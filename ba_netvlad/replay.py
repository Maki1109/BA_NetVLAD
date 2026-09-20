
"""Full BA-NetVLAD pipeline replayed over a query sequence (laptop benchmark).

Simulates the edge control flow:
  sharp frame  -> LCD only at keyframe interval (ORB branch trusted)
  blurry frame -> bypass ORB, descriptor match every frame + temporal voting
Reports recall split by branch and accepted-loop precision/recall.
"""
import numpy as np
import torch
import cv2
from .blur import BlurGate
from .dataset import SequenceDataset, to_tensor_224
from .matcher import CosineDict, TemporalVoter, confirm_loop
from .evaluate import recall_at_n

@torch.no_grad()
def run_replay(model, dict_data, query_root, epsilon=120.0, tau=0.75, delta=0.05,
               r=5, m=3, keyframe_interval=5, loop_index_gap=100,
               cell_size=4.0, device='cpu', synth_blur_lengths=(15, 25),
               margin_window=25, vote_cell=None, project=None):
    """query_root: SequenceDataset-format folder. Ref poses come from dict_data.
    A ground-truth loop for query i = any ref j with ||xy_i - xy_j|| <= cell_size
    AND |i - j| > loop_index_gap (index adjacency excludes trivial matches).

    The SAME |i - j| > loop_index_gap window is applied to retrieval, not just to
    the ground truth. When query_root is the sequence the dictionary was built
    from, the un-masked top-1 is the query frame matching itself at cosine 1.0
    and the rest of the top-k is its own temporal neighbourhood -- none of which
    can ever be a ground-truth loop. Scoring eligible GT against ineligible
    candidates is what caps Recall@5 and would make every accepted loop a
    self-match counted as a false positive.

    `project`: optional callable applied to the raw descriptor (PCA+whitening).
    """
    model.eval()
    ds = SequenceDataset(query_root, cell_size=cell_size, transform=lambda im: im)
    cd = CosineDict(dict_data)
    gate = BlurGate(epsilon=epsilon)
    voter = TemporalVoter(r=r, m=m)

    ref_xy, ref_ids = dict_data['poses'], dict_data['node_ids']
    n_ref = len(ref_xy)
    # original sequence index of each dictionary entry (identity for a dense dict)
    ref_index = np.asarray(dict_data.get('src_index', np.arange(n_ref)))
    # Vote on a node coarse enough to survive keyframe_interval frames of motion.
    # The stock 4 m dictionary cell is ~one keyframe step on KITTI (0.82 m/frame
    # x 5 = 4.1 m), so consecutive attempts land in different cells and m-of-r
    # agreement is unreachable by construction.
    if vote_cell is None:
        vote_nodes = ref_ids
    else:
        vote_nodes = np.array([hash((int(np.floor(x / vote_cell)),
                                     int(np.floor(y / vote_cell)))) & 0x7fffffff
                               for x, y in ref_xy])
    scores_all, gt_all, branch_all, accepts = [], [], [], []
    emb_cache = []

    for i in range(len(ds)):
        img = cv2.imread(ds.files[i])
        beta, is_sharp = gate(img)
        x = to_tensor_224(img).unsqueeze(0).to(device)
        if project is None:
            emb = model(x).cpu().numpy()[0]
        else:   # PCA+whitening is fitted on the pre-head VLAD vector
            emb = project(model(x, return_vlad=True)[1].cpu().numpy()[0])
        emb_cache.append(emb)
        # ineligible = inside the temporal exclusion window around the live frame
        exclude = np.abs(ref_index - i) <= loop_index_gap
        s_full = cd.scores(emb, exclude=exclude)
        idx, sc = cd.topk(emb, k=5, exclude=exclude)
        scores_all.append(s_full)
        # GT: spatial positive, index-far (applied for every i, not only i < n_ref)
        d = np.linalg.norm(ref_xy - ds.xy[i], axis=1)
        gt = (d <= cell_size) & (np.abs(ref_index - i) > loop_index_gap)
        gt_all.append(gt)
        branch_all.append('sharp' if is_sharp else 'blur')
        if not is_sharp or i % keyframe_interval == 0:
            ok, info = confirm_loop(idx, sc, voter, tau, delta,
                                    topk_node_map=vote_nodes, all_scores=s_full,
                                    margin_window=margin_window, ref_index=ref_index)
            if ok:
                accepts.append({**info, 'q_idx': i, 'beta': beta,
                                'correct': bool(gt[info['match_idx']])})
        else:
            voter.vote(TemporalVoter.NO_VOTE)   # keep the ring buffer on real time
    voter.reset()
    S = np.stack(scores_all); G = np.stack(gt_all); B = np.array(branch_all)
    n_loopable = int(G.any(axis=1).sum())
    # lcd_recall = fraction of loop-capable QUERY frames that produced a correct
    # accepted loop (the old version counted distinct matched ref indices, which
    # is not a recall of anything).
    hit_q = {a['q_idx'] for a in accepts if a['correct']}
    res = {'recall_all': recall_at_n(S, G), 'n_frames': len(ds),
           'n_blur_frames': int((~ (B == 'sharp')).sum()),
           'n_loopable_queries': n_loopable,
           'n_accepted_loops': len(accepts),
           'accepted_loops': accepts,
           'lcd_precision': (float(np.mean([a['correct'] for a in accepts]))
                             if accepts else float('nan')),
           'lcd_recall': len(hit_q) / max(n_loopable, 1)}
    for br in ('sharp', 'blur'):
        mask = (B == br) & G.any(axis=1)
        if mask.sum():
            res[f'recall_{br}_pos'] = recall_at_n(S[mask], G[mask])
    return res
