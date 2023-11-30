# Bridging the Gap: A Unified Video Comprehension Framework for Moment Retrieval and Highlight Detection
[Yicheng Xiao*](https://easonxiao-888.github.io/), [Zhuoyan Luo*](https://robertluo1.github.io/), [Yong Liu](https://workforai.github.io/), Yue Ma, Hengwei Bian, Yatai Ji, Yujiu Yang, Xiu Li

<a href='https://arxiv.org/pdf/2311.16464.pdf'><img src='https://img.shields.io/badge/ArXiv-2311.16464-red'></a> 

[THUSIGSCLUB](https://thusigsclub.github.io/thu.github.io/index.html)


## Abstract
Video Moment Retrieval (MR) and Highlight Detection (HD) have attracted significant attention due to the growing demand for video analysis. Recent approaches treat MR and HD as similar video grounding problems and address them together with transformer-based architecture. However, we observe that the emphasis of MR and HD differs, with one necessitating the perception of local relationships and the other prioritizing the understanding of global contexts. Consequently, the lack of task-specific design will inevitably lead to limitations in associating the intrinsic specialty of two tasks. To tackle the issue, we propose a Unified Video COMprehension framework (UVCOM) to bridge the gap and jointly solve MR and HD effectively. By performing progressive integration on intra and inter-modality across multi-granularity, UVCOM achieves the comprehensive understanding in processing a video. Moreover, we present multi-aspect contrastive learning to consolidate the local relation modeling and global knowledge accumulation via well aligned multi-modal space. Extensive experiments on QVHighlights, Charades-STA, TACoS , YouTube Highlights and TVSum datasets demonstrate the effectiveness and rationality of UVCOM which outperforms the state-of-the-art methods by a remarkable margin.

---

<p align="center">
 <img src="assets/framework.png" width="100%">
</p>

## 🎤🎤🎤 Todo

- [ ] Release the code.
- [ ] Release the config and checkpoints.
