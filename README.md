# Latest Adversarial Attack Papers
**update at 2025-08-18 16:21:41**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Random Walk Learning and the Pac-Man Attack**

stat.ML

The updated manuscript represents an incomplete version of the work.  A substantially updated version will be prepared before further dissemination

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2508.05663v2) [paper-pdf](http://arxiv.org/pdf/2508.05663v2)

**Authors**: Xingran Chen, Parimal Parag, Rohit Bhagat, Zonghong Liu, Salim El Rouayheb

**Abstract**: Random walk (RW)-based algorithms have long been popular in distributed systems due to low overheads and scalability, with recent growing applications in decentralized learning. However, their reliance on local interactions makes them inherently vulnerable to malicious behavior. In this work, we investigate an adversarial threat that we term the ``Pac-Man'' attack, in which a malicious node probabilistically terminates any RW that visits it. This stealthy behavior gradually eliminates active RWs from the network, effectively halting the learning process without triggering failure alarms. To counter this threat, we propose the Average Crossing (AC) algorithm--a fully decentralized mechanism for duplicating RWs to prevent RW extinction in the presence of Pac-Man. Our theoretical analysis establishes that (i) the RW population remains almost surely bounded under AC and (ii) RW-based stochastic gradient descent remains convergent under AC, even in the presence of Pac-Man, with a quantifiable deviation from the true optimum. Our extensive empirical results on both synthetic and real-world datasets corroborate our theoretical findings. Furthermore, they uncover a phase transition in the extinction probability as a function of the duplication threshold. We offer theoretical insights by analyzing a simplified variant of the AC, which sheds light on the observed phase transition.



## **2. Robust Convolution Neural ODEs via Contractivity-promoting regularization**

cs.LG

Accepted in IEEE CDC2025, Rio de Janeiro, Brazil

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2508.11432v1) [paper-pdf](http://arxiv.org/pdf/2508.11432v1)

**Authors**: Muhammad Zakwan, Liang Xu, Giancarlo Ferrari-Trecate

**Abstract**: Neural networks can be fragile to input noise and adversarial attacks.   In this work, we consider Convolutional Neural Ordinary Differential Equations (NODEs), a family of continuous-depth neural networks represented by dynamical systems, and propose to use contraction theory to improve their robustness.   For a contractive dynamical system two trajectories starting from different initial conditions converge to each other exponentially fast.   Contractive Convolutional NODEs can enjoy increased robustness as slight perturbations of the features do not cause a significant change in the output.   Contractivity can be induced during training by using a regularization term involving the Jacobian of the system dynamics.   To reduce the computational burden, we show that it can also be promoted using carefully selected weight regularization terms for a class of NODEs with slope-restricted activation functions.   The performance of the proposed regularizers is illustrated through benchmark image classification tasks on MNIST and FashionMNIST datasets, where images are corrupted by different kinds of noise and attacks.



## **3. Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models**

cs.LG

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2506.07468v2) [paper-pdf](http://arxiv.org/pdf/2506.07468v2)

**Authors**: Mickel Liu, Liwei Jiang, Yancheng Liang, Simon Shaolei Du, Yejin Choi, Tim Althoff, Natasha Jaques

**Abstract**: Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL).



## **4. Semantically Guided Adversarial Testing of Vision Models Using Language Models**

cs.CV

12 pages, 4 figures, 3 tables. Submitted for peer review

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2508.11341v1) [paper-pdf](http://arxiv.org/pdf/2508.11341v1)

**Authors**: Katarzyna Filus, Jorge M. Cruz-Duarte

**Abstract**: In targeted adversarial attacks on vision models, the selection of the target label is a critical yet often overlooked determinant of attack success. This target label corresponds to the class that the attacker aims to force the model to predict. Now, existing strategies typically rely on randomness, model predictions, or static semantic resources, limiting interpretability, reproducibility, or flexibility. This paper then proposes a semantics-guided framework for adversarial target selection using the cross-modal knowledge transfer from pretrained language and vision-language models. We evaluate several state-of-the-art models (BERT, TinyLLAMA, and CLIP) as similarity sources to select the most and least semantically related labels with respect to the ground truth, forming best- and worst-case adversarial scenarios. Our experiments on three vision models and five attack methods reveal that these models consistently render practical adversarial targets and surpass static lexical databases, such as WordNet, particularly for distant class relationships. We also observe that static testing of target labels offers a preliminary assessment of the effectiveness of similarity sources, \textit{a priori} testing. Our results corroborate the suitability of pretrained models for constructing interpretable, standardized, and scalable adversarial benchmarks across architectures and datasets.



## **5. MUNBa: Machine Unlearning via Nash Bargaining**

cs.CV

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2411.15537v3) [paper-pdf](http://arxiv.org/pdf/2411.15537v3)

**Authors**: Jing Wu, Mehrtash Harandi

**Abstract**: Machine Unlearning (MU) aims to selectively erase harmful behaviors from models while retaining the overall utility of the model. As a multi-task learning problem, MU involves balancing objectives related to forgetting specific concepts/data and preserving general performance. A naive integration of these forgetting and preserving objectives can lead to gradient conflicts and dominance, impeding MU algorithms from reaching optimal solutions. To address the gradient conflict and dominance issue, we reformulate MU as a two-player cooperative game, where the two players, namely, the forgetting player and the preservation player, contribute via their gradient proposals to maximize their overall gain and balance their contributions. To this end, inspired by the Nash bargaining theory, we derive a closed-form solution to guide the model toward the Pareto stationary point. Our formulation of MU guarantees an equilibrium solution, where any deviation from the final state would lead to a reduction in the overall objectives for both players, ensuring optimality in each objective. We evaluate our algorithm's effectiveness on a diverse set of tasks across image classification and image generation. Extensive experiments with ResNet, vision-language model CLIP, and text-to-image diffusion models demonstrate that our method outperforms state-of-the-art MU algorithms, achieving a better trade-off between forgetting and preserving. Our results also highlight improvements in forgetting precision, preservation of generalization, and robustness against adversarial attacks.



## **6. Towards Physically Realizable Adversarial Attacks in Embodied Vision Navigation**

cs.CV

7 pages, 7 figures, Accept by IEEE/RSJ International Conference on  Intelligent Robots and Systems (IROS) 2025

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2409.10071v5) [paper-pdf](http://arxiv.org/pdf/2409.10071v5)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The significant advancements in embodied vision navigation have raised concerns about its susceptibility to adversarial attacks exploiting deep neural networks. Investigating the adversarial robustness of embodied vision navigation is crucial, especially given the threat of 3D physical attacks that could pose risks to human safety. However, existing attack methods for embodied vision navigation often lack physical feasibility due to challenges in transferring digital perturbations into the physical world. Moreover, current physical attacks for object detection struggle to achieve both multi-view effectiveness and visual naturalness in navigation scenarios. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches to objects, where both opacity and textures are learnable. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which optimizes the patch's texture based on feedback from the vision-based perception model used in navigation. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, in which opacity is fine-tuned after texture optimization. Experimental results demonstrate that our adversarial patches decrease the navigation success rate by an average of 22.39%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav



## **7. SHLIME: Foiling adversarial attacks fooling SHAP and LIME**

cs.LG

7 pages, 7 figures

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.11053v1) [paper-pdf](http://arxiv.org/pdf/2508.11053v1)

**Authors**: Sam Chauhan, Estelle Duguet, Karthik Ramakrishnan, Hugh Van Deventer, Jack Kruger, Ranjan Subbaraman

**Abstract**: Post hoc explanation methods, such as LIME and SHAP, provide interpretable insights into black-box classifiers and are increasingly used to assess model biases and generalizability. However, these methods are vulnerable to adversarial manipulation, potentially concealing harmful biases. Building on the work of Slack et al. (2020), we investigate the susceptibility of LIME and SHAP to biased models and evaluate strategies for improving robustness. We first replicate the original COMPAS experiment to validate prior findings and establish a baseline. We then introduce a modular testing framework enabling systematic evaluation of augmented and ensemble explanation approaches across classifiers of varying performance. Using this framework, we assess multiple LIME/SHAP ensemble configurations on out-of-distribution models, comparing their resistance to bias concealment against the original methods. Our results identify configurations that substantially improve bias detection, highlighting their potential for enhancing transparency in the deployment of high-stakes machine learning systems.



## **8. Byzantine-Resilient Decentralized Online Resource Allocation**

math.OC

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.08658v2) [paper-pdf](http://arxiv.org/pdf/2508.08658v2)

**Authors**: Runhua Wang, Qing Ling, Hoi-To Wai, Zhi Tian

**Abstract**: In this paper, we investigate the problem of decentralized online resource allocation in the presence of Byzantine attacks. In this problem setting, some agents may be compromised due to external manipulations or internal failures, causing them to behave maliciously and disrupt the resource allocation process by sending incorrect messages to their neighbors. Given the non-consensual nature of the resource allocation problem, we formulate it under a primal-dual optimization framework, where the dual variables are aggregated among the agents, enabling the incorporation of robust aggregation mechanisms to mitigate Byzantine attacks. By leveraging the classical Byzantine attack model, we propose a class of Byzantine-resilient decentralized online resource allocation algorithms that judiciously integrate the adaptive robust clipping technique with the existing robust aggregation rules to filter out adversarial messages. We establish theoretical guarantees, showing that the proposed algorithms achieve tight linear dynamic regret and accumulative constraint violation bounds, where the constants depend on the properties of robust aggregation rules. Numerical experiments on decentralized online economic dispatch validate the effectiveness of our approach and support our theoretical results.



## **9. JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example**

cs.LG

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2401.01199v2) [paper-pdf](http://arxiv.org/pdf/2401.01199v2)

**Authors**: Benedetta Tondi, Wei Guo, Niccolò Pancino, Mauro Barni

**Abstract**: Most of the approaches proposed so far to craft targeted adversarial examples against Deep Learning classifiers are highly suboptimal and typically rely on increasing the likelihood of the target class, thus implicitly focusing on one-hot encoding settings. In this paper, a more general, theoretically sound, targeted attack is proposed, which resorts to the minimization of a Jacobian-induced Mahalanobis distance term, taking into account the effort (in the input space) required to move the latent space representation of the input sample in a given direction. The minimization is solved by exploiting the Wolfe duality theorem, reducing the problem to the solution of a Non-Negative Least Square (NNLS) problem. The proposed algorithm (referred to as JMA) provides an optimal solution to a linearised version of the adversarial example problem originally introduced by Szegedy et al. The results of the experiments confirm the generality of the proposed attack which is proven to be effective under a wide variety of output encoding schemes. Noticeably, JMA is also effective in a multi-label classification scenario, being capable to induce a targeted modification of up to half the labels in complex multi-label classification scenarios, a capability that is out of reach of all the attacks proposed so far. As a further advantage, JMA requires very few iterations, thus resulting more efficient than existing methods.



## **10. MCP-Guard: A Defense Framework for Model Context Protocol Integrity in Large Language Model Applications**

cs.CR

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10991v1) [paper-pdf](http://arxiv.org/pdf/2508.10991v1)

**Authors**: Wenpeng Xing, Zhonghao Qi, Yupeng Qin, Yilin Li, Caini Chang, Jiahui Yu, Changting Lin, Zhenzhen Xie, Meng Han

**Abstract**: The integration of Large Language Models (LLMs) with external tools via protocols such as the Model Context Protocol (MCP) introduces critical security vulnerabilities, including prompt injection, data exfiltration, and other threats. To counter these challenges, we propose MCP-Guard, a robust, layered defense architecture designed for LLM--tool interactions. MCP-Guard employs a three-stage detection pipeline that balances efficiency with accuracy: it progresses from lightweight static scanning for overt threats and a deep neural detector for semantic attacks, to our fine-tuned E5-based model achieves (96.01) accuracy in identifying adversarial prompts. Finally, a lightweight LLM arbitrator synthesizes these signals to deliver the final decision while minimizing false positives. To facilitate rigorous training and evaluation, we also introduce MCP-AttackBench, a comprehensive benchmark of over 70,000 samples. Sourced from public datasets and augmented by GPT-4, MCP-AttackBench simulates diverse, real-world attack vectors in the MCP format, providing a foundation for future research into securing LLM-tool ecosystems.



## **11. MirGuard: Towards a Robust Provenance-based Intrusion Detection System Against Graph Manipulation Attacks**

cs.CR

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10639v1) [paper-pdf](http://arxiv.org/pdf/2508.10639v1)

**Authors**: Anyuan Sang, Lu Zhou, Li Yang, Junbo Jia, Huipeng Yang, Pengbin Feng, Jianfeng Ma

**Abstract**: Learning-based Provenance-based Intrusion Detection Systems (PIDSes) have become essential tools for anomaly detection in host systems due to their ability to capture rich contextual and structural information, as well as their potential to detect unknown attacks. However, recent studies have shown that these systems are vulnerable to graph manipulation attacks, where attackers manipulate the graph structure to evade detection. While some previous approaches have discussed this type of attack, none have fully addressed it with a robust detection solution, limiting the practical applicability of PIDSes.   To address this challenge, we propose MirGuard, a robust anomaly detection framework that combines logic-aware multi-view augmentation with contrastive representation learning. Rather than applying arbitrary structural perturbations, MirGuard introduces Logic-Aware Noise Injection (LNI) to generate semantically valid graph views, ensuring that all augmentations preserve the underlying causal semantics of the provenance data. These views are then used in a Logic-Preserving Contrastive Learning framework, which encourages the model to learn representations that are invariant to benign transformations but sensitive to adversarial inconsistencies. Comprehensive evaluations on multiple provenance datasets demonstrate that MirGuard significantly outperforms state-of-the-art detectors in robustness against various graph manipulation attacks without sacrificing detection performance and efficiency. Our work represents the first targeted study to enhance PIDS against such adversarial threats, providing a robust and effective solution to modern cybersecurity challenges.



## **12. Towards Powerful and Practical Patch Attacks for 2D Object Detection in Autonomous Driving**

cs.CV

13 pages, 4 figures

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10600v1) [paper-pdf](http://arxiv.org/pdf/2508.10600v1)

**Authors**: Yuxin Cao, Yedi Zhang, Wentao He, Yifan Liao, Yan Xiao, Chang Li, Zhiyong Huang, Jin Song Dong

**Abstract**: Learning-based autonomous driving systems remain critically vulnerable to adversarial patches, posing serious safety and security risks in their real-world deployment. Black-box attacks, notable for their high attack success rate without model knowledge, are especially concerning, with their transferability extensively studied to reduce computational costs compared to query-based attacks. Previous transferability-based black-box attacks typically adopt mean Average Precision (mAP) as the evaluation metric and design training loss accordingly. However, due to the presence of multiple detected bounding boxes and the relatively lenient Intersection over Union (IoU) thresholds, the attack effectiveness of these approaches is often overestimated, resulting in reduced success rates in practical attacking scenarios. Furthermore, patches trained on low-resolution data often fail to maintain effectiveness on high-resolution images, limiting their transferability to autonomous driving datasets. To fill this gap, we propose P$^3$A, a Powerful and Practical Patch Attack framework for 2D object detection in autonomous driving, specifically optimized for high-resolution datasets. First, we introduce a novel metric, Practical Attack Success Rate (PASR), to more accurately quantify attack effectiveness with greater relevance for pedestrian safety. Second, we present a tailored Localization-Confidence Suppression Loss (LCSL) to improve attack transferability under PASR. Finally, to maintain the transferability for high-resolution datasets, we further incorporate the Probabilistic Scale-Preserving Padding (PSPP) into the patch attack pipeline as a data preprocessing step. Extensive experiments show that P$^3$A outperforms state-of-the-art attacks on unseen models and unseen high-resolution datasets, both under the proposed practical IoU-based evaluation metric and the previous mAP-based metrics.



## **13. Adversarial Robustness in Two-Stage Learning-to-Defer: Algorithms and Guarantees**

stat.ML

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2502.01027v3) [paper-pdf](http://arxiv.org/pdf/2502.01027v3)

**Authors**: Yannis Montreuil, Axel Carlier, Lai Xing Ng, Wei Tsang Ooi

**Abstract**: Two-stage Learning-to-Defer (L2D) enables optimal task delegation by assigning each input to either a fixed main model or one of several offline experts, supporting reliable decision-making in complex, multi-agent environments. However, existing L2D frameworks assume clean inputs and are vulnerable to adversarial perturbations that can manipulate query allocation--causing costly misrouting or expert overload. We present the first comprehensive study of adversarial robustness in two-stage L2D systems. We introduce two novel attack strategie--untargeted and targeted--which respectively disrupt optimal allocations or force queries to specific agents. To defend against such threats, we propose SARD, a convex learning algorithm built on a family of surrogate losses that are provably Bayes-consistent and $(\mathcal{R}, \mathcal{G})$-consistent. These guarantees hold across classification, regression, and multi-task settings. Empirical results demonstrate that SARD significantly improves robustness under adversarial attacks while maintaining strong clean performance, marking a critical step toward secure and trustworthy L2D deployment.



## **14. Contrastive ECOC: Learning Output Codes for Adversarial Defense**

cs.LG

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10491v1) [paper-pdf](http://arxiv.org/pdf/2508.10491v1)

**Authors**: Che-Yu Chou, Hung-Hsuan Chen

**Abstract**: Although one-hot encoding is commonly used for multiclass classification, it is not always the most effective encoding mechanism. Error Correcting Output Codes (ECOC) address multiclass classification by mapping each class to a unique codeword used as a label. Traditional ECOC methods rely on manually designed or randomly generated codebooks, which are labor-intensive and may yield suboptimal, dataset-agnostic results. This paper introduces three models for automated codebook learning based on contrastive learning, allowing codebooks to be learned directly and adaptively from data. Across four datasets, our proposed models demonstrate superior robustness to adversarial attacks compared to two baselines. The source is available at https://github.com/YuChou20/Automated-Codebook-Learning-with-Error-Correcting-Output-Code-Technique.



## **15. Layer-Wise Perturbations via Sparse Autoencoders for Adversarial Text Generation**

cs.CL

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10404v1) [paper-pdf](http://arxiv.org/pdf/2508.10404v1)

**Authors**: Huizhen Shu, Xuying Li, Qirui Wang, Yuji Kosuga, Mengqiu Tian, Zhuo Li

**Abstract**: With the rapid proliferation of Natural Language Processing (NLP), especially Large Language Models (LLMs), generating adversarial examples to jailbreak LLMs remains a key challenge for understanding model vulnerabilities and improving robustness. In this context, we propose a new black-box attack method that leverages the interpretability of large models. We introduce the Sparse Feature Perturbation Framework (SFPF), a novel approach for adversarial text generation that utilizes sparse autoencoders to identify and manipulate critical features in text. After using the SAE model to reconstruct hidden layer representations, we perform feature clustering on the successfully attacked texts to identify features with higher activations. These highly activated features are then perturbed to generate new adversarial texts. This selective perturbation preserves the malicious intent while amplifying safety signals, thereby increasing their potential to evade existing defenses. Our method enables a new red-teaming strategy that balances adversarial effectiveness with safety alignment. Experimental results demonstrate that adversarial texts generated by SFPF can bypass state-of-the-art defense mechanisms, revealing persistent vulnerabilities in current NLP systems.However, the method's effectiveness varies across prompts and layers, and its generalizability to other architectures and larger models remains to be validated.



## **16. BERTector: Intrusion Detection Based on Joint-Dataset Learning**

cs.CR

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10327v1) [paper-pdf](http://arxiv.org/pdf/2508.10327v1)

**Authors**: Haoyang Hu, Xun Huang, Chenyu Wu, Shiwen Liu, Zhichao Lian, Shuangquan Zhang

**Abstract**: Intrusion detection systems (IDS) are facing challenges in generalization and robustness due to the heterogeneity of network traffic and the diversity of attack patterns. To address this issue, we propose a new joint-dataset training paradigm for IDS and propose a scalable BERTector framework based on BERT. BERTector integrates three key components: NSS-Tokenizer for traffic-aware semantic tokenization, supervised fine-tuning with a hybrid dataset, and low-rank adaptation (LoRA) for efficient training. Extensive experiments show that BERTector achieves state-of-the-art detection accuracy, strong cross-dataset generalization capabilities, and excellent robustness to adversarial perturbations. This work establishes a unified and efficient solution for modern IDS in complex and dynamic network environments.



## **17. Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability**

cs.CV

Preprint

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2506.18248v4) [paper-pdf](http://arxiv.org/pdf/2506.18248v4)

**Authors**: Jongoh Jeong, Hunmin Yang, Jaeseok Jeong, Kuk-Jin Yoon

**Abstract**: Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR).



## **18. PromptSafe: Gated Prompt Tuning for Safe Text-to-Image Generation**

cs.CV

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.01272v2) [paper-pdf](http://arxiv.org/pdf/2508.01272v2)

**Authors**: Zonglei Jing, Xiao Yang, Xiaoqian Li, Siyuan Liang, Aishan Liu, Mingchuan Zhang, Xianglong Liu

**Abstract**: Text-to-image (T2I) models have demonstrated remarkable generative capabilities but remain vulnerable to producing not-safe-for-work (NSFW) content, such as violent or explicit imagery. While recent moderation efforts have introduced soft prompt-guided tuning by appending defensive tokens to the input, these approaches often rely on large-scale curated image-text datasets and apply static, one-size-fits-all defenses at inference time. However, this results not only in high computational cost and degraded benign image quality, but also in limited adaptability to the diverse and nuanced safety requirements of real-world prompts. To address these challenges, we propose PromptSafe, a gated prompt tuning framework that combines a lightweight, text-only supervised soft embedding with an inference-time gated control network. Instead of training on expensive image-text datasets, we first rewrite unsafe prompts into semantically aligned but safe alternatives using an LLM, constructing an efficient text-only training corpus. Based on this, we optimize a universal soft prompt that repels unsafe and attracts safe embeddings during the diffusion denoising process. To avoid over-suppressing benign prompts, we introduce a gated mechanism that adaptively adjusts the defensive strength based on estimated prompt toxicity, thereby aligning defense intensity with prompt risk and ensuring strong protection for harmful inputs while preserving benign generation quality. Extensive experiments across multiple benchmarks and T2I models show that PromptSafe achieves a SOTA unsafe generation rate (2.36%), while preserving high benign fidelity. Furthermore, PromptSafe demonstrates strong generalization to unseen harmful categories, robust transferability across diffusion model architectures, and resilience under adaptive adversarial attacks, highlighting its practical value for safe and scalable deployment.



## **19. Detecting Untargeted Attacks and Mitigating Unreliable Updates in Federated Learning for Underground Mining Operations**

cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.10212v1) [paper-pdf](http://arxiv.org/pdf/2508.10212v1)

**Authors**: Md Sazedur Rahman, Mohamed Elmahallawy, Sanjay Madria, Samuel Frimpong

**Abstract**: Underground mining operations rely on distributed sensor networks to collect critical data daily, including mine temperature, toxic gas concentrations, and miner movements for hazard detection and operational decision-making. However, transmitting raw sensor data to a central server for training deep learning models introduces significant privacy risks, potentially exposing sensitive mine-specific information. Federated Learning (FL) offers a transformative solution by enabling collaborative model training while ensuring that raw data remains localized at each mine. Despite its advantages, FL in underground mining faces key challenges: (i) An attacker may compromise a mine's local model by employing techniques such as sign-flipping attacks or additive noise, leading to erroneous predictions; (ii) Low-quality (yet potentially valuable) data, caused by poor lighting conditions or sensor inaccuracies in mines may degrade the FL training process. In response, this paper proposes MineDetect, a defense FL framework that detects and isolates the attacked models while mitigating the impact of mines with low-quality data. MineDetect introduces two key innovations: (i) Detecting attacked models (maliciously manipulated) by developing a history-aware mechanism that leverages local and global averages of gradient updates; (ii) Identifying and eliminating adversarial influences from unreliable models (generated by clients with poor data quality) on the FL training process. Comprehensive simulations across diverse datasets demonstrate that MineDetect outperforms existing methods in both robustness and accuracy, even in challenging non-IID data scenarios. Its ability to counter adversarial influences while maintaining lower computational efficiency makes it a vital advancement for improving safety and operational effectiveness in underground mining.



## **20. An Architecture for Distributed Digital Identities in the Physical World**

cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.10185v1) [paper-pdf](http://arxiv.org/pdf/2508.10185v1)

**Authors**: René Mayrhofer, Michael Roland, Tobias Höller, Philipp Hofer, Mario Lins

**Abstract**: Digital identities are increasingly important for mediating not only digital but also physical service transactions. Managing such identities through centralized providers can cause both availability and privacy concerns: single points of failure and control are ideal targets for global attacks on technical, organizational, or legal fronts. We design, analyze, and build a distributed digital identity architecture for physical world transactions in common scenarios like unlocking doors, public transport, or crossing country borders. This architecture combines (biometric and other) sensors, (established and upcoming) identity authorities, attribute verifiers, and a new core component we call the \emph{Personal Identity Agent (PIA)} that represents individuals with their identity attributes in the digital domain. All transactions are conducted in a completely decentralized manner, and the components for which we currently assume central coordination are optional and only used for assisting with service discovery and latency reduction. We present a first protocol between these parties and formally verify that it achieves relevant security properties based on a realistic threat model including strong global adversaries. A proof-of-concept implementation demonstrates practical feasibility of both architecture and initial protocol for applications that can tolerate end-to-end latencies in the range of a few seconds.



## **21. Security Concerns for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2505.18889v3) [paper-pdf](http://arxiv.org/pdf/2505.18889v3)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as ChatGPT and its competitors have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. This survey provides a comprehensive overview of these emerging concerns, categorizing threats into several key areas: prompt injection and jailbreaking; adversarial attacks, including input perturbations and data poisoning; misuse by malicious actors to generate disinformation, phishing emails, and malware; and the worrisome risks inherent in autonomous LLM agents. Recently, a significant focus is increasingly being placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives, a behavior known as scheming, which may even persist through safety training. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.



## **22. Perturbed Public Voices (P$^{2}$V): A Dataset for Robust Audio Deepfake Detection**

cs.SD

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.10949v1) [paper-pdf](http://arxiv.org/pdf/2508.10949v1)

**Authors**: Chongyang Gao, Marco Postiglione, Isabel Gortner, Sarit Kraus, V. S. Subrahmanian

**Abstract**: Current audio deepfake detectors cannot be trusted. While they excel on controlled benchmarks, they fail when tested in the real world. We introduce Perturbed Public Voices (P$^{2}$V), an IRB-approved dataset capturing three critical aspects of malicious deepfakes: (1) identity-consistent transcripts via LLMs, (2) environmental and adversarial noise, and (3) state-of-the-art voice cloning (2020-2025). Experiments reveal alarming vulnerabilities of 22 recent audio deepfake detectors: models trained on current datasets lose 43% performance when tested on P$^{2}$V, with performance measured as the mean of F1 score on deepfake audio, AUC, and 1-EER. Simple adversarial perturbations induce up to 16% performance degradation, while advanced cloning techniques reduce detectability by 20-30%. In contrast, P$^{2}$V-trained models maintain robustness against these attacks while generalizing to existing datasets, establishing a new benchmark for robust audio deepfake detection. P$^{2}$V will be publicly released upon acceptance by a conference/journal.



## **23. IPG: Incremental Patch Generation for Generalized Adversarial Patch Training**

cs.CV

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.10946v1) [paper-pdf](http://arxiv.org/pdf/2508.10946v1)

**Authors**: Wonho Lee, Hyunsik Na, Jisu Lee, Daeseon Choi

**Abstract**: The advent of adversarial patches poses a significant challenge to the robustness of AI models, particularly in the domain of computer vision tasks such as object detection. In contradistinction to traditional adversarial examples, these patches target specific regions of an image, resulting in the malfunction of AI models. This paper proposes Incremental Patch Generation (IPG), a method that generates adversarial patches up to 11.1 times more efficiently than existing approaches while maintaining comparable attack performance. The efficacy of IPG is demonstrated by experiments and ablation studies including YOLO's feature distribution visualization and adversarial training results, which show that it produces well-generalized patches that effectively cover a broader range of model vulnerabilities. Furthermore, IPG-generated datasets can serve as a robust knowledge foundation for constructing a robust model, enabling structured representation, advanced reasoning, and proactive defenses in AI security ecosystems. The findings of this study suggest that IPG has considerable potential for future utilization not only in adversarial patch defense but also in real-world applications such as autonomous vehicles, security systems, and medical imaging, where AI models must remain resilient to adversarial attacks in dynamic and high-stakes environments.



## **24. MetaGuardian: Enhancing Voice Assistant Security through Advanced Acoustic Metamaterials**

cs.SD

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.09728v1) [paper-pdf](http://arxiv.org/pdf/2508.09728v1)

**Authors**: Zhiyuan Ning, Zheng Wang, Zhanyong Tang

**Abstract**: We present MetaGuardian, a voice assistant (VA) protection system based on acoustic metamaterials. MetaGuardian can be directly integrated into the enclosures of various smart devices, effectively defending against inaudible, adversarial and laser attacks without relying on additional software support or altering the underlying hardware, ensuring usability. To achieve this, MetaGuardian leverages the mutual impedance effects between metamaterial units to extend the signal filtering range to 16-40 kHz to effectively block wide-band inaudible attacks. Additionally, it adopts a carefully designed coiled space structure to precisely interfere with adversarial attacks while ensuring the normal functioning of VAs. Furthermore, MetaGuardian offers a universal structural design, allowing itself to be flexibly adapted to various smart devices, striking a balance between portability and protection effectiveness. In controled evaluation environments, MetaGuardian achieves a high defense success rate against various attack types, including adversarial, inaudible and laser attacks.



## **25. MetaCipher: A Time-Persistent and Universal Multi-Agent Framework for Cipher-Based Jailbreak Attacks for LLMs**

cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2506.22557v2) [paper-pdf](http://arxiv.org/pdf/2506.22557v2)

**Authors**: Boyuan Chen, Minghao Shao, Abdul Basit, Siddharth Garg, Muhammad Shafique

**Abstract**: As large language models (LLMs) grow more capable, they face growing vulnerability to sophisticated jailbreak attacks. While developers invest heavily in alignment finetuning and safety guardrails, researchers continue publishing novel attacks, driving progress through adversarial iteration. This dynamic mirrors a strategic game of continual evolution. However, two major challenges hinder jailbreak development: the high cost of querying top-tier LLMs and the short lifespan of effective attacks due to frequent safety updates. These factors limit cost-efficiency and practical impact of research in jailbreak attacks. To address this, we propose MetaCipher, a low-cost, multi-agent jailbreak framework that generalizes across LLMs with varying safety measures. Using reinforcement learning, MetaCipher is modular and adaptive, supporting extensibility to future strategies. Within as few as 10 queries, MetaCipher achieves state-of-the-art attack success rates on recent malicious prompt benchmarks, outperforming prior jailbreak methods. We conduct a large-scale empirical evaluation across diverse victim models and benchmarks, demonstrating its robustness and adaptability. Warning: This paper contains model outputs that may be offensive or harmful, shown solely to demonstrate jailbreak efficacy.



## **26. LLM Robustness Leaderboard v1 --Technical report**

cs.AI

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.06296v2) [paper-pdf](http://arxiv.org/pdf/2508.06296v2)

**Authors**: Pierre Peigné - Lefebvre, Quentin Feuillade-Montixi, Tom David, Nicolas Miailhe

**Abstract**: This technical report accompanies the LLM robustness leaderboard published by PRISM Eval for the Paris AI Action Summit. We introduce PRISM Eval Behavior Elicitation Tool (BET), an AI system performing automated red-teaming through Dynamic Adversarial Optimization that achieves 100% Attack Success Rate (ASR) against 37 of 41 state-of-the-art LLMs. Beyond binary success metrics, we propose a fine-grained robustness metric estimating the average number of attempts required to elicit harmful behaviors, revealing that attack difficulty varies by over 300-fold across models despite universal vulnerability. We introduce primitive-level vulnerability analysis to identify which jailbreaking techniques are most effective for specific hazard categories. Our collaborative evaluation with trusted third parties from the AI Safety Network demonstrates practical pathways for distributed robustness assessment across the community.



## **27. Guardians and Offenders: A Survey on Harmful Content Generation and Safety Mitigation of LLM**

cs.CL

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.05775v2) [paper-pdf](http://arxiv.org/pdf/2508.05775v2)

**Authors**: Chi Zhang, Changjia Zhu, Junjie Xiong, Xiaoran Xu, Lingyao Li, Yao Liu, Zhuo Lu

**Abstract**: Large Language Models (LLMs) have revolutionized content creation across digital platforms, offering unprecedented capabilities in natural language generation and understanding. These models enable beneficial applications such as content generation, question and answering (Q&A), programming, and code reasoning. Meanwhile, they also pose serious risks by inadvertently or intentionally producing toxic, offensive, or biased content. This dual role of LLMs, both as powerful tools for solving real-world problems and as potential sources of harmful language, presents a pressing sociotechnical challenge. In this survey, we systematically review recent studies spanning unintentional toxicity, adversarial jailbreaking attacks, and content moderation techniques. We propose a unified taxonomy of LLM-related harms and defenses, analyze emerging multimodal and LLM-assisted jailbreak strategies, and assess mitigation efforts, including reinforcement learning with human feedback (RLHF), prompt engineering, and safety alignment. Our synthesis highlights the evolving landscape of LLM safety, identifies limitations in current evaluation methodologies, and outlines future research directions to guide the development of robust and ethically aligned language technologies.



## **28. A Taxonomy of System-Level Attacks on Deep Learning Models in Autonomous Vehicles**

cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2412.04510v2) [paper-pdf](http://arxiv.org/pdf/2412.04510v2)

**Authors**: Masoud Jamshidiyan Tehrani, Jinhan Kim, Rosmael Zidane Lekeufack Foulefack, Alessandro Marchetto, Paolo Tonella

**Abstract**: The advent of deep learning and its astonishing performance has enabled its usage in complex systems, including autonomous vehicles. On the other hand, deep learning models are susceptible to mispredictions when small, adversarial changes are introduced into their input. Such mis-predictions can be triggered in the real world and can result in a failure of the entire system. In recent years, a growing number of research works have investigated ways to mount attacks against autonomous vehicles that exploit deep learning components. Such attacks are directed toward elements of the environment where these systems operate and their effectiveness is assessed in terms of system-level failures triggered by them. There has been however no systematic attempt to analyze and categorize such attacks. In this paper, we present the first taxonomy of system-level attacks against autonomous vehicles. We constructed our taxonomy by selecting 21 highly relevant papers, then we tagged them with 12 top-level taxonomy categories and several sub-categories. The taxonomy allowed us to investigate the attack features, the most attacked components and systems, the underlying threat models, and the failure chains from input perturbation to system-level failure. We distilled several lessons for practitioners and identified possible directions for future work for researchers.



## **29. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects  the up-to-date experimental results

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2409.20002v4) [paper-pdf](http://arxiv.org/pdf/2409.20002v4)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.



## **30. 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation**

cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2507.01367v2) [paper-pdf](http://arxiv.org/pdf/2507.01367v2)

**Authors**: Tianrui Lou, Xiaojun Jia, Siyuan Liang, Jiawei Liang, Ming Zhang, Yanjun Xiao, Xiaochun Cao

**Abstract**: Physical adversarial attack methods expose the vulnerabilities of deep neural networks and pose a significant threat to safety-critical scenarios such as autonomous driving. Camouflage-based physical attack is a more promising approach compared to the patch-based attack, offering stronger adversarial effectiveness in complex physical environments. However, most prior work relies on mesh priors of the target object and virtual environments constructed by simulators, which are time-consuming to obtain and inevitably differ from the real world. Moreover, due to the limitations of the backgrounds in training images, previous methods often fail to produce multi-view robust adversarial camouflage and tend to fall into sub-optimal solutions. Due to these reasons, prior work lacks adversarial effectiveness and robustness across diverse viewpoints and physical environments. We propose a physical attack framework based on 3D Gaussian Splatting (3DGS), named PGA, which provides rapid and precise reconstruction with few images, along with photo-realistic rendering capabilities. Our framework further enhances cross-view robustness and adversarial effectiveness by preventing mutual and self-occlusion among Gaussians and employing a min-max optimization approach that adjusts the imaging background of each viewpoint, helping the algorithm filter out non-robust adversarial features. Extensive experiments validate the effectiveness and superiority of PGA. Our code is available at:https://github.com/TRLou/PGA.



## **31. Exact Verification of Graph Neural Networks with Incremental Constraint Solving**

cs.LG

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.09320v1) [paper-pdf](http://arxiv.org/pdf/2508.09320v1)

**Authors**: Minghao Liu, Chia-Hsuan Lu, Marta Kwiatkowska

**Abstract**: Graph neural networks (GNNs) are increasingly employed in high-stakes applications, such as fraud detection or healthcare, but are susceptible to adversarial attacks. A number of techniques have been proposed to provide adversarial robustness guarantees, but support for commonly used aggregation functions in message-passing GNNs is still lacking. In this paper, we develop an exact (sound and complete) verification method for GNNs to compute guarantees against attribute and structural perturbations that involve edge addition or deletion, subject to budget constraints. Focusing on node classification tasks, our method employs constraint solving with bound tightening, and iteratively solves a sequence of relaxed constraint satisfaction problems while relying on incremental solving capabilities of solvers to improve efficiency. We implement GNNev, a versatile solver for message-passing neural networks, which supports three aggregation functions, sum, max and mean, with the latter two considered here for the first time. Extensive experimental evaluation of GNNev on two standard benchmarks (Cora and CiteSeer) and two real-world fraud datasets (Amazon and Yelp) demonstrates its usability and effectiveness, as well as superior performance compared to existing {exact verification} tools on sum-aggregated node classification tasks.



## **32. Constrained Black-Box Attacks Against Multi-Agent Reinforcement Learning**

cs.LG

Under review in TNNLS

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.09275v1) [paper-pdf](http://arxiv.org/pdf/2508.09275v1)

**Authors**: Amine Andam, Jamal Bentahar, Mustapha Hedabou

**Abstract**: Collaborative multi-agent reinforcement learning (c-MARL) has rapidly evolved, offering state-of-the-art algorithms for real-world applications, including sensitive domains. However, a key challenge to its widespread adoption is the lack of a thorough investigation into its vulnerabilities to adversarial attacks. Existing work predominantly focuses on training-time attacks or unrealistic scenarios, such as access to policy weights or the ability to train surrogate policies. In this paper, we investigate new vulnerabilities under more realistic and constrained conditions, assuming an adversary can only collect and perturb the observations of deployed agents. We also consider scenarios where the adversary has no access at all. We propose simple yet highly effective algorithms for generating adversarial perturbations designed to misalign how victim agents perceive their environment. Our approach is empirically validated on three benchmarks and 22 environments, demonstrating its effectiveness across diverse algorithms and environments. Furthermore, we show that our algorithm is sample-efficient, requiring only 1,000 samples compared to the millions needed by previous methods.



## **33. Fre-CW: Targeted Attack on Time Series Forecasting using Frequency Domain Loss**

cs.LG

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.08955v1) [paper-pdf](http://arxiv.org/pdf/2508.08955v1)

**Authors**: Naifu Feng, Lixing Chen, Junhua Tang, Hua Ding, Jianhua Li, Yang Bai

**Abstract**: Transformer-based models have made significant progress in time series forecasting. However, a key limitation of deep learning models is their susceptibility to adversarial attacks, which has not been studied enough in the context of time series prediction. In contrast to areas such as computer vision, where adversarial robustness has been extensively studied, frequency domain features of time series data play an important role in the prediction task but have not been sufficiently explored in terms of adversarial attacks. This paper proposes a time series prediction attack algorithm based on frequency domain loss. Specifically, we adapt an attack method originally designed for classification tasks to the prediction field and optimize the adversarial samples using both time-domain and frequency-domain losses. To the best of our knowledge, there is no relevant research on using frequency information for time-series adversarial attacks. Our experimental results show that these current time series prediction models are vulnerable to adversarial attacks, and our approach achieves excellent performance on major time series forecasting datasets.



## **34. Exploring Cross-Stage Adversarial Transferability in Class-Incremental Continual Learning**

cs.LG

Accepted at MMSP 2025

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.08920v1) [paper-pdf](http://arxiv.org/pdf/2508.08920v1)

**Authors**: Jungwoo Kim, Jong-Seok Lee

**Abstract**: Class-incremental continual learning addresses catastrophic forgetting by enabling classification models to preserve knowledge of previously learned classes while acquiring new ones. However, the vulnerability of the models against adversarial attacks during this process has not been investigated sufficiently. In this paper, we present the first exploration of vulnerability to stage-transferred attacks, i.e., an adversarial example generated using the model in an earlier stage is used to attack the model in a later stage. Our findings reveal that continual learning methods are highly susceptible to these attacks, raising a serious security issue. We explain this phenomenon through model similarity between stages and gradual robustness degradation. Additionally, we find that existing adversarial training-based defense methods are not sufficiently effective to stage-transferred attacks. Codes are available at https://github.com/mcml-official/CSAT.



## **35. Improving the robustness of neural ODEs with minimal weight perturbation**

math.NA

31 pages, 5 figures, 4 tables

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2501.10740v2) [paper-pdf](http://arxiv.org/pdf/2501.10740v2)

**Authors**: Arturo De Marinis, Nicola Guglielmi, Stefano Sicilia, Francesco Tudisco

**Abstract**: We propose a method to enhance the stability of a neural ordinary differential equation (neural ODE) by reducing the maximum error growth subsequent to a perturbation of the initial value. Since the stability depends on the logarithmic norm of the Jacobian matrix associated with the neural ODE, we control the logarithmic norm by perturbing the weight matrices of the neural ODE by a smallest possible perturbation (in Frobenius norm). We do so by engaging an eigenvalue optimisation problem, for which we propose a nested two-level algorithm. For a given perturbation size of the weight matrix, the inner level computes optimal perturbations of that size, while - at the outer level - we tune the perturbation amplitude until we reach the desired uniform stability bound. We embed the proposed algorithm in the training of the neural ODE to improve its robustness to perturbations of the initial value, as adversarial attacks. Numerical experiments on classical image datasets show that an image classifier including a neural ODE in its architecture trained according to our strategy is more stable than the same classifier trained in the classical way, and therefore, it is more robust and less vulnerable to adversarial attacks.



## **36. Adversarial Video Promotion Against Text-to-Video Retrieval**

cs.CV

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.06964v2) [paper-pdf](http://arxiv.org/pdf/2508.06964v2)

**Authors**: Qiwei Tian, Chenhao Lin, Zhengyu Zhao, Qian Li, Shuai Liu, Chao Shen

**Abstract**: Thanks to the development of cross-modal models, text-to-video retrieval (T2VR) is advancing rapidly, but its robustness remains largely unexamined. Existing attacks against T2VR are designed to push videos away from queries, i.e., suppressing the ranks of videos, while the attacks that pull videos towards selected queries, i.e., promoting the ranks of videos, remain largely unexplored. These attacks can be more impactful as attackers may gain more views/clicks for financial benefits and widespread (mis)information. To this end, we pioneer the first attack against T2VR to promote videos adversarially, dubbed the Video Promotion attack (ViPro). We further propose Modal Refinement (MoRe) to capture the finer-grained, intricate interaction between visual and textual modalities to enhance black-box transferability. Comprehensive experiments cover 2 existing baselines, 3 leading T2VR models, 3 prevailing datasets with over 10k videos, evaluated under 3 scenarios. All experiments are conducted in a multi-target setting to reflect realistic scenarios where attackers seek to promote the video regarding multiple queries simultaneously. We also evaluated our attacks for defences and imperceptibility. Overall, ViPro surpasses other baselines by over $30/10/4\%$ for white/grey/black-box settings on average. Our work highlights an overlooked vulnerability, provides a qualitative analysis on the upper/lower bound of our attacks, and offers insights into potential counterplays. Code will be publicly available at https://github.com/michaeltian108/ViPro.



## **37. Cowpox: Towards the Immunity of VLM-based Multi-Agent Systems**

cs.MA

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.09230v1) [paper-pdf](http://arxiv.org/pdf/2508.09230v1)

**Authors**: Yutong Wu, Jie Zhang, Yiming Li, Chao Zhang, Qing Guo, Nils Lukas, Tianwei Zhang

**Abstract**: Vision Language Model (VLM)-based agents are stateful, autonomous entities capable of perceiving and interacting with their environments through vision and language. Multi-agent systems comprise specialized agents who collaborate to solve a (complex) task. A core security property is robustness, stating that the system should maintain its integrity under adversarial attacks. However, the design of existing multi-agent systems lacks the robustness consideration, as a successful exploit against one agent can spread and infect other agents to undermine the entire system's assurance. To address this, we propose a new defense approach, Cowpox, to provably enhance the robustness of multi-agent systems. It incorporates a distributed mechanism, which improves the recovery rate of agents by limiting the expected number of infections to other agents. The core idea is to generate and distribute a special cure sample that immunizes an agent against the attack before exposure and helps recover the already infected agents. We demonstrate the effectiveness of Cowpox empirically and provide theoretical robustness guarantees.



## **38. PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN**

cs.LG

Best student paper award of ECML-PKDD 2025

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2502.12207v3) [paper-pdf](http://arxiv.org/pdf/2502.12207v3)

**Authors**: Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Silin Liao, Zhibo Jin, Flora D. Salim, Huaming Chen

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original AdvGAN.Moreover, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: https://github.com/LMBTough/PAR



## **39. Evasive Ransomware Attacks Using Low-level Behavioral Adversarial Examples**

cs.CR

\copyright 2025 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.08656v1) [paper-pdf](http://arxiv.org/pdf/2508.08656v1)

**Authors**: Manabu Hirano, Ryotaro Kobayashi

**Abstract**: Protecting state-of-the-art AI-based cybersecurity defense systems from cyber attacks is crucial. Attackers create adversarial examples by adding small changes (i.e., perturbations) to the attack features to evade or fool the deep learning model. This paper introduces the concept of low-level behavioral adversarial examples and its threat model of evasive ransomware. We formulate the method and the threat model to generate the optimal source code of evasive malware. We then examine the method using the leaked source code of Conti ransomware with the micro-behavior control function. The micro-behavior control function is our test component to simulate changing source code in ransomware; ransomware's behavior can be changed by specifying the number of threads, file encryption ratio, and delay after file encryption at the boot time. We evaluated how much an attacker can control the behavioral features of ransomware using the micro-behavior control function to decrease the detection rate of a ransomware detector.



## **40. Securing Educational LLMs: A Generalised Taxonomy of Attacks on LLMs and DREAD Risk Assessment**

cs.CY

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.08629v1) [paper-pdf](http://arxiv.org/pdf/2508.08629v1)

**Authors**: Farzana Zahid, Anjalika Sewwandi, Lee Brandon, Vimal Kumar, Roopak Sinha

**Abstract**: Due to perceptions of efficiency and significant productivity gains, various organisations, including in education, are adopting Large Language Models (LLMs) into their workflows. Educator-facing, learner-facing, and institution-facing LLMs, collectively, Educational Large Language Models (eLLMs), complement and enhance the effectiveness of teaching, learning, and academic operations. However, their integration into an educational setting raises significant cybersecurity concerns. A comprehensive landscape of contemporary attacks on LLMs and their impact on the educational environment is missing. This study presents a generalised taxonomy of fifty attacks on LLMs, which are categorized as attacks targeting either models or their infrastructure. The severity of these attacks is evaluated in the educational sector using the DREAD risk assessment framework. Our risk assessment indicates that token smuggling, adversarial prompts, direct injection, and multi-step jailbreak are critical attacks on eLLMs. The proposed taxonomy, its application in the educational environment, and our risk assessment will help academic and industrial practitioners to build resilient solutions that protect learners and institutions.



## **41. Generative AI for Critical Infrastructure in Smart Grids: A Unified Framework for Synthetic Data Generation and Anomaly Detection**

cs.CR

28 pages, 12 figures

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.08593v1) [paper-pdf](http://arxiv.org/pdf/2508.08593v1)

**Authors**: Aydin Zaboli, Junho Hong

**Abstract**: In digital substations, security events pose significant challenges to the sustained operation of power systems. To mitigate these challenges, the implementation of robust defense strategies is critically important. A thorough process of anomaly identification and detection in information and communication technology (ICT) frameworks is crucial to ensure secure and reliable communication and coordination between interconnected devices within digital substations. Hence, this paper addresses the critical cybersecurity challenges confronting IEC61850-based digital substations within modern smart grids, where the integration of advanced communication protocols, e.g., generic object-oriented substation event (GOOSE), has enhanced energy management and introduced significant vulnerabilities to cyberattacks. Focusing on the limitations of traditional anomaly detection systems (ADSs) in detecting threats, this research proposes a transformative approach by leveraging generative AI (GenAI) to develop robust ADSs. The primary contributions include the suggested advanced adversarial traffic mutation (AATM) technique to generate synthesized and balanced datasets for GOOSE messages, ensuring protocol compliance and enabling realistic zero-day attack pattern creation to address data scarcity. Then, the implementation of GenAI-based ADSs incorporating the task-oriented dialogue (ToD) processes has been explored for improved detection of attack patterns. Finally, a comparison of the GenAI-based ADS with machine learning (ML)-based ADSs has been implemented to showcase the outperformance of the GenAI-based frameworks considering the AATM-generated GOOSE datasets and standard/advanced performance evaluation metrics.



## **42. Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models**

cs.LG

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2505.15130v2) [paper-pdf](http://arxiv.org/pdf/2505.15130v2)

**Authors**: Sajjad Ghiasvand, Haniyeh Ehsani Oskouie, Mahnoosh Alizadeh, Ramtin Pedarsani

**Abstract**: Vision-Language Models (VLMs) such as CLIP have shown remarkable performance in cross-modal tasks through large-scale contrastive pre-training. To adapt these large transformer-based models efficiently for downstream tasks, Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA have emerged as scalable alternatives to full fine-tuning, especially in few-shot scenarios. However, like traditional deep neural networks, VLMs are highly vulnerable to adversarial attacks, where imperceptible perturbations can significantly degrade model performance. Adversarial training remains the most effective strategy for improving model robustness in PEFT. In this work, we propose AdvCLIP-LoRA, the first algorithm designed to enhance the adversarial robustness of CLIP models fine-tuned with LoRA in few-shot settings. Our method formulates adversarial fine-tuning as a minimax optimization problem and provides theoretical guarantees for convergence under smoothness and nonconvex-strong-concavity assumptions. Empirical results across eight datasets using ViT-B/16 and ViT-B/32 models show that AdvCLIP-LoRA significantly improves robustness against common adversarial attacks (e.g., FGSM, PGD), without sacrificing much clean accuracy. These findings highlight AdvCLIP-LoRA as a practical and theoretically grounded approach for robust adaptation of VLMs in resource-constrained settings.



## **43. VISOR: Visual Input-based Steering for Output Redirection in Vision-Language Models**

cs.CV

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08521v1) [paper-pdf](http://arxiv.org/pdf/2508.08521v1)

**Authors**: Mansi Phute, Ravikumar Balakrishnan

**Abstract**: Vision Language Models (VLMs) are increasingly being used in a broad range of applications, bringing their security and behavioral control to the forefront. While existing approaches for behavioral control or output redirection, like system prompting in VLMs, are easily detectable and often ineffective, activation-based steering vectors require invasive runtime access to model internals--incompatible with API-based services and closed-source deployments. We introduce VISOR (Visual Input-based Steering for Output Redirection), a novel method that achieves sophisticated behavioral control through optimized visual inputs alone. By crafting universal steering images that induce target activation patterns, VISOR enables practical deployment across all VLM serving modalities while remaining imperceptible compared to explicit textual instructions. We validate VISOR on LLaVA-1.5-7B across three critical alignment tasks: refusal, sycophancy and survival instinct. A single 150KB steering image matches steering vector performance within 1-2% for positive behavioral shifts while dramatically exceeding it for negative steering--achieving up to 25% shifts from baseline compared to steering vectors' modest changes. Unlike system prompting (3-4% shifts), VISOR provides robust bidirectional control while maintaining 99.9% performance on 14,000 unrelated MMLU tasks. Beyond eliminating runtime overhead and model access requirements, VISOR exposes a critical security vulnerability: adversaries can achieve sophisticated behavioral manipulation through visual channels alone, bypassing text-based defenses. Our work fundamentally re-imagines multimodal model control and highlights the urgent need for defenses against visual steering attacks.



## **44. Designing with Deception: ML- and Covert Gate-Enhanced Camouflaging to Thwart IC Reverse Engineering**

cs.CR

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08462v1) [paper-pdf](http://arxiv.org/pdf/2508.08462v1)

**Authors**: Junling Fan, David Koblah, Domenic Forte

**Abstract**: Integrated circuits (ICs) are essential to modern electronic systems, yet they face significant risks from physical reverse engineering (RE) attacks that compromise intellectual property (IP) and overall system security. While IC camouflage techniques have emerged to mitigate these risks, existing approaches largely focus on localized gate modifications, neglecting comprehensive deception strategies. To address this gap, we present a machine learning (ML)-driven methodology that integrates cryptic and mimetic cyber deception principles to enhance IC security against RE. Our approach leverages a novel And-Inverter Graph Variational Autoencoder (AIG-VAE) to encode circuit representations, enabling dual-layered camouflage through functional preservation and appearance mimicry. By introducing new variants of covert gates -- Fake Inverters, Fake Buffers, and Universal Transmitters -- our methodology achieves robust protection by obscuring circuit functionality while presenting misleading appearances. Experimental results demonstrate the effectiveness of our strategy in maintaining circuit functionality while achieving high camouflage and similarity scores with minimal structural overhead. Additionally, we validate the robustness of our method against advanced artificial intelligence (AI)-enhanced RE attacks, highlighting its practical applicability in securing IC designs. By bridging the gap in mimetic deception for hardware security, our work sets a new standard for IC camouflage, advancing the application of cyber deception principles to protect critical systems from adversarial threats.



## **45. Evaluating lightweight unsupervised online IDS for masquerade attacks in CAN**

cs.CR

22 pages, 10 figures, 4 tables. New title

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2406.13778v3) [paper-pdf](http://arxiv.org/pdf/2406.13778v3)

**Authors**: Pablo Moriano, Steven C. Hespeler, Mingyan Li, Robert A. Bridges

**Abstract**: Vehicular controller area networks (CANs) are susceptible to masquerade attacks by malicious adversaries. In masquerade attacks, adversaries silence a targeted ID and then send malicious frames with forged content at the expected timing of benign frames. As masquerade attacks could seriously harm vehicle functionality and are the stealthiest attacks to detect in CAN, recent work has devoted attention to compare frameworks for detecting masquerade attacks in CAN. However, most existing works report offline evaluations using CAN logs already collected using simulations that do not comply with the domain's real-time constraints. Here we contribute to advance the state of the art by presenting a comparative evaluation of four different non-deep learning (DL)-based unsupervised online intrusion detection systems (IDS) for masquerade attacks in CAN. Our approach differs from existing comparative evaluations in that we analyze the effect of controlling streaming data conditions in a sliding window setting. In doing so, we use realistic masquerade attacks being replayed from the ROAD dataset. We show that although evaluated IDS are not effective at detecting every attack type, the method that relies on detecting changes in the hierarchical structure of clusters of time series produces the best results at the expense of higher computational overhead. We discuss limitations, open challenges, and how the evaluated methods can be used for practical unsupervised online CAN IDS for masquerade attacks.



## **46. Selective KV-Cache Sharing to Mitigate Timing Side-Channels in LLM Inference**

cs.CR

17 pages,17 figures

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08438v1) [paper-pdf](http://arxiv.org/pdf/2508.08438v1)

**Authors**: Kexin Chu, Zecheng Lin, Dawei Xiang, Zixu Shen, Jianchang Su, Cheng Chu, Yiwei Yang, Wenhui Zhang, Wenfei Wu, Wei Zhang

**Abstract**: Global KV-cache sharing has emerged as a key optimization for accelerating large language model (LLM) inference. However, it exposes a new class of timing side-channel attacks, enabling adversaries to infer sensitive user inputs via shared cache entries. Existing defenses, such as per-user isolation, eliminate leakage but degrade performance by up to 38.9% in time-to-first-token (TTFT), making them impractical for high-throughput deployment. To address this gap, we introduce SafeKV (Secure and Flexible KV Cache Sharing), a privacy-aware KV-cache management framework that selectively shares non-sensitive entries while confining sensitive content to private caches. SafeKV comprises three components: (i) a hybrid, multi-tier detection pipeline that integrates rule-based pattern matching, a general-purpose privacy detector, and context-aware validation; (ii) a unified radix-tree index that manages public and private entries across heterogeneous memory tiers (HBM, DRAM, SSD); and (iii) entropy-based access monitoring to detect and mitigate residual information leakage. Our evaluation shows that SafeKV mitigates 94% - 97% of timing-based side-channel attacks. Compared to per-user isolation method, SafeKV improves TTFT by up to 40.58% and throughput by up to 2.66X across diverse LLMs and workloads. SafeKV reduces cache-induced TTFT overhead from 50.41% to 11.74% on Qwen3-235B. By combining fine-grained privacy control with high cache reuse efficiency, SafeKV reclaims the performance advantages of global sharing while providing robust runtime privacy guarantees for LLM inference.



## **47. Towards Effective MLLM Jailbreaking Through Balanced On-Topicness and OOD-Intensity**

cs.CV

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.09218v1) [paper-pdf](http://arxiv.org/pdf/2508.09218v1)

**Authors**: Zuoou Li, Weitong Zhang, Jingyuan Wang, Shuyuan Zhang, Wenjia Bai, Bernhard Kainz, Mengyun Qiao

**Abstract**: Multimodal large language models (MLLMs) are widely used in vision-language reasoning tasks. However, their vulnerability to adversarial prompts remains a serious concern, as safety mechanisms often fail to prevent the generation of harmful outputs. Although recent jailbreak strategies report high success rates, many responses classified as "successful" are actually benign, vague, or unrelated to the intended malicious goal. This mismatch suggests that current evaluation standards may overestimate the effectiveness of such attacks. To address this issue, we introduce a four-axis evaluation framework that considers input on-topicness, input out-of-distribution (OOD) intensity, output harmfulness, and output refusal rate. This framework identifies truly effective jailbreaks. In a substantial empirical study, we reveal a structural trade-off: highly on-topic prompts are frequently blocked by safety filters, whereas those that are too OOD often evade detection but fail to produce harmful content. However, prompts that balance relevance and novelty are more likely to evade filters and trigger dangerous output. Building on this insight, we develop a recursive rewriting strategy called Balanced Structural Decomposition (BSD). The approach restructures malicious prompts into semantically aligned sub-tasks, while introducing subtle OOD signals and visual cues that make the inputs harder to detect. BSD was tested across 13 commercial and open-source MLLMs, where it consistently led to higher attack success rates, more harmful outputs, and fewer refusals. Compared to previous methods, it improves success rates by $67\%$ and harmfulness by $21\%$, revealing a previously underappreciated weakness in current multimodal safety systems.



## **48. Adaptive Learning for IRS-Assisted Wireless Networks: Securing Opportunistic Communications Against Byzantine Eavesdroppers**

eess.SP

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08206v1) [paper-pdf](http://arxiv.org/pdf/2508.08206v1)

**Authors**: Amirhossein Taherpour, Abbas Taherpour, Tamer Khattab

**Abstract**: We propose a joint learning framework for Byzantine-resilient spectrum sensing and secure intelligent reflecting surface (IRS)--assisted opportunistic access under channel state information (CSI) uncertainty. The sensing stage performs logit-domain Bayesian updates with trimmed aggregation and attention-weighted consensus, and the base station (BS) fuses network beliefs with a conservative minimum rule, preserving detection accuracy under a bounded number of Byzantine users. Conditioned on the sensing outcome, we pose downlink design as sum mean-squared error (MSE) minimization under transmit-power and signal-leakage constraints and jointly optimize the BS precoder, IRS phase shifts, and user equalizers. With partial (or known) CSI, we develop an augmented-Lagrangian alternating algorithm with projected updates and provide provable sublinear convergence, with accelerated rates under mild local curvature. With unknown CSI, we perform constrained Bayesian optimization (BO) in a geometry-aware low-dimensional latent space using Gaussian process (GP) surrogates; we prove regret bounds for a constrained upper confidence bound (UCB) variant of the BO module, and demonstrate strong empirical performance of the implemented procedure. Simulations across diverse network conditions show higher detection probability at fixed false-alarm rate under adversarial attacks, large reductions in sum MSE for honest users, strong suppression of eavesdropper signal power, and fast convergence. The framework offers a practical path to secure opportunistic communication that adapts to CSI availability while coherently coordinating sensing and transmission through joint learning.



## **49. Robust Anomaly Detection in O-RAN: Leveraging LLMs against Data Manipulation Attacks**

cs.CR

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08029v1) [paper-pdf](http://arxiv.org/pdf/2508.08029v1)

**Authors**: Thusitha Dayaratne, Ngoc Duy Pham, Viet Vo, Shangqi Lai, Sharif Abuadbba, Hajime Suzuki, Xingliang Yuan, Carsten Rudolph

**Abstract**: The introduction of 5G and the Open Radio Access Network (O-RAN) architecture has enabled more flexible and intelligent network deployments. However, the increased complexity and openness of these architectures also introduce novel security challenges, such as data manipulation attacks on the semi-standardised Shared Data Layer (SDL) within the O-RAN platform through malicious xApps. In particular, malicious xApps can exploit this vulnerability by introducing subtle Unicode-wise alterations (hypoglyphs) into the data that are being used by traditional machine learning (ML)-based anomaly detection methods. These Unicode-wise manipulations can potentially bypass detection and cause failures in anomaly detection systems based on traditional ML, such as AutoEncoders, which are unable to process hypoglyphed data without crashing. We investigate the use of Large Language Models (LLMs) for anomaly detection within the O-RAN architecture to address this challenge. We demonstrate that LLM-based xApps maintain robust operational performance and are capable of processing manipulated messages without crashing. While initial detection accuracy requires further improvements, our results highlight the robustness of LLMs to adversarial attacks such as hypoglyphs in input data. There is potential to use their adaptability through prompt engineering to further improve the accuracy, although this requires further research. Additionally, we show that LLMs achieve low detection latency (under 0.07 seconds), making them suitable for Near-Real-Time (Near-RT) RIC deployments.



## **50. Universally Unfiltered and Unseen:Input-Agnostic Multimodal Jailbreaks against Text-to-Image Model Safeguards**

cs.CR

This paper has been accepted by ACM MM 2025

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.05658v2) [paper-pdf](http://arxiv.org/pdf/2508.05658v2)

**Authors**: Song Yan, Hui Wei, Jinlong Fei, Guoliang Yang, Zhengyu Zhao, Zheng Wang

**Abstract**: Various (text) prompt filters and (image) safety checkers have been implemented to mitigate the misuse of Text-to-Image (T2I) models in creating Not-Safe-For-Work (NSFW) content. In order to expose potential security vulnerabilities of such safeguards, multimodal jailbreaks have been studied. However, existing jailbreaks are limited to prompt-specific and image-specific perturbations, which suffer from poor scalability and time-consuming optimization. To address these limitations, we propose Universally Unfiltered and Unseen (U3)-Attack, a multimodal jailbreak attack method against T2I safeguards. Specifically, U3-Attack optimizes an adversarial patch on the image background to universally bypass safety checkers and optimizes a safe paraphrase set from a sensitive word to universally bypass prompt filters while eliminating redundant computations. Extensive experimental results demonstrate the superiority of our U3-Attack on both open-source and commercial T2I models. For example, on the commercial Runway-inpainting model with both prompt filter and safety checker, our U3-Attack achieves $~4\times$ higher success rates than the state-of-the-art multimodal jailbreak attack, MMA-Diffusion.



