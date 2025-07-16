# Latest Adversarial Attack Papers
**update at 2025-07-16 09:47:30**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. A Generative Approach to LLM Harmfulness Detection with Special Red Flag Tokens**

cs.CL

14 pages, 6 figures

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2502.16366v3) [paper-pdf](http://arxiv.org/pdf/2502.16366v3)

**Authors**: Sophie Xhonneux, David Dobre, Mehrnaz Mofakhami, Leo Schwinn, Gauthier Gidel

**Abstract**: Most safety training methods for large language models (LLMs) are based on fine-tuning that forces models to shift from an unsafe answer to refusal when faced with harmful requests. Unfortunately, these drastic distribution shifts generally compromise model capabilities. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to train the model to insert this token into its response at any time when harmful content is generated or about to be generated. Our approach offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer and provides robustness as good as adversarial training without the need to run attacks during training. Moreover, by encapsulating our safety tuning in a LoRA module, we provide additional defenses against fine-tuning API attacks.



## **2. Robustifying 3D Perception via Least-Squares Graphs for Multi-Agent Object Tracking**

cs.CV

6 pages, 3 figures, 4 tables

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.04762v2) [paper-pdf](http://arxiv.org/pdf/2507.04762v2)

**Authors**: Maria Damanaki, Ioulia Kapsali, Nikos Piperigkos, Alexandros Gkillas, Aris S. Lalos

**Abstract**: The critical perception capabilities of EdgeAI systems, such as autonomous vehicles, are required to be resilient against adversarial threats, by enabling accurate identification and localization of multiple objects in the scene over time, mitigating their impact. Single-agent tracking offers resilience to adversarial attacks but lacks situational awareness, underscoring the need for multi-agent cooperation to enhance context understanding and robustness. This paper proposes a novel mitigation framework on 3D LiDAR scene against adversarial noise by tracking objects based on least-squares graph on multi-agent adversarial bounding boxes. Specifically, we employ the least-squares graph tool to reduce the induced positional error of each detection's centroid utilizing overlapped bounding boxes on a fully connected graph via differential coordinates and anchor points. Hence, the multi-vehicle detections are fused and refined mitigating the adversarial impact, and associated with existing tracks in two stages performing tracking to further suppress the adversarial threat. An extensive evaluation study on the real-world V2V4Real dataset demonstrates that the proposed method significantly outperforms both state-of-the-art single and multi-agent tracking frameworks by up to 23.3% under challenging adversarial conditions, operating as a resilient approach without relying on additional defense mechanisms.



## **3. Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks**

cs.LG

Published in TMLR. Best Paper Award at the AdvML-Frontiers @ NeurIPS  2024 workshop. Code available at https://github.com/saper0/qpcert

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2407.10867v3) [paper-pdf](http://arxiv.org/pdf/2407.10867v3)

**Authors**: Lukas Gosch, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Stephan Günnemann

**Abstract**: Generalization of machine learning models can be severely compromised by data poisoning, where adversarial changes are applied to the training data. This vulnerability has led to interest in certifying (i.e., proving) that such changes up to a certain magnitude do not affect test predictions. We, for the first time, certify Graph Neural Networks (GNNs) against poisoning attacks, including backdoors, targeting the node features of a given graph. Our certificates are white-box and based upon $(i)$ the neural tangent kernel, which characterizes the training dynamics of sufficiently wide networks; and $(ii)$ a novel reformulation of the bilevel optimization problem describing poisoning as a mixed-integer linear program. Consequently, we leverage our framework to provide fundamental insights into the role of graph structure and its connectivity on the worst-case robustness behavior of convolution-based and PageRank-based GNNs. We note that our framework is more general and constitutes the first approach to derive white-box poisoning certificates for NNs, which can be of independent interest beyond graph-related tasks.



## **4. Real-Time Bayesian Detection of Drift-Evasive GNSS Spoofing in Reinforcement Learning Based UAV Deconfliction**

cs.LG

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11173v1) [paper-pdf](http://arxiv.org/pdf/2507.11173v1)

**Authors**: Deepak Kumar Panda, Weisi Guo

**Abstract**: Autonomous unmanned aerial vehicles (UAVs) rely on global navigation satellite system (GNSS) pseudorange measurements for accurate real-time localization and navigation. However, this dependence exposes them to sophisticated spoofing threats, where adversaries manipulate pseudoranges to deceive UAV receivers. Among these, drift-evasive spoofing attacks subtly perturb measurements, gradually diverting the UAVs trajectory without triggering conventional signal-level anti-spoofing mechanisms. Traditional distributional shift detection techniques often require accumulating a threshold number of samples, causing delays that impede rapid detection and timely response. Consequently, robust temporal-scale detection methods are essential to identify attack onset and enable contingency planning with alternative sensing modalities, improving resilience against stealthy adversarial manipulations. This study explores a Bayesian online change point detection (BOCPD) approach that monitors temporal shifts in value estimates from a reinforcement learning (RL) critic network to detect subtle behavioural deviations in UAV navigation. Experimental results show that this temporal value-based framework outperforms conventional GNSS spoofing detectors, temporal semi-supervised learning frameworks, and the Page-Hinkley test, achieving higher detection accuracy and lower false-positive and false-negative rates for drift-evasive spoofing attacks.



## **5. Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs**

cs.CL

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11112v1) [paper-pdf](http://arxiv.org/pdf/2507.11112v1)

**Authors**: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier

**Abstract**: Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.



## **6. The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs**

cs.CL

21 pages, 9 figures, work in progress

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11097v1) [paper-pdf](http://arxiv.org/pdf/2507.11097v1)

**Authors**: Zichen Wen, Jiashu Qu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, Linfeng Zhang

**Abstract**: Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA.



## **7. Crafting Imperceptible On-Manifold Adversarial Attacks for Tabular Data**

cs.LG

32 pages

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.10998v1) [paper-pdf](http://arxiv.org/pdf/2507.10998v1)

**Authors**: Zhipeng He, Alexander Stevens, Chun Ouyang, Johannes De Smedt, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks on tabular data present fundamental challenges distinct from image or text domains due to the heterogeneous nature of mixed categorical and numerical features. Unlike images where pixel perturbations maintain visual similarity, tabular data lacks intuitive similarity metrics, making it difficult to define imperceptible modifications. Additionally, traditional gradient-based methods prioritise $\ell_p$-norm constraints, often producing adversarial examples that deviate from the original data distributions, making them detectable. We propose a latent space perturbation framework using a mixed-input Variational Autoencoder (VAE) to generate imperceptible adversarial examples. The proposed VAE integrates categorical embeddings and numerical features into a unified latent manifold, enabling perturbations that preserve statistical consistency. We specify In-Distribution Success Rate (IDSR) to measure the proportion of adversarial examples that remain statistically indistinguishable from the input distribution. Evaluation across six publicly available datasets and three model architectures demonstrates that our method achieves substantially lower outlier rates and more consistent performance compared to traditional input-space attacks and other VAE-based methods adapted from image domain approaches. Our comprehensive analysis includes hyperparameter sensitivity, sparsity control mechanisms, and generative architectural comparisons, revealing that VAE-based attacks depend critically on reconstruction quality but offer superior practical utility when sufficient training data is available. This work highlights the importance of on-manifold perturbations for realistic adversarial attacks on tabular data, offering a robust approach for practical deployment. The source code can be accessed through https://github.com/ZhipengHe/VAE-TabAttack.



## **8. Representation Bending for Large Language Model Safety**

cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2504.01550v3) [paper-pdf](http://arxiv.org/pdf/2504.01550v3)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.



## **9. A Survey on Speech Deepfake Detection**

cs.SD

38 pages. This paper has been accepted by ACM Computing Surveys

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2404.13914v2) [paper-pdf](http://arxiv.org/pdf/2404.13914v2)

**Authors**: Menglu Li, Yasaman Ahmadiadli, Xiao-Ping Zhang

**Abstract**: The availability of smart devices leads to an exponential increase in multimedia content. However, advancements in deep learning have also enabled the creation of highly sophisticated Deepfake content, including speech Deepfakes, which pose a serious threat by generating realistic voices and spreading misinformation. To combat this, numerous challenges have been organized to advance speech Deepfake detection techniques. In this survey, we systematically analyze more than 200 papers published up to March 2024. We provide a comprehensive review of each component in the detection pipeline, including model architectures, optimization techniques, generalizability, evaluation metrics, performance comparisons, available datasets, and open source availability. For each aspect, we assess recent progress and discuss ongoing challenges. In addition, we explore emerging topics such as partial Deepfake detection, cross-dataset evaluation, and defences against adversarial attacks, while suggesting promising research directions. This survey not only identifies the current state of the art to establish strong baselines for future experiments but also offers clear guidance for researchers aiming to enhance speech Deepfake detection systems.



## **10. REAL-IoT: Characterizing GNN Intrusion Detection Robustness under Practical Adversarial Attack**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10836v1) [paper-pdf](http://arxiv.org/pdf/2507.10836v1)

**Authors**: Zhonghao Zhan, Huichi Zhou, Hamed Haddadi

**Abstract**: Graph Neural Network (GNN)-based network intrusion detection systems (NIDS) are often evaluated on single datasets, limiting their ability to generalize under distribution drift. Furthermore, their adversarial robustness is typically assessed using synthetic perturbations that lack realism. This measurement gap leads to an overestimation of GNN-based NIDS resilience. To address the limitations, we propose \textbf{REAL-IoT}, a comprehensive framework for robustness evaluation of GNN-based NIDS in IoT environments. Our framework presents a methodology that creates a unified dataset from canonical datasets to assess generalization under drift. In addition, it features a novel intrusion dataset collected from a physical IoT testbed, which captures network traffic and attack scenarios under real-world settings. Furthermore, using REAL-IoT, we explore the usage of Large Language Models (LLMs) to analyze network data and mitigate the impact of adversarial examples by filtering suspicious flows. Our evaluations using REAL-IoT reveal performance drops in GNN models compared to results from standard benchmarks, quantifying their susceptibility to drift and realistic attacks. We also demonstrate the potential of LLM-based filtering to enhance robustness. These findings emphasize the necessity of realistic threat modeling and rigorous measurement practices for developing resilient IoT intrusion detection systems.



## **11. Investigating Adversarial Attacks in Software Analytics via Machine Learning Explainability**

cs.SE

This paper has been accepted for publication in Software Quality  Journal

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2408.04124v2) [paper-pdf](http://arxiv.org/pdf/2408.04124v2)

**Authors**: MD Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: With the recent advancements in machine learning (ML), numerous ML-based approaches have been extensively applied in software analytics tasks to streamline software development and maintenance processes. Nevertheless, studies indicate that despite their potential usefulness, ML models are vulnerable to adversarial attacks, which may result in significant monetary losses in these processes. As a result, the ML models' robustness against adversarial attacks must be assessed before they are deployed in software analytics tasks. Despite several techniques being available for adversarial attacks in software analytics tasks, exploring adversarial attacks using ML explainability is largely unexplored. Therefore, this study aims to investigate the relationship between ML explainability and adversarial attacks to measure the robustness of ML models in software analytics tasks. In addition, unlike most existing attacks that directly perturb input-space, our attack approach focuses on perturbing feature-space. Our extensive experiments, involving six datasets, three ML explainability techniques, and seven ML models, demonstrate that ML explainability can be used to conduct successful adversarial attacks on ML models in software analytics tasks. This is achieved by modifying only the top 1-3 important features identified by ML explainability techniques. Consequently, the ML models under attack fail to accurately predict up to 86.6% of instances that were correctly predicted before adversarial attacks, indicating the models' low robustness against such attacks. Finally, our proposed technique demonstrates promising results compared to four state-of-the-art adversarial attack techniques targeting tabular data.



## **12. BURN: Backdoor Unlearning via Adversarial Boundary Analysis**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10491v1) [paper-pdf](http://arxiv.org/pdf/2507.10491v1)

**Authors**: Yanghao Su, Jie Zhang, Yiming Li, Tianwei Zhang, Qing Guo, Weiming Zhang, Nenghai Yu, Nils Lukas, Wenbo Zhou

**Abstract**: Backdoor unlearning aims to remove backdoor-related information while preserving the model's original functionality. However, existing unlearning methods mainly focus on recovering trigger patterns but fail to restore the correct semantic labels of poison samples. This limitation prevents them from fully eliminating the false correlation between the trigger pattern and the target label. To address this, we leverage boundary adversarial attack techniques, revealing two key observations. First, poison samples exhibit significantly greater distances from decision boundaries compared to clean samples, indicating they require larger adversarial perturbations to change their predictions. Second, while adversarial predicted labels for clean samples are uniformly distributed, those for poison samples tend to revert to their original correct labels. Moreover, the features of poison samples restore to closely resemble those of corresponding clean samples after adding adversarial perturbations. Building upon these insights, we propose Backdoor Unlearning via adversaRial bouNdary analysis (BURN), a novel defense framework that integrates false correlation decoupling, progressive data refinement, and model purification. In the first phase, BURN employs adversarial boundary analysis to detect poisoned samples based on their abnormal adversarial boundary distances, then restores their correct semantic labels for fine-tuning. In the second phase, it employs a feedback mechanism that tracks prediction discrepancies between the original backdoored model and progressively sanitized models, guiding both dataset refinement and model purification. Extensive evaluations across multiple datasets, architectures, and seven diverse backdoor attack types confirm that BURN effectively removes backdoor threats while maintaining the model's original performance.



## **13. Some remarks on gradient dominance and LQR policy optimization**

cs.LG

This is a short paper summarizing the first part of the slides  presented at my keynote at the 2025 L4DC (Learning for Dynamics & Control  Conference) in Ann Arbor, Michigan, 05 June 2025. A partial bibliography has  been added. A second part on neural net feedback controllers is to be added

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10452v1) [paper-pdf](http://arxiv.org/pdf/2507.10452v1)

**Authors**: Eduardo D. Sontag

**Abstract**: Solutions of optimization problems, including policy optimization in reinforcement learning, typically rely upon some variant of gradient descent. There has been much recent work in the machine learning, control, and optimization communities applying the Polyak-{\L}ojasiewicz Inequality (PLI) to such problems in order to establish an exponential rate of convergence (a.k.a. ``linear convergence'' in the local-iteration language of numerical analysis) of loss functions to their minima under the gradient flow. Often, as is the case of policy iteration for the continuous-time LQR problem, this rate vanishes for large initial conditions, resulting in a mixed globally linear / locally exponential behavior. This is in sharp contrast with the discrete-time LQR problem, where there is global exponential convergence. That gap between CT and DT behaviors motivates the search for various generalized PLI-like conditions, and this talk will address that topic. Moreover, these generalizations are key to understanding the transient and asymptotic effects of errors in the estimation of the gradient, errors which might arise from adversarial attacks, wrong evaluation by an oracle, early stopping of a simulation, inaccurate and very approximate digital twins, stochastic computations (algorithm ``reproducibility''), or learning by sampling from limited data. We describe an ``input to state stability'' (ISS) analysis of this issue. The lecture also discussed convergence and PLI-like properties of ``linear feedforward neural networks'' in feedback control, but this arXiv skips that part (to be updated). Much of the work described here was done in collaboration with Arthur Castello B. de Oliveira, Leilei Cui, Zhong-Ping Jiang, and Milad Siami.



## **14. Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems**

cs.CR

14 pages, 5 figures, 11 tables. To be published in LLMSec 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2504.11168v3) [paper-pdf](http://arxiv.org/pdf/2504.11168v3)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.



## **15. SCOOTER: A Human Evaluation Framework for Unrestricted Adversarial Examples**

cs.CV

42 pages, 16 figures, 11 tables, Under Review, Code:  https://github.com/DrenFazlija/Scooter, Data:  https://doi.org/10.5281/zenodo.15771501

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.07776v2) [paper-pdf](http://arxiv.org/pdf/2507.07776v2)

**Authors**: Dren Fazlija, Monty-Maximilian Zühlke, Johanna Schrader, Arkadij Orlov, Clara Stein, Iyiola E. Olatunji, Daniel Kudenko

**Abstract**: Unrestricted adversarial attacks aim to fool computer vision models without being constrained by $\ell_p$-norm bounds to remain imperceptible to humans, for example, by changing an object's color. This allows attackers to circumvent traditional, norm-bounded defense strategies such as adversarial training or certified defense strategies. However, due to their unrestricted nature, there are also no guarantees of norm-based imperceptibility, necessitating human evaluations to verify just how authentic these adversarial examples look. While some related work assesses this vital quality of adversarial attacks, none provide statistically significant insights. This issue necessitates a unified framework that supports and streamlines such an assessment for evaluating and comparing unrestricted attacks. To close this gap, we introduce SCOOTER - an open-source, statistically powered framework for evaluating unrestricted adversarial examples. Our contributions are: $(i)$ best-practice guidelines for crowd-study power, compensation, and Likert equivalence bounds to measure imperceptibility; $(ii)$ the first large-scale human vs. model comparison across 346 human participants showing that three color-space attacks and three diffusion-based attacks fail to produce imperceptible images. Furthermore, we found that GPT-4o can serve as a preliminary test for imperceptibility, but it only consistently detects adversarial examples for four out of six tested attacks; $(iii)$ open-source software tools, including a browser-based task template to collect annotations and analysis scripts in Python and R; $(iv)$ an ImageNet-derived benchmark dataset containing 3K real images, 7K adversarial examples, and over 34K human ratings. Our findings demonstrate that automated vision systems do not align with human perception, reinforcing the need for a ground-truth SCOOTER benchmark.



## **16. Bridging Robustness and Generalization Against Word Substitution Attacks in NLP via the Growth Bound Matrix Approach**

cs.CL

Accepted to ACL Findings 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10330v1) [paper-pdf](http://arxiv.org/pdf/2507.10330v1)

**Authors**: Mohammed Bouri, Adnane Saoud

**Abstract**: Despite advancements in Natural Language Processing (NLP), models remain vulnerable to adversarial attacks, such as synonym substitutions. While prior work has focused on improving robustness for feed-forward and convolutional architectures, the robustness of recurrent networks and modern state space models (SSMs), such as S4, remains understudied. These architectures pose unique challenges due to their sequential processing and complex parameter dynamics. In this paper, we introduce a novel regularization technique based on Growth Bound Matrices (GBM) to improve NLP model robustness by reducing the impact of input perturbations on model outputs. We focus on computing the GBM for three architectures: Long Short-Term Memory (LSTM), State Space models (S4), and Convolutional Neural Networks (CNN). Our method aims to (1) enhance resilience against word substitution attacks, (2) improve generalization on clean text, and (3) providing the first systematic analysis of SSM (S4) robustness. Extensive experiments across multiple architectures and benchmark datasets demonstrate that our method improves adversarial robustness by up to 8.8% over existing baselines. These results highlight the effectiveness of our approach, outperforming several state-of-the-art methods in adversarial defense. Codes are available at https://github.com/BouriMohammed/GBM



## **17. Kaleidoscopic Background Attack: Disrupting Pose Estimation with Multi-Fold Radial Symmetry Textures**

cs.CV

Accepted at ICCV 2025. Project page is available at  https://wakuwu.github.io/KBA

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10265v1) [paper-pdf](http://arxiv.org/pdf/2507.10265v1)

**Authors**: Xinlong Ding, Hongwei Yu, Jiawei Li, Feifan Li, Yu Shang, Bochao Zou, Huimin Ma, Jiansheng Chen

**Abstract**: Camera pose estimation is a fundamental computer vision task that is essential for applications like visual localization and multi-view stereo reconstruction. In the object-centric scenarios with sparse inputs, the accuracy of pose estimation can be significantly influenced by background textures that occupy major portions of the images across different viewpoints. In light of this, we introduce the Kaleidoscopic Background Attack (KBA), which uses identical segments to form discs with multi-fold radial symmetry. These discs maintain high similarity across different viewpoints, enabling effective attacks on pose estimation models even with natural texture segments. Additionally, a projected orientation consistency loss is proposed to optimize the kaleidoscopic segments, leading to significant enhancement in the attack effectiveness. Experimental results show that optimized adversarial kaleidoscopic backgrounds can effectively attack various camera pose estimation models.



## **18. Transferring Styles for Reduced Texture Bias and Improved Robustness in Semantic Segmentation Networks**

cs.CV

accepted at ECAI 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10239v1) [paper-pdf](http://arxiv.org/pdf/2507.10239v1)

**Authors**: Ben Hamscher, Edgar Heinert, Annika Mütze, Kira Maag, Matthias Rottmann

**Abstract**: Recent research has investigated the shape and texture biases of deep neural networks (DNNs) in image classification which influence their generalization capabilities and robustness. It has been shown that, in comparison to regular DNN training, training with stylized images reduces texture biases in image classification and improves robustness with respect to image corruptions. In an effort to advance this line of research, we examine whether style transfer can likewise deliver these two effects in semantic segmentation. To this end, we perform style transfer with style varying across artificial image areas. Those random areas are formed by a chosen number of Voronoi cells. The resulting style-transferred data is then used to train semantic segmentation DNNs with the objective of reducing their dependence on texture cues while enhancing their reliance on shape-based features. In our experiments, it turns out that in semantic segmentation, style transfer augmentation reduces texture bias and strongly increases robustness with respect to common image corruptions as well as adversarial attacks. These observations hold for convolutional neural networks and transformer architectures on the Cityscapes dataset as well as on PASCAL Context, showing the generality of the proposed method.



## **19. HASSLE: A Self-Supervised Learning Enhanced Hijacking Attack on Vertical Federated Learning**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10162v1) [paper-pdf](http://arxiv.org/pdf/2507.10162v1)

**Authors**: Weiyang He, Chip-Hong Chang

**Abstract**: Vertical Federated Learning (VFL) enables an orchestrating active party to perform a machine learning task by cooperating with passive parties that provide additional task-related features for the same training data entities. While prior research has leveraged the privacy vulnerability of VFL to compromise its integrity through a combination of label inference and backdoor attacks, their effectiveness is constrained by the low label inference precision and suboptimal backdoor injection conditions. To facilitate a more rigorous security evaluation on VFL without these limitations, we propose HASSLE, a hijacking attack framework composed of a gradient-direction-based label inference module and an adversarial embedding generation algorithm enhanced by self-supervised learning. HASSLE accurately identifies private samples associated with a targeted label using only a single known instance of that label. In the two-party scenario, it demonstrates strong performance with an attack success rate (ASR) of over 99% across four datasets, including both image and tabular modalities, and achieves 85% ASR on the more complex CIFAR-100 dataset. Evaluation of HASSLE against 8 potential defenses further highlights its significant threat while providing new insights into building a trustworthy VFL system.



## **20. Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks**

cs.SE

Accepted to ICSME 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10054v1) [paper-pdf](http://arxiv.org/pdf/2507.10054v1)

**Authors**: Emir Bosnak, Sahand Moslemi, Mayasah Lami, Anil Koyuncu

**Abstract**: Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities or adversarial prompting techniques, this study examines a more direct threat scenario: open-source LLMs generating vulnerable code when prompted either directly or indirectly. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and directness across structured templates; and (2) Reverse Prompting, which derives prompts from real vulnerable code samples to assess vulnerability reproduction accuracy. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, and Gemma) using ESBMC static analysis to assess both the presence of vulnerabilities and the correctness of the generated vulnerability type. Results show all models frequently produce vulnerable outputs, with Qwen2 achieving highest correctness rates. User persona significantly affects success, where student personas achieved higher vulnerability rates than professional roles, while direct prompts were marginally more effective. Vulnerability reproduction followed an inverted-U pattern with cyclomatic complexity, peaking at moderate ranges. Our findings expose limitations of safety mechanisms in open-source models, particularly for seemingly benign educational requests.



## **21. 3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving**

cs.CV

Submitted to WACV 2026

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.09993v1) [paper-pdf](http://arxiv.org/pdf/2507.09993v1)

**Authors**: Yixun Zhang, Lizhi Wang, Junjun Zhao, Wending Zhao, Feng Zhou, Yonghao Dang, Jianqin Yin

**Abstract**: Camera-based object detection systems play a vital role in autonomous driving, yet they remain vulnerable to adversarial threats in real-world environments. While existing 2D and 3D physical attacks typically optimize texture, they often struggle to balance physical realism and attack robustness. In this work, we propose 3D Gaussian-based Adversarial Attack (3DGAA), a novel adversarial object generation framework that leverages the full 14-dimensional parameterization of 3D Gaussian Splatting (3DGS) to jointly optimize geometry and appearance in physically realizable ways. Unlike prior works that rely on patches or texture, 3DGAA jointly perturbs both geometric attributes (shape, scale, rotation) and appearance attributes (color, opacity) to produce physically realistic and transferable adversarial objects. We further introduce a physical filtering module to preserve geometric fidelity, and a physical augmentation module to simulate complex physical scenarios, thus enhancing attack generalization under real-world conditions. We evaluate 3DGAA on both virtual benchmarks and physical-world setups using miniature vehicle models. Experimental results show that 3DGAA achieves to reduce the detection mAP from 87.21% to 7.38%, significantly outperforming existing 3D physical attacks. Moreover, our method maintains high transferability across different physical conditions, demonstrating a new state-of-the-art in physically realizable adversarial attacks. These results validate 3DGAA as a practical attack framework for evaluating the safety of perception systems in autonomous driving.



## **22. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

cs.CL

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2501.03940v3) [paper-pdf](http://arxiv.org/pdf/2501.03940v3)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.



## **23. EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective**

cs.SE

20 pages, 11 figures

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2505.12185v3) [paper-pdf](http://arxiv.org/pdf/2505.12185v3)

**Authors**: Sen Fang, Weiyuan Ding, Bowen Xu

**Abstract**: Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.



## **24. AdvGrasp: Adversarial Attacks on Robotic Grasping from a Physical Perspective**

cs.RO

IJCAI'2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.09857v1) [paper-pdf](http://arxiv.org/pdf/2507.09857v1)

**Authors**: Xiaofei Wang, Mingliang Han, Tianyu Hao, Cegang Li, Yunbo Zhao, Keke Tang

**Abstract**: Adversarial attacks on robotic grasping provide valuable insights into evaluating and improving the robustness of these systems. Unlike studies that focus solely on neural network predictions while overlooking the physical principles of grasping, this paper introduces AdvGrasp, a framework for adversarial attacks on robotic grasping from a physical perspective. Specifically, AdvGrasp targets two core aspects: lift capability, which evaluates the ability to lift objects against gravity, and grasp stability, which assesses resistance to external disturbances. By deforming the object's shape to increase gravitational torque and reduce stability margin in the wrench space, our method systematically degrades these two key grasping metrics, generating adversarial objects that compromise grasp performance. Extensive experiments across diverse scenarios validate the effectiveness of AdvGrasp, while real-world validations demonstrate its robustness and practical applicability



## **25. Game Theory Meets LLM and Agentic AI: Reimagining Cybersecurity for the Age of Intelligent Threats**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10621v1) [paper-pdf](http://arxiv.org/pdf/2507.10621v1)

**Authors**: Quanyan Zhu

**Abstract**: Protecting cyberspace requires not only advanced tools but also a shift in how we reason about threats, trust, and autonomy. Traditional cybersecurity methods rely on manual responses and brittle heuristics. To build proactive and intelligent defense systems, we need integrated theoretical frameworks and software tools. Game theory provides a rigorous foundation for modeling adversarial behavior, designing strategic defenses, and enabling trust in autonomous systems. Meanwhile, software tools process cyber data, visualize attack surfaces, verify compliance, and suggest mitigations. Yet a disconnect remains between theory and practical implementation.   The rise of Large Language Models (LLMs) and agentic AI offers a new path to bridge this gap. LLM-powered agents can operationalize abstract strategies into real-world decisions. Conversely, game theory can inform the reasoning and coordination of these agents across complex workflows. LLMs also challenge classical game-theoretic assumptions, such as perfect rationality or static payoffs, prompting new models aligned with cognitive and computational realities. This co-evolution promises richer theoretical foundations and novel solution concepts. Agentic AI also reshapes software design: systems must now be modular, adaptive, and trust-aware from the outset.   This chapter explores the intersection of game theory, agentic AI, and cybersecurity. We review key game-theoretic frameworks (e.g., static, dynamic, Bayesian, and signaling games) and solution concepts. We then examine how LLM agents can enhance cyber defense and introduce LLM-driven games that embed reasoning into AI agents. Finally, we explore multi-agent workflows and coordination games, outlining how this convergence fosters secure, intelligent, and adaptive cyber systems.



## **26. Concept Steerers: Leveraging K-Sparse Autoencoders for Test-Time Controllable Generations**

cs.CV

23 pages, 18 figures

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2501.19066v2) [paper-pdf](http://arxiv.org/pdf/2501.19066v2)

**Authors**: Dahye Kim, Deepti Ghadiyaram

**Abstract**: Despite the remarkable progress in text-to-image generative models, they are prone to adversarial attacks and inadvertently generate unsafe, unethical content. Existing approaches often rely on fine-tuning models to remove specific concepts, which is computationally expensive, lacks scalability, and/or compromises generation quality. In this work, we propose a novel framework leveraging k-sparse autoencoders (k-SAEs) to enable efficient and interpretable concept manipulation in diffusion models. Specifically, we first identify interpretable monosemantic concepts in the latent space of text embeddings and leverage them to precisely steer the generation away or towards a given concept (e.g., nudity) or to introduce a new concept (e.g., photographic style) -- all during test time. Through extensive experiments, we demonstrate that our approach is very simple, requires no retraining of the base model nor LoRA adapters, does not compromise the generation quality, and is robust to adversarial prompt manipulations. Our method yields an improvement of $\mathbf{20.01\%}$ in unsafe concept removal, is effective in style manipulation, and is $\mathbf{\sim5}$x faster than the current state-of-the-art. Code is available at: https://github.com/kim-dahye/steerers



## **27. Adversarial Activation Patching: A Framework for Detecting and Mitigating Emergent Deception in Safety-Aligned Transformers**

cs.LG

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2507.09406v1) [paper-pdf](http://arxiv.org/pdf/2507.09406v1)

**Authors**: Santhosh Kumar Ravindran

**Abstract**: Large language models (LLMs) aligned for safety through techniques like reinforcement learning from human feedback (RLHF) often exhibit emergent deceptive behaviors, where outputs appear compliant but subtly mislead or omit critical information. This paper introduces adversarial activation patching, a novel mechanistic interpretability framework that leverages activation patching as an adversarial tool to induce, detect, and mitigate such deception in transformer-based models. By sourcing activations from "deceptive" prompts and patching them into safe forward passes at specific layers, we simulate vulnerabilities and quantify deception rates. Through toy neural network simulations across multiple scenarios (e.g., 1000 trials per setup), we demonstrate that adversarial patching increases deceptive outputs to 23.9% from a 0% baseline, with layer-specific variations supporting our hypotheses. We propose six hypotheses, including transferability across models, exacerbation in multimodal settings, and scaling effects. An expanded literature review synthesizes over 20 key works in interpretability, deception, and adversarial attacks. Mitigation strategies, such as activation anomaly detection and robust fine-tuning, are detailed, alongside ethical considerations and future research directions. This work advances AI safety by highlighting patching's dual-use potential and provides a roadmap for empirical studies on large-scale models.



## **28. Single Word Change is All You Need: Using LLMs to Create Synthetic Training Examples for Text Classifiers**

cs.CL

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2401.17196v3) [paper-pdf](http://arxiv.org/pdf/2401.17196v3)

**Authors**: Lei Xu, Sarah Alnegheimish, Laure Berti-Equille, Alfredo Cuesta-Infante, Kalyan Veeramachaneni

**Abstract**: In text classification, creating an adversarial example means subtly perturbing a few words in a sentence without changing its meaning, causing it to be misclassified by a classifier. A concerning observation is that a significant portion of adversarial examples generated by existing methods change only one word. This single-word perturbation vulnerability represents a significant weakness in classifiers, which malicious users can exploit to efficiently create a multitude of adversarial examples. This paper studies this problem and makes the following key contributions: (1) We introduce a novel metric $\rho$ to quantitatively assess a classifier's robustness against single-word perturbation. (2) We present the SP-Attack, designed to exploit the single-word perturbation vulnerability, achieving a higher attack success rate, better preserving sentence meaning, while reducing computation costs compared to state-of-the-art adversarial methods. (3) We propose SP-Defense, which aims to improve \r{ho} by applying data augmentation in learning. Experimental results on 4 datasets and BERT and distilBERT classifiers show that SP-Defense improves $\rho$ by 14.6% and 13.9% and decreases the attack success rate of SP-Attack by 30.4% and 21.2% on two classifiers respectively, and decreases the attack success rate of existing attack methods that involve multiple-word perturbations.



## **29. AdRo-FL: Informed and Secure Client Selection for Federated Learning in the Presence of Adversarial Aggregator**

cs.CR

17 pages

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2506.17805v2) [paper-pdf](http://arxiv.org/pdf/2506.17805v2)

**Authors**: Md. Kamrul Hossain, Walid Aljoby, Anis Elgabli, Ahmed M. Abdelmoniem, Khaled A. Harras

**Abstract**: Federated Learning (FL) enables collaborative learning without exposing clients' data. While clients only share model updates with the aggregator, studies reveal that aggregators can infer sensitive information from these updates. Secure Aggregation (SA) protects individual updates during transmission; however, recent work demonstrates a critical vulnerability where adversarial aggregators manipulate client selection to bypass SA protections, constituting a Biased Selection Attack (BSA). Although verifiable random selection prevents BSA, it precludes informed client selection essential for FL performance. We propose Adversarial Robust Federated Learning (AdRo-FL), which simultaneously enables: informed client selection based on client utility, and robust defense against BSA maintaining privacy-preserving aggregation. AdRo-FL implements two client selection frameworks tailored for distinct settings. The first framework assumes clients are grouped into clusters based on mutual trust, such as different branches of an organization. The second framework handles distributed clients where no trust relationships exist between them. For the cluster-oriented setting, we propose a novel defense against BSA by (1) enforcing a minimum client selection quota from each cluster, supervised by a cluster-head in every round, and (2) introducing a client utility function to prioritize efficient clients. For the distributed setting, we design a two-phase selection protocol: first, the aggregator selects the top clients based on our utility-driven ranking; then, a verifiable random function (VRF) ensures a BSA-resistant final selection. AdRo-FL also applies quantization to reduce communication overhead and sets strict transmission deadlines to improve energy efficiency. AdRo-FL achieves up to $1.85\times$ faster time-to-accuracy and up to $1.06\times$ higher final accuracy compared to insecure baselines.



## **30. Exploiting Leaderboards for Large-Scale Distribution of Malicious Models**

cs.LG

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08983v1) [paper-pdf](http://arxiv.org/pdf/2507.08983v1)

**Authors**: Anshuman Suri, Harsh Chaudhari, Yuefeng Peng, Ali Naseh, Amir Houmansadr, Alina Oprea

**Abstract**: While poisoning attacks on machine learning models have been extensively studied, the mechanisms by which adversaries can distribute poisoned models at scale remain largely unexplored. In this paper, we shed light on how model leaderboards -- ranked platforms for model discovery and evaluation -- can serve as a powerful channel for adversaries for stealthy large-scale distribution of poisoned models. We present TrojanClimb, a general framework that enables injection of malicious behaviors while maintaining competitive leaderboard performance. We demonstrate its effectiveness across four diverse modalities: text-embedding, text-generation, text-to-speech and text-to-image, showing that adversaries can successfully achieve high leaderboard rankings while embedding arbitrary harmful functionalities, from backdoors to bias injection. Our findings reveal a significant vulnerability in the machine learning ecosystem, highlighting the urgent need to redesign leaderboard evaluation mechanisms to detect and filter malicious (e.g., poisoned) models, while exposing broader security implications for the machine learning community regarding the risks of adopting models from unverified sources.



## **31. VIP: Visual Information Protection through Adversarial Attacks on Vision-Language Models**

eess.IV

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08982v1) [paper-pdf](http://arxiv.org/pdf/2507.08982v1)

**Authors**: Hanene F. Z. Brachemi Meftah, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Déforges

**Abstract**: Recent years have witnessed remarkable progress in developing Vision-Language Models (VLMs) capable of processing both textual and visual inputs. These models have demonstrated impressive performance, leading to their widespread adoption in various applications. However, this widespread raises serious concerns regarding user privacy, particularly when models inadvertently process or expose private visual information. In this work, we frame the preservation of privacy in VLMs as an adversarial attack problem. We propose a novel attack strategy that selectively conceals information within designated Region Of Interests (ROIs) in an image, effectively preventing VLMs from accessing sensitive content while preserving the semantic integrity of the remaining image. Unlike conventional adversarial attacks that often disrupt the entire image, our method maintains high coherence in unmasked areas. Experimental results across three state-of-the-art VLMs namely LLaVA, Instruct-BLIP, and BLIP2-T5 demonstrate up to 98% reduction in detecting targeted ROIs, while maintaining global image semantics intact, as confirmed by high similarity scores between clean and adversarial outputs. We believe that this work contributes to a more privacy conscious use of multimodal models and offers a practical tool for further research, with the source code publicly available at: https://github.com/hbrachemi/Vlm_defense-attack.



## **32. Weak-to-Strong Jailbreaking on Large Language Models**

cs.CL

ICML 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2401.17256v4) [paper-pdf](http://arxiv.org/pdf/2401.17256v4)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong



## **33. Entangled Threats: A Unified Kill Chain Model for Quantum Machine Learning Security**

quant-ph

Accepted for publication at IEEE International Conference on Quantum  Computing and Engineering (QCE) 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08623v1) [paper-pdf](http://arxiv.org/pdf/2507.08623v1)

**Authors**: Pascal Debus, Maximilian Wendlinger, Kilian Tscharke, Daniel Herr, Cedric Brügmann, Daniel Ohl de Mello, Juris Ulmanis, Alexander Erhard, Arthur Schmidt, Fabian Petsch

**Abstract**: Quantum Machine Learning (QML) systems inherit vulnerabilities from classical machine learning while introducing new attack surfaces rooted in the physical and algorithmic layers of quantum computing. Despite a growing body of research on individual attack vectors - ranging from adversarial poisoning and evasion to circuit-level backdoors, side-channel leakage, and model extraction - these threats are often analyzed in isolation, with unrealistic assumptions about attacker capabilities and system environments. This fragmentation hampers the development of effective, holistic defense strategies. In this work, we argue that QML security requires more structured modeling of the attack surface, capturing not only individual techniques but also their relationships, prerequisites, and potential impact across the QML pipeline. We propose adapting kill chain models, widely used in classical IT and cybersecurity, to the quantum machine learning context. Such models allow for structured reasoning about attacker objectives, capabilities, and possible multi-stage attack paths - spanning reconnaissance, initial access, manipulation, persistence, and exfiltration. Based on extensive literature analysis, we present a detailed taxonomy of QML attack vectors mapped to corresponding stages in a quantum-aware kill chain framework that is inspired by the MITRE ATLAS for classical machine learning. We highlight interdependencies between physical-level threats (like side-channel leakage and crosstalk faults), data and algorithm manipulation (such as poisoning or circuit backdoors), and privacy attacks (including model extraction and training data inference). This work provides a foundation for more realistic threat modeling and proactive security-in-depth design in the emerging field of quantum machine learning.



## **34. When and Where do Data Poisons Attack Textual Inversion?**

cs.CR

Accepted to ICCV

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.10578v1) [paper-pdf](http://arxiv.org/pdf/2507.10578v1)

**Authors**: Jeremy Styborski, Mingzhi Lyu, Jiayou Lu, Nupur Kapur, Adams Kong

**Abstract**: Poisoning attacks pose significant challenges to the robustness of diffusion models (DMs). In this paper, we systematically analyze when and where poisoning attacks textual inversion (TI), a widely used personalization technique for DMs. We first introduce Semantic Sensitivity Maps, a novel method for visualizing the influence of poisoning on text embeddings. Second, we identify and experimentally verify that DMs exhibit non-uniform learning behavior across timesteps, focusing on lower-noise samples. Poisoning attacks inherit this bias and inject adversarial signals predominantly at lower timesteps. Lastly, we observe that adversarial signals distract learning away from relevant concept regions within training data, corrupting the TI process. Based on these insights, we propose Safe-Zone Training (SZT), a novel defense mechanism comprised of 3 key components: (1) JPEG compression to weaken high-frequency poison signals, (2) restriction to high timesteps during TI training to avoid adversarial signals at lower timesteps, and (3) loss masking to constrain learning to relevant regions. Extensive experiments across multiple poisoning methods demonstrate that SZT greatly enhances the robustness of TI against all poisoning attacks, improving generative quality beyond prior published defenses. Code: www.github.com/JStyborski/Diff_Lab Data: www.github.com/JStyborski/NC10



## **35. The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover**

cs.CR

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.06850v3) [paper-pdf](http://arxiv.org/pdf/2507.06850v3)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.



## **36. On the $(k,\ell)$-multiset anonymity measure for social graphs**

math.CO

25 pages

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08433v1) [paper-pdf](http://arxiv.org/pdf/2507.08433v1)

**Authors**: Alejandro Estrada-Moreno, Elena Fernández, Dorota Kuziak, Manuel Muñoz-Márquez, Rolando Trujillo-Rasua, Ismael G. Yero

**Abstract**: The publication of social graphs must be preceded by a rigorous analysis of privacy threats against social graph users. When the threat comes from inside the social network itself, the threat is called an active attack, and the de-facto privacy measure used to quantify the resistance to such an attack is the $(k,\ell)$-anonymity. The original formulation of $(k,\ell)$-anonymity represents the adversary's knowledge as a vector of distances to the set of attacker nodes. In this article, we argue that such adversary is too strong when it comes to counteracting active attacks. We, instead, propose a new formulation where the adversary's knowledge is the multiset of distances to the set of attacker nodes. The goal of this article is to study the $(k,\ell)$-multiset anonymity from a graph theoretical point of view, while establishing its relationship to $(k,\ell)$-anonymity in one hand, and considering the $k$-multiset antiresolving sets as its theoretical frame, in a second one. That is, we prove properties of some graph families in relation to whether they contain a set of attacker nodes that breaks the $(k,\ell)$-multiset anonymity. From a practical point of view, we develop a linear programming formulation of the $k$-multiset antiresolving sets that allows us to calculate the resistance of social graphs against active attacks. This is useful for analysts who wish to know the level of privacy offered by a graph.



## **37. Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving**

cs.RO

Accepted in the 36th IEEE Intelligent Vehicles Symposium (IV 2025)

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2505.06740v2) [paper-pdf](http://arxiv.org/pdf/2505.06740v2)

**Authors**: Ahmed Abouelazm, Mianzhi Liu, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner

**Abstract**: Accurate prediction of surrounding road users' trajectories is essential for safe and efficient autonomous driving. While deep learning models have improved performance, challenges remain in preventing off-road predictions and ensuring kinematic feasibility. Existing methods incorporate road-awareness modules and enforce kinematic constraints but lack plausibility guarantees and often introduce trade-offs in complexity and flexibility. This paper proposes a novel framework that formulates trajectory prediction as a constrained regression guided by permissible driving directions and their boundaries. Using the agent's current state and an HD map, our approach defines the valid boundaries and ensures on-road predictions by training the network to learn superimposed paths between left and right boundary polylines. To guarantee feasibility, the model predicts acceleration profiles that determine the vehicle's travel distance along these paths while adhering to kinematic constraints. We evaluate our approach on the Argoverse-2 dataset against the HPTR baseline. Our approach shows a slight decrease in benchmark metrics compared to HPTR but notably improves final displacement error and eliminates infeasible trajectories. Moreover, the proposed approach has superior generalization to less prevalent maneuvers and unseen out-of-distribution scenarios, reducing the off-road rate under adversarial attacks from 66% to just 1%. These results highlight the effectiveness of our approach in generating feasible and robust predictions.



## **38. Minerva: A File-Based Ransomware Detector**

cs.CR

Accepted for publication at The 20th ACM ASIA Conference on Computer  and Communications Security (ACM ASIACCS 2025), Meli\'a Hanoi

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2301.11050v4) [paper-pdf](http://arxiv.org/pdf/2301.11050v4)

**Authors**: Dorjan Hitaj, Giulio Pagnotta, Fabio De Gaspari, Lorenzo De Carli, Luigi V. Mancini

**Abstract**: Ransomware attacks have caused billions of dollars in damages in recent years, and are expected to cause billions more in the future. Consequently, significant effort has been devoted to ransomware detection and mitigation. Behavioral-based ransomware detection approaches have garnered considerable attention recently. These behavioral detectors typically rely on process-based behavioral profiles to identify malicious behaviors. However, with an increasing body of literature highlighting the vulnerability of such approaches to evasion attacks, a comprehensive solution to the ransomware problem remains elusive. This paper presents Minerva, a novel, robust approach to ransomware detection. Minerva is engineered to be robust by design against evasion attacks, with architectural and feature selection choices informed by their resilience to adversarial manipulation. We conduct a comprehensive analysis of Minerva across a diverse spectrum of ransomware types, encompassing unseen ransomware as well as variants designed specifically to evade Minerva. Our evaluation showcases the ability of Minerva to accurately identify ransomware, generalize to unseen threats, and withstand evasion attacks. Furthermore, over 99% of detected ransomware are identified within 0.52sec of activity, enabling the adoption of data loss prevention techniques with near-zero overhead.



## **39. Towards Imperceptible JPEG Image Hiding: Multi-range Representations-driven Adversarial Stego Generation**

cs.CV

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08343v1) [paper-pdf](http://arxiv.org/pdf/2507.08343v1)

**Authors**: Junxue Yang, Xin Liao, Weixuan Tang, Jianhua Yang, Zheng Qin

**Abstract**: Deep hiding has been exploring the hiding capability of deep learning-based models, aiming to conceal image-level messages into cover images and reveal them from generated stego images. Existing schemes are easily detected by steganalyzers due to their large payloads and their limitation to feature extraction based solely on either pure convolution or pure transformer operators within a single range, as well as pixel-level loss constraints. To address the issue, in this paper, we introduce generation-based adversarial attacks into color JPEG image deep hiding and propose a multi-range representations-driven adversarial stego generation framework called MRAG from a steganalysis perspective. Specifically, we integrate the local-range neighbor reception characteristic of the convolution and the global-range dependency modeling of the transformer to construct MRAG. Meanwhile, we use the transformed images obtained through coarse-grained and fine-grained frequency decomposition as inputs, introducing multi-grained information. Furthermore, a features angle-norm disentanglement loss is designed to constrain the generated stegos closer to covers in the angle and norm space of the steganalyzer's classified features. Consequently, small yet effective adversarial perturbations can be injected into the process of generating stegos, ensuring that stegos maintain favorable secret restorability and imperceptibility. Extensive experiments demonstrate that MRAG can achieve state-of-the-art performance.



## **40. Learning Robust Motion Skills via Critical Adversarial Attacks for Humanoid Robots**

cs.RO

10 pages, 9 figures

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08303v1) [paper-pdf](http://arxiv.org/pdf/2507.08303v1)

**Authors**: Yang Zhang, Zhanxiang Cao, Buqing Nie, Haoyang Li, Yue Gao

**Abstract**: Humanoid robots show significant potential in daily tasks. However, reinforcement learning-based motion policies often suffer from robustness degradation due to the sim-to-real dynamics gap, thereby affecting the agility of real robots. In this work, we propose a novel robust adversarial training paradigm designed to enhance the robustness of humanoid motion policies in real worlds. The paradigm introduces a learnable adversarial attack network that precisely identifies vulnerabilities in motion policies and applies targeted perturbations, forcing the motion policy to enhance its robustness against perturbations through dynamic adversarial training. We conduct experiments on the Unitree G1 humanoid robot for both perceptive locomotion and whole-body control tasks. The results demonstrate that our proposed method significantly enhances the robot's motion robustness in real world environments, enabling successful traversal of challenging terrains and highly agile whole-body trajectory tracking.



## **41. Lightweight Safety Guardrails via Synthetic Data and RL-guided Adversarial Training**

cs.LG

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08284v1) [paper-pdf](http://arxiv.org/pdf/2507.08284v1)

**Authors**: Aleksei Ilin, Gor Matevosyan, Xueying Ma, Vladimir Eremin, Suhaa Dada, Muqun Li, Riyaaz Shaik, Haluk Noyan Tokgozoglu

**Abstract**: We introduce a lightweight yet highly effective safety guardrail framework for language models, demonstrating that small-scale language models can achieve, and even surpass, the performance of larger counterparts in content moderation tasks. This is accomplished through high-fidelity synthetic data generation and adversarial training. The synthetic data generation process begins with human-curated seed data, which undergoes query augmentation and paraphrasing to create diverse and contextually rich examples. This augmented data is then subjected to multiple rounds of curation, ensuring high fidelity and relevance. Inspired by recent advances in the Generative Adversarial Network (GAN) architecture, our adversarial training employs reinforcement learning to guide a generator that produces challenging synthetic examples. These examples are used to fine-tune the safety classifier, enhancing its ability to detect and mitigate harmful content. Additionally, we incorporate strategies from recent research on efficient LLM training, leveraging the capabilities of smaller models to improve the performance of larger generative models. With iterative adversarial training and the generation of diverse, high-quality synthetic data, our framework enables small language models (SLMs) to serve as robust safety guardrails. This approach not only reduces computational overhead but also enhances resilience against adversarial attacks, offering a scalable and efficient solution for content moderation in AI systems.



## **42. Admissibility of Stein Shrinkage for Batch Normalization in the Presence of Adversarial Attacks**

stat.ML

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08261v1) [paper-pdf](http://arxiv.org/pdf/2507.08261v1)

**Authors**: Sofia Ivolgina, P. Thomas Fletcher, Baba C. Vemuri

**Abstract**: Batch normalization (BN) is a ubiquitous operation in deep neural networks used primarily to achieve stability and regularization during network training. BN involves feature map centering and scaling using sample means and variances, respectively. Since these statistics are being estimated across the feature maps within a batch, this problem is ideally suited for the application of Stein's shrinkage estimation, which leads to a better, in the mean-squared-error sense, estimate of the mean and variance of the batch. In this paper, we prove that the Stein shrinkage estimator for the mean and variance dominates over the sample mean and variance estimators in the presence of adversarial attacks when modeling these attacks using sub-Gaussian distributions. This facilitates and justifies the application of Stein shrinkage to estimate the mean and variance parameters in BN and use it in image classification (segmentation) tasks with and without adversarial attacks. We present SOTA performance results using this Stein corrected batch norm in a standard ResNet architecture applied to the task of image classification using CIFAR-10 data, 3D CNN on PPMI (neuroimaging) data and image segmentation using HRNet on Cityscape data with and without adversarial attacks.



## **43. Pushing the Limits of Safety: A Technical Report on the ATLAS Challenge 2025**

cs.CR

AdvML@CVPR Challenge Report

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2506.12430v2) [paper-pdf](http://arxiv.org/pdf/2506.12430v2)

**Authors**: Zonghao Ying, Siyang Wu, Run Hao, Peng Ying, Shixuan Sun, Pengyu Chen, Junze Chen, Hao Du, Kaiwen Shen, Shangkun Wu, Jiwei Wei, Shiyuan He, Yang Yang, Xiaohai Xu, Ke Ma, Qianqian Xu, Qingming Huang, Shi Lin, Xun Wang, Changting Lin, Meng Han, Yilei Jiang, Siqi Lai, Yaozhi Zheng, Yifei Song, Xiangyu Yue, Zonglei Jing, Tianyuan Zhang, Zhilei Zhu, Aishan Liu, Jiakai Wang, Siyuan Liang, Xianglong Kong, Hainan Li, Junjie Mu, Haotong Qin, Yue Yu, Lei Chen, Felix Juefei-Xu, Qing Guo, Xinyun Chen, Yew Soon Ong, Xianglong Liu, Dawn Song, Alan Yuille, Philip Torr, Dacheng Tao

**Abstract**: Multimodal Large Language Models (MLLMs) have enabled transformative advancements across diverse applications but remain susceptible to safety threats, especially jailbreak attacks that induce harmful outputs. To systematically evaluate and improve their safety, we organized the Adversarial Testing & Large-model Alignment Safety Grand Challenge (ATLAS) 2025}. This technical report presents findings from the competition, which involved 86 teams testing MLLM vulnerabilities via adversarial image-text attacks in two phases: white-box and black-box evaluations. The competition results highlight ongoing challenges in securing MLLMs and provide valuable guidance for developing stronger defense mechanisms. The challenge establishes new benchmarks for MLLM safety evaluation and lays groundwork for advancing safer multimodal AI systems. The code and data for this challenge are openly available at https://github.com/NY1024/ATLAS_Challenge_2025.



## **44. A Dynamic Stackelberg Game Framework for Agentic AI Defense Against LLM Jailbreaking**

cs.AI

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.08207v1) [paper-pdf](http://arxiv.org/pdf/2507.08207v1)

**Authors**: Zhengye Han, Quanyan Zhu

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, the challenge of jailbreaking, where adversaries manipulate the models to bypass safety mechanisms, has become a significant concern. This paper presents a dynamic Stackelberg game framework to model the interactions between attackers and defenders in the context of LLM jailbreaking. The framework treats the prompt-response dynamics as a sequential extensive-form game, where the defender, as the leader, commits to a strategy while anticipating the attacker's optimal responses. We propose a novel agentic AI solution, the "Purple Agent," which integrates adversarial exploration and defensive strategies using Rapidly-exploring Random Trees (RRT). The Purple Agent actively simulates potential attack trajectories and intervenes proactively to prevent harmful outputs. This approach offers a principled method for analyzing adversarial dynamics and provides a foundation for mitigating the risk of jailbreaking.



## **45. Beyond the Worst Case: Extending Differential Privacy Guarantees to Realistic Adversaries**

cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.08158v1) [paper-pdf](http://arxiv.org/pdf/2507.08158v1)

**Authors**: Marika Swanberg, Meenatchi Sundaram Muthu Selva Annamalai, Jamie Hayes, Borja Balle, Adam Smith

**Abstract**: Differential Privacy (DP) is a family of definitions that bound the worst-case privacy leakage of a mechanism. One important feature of the worst-case DP guarantee is it naturally implies protections against adversaries with less prior information, more sophisticated attack goals, and complex measures of a successful attack. However, the analytical tradeoffs between the adversarial model and the privacy protections conferred by DP are not well understood thus far. To that end, this work sheds light on what the worst-case guarantee of DP implies about the success of attackers that are more representative of real-world privacy risks.   In this paper, we present a single flexible framework that generalizes and extends the patchwork of bounds on DP mechanisms found in prior work. Our framework allows us to compute high-probability guarantees for DP mechanisms on a large family of natural attack settings that previous bounds do not capture. One class of such settings is the approximate reconstruction of multiple individuals' data, such as inferring nearly entire columns of a tabular data set from noisy marginals and extracting sensitive information from DP-trained language models.   We conduct two empirical case studies to illustrate the versatility of our bounds and compare them to the success of state-of-the-art attacks. Specifically, we study attacks that extract non-uniform PII from a DP-trained language model, as well as multi-column reconstruction attacks where the adversary has access to some columns in the clear and attempts to reconstruct the remaining columns for each person's record. We find that the absolute privacy risk of attacking non-uniform data is highly dependent on the adversary's prior probability of success. Our high probability bounds give us a nuanced understanding of the privacy leakage of DP mechanisms in a variety of previously understudied attack settings.



## **46. Hedge Funds on a Swamp: Analyzing Patterns, Vulnerabilities, and Defense Measures in Blockchain Bridges [Experiment, Analysis & Benchmark]**

cs.ET

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.06156v2) [paper-pdf](http://arxiv.org/pdf/2507.06156v2)

**Authors**: Poupak Azad, Jiahua Xu, Yebo Feng, Preston Strowbridge, Cuneyt Akcora

**Abstract**: Blockchain bridges have become essential infrastructure for enabling interoperability across different blockchain networks, with more than $24B monthly bridge transaction volume. However, their growing adoption has been accompanied by a disproportionate rise in security breaches, making them the single largest source of financial loss in Web3. For cross-chain ecosystems to be robust and sustainable, it is essential to understand and address these vulnerabilities. In this study, we present a comprehensive systematization of blockchain bridge design and security. We define three bridge security priors, formalize the architectural structure of 13 prominent bridges, and identify 23 attack vectors grounded in real-world blockchain exploits. Using this foundation, we evaluate 43 representative attack scenarios and introduce a layered threat model that captures security failures across source chain, off-chain, and destination chain components.   Our analysis at the static code and transaction network levels reveals recurring design flaws, particularly in access control, validator trust assumptions, and verification logic, and identifies key patterns in adversarial behavior based on transaction-level traces. To support future development, we propose a decision framework for bridge architecture design, along with defense mechanisms such as layered validation and circuit breakers. This work provides a data-driven foundation for evaluating bridge security and lays the groundwork for standardizing resilient cross-chain infrastructure.



## **47. KeyDroid: A Large-Scale Analysis of Secure Key Storage in Android Apps**

cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07927v1) [paper-pdf](http://arxiv.org/pdf/2507.07927v1)

**Authors**: Jenny Blessing, Ross J. Anderson, Alastair R. Beresford

**Abstract**: Most contemporary mobile devices offer hardware-backed storage for cryptographic keys, user data, and other sensitive credentials. Such hardware protects credentials from extraction by an adversary who has compromised the main operating system, such as a malicious third-party app. Since 2011, Android app developers can access trusted hardware via the Android Keystore API. In this work, we conduct the first comprehensive survey of hardware-backed key storage in Android devices. We analyze 490 119 Android apps, collecting data on how trusted hardware is used by app developers (if used at all) and cross-referencing our findings with sensitive user data collected by each app, as self-reported by developers via the Play Store's data safety labels.   We find that despite industry-wide initiatives to encourage adoption, 56.3% of apps self-reporting as processing sensitive user data do not use Android's trusted hardware capabilities at all, while just 5.03% of apps collecting some form of sensitive data use the strongest form of trusted hardware, a secure element distinct from the main processor. To better understand the potential downsides of using secure hardware, we conduct the first empirical analysis of trusted hardware performance in mobile devices, measuring the runtime of common cryptographic operations across both software- and hardware-backed keystores. We find that while hardware-backed key storage using a coprocessor is viable for most common cryptographic operations, secure elements capable of preventing more advanced attacks make performance infeasible for symmetric encryption with non-negligible payloads and any kind of asymmetric encryption.



## **48. Bayes-Nash Generative Privacy Against Membership Inference Attacks**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2406.01811

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2410.07414v5) [paper-pdf](http://arxiv.org/pdf/2410.07414v5)

**Authors**: Tao Zhang, Rajagopal Venkatesaramani, Rajat K. De, Bradley A. Malin, Yevgeniy Vorobeychik

**Abstract**: Membership inference attacks (MIAs) pose significant privacy risks by determining whether individual data is in a dataset. While differential privacy (DP) mitigates these risks, it has limitations including limited resolution in expressing privacy-utility tradeoffs and intractable sensitivity calculations for tight guarantees. We propose a game-theoretic framework modeling privacy protection as a Bayesian game between defender and attacker, where privacy loss corresponds to the attacker's membership inference ability. To address strategic complexity, we represent the defender's mixed strategy as a neural network generator mapping private datasets to public representations (e.g., noisy statistics) and the attacker's strategy as a discriminator making membership claims. This \textit{general-sum Generative Adversarial Network} trains iteratively through alternating updates, yielding \textit{Bayes-Nash Generative Privacy (BNGP)} strategies. BNGP avoids worst-case privacy proofs such as sensitivity calculations, supports correlated mechanism compositions, handles heterogeneous attacker preferences. Empirical studies on sensitive dataset summary statistics show our approach significantly outperforms state-of-the-art methods by generating stronger attacks and achieving better privacy-utility tradeoffs.



## **49. Identifying the Smallest Adversarial Load Perturbations that Render DC-OPF Infeasible**

eess.SY

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07850v1) [paper-pdf](http://arxiv.org/pdf/2507.07850v1)

**Authors**: Samuel Chevalier, William A. Wheeler

**Abstract**: What is the globally smallest load perturbation that renders DC-OPF infeasible? Reliably identifying such "adversarial attack" perturbations has useful applications in a variety of emerging grid-related contexts, including machine learning performance verification, cybersecurity, and operational robustness of power systems dominated by stochastic renewable energy resources. In this paper, we formulate the inherently nonconvex adversarial attack problem by applying a parameterized version of Farkas' lemma to a perturbed set of DC-OPF equations. Since the resulting formulation is very hard to globally optimize, we also propose a parameterized generation control policy which, when applied to the primal DC-OPF problem, provides solvability guarantees. Together, these nonconvex problems provide guaranteed upper and lower bounds on adversarial attack size; by combining them into a single optimization problem, we can efficiently "squeeze" these bounds towards a common global solution. We apply these methods on a range of small- to medium-sized test cases from PGLib, benchmarking our results against the best adversarial attack lower bounds provided by Gurobi 12.0's spatial Branch and Bound solver.



## **50. "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models**

cs.LG

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2502.00718v2) [paper-pdf](http://arxiv.org/pdf/2502.00718v2)

**Authors**: Isha Gupta, David Khachaturov, Robert Mullins

**Abstract**: The rise of multimodal large language models has introduced innovative human-machine interaction paradigms but also significant challenges in machine learning safety. Audio-Language Models (ALMs) are especially relevant due to the intuitive nature of spoken communication, yet little is known about their failure modes. This paper explores audio jailbreaks targeting ALMs, focusing on their ability to bypass alignment mechanisms. We construct adversarial perturbations that generalize across prompts, tasks, and even base audio samples, demonstrating the first universal jailbreaks in the audio modality, and show that these remain effective in simulated real-world conditions. Beyond demonstrating attack feasibility, we analyze how ALMs interpret these audio adversarial examples and reveal them to encode imperceptible first-person toxic speech - suggesting that the most effective perturbations for eliciting toxic outputs specifically embed linguistic features within the audio signal. These results have important implications for understanding the interactions between different modalities in multimodal models, and offer actionable insights for enhancing defenses against adversarial audio attacks.



