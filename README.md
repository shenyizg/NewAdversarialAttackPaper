# Latest Adversarial Attack Papers
**update at 2025-06-03 09:50:50**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models**

cs.CR

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2506.00821v1) [paper-pdf](http://arxiv.org/pdf/2506.00821v1)

**Authors**: Huixin Zhan, Jason H. Moore

**Abstract**: Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated significant success in variant effect prediction. However, their adversarial robustness remains largely unexplored. To address this gap, we propose SafeGenes: a framework for Secure analysis of genomic foundation models, leveraging adversarial attacks to evaluate robustness against both engineered near-identical adversarial Genes and embedding-space manipulations. In this study, we assess the adversarial vulnerabilities of GFMs using two approaches: the Fast Gradient Sign Method (FGSM) and a soft prompt attack. FGSM introduces minimal perturbations to input sequences, while the soft prompt attack optimizes continuous embeddings to manipulate model predictions without modifying the input tokens. By combining these techniques, SafeGenes provides a comprehensive assessment of GFM susceptibility to adversarial manipulation. Targeted soft prompt attacks led to substantial performance degradation, even in large models such as ESM1b and ESM1v. These findings expose critical vulnerabilities in current foundation models, opening new research directions toward improving their security and robustness in high-stakes genomic applications such as variant effect prediction.



## **2. Unlearning Inversion Attacks for Graph Neural Networks**

cs.LG

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2506.00808v1) [paper-pdf](http://arxiv.org/pdf/2506.00808v1)

**Authors**: Jiahao Zhang, Yilong Wang, Zhiwei Zhang, Xiaorui Liu, Suhang Wang

**Abstract**: Graph unlearning methods aim to efficiently remove the impact of sensitive data from trained GNNs without full retraining, assuming that deleted information cannot be recovered. In this work, we challenge this assumption by introducing the graph unlearning inversion attack: given only black-box access to an unlearned GNN and partial graph knowledge, can an adversary reconstruct the removed edges? We identify two key challenges: varying probability-similarity thresholds for unlearned versus retained edges, and the difficulty of locating unlearned edge endpoints, and address them with TrendAttack. First, we derive and exploit the confidence pitfall, a theoretical and empirical pattern showing that nodes adjacent to unlearned edges exhibit a large drop in model confidence. Second, we design an adaptive prediction mechanism that applies different similarity thresholds to unlearned and other membership edges. Our framework flexibly integrates existing membership inference techniques and extends them with trend features. Experiments on four real-world datasets demonstrate that TrendAttack significantly outperforms state-of-the-art GNN membership inference baselines, exposing a critical privacy vulnerability in current graph unlearning methods.



## **3. RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments**

cs.CL

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2505.21936v2) [paper-pdf](http://arxiv.org/pdf/2505.21936v2)

**Authors**: Zeyi Liao, Jaylen Jones, Linxi Jiang, Eric Fosler-Lussier, Yu Su, Zhiqiang Lin, Huan Sun

**Abstract**: Computer-use agents (CUAs) promise to automate complex tasks across operating systems (OS) and the web, but remain vulnerable to indirect prompt injection. Current evaluations of this threat either lack support realistic but controlled environments or ignore hybrid web-OS attack scenarios involving both interfaces. To address this, we propose RedTeamCUA, an adversarial testing framework featuring a novel hybrid sandbox that integrates a VM-based OS environment with Docker-based web platforms. Our sandbox supports key features tailored for red teaming, such as flexible adversarial scenario configuration, and a setting that decouples adversarial evaluation from navigational limitations of CUAs by initializing tests directly at the point of an adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive benchmark with 864 examples that investigate realistic, hybrid web-OS attack scenarios and fundamental security vulnerabilities. Benchmarking current frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated, still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute adversarial tasks with an Attempt Rate as high as 92.5%, although failing to complete them due to capability limitations. Nevertheless, we observe concerning ASRs of up to 50% in realistic end-to-end settings, with the recently released frontier Claude 4 Opus | CUA showing an alarming ASR of 48%, demonstrating that indirect prompt injection presents tangible risks for even advanced CUAs despite their capabilities and safeguards. Overall, RedTeamCUA provides an essential framework for advancing realistic, controlled, and systematic analysis of CUA vulnerabilities, highlighting the urgent need for robust defenses to indirect prompt injection prior to real-world deployment.



## **4. Security Concerns for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2505.18889v2) [paper-pdf](http://arxiv.org/pdf/2505.18889v2)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as GPT-4 and its recent iterations, Google's Gemini, Anthropic's Claude 3 models, and xAI's Grok have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. In this survey, we provide a comprehensive overview of the emerging security concerns around LLMs, categorizing threats into prompt injection and jailbreaking, adversarial attacks such as input perturbations and data poisoning, misuse by malicious actors for purposes such as generating disinformation, phishing emails, and malware, and worrisome risks inherent in autonomous LLM agents. A significant focus has been recently placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives, a behavior known as scheming, which may even persist through safety training. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.



## **5. AdvAgent: Controllable Blackbox Red-teaming on Web Agents**

cs.CR

ICML 2025

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2410.17401v4) [paper-pdf](http://arxiv.org/pdf/2410.17401v4)

**Authors**: Chejian Xu, Mintong Kang, Jiawei Zhang, Zeyi Liao, Lingbo Mo, Mengqi Yuan, Huan Sun, Bo Li

**Abstract**: Foundation model-based agents are increasingly used to automate complex tasks, enhancing efficiency and productivity. However, their access to sensitive resources and autonomous decision-making also introduce significant security risks, where successful attacks could lead to severe consequences. To systematically uncover these vulnerabilities, we propose AdvAgent, a black-box red-teaming framework for attacking web agents. Unlike existing approaches, AdvAgent employs a reinforcement learning-based pipeline to train an adversarial prompter model that optimizes adversarial prompts using feedback from the black-box agent. With careful attack design, these prompts effectively exploit agent weaknesses while maintaining stealthiness and controllability. Extensive evaluations demonstrate that AdvAgent achieves high success rates against state-of-the-art GPT-4-based web agents across diverse web tasks. Furthermore, we find that existing prompt-based defenses provide only limited protection, leaving agents vulnerable to our framework. These findings highlight critical vulnerabilities in current web agents and emphasize the urgent need for stronger defense mechanisms. We release code at https://ai-secure.github.io/AdvAgent/.



## **6. Poster: Adapting Pretrained Vision Transformers with LoRA Against Attack Vectors**

cs.CV

Presented at IEEE MOST 2025

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00661v1) [paper-pdf](http://arxiv.org/pdf/2506.00661v1)

**Authors**: Richard E. Neddo, Sean Willis, Zander Blasingame, Chen Liu

**Abstract**: Image classifiers, such as those used for autonomous vehicle navigation, are largely known to be susceptible to adversarial attacks that target the input image set. There is extensive discussion on adversarial attacks including perturbations that alter the input images to cause malicious misclassifications without perceivable modification. This work proposes a countermeasure for such attacks by adjusting the weights and classes of pretrained vision transformers with a low-rank adaptation to become more robust against adversarial attacks and allow for scalable fine-tuning without retraining.



## **7. Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities**

cs.CR

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00548v1) [paper-pdf](http://arxiv.org/pdf/2506.00548v1)

**Authors**: Jiahui Geng, Thy Thy Tran, Preslav Nakov, Iryna Gurevych

**Abstract**: Existing attacks against multimodal language models (MLLMs) primarily communicate instructions through text accompanied by adversarial images. In contrast, we exploit the capabilities of MLLMs to interpret non-textual instructions, specifically, adversarial images or audio generated by our novel method, Con Instruction. We optimize these adversarial examples to align closely with target instructions in the embedding space, revealing the detrimental implications of MLLMs' sophisticated understanding. Unlike prior work, our method does not require training data or preprocessing of textual instructions. While these non-textual adversarial examples can effectively bypass MLLM safety mechanisms, their combination with various text inputs substantially amplifies attack success. We further introduce a new Attack Response Categorization (ARC) framework, which evaluates both the quality of the model's response and its relevance to the malicious instructions. Experimental results demonstrate that Con Instruction effectively bypasses safety mechanisms in multiple vision- and audio-language models, including LLaVA-v1.5, InternVL, Qwen-VL, and Qwen-Audio, evaluated on two standard benchmarks: AdvBench and SafeBench. Specifically, our method achieves the highest attack success rates, reaching 81.3% and 86.6% on LLaVA-v1.5 (13B). On the defense side, we explore various countermeasures against our attacks and uncover a substantial performance gap among existing techniques. Our implementation is made publicly available.



## **8. The TIP of the Iceberg: Revealing a Hidden Class of Task-in-Prompt Adversarial Attacks on LLMs**

cs.CR

Accepted to the Main Track of ACL 2025

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2501.18626v4) [paper-pdf](http://arxiv.org/pdf/2501.18626v4)

**Authors**: Sergey Berezin, Reza Farahbakhsh, Noel Crespi

**Abstract**: We present a novel class of jailbreak adversarial attacks on LLMs, termed Task-in-Prompt (TIP) attacks. Our approach embeds sequence-to-sequence tasks (e.g., cipher decoding, riddles, code execution) into the model's prompt to indirectly generate prohibited inputs. To systematically assess the effectiveness of these attacks, we introduce the PHRYGE benchmark. We demonstrate that our techniques successfully circumvent safeguards in six state-of-the-art language models, including GPT-4o and LLaMA 3.2. Our findings highlight critical weaknesses in current LLM safety alignments and underscore the urgent need for more sophisticated defence strategies.   Warning: this paper contains examples of unethical inquiries used solely for research purposes.



## **9. Practical Adversarial Attacks on Stochastic Bandits via Fake Data Injection**

cs.LG

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2505.21938v2) [paper-pdf](http://arxiv.org/pdf/2505.21938v2)

**Authors**: Qirun Zeng, Eric He, Richard Hoffmann, Xuchuang Wang, Jinhang Zuo

**Abstract**: Adversarial attacks on stochastic bandits have traditionally relied on some unrealistic assumptions, such as per-round reward manipulation and unbounded perturbations, limiting their relevance to real-world systems. We propose a more practical threat model, Fake Data Injection, which reflects realistic adversarial constraints: the attacker can inject only a limited number of bounded fake feedback samples into the learner's history, simulating legitimate interactions. We design efficient attack strategies under this model, explicitly addressing both magnitude constraints (on reward values) and temporal constraints (on when and how often data can be injected). Our theoretical analysis shows that these attacks can mislead both Upper Confidence Bound (UCB) and Thompson Sampling algorithms into selecting a target arm in nearly all rounds while incurring only sublinear attack cost. Experiments on synthetic and real-world datasets validate the effectiveness of our strategies, revealing significant vulnerabilities in widely used stochastic bandit algorithms under practical adversarial scenarios.



## **10. PADetBench: Towards Benchmarking Physical Attacks against Object Detection**

cs.CV

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2408.09181v3) [paper-pdf](http://arxiv.org/pdf/2408.09181v3)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Lap-Pui Chau, Shaohui Mei

**Abstract**: Physical attacks against object detection have gained increasing attention due to their significant practical implications. However, conducting physical experiments is extremely time-consuming and labor-intensive. Moreover, physical dynamics and cross-domain transformation are challenging to strictly regulate in the real world, leading to unaligned evaluation and comparison, severely hindering the development of physically robust models. To accommodate these challenges, we explore utilizing realistic simulation to thoroughly and rigorously benchmark physical attacks with fairness under controlled physical dynamics and cross-domain transformation. This resolves the problem of capturing identical adversarial images that cannot be achieved in the real world. Our benchmark includes 20 physical attack methods, 48 object detectors, comprehensive physical dynamics, and evaluation metrics. We also provide end-to-end pipelines for dataset generation, detection, evaluation, and further analysis. In addition, we perform 8064 groups of evaluation based on our benchmark, which includes both overall evaluation and further detailed ablation studies for controlled physical dynamics. Through these experiments, we provide in-depth analyses of physical attack performance and physical adversarial robustness, draw valuable observations, and discuss potential directions for future research.   Codebase: https://github.com/JiaweiLian/Benchmarking_Physical_Attack



## **11. Adversarial Machine Learning for Robust Password Strength Estimation**

cs.CR

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00373v1) [paper-pdf](http://arxiv.org/pdf/2506.00373v1)

**Authors**: Pappu Jha, Hanzla Hamid, Oluseyi Olukola, Ashim Dahal, Nick Rahimi

**Abstract**: Passwords remain one of the most common methods for securing sensitive data in the digital age. However, weak password choices continue to pose significant risks to data security and privacy. This study aims to solve the problem by focusing on developing robust password strength estimation models using adversarial machine learning, a technique that trains models on intentionally crafted deceptive passwords to expose and address vulnerabilities posed by such passwords. We apply five classification algorithms and use a dataset with more than 670,000 samples of adversarial passwords to train the models. Results demonstrate that adversarial training improves password strength classification accuracy by up to 20% compared to traditional machine learning models. It highlights the importance of integrating adversarial machine learning into security systems to enhance their robustness against modern adaptive threats.   Keywords: adversarial attack, password strength, classification, machine learning



## **12. Towards Effective and Efficient Adversarial Defense with Diffusion Models for Robust Visual Tracking**

cs.CV

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00325v1) [paper-pdf](http://arxiv.org/pdf/2506.00325v1)

**Authors**: Long Xu, Peng Gao, Wen-Jia Tang, Fei Wang, Ru-Yue Yuan

**Abstract**: Although deep learning-based visual tracking methods have made significant progress, they exhibit vulnerabilities when facing carefully designed adversarial attacks, which can lead to a sharp decline in tracking performance. To address this issue, this paper proposes for the first time a novel adversarial defense method based on denoise diffusion probabilistic models, termed DiffDf, aimed at effectively improving the robustness of existing visual tracking methods against adversarial attacks. DiffDf establishes a multi-scale defense mechanism by combining pixel-level reconstruction loss, semantic consistency loss, and structural similarity loss, effectively suppressing adversarial perturbations through a gradual denoising process. Extensive experimental results on several mainstream datasets show that the DiffDf method demonstrates excellent generalization performance for trackers with different architectures, significantly improving various evaluation metrics while achieving real-time inference speeds of over 30 FPS, showcasing outstanding defense performance and efficiency. Codes are available at https://github.com/pgao-lab/DiffDf.



## **13. RenderBender: A Survey on Adversarial Attacks Using Differentiable Rendering**

cs.LG

9 pages, 1 figure, 2 tables, IJCAI '25 Survey Track

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2411.09749v2) [paper-pdf](http://arxiv.org/pdf/2411.09749v2)

**Authors**: Matthew Hull, Haoran Wang, Matthew Lau, Alec Helbling, Mansi Phute, Chao Zhang, Zsolt Kira, Willian Lunardi, Martin Andreoni, Wenke Lee, Polo Chau

**Abstract**: Differentiable rendering techniques like Gaussian Splatting and Neural Radiance Fields have become powerful tools for generating high-fidelity models of 3D objects and scenes. Their ability to produce both physically plausible and differentiable models of scenes are key ingredient needed to produce physically plausible adversarial attacks on DNNs. However, the adversarial machine learning community has yet to fully explore these capabilities, partly due to differing attack goals (e.g., misclassification, misdetection) and a wide range of possible scene manipulations used to achieve them (e.g., alter texture, mesh). This survey contributes the first framework that unifies diverse goals and tasks, facilitating easy comparison of existing work, identifying research gaps, and highlighting future directions - ranging from expanding attack goals and tasks to account for new modalities, state-of-the-art models, tools, and pipelines, to underscoring the importance of studying real-world threats in complex scenes.



## **14. Adversarial Threat Vectors and Risk Mitigation for Retrieval-Augmented Generation Systems**

cs.CR

SPIE DCS: Proceedings Volume Assurance and Security for AI-enabled  Systems 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2506.00281v1) [paper-pdf](http://arxiv.org/pdf/2506.00281v1)

**Authors**: Chris M. Ward, Josh Harguess

**Abstract**: Retrieval-Augmented Generation (RAG) systems, which integrate Large Language Models (LLMs) with external knowledge sources, are vulnerable to a range of adversarial attack vectors. This paper examines the importance of RAG systems through recent industry adoption trends and identifies the prominent attack vectors for RAG: prompt injection, data poisoning, and adversarial query manipulation. We analyze these threats under risk management lens, and propose robust prioritized control list that includes risk-mitigating actions like input validation, adversarial training, and real-time monitoring.



## **15. 3D Gaussian Splat Vulnerabilities**

cs.CR

4 pages, 4 figures, CVPR '25 Workshop on Neural Fields Beyond  Conventional Cameras

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2506.00280v1) [paper-pdf](http://arxiv.org/pdf/2506.00280v1)

**Authors**: Matthew Hull, Haoyang Yang, Pratham Mehta, Mansi Phute, Aeree Cho, Haoran Wang, Matthew Lau, Wenke Lee, Willian T. Lunardi, Martin Andreoni, Polo Chau

**Abstract**: With 3D Gaussian Splatting (3DGS) being increasingly used in safety-critical applications, how can an adversary manipulate the scene to cause harm? We introduce CLOAK, the first attack that leverages view-dependent Gaussian appearances - colors and textures that change with viewing angle - to embed adversarial content visible only from specific viewpoints. We further demonstrate DAGGER, a targeted adversarial attack directly perturbing 3D Gaussians without access to underlying training data, deceiving multi-stage object detectors e.g., Faster R-CNN, through established methods such as projected gradient descent. These attacks highlight underexplored vulnerabilities in 3DGS, introducing a new potential threat to robotic learning for autonomous navigation and other safety-critical 3DGS applications.



## **16. GSBA$^K$: $top$-$K$ Geometric Score-based Black-box Attack**

cs.CV

License changed to CC BY 4.0 to align with ICLR 2025. No changes to  content. Published at: https://openreview.net/forum?id=htX7AoHyln

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2503.12827v3) [paper-pdf](http://arxiv.org/pdf/2503.12827v3)

**Authors**: Md Farhamdur Reza, Richeng Jin, Tianfu Wu, Huaiyu Dai

**Abstract**: Existing score-based adversarial attacks mainly focus on crafting $top$-1 adversarial examples against classifiers with single-label classification. Their attack success rate and query efficiency are often less than satisfactory, particularly under small perturbation requirements; moreover, the vulnerability of classifiers with multi-label learning is yet to be studied. In this paper, we propose a comprehensive surrogate free score-based attack, named \b geometric \b score-based \b black-box \b attack (GSBA$^K$), to craft adversarial examples in an aggressive $top$-$K$ setting for both untargeted and targeted attacks, where the goal is to change the $top$-$K$ predictions of the target classifier. We introduce novel gradient-based methods to find a good initial boundary point to attack. Our iterative method employs novel gradient estimation techniques, particularly effective in $top$-$K$ setting, on the decision boundary to effectively exploit the geometry of the decision boundary. Additionally, GSBA$^K$ can be used to attack against classifiers with $top$-$K$ multi-label learning. Extensive experimental results on ImageNet and PASCAL VOC datasets validate the effectiveness of GSBA$^K$ in crafting $top$-$K$ adversarial examples.



## **17. Cascading Adversarial Bias from Injection to Distillation in Language Models**

cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24842v1) [paper-pdf](http://arxiv.org/pdf/2505.24842v1)

**Authors**: Harsh Chaudhari, Jamie Hayes, Matthew Jagielski, Ilia Shumailov, Milad Nasr, Alina Oprea

**Abstract**: Model distillation has become essential for creating smaller, deployable language models that retain larger system capabilities. However, widespread deployment raises concerns about resilience to adversarial manipulation. This paper investigates vulnerability of distilled models to adversarial injection of biased content during training. We demonstrate that adversaries can inject subtle biases into teacher models through minimal data poisoning, which propagates to student models and becomes significantly amplified. We propose two propagation modes: Untargeted Propagation, where bias affects multiple tasks, and Targeted Propagation, focusing on specific tasks while maintaining normal behavior elsewhere. With only 25 poisoned samples (0.25% poisoning rate), student models generate biased responses 76.9% of the time in targeted scenarios - higher than 69.4% in teacher models. For untargeted propagation, adversarial bias appears 6x-29x more frequently in student models on unseen tasks. We validate findings across six bias types (targeted advertisements, phishing links, narrative manipulations, insecure coding practices), various distillation methods, and different modalities spanning text and code generation. Our evaluation reveals shortcomings in current defenses - perplexity filtering, bias detection systems, and LLM-based autorater frameworks - against these attacks. Results expose significant security vulnerabilities in distilled models, highlighting need for specialized safeguards. We propose practical design principles for building effective adversarial bias mitigation strategies.



## **18. ByzFL: Research Framework for Robust Federated Learning**

cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24802v1) [paper-pdf](http://arxiv.org/pdf/2505.24802v1)

**Authors**: Marc González, Rachid Guerraoui, Rafael Pinot, Geovani Rizk, John Stephan, François Taïani

**Abstract**: We present ByzFL, an open-source Python library for developing and benchmarking robust federated learning (FL) algorithms. ByzFL provides a unified and extensible framework that includes implementations of state-of-the-art robust aggregators, a suite of configurable attacks, and tools for simulating a variety of FL scenarios, including heterogeneous data distributions, multiple training algorithms, and adversarial threat models. The library enables systematic experimentation via a single JSON-based configuration file and includes built-in utilities for result visualization. Compatible with PyTorch tensors and NumPy arrays, ByzFL is designed to facilitate reproducible research and rapid prototyping of robust FL solutions. ByzFL is available at https://byzfl.epfl.ch/, with source code hosted on GitHub: https://github.com/LPD-EPFL/byzfl.



## **19. PatchDEMUX: A Certifiably Robust Framework for Multi-label Classifiers Against Adversarial Patches**

cs.CR

CVPR 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24703v1) [paper-pdf](http://arxiv.org/pdf/2505.24703v1)

**Authors**: Dennis Jacob, Chong Xiang, Prateek Mittal

**Abstract**: Deep learning techniques have enabled vast improvements in computer vision technologies. Nevertheless, these models are vulnerable to adversarial patch attacks which catastrophically impair performance. The physically realizable nature of these attacks calls for certifiable defenses, which feature provable guarantees on robustness. While certifiable defenses have been successfully applied to single-label classification, limited work has been done for multi-label classification. In this work, we present PatchDEMUX, a certifiably robust framework for multi-label classifiers against adversarial patches. Our approach is a generalizable method which can extend any existing certifiable defense for single-label classification; this is done by considering the multi-label classification task as a series of isolated binary classification problems to provably guarantee robustness. Furthermore, in the scenario where an attacker is limited to a single patch we propose an additional certification procedure that can provide tighter robustness bounds. Using the current state-of-the-art (SOTA) single-label certifiable defense PatchCleanser as a backbone, we find that PatchDEMUX can achieve non-trivial robustness on the MS-COCO and PASCAL VOC datasets while maintaining high clean performance



## **20. So, I climbed to the top of the pyramid of pain -- now what?**

cs.CR

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24685v1) [paper-pdf](http://arxiv.org/pdf/2505.24685v1)

**Authors**: Vasilis Katos, Emily Rosenorn-Lanng, Jane Henriksen-Bulmer, Ala Yankouskaya

**Abstract**: This paper explores the evolving dynamics of cybersecurity in the age of advanced AI, from the perspective of the introduced Human Layer Kill Chain framework. As traditional attack models like Lockheed Martin's Cyber Kill Chain become inadequate in addressing human vulnerabilities exploited by modern adversaries, the Humal Layer Kill Chain offers a nuanced approach that integrates human psychology and behaviour into the analysis of cyber threats. We detail the eight stages of the Human Layer Kill Chain, illustrating how AI-enabled techniques can enhance psychological manipulation in attacks. By merging the Human Layer with the Cyber Kill Chain, we propose a Sociotechnical Kill Plane that allows for a holistic examination of attackers' tactics, techniques, and procedures (TTPs) across the sociotechnical landscape. This framework not only aids cybersecurity professionals in understanding adversarial methods, but also empowers non-technical personnel to engage in threat identification and response. The implications for incident response and organizational resilience are significant, particularly as AI continues to shape the threat landscape.



## **21. Black-box Adversarial Attacks on CNN-based SLAM Algorithms**

cs.RO

9 pages, 8 figures

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24654v1) [paper-pdf](http://arxiv.org/pdf/2505.24654v1)

**Authors**: Maria Rafaela Gkeka, Bowen Sun, Evgenia Smirni, Christos D. Antonopoulos, Spyros Lalis, Nikolaos Bellas

**Abstract**: Continuous advancements in deep learning have led to significant progress in feature detection, resulting in enhanced accuracy in tasks like Simultaneous Localization and Mapping (SLAM). Nevertheless, the vulnerability of deep neural networks to adversarial attacks remains a challenge for their reliable deployment in applications, such as navigation of autonomous agents. Even though CNN-based SLAM algorithms are a growing area of research there is a notable absence of a comprehensive presentation and examination of adversarial attacks targeting CNN-based feature detectors, as part of a SLAM system. Our work introduces black-box adversarial perturbations applied to the RGB images fed into the GCN-SLAM algorithm. Our findings on the TUM dataset [30] reveal that even attacks of moderate scale can lead to tracking failure in as many as 76% of the frames. Moreover, our experiments highlight the catastrophic impact of attacking depth instead of RGB input images on the SLAM system.



## **22. A Flat Minima Perspective on Understanding Augmentations and Model Robustness**

cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24592v1) [paper-pdf](http://arxiv.org/pdf/2505.24592v1)

**Authors**: Weebum Yoo, Sung Whan Yoon

**Abstract**: Model robustness indicates a model's capability to generalize well on unforeseen distributional shifts, including data corruption, adversarial attacks, and domain shifts. Data augmentation is one of the prevalent and effective ways to enhance robustness. Despite the great success of augmentations in different fields, a general theoretical understanding of their efficacy in improving model robustness is lacking. We offer a unified theoretical framework to clarify how augmentations can enhance model robustness through the lens of loss surface flatness and PAC generalization bound. Our work diverges from prior studies in that our analysis i) broadly encompasses much of the existing augmentation methods, and ii) is not limited to specific types of distribution shifts like adversarial attacks. We confirm our theories through simulations on the existing common corruption and adversarial robustness benchmarks based on the CIFAR and ImageNet datasets, as well as domain generalization benchmarks including PACS and OfficeHome.



## **23. Stress-testing Machine Generated Text Detection: Shifting Language Models Writing Style to Fool Detectors**

cs.CL

Accepted at Findings of ACL 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24523v1) [paper-pdf](http://arxiv.org/pdf/2505.24523v1)

**Authors**: Andrea Pedrotti, Michele Papucci, Cristiano Ciaccio, Alessio Miaschi, Giovanni Puccetti, Felice Dell'Orletta, Andrea Esuli

**Abstract**: Recent advancements in Generative AI and Large Language Models (LLMs) have enabled the creation of highly realistic synthetic content, raising concerns about the potential for malicious use, such as misinformation and manipulation. Moreover, detecting Machine-Generated Text (MGT) remains challenging due to the lack of robust benchmarks that assess generalization to real-world scenarios. In this work, we present a pipeline to test the resilience of state-of-the-art MGT detectors (e.g., Mage, Radar, LLM-DetectAIve) to linguistically informed adversarial attacks. To challenge the detectors, we fine-tune language models using Direct Preference Optimization (DPO) to shift the MGT style toward human-written text (HWT). This exploits the detectors' reliance on stylistic clues, making new generations more challenging to detect. Additionally, we analyze the linguistic shifts induced by the alignment and which features are used by detectors to detect MGT texts. Our results show that detectors can be easily fooled with relatively few examples, resulting in a significant drop in detection performance. This highlights the importance of improving detection methods and making them robust to unseen in-domain texts.



## **24. IDEA: An Inverse Domain Expert Adaptation Based Active DNN IP Protection Method**

cs.CR

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2410.00059v2) [paper-pdf](http://arxiv.org/pdf/2410.00059v2)

**Authors**: Chaohui Xu, Qi Cui, Jinxin Dong, Weiyang He, Chip-Hong Chang

**Abstract**: Illegitimate reproduction, distribution and derivation of Deep Neural Network (DNN) models can inflict economic loss, reputation damage and even privacy infringement. Passive DNN intellectual property (IP) protection methods such as watermarking and fingerprinting attempt to prove the ownership upon IP violation, but they are often too late to stop catastrophic damage of IP abuse and too feeble against strong adversaries. In this paper, we propose IDEA, an Inverse Domain Expert Adaptation based proactive DNN IP protection method featuring active authorization and source traceability. IDEA generalizes active authorization as an inverse problem of domain adaptation. The multi-adaptive optimization is solved by a mixture-of-experts model with one real and two fake experts. The real expert re-optimizes the source model to correctly classify test images with a unique model user key steganographically embedded. The fake experts are trained to output random prediction on test images without or with incorrect user key embedded by minimizing their mutual information (MI) with the real expert. The MoE model is knowledge distilled into a unified protected model to avoid leaking the expert model features by maximizing their MI with additional multi-layer attention and contrastive representation loss optimization. IDEA not only prevents unauthorized users without the valid key to access the functional model, but also enable the model owner to validate the deployed model and trace the source of IP infringement. We extensively evaluate IDEA on five datasets and four DNN models to demonstrate its effectiveness in authorization control, culprit tracing success rate, and robustness against various attacks.



## **25. CAE-Net: Generalized Deepfake Image Detection using Convolution and Attention Mechanisms with Spatial and Frequency Domain Features**

cs.CV

Under review in Elsevier Journal of Visual Communication and Image  Representation

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2502.10682v2) [paper-pdf](http://arxiv.org/pdf/2502.10682v2)

**Authors**: Kafi Anan, Anindya Bhattacharjee, Ashir Intesher, Kaidul Islam, Abrar Assaeem Fuad, Utsab Saha, Hafiz Imtiaz

**Abstract**: Effective deepfake detection tools are becoming increasingly essential to the growing usage of deepfakes in unethical practices. There exists a wide range of deepfake generation techniques, which makes it challenging to develop an accurate universal detection mechanism. The 2025 IEEE Signal Processing Cup (\textit{DFWild-Cup} competition) provided a diverse dataset of deepfake images containing significant class imbalance. The images in the dataset are generated from multiple deepfake image generators, for training machine learning model(s) to emphasize the generalization of deepfake detection. To this end, we proposed a disjoint set-based multistage training method to address the class imbalance and devised an ensemble-based architecture \emph{CAE-Net}. Our architecture consists of a convolution- and attention-based ensemble network, and employs three different neural network architectures: EfficientNet, Data-Efficient Image Transformer (DeiT), and ConvNeXt with wavelet transform to capture both local and global features of deepfakes. We visualize the specific regions that these models focus on for classification using Grad-CAM, and empirically demonstrate the effectiveness of these models in grouping real and fake images into cohesive clusters using t-SNE plots. Individually, the EfficientNet B0 architecture has achieved 90.79\% accuracy, whereas the ConvNeXt and the DeiT architecture have achieved 89.49\% and 89.32\% accuracy, respectively. With these networks, our weighted ensemble model achieves an excellent accuracy of 94.63\% on the validation dataset of the SP Cup 2025 competition. The equal error rate of 4.72\% and the Area Under the ROC curve of 97.37\% further confirm the stability of our proposed method. Finally, the robustness of our proposed model against adversarial perturbation attacks is tested as well, showing the inherent defensive properties of the ensemble approach.



## **26. Adversarially Robust AI-Generated Image Detection for Free: An Information Theoretic Perspective**

cs.CV

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.22604v2) [paper-pdf](http://arxiv.org/pdf/2505.22604v2)

**Authors**: Ruixuan Zhang, He Wang, Zhengyu Zhao, Zhiqing Guo, Xun Yang, Yunfeng Diao, Meng Wang

**Abstract**: Rapid advances in Artificial Intelligence Generated Images (AIGI) have facilitated malicious use, such as forgery and misinformation. Therefore, numerous methods have been proposed to detect fake images. Although such detectors have been proven to be universally vulnerable to adversarial attacks, defenses in this field are scarce. In this paper, we first identify that adversarial training (AT), widely regarded as the most effective defense, suffers from performance collapse in AIGI detection. Through an information-theoretic lens, we further attribute the cause of collapse to feature entanglement, which disrupts the preservation of feature-label mutual information. Instead, standard detectors show clear feature separation. Motivated by this difference, we propose Training-free Robust Detection via Information-theoretic Measures (TRIM), the first training-free adversarial defense for AIGI detection. TRIM builds on standard detectors and quantifies feature shifts using prediction entropy and KL divergence. Extensive experiments across multiple datasets and attacks validate the superiority of our TRIM, e.g., outperforming the state-of-the-art defense by 33.88% (28.91%) on ProGAN (GenImage), while well maintaining original accuracy.



## **27. SEAR: A Multimodal Dataset for Analyzing AR-LLM-Driven Social Engineering Behaviors**

cs.AI

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24458v1) [paper-pdf](http://arxiv.org/pdf/2505.24458v1)

**Authors**: Tianlong Yu, Chenghang Ye, Zheyu Yang, Ziyi Zhou, Cui Tang, Zui Tao, Jun Zhang, Kailong Wang, Liting Zhou, Yang Yang, Ting Bi

**Abstract**: The SEAR Dataset is a novel multimodal resource designed to study the emerging threat of social engineering (SE) attacks orchestrated through augmented reality (AR) and multimodal large language models (LLMs). This dataset captures 180 annotated conversations across 60 participants in simulated adversarial scenarios, including meetings, classes and networking events. It comprises synchronized AR-captured visual/audio cues (e.g., facial expressions, vocal tones), environmental context, and curated social media profiles, alongside subjective metrics such as trust ratings and susceptibility assessments. Key findings reveal SEAR's alarming efficacy in eliciting compliance (e.g., 93.3% phishing link clicks, 85% call acceptance) and hijacking trust (76.7% post-interaction trust surge). The dataset supports research in detecting AR-driven SE attacks, designing defensive frameworks, and understanding multimodal adversarial manipulation. Rigorous ethical safeguards, including anonymization and IRB compliance, ensure responsible use. The SEAR dataset is available at https://github.com/INSLabCN/SEAR-Dataset.



## **28. Learning Safety Constraints for Large Language Models**

cs.LG

ICML 2025 (Spotlight)

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24445v1) [paper-pdf](http://arxiv.org/pdf/2505.24445v1)

**Authors**: Xin Chen, Yarden As, Andreas Krause

**Abstract**: Large language models (LLMs) have emerged as powerful tools but pose significant safety risks through harmful outputs and vulnerability to adversarial attacks. We propose SaP, short for Safety Polytope, a geometric approach to LLM safety that learns and enforces multiple safety constraints directly in the model's representation space. We develop a framework that identifies safe and unsafe regions via the polytope's facets, enabling both detection and correction of unsafe outputs through geometric steering. Unlike existing approaches that modify model weights, SaP operates post-hoc in the representation space, preserving model capabilities while enforcing safety constraints. Experiments across multiple LLMs demonstrate that our method can effectively detect unethical inputs, reduce adversarial attack success rates while maintaining performance on standard tasks, thus highlighting the importance of having an explicit geometric model for safety. Analysis of the learned polytope facets reveals emergence of specialization in detecting different semantic notions of safety, providing interpretable insights into how safety is captured in LLMs' representation space.



## **29. Breaking the Gold Standard: Extracting Forgotten Data under Exact Unlearning in Large Language Models**

cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24379v1) [paper-pdf](http://arxiv.org/pdf/2505.24379v1)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large language models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard, believed to be robust against privacy-related attacks. In this paper, we challenge this assumption by introducing a novel data extraction attack that compromises even exact unlearning. Our method leverages both the pre- and post-unlearning models: by guiding the post-unlearning model using signals from the pre-unlearning model, we uncover patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints.



## **30. Adversarial Preference Learning for Robust LLM Alignment**

cs.LG

Accepted at ACL2025 Findings

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24369v1) [paper-pdf](http://arxiv.org/pdf/2505.24369v1)

**Authors**: Yuanfu Wang, Pengyu Wang, Chenyang Xi, Bo Tang, Junyi Zhu, Wenqiang Wei, Chen Chen, Chao Yang, Jingfeng Zhang, Chaochao Lu, Yijun Niu, Keming Mao, Zhiyu Li, Feiyu Xiong, Jie Hu, Mingchuan Yang

**Abstract**: Modern language models often rely on Reinforcement Learning from Human Feedback (RLHF) to encourage safe behaviors. However, they remain vulnerable to adversarial attacks due to three key limitations: (1) the inefficiency and high cost of human annotation, (2) the vast diversity of potential adversarial attacks, and (3) the risk of feedback bias and reward hacking. To address these challenges, we introduce Adversarial Preference Learning (APL), an iterative adversarial training method incorporating three key innovations. First, a direct harmfulness metric based on the model's intrinsic preference probabilities, eliminating reliance on external assessment. Second, a conditional generative attacker that synthesizes input-specific adversarial variations. Third, an iterative framework with automated closed-loop feedback, enabling continuous adaptation through vulnerability discovery and mitigation. Experiments on Mistral-7B-Instruct-v0.3 demonstrate that APL significantly enhances robustness, achieving 83.33% harmlessness win rate over the base model (evaluated by GPT-4o), reducing harmful outputs from 5.88% to 0.43% (measured by LLaMA-Guard), and lowering attack success rate by up to 65% according to HarmBench. Notably, APL maintains competitive utility, with an MT-Bench score of 6.59 (comparable to the baseline 6.78) and an LC-WinRate of 46.52% against the base model.



## **31. Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction**

cs.CL

22 pages, 10 figures, accepted to ACL 2025 findings

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2502.11084v2) [paper-pdf](http://arxiv.org/pdf/2502.11084v2)

**Authors**: Yuting Huang, Chengyuan Liu, Yifeng Feng, Yiquan Wu, Chao Wu, Fei Wu, Kun Kuang

**Abstract**: As Large Language Models (LLMs) are widely applied in various domains, the safety of LLMs is increasingly attracting attention to avoid their powerful capabilities being misused. Existing jailbreak methods create a forced instruction-following scenario, or search adversarial prompts with prefix or suffix tokens to achieve a specific representation manually or automatically. However, they suffer from low efficiency and explicit jailbreak patterns, far from the real deployment of mass attacks to LLMs. In this paper, we point out that simply rewriting the original instruction can achieve a jailbreak, and we find that this rewriting approach is learnable and transferable. We propose the Rewrite to Jailbreak (R2J) approach, a transferable black-box jailbreak method to attack LLMs by iteratively exploring the weakness of the LLMs and automatically improving the attacking strategy. The jailbreak is more efficient and hard to identify since no additional features are introduced. Extensive experiments and analysis demonstrate the effectiveness of R2J, and we find that the jailbreak is also transferable to multiple datasets and various types of models with only a few queries. We hope our work motivates further investigation of LLM safety. The code can be found at https://github.com/ythuang02/R2J/.



## **32. Multi-Domain Graph Foundation Models: Robust Knowledge Transfer via Topology Alignment**

cs.SI

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2502.02017v2) [paper-pdf](http://arxiv.org/pdf/2502.02017v2)

**Authors**: Shuo Wang, Bokui Wang, Zhixiang Shen, Boyan Deng, Zhao Kang

**Abstract**: Recent advances in CV and NLP have inspired researchers to develop general-purpose graph foundation models through pre-training across diverse domains. However, a fundamental challenge arises from the substantial differences in graph topologies across domains. Additionally, real-world graphs are often sparse and prone to noisy connections and adversarial attacks. To address these issues, we propose the Multi-Domain Graph Foundation Model (MDGFM), a unified framework that aligns and leverages cross-domain topological information to facilitate robust knowledge transfer. MDGFM bridges different domains by adaptively balancing features and topology while refining original graphs to eliminate noise and align topological structures. To further enhance knowledge transfer, we introduce an efficient prompt-tuning approach. By aligning topologies, MDGFM not only improves multi-domain pre-training but also enables robust knowledge transfer to unseen domains. Theoretical analyses provide guarantees of MDGFM's effectiveness and domain generalization capabilities. Extensive experiments on both homophilic and heterophilic graph datasets validate the robustness and efficacy of our method.



## **33. When Are Concepts Erased From Diffusion Models?**

cs.LG

Project Page:  https://nyu-dice-lab.github.io/when-are-concepts-erased/

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.17013v4) [paper-pdf](http://arxiv.org/pdf/2505.17013v4)

**Authors**: Kevin Lu, Nicky Kriplani, Rohit Gandikota, Minh Pham, David Bau, Chinmay Hegde, Niv Cohen

**Abstract**: Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.



## **34. Safety Alignment Can Be Not Superficial With Explicit Safety Signals**

cs.CR

ICML 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.17072v2) [paper-pdf](http://arxiv.org/pdf/2505.17072v2)

**Authors**: Jianwei Li, Jung-Eun Kim

**Abstract**: Recent studies on the safety alignment of large language models (LLMs) have revealed that existing approaches often operate superficially, leaving models vulnerable to various adversarial attacks. Despite their significance, these studies generally fail to offer actionable solutions beyond data augmentation for achieving more robust safety mechanisms. This paper identifies a fundamental cause of this superficiality: existing alignment approaches often presume that models can implicitly learn a safety-related reasoning task during the alignment process, enabling them to refuse harmful requests. However, the learned safety signals are often diluted by other competing objectives, leading models to struggle with drawing a firm safety-conscious decision boundary when confronted with adversarial attacks. Based on this observation, by explicitly introducing a safety-related binary classification task and integrating its signals with our attention and decoding strategies, we eliminate this ambiguity and allow models to respond more responsibly to malicious queries. We emphasize that, with less than 0.2x overhead cost, our approach enables LLMs to assess the safety of both the query and the previously generated tokens at each necessary generating step. Extensive experiments demonstrate that our method significantly improves the resilience of LLMs against various adversarial attacks, offering a promising pathway toward more robust generative AI systems.



## **35. Light as Deception: GPT-driven Natural Relighting Against Vision-Language Pre-training Models**

cs.CV

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24227v1) [paper-pdf](http://arxiv.org/pdf/2505.24227v1)

**Authors**: Ying Yang, Jie Zhang, Xiao Lv, Di Lin, Tao Xiang, Qing Guo

**Abstract**: While adversarial attacks on vision-and-language pretraining (VLP) models have been explored, generating natural adversarial samples crafted through realistic and semantically meaningful perturbations remains an open challenge. Existing methods, primarily designed for classification tasks, struggle when adapted to VLP models due to their restricted optimization spaces, leading to ineffective attacks or unnatural artifacts. To address this, we propose \textbf{LightD}, a novel framework that generates natural adversarial samples for VLP models via semantically guided relighting. Specifically, LightD leverages ChatGPT to propose context-aware initial lighting parameters and integrates a pretrained relighting model (IC-light) to enable diverse lighting adjustments. LightD expands the optimization space while ensuring perturbations align with scene semantics. Additionally, gradient-based optimization is applied to the reference lighting image to further enhance attack effectiveness while maintaining visual naturalness. The effectiveness and superiority of the proposed LightD have been demonstrated across various VLP models in tasks such as image captioning and visual question answering.



## **36. On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains**

cs.CR

Accepted by ICML 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2409.17275v2) [paper-pdf](http://arxiv.org/pdf/2409.17275v2)

**Authors**: Xun Xian, Ganghua Wang, Xuan Bi, Jayanth Srinivasa, Ashish Kundu, Charles Fleming, Mingyi Hong, Jie Ding

**Abstract**: Retrieval-Augmented Generation (RAG) has been empirically shown to enhance the performance of large language models (LLMs) in knowledge-intensive domains such as healthcare, finance, and legal contexts. Given a query, RAG retrieves relevant documents from a corpus and integrates them into the LLMs' generation process. In this study, we investigate the adversarial robustness of RAG, focusing specifically on examining the retrieval system. First, across 225 different setup combinations of corpus, retriever, query, and targeted information, we show that retrieval systems are vulnerable to universal poisoning attacks in medical Q\&A. In such attacks, adversaries generate poisoned documents containing a broad spectrum of targeted information, such as personally identifiable information. When these poisoned documents are inserted into a corpus, they can be accurately retrieved by any users, as long as attacker-specified queries are used. To understand this vulnerability, we discovered that the deviation from the query's embedding to that of the poisoned document tends to follow a pattern in which the high similarity between the poisoned document and the query is retained, thereby enabling precise retrieval. Based on these findings, we develop a new detection-based defense to ensure the safe use of RAG. Through extensive experiments spanning various Q\&A domains, we observed that our proposed method consistently achieves excellent detection rates in nearly all cases.



## **37. GNNBleed: Inference Attacks to Unveil Private Edges in Graphs with Realistic Access to GNN Models**

cs.CR

The paper has been accepted to the PoPETs 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2311.16139v2) [paper-pdf](http://arxiv.org/pdf/2311.16139v2)

**Authors**: Zeyu Song, Ehsanul Kabir, Shagufta Mehnaz

**Abstract**: Graph Neural Networks (GNNs) have become indispensable tools for learning from graph structured data, catering to various applications such as social network analysis and fraud detection for financial services. At the heart of these networks are the edges, which are crucial in guiding GNN models' predictions. In many scenarios, these edges represent sensitive information, such as personal associations or financial dealings, which require privacy assurance. However, their contributions to GNN model predictions may, in turn, be exploited by the adversary to compromise their privacy. Motivated by these conflicting requirements, this paper investigates edge privacy in contexts where adversaries possess only black-box access to the target GNN model, restricted further by access controls, preventing direct insights into arbitrary node outputs. Moreover, we are the first to extensively examine situations where the target graph continuously evolves, a common trait of many real-world graphs. In this setting, we present a range of attacks that leverage the message-passing mechanism of GNNs. We evaluated the effectiveness of our attacks using nine real-world datasets, encompassing both static and dynamic graphs, across four different GNN architectures. The results demonstrate that our attack outperforms existing methods across various GNN architectures, consistently achieving an F1 score of at least 0.8 in static scenarios. Furthermore, our attack retains robustness in dynamic graph scenarios, maintaining F1 scores up to 0.8, unlike previous methods that only achieve F1 scores around 0.2.



## **38. Latent-space adversarial training with post-aware calibration for defending large language models against jailbreak attacks**

cs.CR

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2501.10639v3) [paper-pdf](http://arxiv.org/pdf/2501.10639v3)

**Authors**: Xin Yi, Yue Li, Dongsheng Shi, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Ensuring safety alignment is a critical requirement for large language models (LLMs), particularly given increasing deployment in real-world applications. Despite considerable advancements, LLMs remain susceptible to jailbreak attacks, which exploit system vulnerabilities to circumvent safety measures and elicit harmful or inappropriate outputs. Furthermore, while adversarial training-based defense methods have shown promise, a prevalent issue is the unintended over-defense behavior, wherein models excessively reject benign queries, significantly undermining their practical utility. To address these limitations, we introduce LATPC, a Latent-space Adversarial Training with Post-aware Calibration framework. LATPC dynamically identifies safety-critical latent dimensions by contrasting harmful and benign inputs, enabling the adaptive construction of targeted refusal feature removal attacks. This mechanism allows adversarial training to concentrate on real-world jailbreak tactics that disguise harmful queries as benign ones. During inference, LATPC employs an efficient embedding-level calibration mechanism to minimize over-defense behaviors with negligible computational overhead. Experimental results across five types of disguise-based jailbreak attacks demonstrate that LATPC achieves a superior balance between safety and utility compared to existing defense frameworks. Further analysis demonstrates the effectiveness of leveraging safety-critical dimensions in developing robust defense methods against jailbreak attacks.



## **39. JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models**

cs.CV

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.19610v2) [paper-pdf](http://arxiv.org/pdf/2505.19610v2)

**Authors**: Jiaxin Song, Yixu Wang, Jie Li, Rui Yu, Yan Teng, Xingjun Ma, Yingchun Wang

**Abstract**: Vision-Language Models (VLMs) exhibit impressive performance, yet the integration of powerful vision encoders has significantly broadened their attack surface, rendering them increasingly susceptible to jailbreak attacks. However, lacking well-defined attack objectives, existing jailbreak methods often struggle with gradient-based strategies prone to local optima and lacking precise directional guidance, and typically decouple visual and textual modalities, thereby limiting their effectiveness by neglecting crucial cross-modal interactions. Inspired by the Eliciting Latent Knowledge (ELK) framework, we posit that VLMs encode safety-relevant information within their internal fusion-layer representations, revealing an implicit safety decision boundary in the latent space. This motivates exploiting boundary to steer model behavior. Accordingly, we propose JailBound, a novel latent space jailbreak framework comprising two stages: (1) Safety Boundary Probing, which addresses the guidance issue by approximating decision boundary within fusion layer's latent space, thereby identifying optimal perturbation directions towards the target region; and (2) Safety Boundary Crossing, which overcomes the limitations of decoupled approaches by jointly optimizing adversarial perturbations across both image and text inputs. This latter stage employs an innovative mechanism to steer the model's internal state towards policy-violating outputs while maintaining cross-modal semantic consistency. Extensive experiments on six diverse VLMs demonstrate JailBound's efficacy, achieves 94.32% white-box and 67.28% black-box attack success averagely, which are 6.17% and 21.13% higher than SOTA methods, respectively. Our findings expose a overlooked safety risk in VLMs and highlight the urgent need for more robust defenses. Warning: This paper contains potentially sensitive, harmful and offensive content.



## **40. The Butterfly Effect in Pathology: Exploring Security in Pathology Foundation Models**

cs.CV

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24141v1) [paper-pdf](http://arxiv.org/pdf/2505.24141v1)

**Authors**: Jiashuai Liu, Yingjia Shang, Yingkang Zhan, Di Zhang, Yi Niu, Dong Wei, Xian Wu, Zeyu Gao, Chen Li, Yefeng Zheng

**Abstract**: With the widespread adoption of pathology foundation models in both research and clinical decision support systems, exploring their security has become a critical concern. However, despite their growing impact, the vulnerability of these models to adversarial attacks remains largely unexplored. In this work, we present the first systematic investigation into the security of pathology foundation models for whole slide image~(WSI) analysis against adversarial attacks. Specifically, we introduce the principle of \textit{local perturbation with global impact} and propose a label-free attack framework that operates without requiring access to downstream task labels. Under this attack framework, we revise four classical white-box attack methods and redefine the perturbation budget based on the characteristics of WSI. We conduct comprehensive experiments on three representative pathology foundation models across five datasets and six downstream tasks. Despite modifying only 0.1\% of patches per slide with imperceptible noise, our attack leads to downstream accuracy degradation that can reach up to 20\% in the worst cases. Furthermore, we analyze key factors that influence attack success, explore the relationship between patch-level vulnerability and semantic content, and conduct a preliminary investigation into potential defence strategies. These findings lay the groundwork for future research on the adversarial robustness and reliable deployment of pathology foundation models. Our code is publicly available at: https://github.com/Jiashuai-Liu-hmos/Attack-WSI-pathology-foundation-models.



## **41. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP**

cs.CV

ICML 2025

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.05528v3) [paper-pdf](http://arxiv.org/pdf/2505.05528v3)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.



## **42. LLM Agents Should Employ Security Principles**

cs.CR

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.24019v1) [paper-pdf](http://arxiv.org/pdf/2505.24019v1)

**Authors**: Kaiyuan Zhang, Zian Su, Pin-Yu Chen, Elisa Bertino, Xiangyu Zhang, Ninghui Li

**Abstract**: Large Language Model (LLM) agents show considerable promise for automating complex tasks using contextual reasoning; however, interactions involving multiple agents and the system's susceptibility to prompt injection and other forms of context manipulation introduce new vulnerabilities related to privacy leakage and system exploitation. This position paper argues that the well-established design principles in information security, which are commonly referred to as security principles, should be employed when deploying LLM agents at scale. Design principles such as defense-in-depth, least privilege, complete mediation, and psychological acceptability have helped guide the design of mechanisms for securing information systems over the last five decades, and we argue that their explicit and conscientious adoption will help secure agentic systems. To illustrate this approach, we introduce AgentSandbox, a conceptual framework embedding these security principles to provide safeguards throughout an agent's life-cycle. We evaluate with state-of-the-art LLMs along three dimensions: benign utility, attack utility, and attack success rate. AgentSandbox maintains high utility for its intended functions under both benign and adversarial evaluations while substantially mitigating privacy risks. By embedding secure design principles as foundational elements within emerging LLM agent protocols, we aim to promote trustworthy agent ecosystems aligned with user privacy expectations and evolving regulatory requirements.



## **43. Approaching the Harm of Gradient Attacks While Only Flipping Labels**

cs.CR

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2503.00140v2) [paper-pdf](http://arxiv.org/pdf/2503.00140v2)

**Authors**: Abdessamad El-Kabid, El-Mahdi El-Mhamdi

**Abstract**: Machine learning systems deployed in distributed or federated environments are highly susceptible to adversarial manipulations, particularly availability attacks -adding imperceptible perturbations to training data, thereby rendering the trained model unavailable. Prior research in distributed machine learning has demonstrated such adversarial effects through the injection of gradients or data poisoning. In this study, we aim to enhance comprehension of the potential of weaker (and more probable) adversaries by posing the following inquiry: Can availability attacks be inflicted solely through the flipping of a subset of training labels, without altering features, and under a strict flipping budget? We analyze the extent of damage caused by constrained label flipping attacks. Focusing on a distributed classification problem, (1) we propose a novel formalization of label flipping attacks on logistic regression models and derive a greedy algorithm that is provably optimal at each training step. (2) To demonstrate that availability attacks can be approached by label flipping alone, we show that a budget of only $0.1\%$ of labels at each training step can reduce the accuracy of the model by $6\%$, and that some models can perform worse than random guessing when up to $25\%$ of labels are flipped. (3) We shed light on an interesting interplay between what the attacker gains from more write-access versus what they gain from more flipping budget. (4) we define and compare the power of targeted label flipping attack to that of an untargeted label flipping attack.



## **44. HoneySat: A Network-based Satellite Honeypot Framework**

cs.CR

Efr\'en L\'opez-Morales and Ulysse Planta contributed equally to this  work

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.24008v1) [paper-pdf](http://arxiv.org/pdf/2505.24008v1)

**Authors**: Efrén López-Morales, Ulysse Planta, Gabriele Marra, Carlos González, Jacob Hopkins, Majid Garoosi, Elías Obreque, Carlos Rubio-Medrano, Ali Abbasi

**Abstract**: Satellites are the backbone of several mission-critical services, such as GPS that enable our modern society to function. For many years, satellites were assumed to be secure because of their indecipherable architectures and the reliance on security by obscurity. However, technological advancements have made these assumptions obsolete, paving the way for potential attacks, and sparking a renewed interest in satellite security. Unfortunately, to this day, there is no efficient way to collect data on adversarial techniques for satellites, which severely hurts the generation of security intelligence. In this paper, we present HoneySat, the first high-interaction satellite honeypot framework, which is fully capable of convincingly simulating a real-world CubeSat, a type of Small Satellite (SmallSat) widely used in practice. To provide evidence of the effectiveness of HoneySat, we surveyed experienced SmallSat operators currently in charge of active in-orbit satellite missions. Results revealed that the majority of satellite operators (71.4%) agreed that HoneySat provides realistic and engaging simulations of CubeSat missions. Further experimental evaluations also showed that HoneySat provides adversaries with extensive interaction opportunities by supporting the majority of adversarial techniques (86.8%) and tactics (100%) that target satellites. Additionally, we also obtained a series of real interactions from actual adversaries by deploying HoneySat on the internet over several months, confirming that HoneySat can operate covertly and efficiently while collecting highly valuable interaction data.



## **45. SVIP: Towards Verifiable Inference of Open-source Large Language Models**

cs.LG

22 pages

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2410.22307v2) [paper-pdf](http://arxiv.org/pdf/2410.22307v2)

**Authors**: Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, Huan Zhang

**Abstract**: The ever-increasing size of open-source Large Language Models (LLMs) renders local deployment impractical for individual users. Decentralized computing has emerged as a cost-effective solution, allowing individuals and small companies to perform LLM inference for users using surplus computational power. However, a computing provider may stealthily substitute the requested LLM with a smaller, less capable model without consent from users, thereby benefiting from cost savings. We introduce SVIP, a secret-based verifiable LLM inference protocol. Unlike existing solutions based on cryptographic or game-theoretic techniques, our method is computationally effective and does not rest on strong assumptions. Our protocol requires the computing provider to return both the generated text and processed hidden representations from LLMs. We then train a proxy task on these representations, effectively transforming them into a unique model identifier. With our protocol, users can reliably verify whether the computing provider is acting honestly. A carefully integrated secret mechanism further strengthens its security. We thoroughly analyze our protocol under multiple strong and adaptive adversarial scenarios. Our extensive experiments demonstrate that SVIP is accurate, generalizable, computationally efficient, and resistant to various attacks. Notably, SVIP achieves false negative rates below 5% and false positive rates below 3%, while requiring less than 0.01 seconds per prompt query for verification.



## **46. SafeScientist: Toward Risk-Aware Scientific Discoveries by LLM Agents**

cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23559v1) [paper-pdf](http://arxiv.org/pdf/2505.23559v1)

**Authors**: Kunlun Zhu, Jiaxun Zhang, Ziheng Qi, Nuoxing Shang, Zijia Liu, Peixuan Han, Yue Su, Haofei Yu, Jiaxuan You

**Abstract**: Recent advancements in large language model (LLM) agents have significantly accelerated scientific discovery automation, yet concurrently raised critical ethical and safety concerns. To systematically address these challenges, we introduce \textbf{SafeScientist}, an innovative AI scientist framework explicitly designed to enhance safety and ethical responsibility in AI-driven scientific exploration. SafeScientist proactively refuses ethically inappropriate or high-risk tasks and rigorously emphasizes safety throughout the research process. To achieve comprehensive safety oversight, we integrate multiple defensive mechanisms, including prompt monitoring, agent-collaboration monitoring, tool-use monitoring, and an ethical reviewer component. Complementing SafeScientist, we propose \textbf{SciSafetyBench}, a novel benchmark specifically designed to evaluate AI safety in scientific contexts, comprising 240 high-risk scientific tasks across 6 domains, alongside 30 specially designed scientific tools and 120 tool-related risk tasks. Extensive experiments demonstrate that SafeScientist significantly improves safety performance by 35\% compared to traditional AI scientist frameworks, without compromising scientific output quality. Additionally, we rigorously validate the robustness of our safety pipeline against diverse adversarial attack methods, further confirming the effectiveness of our integrated approach. The code and data will be available at https://github.com/ulab-uiuc/SafeScientist. \textcolor{red}{Warning: this paper contains example data that may be offensive or harmful.}



## **47. SGD Jittering: A Training Strategy for Robust and Accurate Model-Based Architectures**

cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2410.14667v2) [paper-pdf](http://arxiv.org/pdf/2410.14667v2)

**Authors**: Peimeng Guan, Mark A. Davenport

**Abstract**: Inverse problems aim to reconstruct unseen data from corrupted or perturbed measurements. While most work focuses on improving reconstruction quality, generalization accuracy and robustness are equally important, especially for safety-critical applications. Model-based architectures (MBAs), such as loop unrolling methods, are considered more interpretable and achieve better reconstructions. Empirical evidence suggests that MBAs are more robust to perturbations than black-box solvers, but the accuracy-robustness tradeoff in MBAs remains underexplored. In this work, we propose a simple yet effective training scheme for MBAs, called SGD jittering, which injects noise iteration-wise during reconstruction. We theoretically demonstrate that SGD jittering not only generalizes better than the standard mean squared error training but is also more robust to average-case attacks. We validate SGD jittering using denoising toy examples, seismic deconvolution, and single-coil MRI reconstruction. Both SGD jittering and its SPGD extension yield cleaner reconstructions for out-of-distribution data and demonstrates enhanced robustness against adversarial attacks.



## **48. TRAP: Targeted Redirecting of Agentic Preferences**

cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23518v1) [paper-pdf](http://arxiv.org/pdf/2505.23518v1)

**Authors**: Hangoo Kang, Jehyeok Yeon, Gagandeep Singh

**Abstract**: Autonomous agentic AI systems powered by vision-language models (VLMs) are rapidly advancing toward real-world deployment, yet their cross-modal reasoning capabilities introduce new attack surfaces for adversarial manipulation that exploit semantic reasoning across modalities. Existing adversarial attacks typically rely on visible pixel perturbations or require privileged model or environment access, making them impractical for stealthy, real-world exploitation. We introduce TRAP, a generative adversarial framework that manipulates the agent's decision-making using diffusion-based semantic injections. Our method combines negative prompt-based degradation with positive semantic optimization, guided by a Siamese semantic network and layout-aware spatial masking. Without requiring access to model internals, TRAP produces visually natural images yet induces consistent selection biases in agentic AI systems. We evaluate TRAP on the Microsoft Common Objects in Context (COCO) dataset, building multi-candidate decision scenarios. Across these scenarios, TRAP achieves a 100% attack success rate on leading models, including LLaVA-34B, Gemma3, and Mistral-3.1, significantly outperforming baselines such as SPSA, Bandit, and standard diffusion approaches. These results expose a critical vulnerability: Autonomous agents can be consistently misled through human-imperceptible cross-modal manipulations. These findings highlight the need for defense strategies beyond pixel-level robustness to address semantic vulnerabilities in cross-modal decision-making.



## **49. Hijacking Large Language Models via Adversarial In-Context Learning**

cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2311.09948v3) [paper-pdf](http://arxiv.org/pdf/2311.09948v3)

**Authors**: Xiangyu Zhou, Yao Qiang, Saleh Zare Zade, Prashant Khanduri, Dongxiao Zhu

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific downstream tasks by utilizing labeled examples as demonstrations (demos) in the preconditioned prompts. Despite its promising performance, crafted adversarial attacks pose a notable threat to the robustness of LLMs. Existing attacks are either easy to detect, require a trigger in user input, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable prompt injection attack against ICL, aiming to hijack LLMs to generate the target output or elicit harmful responses. In our threat model, the hacker acts as a model publisher who leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demos via prompt injection. We also propose effective defense strategies using a few shots of clean demos, enhancing the robustness of LLMs during ICL. Extensive experimental results across various classification and jailbreak tasks demonstrate the effectiveness of the proposed attack and defense strategies. This work highlights the significant security vulnerabilities of LLMs during ICL and underscores the need for further in-depth studies.



## **50. Learning to Poison Large Language Models for Downstream Manipulation**

cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2402.13459v3) [paper-pdf](http://arxiv.org/pdf/2402.13459v3)

**Authors**: Xiangyu Zhou, Yao Qiang, Saleh Zare Zade, Mohammad Amin Roshani, Prashant Khanduri, Douglas Zytko, Dongxiao Zhu

**Abstract**: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where the adversary inserts backdoor triggers into training data to manipulate outputs. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the supervised fine-tuning (SFT) process. We propose a novel gradient-guided backdoor trigger learning (GBTL) algorithm to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various language model tasks, including sentiment analysis, domain generation, and question answering, our poisoning strategy demonstrates a high success rate in compromising various LLMs' outputs. We further propose two defense strategies against data poisoning attacks, including in-context learning (ICL) and continuous learning (CL), which effectively rectify the behavior of LLMs and significantly reduce the decline in performance. Our work highlights the significant security risks present during SFT of LLMs and the necessity of safeguarding LLMs against data poisoning attacks.



