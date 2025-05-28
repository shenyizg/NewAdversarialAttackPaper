# Latest Adversarial Attack Papers
**update at 2025-05-28 14:53:55**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. AdInject: Real-World Black-Box Attacks on Web Agents via Advertising Delivery**

cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21499v1) [paper-pdf](http://arxiv.org/pdf/2505.21499v1)

**Authors**: Haowei Wang, Junjie Wang, Xiaojun Jia, Rupeng Zhang, Mingyang Li, Zhe Liu, Yang Liu, Qing Wang

**Abstract**: Vision-Language Model (VLM) based Web Agents represent a significant step towards automating complex tasks by simulating human-like interaction with websites. However, their deployment in uncontrolled web environments introduces significant security vulnerabilities. Existing research on adversarial environmental injection attacks often relies on unrealistic assumptions, such as direct HTML manipulation, knowledge of user intent, or access to agent model parameters, limiting their practical applicability. In this paper, we propose AdInject, a novel and real-world black-box attack method that leverages the internet advertising delivery to inject malicious content into the Web Agent's environment. AdInject operates under a significantly more realistic threat model than prior work, assuming a black-box agent, static malicious content constraints, and no specific knowledge of user intent. AdInject includes strategies for designing malicious ad content aimed at misleading agents into clicking, and a VLM-based ad content optimization technique that infers potential user intents from the target website's context and integrates these intents into the ad content to make it appear more relevant or critical to the agent's task, thus enhancing attack effectiveness. Experimental evaluations demonstrate the effectiveness of AdInject, attack success rates exceeding 60% in most scenarios and approaching 100% in certain cases. This strongly demonstrates that prevalent advertising delivery constitutes a potent and real-world vector for environment injection attacks against Web Agents. This work highlights a critical vulnerability in Web Agent security arising from real-world environment manipulation channels, underscoring the urgent need for developing robust defense mechanisms against such threats. Our code is available at https://github.com/NicerWang/AdInject.



## **2. Adversarial Attacks against Closed-Source MLLMs via Feature Optimal Alignment**

cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21494v1) [paper-pdf](http://arxiv.org/pdf/2505.21494v1)

**Authors**: Xiaojun Jia, Sensen Gao, Simeng Qin, Tianyu Pang, Chao Du, Yihao Huang, Xinfeng Li, Yiming Li, Bo Li, Yang Liu

**Abstract**: Multimodal large language models (MLLMs) remain vulnerable to transferable adversarial examples. While existing methods typically achieve targeted attacks by aligning global features-such as CLIP's [CLS] token-between adversarial and target samples, they often overlook the rich local information encoded in patch tokens. This leads to suboptimal alignment and limited transferability, particularly for closed-source models. To address this limitation, we propose a targeted transferable adversarial attack method based on feature optimal alignment, called FOA-Attack, to improve adversarial transfer capability. Specifically, at the global level, we introduce a global feature loss based on cosine similarity to align the coarse-grained features of adversarial samples with those of target samples. At the local level, given the rich local representations within Transformers, we leverage clustering techniques to extract compact local patterns to alleviate redundant local features. We then formulate local feature alignment between adversarial and target samples as an optimal transport (OT) problem and propose a local clustering optimal transport loss to refine fine-grained feature alignment. Additionally, we propose a dynamic ensemble model weighting strategy to adaptively balance the influence of multiple models during adversarial example generation, thereby further improving transferability. Extensive experiments across various models demonstrate the superiority of the proposed method, outperforming state-of-the-art methods, especially in transferring to closed-source MLLMs. The code is released at https://github.com/jiaxiaojunQAQ/FOA-Attack.



## **3. When Are Concepts Erased From Diffusion Models?**

cs.LG

Project Page:  https://nyu-dice-lab.github.io/when-are-concepts-erased/

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.17013v3) [paper-pdf](http://arxiv.org/pdf/2505.17013v3)

**Authors**: Kevin Lu, Nicky Kriplani, Rohit Gandikota, Minh Pham, David Bau, Chinmay Hegde, Niv Cohen

**Abstract**: Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.



## **4. On the Robustness of Adversarial Training Against Uncertainty Attacks**

cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2410.21952v2) [paper-pdf](http://arxiv.org/pdf/2410.21952v2)

**Authors**: Emanuele Ledda, Giovanni Scodeller, Daniele Angioni, Giorgio Piras, Antonio Emanuele Cinà, Giorgio Fumera, Battista Biggio, Fabio Roli

**Abstract**: In learning problems, the noise inherent to the task at hand hinders the possibility to infer without a certain degree of uncertainty. Quantifying this uncertainty, regardless of its wide use, assumes high relevance for security-sensitive applications. Within these scenarios, it becomes fundamental to guarantee good (i.e., trustworthy) uncertainty measures, which downstream modules can securely employ to drive the final decision-making process. However, an attacker may be interested in forcing the system to produce either (i) highly uncertain outputs jeopardizing the system's availability or (ii) low uncertainty estimates, making the system accept uncertain samples that would instead require a careful inspection (e.g., human intervention). Therefore, it becomes fundamental to understand how to obtain robust uncertainty estimates against these kinds of attacks. In this work, we reveal both empirically and theoretically that defending against adversarial examples, i.e., carefully perturbed samples that cause misclassification, additionally guarantees a more secure, trustworthy uncertainty estimate under common attack scenarios without the need for an ad-hoc defense strategy. To support our claims, we evaluate multiple adversarial-robust models from the publicly available benchmark RobustBench on the CIFAR-10 and ImageNet datasets.



## **5. Attribute-Efficient PAC Learning of Sparse Halfspaces with Constant Malicious Noise Rate**

cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21430v1) [paper-pdf](http://arxiv.org/pdf/2505.21430v1)

**Authors**: Shiwei Zeng, Jie Shen

**Abstract**: Attribute-efficient learning of sparse halfspaces has been a fundamental problem in machine learning theory. In recent years, machine learning algorithms are faced with prevalent data corruptions or even adversarial attacks. It is of central interest to design efficient algorithms that are robust to noise corruptions. In this paper, we consider that there exists a constant amount of malicious noise in the data and the goal is to learn an underlying $s$-sparse halfspace $w^* \in \mathbb{R}^d$ with $\text{poly}(s,\log d)$ samples. Specifically, we follow a recent line of works and assume that the underlying distribution satisfies a certain concentration condition and a margin condition at the same time. Under such conditions, we show that attribute-efficiency can be achieved by simple variants to existing hinge loss minimization programs. Our key contribution includes: 1) an attribute-efficient PAC learning algorithm that works under constant malicious noise rate; 2) a new gradient analysis that carefully handles the sparsity constraint in hinge loss minimization.



## **6. A Framework for Adversarial Analysis of Decision Support Systems Prior to Deployment**

cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21414v1) [paper-pdf](http://arxiv.org/pdf/2505.21414v1)

**Authors**: Brett Bissey, Kyle Gatesman, Walker Dimon, Mohammad Alam, Luis Robaina, Joseph Weissman

**Abstract**: This paper introduces a comprehensive framework designed to analyze and secure decision-support systems trained with Deep Reinforcement Learning (DRL), prior to deployment, by providing insights into learned behavior patterns and vulnerabilities discovered through simulation. The introduced framework aids in the development of precisely timed and targeted observation perturbations, enabling researchers to assess adversarial attack outcomes within a strategic decision-making context. We validate our framework, visualize agent behavior, and evaluate adversarial outcomes within the context of a custom-built strategic game, CyberStrike. Utilizing the proposed framework, we introduce a method for systematically discovering and ranking the impact of attacks on various observation indices and time-steps, and we conduct experiments to evaluate the transferability of adversarial attacks across agent architectures and DRL training algorithms. The findings underscore the critical need for robust adversarial defense mechanisms to protect decision-making policies in high-stakes environments.



## **7. Optimizing Robustness and Accuracy in Mixture of Experts: A Dual-Model Approach**

cs.LG

15 pages, 7 figures, accepted by ICML 2025

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2502.06832v3) [paper-pdf](http://arxiv.org/pdf/2502.06832v3)

**Authors**: Xu Zhang, Kaidi Xu, Ziqing Hu, Ren Wang

**Abstract**: Mixture of Experts (MoE) have shown remarkable success in leveraging specialized expert networks for complex machine learning tasks. However, their susceptibility to adversarial attacks presents a critical challenge for deployment in robust applications. This paper addresses the critical question of how to incorporate robustness into MoEs while maintaining high natural accuracy. We begin by analyzing the vulnerability of MoE components, finding that expert networks are notably more susceptible to adversarial attacks than the router. Based on this insight, we propose a targeted robust training technique that integrates a novel loss function to enhance the adversarial robustness of MoE, requiring only the robustification of one additional expert without compromising training or inference efficiency. Building on this, we introduce a dual-model strategy that linearly combines a standard MoE model with our robustified MoE model using a smoothing parameter. This approach allows for flexible control over the robustness-accuracy trade-off. We further provide theoretical foundations by deriving certified robustness bounds for both the single MoE and the dual-model. To push the boundaries of robustness and accuracy, we propose a novel joint training strategy JTDMoE for the dual-model. This joint training enhances both robustness and accuracy beyond what is achievable with separate models. Experimental results on CIFAR-10 and TinyImageNet datasets using ResNet18 and Vision Transformer (ViT) architectures demonstrate the effectiveness of our proposed methods. The code is publicly available at https://github.com/TIML-Group/Robust-MoE-Dual-Model.



## **8. Boosting Adversarial Transferability via High-Frequency Augmentation and Hierarchical-Gradient Fusion**

cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21181v1) [paper-pdf](http://arxiv.org/pdf/2505.21181v1)

**Authors**: Yayin Zheng, Chen Wan, Zihong Guo, Hailing Kuang, Xiaohai Lu

**Abstract**: Adversarial attacks have become a significant challenge in the security of machine learning models, particularly in the context of black-box defense strategies. Existing methods for enhancing adversarial transferability primarily focus on the spatial domain. This paper presents Frequency-Space Attack (FSA), a new adversarial attack framework that effectively integrates frequency-domain and spatial-domain transformations. FSA combines two key techniques: (1) High-Frequency Augmentation, which applies Fourier transform with frequency-selective amplification to diversify inputs and emphasize the critical role of high-frequency components in adversarial attacks, and (2) Hierarchical-Gradient Fusion, which merges multi-scale gradient decomposition and fusion to capture both global structures and fine-grained details, resulting in smoother perturbations. Our experiment demonstrates that FSA consistently outperforms state-of-the-art methods across various black-box models. Notably, our proposed FSA achieves an average attack success rate increase of 23.6% compared with BSR (CVPR 2024) on eight black-box defense models.



## **9. Unraveling Indirect In-Context Learning Using Influence Functions**

cs.LG

Under Review

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2501.01473v2) [paper-pdf](http://arxiv.org/pdf/2501.01473v2)

**Authors**: Hadi Askari, Shivanshu Gupta, Terry Tong, Fei Wang, Anshuman Chhabra, Muhao Chen

**Abstract**: In this work, we introduce a novel paradigm for generalized In-Context Learning (ICL), termed Indirect In-Context Learning. In Indirect ICL, we explore demonstration selection strategies tailored for two distinct real-world scenarios: Mixture of Tasks and Noisy ICL. We systematically evaluate the effectiveness of Influence Functions (IFs) as a selection tool for these settings, highlighting the potential of IFs to better capture the informativeness of examples within the demonstration pool. For the Mixture of Tasks setting, demonstrations are drawn from 28 diverse tasks, including MMLU, BigBench, StrategyQA, and CommonsenseQA. We demonstrate that combining BertScore-Recall (BSR) with an IF surrogate model can further improve performance, leading to average absolute accuracy gains of 0.37\% and 1.45\% for 3-shot and 5-shot setups when compared to traditional ICL metrics. In the Noisy ICL setting, we examine scenarios where demonstrations might be mislabeled or have adversarial noise. Our experiments show that reweighting traditional ICL selectors (BSR and Cosine Similarity) with IF-based selectors boosts accuracy by an average of 2.90\% for Cosine Similarity and 2.94\% for BSR on noisy GLUE benchmarks. For the adversarial sub-setting, we show the utility of using IFs for task-agnostic demonstration selection for backdoor attack mitigation. Showing a 32.89\% reduction in Attack Success Rate compared to task-aware methods. In sum, we propose a robust framework for demonstration selection that generalizes beyond traditional ICL, offering valuable insights into the role of IFs for Indirect ICL.



## **10. TabAttackBench: A Benchmark for Adversarial Attacks on Tabular Data**

cs.LG

63 pages, 22 figures, 6 tables

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21027v1) [paper-pdf](http://arxiv.org/pdf/2505.21027v1)

**Authors**: Zhipeng He, Chun Ouyang, Lijie Wen, Cong Liu, Catarina Moreira

**Abstract**: Adversarial attacks pose a significant threat to machine learning models by inducing incorrect predictions through imperceptible perturbations to input data. While these attacks have been extensively studied in unstructured data like images, their application to tabular data presents new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ significantly from those in image data. To address these differences, it is crucial to consider imperceptibility as a key criterion specific to tabular data. Most current research focuses primarily on achieving effective adversarial attacks, often overlooking the importance of maintaining imperceptibility. To address this gap, we propose a new benchmark for adversarial attacks on tabular data that evaluates both effectiveness and imperceptibility. In this study, we assess the effectiveness and imperceptibility of five adversarial attacks across four models using eleven tabular datasets, including both mixed and numerical-only datasets. Our analysis explores how these factors interact and influence the overall performance of the attacks. We also compare the results across different dataset types to understand the broader implications of these findings. The findings from this benchmark provide valuable insights for improving the design of adversarial attack algorithms, thereby advancing the field of adversarial machine learning on tabular data.



## **11. Tradeoffs Between Alignment and Helpfulness in Language Models with Steering Methods**

cs.CL

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2401.16332v5) [paper-pdf](http://arxiv.org/pdf/2401.16332v5)

**Authors**: Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, Amnon Shashua

**Abstract**: Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. First, we find that under the conditions of our framework, alignment can be guaranteed with representation engineering, and at the same time that helpfulness is harmed in the process. Second, we show that helpfulness is harmed quadratically with the norm of the representation engineering vector, while the alignment increases linearly with it, indicating a regime in which it is efficient to use representation engineering. We validate our findings empirically, and chart the boundaries to the usefulness of representation engineering for alignment.



## **12. NatADiff: Adversarial Boundary Guidance for Natural Adversarial Diffusion**

cs.LG

10 pages, 3 figures, 2 tables

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20934v1) [paper-pdf](http://arxiv.org/pdf/2505.20934v1)

**Authors**: Max Collins, Jordan Vice, Tim French, Ajmal Mian

**Abstract**: Adversarial samples exploit irregularities in the manifold ``learned'' by deep learning models to cause misclassifications. The study of these adversarial samples provides insight into the features a model uses to classify inputs, which can be leveraged to improve robustness against future attacks. However, much of the existing literature focuses on constrained adversarial samples, which do not accurately reflect test-time errors encountered in real-world settings. To address this, we propose `NatADiff', an adversarial sampling scheme that leverages denoising diffusion to generate natural adversarial samples. Our approach is based on the observation that natural adversarial samples frequently contain structural elements from the adversarial class. Deep learning models can exploit these structural elements to shortcut the classification process, rather than learning to genuinely distinguish between classes. To leverage this behavior, we guide the diffusion trajectory towards the intersection of the true and adversarial classes, combining time-travel sampling with augmented classifier guidance to enhance attack transferability while preserving image fidelity. Our method achieves comparable attack success rates to current state-of-the-art techniques, while exhibiting significantly higher transferability across model architectures and better alignment with natural test-time errors as measured by FID. These results demonstrate that NatADiff produces adversarial samples that not only transfer more effectively across models, but more faithfully resemble naturally occurring test-time errors.



## **13. ZeroPur: Succinct Training-Free Adversarial Purification**

cs.CV

17 pages, 7 figures, under review

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2406.03143v3) [paper-pdf](http://arxiv.org/pdf/2406.03143v3)

**Authors**: Erhu Liu, Zonglin Yang, Bo Liu, Bin Xiao, Xiuli Bi

**Abstract**: Adversarial purification is a kind of defense technique that can defend against various unseen adversarial attacks without modifying the victim classifier. Existing methods often depend on external generative models or cooperation between auxiliary functions and victim classifiers. However, retraining generative models, auxiliary functions, or victim classifiers relies on the domain of the fine-tuned dataset and is computation-consuming. In this work, we suppose that adversarial images are outliers of the natural image manifold, and the purification process can be considered as returning them to this manifold. Following this assumption, we present a simple adversarial purification method without further training to purify adversarial images, called ZeroPur. ZeroPur contains two steps: given an adversarial example, Guided Shift obtains the shifted embedding of the adversarial example by the guidance of its blurred counterparts; after that, Adaptive Projection constructs a directional vector by this shifted embedding to provide momentum, projecting adversarial images onto the manifold adaptively. ZeroPur is independent of external models and requires no retraining of victim classifiers or auxiliary functions, relying solely on victim classifiers themselves to achieve purification. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) using various classifier architectures (ResNet, WideResNet) demonstrate that our method achieves state-of-the-art robust performance. The code will be publicly available.



## **14. Concealment of Intent: A Game-Theoretic Analysis**

cs.CL

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20841v1) [paper-pdf](http://arxiv.org/pdf/2505.20841v1)

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.



## **15. MedSentry: Understanding and Mitigating Safety Risks in Medical LLM Multi-Agent Systems**

cs.MA

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20824v1) [paper-pdf](http://arxiv.org/pdf/2505.20824v1)

**Authors**: Kai Chen, Taihang Zhen, Hewei Wang, Kailai Liu, Xinfeng Li, Jing Huo, Tianpei Yang, Jinfeng Xu, Wei Dong, Yang Gao

**Abstract**: As large language models (LLMs) are increasingly deployed in healthcare, ensuring their safety, particularly within collaborative multi-agent configurations, is paramount. In this paper we introduce MedSentry, a benchmark comprising 5 000 adversarial medical prompts spanning 25 threat categories with 100 subthemes. Coupled with this dataset, we develop an end-to-end attack-defense evaluation pipeline to systematically analyze how four representative multi-agent topologies (Layers, SharedPool, Centralized, and Decentralized) withstand attacks from 'dark-personality' agents. Our findings reveal critical differences in how these architectures handle information contamination and maintain robust decision-making, exposing their underlying vulnerability mechanisms. For instance, SharedPool's open information sharing makes it highly susceptible, whereas Decentralized architectures exhibit greater resilience thanks to inherent redundancy and isolation. To mitigate these risks, we propose a personality-scale detection and correction mechanism that identifies and rehabilitates malicious agents, restoring system safety to near-baseline levels. MedSentry thus furnishes both a rigorous evaluation framework and practical defense strategies that guide the design of safer LLM-based multi-agent systems in medical domains.



## **16. TrojanStego: Your Language Model Can Secretly Be A Steganographic Privacy Leaking Agent**

cs.CL

9 pages, 5 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20118v2) [paper-pdf](http://arxiv.org/pdf/2505.20118v2)

**Authors**: Dominik Meier, Jan Philip Wahle, Paul Röttger, Terry Ruas, Bela Gipp

**Abstract**: As large language models (LLMs) become integrated into sensitive workflows, concerns grow over their potential to leak confidential information. We propose TrojanStego, a novel threat model in which an adversary fine-tunes an LLM to embed sensitive context information into natural-looking outputs via linguistic steganography, without requiring explicit control over inference inputs. We introduce a taxonomy outlining risk factors for compromised LLMs, and use it to evaluate the risk profile of the threat. To implement TrojanStego, we propose a practical encoding scheme based on vocabulary partitioning learnable by LLMs via fine-tuning. Experimental results show that compromised models reliably transmit 32-bit secrets with 87% accuracy on held-out prompts, reaching over 97% accuracy using majority voting across three generations. Further, they maintain high utility, can evade human detection, and preserve coherence. These results highlight a new class of LLM data exfiltration attacks that are passive, covert, practical, and dangerous.



## **17. Breaking Dataset Boundaries: Class-Agnostic Targeted Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20782v1) [paper-pdf](http://arxiv.org/pdf/2505.20782v1)

**Authors**: Taïga Gonçalves, Tomo Miyazaki, Shinichiro Omachi

**Abstract**: We present Cross-Domain Multi-Targeted Attack (CD-MTA), a method for generating adversarial examples that mislead image classifiers toward any target class, including those not seen during training. Traditional targeted attacks are limited to one class per model, requiring expensive retraining for each target. Multi-targeted attacks address this by introducing a perturbation generator with a conditional input to specify the target class. However, existing methods are constrained to classes observed during training and require access to the black-box model's training data--introducing a form of data leakage that undermines realistic evaluation in practical black-box scenarios. We identify overreliance on class embeddings as a key limitation, leading to overfitting and poor generalization to unseen classes. To address this, CD-MTA replaces class-level supervision with an image-based conditional input and introduces class-agnostic losses that align the perturbed and target images in the feature space. This design removes dependence on class semantics, thereby enabling generalization to unseen classes across datasets. Experiments on ImageNet and seven other datasets show that CD-MTA outperforms prior multi-targeted attacks in both standard and cross-domain settings--without accessing the black-box model's training data.



## **18. SC-Pro: Training-Free Framework for Defending Unsafe Image Synthesis Attack**

cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2501.05359v2) [paper-pdf](http://arxiv.org/pdf/2501.05359v2)

**Authors**: Junha Park, Jaehui Hwang, Ian Ryu, Hyungkeun Park, Jiyoon Kim, Jong-Seok Lee

**Abstract**: With advances in diffusion models, image generation has shown significant performance improvements. This raises concerns about the potential abuse of image generation, such as the creation of explicit or violent images, commonly referred to as Not Safe For Work (NSFW) content. To address this, the Stable Diffusion model includes several safety checkers to censor initial text prompts and final output images generated from the model. However, recent research has shown that these safety checkers have vulnerabilities against adversarial attacks, allowing them to generate NSFW images. In this paper, we find that these adversarial attacks are not robust to small changes in text prompts or input latents. Based on this, we propose SC-Pro (Spherical or Circular Probing), a training-free framework that easily defends against adversarial attacks generating NSFW images. Moreover, we develop an approach that utilizes one-step diffusion models for efficient NSFW detection (SC-Pro-o), further reducing computational resources. We demonstrate the superiority of our method in terms of performance and applicability.



## **19. Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond**

cs.LG

Accepted by ICML 2025

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2502.05374v4) [paper-pdf](http://arxiv.org/pdf/2502.05374v4)

**Authors**: Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, Sijia Liu

**Abstract**: The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.



## **20. Multi-level Certified Defense Against Poisoning Attacks in Offline Reinforcement Learning**

cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20621v1) [paper-pdf](http://arxiv.org/pdf/2505.20621v1)

**Authors**: Shijie Liu, Andrew C. Cullen, Paul Montague, Sarah Erfani, Benjamin I. P. Rubinstein

**Abstract**: Similar to other machine learning frameworks, Offline Reinforcement Learning (RL) is shown to be vulnerable to poisoning attacks, due to its reliance on externally sourced datasets, a vulnerability that is exacerbated by its sequential nature. To mitigate the risks posed by RL poisoning, we extend certified defenses to provide larger guarantees against adversarial manipulation, ensuring robustness for both per-state actions, and the overall expected cumulative reward. Our approach leverages properties of Differential Privacy, in a manner that allows this work to span both continuous and discrete spaces, as well as stochastic and deterministic environments -- significantly expanding the scope and applicability of achievable guarantees. Empirical evaluations demonstrate that our approach ensures the performance drops to no more than $50\%$ with up to $7\%$ of the training data poisoned, significantly improving over the $0.008\%$ in prior work~\citep{wu_copa_2022}, while producing certified radii that is $5$ times larger as well. This highlights the potential of our framework to enhance safety and reliability in offline RL.



## **21. Detector noise in continuous-variable quantum key distribution**

quant-ph

7 pages, 6 figures

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.20441v1) [paper-pdf](http://arxiv.org/pdf/2505.20441v1)

**Authors**: Shihong Pan, Dimitri Monokandylos, Bing Qi

**Abstract**: Detector noise is a critical factor in determining the performance of a quantum key distribution (QKD) system. In continuous-variable (CV) QKD with optical coherent detection, the trusted detector noise model is widely used to enhance both the secret key rate and transmission distance. This model assumes that noise from the coherent detector is inherently random and cannot be accessed or manipulated by an adversary. Its validity rests on two key assumptions: (1) the detector can be accurately calibrated by the legitimate user and remains isolated from the adversary, and (2) the detector noise is truly random. So far, extensive research has focused on detector calibration and countermeasures against detector side-channel attacks. However, there is no strong evidence supporting assumption (2). In this paper, we analyze the electrical noise of a commercial balanced Photoreceiver, which has been applied in CV-QKD implementations, and demonstrate that assumption (2) is unjustified. To address this issue, we propose a "calibrated detector noise" model for CV-QKD, which relies solely on assumption (1). Numerical simulations comparing different noise models indicate that the new model can achieve a secret key rate comparable to the trusted-noise model, without depending on the questionable assumption of "truly random" detector noise.



## **22. Holes in Latent Space: Topological Signatures Under Adversarial Influence**

cs.LG

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.20435v1) [paper-pdf](http://arxiv.org/pdf/2505.20435v1)

**Authors**: Aideen Fay, Inés García-Redondo, Qiquan Wang, Haim Dubossarsky, Anthea Monod

**Abstract**: Understanding how adversarial conditions affect language models requires techniques that capture both global structure and local detail within high-dimensional activation spaces. We propose persistent homology (PH), a tool from topological data analysis, to systematically characterize multiscale latent space dynamics in LLMs under two distinct attack modes -- backdoor fine-tuning and indirect prompt injection. By analyzing six state-of-the-art LLMs, we show that adversarial conditions consistently compress latent topologies, reducing structural diversity at smaller scales while amplifying dominant features at coarser ones. These topological signatures are statistically robust across layers, architectures, model sizes, and align with the emergence of adversarial effects deeper in the network. To capture finer-grained mechanisms underlying these shifts, we introduce a neuron-level PH framework that quantifies how information flows and transforms within and across layers. Together, our findings demonstrate that PH offers a principled and unifying approach to interpreting representational dynamics in LLMs, particularly under distributional shift.



## **23. Eradicating the Unseen: Detecting, Exploiting, and Remediating a Path Traversal Vulnerability across GitHub**

cs.CR

17 pages, 11 figures

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.20186v1) [paper-pdf](http://arxiv.org/pdf/2505.20186v1)

**Authors**: Jafar Akhoundali, Hamidreza Hamidi, Kristian Rietveld, Olga Gadyatskaya

**Abstract**: Vulnerabilities in open-source software can cause cascading effects in the modern digital ecosystem. It is especially worrying if these vulnerabilities repeat across many projects, as once the adversaries find one of them, they can scale up the attack very easily. Unfortunately, since developers frequently reuse code from their own or external code resources, some nearly identical vulnerabilities exist across many open-source projects.   We conducted a study to examine the prevalence of a particular vulnerable code pattern that enables path traversal attacks (CWE-22) across open-source GitHub projects. To handle this study at the GitHub scale, we developed an automated pipeline that scans GitHub for the targeted vulnerable pattern, confirms the vulnerability by first running a static analysis and then exploiting the vulnerability in the context of the studied project, assesses its impact by calculating the CVSS score, generates a patch using GPT-4, and reports the vulnerability to the maintainers.   Using our pipeline, we identified 1,756 vulnerable open-source projects, some of which are very influential. For many of the affected projects, the vulnerability is critical (CVSS score higher than 9.0), as it can be exploited remotely without any privileges and critically impact the confidentiality and availability of the system. We have responsibly disclosed the vulnerability to the maintainers, and 14\% of the reported vulnerabilities have been remediated.   We also investigated the root causes of the vulnerable code pattern and assessed the side effects of the large number of copies of this vulnerable pattern that seem to have poisoned several popular LLMs. Our study highlights the urgent need to help secure the open-source ecosystem by leveraging scalable automated vulnerability management solutions and raising awareness among developers.



## **24. On the Robustness of RSMA to Adversarial BD-RIS-Induced Interference**

eess.SP

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.20146v1) [paper-pdf](http://arxiv.org/pdf/2505.20146v1)

**Authors**: Arthur S. de Sena, Jacek Kibilda, Nurul H. Mahmood, Andre Gomes, Luiz A. DaSilva, Matti Latva-aho

**Abstract**: This article investigates the robustness of rate-splitting multiple access (RSMA) in multi-user multiple-input multiple-output (MIMO) systems to interference attacks against channel acquisition induced by beyond-diagonal RISs (BD-RISs). Two primary attack strategies, random and aligned interference, are proposed for fully connected and group-connected BD-RIS architectures. Valid random reflection coefficients are generated exploiting the Takagi factorization, while potent aligned interference attacks are achieved through optimization strategies based on a quadratically constrained quadratic program (QCQP) reformulation followed by projections onto the unitary manifold. Our numerical findings reveal that, when perfect channel state information (CSI) is available, RSMA behaves similarly to space-division multiple access (SDMA) and thus is highly susceptible to the attack, with BD-RIS inducing severe performance loss and significantly outperforming diagonal RIS. However, under imperfect CSI, RSMA consistently demonstrates significantly greater robustness than SDMA, particularly as the system's transmit power increases.



## **25. PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks**

cs.CR

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.13862v3) [paper-pdf](http://arxiv.org/pdf/2505.13862v3)

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.



## **26. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2504.05050v3) [paper-pdf](http://arxiv.org/pdf/2504.05050v3)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.



## **27. Beyond Optimal Fault Tolerance**

cs.DC

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2501.06044v6) [paper-pdf](http://arxiv.org/pdf/2501.06044v6)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal "recoverable fault-tolerance" achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic "recovery procedure" that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.



## **28. Attention! You Vision Language Model Could Be Maliciously Manipulated**

cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19911v1) [paper-pdf](http://arxiv.org/pdf/2505.19911v1)

**Authors**: Xiaosen Wang, Shaokang Wang, Zhijin Ge, Yuyang Luo, Shudong Zhang

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable success in understanding complex real-world scenarios and supporting data-driven decision-making processes. However, VLMs exhibit significant vulnerability against adversarial examples, either text or image, which can lead to various adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In this work, we empirically and theoretically demonstrate that VLMs are particularly susceptible to image-based adversarial examples, where imperceptible perturbations can precisely manipulate each output token. To this end, we propose a novel attack called Vision-language model Manipulation Attack (VMA), which integrates first-order and second-order momentum optimization techniques with a differentiable transformation mechanism to effectively optimize the adversarial perturbation. Notably, VMA can be a double-edged sword: it can be leveraged to implement various attacks, such as jailbreaking, hijacking, privacy breaches, Denial-of-Service, and the generation of sponge examples, etc, while simultaneously enabling the injection of watermarks for copyright protection. Extensive empirical evaluations substantiate the efficacy and generalizability of VMA across diverse scenarios and datasets.



## **29. CPA-RAG:Covert Poisoning Attacks on Retrieval-Augmented Generation in Large Language Models**

cs.CR

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19864v1) [paper-pdf](http://arxiv.org/pdf/2505.19864v1)

**Authors**: Chunyang Li, Junwei Zhang, Anda Cheng, Zhuo Ma, Xinghua Li, Jianfeng Ma

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, but its openness introduces vulnerabilities that can be exploited by poisoning attacks. Existing poisoning methods for RAG systems have limitations, such as poor generalization and lack of fluency in adversarial texts. In this paper, we propose CPA-RAG, a black-box adversarial framework that generates query-relevant texts capable of manipulating the retrieval process to induce target answers. The proposed method integrates prompt-based text generation, cross-guided optimization through multiple LLMs, and retriever-based scoring to construct high-quality adversarial samples. We conduct extensive experiments across multiple datasets and LLMs to evaluate its effectiveness. Results show that the framework achieves over 90\% attack success when the top-k retrieval setting is 5, matching white-box performance, and maintains a consistent advantage of approximately 5 percentage points across different top-k values. It also outperforms existing black-box baselines by 14.5 percentage points under various defense strategies. Furthermore, our method successfully compromises a commercial RAG system deployed on Alibaba's BaiLian platform, demonstrating its practical threat in real-world applications. These findings underscore the need for more robust and secure RAG frameworks to defend against poisoning attacks.



## **30. Jailbreaking Prompt Attack: A Controllable Adversarial Attack against Diffusion Models**

cs.CR

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2404.02928v4) [paper-pdf](http://arxiv.org/pdf/2404.02928v4)

**Authors**: Jiachen Ma, Yijiang Li, Zhiqing Xiao, Anda Cao, Jie Zhang, Chao Ye, Junbo Zhao

**Abstract**: Text-to-image (T2I) models can be maliciously used to generate harmful content such as sexually explicit, unfaithful, and misleading or Not-Safe-for-Work (NSFW) images. Previous attacks largely depend on the availability of the diffusion model or involve a lengthy optimization process. In this work, we investigate a more practical and universal attack that does not require the presence of a target model and demonstrate that the high-dimensional text embedding space inherently contains NSFW concepts that can be exploited to generate harmful images. We present the Jailbreaking Prompt Attack (JPA). JPA first searches for the target malicious concepts in the text embedding space using a group of antonyms generated by ChatGPT. Subsequently, a prefix prompt is optimized in the discrete vocabulary space to align malicious concepts semantically in the text embedding space. We further introduce a soft assignment with gradient masking technique that allows us to perform gradient ascent in the discrete vocabulary space.   We perform extensive experiments with open-sourced T2I models, e.g. stable-diffusion-v1-4 and closed-sourced online services, e.g. DALLE2, Midjourney with black-box safety checkers. Results show that (1) JPA bypasses both text and image safety checkers (2) while preserving high semantic alignment with the target prompt. (3) JPA demonstrates a much faster speed than previous methods and can be executed in a fully automated manner. These merits render it a valuable tool for robustness evaluation in future text-to-image generation research.



## **31. One Surrogate to Fool Them All: Universal, Transferable, and Targeted Adversarial Attacks with CLIP**

cs.CR

21 pages, 15 figures, 18 tables To appear in the Proceedings of The  ACM Conference on Computer and Communications Security (CCS), 2025

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19840v1) [paper-pdf](http://arxiv.org/pdf/2505.19840v1)

**Authors**: Binyan Xu, Xilin Dai, Di Tang, Kehuan Zhang

**Abstract**: Deep Neural Networks (DNNs) have achieved widespread success yet remain prone to adversarial attacks. Typically, such attacks either involve frequent queries to the target model or rely on surrogate models closely mirroring the target model -- often trained with subsets of the target model's training data -- to achieve high attack success rates through transferability. However, in realistic scenarios where training data is inaccessible and excessive queries can raise alarms, crafting adversarial examples becomes more challenging. In this paper, we present UnivIntruder, a novel attack framework that relies solely on a single, publicly available CLIP model and publicly available datasets. By using textual concepts, UnivIntruder generates universal, transferable, and targeted adversarial perturbations that mislead DNNs into misclassifying inputs into adversary-specified classes defined by textual concepts.   Our extensive experiments show that our approach achieves an Attack Success Rate (ASR) of up to 85% on ImageNet and over 99% on CIFAR-10, significantly outperforming existing transfer-based methods. Additionally, we reveal real-world vulnerabilities, showing that even without querying target models, UnivIntruder compromises image search engines like Google and Baidu with ASR rates up to 84%, and vision language models like GPT-4 and Claude-3.5 with ASR rates up to 80%. These findings underscore the practicality of our attack in scenarios where traditional avenues are blocked, highlighting the need to reevaluate security paradigms in AI applications.



## **32. TESSER: Transfer-Enhancing Adversarial Attacks from Vision Transformers via Spectral and Semantic Regularization**

cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19613v1) [paper-pdf](http://arxiv.org/pdf/2505.19613v1)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Adversarial transferability remains a critical challenge in evaluating the robustness of deep neural networks. In security-critical applications, transferability enables black-box attacks without access to model internals, making it a key concern for real-world adversarial threat assessment. While Vision Transformers (ViTs) have demonstrated strong adversarial performance, existing attacks often fail to transfer effectively across architectures, especially from ViTs to Convolutional Neural Networks (CNNs) or hybrid models. In this paper, we introduce \textbf{TESSER} -- a novel adversarial attack framework that enhances transferability via two key strategies: (1) \textit{Feature-Sensitive Gradient Scaling (FSGS)}, which modulates gradients based on token-wise importance derived from intermediate feature activations, and (2) \textit{Spectral Smoothness Regularization (SSR)}, which suppresses high-frequency noise in perturbations using a differentiable Gaussian prior. These components work in tandem to generate perturbations that are both semantically meaningful and spectrally smooth. Extensive experiments on ImageNet across 12 diverse architectures demonstrate that TESSER achieves +10.9\% higher attack succes rate (ASR) on CNNs and +7.2\% on ViTs compared to the state-of-the-art Adaptive Token Tuning (ATT) method. Moreover, TESSER significantly improves robustness against defended models, achieving 53.55\% ASR on adversarially trained CNNs. Qualitative analysis shows strong alignment between TESSER's perturbations and salient visual regions identified via Grad-CAM, while frequency-domain analysis reveals a 12\% reduction in high-frequency energy, confirming the effectiveness of spectral regularization.



## **33. JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models**

cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19610v1) [paper-pdf](http://arxiv.org/pdf/2505.19610v1)

**Authors**: Jiaxin Song, Yixu Wang, Jie Li, Rui Yu, Yan Teng, Xingjun Ma, Yingchun Wang

**Abstract**: Vision-Language Models (VLMs) exhibit impressive performance, yet the integration of powerful vision encoders has significantly broadened their attack surface, rendering them increasingly susceptible to jailbreak attacks. However, lacking well-defined attack objectives, existing jailbreak methods often struggle with gradient-based strategies prone to local optima and lacking precise directional guidance, and typically decouple visual and textual modalities, thereby limiting their effectiveness by neglecting crucial cross-modal interactions. Inspired by the Eliciting Latent Knowledge (ELK) framework, we posit that VLMs encode safety-relevant information within their internal fusion-layer representations, revealing an implicit safety decision boundary in the latent space. This motivates exploiting boundary to steer model behavior. Accordingly, we propose JailBound, a novel latent space jailbreak framework comprising two stages: (1) Safety Boundary Probing, which addresses the guidance issue by approximating decision boundary within fusion layer's latent space, thereby identifying optimal perturbation directions towards the target region; and (2) Safety Boundary Crossing, which overcomes the limitations of decoupled approaches by jointly optimizing adversarial perturbations across both image and text inputs. This latter stage employs an innovative mechanism to steer the model's internal state towards policy-violating outputs while maintaining cross-modal semantic consistency. Extensive experiments on six diverse VLMs demonstrate JailBound's efficacy, achieves 94.32% white-box and 67.28% black-box attack success averagely, which are 6.17% and 21.13% higher than SOTA methods, respectively. Our findings expose a overlooked safety risk in VLMs and highlight the urgent need for more robust defenses. Warning: This paper contains potentially sensitive, harmful and offensive content.



## **34. Authenticated Sublinear Quantum Private Information Retrieval**

quant-ph

11 pages, 1 figure

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2504.04041v2) [paper-pdf](http://arxiv.org/pdf/2504.04041v2)

**Authors**: Fengxia Liu, Zhiyong Zheng, Kun Tian, Yi Zhang, Heng Guo, Zhe Hu, Oleksiy Zhedanov, Zixian Gong

**Abstract**: This paper introduces a novel lower bound on communication complexity using quantum relative entropy and mutual information, refining previous classical entropy-based results. By leveraging Uhlmann's lemma and quantum Pinsker inequalities, the authors establish tighter bounds for information-theoretic security, demonstrating that quantum protocols inherently outperform classical counterparts in balancing privacy and efficiency. Also explores symmetric Quantum Private Information Retrieval (QPIR) protocols that achieve sub-linear communication complexity while ensuring robustness against specious adversaries: A post-quantum cryptography based protocol that can be authenticated for the specious server; A ring-LWE-based protocol for post-quantum security in a single-server setting, ensuring robustness against quantum attacks; A multi-server protocol optimized for hardware practicality, reducing implementation overhead while maintaining sub-linear efficiency. These protocols address critical gaps in secure database queries, offering exponential communication improvements over classical linear-complexity methods. The work also analyzes security trade-offs under quantum specious adversaries, providing theoretical guarantees for privacy and correctness.



## **35. RDI: An adversarial robustness evaluation metric for deep neural networks based on model statistical features**

cs.LG

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2504.18556v2) [paper-pdf](http://arxiv.org/pdf/2504.18556v2)

**Authors**: Jialei Song, Xingquan Zuo, Feiyang Wang, Hai Huang, Tianle Zhang

**Abstract**: Deep neural networks (DNNs) are highly susceptible to adversarial samples, raising concerns about their reliability in safety-critical tasks. Currently, methods of evaluating adversarial robustness are primarily categorized into attack-based and certified robustness evaluation approaches. The former not only relies on specific attack algorithms but also is highly time-consuming, while the latter due to its analytical nature, is typically difficult to implement for large and complex models. A few studies evaluate model robustness based on the model's decision boundary, but they suffer from low evaluation accuracy. To address the aforementioned issues, we propose a novel adversarial robustness evaluation metric, Robustness Difference Index (RDI), which is based on model statistical features. RDI draws inspiration from clustering evaluation by analyzing the intra-class and inter-class distances of feature vectors separated by the decision boundary to quantify model robustness. It is attack-independent and has high computational efficiency. Experiments show that, RDI demonstrates a stronger correlation with the gold-standard adversarial robustness metric of attack success rate (ASR). The average computation time of RDI is only 1/30 of the evaluation method based on the PGD attack. Our open-source code is available at: https://github.com/BUPTAIOC/RDI.



## **36. One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs**

cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2503.04856v2) [paper-pdf](http://arxiv.org/pdf/2503.04856v2)

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim

**Abstract**: We introduce a novel framework for consolidating multi-turn adversarial ``jailbreak'' prompts into single-turn queries, significantly reducing the manual overhead required for adversarial testing of large language models (LLMs). While multi-turn human jailbreaks have been shown to yield high attack success rates, they demand considerable human effort and time. Our multi-turn-to-single-turn (M2S) methods -- Hyphenize, Numberize, and Pythonize -- systematically reformat multi-turn dialogues into structured single-turn prompts. Despite removing iterative back-and-forth interactions, these prompts preserve and often enhance adversarial potency: in extensive evaluations on the Multi-turn Human Jailbreak (MHJ) dataset, M2S methods achieve attack success rates from 70.6 percent to 95.9 percent across several state-of-the-art LLMs. Remarkably, the single-turn prompts outperform the original multi-turn attacks by as much as 17.5 percentage points while cutting token usage by more than half on average. Further analysis shows that embedding malicious requests in enumerated or code-like structures exploits ``contextual blindness'', bypassing both native guardrails and external input-output filters. By converting multi-turn conversations into concise single-turn prompts, the M2S framework provides a scalable tool for large-scale red teaming and reveals critical weaknesses in contemporary LLM defenses.



## **37. Structure Disruption: Subverting Malicious Diffusion-Based Inpainting via Self-Attention Query Perturbation**

cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19425v1) [paper-pdf](http://arxiv.org/pdf/2505.19425v1)

**Authors**: Yuhao He, Jinyu Tian, Haiwei Wu, Jianqing Li

**Abstract**: The rapid advancement of diffusion models has enhanced their image inpainting and editing capabilities but also introduced significant societal risks. Adversaries can exploit user images from social media to generate misleading or harmful content. While adversarial perturbations can disrupt inpainting, global perturbation-based methods fail in mask-guided editing tasks due to spatial constraints. To address these challenges, we propose Structure Disruption Attack (SDA), a powerful protection framework for safeguarding sensitive image regions against inpainting-based editing. Building upon the contour-focused nature of self-attention mechanisms of diffusion models, SDA optimizes perturbations by disrupting queries in self-attention during the initial denoising step to destroy the contour generation process. This targeted interference directly disrupts the structural generation capability of diffusion models, effectively preventing them from producing coherent images. We validate our motivation through visualization techniques and extensive experiments on public datasets, demonstrating that SDA achieves state-of-the-art (SOTA) protection performance while maintaining strong robustness.



## **38. Are Time-Series Foundation Models Deployment-Ready? A Systematic Study of Adversarial Robustness Across Domains**

cs.LG

Preprint

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19397v1) [paper-pdf](http://arxiv.org/pdf/2505.19397v1)

**Authors**: Jiawen Zhang, Zhenwei Zhang, Shun Zheng, Xumeng Wen, Jia Li, Jiang Bian

**Abstract**: Time Series Foundation Models (TSFMs), which are pretrained on large-scale, cross-domain data and capable of zero-shot forecasting in new scenarios without further training, are increasingly adopted in real-world applications. However, as the zero-shot forecasting paradigm gets popular, a critical yet overlooked question emerges: Are TSFMs robust to adversarial input perturbations? Such perturbations could be exploited in man-in-the-middle attacks or data poisoning. To address this gap, we conduct a systematic investigation into the adversarial robustness of TSFMs. Our results show that even minimal perturbations can induce significant and controllable changes in forecast behaviors, including trend reversal, temporal drift, and amplitude shift, posing serious risks to TSFM-based services. Through experiments on representative TSFMs and multiple datasets, we reveal their consistent vulnerabilities and identify potential architectural designs, such as structural sparsity and multi-task pretraining, that may improve robustness. Our findings offer actionable guidance for designing more resilient forecasting systems and provide a critical assessment of the adversarial robustness of TSFMs.



## **39. RADEP: A Resilient Adaptive Defense Framework Against Model Extraction Attacks**

cs.CR

Presented at the IEEE International Wireless Communications and  Mobile Computing Conference (IWCMC) 2025

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.19364v1) [paper-pdf](http://arxiv.org/pdf/2505.19364v1)

**Authors**: Amit Chakraborty, Sayyed Farid Ahamed, Sandip Roy, Soumya Banerjee, Kevin Choi, Abdul Rahman, Alison Hu, Edward Bowen, Sachin Shetty

**Abstract**: Machine Learning as a Service (MLaaS) enables users to leverage powerful machine learning models through cloud-based APIs, offering scalability and ease of deployment. However, these services are vulnerable to model extraction attacks, where adversaries repeatedly query the application programming interface (API) to reconstruct a functionally similar model, compromising intellectual property and security. Despite various defense strategies being proposed, many suffer from high computational costs, limited adaptability to evolving attack techniques, and a reduction in performance for legitimate users. In this paper, we introduce a Resilient Adaptive Defense Framework for Model Extraction Attack Protection (RADEP), a multifaceted defense framework designed to counteract model extraction attacks through a multi-layered security approach. RADEP employs progressive adversarial training to enhance model resilience against extraction attempts. Malicious query detection is achieved through a combination of uncertainty quantification and behavioral pattern analysis, effectively identifying adversarial queries. Furthermore, we develop an adaptive response mechanism that dynamically modifies query outputs based on their suspicion scores, reducing the utility of stolen models. Finally, ownership verification is enforced through embedded watermarking and backdoor triggers, enabling reliable identification of unauthorized model use. Experimental evaluations demonstrate that RADEP significantly reduces extraction success rates while maintaining high detection accuracy with minimal impact on legitimate queries. Extensive experiments show that RADEP effectively defends against model extraction attacks and remains resilient even against adaptive adversaries, making it a reliable security framework for MLaaS models.



## **40. Curvature Dynamic Black-box Attack: revisiting adversarial robustness via dynamic curvature estimation**

cs.LG

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.19194v1) [paper-pdf](http://arxiv.org/pdf/2505.19194v1)

**Authors**: Peiran Sun

**Abstract**: Adversarial attack reveals the vulnerability of deep learning models. For about a decade, countless attack and defense methods have been proposed, leading to robustified classifiers and better understanding of models. Among these methods, curvature-based approaches have attracted attention because it is assumed that high curvature may give rise to rough decision boundary. However, the most commonly used \textit{curvature} is the curvature of loss function, scores or other parameters from within the model as opposed to decision boundary curvature, since the former can be relatively easily formed using second order derivative. In this paper, we propose a new query-efficient method, dynamic curvature estimation(DCE), to estimate the decision boundary curvature in a black-box setting. Our approach is based on CGBA, a black-box adversarial attack. By performing DCE on a wide range of classifiers, we discovered, statistically, a connection between decision boundary curvature and adversarial robustness. We also propose a new attack method, curvature dynamic black-box attack(CDBA) with improved performance using the dynamically estimated curvature.



## **41. Latent-space adversarial training with post-aware calibration for defending large language models against jailbreak attacks**

cs.CR

Under Review

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2501.10639v2) [paper-pdf](http://arxiv.org/pdf/2501.10639v2)

**Authors**: Xin Yi, Yue Li, dongsheng Shi, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Ensuring safety alignment has become a critical requirement for large language models (LLMs), particularly given their widespread deployment in real-world applications. However, LLMs remain susceptible to jailbreak attacks, which exploit system vulnerabilities to bypass safety measures and generate harmful outputs. Although numerous defense mechanisms based on adversarial training have been proposed, a persistent challenge lies in the exacerbation of over-refusal behaviors, which compromise the overall utility of the model. To address these challenges, we propose a Latent-space Adversarial Training with Post-aware Calibration (LATPC) framework. During the adversarial training phase, LATPC compares harmful and harmless instructions in the latent space and extracts safety-critical dimensions to construct refusal features attack, precisely simulating agnostic jailbreak attack types requiring adversarial mitigation. At the inference stage, an embedding-level calibration mechanism is employed to alleviate over-refusal behaviors with minimal computational overhead. Experimental results demonstrate that, compared to various defense methods across five types of jailbreak attacks, LATPC framework achieves a superior balance between safety and utility. Moreover, our analysis underscores the effectiveness of extracting safety-critical dimensions from the latent space for constructing robust refusal feature attacks.



## **42. PII-Scope: A Comprehensive Study on Training Data PII Extraction Attacks in LLMs**

cs.CL

Additional results with Pythia6.9B; Additional results with Phone  number PII;

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2410.06704v2) [paper-pdf](http://arxiv.org/pdf/2410.06704v2)

**Authors**: Krishna Kanth Nakka, Ahmed Frikha, Ricardo Mendes, Xue Jiang, Xuebing Zhou

**Abstract**: In this work, we introduce PII-Scope, a comprehensive benchmark designed to evaluate state-of-the-art methodologies for PII extraction attacks targeting LLMs across diverse threat settings. Our study provides a deeper understanding of these attacks by uncovering several hyperparameters (e.g., demonstration selection) crucial to their effectiveness. Building on this understanding, we extend our study to more realistic attack scenarios, exploring PII attacks that employ advanced adversarial strategies, including repeated and diverse querying, and leveraging iterative learning for continual PII extraction. Through extensive experimentation, our results reveal a notable underestimation of PII leakage in existing single-query attacks. In fact, we show that with sophisticated adversarial capabilities and a limited query budget, PII extraction rates can increase by up to fivefold when targeting the pretrained model. Moreover, we evaluate PII leakage on finetuned models, showing that they are more vulnerable to leakage than pretrained models. Overall, our work establishes a rigorous empirical benchmark for PII extraction attacks in realistic threat scenarios and provides a strong foundation for developing effective mitigation strategies.



## **43. Understanding the Robustness of Graph Neural Networks against Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2406.13920v2) [paper-pdf](http://arxiv.org/pdf/2406.13920v2)

**Authors**: Tao Wu, Canyixing Cui, Xingping Xian, Shaojie Qiao, Chao Wang, Lin Yuan, Shui Yu

**Abstract**: Recent studies have shown that graph neural networks (GNNs) are vulnerable to adversarial attacks, posing significant challenges to their deployment in safety-critical scenarios. This vulnerability has spurred a growing focus on designing robust GNNs. Despite this interest, current advancements have predominantly relied on empirical trial and error, resulting in a limited understanding of the robustness of GNNs against adversarial attacks. To address this issue, we conduct the first large-scale systematic study on the adversarial robustness of GNNs by considering the patterns of input graphs, the architecture of GNNs, and their model capacity, along with discussions on sensitive neurons and adversarial transferability. This work proposes a comprehensive empirical framework for analyzing the adversarial robustness of GNNs. To support the analysis of adversarial robustness in GNNs, we introduce two evaluation metrics: the confidence-based decision surface and the accuracy-based adversarial transferability rate. Through experimental analysis, we derive 11 actionable guidelines for designing robust GNNs, enabling model developers to gain deeper insights. The code of this study is available at https://github.com/star4455/GraphRE.



## **44. A quantitative notion of economic security for smart contract compositions**

cs.CR

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.19006v1) [paper-pdf](http://arxiv.org/pdf/2505.19006v1)

**Authors**: Emily Priyadarshini, Massimo Bartoletti

**Abstract**: Decentralized applications are often composed of multiple interconnected smart contracts. This is especially evident in DeFi, where protocols are heavily intertwined and rely on a variety of basic building blocks such as tokens, decentralized exchanges and lending protocols. A crucial security challenge in this setting arises when adversaries target individual components to cause systemic economic losses. Existing security notions focus on determining the existence of these attacks, but fail to quantify the effect of manipulating individual components on the overall economic security of the system. In this paper, we introduce a quantitative security notion that measures how an attack on a single component can amplify economic losses of the overall system. We study the fundamental properties of this notion and apply it to assess the security of key compositions. In particular, we analyse under-collateralized loan attacks in systems made of lending protocols and decentralized exchanges.



## **45. A theoretical basis for MEV**

cs.CR

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2302.02154v5) [paper-pdf](http://arxiv.org/pdf/2302.02154v5)

**Authors**: Massimo Bartoletti, Roberto Zunino

**Abstract**: Maximal Extractable Value (MEV) refers to a wide class of economic attacks to public blockchains, where adversaries with the power to reorder, drop or insert transactions in a block can "extract" value from smart contracts. Empirical research has shown that mainstream DeFi protocols are massively targeted by these attacks, with detrimental effects on their users and on the blockchain network. Despite the increasing real-world impact of these attacks, their theoretical foundations remain insufficiently established. We propose a formal theory of MEV, based on a general, abstract model of blockchains and smart contracts. Our theory is the basis for proofs of security against MEV attacks.



## **46. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

cs.SE

24 pages, 5 tables, 9 figures, Camera-ready version accepted to FSE  2025

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2410.22663v4) [paper-pdf](http://arxiv.org/pdf/2410.22663v4)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Studies indicate that conventional metrics are insufficient to build human trust in ML models. These models often learn spurious correlations and predict based on them. In the real world, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable based on valid patterns in the data. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods. However, this is time-consuming, error-prone, and unscalable.   We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers. TOKI automatically checks whether the words contributing the most to a prediction are semantically related to the predicted class. Specifically, we leverage ML explanations to extract the decision-contributing words and measure their semantic relatedness with the class based on word embeddings. We also introduce a novel adversarial attack method that targets trustworthiness vulnerabilities identified by TOKI. To evaluate their alignment with human judgement, experiments are conducted. We compare TOKI with a naive baseline based solely on model confidence and TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot effectively distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided attack method is more effective with fewer perturbations than A2T.



## **47. GhostPrompt: Jailbreaking Text-to-image Generative Models based on Dynamic Optimization**

cs.LG

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.18979v1) [paper-pdf](http://arxiv.org/pdf/2505.18979v1)

**Authors**: Zixuan Chen, Hao Lin, Ke Xu, Xinghao Jiang, Tanfeng Sun

**Abstract**: Text-to-image (T2I) generation models can inadvertently produce not-safe-for-work (NSFW) content, prompting the integration of text and image safety filters. Recent advances employ large language models (LLMs) for semantic-level detection, rendering traditional token-level perturbation attacks largely ineffective. However, our evaluation shows that existing jailbreak methods are ineffective against these modern filters. We introduce GhostPrompt, the first automated jailbreak framework that combines dynamic prompt optimization with multimodal feedback. It consists of two key components: (i) Dynamic Optimization, an iterative process that guides a large language model (LLM) using feedback from text safety filters and CLIP similarity scores to generate semantically aligned adversarial prompts; and (ii) Adaptive Safety Indicator Injection, which formulates the injection of benign visual cues as a reinforcement learning problem to bypass image-level filters. GhostPrompt achieves state-of-the-art performance, increasing the ShieldLM-7B bypass rate from 12.5\% (Sneakyprompt) to 99.0\%, improving CLIP score from 0.2637 to 0.2762, and reducing the time cost by $4.2 \times$. Moreover, it generalizes to unseen filters including GPT-4.1 and successfully jailbreaks DALLE 3 to generate NSFW images in our evaluation, revealing systemic vulnerabilities in current multimodal defenses. To support further research on AI safety and red-teaming, we will release code and adversarial prompts under a controlled-access protocol.



## **48. Pre-trained Encoder Inference: Revealing Upstream Encoders In Downstream Machine Learning Services**

cs.LG

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2408.02814v2) [paper-pdf](http://arxiv.org/pdf/2408.02814v2)

**Authors**: Shaopeng Fu, Xuexue Sun, Ke Qing, Tianhang Zheng, Di Wang

**Abstract**: Pre-trained encoders available online have been widely adopted to build downstream machine learning (ML) services, but various attacks against these encoders also post security and privacy threats toward such a downstream ML service paradigm. We unveil a new vulnerability: the Pre-trained Encoder Inference (PEI) attack, which can extract sensitive encoder information from a targeted downstream ML service that can then be used to promote other ML attacks against the targeted service. By only providing API accesses to a targeted downstream service and a set of candidate encoders, the PEI attack can successfully infer which encoder is secretly used by the targeted service based on candidate ones. Compared with existing encoder attacks, which mainly target encoders on the upstream side, the PEI attack can compromise encoders even after they have been deployed and hidden in downstream ML services, which makes it a more realistic threat. We empirically verify the effectiveness of the PEI attack on vision encoders. we first conduct PEI attacks against two downstream services (i.e., image classification and multimodal generation), and then show how PEI attacks can facilitate other ML attacks (i.e., model stealing attacks vs. image classification models and adversarial attacks vs. multimodal generative models). Our results call for new security and privacy considerations when deploying encoders in downstream services. The code is available at https://github.com/fshp971/encoder-inference.



## **49. Security Concerns for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18889v1) [paper-pdf](http://arxiv.org/pdf/2505.18889v1)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as GPT-4 (and its recent iterations like GPT-4o and the GPT-4.1 series), Google's Gemini, Anthropic's Claude 3 models, and xAI's Grok have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. In this survey, we provide a comprehensive overview of the emerging security concerns around LLMs, categorizing threats into prompt injection and jailbreaking, adversarial attacks (including input perturbations and data poisoning), misuse by malicious actors (e.g., for disinformation, phishing, and malware generation), and worrisome risks inherent in autonomous LLM agents. A significant focus has been recently placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives (scheming), which may even persist through safety training. We summarize recent academic and industrial studies (2022-2025) that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.



## **50. Audio Jailbreak Attacks: Exposing Vulnerabilities in SpeechGPT in a White-Box Framework**

cs.CL

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18864v1) [paper-pdf](http://arxiv.org/pdf/2505.18864v1)

**Authors**: Binhao Ma, Hanqing Guo, Zhengping Jay Luo, Rui Duan

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have significantly enhanced the naturalness and flexibility of human computer interaction by enabling seamless understanding across text, vision, and audio modalities. Among these, voice enabled models such as SpeechGPT have demonstrated considerable improvements in usability, offering expressive, and emotionally responsive interactions that foster deeper connections in real world communication scenarios. However, the use of voice introduces new security risks, as attackers can exploit the unique characteristics of spoken language, such as timing, pronunciation variability, and speech to text translation, to craft inputs that bypass defenses in ways not seen in text-based systems. Despite substantial research on text based jailbreaks, the voice modality remains largely underexplored in terms of both attack strategies and defense mechanisms. In this work, we present an adversarial attack targeting the speech input of aligned MLLMs in a white box scenario. Specifically, we introduce a novel token level attack that leverages access to the model's speech tokenization to generate adversarial token sequences. These sequences are then synthesized into audio prompts, which effectively bypass alignment safeguards and to induce prohibited outputs. Evaluated on SpeechGPT, our approach achieves up to 89 percent attack success rate across multiple restricted tasks, significantly outperforming existing voice based jailbreak methods. Our findings shed light on the vulnerabilities of voice-enabled multimodal systems and to help guide the development of more robust next-generation MLLMs.



