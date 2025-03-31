# Latest Adversarial Attack Papers
**update at 2025-03-31 15:47:47**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. CoRPA: Adversarial Image Generation for Chest X-rays Using Concept Vector Perturbations and Generative Models**

eess.IV

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2502.05214v2) [paper-pdf](http://arxiv.org/pdf/2502.05214v2)

**Authors**: Amy Rafferty, Rishi Ramaesh, Ajitha Rajan

**Abstract**: Deep learning models for medical image classification tasks are becoming widely implemented in AI-assisted diagnostic tools, aiming to enhance diagnostic accuracy, reduce clinician workloads, and improve patient outcomes. However, their vulnerability to adversarial attacks poses significant risks to patient safety. Current attack methodologies use general techniques such as model querying or pixel value perturbations to generate adversarial examples designed to fool a model. These approaches may not adequately address the unique characteristics of clinical errors stemming from missed or incorrectly identified clinical features. We propose the Concept-based Report Perturbation Attack (CoRPA), a clinically-focused black-box adversarial attack framework tailored to the medical imaging domain. CoRPA leverages clinical concepts to generate adversarial radiological reports and images that closely mirror realistic clinical misdiagnosis scenarios. We demonstrate the utility of CoRPA using the MIMIC-CXR-JPG dataset of chest X-rays and radiological reports. Our evaluation reveals that deep learning models exhibiting strong resilience to conventional adversarial attacks are significantly less robust when subjected to CoRPA's clinically-focused perturbations. This underscores the importance of addressing domain-specific vulnerabilities in medical AI systems. By introducing a specialized adversarial attack framework, this study provides a foundation for developing robust, real-world-ready AI models in healthcare, ensuring their safe and reliable deployment in high-stakes clinical environments.



## **2. Leveraging Expert Input for Robust and Explainable AI-Assisted Lung Cancer Detection in Chest X-rays**

cs.LG

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2403.19444v2) [paper-pdf](http://arxiv.org/pdf/2403.19444v2)

**Authors**: Amy Rafferty, Rishi Ramaesh, Ajitha Rajan

**Abstract**: Deep learning models show significant potential for advancing AI-assisted medical diagnostics, particularly in detecting lung cancer through medical image modalities such as chest X-rays. However, the black-box nature of these models poses challenges to their interpretability and trustworthiness, limiting their adoption in clinical practice. This study examines both the interpretability and robustness of a high-performing lung cancer detection model based on InceptionV3, utilizing a public dataset of chest X-rays and radiological reports. We evaluate the clinical utility of multiple explainable AI (XAI) techniques, including both post-hoc and ante-hoc approaches, and find that existing methods often fail to provide clinically relevant explanations, displaying inconsistencies and divergence from expert radiologist assessments. To address these limitations, we collaborated with a radiologist to define diagnosis-specific clinical concepts and developed ClinicXAI, an expert-driven approach leveraging the concept bottleneck methodology. ClinicXAI generated clinically meaningful explanations which closely aligned with the practical requirements of clinicians while maintaining high diagnostic accuracy. We also assess the robustness of ClinicXAI in comparison to the original InceptionV3 model by subjecting both to a series of widely utilized adversarial attacks. Our analysis demonstrates that ClinicXAI exhibits significantly greater resilience to adversarial perturbations. These findings underscore the importance of incorporating domain expertise into the design of interpretable and robust AI systems for medical diagnostics, paving the way for more trustworthy and effective AI solutions in healthcare.



## **3. Privacy-Preserving Secure Neighbor Discovery for Wireless Networks**

cs.CR

10 pages, 6 figures. Author's version; accepted and presented at the  IEEE 23rd International Conference on Trust, Security and Privacy in  Computing and Communications (TrustCom) 2024

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2503.22232v1) [paper-pdf](http://arxiv.org/pdf/2503.22232v1)

**Authors**: Ahmed Mohamed Hussain, Panos Papadimitratos

**Abstract**: Traditional Neighbor Discovery (ND) and Secure Neighbor Discovery (SND) are key elements for network functionality. SND is a hard problem, satisfying not only typical security properties (authentication, integrity) but also verification of direct communication, which involves distance estimation based on time measurements and device coordinates. Defeating relay attacks, also known as "wormholes", leading to stealthy Byzantine links and significant degradation of communication and adversarial control, is key in many wireless networked systems. However, SND is not concerned with privacy; it necessitates revealing the identity and location of the device(s) participating in the protocol execution. This can be a deterrent for deployment, especially involving user-held devices in the emerging Internet of Things (IoT) enabled smart environments. To address this challenge, we present a novel Privacy-Preserving Secure Neighbor Discovery (PP-SND) protocol, enabling devices to perform SND without revealing their actual identities and locations, effectively decoupling discovery from the exposure of sensitive information. We use Homomorphic Encryption (HE) for computing device distances without revealing their actual coordinates, as well as employing a pseudonymous device authentication to hide identities while preserving communication integrity. PP-SND provides SND [1] along with pseudonymity, confidentiality, and unlinkability. Our presentation here is not specific to one wireless technology, and we assess the performance of the protocols (cryptographic overhead) on a Raspberry Pi 4 and provide a security and privacy analysis.



## **4. Data-Free Universal Attack by Exploiting the Intrinsic Vulnerability of Deep Models**

cs.LG

Accepted in AAAI 2025

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2503.22205v1) [paper-pdf](http://arxiv.org/pdf/2503.22205v1)

**Authors**: YangTian Yan, Jinyu Tian

**Abstract**: Deep neural networks (DNNs) are susceptible to Universal Adversarial Perturbations (UAPs), which are instance agnostic perturbations that can deceive a target model across a wide range of samples. Unlike instance-specific adversarial examples, UAPs present a greater challenge as they must generalize across different samples and models. Generating UAPs typically requires access to numerous examples, which is a strong assumption in real-world tasks. In this paper, we propose a novel data-free method called Intrinsic UAP (IntriUAP), by exploiting the intrinsic vulnerabilities of deep models. We analyze a series of popular deep models composed of linear and nonlinear layers with a Lipschitz constant of 1, revealing that the vulnerability of these models is predominantly influenced by their linear components. Based on this observation, we leverage the ill-conditioned nature of the linear components by aligning the UAP with the right singular vectors corresponding to the maximum singular value of each linear layer. Remarkably, our method achieves highly competitive performance in attacking popular image classification deep models without using any image samples. We also evaluate the black-box attack performance of our method, showing that it matches the state-of-the-art baseline for data-free methods on models that conform to our theoretical framework. Beyond the data-free assumption, IntriUAP also operates under a weaker assumption, where the adversary only can access a few of the victim model's layers. Experiments demonstrate that the attack success rate decreases by only 4% when the adversary has access to just 50% of the linear layers in the victim model.



## **5. AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models**

cs.LG

CVPR 2025

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2410.05346v3) [paper-pdf](http://arxiv.org/pdf/2410.05346v3)

**Authors**: Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Yunhao Chen, Jitao Sang, Dit-Yan Yeung

**Abstract**: Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks. Traditional targeted adversarial attacks require specific targets and labels, limiting their real-world impact.We present AnyAttack, a self-supervised framework that transcends the limitations of conventional attacks through a novel foundation model approach. By pre-training on the massive LAION-400M dataset without label supervision, AnyAttack achieves unprecedented flexibility - enabling any image to be transformed into an attack vector targeting any desired output across different VLMs.This approach fundamentally changes the threat landscape, making adversarial capabilities accessible at an unprecedented scale. Our extensive validation across five open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) demonstrates AnyAttack's effectiveness across diverse multimodal tasks. Most concerning, AnyAttack seamlessly transfers to commercial systems including Google Gemini, Claude Sonnet, Microsoft Copilot and OpenAI GPT, revealing a systemic vulnerability requiring immediate attention.



## **6. Foot-In-The-Door: A Multi-turn Jailbreak for LLMs**

cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2502.19820v3) [paper-pdf](http://arxiv.org/pdf/2502.19820v3)

**Authors**: Zixuan Weng, Xiaolong Jin, Jinyuan Jia, Xiangyu Zhang

**Abstract**: Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.



## **7. Learning to Lie: Reinforcement Learning Attacks Damage Human-AI Teams and Teams of LLMs**

cs.HC

17 pages, 9 figures, accepted to ICLR 2025 Workshop on Human-AI  Coevolution

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21983v1) [paper-pdf](http://arxiv.org/pdf/2503.21983v1)

**Authors**: Abed Kareem Musaffar, Anand Gokhale, Sirui Zeng, Rasta Tadayon, Xifeng Yan, Ambuj Singh, Francesco Bullo

**Abstract**: As artificial intelligence (AI) assistants become more widely adopted in safety-critical domains, it becomes important to develop safeguards against potential failures or adversarial attacks. A key prerequisite to developing these safeguards is understanding the ability of these AI assistants to mislead human teammates. We investigate this attack problem within the context of an intellective strategy game where a team of three humans and one AI assistant collaborate to answer a series of trivia questions. Unbeknownst to the humans, the AI assistant is adversarial. Leveraging techniques from Model-Based Reinforcement Learning (MBRL), the AI assistant learns a model of the humans' trust evolution and uses that model to manipulate the group decision-making process to harm the team. We evaluate two models -- one inspired by literature and the other data-driven -- and find that both can effectively harm the human team. Moreover, we find that in this setting our data-driven model is capable of accurately predicting how human agents appraise their teammates given limited information on prior interactions. Finally, we compare the performance of state-of-the-art LLM models to human agents on our influence allocation task to evaluate whether the LLMs allocate influence similarly to humans or if they are more robust to our attack. These results enhance our understanding of decision-making dynamics in small human-AI teams and lay the foundation for defense strategies.



## **8. Harnessing Chain-of-Thought Metadata for Task Routing and Adversarial Prompt Detection**

cs.CL

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21464v1) [paper-pdf](http://arxiv.org/pdf/2503.21464v1)

**Authors**: Ryan Marinelli, Josef Pichlmeier, Tamas Bisztray

**Abstract**: In this work, we propose a metric called Number of Thoughts (NofT) to determine the difficulty of tasks pre-prompting and support Large Language Models (LLMs) in production contexts. By setting thresholds based on the number of thoughts, this metric can discern the difficulty of prompts and support more effective prompt routing. A 2% decrease in latency is achieved when routing prompts from the MathInstruct dataset through quantized, distilled versions of Deepseek with 1.7 billion, 7 billion, and 14 billion parameters. Moreover, this metric can be used to detect adversarial prompts used in prompt injection attacks with high efficacy. The Number of Thoughts can inform a classifier that achieves 95% accuracy in adversarial prompt detection. Our experiments ad datasets used are available on our GitHub page: https://github.com/rymarinelli/Number_Of_Thoughts/tree/main.



## **9. Tricking Retrievers with Influential Tokens: An Efficient Black-Box Corpus Poisoning Attack**

cs.LG

Accepted to NAACL 2025 Main Track

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21315v1) [paper-pdf](http://arxiv.org/pdf/2503.21315v1)

**Authors**: Cheng Wang, Yiwei Wang, Yujun Cai, Bryan Hooi

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models by incorporating external knowledge, addressing issues like outdated internal knowledge and hallucination. However, their reliance on external knowledge bases makes them vulnerable to corpus poisoning attacks, where adversarial passages can be injected to manipulate retrieval results. Existing methods for crafting such passages, such as random token replacement or training inversion models, are often slow and computationally expensive, requiring either access to retriever's gradients or large computational resources. To address these limitations, we propose Dynamic Importance-Guided Genetic Algorithm (DIGA), an efficient black-box method that leverages two key properties of retrievers: insensitivity to token order and bias towards influential tokens. By focusing on these characteristics, DIGA dynamically adjusts its genetic operations to generate effective adversarial passages with significantly reduced time and memory usage. Our experimental evaluation shows that DIGA achieves superior efficiency and scalability compared to existing methods, while maintaining comparable or better attack success rates across multiple datasets.



## **10. Clean Image May be Dangerous: Data Poisoning Attacks Against Deep Hashing**

cs.CV

Accepted by TMM

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21236v1) [paper-pdf](http://arxiv.org/pdf/2503.21236v1)

**Authors**: Shuai Li, Jie Zhang, Yuang Qi, Kejiang Chen, Tianwei Zhang, Weiming Zhang, Nenghai Yu

**Abstract**: Large-scale image retrieval using deep hashing has become increasingly popular due to the exponential growth of image data and the remarkable feature extraction capabilities of deep neural networks (DNNs). However, deep hashing methods are vulnerable to malicious attacks, including adversarial and backdoor attacks. It is worth noting that these attacks typically involve altering the query images, which is not a practical concern in real-world scenarios. In this paper, we point out that even clean query images can be dangerous, inducing malicious target retrieval results, like undesired or illegal images. To the best of our knowledge, we are the first to study data \textbf{p}oisoning \textbf{a}ttacks against \textbf{d}eep \textbf{hash}ing \textbf{(\textit{PADHASH})}. Specifically, we first train a surrogate model to simulate the behavior of the target deep hashing model. Then, a strict gradient matching strategy is proposed to generate the poisoned images. Extensive experiments on different models, datasets, hash methods, and hash code lengths demonstrate the effectiveness and generality of our attack method.



## **11. Adversarial Wear and Tear: Exploiting Natural Damage for Generating Physical-World Adversarial Examples**

cs.CV

11 pages, 9 figures

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21164v1) [paper-pdf](http://arxiv.org/pdf/2503.21164v1)

**Authors**: Samra Irshad, Seungkyu Lee, Nassir Navab, Hong Joo Lee, Seong Tae Kim

**Abstract**: The presence of adversarial examples in the physical world poses significant challenges to the deployment of Deep Neural Networks in safety-critical applications such as autonomous driving. Most existing methods for crafting physical-world adversarial examples are ad-hoc, relying on temporary modifications like shadows, laser beams, or stickers that are tailored to specific scenarios. In this paper, we introduce a new class of physical-world adversarial examples, AdvWT, which draws inspiration from the naturally occurring phenomenon of `wear and tear', an inherent property of physical objects. Unlike manually crafted perturbations, `wear and tear' emerges organically over time due to environmental degradation, as seen in the gradual deterioration of outdoor signboards. To achieve this, AdvWT follows a two-step approach. First, a GAN-based, unsupervised image-to-image translation network is employed to model these naturally occurring damages, particularly in the context of outdoor signboards. The translation network encodes the characteristics of damaged signs into a latent `damage style code'. In the second step, we introduce adversarial perturbations into the style code, strategically optimizing its transformation process. This manipulation subtly alters the damage style representation, guiding the network to generate adversarial images where the appearance of damages remains perceptually realistic, while simultaneously ensuring their effectiveness in misleading neural networks. Through comprehensive experiments on two traffic sign datasets, we show that AdvWT effectively misleads DNNs in both digital and physical domains. AdvWT achieves an effective attack success rate, greater robustness, and a more natural appearance compared to existing physical-world adversarial examples. Additionally, integrating AdvWT into training enhances a model's generalizability to real-world damaged signs.



## **12. Robust Federated Learning Against Poisoning Attacks: A GAN-Based Defense Framework**

cs.CR

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20884v1) [paper-pdf](http://arxiv.org/pdf/2503.20884v1)

**Authors**: Usama Zafar, André Teixeira, Salman Toor

**Abstract**: Federated Learning (FL) enables collaborative model training across decentralized devices without sharing raw data, but it remains vulnerable to poisoning attacks that compromise model integrity. Existing defenses often rely on external datasets or predefined heuristics (e.g. number of malicious clients), limiting their effectiveness and scalability. To address these limitations, we propose a privacy-preserving defense framework that leverages a Conditional Generative Adversarial Network (cGAN) to generate synthetic data at the server for authenticating client updates, eliminating the need for external datasets. Our framework is scalable, adaptive, and seamlessly integrates into FL workflows. Extensive experiments on benchmark datasets demonstrate its robust performance against a variety of poisoning attacks, achieving high True Positive Rate (TPR) and True Negative Rate (TNR) of malicious and benign clients, respectively, while maintaining model accuracy. The proposed framework offers a practical and effective solution for securing federated learning systems.



## **13. The mathematics of adversarial attacks in AI -- Why deep learning is unstable despite the existence of stable neural networks**

cs.LG

31 pages, 1 figure. Revised to make minor changes to notation and  references

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2109.06098v2) [paper-pdf](http://arxiv.org/pdf/2109.06098v2)

**Authors**: Alexander Bastounis, Anders C Hansen, Verner Vlačić

**Abstract**: The unprecedented success of deep learning (DL) makes it unchallenged when it comes to classification problems. However, it is well established that the current DL methodology produces universally unstable neural networks (NNs). The instability problem has caused an enormous research effort -- with a vast literature on so-called adversarial attacks -- yet there has been no solution to the problem. Our paper addresses why there has been no solution to the problem, as we prove the following mathematical paradox: any training procedure based on training neural networks for classification problems with a fixed architecture will yield neural networks that are either inaccurate or unstable (if accurate) -- despite the provable existence of both accurate and stable neural networks for the same classification problems. The key is that the stable and accurate neural networks must have variable dimensions depending on the input, in particular, variable dimensions is a necessary condition for stability.   Our result points towards the paradox that accurate and stable neural networks exist, however, modern algorithms do not compute them. This yields the question: if the existence of neural networks with desirable properties can be proven, can one also find algorithms that compute them? There are cases in mathematics where provable existence implies computability, but will this be the case for neural networks? The contrary is true, as we demonstrate how neural networks can provably exist as approximate minimisers to standard optimisation problems with standard cost functions, however, no randomised algorithm can compute them with probability better than 1/2.



## **14. Intelligent Code Embedding Framework for High-Precision Ransomware Detection via Multimodal Execution Path Analysis**

cs.CR

arXiv admin note: This paper has been withdrawn by arXiv due to  disputed and unverifiable authorship

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2501.15836v2) [paper-pdf](http://arxiv.org/pdf/2501.15836v2)

**Authors**: Levi Gareth, Maximilian Fairbrother, Peregrine Blackwood, Lucasta Underhill, Benedict Ruthermore

**Abstract**: Modern threat landscapes continue to evolve with increasing sophistication, challenging traditional detection methodologies and necessitating innovative solutions capable of addressing complex adversarial tactics. A novel framework was developed to identify ransomware activity through multimodal execution path analysis, integrating high-dimensional embeddings and dynamic heuristic derivation mechanisms to capture behavioral patterns across diverse attack variants. The approach demonstrated high adaptability, effectively mitigating obfuscation strategies and polymorphic characteristics often employed by ransomware families to evade detection. Comprehensive experimental evaluations revealed significant advancements in precision, recall, and accuracy metrics compared to baseline techniques, particularly under conditions of variable encryption speeds and obfuscated execution flows. The framework achieved scalable and computationally efficient performance, ensuring robust applicability across a range of system configurations, from resource-constrained environments to high-performance infrastructures. Notable findings included reduced false positive rates and enhanced detection latency, even for ransomware families employing sophisticated encryption mechanisms. The modular design allowed seamless integration of additional modalities, enabling extensibility and future-proofing against emerging threat vectors. Quantitative analyses further highlighted the system's energy efficiency, emphasizing its practicality for deployment in environments with stringent operational constraints. The results underline the importance of integrating advanced computational techniques and dynamic adaptability to safeguard digital ecosystems from increasingly complex threats.



## **15. A Survey of Secure Semantic Communications**

cs.CR

160 pages, 27 figures

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2501.00842v2) [paper-pdf](http://arxiv.org/pdf/2501.00842v2)

**Authors**: Rui Meng, Song Gao, Dayu Fan, Haixiao Gao, Yining Wang, Xiaodong Xu, Bizhu Wang, Suyu Lv, Zhidi Zhang, Mengying Sun, Shujun Han, Chen Dong, Xiaofeng Tao, Ping Zhang

**Abstract**: Semantic communication (SemCom) is regarded as a promising and revolutionary technology in 6G, aiming to transcend the constraints of ``Shannon's trap" by filtering out redundant information and extracting the core of effective data. Compared to traditional communication paradigms, SemCom offers several notable advantages, such as reducing the burden on data transmission, enhancing network management efficiency, and optimizing resource allocation. Numerous researchers have extensively explored SemCom from various perspectives, including network architecture, theoretical analysis, potential technologies, and future applications. However, as SemCom continues to evolve, a multitude of security and privacy concerns have arisen, posing threats to the confidentiality, integrity, and availability of SemCom systems. This paper presents a comprehensive survey of the technologies that can be utilized to secure SemCom. Firstly, we elaborate on the entire life cycle of SemCom, which includes the model training, model transfer, and semantic information transmission phases. Then, we identify the security and privacy issues that emerge during these three stages. Furthermore, we summarize the techniques available to mitigate these security and privacy threats, including data cleaning, robust learning, defensive strategies against backdoor attacks, adversarial training, differential privacy, cryptography, blockchain technology, model compression, and physical-layer security. Lastly, this paper outlines future research directions to guide researchers in related fields.



## **16. $β$-GNN: A Robust Ensemble Approach Against Graph Structure Perturbation**

cs.LG

This is the author's version of the paper accepted at EuroMLSys 2025

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20630v1) [paper-pdf](http://arxiv.org/pdf/2503.20630v1)

**Authors**: Haci Ismail Aslan, Philipp Wiesner, Ping Xiong, Odej Kao

**Abstract**: Graph Neural Networks (GNNs) are playing an increasingly important role in the efficient operation and security of computing systems, with applications in workload scheduling, anomaly detection, and resource management. However, their vulnerability to network perturbations poses a significant challenge. We propose $\beta$-GNN, a model enhancing GNN robustness without sacrificing clean data performance. $\beta$-GNN uses a weighted ensemble, combining any GNN with a multi-layer perceptron. A learned dynamic weight, $\beta$, modulates the GNN's contribution. This $\beta$ not only weights GNN influence but also indicates data perturbation levels, enabling proactive mitigation. Experimental results on diverse datasets show $\beta$-GNN's superior adversarial accuracy and attack severity quantification. Crucially, $\beta$-GNN avoids perturbation assumptions, preserving clean data structure and performance.



## **17. Robust Deep Reinforcement Learning in Robotics via Adaptive Gradient-Masked Adversarial Attacks**

cs.LG

9 pages, 6 figures

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20844v1) [paper-pdf](http://arxiv.org/pdf/2503.20844v1)

**Authors**: Zongyuan Zhang, Tianyang Duan, Zheng Lin, Dong Huang, Zihan Fang, Zekai Sun, Ling Xiong, Hongbin Liang, Heming Cui, Yong Cui, Yue Gao

**Abstract**: Deep reinforcement learning (DRL) has emerged as a promising approach for robotic control, but its realworld deployment remains challenging due to its vulnerability to environmental perturbations. Existing white-box adversarial attack methods, adapted from supervised learning, fail to effectively target DRL agents as they overlook temporal dynamics and indiscriminately perturb all state dimensions, limiting their impact on long-term rewards. To address these challenges, we propose the Adaptive Gradient-Masked Reinforcement (AGMR) Attack, a white-box attack method that combines DRL with a gradient-based soft masking mechanism to dynamically identify critical state dimensions and optimize adversarial policies. AGMR selectively allocates perturbations to the most impactful state features and incorporates a dynamic adjustment mechanism to balance exploration and exploitation during training. Extensive experiments demonstrate that AGMR outperforms state-of-the-art adversarial attack methods in degrading the performance of the victim agent and enhances the victim agent's robustness through adversarial defense mechanisms.



## **18. State-Aware Perturbation Optimization for Robust Deep Reinforcement Learning**

cs.LG

15 pages, 11 figures

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20613v1) [paper-pdf](http://arxiv.org/pdf/2503.20613v1)

**Authors**: Zongyuan Zhang, Tianyang Duan, Zheng Lin, Dong Huang, Zihan Fang, Zekai Sun, Ling Xiong, Hongbin Liang, Heming Cui, Yong Cui

**Abstract**: Recently, deep reinforcement learning (DRL) has emerged as a promising approach for robotic control. However, the deployment of DRL in real-world robots is hindered by its sensitivity to environmental perturbations. While existing whitebox adversarial attacks rely on local gradient information and apply uniform perturbations across all states to evaluate DRL robustness, they fail to account for temporal dynamics and state-specific vulnerabilities. To combat the above challenge, we first conduct a theoretical analysis of white-box attacks in DRL by establishing the adversarial victim-dynamics Markov decision process (AVD-MDP), to derive the necessary and sufficient conditions for a successful attack. Based on this, we propose a selective state-aware reinforcement adversarial attack method, named STAR, to optimize perturbation stealthiness and state visitation dispersion. STAR first employs a soft mask-based state-targeting mechanism to minimize redundant perturbations, enhancing stealthiness and attack effectiveness. Then, it incorporates an information-theoretic optimization objective to maximize mutual information between perturbations, environmental states, and victim actions, ensuring a dispersed state-visitation distribution that steers the victim agent into vulnerable states for maximum return reduction. Extensive experiments demonstrate that STAR outperforms state-of-the-art benchmarks.



## **19. Feature Statistics with Uncertainty Help Adversarial Robustness**

cs.LG

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20583v1) [paper-pdf](http://arxiv.org/pdf/2503.20583v1)

**Authors**: Ran Wang, Xinlei Zhou, Rihao Li, Meng Hu, Wenhui Wu, Yuheng Jia

**Abstract**: Despite the remarkable success of deep neural networks (DNNs), the security threat of adversarial attacks poses a significant challenge to the reliability of DNNs. By introducing randomness into different parts of DNNs, stochastic methods can enable the model to learn some uncertainty, thereby improving model robustness efficiently. In this paper, we theoretically discover a universal phenomenon that adversarial attacks will shift the distributions of feature statistics. Motivated by this theoretical finding, we propose a robustness enhancement module called Feature Statistics with Uncertainty (FSU). It resamples channel-wise feature means and standard deviations of examples from multivariate Gaussian distributions, which helps to reconstruct the attacked examples and calibrate the shifted distributions. The calibration recovers some domain characteristics of the data for classification, thereby mitigating the influence of perturbations and weakening the ability of attacks to deceive models. The proposed FSU module has universal applicability in training, attacking, predicting and fine-tuning, demonstrating impressive robustness enhancement ability at trivial additional time cost. For example, against powerful optimization-based CW attacks, by incorporating FSU into attacking and predicting phases, it endows many collapsed state-of-the-art models with 50%-80% robust accuracy on CIFAR10, CIFAR100 and SVHN.



## **20. Aligning Visual Contrastive learning models via Preference Optimization**

cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2411.08923v3) [paper-pdf](http://arxiv.org/pdf/2411.08923v3)

**Authors**: Amirabbas Afzali, Borna Khodabandeh, Ali Rasekh, Mahyar JafariNodeh, Sepehr kazemi, Simon Gottschalk

**Abstract**: Contrastive learning models have demonstrated impressive abilities to capture semantic similarities by aligning representations in the embedding space. However, their performance can be limited by the quality of the training data and its inherent biases. While Preference Optimization (PO) methods such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) have been applied to align generative models with human preferences, their use in contrastive learning has yet to be explored. This paper introduces a novel method for training contrastive learning models using different PO methods to break down complex concepts. Our method systematically aligns model behavior with desired preferences, enhancing performance on the targeted task. In particular, we focus on enhancing model robustness against typographic attacks and inductive biases, commonly seen in contrastive vision-language models like CLIP. Our experiments demonstrate that models trained using PO outperform standard contrastive learning techniques while retaining their ability to handle adversarial challenges and maintain accuracy on other downstream tasks. This makes our method well-suited for tasks requiring fairness, robustness, and alignment with specific preferences. We evaluate our method for tackling typographic attacks on images and explore its ability to disentangle gender concepts and mitigate gender bias, showcasing the versatility of our approach.



## **21. Lipschitz Constant Meets Condition Number: Learning Robust and Compact Deep Neural Networks**

cs.LG

13 pages, 6 figures

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20454v1) [paper-pdf](http://arxiv.org/pdf/2503.20454v1)

**Authors**: Yangqi Feng, Shing-Ho J. Lin, Baoyuan Gao, Xian Wei

**Abstract**: Recent research has revealed that high compression of Deep Neural Networks (DNNs), e.g., massive pruning of the weight matrix of a DNN, leads to a severe drop in accuracy and susceptibility to adversarial attacks. Integration of network pruning into an adversarial training framework has been proposed to promote adversarial robustness. It has been observed that a highly pruned weight matrix tends to be ill-conditioned, i.e., increasing the condition number of the weight matrix. This phenomenon aggravates the vulnerability of a DNN to input noise. Although a highly pruned weight matrix is considered to be able to lower the upper bound of the local Lipschitz constant to tolerate large distortion, the ill-conditionedness of such a weight matrix results in a non-robust DNN model. To overcome this challenge, this work develops novel joint constraints to adjust the weight distribution of networks, namely, the Transformed Sparse Constraint joint with Condition Number Constraint (TSCNC), which copes with smoothing distribution and differentiable constraint functions to reduce condition number and thus avoid the ill-conditionedness of weight matrices. Furthermore, our theoretical analyses unveil the relevance between the condition number and the local Lipschitz constant of the weight matrix, namely, the sharply increasing condition number becomes the dominant factor that restricts the robustness of over-sparsified models. Extensive experiments are conducted on several public datasets, and the results show that the proposed constraints significantly improve the robustness of a DNN with high pruning rates.



## **22. UnReference: analysis of the effect of spoofing on RTK reference stations for connected rovers**

cs.CR

To appear the the 2025 IEEE/ION Position, Navigation and Localization  Symposium

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20364v1) [paper-pdf](http://arxiv.org/pdf/2503.20364v1)

**Authors**: Marco Spanghero, Panos Papadimitratos

**Abstract**: Global Navigation Satellite Systems (GNSS) provide standalone precise navigation for a wide gamut of applications. Nevertheless, applications or systems such as unmanned vehicles (aerial or ground vehicles and surface vessels) generally require a much higher level of accuracy than those provided by standalone receivers. The most effective and economical way of achieving centimeter-level accuracy is to rely on corrections provided by fixed \emph{reference station} receivers to improve the satellite ranging measurements. Differential GNSS (DGNSS) and Real Time Kinematics (RTK) provide centimeter-level accuracy by distributing online correction streams to connected nearby mobile receivers typically termed \emph{rovers}. However, due to their static nature, reference stations are prime targets for GNSS attacks, both simplistic jamming and advanced spoofing, with different levels of adversarial control and complexity. Jamming the reference station would deny corrections and thus accuracy to the rovers. Spoofing the reference station would force it to distribute misleading corrections. As a result, all connected rovers using those corrections will be equally influenced by the adversary independently of their actual trajectory. We evaluate a battery of tests generated with an RF simulator to test the robustness of a common DGNSS/RTK processing library and receivers. We test both jamming and synchronized spoofing to demonstrate that adversarial action on the rover using reference spoofing is both effective and convenient from an adversarial perspective. Additionally, we discuss possible strategies based on existing countermeasures (self-validation of the PNT solution and monitoring of own clock drift) that the rover and the reference station can adopt to avoid using or distributing bogus corrections.



## **23. Enabling Heterogeneous Adversarial Transferability via Feature Permutation Attacks**

cs.CV

PAKDD 2025. Main Track

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20310v1) [paper-pdf](http://arxiv.org/pdf/2503.20310v1)

**Authors**: Tao Wu, Tie Luo

**Abstract**: Adversarial attacks in black-box settings are highly practical, with transfer-based attacks being the most effective at generating adversarial examples (AEs) that transfer from surrogate models to unseen target models. However, their performance significantly degrades when transferring across heterogeneous architectures -- such as CNNs, MLPs, and Vision Transformers (ViTs) -- due to fundamental architectural differences. To address this, we propose Feature Permutation Attack (FPA), a zero-FLOP, parameter-free method that enhances adversarial transferability across diverse architectures. FPA introduces a novel feature permutation (FP) operation, which rearranges pixel values in selected feature maps to simulate long-range dependencies, effectively making CNNs behave more like ViTs and MLPs. This enhances feature diversity and improves transferability both across heterogeneous architectures and within homogeneous CNNs. Extensive evaluations on 14 state-of-the-art architectures show that FPA achieves maximum absolute gains in attack success rates of 7.68% on CNNs, 14.57% on ViTs, and 14.48% on MLPs, outperforming existing black-box attacks. Additionally, FPA is highly generalizable and can seamlessly integrate with other transfer-based attacks to further boost their performance. Our findings establish FPA as a robust, efficient, and computationally lightweight strategy for enhancing adversarial transferability across heterogeneous architectures.



## **24. Are We There Yet? Unraveling the State-of-the-Art Graph Network Intrusion Detection Systems**

cs.CR

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20281v1) [paper-pdf](http://arxiv.org/pdf/2503.20281v1)

**Authors**: Chenglong Wang, Pujia Zheng, Jiaping Gui, Cunqing Hua, Wajih Ul Hassan

**Abstract**: Network Intrusion Detection Systems (NIDS) are vital for ensuring enterprise security. Recently, Graph-based NIDS (GIDS) have attracted considerable attention because of their capability to effectively capture the complex relationships within the graph structures of data communications. Despite their promise, the reproducibility and replicability of these GIDS remain largely unexplored, posing challenges for developing reliable and robust detection systems. This study bridges this gap by designing a systematic approach to evaluate state-of-the-art GIDS, which includes critically assessing, extending, and clarifying the findings of these systems. We further assess the robustness of GIDS under adversarial attacks. Evaluations were conducted on three public datasets as well as a newly collected large-scale enterprise dataset. Our findings reveal significant performance discrepancies, highlighting challenges related to dataset scale, model inputs, and implementation settings. We demonstrate difficulties in reproducing and replicating results, particularly concerning false positive rates and robustness against adversarial attacks. This work provides valuable insights and recommendations for future research, emphasizing the importance of rigorous reproduction and replication studies in developing robust and generalizable GIDS solutions.



## **25. Hi-ALPS -- An Experimental Robustness Quantification of Six LiDAR-based Object Detection Systems for Autonomous Driving**

cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.17168v2) [paper-pdf](http://arxiv.org/pdf/2503.17168v2)

**Authors**: Alexandra Arzberger, Ramin Tavakoli Kolagari

**Abstract**: Light Detection and Ranging (LiDAR) is an essential sensor technology for autonomous driving as it can capture high-resolution 3D data. As 3D object detection systems (OD) can interpret such point cloud data, they play a key role in the driving decisions of autonomous vehicles. Consequently, such 3D OD must be robust against all types of perturbations and must therefore be extensively tested. One approach is the use of adversarial examples, which are small, sometimes sophisticated perturbations in the input data that change, i.e., falsify, the prediction of the OD. These perturbations are carefully designed based on the weaknesses of the OD. The robustness of the OD cannot be quantified with adversarial examples in general, because if the OD is vulnerable to a given attack, it is unclear whether this is due to the robustness of the OD or whether the attack algorithm produces particularly strong adversarial examples. The contribution of this work is Hi-ALPS -- Hierarchical Adversarial-example-based LiDAR Perturbation Level System, where higher robustness of the OD is required to withstand the perturbations as the perturbation levels increase. In doing so, the Hi-ALPS levels successively implement a heuristic followed by established adversarial example approaches. In a series of comprehensive experiments using Hi-ALPS, we quantify the robustness of six state-of-the-art 3D OD under different types of perturbations. The results of the experiments show that none of the OD is robust against all Hi-ALPS levels; an important factor for the ranking is that human observers can still correctly recognize the perturbed objects, as the respective perturbations are small. To increase the robustness of the OD, we discuss the applicability of state-of-the-art countermeasures. In addition, we derive further suggestions for countermeasures based on our experimental results.



## **26. How Secure is Forgetting? Linking Machine Unlearning to Machine Learning Attacks**

cs.CR

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20257v1) [paper-pdf](http://arxiv.org/pdf/2503.20257v1)

**Authors**: Muhammed Shafi K. P., Serena Nicolazzo, Antonino Nocera, Vinod P

**Abstract**: As Machine Learning (ML) evolves, the complexity and sophistication of security threats against this paradigm continue to grow as well, threatening data privacy and model integrity. In response, Machine Unlearning (MU) is a recent technology that aims to remove the influence of specific data from a trained model, enabling compliance with privacy regulations and user requests. This can be done for privacy compliance (e.g., GDPR's right to be forgotten) or model refinement. However, the intersection between classical threats in ML and MU remains largely unexplored. In this Systematization of Knowledge (SoK), we provide a structured analysis of security threats in ML and their implications for MU. We analyze four major attack classes, namely, Backdoor Attacks, Membership Inference Attacks (MIA), Adversarial Attacks, and Inversion Attacks, we investigate their impact on MU and propose a novel classification based on how they are usually used in this context. Finally, we identify open challenges, including ethical considerations, and explore promising future research directions, paving the way for future research in secure and privacy-preserving Machine Unlearning.



## **27. Defending against Backdoor Attack on Deep Neural Networks**

cs.CR

This workshop manuscript is not a publication and will not be  published anywhere

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2002.12162v3) [paper-pdf](http://arxiv.org/pdf/2002.12162v3)

**Authors**: Hao Cheng, Kaidi Xu, Sijia Liu, Pin-Yu Chen, Pu Zhao, Xue Lin

**Abstract**: Although deep neural networks (DNNs) have achieved a great success in various computer vision tasks, it is recently found that they are vulnerable to adversarial attacks. In this paper, we focus on the so-called \textit{backdoor attack}, which injects a backdoor trigger to a small portion of training data (also known as data poisoning) such that the trained DNN induces misclassification while facing examples with this trigger. To be specific, we carefully study the effect of both real and synthetic backdoor attacks on the internal response of vanilla and backdoored DNNs through the lens of Gard-CAM. Moreover, we show that the backdoor attack induces a significant bias in neuron activation in terms of the $\ell_\infty$ norm of an activation map compared to its $\ell_1$ and $\ell_2$ norm. Spurred by our results, we propose the \textit{$\ell_\infty$-based neuron pruning} to remove the backdoor from the backdoored DNN. Experiments show that our method could effectively decrease the attack success rate, and also hold a high classification accuracy for clean images.



## **28. Persistence of Backdoor-based Watermarks for Neural Networks: A Comprehensive Evaluation**

cs.LG

Preprint. Under Review

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2501.02704v2) [paper-pdf](http://arxiv.org/pdf/2501.02704v2)

**Authors**: Anh Tu Ngo, Chuan Song Heng, Nandish Chattopadhyay, Anupam Chattopadhyay

**Abstract**: Deep Neural Networks (DNNs) have gained considerable traction in recent years due to the unparalleled results they gathered. However, the cost behind training such sophisticated models is resource intensive, resulting in many to consider DNNs to be intellectual property (IP) to model owners. In this era of cloud computing, high-performance DNNs are often deployed all over the internet so that people can access them publicly. As such, DNN watermarking schemes, especially backdoor-based watermarks, have been actively developed in recent years to preserve proprietary rights. Nonetheless, there lies much uncertainty on the robustness of existing backdoor watermark schemes, towards both adversarial attacks and unintended means such as fine-tuning neural network models. One reason for this is that no complete guarantee of robustness can be assured in the context of backdoor-based watermark. In this paper, we extensively evaluate the persistence of recent backdoor-based watermarks within neural networks in the scenario of fine-tuning, we propose/develop a novel data-driven idea to restore watermark after fine-tuning without exposing the trigger set. Our empirical results show that by solely introducing training data after fine-tuning, the watermark can be restored if model parameters do not shift dramatically during fine-tuning. Depending on the types of trigger samples used, trigger accuracy can be reinstated to up to 100%. Our study further explores how the restoration process works using loss landscape visualization, as well as the idea of introducing training data in fine-tuning stage to alleviate watermark vanishing.



## **29. Joint Task Offloading and User Scheduling in 5G MEC under Jamming Attacks**

cs.CR

6 pages, 5 figures, Accepted to IEEE International Conference in  Communications (ICC) 2025

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2501.13227v2) [paper-pdf](http://arxiv.org/pdf/2501.13227v2)

**Authors**: Mohammadreza Amini, Burak Kantarci, Claude D'Amours, Melike Erol-Kantarci

**Abstract**: In this paper, we propose a novel joint task offloading and user scheduling (JTO-US) framework for 5G mobile edge computing (MEC) systems under security threats from jamming attacks. The goal is to minimize the delay and the ratio of dropped tasks, taking into account both communication and computation delays. The system model includes a 5G network equipped with MEC servers and an adversarial on-off jammer that disrupts communication. The proposed framework optimally schedules tasks and users to minimize the impact of jamming while ensuring that high-priority tasks are processed efficiently. Genetic algorithm (GA) is used to solve the optimization problem, and the results are compared with benchmark methods such as GA without considering jamming effect, Shortest Job First (SJF), and Shortest Deadline First (SDF). The simulation results demonstrate that the proposed JTO-US framework achieves the lowest drop ratio and effectively manages priority tasks, outperforming existing methods. Particularly, when the jamming probability is 0.8, the proposed framework mitigates the jammer's impact by reducing the drop ratio to 63%, compared to 89% achieved by the next best method.



## **30. Exploring Adversarial Threat Models in Cyber Physical Battery Systems**

eess.SY

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2401.13801v2) [paper-pdf](http://arxiv.org/pdf/2401.13801v2)

**Authors**: Shanthan Kumar Padisala, Shashank Dhananjay Vyas, Satadru Dey

**Abstract**: Technological advancements like the Internet of Things (IoT) have facilitated data exchange across various platforms. This data exchange across various platforms has transformed the traditional battery system into a cyber physical system. Such connectivity makes modern cyber physical battery systems vulnerable to cyber threats where a cyber attacker can manipulate sensing and actuation signals to bring the battery system into an unsafe operating condition. Hence, it is essential to build resilience in modern cyber physical battery systems (CPBS) under cyber attacks. The first step of building such resilience is to analyze potential adversarial behavior, that is, how the adversaries can inject attacks into the battery systems. However, it has been found that in this under-explored area of battery cyber physical security, such an adversarial threat model has not been studied in a systematic manner. In this study, we address this gap and explore adversarial attack generation policies based on optimal control framework. The framework is developed by performing theoretical analysis, which is subsequently supported by evaluation with experimental data generated from a commercial battery cell.



## **31. Bitstream Collisions in Neural Image Compression via Adversarial Perturbations**

cs.CR

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19817v1) [paper-pdf](http://arxiv.org/pdf/2503.19817v1)

**Authors**: Jordan Madden, Lhamo Dorje, Xiaohua Li

**Abstract**: Neural image compression (NIC) has emerged as a promising alternative to classical compression techniques, offering improved compression ratios. Despite its progress towards standardization and practical deployment, there has been minimal exploration into it's robustness and security. This study reveals an unexpected vulnerability in NIC - bitstream collisions - where semantically different images produce identical compressed bitstreams. Utilizing a novel whitebox adversarial attack algorithm, this paper demonstrates that adding carefully crafted perturbations to semantically different images can cause their compressed bitstreams to collide exactly. The collision vulnerability poses a threat to the practical usability of NIC, particularly in security-critical applications. The cause of the collision is analyzed, and a simple yet effective mitigation method is presented.



## **32. SITA: Structurally Imperceptible and Transferable Adversarial Attacks for Stylized Image Generation**

cs.CV

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19791v1) [paper-pdf](http://arxiv.org/pdf/2503.19791v1)

**Authors**: Jingdan Kang, Haoxin Yang, Yan Cai, Huaidong Zhang, Xuemiao Xu, Yong Du, Shengfeng He

**Abstract**: Image generation technology has brought significant advancements across various fields but has also raised concerns about data misuse and potential rights infringements, particularly with respect to creating visual artworks. Current methods aimed at safeguarding artworks often employ adversarial attacks. However, these methods face challenges such as poor transferability, high computational costs, and the introduction of noticeable noise, which compromises the aesthetic quality of the original artwork. To address these limitations, we propose a Structurally Imperceptible and Transferable Adversarial (SITA) attacks. SITA leverages a CLIP-based destylization loss, which decouples and disrupts the robust style representation of the image. This disruption hinders style extraction during stylized image generation, thereby impairing the overall stylization process. Importantly, SITA eliminates the need for a surrogate diffusion model, leading to significantly reduced computational overhead. The method's robust style feature disruption ensures high transferability across diverse models. Moreover, SITA introduces perturbations by embedding noise within the imperceptible structural details of the image. This approach effectively protects against style extraction without compromising the visual quality of the artwork. Extensive experiments demonstrate that SITA offers superior protection for artworks against unauthorized use in stylized generation. It significantly outperforms existing methods in terms of transferability, computational efficiency, and noise imperceptibility. Code is available at https://github.com/A-raniy-day/SITA.



## **33. Lifting Linear Sketches: Optimal Bounds and Adversarial Robustness**

cs.DS

To appear in STOC 2025

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19629v1) [paper-pdf](http://arxiv.org/pdf/2503.19629v1)

**Authors**: Elena Gribelyuk, Honghao Lin, David P. Woodruff, Huacheng Yu, Samson Zhou

**Abstract**: We introduce a novel technique for ``lifting'' dimension lower bounds for linear sketches in the real-valued setting to dimension lower bounds for linear sketches with polynomially-bounded integer entries when the input is a polynomially-bounded integer vector. Using this technique, we obtain the first optimal sketching lower bounds for discrete inputs in a data stream, for classical problems such as approximating the frequency moments, estimating the operator norm, and compressed sensing. Additionally, we lift the adaptive attack of Hardt and Woodruff (STOC, 2013) for breaking any real-valued linear sketch via a sequence of real-valued queries, and show how to obtain an attack on any integer-valued linear sketch using integer-valued queries. This shows that there is no linear sketch in a data stream with insertions and deletions that is adversarially robust for approximating any $L_p$ norm of the input, resolving a central open question for adversarially robust streaming algorithms. To do so, we introduce a new pre-processing technique of independent interest which, given an integer-valued linear sketch, increases the dimension of the sketch by only a constant factor in order to make the orthogonal lattice to its row span smooth. This pre-processing then enables us to leverage results in lattice theory on discrete Gaussian distributions and reason that efficient discrete sketches imply efficient continuous sketches. Our work resolves open questions from the Banff '14 and '17 workshops on Communication Complexity and Applications, as well as the STOC '21 and FOCS '23 workshops on adaptivity and robustness.



## **34. Semantic Entanglement-Based Ransomware Detection via Probabilistic Latent Encryption Mapping**

cs.CR

arXiv admin note: This paper has been withdrawn by arXiv due to  disputed and unverifiable authorship

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2502.02730v2) [paper-pdf](http://arxiv.org/pdf/2502.02730v2)

**Authors**: Mohammad Eisa, Quentin Yardley, Rafael Witherspoon, Harriet Pendlebury, Clement Rutherford

**Abstract**: Encryption-based attacks have introduced significant challenges for detection mechanisms that rely on predefined signatures, heuristic indicators, or static rule-based classifications. Probabilistic Latent Encryption Mapping presents an alternative detection framework that models ransomware-induced encryption behaviors through statistical representations of entropy deviations and probabilistic dependencies in execution traces. Unlike conventional approaches that depend on explicit bytecode analysis or predefined cryptographic function call monitoring, probabilistic inference techniques classify encryption anomalies based on their underlying statistical characteristics, ensuring greater adaptability to polymorphic attack strategies. Evaluations demonstrate that entropy-driven classification reduces false positive rates while maintaining high detection accuracy across diverse ransomware families and encryption methodologies. Experimental results further highlight the framework's ability to differentiate between benign encryption workflows and adversarial cryptographic manipulations, ensuring that classification performance remains effective across cloud-based and localized execution environments. Benchmark comparisons illustrate that probabilistic modeling exhibits advantages over heuristic and machine learning-based detection approaches, particularly in handling previously unseen encryption techniques and adversarial obfuscation strategies. Computational efficiency analysis confirms that detection latency remains within operational feasibility constraints, reinforcing the viability of probabilistic encryption classification for real-time security infrastructures. The ability to systematically infer encryption-induced deviations without requiring static attack signatures strengthens detection robustness against adversarial evasion techniques.



## **35. Nanopass Back-Translation of Call-Return Trees for Mechanized Secure Compilation Proofs**

cs.PL

ITP'25 submission

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19609v1) [paper-pdf](http://arxiv.org/pdf/2503.19609v1)

**Authors**: Jérémy Thibault, Joseph Lenormand, Catalin Hritcu

**Abstract**: Researchers aim to build secure compilation chains enforcing that if there is no attack a source context can mount against a source program then there is also no attack an adversarial target context can mount against the compiled program. Proving that these compilation chains are secure is, however, challenging, and involves a non-trivial back-translation step: for any attack a target context mounts against the compiled program one has to exhibit a source context mounting the same attack against the source program. We describe a novel back-translation technique, which results in simpler proofs that can be more easily mechanized in a proof assistant. Given a finite set of finite trace prefixes, capturing the interaction recorded during an attack between a target context and the compiled program, we build a call-return tree that we back-translate into a source context producing the same trace prefixes. We use state in the generated source context to record the current location in the call-return tree. The back-translation is done in several small steps, each adding to the tree new information describing how the location should change depending on how the context regains control. To prove this back-translation correct we give semantics to every intermediate call-return tree language, using ghost state to store information and explicitly enforce execution invariants. We prove several small forward simulations, basically seeing the back-translation as a verified nanopass compiler. Thanks to this modular structure, we are able to mechanize this complex back-translation and its correctness proof in the Rocq prover without too much effort.



## **36. Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?**

cs.CL

Accepted in ICLR 2025

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2412.03235v2) [paper-pdf](http://arxiv.org/pdf/2412.03235v2)

**Authors**: Sravanti Addepalli, Yerram Varun, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.



## **37. Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond**

cs.LG

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2502.05374v3) [paper-pdf](http://arxiv.org/pdf/2502.05374v3)

**Authors**: Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, Sijia Liu

**Abstract**: The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.



## **38. Boosting the Transferability of Audio Adversarial Examples with Acoustic Representation Optimization**

cs.SD

Accepted to ICME 2025

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19591v1) [paper-pdf](http://arxiv.org/pdf/2503.19591v1)

**Authors**: Weifei Jin, Junjie Su, Hejia Wang, Yulin Ye, Jie Hao

**Abstract**: With the widespread application of automatic speech recognition (ASR) systems, their vulnerability to adversarial attacks has been extensively studied. However, most existing adversarial examples are generated on specific individual models, resulting in a lack of transferability. In real-world scenarios, attackers often cannot access detailed information about the target model, making query-based attacks unfeasible. To address this challenge, we propose a technique called Acoustic Representation Optimization that aligns adversarial perturbations with low-level acoustic characteristics derived from speech representation models. Rather than relying on model-specific, higher-layer abstractions, our approach leverages fundamental acoustic representations that remain consistent across diverse ASR architectures. By enforcing an acoustic representation loss to guide perturbations toward these robust, lower-level representations, we enhance the cross-model transferability of adversarial examples without degrading audio quality. Our method is plug-and-play and can be integrated with any existing attack methods. We evaluate our approach on three modern ASR models, and the experimental results demonstrate that our method significantly improves the transferability of adversarial examples generated by previous methods while preserving the audio quality.



## **39. Towards Imperceptible Adversarial Attacks for Time Series Classification with Local Perturbations and Frequency Analysis**

cs.CR

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19519v1) [paper-pdf](http://arxiv.org/pdf/2503.19519v1)

**Authors**: Wenwei Gu, Renyi Zhong, Jianping Zhang, Michael R. Lyu

**Abstract**: Adversarial attacks in time series classification (TSC) models have recently gained attention due to their potential to compromise model robustness. Imperceptibility is crucial, as adversarial examples detected by the human vision system (HVS) can render attacks ineffective. Many existing methods fail to produce high-quality imperceptible examples, often generating perturbations with more perceptible low-frequency components, like square waves, and global perturbations that reduce stealthiness. This paper aims to improve the imperceptibility of adversarial attacks on TSC models by addressing frequency components and time series locality. We propose the Shapelet-based Frequency-domain Attack (SFAttack), which uses local perturbations focused on time series shapelets to enhance discriminative information and stealthiness. Additionally, we introduce a low-frequency constraint to confine perturbations to high-frequency components, enhancing imperceptibility.



## **40. Using Anomaly Detection to Detect Poisoning Attacks in Federated Learning Applications**

cs.LG

We will updated this article soon

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2207.08486v4) [paper-pdf](http://arxiv.org/pdf/2207.08486v4)

**Authors**: Ali Raza, Shujun Li, Kim-Phuc Tran, Ludovic Koehl, Kim Duc Tran

**Abstract**: Adversarial attacks such as poisoning attacks have attracted the attention of many machine learning researchers. Traditionally, poisoning attacks attempt to inject adversarial training data in order to manipulate the trained model. In federated learning (FL), data poisoning attacks can be generalized to model poisoning attacks, which cannot be detected by simpler methods due to the lack of access to local training data by the detector. State-of-the-art poisoning attack detection methods for FL have various weaknesses, e.g., the number of attackers has to be known or not high enough, working with i.i.d. data only, and high computational complexity. To overcome above weaknesses, we propose a novel framework for detecting poisoning attacks in FL, which employs a reference model based on a public dataset and an auditor model to detect malicious updates. We implemented a detector based on the proposed framework and using a one-class support vector machine (OC-SVM), which reaches the lowest possible computational complexity O(K) where K is the number of clients. We evaluated our detector's performance against state-of-the-art (SOTA) poisoning attacks for two typical applications of FL: electrocardiograph (ECG) classification and human activity recognition (HAR). Our experimental results validated the performance of our detector over other SOTA detection methods.



## **41. NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary**

cs.CV

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.00063v3) [paper-pdf](http://arxiv.org/pdf/2503.00063v3)

**Authors**: Zezeng Li, Xiaoyu Du, Na Lei, Liming Chen, Weimin Wang

**Abstract**: Adversarial attacks exploit the vulnerability of deep models against adversarial samples. Existing point cloud attackers are tailored to specific models, iteratively optimizing perturbations based on gradients in either a white-box or black-box setting. Despite their promising attack performance, they often struggle to produce transferable adversarial samples due to overfitting the specific parameters of surrogate models. To overcome this issue, we shift our focus to the data distribution itself and introduce a novel approach named NoPain, which employs optimal transport (OT) to identify the inherent singular boundaries of the data manifold for cross-network point cloud attacks. Specifically, we first calculate the OT mapping from noise to the target feature space, then identify singular boundaries by locating non-differentiable positions. Finally, we sample along singular boundaries to generate adversarial point clouds. Once the singular boundaries are determined, NoPain can efficiently produce adversarial samples without the need of iterative updates or guidance from the surrogate classifiers. Extensive experiments demonstrate that the proposed end-to-end method outperforms baseline approaches in terms of both transferability and efficiency, while also maintaining notable advantages even against defense strategies. Code and model are available at https://github.com/cognaclee/nopain



## **42. Improving Transferable Targeted Attacks with Feature Tuning Mixup**

cs.CV

CVPR 2025

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2411.15553v2) [paper-pdf](http://arxiv.org/pdf/2411.15553v2)

**Authors**: Kaisheng Liang, Xuelong Dai, Yanjie Li, Dong Wang, Bin Xiao

**Abstract**: Deep neural networks (DNNs) exhibit vulnerability to adversarial examples that can transfer across different DNN models. A particularly challenging problem is developing transferable targeted attacks that can mislead DNN models into predicting specific target classes. While various methods have been proposed to enhance attack transferability, they often incur substantial computational costs while yielding limited improvements. Recent clean feature mixup methods use random clean features to perturb the feature space but lack optimization for disrupting adversarial examples, overlooking the advantages of attack-specific perturbations. In this paper, we propose Feature Tuning Mixup (FTM), a novel method that enhances targeted attack transferability by combining both random and optimized noises in the feature space. FTM introduces learnable feature perturbations and employs an efficient stochastic update strategy for optimization. These learnable perturbations facilitate the generation of more robust adversarial examples with improved transferability. We further demonstrate that attack performance can be enhanced through an ensemble of multiple FTM-perturbed surrogate models. Extensive experiments on the ImageNet-compatible dataset across various DNN models demonstrate that our method achieves significant improvements over state-of-the-art methods while maintaining low computational cost.



## **43. ImF: Implicit Fingerprint for Large Language Models**

cs.CL

16 pages, 7 figures

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.21805v1) [paper-pdf](http://arxiv.org/pdf/2503.21805v1)

**Authors**: Wu jiaxuan, Peng Wanli, Fu hang, Xue Yiming, Wen juan

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making intellectual property (IP) protection essential. Most existing model fingerprint methods inject fingerprints into LLMs to protect model ownership. These methods create fingerprint pairs with weak semantic correlations, lacking the contextual coherence and semantic relatedness founded in normal question-answer (QA) pairs in LLMs. In this paper, we propose a Generation Revision Intervention (GRI) attack that can effectively exploit this flaw to erase fingerprints, highlighting the need for more secure model fingerprint methods. Thus, we propose a novel injected fingerprint paradigm called Implicit Fingerprints (ImF). ImF constructs fingerprint pairs with strong semantic correlations, disguising them as natural QA pairs within LLMs. This ensures the fingerprints are consistent with normal model behavior, making them indistinguishable and robust against detection and removal. Our experiment on multiple LLMs demonstrates that ImF retains high verification success rates under adversarial conditions, offering a reliable solution for protecting LLM ownership.



## **44. Stop Walking in Circles! Bailing Out Early in Projected Gradient Descent**

cs.CV

To appear in the 2025 IEEE/CVF Conference on Computer Vision and  Pattern Recognition (CVPR)

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19347v1) [paper-pdf](http://arxiv.org/pdf/2503.19347v1)

**Authors**: Philip Doldo, Derek Everett, Amol Khanna, Andre T Nguyen, Edward Raff

**Abstract**: Projected Gradient Descent (PGD) under the $L_\infty$ ball has become one of the defacto methods used in adversarial robustness evaluation for computer vision (CV) due to its reliability and efficacy, making a strong and easy-to-implement iterative baseline. However, PGD is computationally demanding to apply, especially when using thousands of iterations is the current best-practice recommendation to generate an adversarial example for a single image. In this work, we introduce a simple novel method for early termination of PGD based on cycle detection by exploiting the geometry of how PGD is implemented in practice and show that it can produce large speedup factors while providing the \emph{exact} same estimate of model robustness as standard PGD. This method substantially speeds up PGD without sacrificing any attack strength, enabling evaluations of robustness that were previously computationally intractable.



## **45. Efficient Adversarial Detection Frameworks for Vehicle-to-Microgrid Services in Edge Computing**

cs.CR

6 pages, 3 figures, Accepted to 2025 IEEE International Conference on  Communications (ICC) Workshops

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19318v1) [paper-pdf](http://arxiv.org/pdf/2503.19318v1)

**Authors**: Ahmed Omara, Burak Kantarci

**Abstract**: As Artificial Intelligence (AI) becomes increasingly integrated into microgrid control systems, the risk of malicious actors exploiting vulnerabilities in Machine Learning (ML) algorithms to disrupt power generation and distribution grows. Detection models to identify adversarial attacks need to meet the constraints of edge environments, where computational power and memory are often limited. To address this issue, we propose a novel strategy that optimizes detection models for Vehicle-to-Microgrid (V2M) edge environments without compromising performance against inference and evasion attacks. Our approach integrates model design and compression into a unified process and results in a highly compact detection model that maintains high accuracy. We evaluated our method against four benchmark evasion attacks-Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), Carlini & Wagner method (C&W) and Conditional Generative Adversarial Network (CGAN) method-and two knowledge-based attacks, white-box and gray-box. Our optimized model reduces memory usage from 20MB to 1.3MB, inference time from 3.2 seconds to 0.9 seconds, and GPU utilization from 5% to 2.68%.



## **46. Robustness of Proof of Team Sprint (PoTS) Against Attacks: A Simulation-Based Analysis**

cs.DC

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19293v1) [paper-pdf](http://arxiv.org/pdf/2503.19293v1)

**Authors**: Naoki Yonezawa

**Abstract**: This study evaluates the robustness of Proof of Team Sprint (PoTS) against adversarial attacks through simulations, focusing on the attacker win rate and computational efficiency under varying team sizes (\( N \)) and attacker ratios (\( \alpha \)). Our results demonstrate that PoTS effectively reduces an attacker's ability to dominate the consensus process. For instance, when \( \alpha = 0.5 \), the attacker win rate decreases from 50.7\% at \( N = 1 \) to below 0.4\% at \( N = 8 \), effectively neutralizing adversarial influence. Similarly, at \( \alpha = 0.8 \), the attacker win rate drops from 80.47\% at \( N = 1 \) to only 2.79\% at \( N = 16 \). In addition to its strong security properties, PoTS maintains high computational efficiency. We introduce the concept of Normalized Computation Efficiency (NCE) to quantify this efficiency gain, showing that PoTS significantly improves resource utilization as team size increases. The results indicate that as \( N \) grows, PoTS not only enhances security but also achieves better computational efficiency due to the averaging effects of execution time variations. These findings highlight PoTS as a promising alternative to traditional consensus mechanisms, offering both robust security and efficient resource utilization. By leveraging team-based block generation and randomized participant reassignment, PoTS provides a scalable and resilient approach to decentralized consensus.



## **47. Activation Functions Considered Harmful: Recovering Neural Network Weights through Controlled Channels**

cs.CR

17 pages, 5 figures

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.19142v1) [paper-pdf](http://arxiv.org/pdf/2503.19142v1)

**Authors**: Jesse Spielman, David Oswald, Mark Ryan, Jo Van Bulck

**Abstract**: With high-stakes machine learning applications increasingly moving to untrusted end-user or cloud environments, safeguarding pre-trained model parameters becomes essential for protecting intellectual property and user privacy. Recent advancements in hardware-isolated enclaves, notably Intel SGX, hold the promise to secure the internal state of machine learning applications even against compromised operating systems. However, we show that privileged software adversaries can exploit input-dependent memory access patterns in common neural network activation functions to extract secret weights and biases from an SGX enclave.   Our attack leverages the SGX-Step framework to obtain a noise-free, instruction-granular page-access trace. In a case study of an 11-input regression network using the Tensorflow Microlite library, we demonstrate complete recovery of all first-layer weights and biases, as well as partial recovery of parameters from deeper layers under specific conditions. Our novel attack technique requires only 20 queries per input per weight to obtain all first-layer weights and biases with an average absolute error of less than 1%, improving over prior model stealing attacks.   Additionally, a broader ecosystem analysis reveals the widespread use of activation functions with input-dependent memory access patterns in popular machine learning frameworks (either directly or via underlying math libraries). Our findings highlight the limitations of deploying confidential models in SGX enclaves and emphasise the need for stricter side-channel validation of machine learning implementations, akin to the vetting efforts applied to secure cryptographic libraries.



## **48. Masks and Mimicry: Strategic Obfuscation and Impersonation Attacks on Authorship Verification**

cs.CL

Accepted at NLP4DH Workshop @ NAACL 2025

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.19099v1) [paper-pdf](http://arxiv.org/pdf/2503.19099v1)

**Authors**: Kenneth Alperin, Rohan Leekha, Adaku Uchendu, Trang Nguyen, Srilakshmi Medarametla, Carlos Levya Capote, Seth Aycock, Charlie Dagli

**Abstract**: The increasing use of Artificial Intelligence (AI) technologies, such as Large Language Models (LLMs) has led to nontrivial improvements in various tasks, including accurate authorship identification of documents. However, while LLMs improve such defense techniques, they also simultaneously provide a vehicle for malicious actors to launch new attack vectors. To combat this security risk, we evaluate the adversarial robustness of authorship models (specifically an authorship verification model) to potent LLM-based attacks. These attacks include untargeted methods - \textit{authorship obfuscation} and targeted methods - \textit{authorship impersonation}. For both attacks, the objective is to mask or mimic the writing style of an author while preserving the original texts' semantics, respectively. Thus, we perturb an accurate authorship verification model, and achieve maximum attack success rates of 92\% and 78\% for both obfuscation and impersonation attacks, respectively.



## **49. Quantum Byzantine Multiple Access Channels**

cs.IT

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2502.12047v2) [paper-pdf](http://arxiv.org/pdf/2502.12047v2)

**Authors**: Minglai Cai, Christian Deppe

**Abstract**: In communication theory, attacks like eavesdropping or jamming are typically assumed to occur at the channel level, while communication parties are expected to follow established protocols. But what happens if one of the parties turns malicious? In this work, we investigate a compelling scenario: a multiple-access channel with two transmitters and one receiver, where one transmitter deviates from the protocol and acts dishonestly. To address this challenge, we introduce the Byzantine multiple-access classical-quantum channel and derive an achievable communication rate for this adversarial setting.



## **50. MF-CLIP: Leveraging CLIP as Surrogate Models for No-box Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2307.06608v3) [paper-pdf](http://arxiv.org/pdf/2307.06608v3)

**Authors**: Jiaming Zhang, Lingyu Qiu, Qi Yi, Yige Li, Jitao Sang, Changsheng Xu, Dit-Yan Yeung

**Abstract**: The vulnerability of Deep Neural Networks (DNNs) to adversarial attacks poses a significant challenge to their deployment in safety-critical applications. While extensive research has addressed various attack scenarios, the no-box attack setting where adversaries have no prior knowledge, including access to training data of the target model, remains relatively underexplored despite its practical relevance. This work presents a systematic investigation into leveraging large-scale Vision-Language Models (VLMs), particularly CLIP, as surrogate models for executing no-box attacks. Our theoretical and empirical analyses reveal a key limitation in the execution of no-box attacks stemming from insufficient discriminative capabilities for direct application of vanilla CLIP as a surrogate model. To address this limitation, we propose MF-CLIP: a novel framework that enhances CLIP's effectiveness as a surrogate model through margin-aware feature space optimization. Comprehensive evaluations across diverse architectures and datasets demonstrate that MF-CLIP substantially advances the state-of-the-art in no-box attacks, surpassing existing baselines by 15.23% on standard models and achieving a 9.52% improvement on adversarially trained models. Our code will be made publicly available to facilitate reproducibility and future research in this direction.



