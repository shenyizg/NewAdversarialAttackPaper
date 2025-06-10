# Latest Adversarial Attack Papers
**update at 2025-06-10 15:44:01**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Adversarial Attack Classification and Robustness Testing for Large Language Models for Code**

cs.SE

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07942v1) [paper-pdf](http://arxiv.org/pdf/2506.07942v1)

**Authors**: Yang Liu, Armstrong Foundjem, Foutse Khomh, Heng Li

**Abstract**: Large Language Models (LLMs) have become vital tools in software development tasks such as code generation, completion, and analysis. As their integration into workflows deepens, ensuring robustness against vulnerabilities especially those triggered by diverse or adversarial inputs becomes increasingly important. Such vulnerabilities may lead to incorrect or insecure code generation when models encounter perturbed task descriptions, code, or comments. Prior research often overlooks the role of natural language in guiding code tasks. This study investigates how adversarial perturbations in natural language inputs including prompts, comments, and descriptions affect LLMs for Code (LLM4Code). It examines the effects of perturbations at the character, word, and sentence levels to identify the most impactful vulnerabilities. We analyzed multiple projects (e.g., ReCode, OpenAttack) and datasets (e.g., HumanEval, MBPP), establishing a taxonomy of adversarial attacks. The first dimension classifies the input type code, prompts, or comments while the second dimension focuses on granularity: character, word, or sentence-level changes. We adopted a mixed-methods approach, combining quantitative performance metrics with qualitative vulnerability analysis. LLM4Code models show varying robustness across perturbation types. Sentence-level attacks were least effective, suggesting models are resilient to broader contextual changes. In contrast, word-level perturbations posed serious challenges, exposing semantic vulnerabilities. Character-level effects varied, showing model sensitivity to subtle syntactic deviations.Our study offers a structured framework for testing LLM4Code robustness and emphasizes the critical role of natural language in adversarial evaluation. Improving model resilience to semantic-level disruptions is essential for secure and reliable code-generation systems.



## **2. CAPAA: Classifier-Agnostic Projector-Based Adversarial Attack**

cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.00978v2) [paper-pdf](http://arxiv.org/pdf/2506.00978v2)

**Authors**: Zhan Li, Mingyu Zhao, Xin Dong, Haibin Ling, Bingyao Huang

**Abstract**: Projector-based adversarial attack aims to project carefully designed light patterns (i.e., adversarial projections) onto scenes to deceive deep image classifiers. It has potential applications in privacy protection and the development of more robust classifiers. However, existing approaches primarily focus on individual classifiers and fixed camera poses, often neglecting the complexities of multi-classifier systems and scenarios with varying camera poses. This limitation reduces their effectiveness when introducing new classifiers or camera poses. In this paper, we introduce Classifier-Agnostic Projector-Based Adversarial Attack (CAPAA) to address these issues. First, we develop a novel classifier-agnostic adversarial loss and optimization framework that aggregates adversarial and stealthiness loss gradients from multiple classifiers. Then, we propose an attention-based gradient weighting mechanism that concentrates perturbations on regions of high classification activation, thereby improving the robustness of adversarial projections when applied to scenes with varying camera poses. Our extensive experimental evaluations demonstrate that CAPAA achieves both a higher attack success rate and greater stealthiness compared to existing baselines. Codes are available at: https://github.com/ZhanLiQxQ/CAPAA.



## **3. Enhancing Adversarial Robustness with Conformal Prediction: A Framework for Guaranteed Model Reliability**

cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07804v1) [paper-pdf](http://arxiv.org/pdf/2506.07804v1)

**Authors**: Jie Bao, Chuangyin Dang, Rui Luo, Hanwei Zhang, Zhixin Zhou

**Abstract**: As deep learning models are increasingly deployed in high-risk applications, robust defenses against adversarial attacks and reliable performance guarantees become paramount. Moreover, accuracy alone does not provide sufficient assurance or reliable uncertainty estimates for these models. This study advances adversarial training by leveraging principles from Conformal Prediction. Specifically, we develop an adversarial attack method, termed OPSA (OPtimal Size Attack), designed to reduce the efficiency of conformal prediction at any significance level by maximizing model uncertainty without requiring coverage guarantees. Correspondingly, we introduce OPSA-AT (Adversarial Training), a defense strategy that integrates OPSA within a novel conformal training paradigm. Experimental evaluations demonstrate that our OPSA attack method induces greater uncertainty compared to baseline approaches for various defenses. Conversely, our OPSA-AT defensive model significantly enhances robustness not only against OPSA but also other adversarial attacks, and maintains reliable prediction. Our findings highlight the effectiveness of this integrated approach for developing trustworthy and resilient deep learning models for safety-critical domains. Our code is available at https://github.com/bjbbbb/Enhancing-Adversarial-Robustness-with-Conformal-Prediction.



## **4. Trial and Trust: Addressing Byzantine Attacks with Comprehensive Defense Strategy**

cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2505.07614v2) [paper-pdf](http://arxiv.org/pdf/2505.07614v2)

**Authors**: Gleb Molodtsov, Daniil Medyakov, Sergey Skorik, Nikolas Khachaturov, Shahane Tigranyan, Vladimir Aletov, Aram Avetisyan, Martin Takáč, Aleksandr Beznosikov

**Abstract**: Recent advancements in machine learning have improved performance while also increasing computational demands. While federated and distributed setups address these issues, their structure is vulnerable to malicious influences. In this paper, we address a specific threat, Byzantine attacks, where compromised clients inject adversarial updates to derail global convergence. We combine the trust scores concept with trial function methodology to dynamically filter outliers. Our methods address the critical limitations of previous approaches, allowing functionality even when Byzantine nodes are in the majority. Moreover, our algorithms adapt to widely used scaled methods like Adam and RMSProp, as well as practical scenarios, including local training and partial participation. We validate the robustness of our methods by conducting extensive experiments on both synthetic and real ECG data collected from medical institutions. Furthermore, we provide a broad theoretical analysis of our algorithms and their extensions to aforementioned practical setups. The convergence guarantees of our methods are comparable to those of classical algorithms developed without Byzantine interference.



## **5. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

cs.AI

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07736v1) [paper-pdf](http://arxiv.org/pdf/2506.07736v1)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.



## **6. Representation Bending for Large Language Model Safety**

cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2504.01550v2) [paper-pdf](http://arxiv.org/pdf/2504.01550v2)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.



## **7. EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

cs.CL

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2505.17654v2) [paper-pdf](http://arxiv.org/pdf/2505.17654v2)

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.



## **8. ProARD: progressive adversarial robustness distillation: provide wide range of robust students**

cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07666v1) [paper-pdf](http://arxiv.org/pdf/2506.07666v1)

**Authors**: Seyedhamidreza Mousavi, Seyedali Mousavi, Masoud Daneshtalab

**Abstract**: Adversarial Robustness Distillation (ARD) has emerged as an effective method to enhance the robustness of lightweight deep neural networks against adversarial attacks. Current ARD approaches have leveraged a large robust teacher network to train one robust lightweight student. However, due to the diverse range of edge devices and resource constraints, current approaches require training a new student network from scratch to meet specific constraints, leading to substantial computational costs and increased CO2 emissions. This paper proposes Progressive Adversarial Robustness Distillation (ProARD), enabling the efficient one-time training of a dynamic network that supports a diverse range of accurate and robust student networks without requiring retraining. We first make a dynamic deep neural network based on dynamic layers by encompassing variations in width, depth, and expansion in each design stage to support a wide range of architectures. Then, we consider the student network with the largest size as the dynamic teacher network. ProARD trains this dynamic network using a weight-sharing mechanism to jointly optimize the dynamic teacher network and its internal student networks. However, due to the high computational cost of calculating exact gradients for all the students within the dynamic network, a sampling mechanism is required to select a subset of students. We show that random student sampling in each iteration fails to produce accurate and robust students.



## **9. Feature Statistics with Uncertainty Help Adversarial Robustness**

cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2503.20583v2) [paper-pdf](http://arxiv.org/pdf/2503.20583v2)

**Authors**: Ran Wang, Xinlei Zhou, Meng Hu, Rihao Li, Wenhui Wu, Yuheng Jia

**Abstract**: Despite the remarkable success of deep neural networks (DNNs), the security threat of adversarial attacks poses a significant challenge to the reliability of DNNs. In this paper, both theoretically and empirically, we discover a universal phenomenon that has been neglected in previous works, i.e., adversarial attacks tend to shift the distributions of feature statistics. Motivated by this finding, and by leveraging the advantages of uncertainty-aware stochastic methods in building robust models efficiently, we propose an uncertainty-driven feature statistics adjustment module for robustness enhancement, named Feature Statistics with Uncertainty (FSU). It randomly resamples channel-wise feature means and standard deviations of examples from multivariate Gaussian distributions, which helps to reconstruct the perturbed examples and calibrate the shifted distributions. The calibration recovers some domain characteristics of the data for classification, thereby mitigating the influence of perturbations and weakening the ability of attacks to deceive models. The proposed FSU module has universal applicability in training, attacking, predicting, and fine-tuning, demonstrating impressive robustness enhancement ability at a trivial additional time cost. For example, by fine-tuning the well-established models with FSU, the state-of-the-art methods achieve up to 17.13% and 34.82% robustness improvement against powerful AA and CW attacks on benchmark datasets.



## **10. RAID: A Dataset for Testing the Adversarial Robustness of AI-Generated Image Detectors**

cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.03988v3) [paper-pdf](http://arxiv.org/pdf/2506.03988v3)

**Authors**: Hicham Eddoubi, Jonas Ricker, Federico Cocchi, Lorenzo Baraldi, Angelo Sotgiu, Maura Pintor, Marcella Cornia, Lorenzo Baraldi, Asja Fischer, Rita Cucchiara, Battista Biggio

**Abstract**: AI-generated images have reached a quality level at which humans are incapable of reliably distinguishing them from real images. To counteract the inherent risk of fraud and disinformation, the detection of AI-generated images is a pressing challenge and an active research topic. While many of the presented methods claim to achieve high detection accuracy, they are usually evaluated under idealized conditions. In particular, the adversarial robustness is often neglected, potentially due to a lack of awareness or the substantial effort required to conduct a comprehensive robustness analysis. In this work, we tackle this problem by providing a simpler means to assess the robustness of AI-generated image detectors. We present RAID (Robust evaluation of AI-generated image Detectors), a dataset of 72k diverse and highly transferable adversarial examples. The dataset is created by running attacks against an ensemble of seven state-of-the-art detectors and images generated by four different text-to-image models. Extensive experiments show that our methodology generates adversarial images that transfer with a high success rate to unseen detectors, which can be used to quickly provide an approximate yet still reliable estimate of a detector's adversarial robustness. Our findings indicate that current state-of-the-art AI-generated image detectors can be easily deceived by adversarial examples, highlighting the critical need for the development of more robust methods. We release our dataset at https://huggingface.co/datasets/aimagelab/RAID and evaluation code at https://github.com/pralab/RAID.



## **11. Explore the vulnerability of black-box models via diffusion models**

cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07590v1) [paper-pdf](http://arxiv.org/pdf/2506.07590v1)

**Authors**: Jiacheng Shi, Yanfu Zhang, Huajie Shao, Ashley Gao

**Abstract**: Recent advancements in diffusion models have enabled high-fidelity and photorealistic image generation across diverse applications. However, these models also present security and privacy risks, including copyright violations, sensitive information leakage, and the creation of harmful or offensive content that could be exploited maliciously. In this study, we uncover a novel security threat where an attacker leverages diffusion model APIs to generate synthetic images, which are then used to train a high-performing substitute model. This enables the attacker to execute model extraction and transfer-based adversarial attacks on black-box classification models with minimal queries, without needing access to the original training data. The generated images are sufficiently high-resolution and diverse to train a substitute model whose outputs closely match those of the target model. Across the seven benchmarks, including CIFAR and ImageNet subsets, our method shows an average improvement of 27.37% over state-of-the-art methods while using just 0.01 times of the query budget, achieving a 98.68% success rate in adversarial attacks on the target model.



## **12. MalGEN: A Generative Agent Framework for Modeling Malicious Software in Cybersecurity**

cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07586v1) [paper-pdf](http://arxiv.org/pdf/2506.07586v1)

**Authors**: Bikash Saha, Sandeep Kumar Shukla

**Abstract**: The dual use nature of Large Language Models (LLMs) presents a growing challenge in cybersecurity. While LLM enhances automation and reasoning for defenders, they also introduce new risks, particularly their potential to be misused for generating evasive, AI crafted malware. Despite this emerging threat, the research community currently lacks controlled and extensible tools that can simulate such behavior for testing and defense preparation. We present MalGEN, a multi agent framework that simulates coordinated adversarial behavior to generate diverse, activity driven malware samples. The agents work collaboratively to emulate attacker workflows, including payload planning, capability selection, and evasion strategies, within a controlled environment built for ethical and defensive research. Using MalGEN, we synthesized ten novel malware samples and evaluated them against leading antivirus and behavioral detection engines. Several samples exhibited stealthy and evasive characteristics that bypassed current defenses, validating MalGEN's ability to model sophisticated and new threats. By transforming the threat of LLM misuse into an opportunity for proactive defense, MalGEN offers a valuable framework for evaluating and strengthening cybersecurity systems. The framework addresses data scarcity, enables rigorous testing, and supports the development of resilient and future ready detection strategies.



## **13. HSF: Defending against Jailbreak Attacks with Hidden State Filtering**

cs.CR

WWW2025 WSAI BESTPAPER

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2409.03788v2) [paper-pdf](http://arxiv.org/pdf/2409.03788v2)

**Authors**: Cheng Qian, Hainan Zhang, Lei Sha, Zhiming Zheng

**Abstract**: With the growing deployment of LLMs in daily applications like chatbots and content generation, efforts to ensure outputs align with human values and avoid harmful content have intensified. However, increasingly sophisticated jailbreak attacks threaten this alignment, aiming to induce unsafe outputs. Current defense efforts either focus on prompt rewriting or detection, which are limited in effectiveness due to the various design of jailbreak prompts, or on output control and detection, which are computationally expensive as they require LLM inference. Therefore, designing a pre-inference defense method that resists diverse jailbreak prompts is crucial for preventing LLM jailbreak attacks. We observe that jailbreak attacks, safe queries, and harmful queries exhibit different clustering patterns within the LLM's hidden state representation space. This suggests that by leveraging the LLM's hidden state representational capabilities, we can analyze the LLM's forthcoming behavior and proactively intervene for defense. In this paper, we propose a jailbreak attack defense strategy based on a Hidden State Filter (HSF), a lossless architectural defense mechanism that enables the model to preemptively identify and reject adversarial inputs before the inference process begins. We activate its defensive potential through an additional plugin module, effectively framing the defense task as a classification problem. Experimental results on two benchmark datasets, utilizing three different LLMs, show that HSF significantly enhances resilience against six cutting-edge jailbreak attacks. It significantly reduces the success rate of jailbreak attacks while minimally impacting responses to benign user queries, with negligible inference overhead, and outperforming defense baselines.Our code and data are available at https://anonymous.4open.science/r/Hidden-State-Filtering-8652/



## **14. Attacking Attention of Foundation Models Disrupts Downstream Tasks**

cs.CR

Paper published at CVPR 2025 Workshop Advml

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.05394v2) [paper-pdf](http://arxiv.org/pdf/2506.05394v2)

**Authors**: Hondamunige Prasanna Silva, Federico Becattini, Lorenzo Seidenari

**Abstract**: Foundation models represent the most prominent and recent paradigm shift in artificial intelligence. Foundation models are large models, trained on broad data that deliver high accuracy in many downstream tasks, often without fine-tuning. For this reason, models such as CLIP , DINO or Vision Transfomers (ViT), are becoming the bedrock of many industrial AI-powered applications. However, the reliance on pre-trained foundation models also introduces significant security concerns, as these models are vulnerable to adversarial attacks. Such attacks involve deliberately crafted inputs designed to deceive AI systems, jeopardizing their reliability. This paper studies the vulnerabilities of vision foundation models, focusing specifically on CLIP and ViTs, and explores the transferability of adversarial attacks to downstream tasks. We introduce a novel attack, targeting the structure of transformer-based architectures in a task-agnostic fashion. We demonstrate the effectiveness of our attack on several downstream tasks: classification, captioning, image/text retrieval, segmentation and depth estimation. Code available at:https://github.com/HondamunigePrasannaSilva/attack-attention



## **15. Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models**

cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07468v1) [paper-pdf](http://arxiv.org/pdf/2506.07468v1)

**Authors**: Mickel Liu, Liwei Jiang, Yancheng Liang, Simon Shaolei Du, Yejin Choi, Tim Althoff, Natasha Jaques

**Abstract**: Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL).



## **16. A Red Teaming Roadmap Towards System-Level Safety**

cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.05376v2) [paper-pdf](http://arxiv.org/pdf/2506.05376v2)

**Authors**: Zifan Wang, Christina Q. Knight, Jeremy Kritz, Willow E. Primack, Julian Michael

**Abstract**: Large Language Model (LLM) safeguards, which implement request refusals, have become a widely adopted mitigation strategy against misuse. At the intersection of adversarial machine learning and AI safety, safeguard red teaming has effectively identified critical vulnerabilities in state-of-the-art refusal-trained LLMs. However, in our view the many conference submissions on LLM red teaming do not, in aggregate, prioritize the right research problems. First, testing against clear product safety specifications should take a higher priority than abstract social biases or ethical principles. Second, red teaming should prioritize realistic threat models that represent the expanding risk landscape and what real attackers might do. Finally, we contend that system-level safety is a necessary step to move red teaming research forward, as AI models present new threats as well as affordances for threat mitigation (e.g., detection and banning of malicious users) once placed in a deployment context. Adopting these priorities will be necessary in order for red teaming research to adequately address the slate of new threats that rapid AI advances present today and will present in the very near future.



## **17. Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models**

cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2502.20650v3) [paper-pdf](http://arxiv.org/pdf/2502.20650v3)

**Authors**: Yu Pan, Jiahao Chen, Bingrong Dai, Lin Wang, Yi Du, Jiao Liu

**Abstract**: In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.



## **18. On the Impact of Uncertainty and Calibration on Likelihood-Ratio Membership Inference Attacks**

cs.IT

16 pages, 28 figures

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2402.10686v5) [paper-pdf](http://arxiv.org/pdf/2402.10686v5)

**Authors**: Meiyi Zhu, Caili Guo, Chunyan Feng, Osvaldo Simeone

**Abstract**: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in which an adaptive prediction set is produced as in conformal prediction. We derive bounds on the advantage of an MIA adversary with the aim of offering insights into the impact of uncertainty and calibration on the effectiveness of MIAs. Simulation results demonstrate that the derived analytical bounds predict well the effectiveness of MIAs.



## **19. Defending Against Diverse Attacks in Federated Learning Through Consensus-Based Bi-Level Optimization**

cs.LG

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2412.02535v2) [paper-pdf](http://arxiv.org/pdf/2412.02535v2)

**Authors**: Nicolás García Trillos, Aditya Kumar Akash, Sixu Li, Konstantin Riedl, Yuhua Zhu

**Abstract**: Adversarial attacks pose significant challenges in many machine learning applications, particularly in the setting of distributed training and federated learning, where malicious agents seek to corrupt the training process with the goal of jeopardizing and compromising the performance and reliability of the final models. In this paper, we address the problem of robust federated learning in the presence of such attacks by formulating the training task as a bi-level optimization problem. We conduct a theoretical analysis of the resilience of consensus-based bi-level optimization (CB$^2$O), an interacting multi-particle metaheuristic optimization method, in adversarial settings. Specifically, we provide a global convergence analysis of CB$^2$O in mean-field law in the presence of malicious agents, demonstrating the robustness of CB$^2$O against a diverse range of attacks. Thereby, we offer insights into how specific hyperparameter choices enable to mitigate adversarial effects. On the practical side, we extend CB$^2$O to the clustered federated learning setting by proposing FedCB$^2$O, a novel interacting multi-particle system, and design a practical algorithm that addresses the demands of real-world applications. Extensive experiments demonstrate the robustness of the FedCB$^2$O algorithm against label-flipping attacks in decentralized clustered federated learning scenarios, showcasing its effectiveness in practical contexts.



## **20. Backdoor Attack on Vision Language Models with Stealthy Semantic Manipulation**

cs.CV

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07214v1) [paper-pdf](http://arxiv.org/pdf/2506.07214v1)

**Authors**: Zhiyuan Zhong, Zhen Sun, Yepang Liu, Xinlei He, Guanhong Tao

**Abstract**: Vision Language Models (VLMs) have shown remarkable performance, but are also vulnerable to backdoor attacks whereby the adversary can manipulate the model's outputs through hidden triggers. Prior attacks primarily rely on single-modality triggers, leaving the crucial cross-modal fusion nature of VLMs largely unexplored. Unlike prior work, we identify a novel attack surface that leverages cross-modal semantic mismatches as implicit triggers. Based on this insight, we propose BadSem (Backdoor Attack with Semantic Manipulation), a data poisoning attack that injects stealthy backdoors by deliberately misaligning image-text pairs during training. To perform the attack, we construct SIMBad, a dataset tailored for semantic manipulation involving color and object attributes. Extensive experiments across four widely used VLMs show that BadSem achieves over 98% average ASR, generalizes well to out-of-distribution datasets, and can transfer across poisoning modalities. Our detailed analysis using attention visualization shows that backdoored models focus on semantically sensitive regions under mismatched conditions while maintaining normal behavior on clean inputs. To mitigate the attack, we try two defense strategies based on system prompt and supervised fine-tuning but find that both of them fail to mitigate the semantic backdoor. Our findings highlight the urgent need to address semantic vulnerabilities in VLMs for their safer deployment.



## **21. Quality-Diversity Red-Teaming: Automated Generation of High-Quality and Diverse Attackers for Large Language Models**

cs.LG

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07121v1) [paper-pdf](http://arxiv.org/pdf/2506.07121v1)

**Authors**: Ren-Jian Wang, Ke Xue, Zeyu Qin, Ziniu Li, Sheng Tang, Hao-Tian Li, Shengcai Liu, Chao Qian

**Abstract**: Ensuring safety of large language models (LLMs) is important. Red teaming--a systematic approach to identifying adversarial prompts that elicit harmful responses from target LLMs--has emerged as a crucial safety evaluation method. Within this framework, the diversity of adversarial prompts is essential for comprehensive safety assessments. We find that previous approaches to red-teaming may suffer from two key limitations. First, they often pursue diversity through simplistic metrics like word frequency or sentence embedding similarity, which may not capture meaningful variation in attack strategies. Second, the common practice of training a single attacker model restricts coverage across potential attack styles and risk categories. This paper introduces Quality-Diversity Red-Teaming (QDRT), a new framework designed to address these limitations. QDRT achieves goal-driven diversity through behavior-conditioned training and implements a behavioral replay buffer in an open-ended manner. Additionally, it trains multiple specialized attackers capable of generating high-quality attacks across diverse styles and risk categories. Our empirical evaluation demonstrates that QDRT generates attacks that are both more diverse and more effective against a wide range of target LLMs, including GPT-2, Llama-3, Gemma-2, and Qwen2.5. This work advances the field of LLM safety by providing a systematic and effective approach to automated red-teaming, ultimately supporting the responsible deployment of LLMs.



## **22. D2R: dual regularization loss with collaborative adversarial generation for model robustness**

cs.CV

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07056v1) [paper-pdf](http://arxiv.org/pdf/2506.07056v1)

**Authors**: Zhenyu Liu, Huizhi Liang, Rajiv Ranjan, Zhanxing Zhu, Vaclav Snasel, Varun Ojha

**Abstract**: The robustness of Deep Neural Network models is crucial for defending models against adversarial attacks. Recent defense methods have employed collaborative learning frameworks to enhance model robustness. Two key limitations of existing methods are (i) insufficient guidance of the target model via loss functions and (ii) non-collaborative adversarial generation. We, therefore, propose a dual regularization loss (D2R Loss) method and a collaborative adversarial generation (CAG) strategy for adversarial training. D2R loss includes two optimization steps. The adversarial distribution and clean distribution optimizations enhance the target model's robustness by leveraging the strengths of different loss functions obtained via a suitable function space exploration to focus more precisely on the target model's distribution. CAG generates adversarial samples using a gradient-based collaboration between guidance and target models. We conducted extensive experiments on three benchmark databases, including CIFAR-10, CIFAR-100, Tiny ImageNet, and two popular target models, WideResNet34-10 and PreActResNet18. Our results show that D2R loss with CAG produces highly robust models.



## **23. AHSG: Adversarial Attack on High-level Semantics in Graph Neural Networks**

cs.LG

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2412.07468v3) [paper-pdf](http://arxiv.org/pdf/2412.07468v3)

**Authors**: Kai Yuan, Jiahao Zhang, Yidi Wang, Xiaobing Pei

**Abstract**: Adversarial attacks on Graph Neural Networks aim to perturb the performance of the learner by carefully modifying the graph topology and node attributes. Existing methods achieve attack stealthiness by constraining the modification budget and differences in graph properties. However, these methods typically disrupt task-relevant primary semantics directly, which results in low defensibility and detectability of the attack. In this paper, we propose an Adversarial Attack on High-level Semantics for Graph Neural Networks (AHSG), which is a graph structure attack model that ensures the retention of primary semantics. By combining latent representations with shared primary semantics, our model retains detectable attributes and relational patterns of the original graph while leveraging more subtle changes to carry out the attack. Then we use the Projected Gradient Descent algorithm to map the latent representations with attack effects to the adversarial graph. Through experiments on robust graph deep learning models equipped with defense strategies, we demonstrate that AHSG outperforms other state-of-the-art methods in attack effectiveness. Additionally, using Contextual Stochastic Block Models to detect the attacked graph further validates that our method preserves the primary semantics of the graph.



## **24. NanoZone: Scalable, Efficient, and Secure Memory Protection for Arm CCA**

cs.CR

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07034v1) [paper-pdf](http://arxiv.org/pdf/2506.07034v1)

**Authors**: Shiqi Liu, Yongpeng Gao, Mingyang Zhang, Jie Wang

**Abstract**: Arm Confidential Computing Architecture (CCA) currently isolates at the granularity of an entire Confidential Virtual Machine (CVM), leaving intra-VM bugs such as Heartbleed unmitigated. The state-of-the-art narrows this to the process level, yet still cannot stop attacks that pivot within the same process, and prior intra-enclave schemes are either too slow or incompatible with CVM-style isolation. We extend CCA with a three-tier zone model that spawns an unlimited number of lightweight isolation domains inside a single process, while shielding them from kernel-space adversaries. To block domain-switch abuse, we also add a fast user-level Code-Pointer Integrity (CPI) mechanism. We developed two prototypes: a functional version on Arm's official simulator to validate resistance against intra-process and kernel-space adversaries, and a performance variant on Arm development boards evaluated for session-key isolation within server applications, in-memory key-value protection, and non-volatile-memory data isolation. NanoZone incurs roughly a 20% performance overhead while retaining 95% throughput compared to the system without fine-grained isolation.



## **25. Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text**

cs.CL

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07001v1) [paper-pdf](http://arxiv.org/pdf/2506.07001v1)

**Authors**: Yize Cheng, Vinu Sankar Sadasivan, Mehrdad Saberi, Shoumik Saha, Soheil Feizi

**Abstract**: The increasing capabilities of Large Language Models (LLMs) have raised concerns about their misuse in AI-generated plagiarism and social engineering. While various AI-generated text detectors have been proposed to mitigate these risks, many remain vulnerable to simple evasion techniques such as paraphrasing. However, recent detectors have shown greater robustness against such basic attacks. In this work, we introduce Adversarial Paraphrasing, a training-free attack framework that universally humanizes any AI-generated text to evade detection more effectively. Our approach leverages an off-the-shelf instruction-following LLM to paraphrase AI-generated content under the guidance of an AI text detector, producing adversarial examples that are specifically optimized to bypass detection. Extensive experiments show that our attack is both broadly effective and highly transferable across several detection systems. For instance, compared to simple paraphrasing attack--which, ironically, increases the true positive at 1% false positive (T@1%F) by 8.57% on RADAR and 15.03% on Fast-DetectGPT--adversarial paraphrasing, guided by OpenAI-RoBERTa-Large, reduces T@1%F by 64.49% on RADAR and a striking 98.96% on Fast-DetectGPT. Across a diverse set of detectors--including neural network-based, watermark-based, and zero-shot approaches--our attack achieves an average T@1%F reduction of 87.88% under the guidance of OpenAI-RoBERTa-Large. We also analyze the tradeoff between text quality and attack success to find that our method can significantly reduce detection rates, with mostly a slight degradation in text quality. Our adversarial setup highlights the need for more robust and resilient detection strategies in the light of increasingly sophisticated evasion techniques.



## **26. Boosting Adversarial Transferability via Commonality-Oriented Gradient Optimization**

cs.CV

22 pages

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.06992v1) [paper-pdf](http://arxiv.org/pdf/2506.06992v1)

**Authors**: Yanting Gao, Yepeng Liu, Junming Liu, Qi Zhang, Hongyun Zhang, Duoqian Miao, Cairong Zhao

**Abstract**: Exploring effective and transferable adversarial examples is vital for understanding the characteristics and mechanisms of Vision Transformers (ViTs). However, adversarial examples generated from surrogate models often exhibit weak transferability in black-box settings due to overfitting. Existing methods improve transferability by diversifying perturbation inputs or applying uniform gradient regularization within surrogate models, yet they have not fully leveraged the shared and unique features of surrogate models trained on the same task, leading to suboptimal transfer performance. Therefore, enhancing perturbations of common information shared by surrogate models and suppressing those tied to individual characteristics offers an effective way to improve transferability. Accordingly, we propose a commonality-oriented gradient optimization strategy (COGO) consisting of two components: Commonality Enhancement (CE) and Individuality Suppression (IS). CE perturbs the mid-to-low frequency regions, leveraging the fact that ViTs trained on the same dataset tend to rely more on mid-to-low frequency information for classification. IS employs adaptive thresholds to evaluate the correlation between backpropagated gradients and model individuality, assigning weights to gradients accordingly. Extensive experiments demonstrate that COGO significantly improves the transfer success rates of adversarial attacks, outperforming current state-of-the-art methods.



## **27. Rewriting the Budget: A General Framework for Black-Box Attacks Under Cost Asymmetry**

cs.LG

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2506.06933v1) [paper-pdf](http://arxiv.org/pdf/2506.06933v1)

**Authors**: Mahdi Salmani, Alireza Abdollahpoorrostam, Seyed-Mohsen Moosavi-Dezfooli

**Abstract**: Traditional decision-based black-box adversarial attacks on image classifiers aim to generate adversarial examples by slightly modifying input images while keeping the number of queries low, where each query involves sending an input to the model and observing its output. Most existing methods assume that all queries have equal cost. However, in practice, queries may incur asymmetric costs; for example, in content moderation systems, certain output classes may trigger additional review, enforcement, or penalties, making them more costly than others. While prior work has considered such asymmetric cost settings, effective algorithms for this scenario remain underdeveloped. In this paper, we propose a general framework for decision-based attacks under asymmetric query costs, which we refer to as asymmetric black-box attacks. We modify two core components of existing attacks: the search strategy and the gradient estimation process. Specifically, we propose Asymmetric Search (AS), a more conservative variant of binary search that reduces reliance on high-cost queries, and Asymmetric Gradient Estimation (AGREST), which shifts the sampling distribution to favor low-cost queries. We design efficient algorithms that minimize total attack cost by balancing different query types, in contrast to earlier methods such as stealthy attacks that focus only on limiting expensive (high-cost) queries. Our method can be integrated into a range of existing black-box attacks with minimal changes. We perform both theoretical analysis and empirical evaluation on standard image classification benchmarks. Across various cost regimes, our method consistently achieves lower total query cost and smaller perturbations than existing approaches, with improvements of up to 40% in some settings.



## **28. KNN-Defense: Defense against 3D Adversarial Point Clouds using Nearest-Neighbor Search**

cs.CV

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2506.06906v1) [paper-pdf](http://arxiv.org/pdf/2506.06906v1)

**Authors**: Nima Jamali, Matina Mahdizadeh Sani, Hanieh Naderi, Shohreh Kasaei

**Abstract**: Deep neural networks (DNNs) have demonstrated remarkable performance in analyzing 3D point cloud data. However, their vulnerability to adversarial attacks-such as point dropping, shifting, and adding-poses a critical challenge to the reliability of 3D vision systems. These attacks can compromise the semantic and structural integrity of point clouds, rendering many existing defense mechanisms ineffective. To address this issue, a defense strategy named KNN-Defense is proposed, grounded in the manifold assumption and nearest-neighbor search in feature space. Instead of reconstructing surface geometry or enforcing uniform point distributions, the method restores perturbed inputs by leveraging the semantic similarity of neighboring samples from the training set. KNN-Defense is lightweight and computationally efficient, enabling fast inference and making it suitable for real-time and practical applications. Empirical results on the ModelNet40 dataset demonstrated that KNN-Defense significantly improves robustness across various attack types. In particular, under point-dropping attacks-where many existing methods underperform due to the targeted removal of critical points-the proposed method achieves accuracy gains of 20.1%, 3.6%, 3.44%, and 7.74% on PointNet, PointNet++, DGCNN, and PCT, respectively. These findings suggest that KNN-Defense offers a scalable and effective solution for enhancing the adversarial resilience of 3D point cloud classifiers. (An open-source implementation of the method, including code and data, is available at https://github.com/nimajam41/3d-knn-defense).



## **29. Robustifying Vision-Language Models via Dynamic Token Reweighting**

cs.CV

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2505.17132v2) [paper-pdf](http://arxiv.org/pdf/2505.17132v2)

**Authors**: Tanqiu Jiang, Jiacheng Liang, Rongyi Zhu, Jiawei Zhou, Fenglong Ma, Ting Wang

**Abstract**: Large vision-language models (VLMs) are highly vulnerable to jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that \sys outperforms existing defenses in both attack robustness and benign task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. (warning: this paper contains potentially harmful content generated by VLMs.)



## **30. Can In-Context Reinforcement Learning Recover From Reward Poisoning Attacks?**

cs.LG

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2506.06891v1) [paper-pdf](http://arxiv.org/pdf/2506.06891v1)

**Authors**: Paulius Sasnauskas, Yiğit Yalın, Goran Radanović

**Abstract**: We study the corruption-robustness of in-context reinforcement learning (ICRL), focusing on the Decision-Pretrained Transformer (DPT, Lee et al., 2023). To address the challenge of reward poisoning attacks targeting the DPT, we propose a novel adversarial training framework, called Adversarially Trained Decision-Pretrained Transformer (AT-DPT). Our method simultaneously trains an attacker to minimize the true reward of the DPT by poisoning environment rewards, and a DPT model to infer optimal actions from the poisoned data. We evaluate the effectiveness of our approach against standard bandit algorithms, including robust baselines designed to handle reward contamination. Our results show that the proposed method significantly outperforms these baselines in bandit settings, under a learned attacker. We additionally evaluate AT-DPT on an adaptive attacker, and observe similar results. Furthermore, we extend our evaluation to the MDP setting, confirming that the robustness observed in bandit scenarios generalizes to more complex environments.



## **31. Watermark under Fire: A Robustness Evaluation of LLM Watermarking**

cs.CR

22 pages

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2411.13425v3) [paper-pdf](http://arxiv.org/pdf/2411.13425v3)

**Authors**: Jiacheng Liang, Zian Wang, Lauren Hong, Shouling Ji, Ting Wang

**Abstract**: Various watermarking methods (``watermarkers'') have been proposed to identify LLM-generated texts; yet, due to the lack of unified evaluation platforms, many critical questions remain under-explored: i) What are the strengths/limitations of various watermarkers, especially their attack robustness? ii) How do various design choices impact their robustness? iii) How to optimally operate watermarkers in adversarial environments? To fill this gap, we systematize existing LLM watermarkers and watermark removal attacks, mapping out their design spaces. We then develop WaterPark, a unified platform that integrates 10 state-of-the-art watermarkers and 12 representative attacks. More importantly, by leveraging WaterPark, we conduct a comprehensive assessment of existing watermarkers, unveiling the impact of various design choices on their attack robustness. We further explore the best practices to operate watermarkers in adversarial environments. We believe our study sheds light on current LLM watermarking techniques while WaterPark serves as a valuable testbed to facilitate future research.



## **32. LLM-attacker: Enhancing Closed-loop Adversarial Scenario Generation for Autonomous Driving with Large Language Models**

cs.LG

Accepted as a regular paper at IEEE TITS 2025

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2501.15850v2) [paper-pdf](http://arxiv.org/pdf/2501.15850v2)

**Authors**: Yuewen Mei, Tong Nie, Jian Sun, Ye Tian

**Abstract**: Ensuring and improving the safety of autonomous driving systems (ADS) is crucial for the deployment of highly automated vehicles, especially in safety-critical events. To address the rarity issue, adversarial scenario generation methods are developed, in which behaviors of traffic participants are manipulated to induce safety-critical events. However, existing methods still face two limitations. First, identification of the adversarial participant directly impacts the effectiveness of the generation. However, the complexity of real-world scenarios, with numerous participants and diverse behaviors, makes identification challenging. Second, the potential of generated safety-critical scenarios to continuously improve ADS performance remains underexplored. To address these issues, we propose LLM-attacker: a closed-loop adversarial scenario generation framework leveraging large language models (LLMs). Specifically, multiple LLM agents are designed and coordinated to identify optimal attackers. Then, the trajectories of the attackers are optimized to generate adversarial scenarios. These scenarios are iteratively refined based on the performance of ADS, forming a feedback loop to improve ADS. Experimental results show that LLM-attacker can create more dangerous scenarios than other methods, and the ADS trained with it achieves a collision rate half that of training with normal scenarios. This indicates the ability of LLM-attacker to test and enhance the safety and robustness of ADS. Video demonstrations are provided at: https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view.



## **33. Homophily-Driven Sanitation View for Robust Graph Contrastive Learning**

cs.LG

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2307.12555v2) [paper-pdf](http://arxiv.org/pdf/2307.12555v2)

**Authors**: Yulin Zhu, Xing Ai, Yevgeniy Vorobeychik, Kai Zhou

**Abstract**: We investigate adversarial robustness of unsupervised Graph Contrastive Learning (GCL) against structural attacks. First, we provide a comprehensive empirical and theoretical analysis of existing attacks, revealing how and why they downgrade the performance of GCL. Inspired by our analytic results, we present a robust GCL framework that integrates a homophily-driven sanitation view, which can be learned jointly with contrastive learning. A key challenge this poses, however, is the non-differentiable nature of the sanitation objective. To address this challenge, we propose a series of techniques to enable gradient-based end-to-end robust GCL. Moreover, we develop a fully unsupervised hyperparameter tuning method which, unlike prior approaches, does not require knowledge of node labels. We conduct extensive experiments to evaluate the performance of our proposed model, GCHS (Graph Contrastive Learning with Homophily-driven Sanitation View), against two state of the art structural attacks on GCL. Our results demonstrate that GCHS consistently outperforms all state of the art baselines in terms of the quality of generated node embeddings as well as performance on two important downstream tasks.



## **34. Refining Adaptive Zeroth-Order Optimization at Ease**

cs.LG

Published as a conference paper at ICML 2025

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2502.01014v2) [paper-pdf](http://arxiv.org/pdf/2502.01014v2)

**Authors**: Yao Shu, Qixin Zhang, Kun He, Zhongxiang Dai

**Abstract**: Recently, zeroth-order (ZO) optimization plays an essential role in scenarios where gradient information is inaccessible or unaffordable, such as black-box systems and resource-constrained environments. While existing adaptive methods such as ZO-AdaMM have shown promise, they are fundamentally limited by their underutilization of moment information during optimization, usually resulting in underperforming convergence. To overcome these limitations, this paper introduces Refined Adaptive Zeroth-Order Optimization (R-AdaZO). Specifically, we first show the untapped variance reduction effect of first moment estimate on ZO gradient estimation, which improves the accuracy and stability of ZO updates. We then refine the second moment estimate based on these variance-reduced gradient estimates to better capture the geometry of the optimization landscape, enabling a more effective scaling of ZO updates. We present rigorous theoretical analysis to show (a) the first analysis to the variance reduction of first moment estimate in ZO optimization, (b) the improved second moment estimates with a more accurate approximation of its variance-free ideal, (c) the first variance-aware convergence framework for adaptive ZO methods, which may be of independent interest, and (d) the faster convergence of R-AdaZO than existing baselines like ZO-AdaMM. Our extensive experiments, including synthetic problems, black-box adversarial attack, and memory-efficient fine-tuning of large language models (LLMs), further verify the superior convergence of R-AdaZO, indicating that R-AdaZO offers an improved solution for real-world ZO optimization challenges.



## **35. On Adversarial Robustness of Language Models in Transfer Learning**

cs.CL

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2501.00066v2) [paper-pdf](http://arxiv.org/pdf/2501.00066v2)

**Authors**: Bohdan Turbal, Anastasiia Mazur, Jiaxu Zhao, Mykola Pechenizkiy

**Abstract**: We investigate the adversarial robustness of LLMs in transfer learning scenarios. Through comprehensive experiments on multiple datasets (MBIB Hate Speech, MBIB Political Bias, MBIB Gender Bias) and various model architectures (BERT, RoBERTa, GPT-2, Gemma, Phi), we reveal that transfer learning, while improving standard performance metrics, often leads to increased vulnerability to adversarial attacks. Our findings demonstrate that larger models exhibit greater resilience to this phenomenon, suggesting a complex interplay between model size, architecture, and adaptation methods. Our work highlights the crucial need for considering adversarial robustness in transfer learning scenarios and provides insights into maintaining model security without compromising performance. These findings have significant implications for the development and deployment of LLMs in real-world applications where both performance and robustness are paramount.



## **36. RED QUEEN: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking**

cs.CR

Accepted in ACL 2025 Findings

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2409.17458v2) [paper-pdf](http://arxiv.org/pdf/2409.17458v2)

**Authors**: Yifan Jiang, Kriti Aggarwal, Tanmay Laud, Kashif Munir, Jay Pujara, Subhabrata Mukherjee

**Abstract**: The rapid progress of Large Language Models (LLMs) has opened up new opportunities across various domains and applications; yet it also presents challenges related to potential misuse. To mitigate such risks, red teaming has been employed as a proactive security measure to probe language models for harmful outputs via jailbreak attacks. However, current jailbreak attack approaches are single-turn with explicit malicious queries that do not fully capture the complexity of real-world interactions. In reality, users can engage in multi-turn interactions with LLM-based chat assistants, allowing them to conceal their true intentions in a more covert manner. To bridge this gap, we, first, propose a new jailbreak approach, RED QUEEN ATTACK. This method constructs a multi-turn scenario, concealing the malicious intent under the guise of preventing harm. We craft 40 scenarios that vary in turns and select 14 harmful categories to generate 56k multi-turn attack data points. We conduct comprehensive experiments on the RED QUEEN ATTACK with four representative LLM families of different sizes. Our experiments reveal that all LLMs are vulnerable to RED QUEEN ATTACK, reaching 87.62% attack success rate on GPT-4o and 75.4% on Llama3-70B. Further analysis reveals that larger models are more susceptible to the RED QUEEN ATTACK, with multi-turn structures and concealment strategies contributing to its success. To prioritize safety, we introduce a straightforward mitigation strategy called RED QUEEN GUARD, which aligns LLMs to effectively counter adversarial attacks. This approach reduces the attack success rate to below 1% while maintaining the model's performance across standard benchmarks. Full implementation and dataset are publicly accessible at https://github.com/kriti-hippo/red_queen.



## **37. from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors**

cs.CL

arXiv admin note: substantial text overlap with arXiv:2412.12145

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2503.00038v3) [paper-pdf](http://arxiv.org/pdf/2503.00038v3)

**Authors**: Yu Yan, Sheng Sun, Zenghao Duan, Teli Liu, Min Liu, Zhiyi Yin, Jiangyu Lei, Qi Li

**Abstract**: Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs.



## **38. Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks: Theoretical and Empirical Evidence**

cs.LG

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2502.04204v2) [paper-pdf](http://arxiv.org/pdf/2502.04204v2)

**Authors**: Shaopeng Fu, Liang Ding, Jingfeng Zhang, Di Wang

**Abstract**: Jailbreak attacks against large language models (LLMs) aim to induce harmful behaviors in LLMs through carefully crafted adversarial prompts. To mitigate attacks, one way is to perform adversarial training (AT)-based alignment, i.e., training LLMs on some of the most adversarial prompts to help them learn how to behave safely under attacks. During AT, the length of adversarial prompts plays a critical role in the robustness of aligned LLMs. While long-length adversarial prompts during AT might lead to strong LLM robustness, their synthesis however is very resource-consuming, which may limit the application of LLM AT. This paper focuses on adversarial suffix jailbreak attacks and unveils that to defend against a jailbreak attack with an adversarial suffix of length $\Theta(M)$, it is enough to align LLMs on prompts with adversarial suffixes of length $\Theta(\sqrt{M})$. Theoretically, we analyze the adversarial in-context learning of linear transformers on linear regression tasks and prove a robust generalization bound for trained transformers. The bound depends on the term $\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$, where $M_{\text{train}}$ and $M_{\text{test}}$ are the numbers of adversarially perturbed in-context samples during training and testing. Empirically, we conduct AT on popular open-source LLMs and evaluate their robustness against jailbreak attacks of different adversarial suffix lengths. Results confirm a positive correlation between the attack success rate and the ratio of the square root of the adversarial suffix length during jailbreaking to the length during AT. Our findings show that it is practical to defend against ``long-length'' jailbreak attacks via efficient ``short-length'' AT. The code is available at https://github.com/fshp971/adv-icl.



## **39. Stochastic Training for Side-Channel Resilient AI**

cs.CR

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2506.06597v1) [paper-pdf](http://arxiv.org/pdf/2506.06597v1)

**Authors**: Anuj Dubey, Aydin Aysu

**Abstract**: The confidentiality of trained AI models on edge devices is at risk from side-channel attacks exploiting power and electromagnetic emissions. This paper proposes a novel training methodology to enhance resilience against such threats by introducing randomized and interchangeable model configurations during inference. Experimental results on Google Coral Edge TPU show a reduction in side-channel leakage and a slower increase in t-scores over 20,000 traces, demonstrating robustness against adversarial observations. The defense maintains high accuracy, with about 1% degradation in most configurations, and requires no additional hardware or software changes, making it the only applicable solution for existing Edge TPUs.



## **40. Adapting Under Fire: Multi-Agent Reinforcement Learning for Adversarial Drift in Network Security**

cs.CR

In Proceedings of the 22nd International Conference on Security and  Cryptography, ISBN 978-989-758-760-3, ISSN 2184-7711, pages 547-554

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06565v1) [paper-pdf](http://arxiv.org/pdf/2506.06565v1)

**Authors**: Emilia Rivas, Sabrina Saika, Ahtesham Bakht, Aritran Piplai, Nathaniel D. Bastian, Ankit Shah

**Abstract**: Evolving attacks are a critical challenge for the long-term success of Network Intrusion Detection Systems (NIDS). The rise of these changing patterns has exposed the limitations of traditional network security methods. While signature-based methods are used to detect different types of attacks, they often fail to detect unknown attacks. Moreover, the system requires frequent updates with new signatures as the attackers are constantly changing their tactics. In this paper, we design an environment where two agents improve their policies over time. The adversarial agent, referred to as the red agent, perturbs packets to evade the intrusion detection mechanism, whereas the blue agent learns new defensive policies using drift adaptation techniques to counter the attacks. Both agents adapt iteratively: the red agent responds to the evolving NIDS, while the blue agent adjusts to emerging attack patterns. By studying the model's learned policy, we offer concrete insights into drift adaptation techniques with high utility. Experiments show that the blue agent boosts model accuracy by 30% with just 2 to 3 adaptation steps using only 25 to 30 samples each.



## **41. Securing Traffic Sign Recognition Systems in Autonomous Vehicles**

cs.CV

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06563v1) [paper-pdf](http://arxiv.org/pdf/2506.06563v1)

**Authors**: Thushari Hapuarachchi, Long Dang, Kaiqi Xiong

**Abstract**: Deep Neural Networks (DNNs) are widely used for traffic sign recognition because they can automatically extract high-level features from images. These DNNs are trained on large-scale datasets obtained from unknown sources. Therefore, it is important to ensure that the models remain secure and are not compromised or poisoned during training. In this paper, we investigate the robustness of DNNs trained for traffic sign recognition. First, we perform the error-minimizing attacks on DNNs used for traffic sign recognition by adding imperceptible perturbations on training data. Then, we propose a data augmentation-based training method to mitigate the error-minimizing attacks. The proposed training method utilizes nonlinear transformations to disrupt the perturbations and improve the model robustness. We experiment with two well-known traffic sign datasets to demonstrate the severity of the attack and the effectiveness of our mitigation scheme. The error-minimizing attacks reduce the prediction accuracy of the DNNs from 99.90% to 10.6%. However, our mitigation scheme successfully restores the prediction accuracy to 96.05%. Moreover, our approach outperforms adversarial training in mitigating the error-minimizing attacks. Furthermore, we propose a detection model capable of identifying poisoned data even when the perturbations are imperceptible to human inspection. Our detection model achieves a success rate of over 99% in identifying the attack. This research highlights the need to employ advanced training methods for DNNs in traffic sign recognition systems to mitigate the effects of data poisoning attacks.



## **42. SDN-Based False Data Detection With Its Mitigation and Machine Learning Robustness for In-Vehicle Networks**

cs.LG

The 34th International Conference on Computer Communications and  Networks (ICCCN 2025)

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06556v1) [paper-pdf](http://arxiv.org/pdf/2506.06556v1)

**Authors**: Long Dang, Thushari Hapuarachchi, Kaiqi Xiong, Yi Li

**Abstract**: As the development of autonomous and connected vehicles advances, the complexity of modern vehicles increases, with numerous Electronic Control Units (ECUs) integrated into the system. In an in-vehicle network, these ECUs communicate with one another using an standard protocol called Controller Area Network (CAN). Securing communication among ECUs plays a vital role in maintaining the safety and security of the vehicle. This paper proposes a robust SDN-based False Data Detection and Mitigation System (FDDMS) for in-vehicle networks. Leveraging the unique capabilities of Software-Defined Networking (SDN), FDDMS is designed to monitor and detect false data injection attacks in real-time. Specifically, we focus on brake-related ECUs within an SDN-enabled in-vehicle network. First, we decode raw CAN data to create an attack model that illustrates how false data can be injected into the system. Then, FDDMS, incorporating a Long Short Term Memory (LSTM)-based detection model, is used to identify false data injection attacks. We further propose an effective variant of DeepFool attack to evaluate the model's robustness. To countermeasure the impacts of four adversarial attacks including Fast gradient descent method, Basic iterative method, DeepFool, and the DeepFool variant, we further enhance a re-training technique method with a threshold based selection strategy. Finally, a mitigation scheme is implemented to redirect attack traffic by dynamically updating flow rules through SDN. Our experimental results show that the proposed FDDMS is robust against adversarial attacks and effectively detects and mitigates false data injection attacks in real-time.



## **43. JailbreakLens: Visual Analysis of Jailbreak Attacks Against Large Language Models**

cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2404.08793v2) [paper-pdf](http://arxiv.org/pdf/2404.08793v2)

**Authors**: Yingchaojie Feng, Zhizhang Chen, Zhining Kang, Sijia Wang, Haoyu Tian, Wei Zhang, Minfeng Zhu, Wei Chen

**Abstract**: The proliferation of large language models (LLMs) has underscored concerns regarding their security vulnerabilities, notably against jailbreak attacks, where adversaries design jailbreak prompts to circumvent safety mechanisms for potential misuse. Addressing these concerns necessitates a comprehensive analysis of jailbreak prompts to evaluate LLMs' defensive capabilities and identify potential weaknesses. However, the complexity of evaluating jailbreak performance and understanding prompt characteristics makes this analysis laborious. We collaborate with domain experts to characterize problems and propose an LLM-assisted framework to streamline the analysis process. It provides automatic jailbreak assessment to facilitate performance evaluation and support analysis of components and keywords in prompts. Based on the framework, we design JailbreakLens, a visual analysis system that enables users to explore the jailbreak performance against the target model, conduct multi-level analysis of prompt characteristics, and refine prompt instances to verify findings. Through a case study, technical evaluations, and expert interviews, we demonstrate our system's effectiveness in helping users evaluate model security and identify model weaknesses.



## **44. Membership Inference Attacks for Unseen Classes**

cs.LG

Preprint

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06488v1) [paper-pdf](http://arxiv.org/pdf/2506.06488v1)

**Authors**: Pratiksha Thaker, Neil Kale, Zhiwei Steven Wu, Virginia Smith

**Abstract**: Shadow model attacks are the state-of-the-art approach for membership inference attacks on machine learning models. However, these attacks typically assume an adversary has access to a background (nonmember) data distribution that matches the distribution the target model was trained on. We initiate a study of membership inference attacks where the adversary or auditor cannot access an entire subclass from the distribution -- a more extreme but realistic version of distribution shift than has been studied previously. In this setting, we first show that the performance of shadow model attacks degrades catastrophically, and then demonstrate the promise of another approach, quantile regression, that does not have the same limitations. We show that quantile regression attacks consistently outperform shadow model attacks in the class dropout setting -- for example, quantile regression attacks achieve up to 11$\times$ the TPR of shadow models on the unseen class on CIFAR-100, and achieve nontrivial TPR on ImageNet even with 90% of training classes removed. We also provide a theoretical model that illustrates the potential and limitations of this approach.



## **45. Adaptive and Robust Watermark for Generative Tabular Data**

cs.CR

15 pages of main body, 5 figures, 5 tables

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2409.14700v2) [paper-pdf](http://arxiv.org/pdf/2409.14700v2)

**Authors**: Dung Daniel Ngo, Daniel Scott, Saheed Obitayo, Archan Ray, Akshay Seshadri, Niraj Kumar, Vamsi K. Potluru, Marco Pistoia, Manuela Veloso

**Abstract**: Recent development in generative models has demonstrated its ability to create high-quality synthetic data. However, the pervasiveness of synthetic content online also brings forth growing concerns that it can be used for malicious purpose. To ensure the authenticity of the data, watermarking techniques have recently emerged as a promising solution due to their strong statistical guarantees. In this paper, we propose a flexible and robust watermarking mechanism for generative tabular data. Specifically, a data provider with knowledge of the downstream tasks can partition the feature space into pairs of (key, value) columns. Within each pair, the data provider first uses elements in the key column to generate a randomized set of ``green'' intervals, then encourages elements of the value column to be in one of these ``green'' intervals. We show theoretically and empirically that the watermarked datasets (i) have negligible impact on the data quality and downstream utility, (ii) can be efficiently detected, (iii) are robust against multiple attacks commonly observed in data science, and (iv) maintain strong security against adversary attempting to learn the underlying watermark scheme.



## **46. Fréchet Radiomic Distance (FRD): A Versatile Metric for Comparing Medical Imaging Datasets**

cs.CV

Codebase for FRD computation:  https://github.com/RichardObi/frd-score. Codebase for medical image  similarity metric evaluation framework:  https://github.com/mazurowski-lab/medical-image-similarity-metrics

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2412.01496v2) [paper-pdf](http://arxiv.org/pdf/2412.01496v2)

**Authors**: Nicholas Konz, Richard Osuala, Preeti Verma, Yuwen Chen, Hanxue Gu, Haoyu Dong, Yaqian Chen, Andrew Marshall, Lidia Garrucho, Kaisar Kushibar, Daniel M. Lang, Gene S. Kim, Lars J. Grimm, John M. Lewin, James S. Duncan, Julia A. Schnabel, Oliver Diaz, Karim Lekadir, Maciej A. Mazurowski

**Abstract**: Determining whether two sets of images belong to the same or different distributions or domains is a crucial task in modern medical image analysis and deep learning; for example, to evaluate the output quality of image generative models. Currently, metrics used for this task either rely on the (potentially biased) choice of some downstream task, such as segmentation, or adopt task-independent perceptual metrics (e.g., Fr\'echet Inception Distance/FID) from natural imaging, which we show insufficiently capture anatomical features. To this end, we introduce a new perceptual metric tailored for medical images, FRD (Fr\'echet Radiomic Distance), which utilizes standardized, clinically meaningful, and interpretable image features. We show that FRD is superior to other image distribution metrics for a range of medical imaging applications, including out-of-domain (OOD) detection, the evaluation of image-to-image translation (by correlating more with downstream task performance as well as anatomical consistency and realism), and the evaluation of unconditional image generation. Moreover, FRD offers additional benefits such as stability and computational efficiency at low sample sizes, sensitivity to image corruptions and adversarial attacks, feature interpretability, and correlation with radiologist-perceived image quality. Additionally, we address key gaps in the literature by presenting an extensive framework for the multifaceted evaluation of image similarity metrics in medical imaging -- including the first large-scale comparative study of generative models for medical image translation -- and release an accessible codebase to facilitate future research. Our results are supported by thorough experiments spanning a variety of datasets, modalities, and downstream tasks, highlighting the broad potential of FRD for medical image analysis.



## **47. ByzSecAgg: A Byzantine-Resistant Secure Aggregation Scheme for Federated Learning Based on Coded Computing and Vector Commitment**

cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2302.09913v4) [paper-pdf](http://arxiv.org/pdf/2302.09913v4)

**Authors**: Tayyebeh Jahani-Nezhad, Mohammad Ali Maddah-Ali, Giuseppe Caire

**Abstract**: In this paper, we propose ByzSecAgg, an efficient secure aggregation scheme for federated learning that is resistant to Byzantine attacks and privacy leakages. Processing individual updates to manage adversarial behavior, while preserving the privacy of the data against colluding nodes, requires some sort of secure secret sharing. However, the communication load for secret sharing of long vectors of updates can be very high. In federated settings, where users are often edge devices with potential bandwidth constraints, excessive communication overhead is undesirable. ByzSecAgg solves this problem by partitioning local updates into smaller sub-vectors and sharing them using ramp secret sharing. However, this sharing method does not admit bilinear computations, such as pairwise distances calculations, which are needed for distance-based outlier-detection algorithms, and effective methods for mitigating Byzantine attacks. To overcome this issue, each user runs another round of ramp sharing, with a different embedding of the data in the sharing polynomial. This technique, motivated by ideas from coded computing, enables secure computation of pairwise distance. In addition, to maintain the integrity and privacy of the local update, ByzSecAgg also uses a vector commitment method, in which the commitment size remains constant (i.e., does not increase with the length of the local update), while simultaneously allowing verification of the secret sharing process. In terms of communication load, ByzSecAgg significantly outperforms the related baseline scheme, known as BREA.



## **48. SATversary: Adversarial Attacks on Satellite Fingerprinting**

cs.CR

19 pages, 18 figures, 2 tables

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06119v1) [paper-pdf](http://arxiv.org/pdf/2506.06119v1)

**Authors**: Joshua Smailes, Sebastian Köhler, Simon Birnbach, Martin Strohmeier, Ivan Martinovic

**Abstract**: As satellite systems become increasingly vulnerable to physical layer attacks via SDRs, novel countermeasures are being developed to protect critical systems, particularly those lacking cryptographic protection, or those which cannot be upgraded to support modern cryptography. Among these is transmitter fingerprinting, which provides mechanisms by which communication can be authenticated by looking at characteristics of the transmitter, expressed as impairments on the signal.   Previous works show that fingerprinting can be used to classify satellite transmitters, or authenticate them against SDR-equipped attackers under simple replay scenarios. In this paper we build upon this by looking at attacks directly targeting the fingerprinting system, with an attacker optimizing for maximum impact in jamming, spoofing, and dataset poisoning attacks, and demonstrate these attacks on the SatIQ system designed to authenticate Iridium transmitters. We show that an optimized jamming signal can cause a 50% error rate with attacker-to-victim ratios as low as -30dB (far less power than traditional jamming) and demonstrate successful identity forgery during spoofing attacks, with an attacker successfully removing their own transmitter's fingerprint from messages. We also present a data poisoning attack, enabling persistent message spoofing by altering the data used to authenticate incoming messages to include the fingerprint of the attacker's transmitter.   Finally, we show that our model trained to optimize spoofing attacks can also be used to detect spoofing and replay attacks, even when it has never seen the attacker's transmitter before. Furthermore, this technique works even when the training dataset includes only a single transmitter, enabling fingerprinting to be used to protect small constellations and even individual satellites, providing additional protection where it is needed the most.



## **49. The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text**

cs.CL

42nd International Conference on Machine Learning (ICML 2025)

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2502.14921v2) [paper-pdf](http://arxiv.org/pdf/2502.14921v2)

**Authors**: Matthieu Meeus, Lukas Wutschitz, Santiago Zanella-Béguelin, Shruti Tople, Reza Shokri

**Abstract**: How much information about training samples can be leaked through synthetic data generated by Large Language Models (LLMs)? Overlooking the subtleties of information flow in synthetic data generation pipelines can lead to a false sense of privacy. In this paper, we assume an adversary has access to some synthetic data generated by a LLM. We design membership inference attacks (MIAs) that target the training data used to fine-tune the LLM that is then used to synthesize data. The significant performance of our MIA shows that synthetic data leak information about the training data. Further, we find that canaries crafted for model-based MIAs are sub-optimal for privacy auditing when only synthetic data is released. Such out-of-distribution canaries have limited influence on the model's output when prompted to generate useful, in-distribution synthetic data, which drastically reduces their effectiveness. To tackle this problem, we leverage the mechanics of auto-regressive models to design canaries with an in-distribution prefix and a high-perplexity suffix that leave detectable traces in synthetic data. This enhances the power of data-based MIAs and provides a better assessment of the privacy risks of releasing synthetic data generated by LLMs.



## **50. One Stone, Two Birds: Enhancing Adversarial Defense Through the Lens of Distributional Discrepancy**

cs.LG

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2503.02169v2) [paper-pdf](http://arxiv.org/pdf/2503.02169v2)

**Authors**: Jiacheng Zhang, Benjamin I. P. Rubinstein, Jingfeng Zhang, Feng Liu

**Abstract**: Statistical adversarial data detection (SADD) detects whether an upcoming batch contains adversarial examples (AEs) by measuring the distributional discrepancies between clean examples (CEs) and AEs. In this paper, we explore the strength of SADD-based methods by theoretically showing that minimizing distributional discrepancy can help reduce the expected loss on AEs. Despite these advantages, SADD-based methods have a potential limitation: they discard inputs that are detected as AEs, leading to the loss of useful information within those inputs. To address this limitation, we propose a two-pronged adversarial defense method, named Distributional-discrepancy-based Adversarial Defense (DAD). In the training phase, DAD first optimizes the test power of the maximum mean discrepancy (MMD) to derive MMD-OPT, which is a stone that kills two birds. MMD-OPT first serves as a guiding signal to minimize the distributional discrepancy between CEs and AEs to train a denoiser. Then, it serves as a discriminator to differentiate CEs and AEs during inference. Overall, in the inference stage, DAD consists of a two-pronged process: (1) directly feeding the detected CEs into the classifier, and (2) removing noise from the detected AEs by the distributional-discrepancy-based denoiser. Extensive experiments show that DAD outperforms current state-of-the-art (SOTA) defense methods by simultaneously improving clean and robust accuracy on CIFAR-10 and ImageNet-1K against adaptive white-box attacks. Codes are publicly available at: https://github.com/tmlr-group/DAD.



