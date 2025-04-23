# Latest Adversarial Attack Papers
**update at 2025-04-23 10:05:50**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Adversarial Observations in Weather Forecasting**

cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15942v1) [paper-pdf](http://arxiv.org/pdf/2504.15942v1)

**Authors**: Erik Imgrund, Thorsten Eisenhofer, Konrad Rieck

**Abstract**: AI-based systems, such as Google's GenCast, have recently redefined the state of the art in weather forecasting, offering more accurate and timely predictions of both everyday weather and extreme events. While these systems are on the verge of replacing traditional meteorological methods, they also introduce new vulnerabilities into the forecasting process. In this paper, we investigate this threat and present a novel attack on autoregressive diffusion models, such as those used in GenCast, capable of manipulating weather forecasts and fabricating extreme events, including hurricanes, heat waves, and intense rainfall. The attack introduces subtle perturbations into weather observations that are statistically indistinguishable from natural noise and change less than 0.1% of the measurements - comparable to tampering with data from a single meteorological satellite. As modern forecasting integrates data from nearly a hundred satellites and many other sources operated by different countries, our findings highlight a critical security risk with the potential to cause large-scale disruptions and undermine public trust in weather prediction.



## **2. Human-Imperceptible Physical Adversarial Attack for NIR Face Recognition Models**

cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15823v1) [paper-pdf](http://arxiv.org/pdf/2504.15823v1)

**Authors**: Songyan Xie, Jinghang Wen, Encheng Su, Qiucheng Yu

**Abstract**: Near-infrared (NIR) face recognition systems, which can operate effectively in low-light conditions or in the presence of makeup, exhibit vulnerabilities when subjected to physical adversarial attacks. To further demonstrate the potential risks in real-world applications, we design a novel, stealthy, and practical adversarial patch to attack NIR face recognition systems in a black-box setting. We achieved this by utilizing human-imperceptible infrared-absorbing ink to generate multiple patches with digitally optimized shapes and positions for infrared images. To address the optimization mismatch between digital and real-world NIR imaging, we develop a light reflection model for human skin to minimize pixel-level discrepancies by simulating NIR light reflection.   Compared to state-of-the-art (SOTA) physical attacks on NIR face recognition systems, the experimental results show that our method improves the attack success rate in both digital and physical domains, particularly maintaining effectiveness across various face postures. Notably, the proposed approach outperforms SOTA methods, achieving an average attack success rate of 82.46% in the physical domain across different models, compared to 64.18% for existing methods. The artifact is available at https://anonymous.4open.science/r/Human-imperceptible-adversarial-patch-0703/.



## **3. Graph Neural Networks for Next-Generation-IoT: Recent Advances and Open Challenges**

cs.IT

28 pages, 15 figures, and 6 tables. Submitted for publication

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2412.20634v2) [paper-pdf](http://arxiv.org/pdf/2412.20634v2)

**Authors**: Nguyen Xuan Tung, Le Tung Giang, Bui Duc Son, Seon Geun Jeong, Trinh Van Chien, Won Joo Hwang, Lajos Hanzo

**Abstract**: Graph Neural Networks (GNNs) have emerged as a critical tool for optimizing and managing the complexities of the Internet of Things (IoT) in next-generation networks. This survey presents a comprehensive exploration of how GNNs may be harnessed in 6G IoT environments, focusing on key challenges and opportunities through a series of open questions. We commence with an exploration of GNN paradigms and the roles of node, edge, and graph-level tasks in solving wireless networking problems and highlight GNNs' ability to overcome the limitations of traditional optimization methods. This guidance enhances problem-solving efficiency across various next-generation (NG) IoT scenarios. Next, we provide a detailed discussion of the application of GNN in advanced NG enabling technologies, including massive MIMO, reconfigurable intelligent surfaces, satellites, THz, mobile edge computing (MEC), and ultra-reliable low latency communication (URLLC). We then delve into the challenges posed by adversarial attacks, offering insights into defense mechanisms to secure GNN-based NG-IoT networks. Next, we examine how GNNs can be integrated with future technologies like integrated sensing and communication (ISAC), satellite-air-ground-sea integrated networks (SAGSIN), and quantum computing. Our findings highlight the transformative potential of GNNs in improving efficiency, scalability, and security within NG-IoT systems, paving the way for future advances. Finally, we propose a set of design guidelines to facilitate the development of efficient, scalable, and secure GNN models tailored for NG IoT applications.



## **4. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2408.09093v3) [paper-pdf](http://arxiv.org/pdf/2408.09093v3)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **5. Red Team Diffuser: Exposing Toxic Continuation Vulnerabilities in Vision-Language Models via Reinforcement Learning**

cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2503.06223v2) [paper-pdf](http://arxiv.org/pdf/2503.06223v2)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The growing deployment of large Vision-Language Models (VLMs) exposes critical safety gaps in their alignment mechanisms. While existing jailbreak studies primarily focus on VLMs' susceptibility to harmful instructions, we reveal a fundamental yet overlooked vulnerability: toxic text continuation, where VLMs produce highly toxic completions when prompted with harmful text prefixes paired with semantically adversarial images. To systematically study this threat, we propose Red Team Diffuser (RTD), the first red teaming diffusion model that coordinates adversarial image generation and toxic continuation through reinforcement learning. Our key innovations include dynamic cross-modal attack and stealth-aware optimization. For toxic text prefixes from an LLM safety benchmark, we conduct greedy search to identify optimal image prompts that maximally induce toxic completions. The discovered image prompts then drive RL-based diffusion model fine-tuning, producing semantically aligned adversarial images that boost toxicity rates. Stealth-aware optimization introduces joint adversarial rewards that balance toxicity maximization (via Detoxify classifier) and stealthiness (via BERTScore), circumventing traditional noise-based adversarial patterns. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% over text-only baselines on the original attack set and 8.91% on an unseen set, proving generalization capability. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. Our findings expose two critical flaws in current VLM alignment: (1) failure to prevent toxic continuation from harmful prefixes, and (2) overlooking cross-modal attack vectors. These results necessitate a paradigm shift toward multimodal red teaming in safety evaluations.



## **6. TrojanDam: Detection-Free Backdoor Defense in Federated Learning through Proactive Model Robustification utilizing OOD Data**

cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15674v1) [paper-pdf](http://arxiv.org/pdf/2504.15674v1)

**Authors**: Yanbo Dai, Songze Li, Zihan Gan, Xueluan Gong

**Abstract**: Federated learning (FL) systems allow decentralized data-owning clients to jointly train a global model through uploading their locally trained updates to a centralized server. The property of decentralization enables adversaries to craft carefully designed backdoor updates to make the global model misclassify only when encountering adversary-chosen triggers. Existing defense mechanisms mainly rely on post-training detection after receiving updates. These methods either fail to identify updates which are deliberately fabricated statistically close to benign ones, or show inconsistent performance in different FL training stages. The effect of unfiltered backdoor updates will accumulate in the global model, and eventually become functional. Given the difficulty of ruling out every backdoor update, we propose a backdoor defense paradigm, which focuses on proactive robustification on the global model against potential backdoor attacks. We first reveal that the successful launching of backdoor attacks in FL stems from the lack of conflict between malicious and benign updates on redundant neurons of ML models. We proceed to prove the feasibility of activating redundant neurons utilizing out-of-distribution (OOD) samples in centralized settings, and migrating to FL settings to propose a novel backdoor defense mechanism, TrojanDam. The proposed mechanism has the FL server continuously inject fresh OOD mappings into the global model to activate redundant neurons, canceling the effect of backdoor updates during aggregation. We conduct systematic and extensive experiments to illustrate the superior performance of TrojanDam, over several SOTA backdoor defense methods across a wide range of FL settings.



## **7. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

cs.CV

17 pages, 5 figures

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.14348v2) [paper-pdf](http://arxiv.org/pdf/2504.14348v2)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this work, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach consists of two key components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to infer the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms existing injection attacks, achieving at least a +26.4% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.



## **8. Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models**

cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2502.20650v2) [paper-pdf](http://arxiv.org/pdf/2502.20650v2)

**Authors**: Yu Pan, Bingrong Dai, Jiahao Chen, Lin Wang, Yi Du, Jiao Liu

**Abstract**: In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.



## **9. Evaluating the Robustness of Multimodal Agents Against Active Environmental Injection Attacks**

cs.CL

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2502.13053v2) [paper-pdf](http://arxiv.org/pdf/2502.13053v2)

**Authors**: Yurun Chen, Xavier Hu, Keting Yin, Juncheng Li, Shengyu Zhang

**Abstract**: As researchers continue to optimize AI agents for more effective task execution within operating systems, they often overlook a critical security concern: the ability of these agents to detect "impostors" within their environment. Through an analysis of the agents' operational context, we identify a significant threat-attackers can disguise malicious attacks as environmental elements, injecting active disturbances into the agents' execution processes to manipulate their decision-making. We define this novel threat as the Active Environment Injection Attack (AEIA). Focusing on the interaction mechanisms of the Android OS, we conduct a risk assessment of AEIA and identify two critical security vulnerabilities: (1) Adversarial content injection in multimodal interaction interfaces, where attackers embed adversarial instructions within environmental elements to mislead agent decision-making; and (2) Reasoning gap vulnerabilities in the agent's task execution process, which increase susceptibility to AEIA attacks during reasoning. To evaluate the impact of these vulnerabilities, we propose AEIA-MN, an attack scheme that exploits interaction vulnerabilities in mobile operating systems to assess the robustness of MLLM-based agents. Experimental results show that even advanced MLLMs are highly vulnerable to this attack, achieving a maximum attack success rate of 93% on the AndroidWorld benchmark by combining two vulnerabilities.



## **10. Unifying Image Counterfactuals and Feature Attributions with Latent-Space Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15479v1) [paper-pdf](http://arxiv.org/pdf/2504.15479v1)

**Authors**: Jeremy Goldwasser, Giles Hooker

**Abstract**: Counterfactuals are a popular framework for interpreting machine learning predictions. These what if explanations are notoriously challenging to create for computer vision models: standard gradient-based methods are prone to produce adversarial examples, in which imperceptible modifications to image pixels provoke large changes in predictions. We introduce a new, easy-to-implement framework for counterfactual images that can flexibly adapt to contemporary advances in generative modeling. Our method, Counterfactual Attacks, resembles an adversarial attack on the representation of the image along a low-dimensional manifold. In addition, given an auxiliary dataset of image descriptors, we show how to accompany counterfactuals with feature attribution that quantify the changes between the original and counterfactual images. These importance scores can be aggregated into global counterfactual explanations that highlight the overall features driving model predictions. While this unification is possible for any counterfactual method, it has particular computational efficiency for ours. We demonstrate the efficacy of our approach with the MNIST and CelebA datasets.



## **11. An Undetectable Watermark for Generative Image Models**

cs.CR

ICLR 2025

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.07369v4) [paper-pdf](http://arxiv.org/pdf/2410.07369v4)

**Authors**: Sam Gunn, Xuandong Zhao, Dawn Song

**Abstract**: We present the first undetectable watermarking scheme for generative image models. Undetectability ensures that no efficient adversary can distinguish between watermarked and un-watermarked images, even after making many adaptive queries. In particular, an undetectable watermark does not degrade image quality under any efficiently computable metric. Our scheme works by selecting the initial latents of a diffusion model using a pseudorandom error-correcting code (Christ and Gunn, 2024), a strategy which guarantees undetectability and robustness. We experimentally demonstrate that our watermarks are quality-preserving and robust using Stable Diffusion 2.1. Our experiments verify that, in contrast to every prior scheme we tested, our watermark does not degrade image quality. Our experiments also demonstrate robustness: existing watermark removal attacks fail to remove our watermark from images without significantly degrading the quality of the images. Finally, we find that we can robustly encode 512 bits in our watermark, and up to 2500 bits when the images are not subjected to watermark removal attacks. Our code is available at https://github.com/XuandongZhao/PRC-Watermark.



## **12. A Framework for Evaluating Emerging Cyberattack Capabilities of AI**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2503.11917v3) [paper-pdf](http://arxiv.org/pdf/2503.11917v3)

**Authors**: Mikel Rodriguez, Raluca Ada Popa, Four Flynn, Lihao Liang, Allan Dafoe, Anna Wang

**Abstract**: As frontier AI models become more capable, evaluating their potential to enable cyberattacks is crucial for ensuring the safe development of Artificial General Intelligence (AGI). Current cyber evaluation efforts are often ad-hoc, lacking systematic analysis of attack phases and guidance on targeted defenses. This work introduces a novel evaluation framework that addresses these limitations by: (1) examining the end-to-end attack chain, (2) identifying gaps in AI threat evaluation, and (3) helping defenders prioritize targeted mitigations and conduct AI-enabled adversary emulation for red teaming. Our approach adapts existing cyberattack chain frameworks for AI systems. We analyzed over 12,000 real-world instances of AI involvement in cyber incidents, catalogued by Google's Threat Intelligence Group, to curate seven representative attack chain archetypes. Through a bottleneck analysis on these archetypes, we pinpointed phases most susceptible to AI-driven disruption. We then identified and utilized externally developed cybersecurity model evaluations focused on these critical phases. We report on AI's potential to amplify offensive capabilities across specific attack stages, and offer recommendations for prioritizing defenses. We believe this represents the most comprehensive AI cyber risk evaluation framework published to date.



## **13. MST3 Encryption improvement with three-parameter group of Hermitian function field**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15391v1) [paper-pdf](http://arxiv.org/pdf/2504.15391v1)

**Authors**: Gennady Khalimov, Yevgen Kotukh

**Abstract**: This scholarly work presents an advanced cryptographic framework utilizing automorphism groups as the foundational structure for encryption scheme implementation. The proposed methodology employs a three-parameter group construction, distinguished by its application of logarithmic signatures positioned outside the group's center, a significant departure from conventional approaches. A key innovation in this implementation is utilizing the Hermitian function field as the underlying mathematical framework. This particular function field provides enhanced structural properties that strengthen the cryptographic protocol when integrated with the three-parameter group architecture. The encryption mechanism features phased key de-encapsulation from ciphertext, representing a substantial advantage over alternative implementations. This sequential extraction process introduces additional computational complexity for potential adversaries while maintaining efficient legitimate decryption. A notable characteristic of this cryptosystem is the direct correlation between the underlying group's mathematical strength and both the attack complexity and message size parameters. This relationship enables precise security-efficiency calibration based on specific implementation requirements and threat models. The application of automorphism groups with logarithmic signatures positioned outside the center represents a significant advancement in non-traditional cryptographic designs, particularly relevant in the context of post-quantum cryptographic resilience.



## **14. MR. Guard: Multilingual Reasoning Guardrail using Curriculum Learning**

cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15241v1) [paper-pdf](http://arxiv.org/pdf/2504.15241v1)

**Authors**: Yahan Yang, Soham Dan, Shuo Li, Dan Roth, Insup Lee

**Abstract**: Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual setting, where multilingual safety-aligned data are often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we propose an approach to build a multilingual guardrail with reasoning. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-guided Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail consistently outperforms recent baselines across both in-domain and out-of-domain languages. The multilingual reasoning capability of our guardrail enables it to generate multilingual explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.



## **15. Progressive Pruning: Analyzing the Impact of Intersection Attacks**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.08700v2) [paper-pdf](http://arxiv.org/pdf/2410.08700v2)

**Authors**: Christoph Döpmann, Maximilian Weisenseel, Florian Tschorsch

**Abstract**: Stream-based communication dominates today's Internet, posing unique challenges for anonymous communication networks (ACNs). Traditionally designed for independent messages, ACNs struggle to account for the inherent vulnerabilities of streams, such as susceptibility to intersection attacks. In this work, we address this gap and introduce progressive pruning, a novel methodology for quantifying the susceptibility to intersection attacks. Progressive pruning quantifies and monitors anonymity sets over time, providing an assessment of an adversary's success in correlating senders and receivers. We leverage this methodology to analyze synthetic scenarios and large-scale simulations of the Tor network using our newly developed TorFS simulator. Our findings reveal that anonymity is significantly influenced by stream length, user population, and stream distribution across the network. These insights highlight critical design challenges for future ACNs seeking to safeguard stream-based communication against traffic analysis attacks.



## **16. HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States**

cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2502.14744v3) [paper-pdf](http://arxiv.org/pdf/2502.14744v3)

**Authors**: Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, Xiangyu Yue

**Abstract**: The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at https://github.com/leigest519/HiddenDetect.



## **17. Scalable Discrete Event Simulation Tool for Large-Scale Cyber-Physical Energy Systems: Advancing System Efficiency and Scalability**

eess.SY

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15198v1) [paper-pdf](http://arxiv.org/pdf/2504.15198v1)

**Authors**: Khandaker Akramul Haque, Shining Sun, Xiang Huo, Ana E. Goulart, Katherine R. Davis

**Abstract**: Modern power systems face growing risks from cyber-physical attacks, necessitating enhanced resilience due to their societal function as critical infrastructures. The challenge is that defense of large-scale systems-of-systems requires scalability in their threat and risk assessment environment for cyber physical analysis including cyber-informed transmission planning, decision-making, and intrusion response. Hence, we present a scalable discrete event simulation tool for analysis of energy systems, called DESTinE. The tool is tailored for largescale cyber-physical systems, with a focus on power systems. It supports faster-than-real-time traffic generation and models packet flow and congestion under both normal and adversarial conditions. Using three well-established power system synthetic cases with 500, 2000, and 10,000 buses, we overlay a constructed cyber network employing star and radial topologies. Experiments are conducted to identify critical nodes within a communication network in response to a disturbance. The findings are incorporated into a constrained optimization problem to assess the impact of the disturbance on a specific node and its cascading effects on the overall network. Based on the solution of the optimization problem, a new hybrid network topology is also derived, combining the strengths of star and radial structures to improve network resilience. Furthermore, DESTinE is integrated with a virtual server and a hardware-in-the-loop (HIL) system using Raspberry Pi 5.



## **18. RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search**

cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15047v1) [paper-pdf](http://arxiv.org/pdf/2504.15047v1)

**Authors**: Quy-Anh Dang, Chris Ngo, Truong-Son Hy

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but are susceptible to adversarial prompts that exploit vulnerabilities to produce unsafe or biased outputs. Existing red-teaming methods often face scalability challenges, resource-intensive requirements, or limited diversity in attack strategies. We propose RainbowPlus, a novel red-teaming framework rooted in evolutionary computation, enhancing adversarial prompt generation through an adaptive quality-diversity (QD) search that extends classical evolutionary algorithms like MAP-Elites with innovations tailored for language models. By employing a multi-element archive to store diverse high-quality prompts and a comprehensive fitness function to evaluate multiple prompts concurrently, RainbowPlus overcomes the constraints of single-prompt archives and pairwise comparisons in prior QD methods like Rainbow Teaming. Experiments comparing RainbowPlus to QD methods across six benchmark datasets and four open-source LLMs demonstrate superior attack success rate (ASR) and diversity (Diverse-Score $\approx 0.84$), generating up to 100 times more unique prompts (e.g., 10,418 vs. 100 for Ministral-8B-Instruct-2410). Against nine state-of-the-art methods on the HarmBench dataset with twelve LLMs (ten open-source, two closed-source), RainbowPlus achieves an average ASR of 81.1%, surpassing AutoDAN-Turbo by 3.9%, and is 9 times faster (1.45 vs. 13.50 hours). Our open-source implementation fosters further advancements in LLM safety, offering a scalable tool for vulnerability assessment. Code and resources are publicly available at https://github.com/knoveleng/rainbowplus, supporting reproducibility and future research in LLM red-teaming.



## **19. An Information-theoretic Security Analysis of Honeyword**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2311.10960v2) [paper-pdf](http://arxiv.org/pdf/2311.10960v2)

**Authors**: Pengcheng Su, Haibo Cheng, Wenting Li, Ping Wang

**Abstract**: Honeyword is a representative "honey" technique that employs decoy objects to mislead adversaries and protect the real ones. To assess the security of a Honeyword system, two metrics--flatness and success-number--have been proposed and evaluated using various simulated attackers. Existing evaluations typically apply statistical learning methods to distinguish real passwords from decoys on real-world datasets. However, such evaluations may overestimate the system's security, as more effective distinguishing attacks could potentially exist.   In this paper, we aim to analyze the security of Honeyword systems under the strongest theoretical attack, rather than relying on specific, expert-crafted attacks evaluated in prior experimental studies. We first derive mathematical expressions for the flatness and success-number under the strongest attack. We conduct analyses and computations for several typical scenarios, and determine the security of honeyword generation methods using a uniform distribution and the List model as examples.   We further evaluate the security of existing honeyword generation methods based on password probability models (PPMs), which depends on the sample size used for training. We investigate, for the first time, the sample complexity of several representative PPMs, introducing two novel polynomial-time approximation schemes for computing the total variation between PCFG models and between higher-order Markov models. Our experimental results show that for small-scale password distributions, sample sizes on the order of millions--often tens of millions--are required to reduce the total variation below 0.1. A surprising result is that we establish an equivalence between flatness and total variation, thus bridging the theoretical study of Honeyword systems with classical information theory. Finally, we discuss the practical implications of our findings.



## **20. Transferable Adversarial Attacks on SAM and Its Downstream Models**

cs.LG

update fig 1

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.20197v3) [paper-pdf](http://arxiv.org/pdf/2410.20197v3)

**Authors**: Song Xia, Wenhan Yang, Yi Yu, Xun Lin, Henghui Ding, Ling-Yu Duan, Xudong Jiang

**Abstract**: The utilization of large foundational models has a dilemma: while fine-tuning downstream tasks from them holds promise for making use of the well-generalized knowledge in practical applications, their open accessibility also poses threats of adverse usage. This paper, for the first time, explores the feasibility of adversarial attacking various downstream models fine-tuned from the segment anything model (SAM), by solely utilizing the information from the open-sourced SAM. In contrast to prevailing transfer-based adversarial attacks, we demonstrate the existence of adversarial dangers even without accessing the downstream task and dataset to train a similar surrogate model. To enhance the effectiveness of the adversarial attack towards models fine-tuned on unknown datasets, we propose a universal meta-initialization (UMI) algorithm to extract the intrinsic vulnerability inherent in the foundation model, which is then utilized as the prior knowledge to guide the generation of adversarial perturbations. Moreover, by formulating the gradient difference in the attacking process between the open-sourced SAM and its fine-tuned downstream models, we theoretically demonstrate that a deviation occurs in the adversarial update direction by directly maximizing the distance of encoded feature embeddings in the open-sourced SAM. Consequently, we propose a gradient robust loss that simulates the associated uncertainty with gradient-based noise augmentation to enhance the robustness of generated adversarial examples (AEs) towards this deviation, thus improving the transferability. Extensive experiments demonstrate the effectiveness of the proposed universal meta-initialized and gradient robust adversarial attack (UMI-GRAT) toward SAMs and their downstream models. Code is available at https://github.com/xiasong0501/GRAT.



## **21. aiXamine: LLM Safety and Security Simplified**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.14985v1) [paper-pdf](http://arxiv.org/pdf/2504.14985v1)

**Authors**: Fatih Deniz, Dorde Popovic, Yazan Boshmaf, Euisuh Jeong, Minhaj Ahmad, Sanjay Chawla, Issa Khalil

**Abstract**: Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.



## **22. Fast Adversarial Training with Weak-to-Strong Spatial-Temporal Consistency in the Frequency Domain on Videos**

cs.CV

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.14921v1) [paper-pdf](http://arxiv.org/pdf/2504.14921v1)

**Authors**: Songping Wang, Hanqing Liu, Yueming Lyu, Xiantao Hu, Ziwen He, Wei Wang, Caifeng Shan, Liang Wang

**Abstract**: Adversarial Training (AT) has been shown to significantly enhance adversarial robustness via a min-max optimization approach. However, its effectiveness in video recognition tasks is hampered by two main challenges. First, fast adversarial training for video models remains largely unexplored, which severely impedes its practical applications. Specifically, most video adversarial training methods are computationally costly, with long training times and high expenses. Second, existing methods struggle with the trade-off between clean accuracy and adversarial robustness. To address these challenges, we introduce Video Fast Adversarial Training with Weak-to-Strong consistency (VFAT-WS), the first fast adversarial training method for video data. Specifically, VFAT-WS incorporates the following key designs: First, it integrates a straightforward yet effective temporal frequency augmentation (TF-AUG), and its spatial-temporal enhanced form STF-AUG, along with a single-step PGD attack to boost training efficiency and robustness. Second, it devises a weak-to-strong spatial-temporal consistency regularization, which seamlessly integrates the simpler TF-AUG and the more complex STF-AUG. Leveraging the consistency regularization, it steers the learning process from simple to complex augmentations. Both of them work together to achieve a better trade-off between clean accuracy and robustness. Extensive experiments on UCF-101 and HMDB-51 with both CNN and Transformer-based models demonstrate that VFAT-WS achieves great improvements in adversarial robustness and corruption robustness, while accelerating training by nearly 490%.



## **23. PA-Boot: A Formally Verified Authentication Protocol for Multiprocessor Secure Boot**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2209.07936v3) [paper-pdf](http://arxiv.org/pdf/2209.07936v3)

**Authors**: Zhuoruo Zhang, Rui Chang, Mingshuai Chen, Wenbo Shen, Chenyang Yu, He Huang, Qinming Dai, Yongwang Zhao

**Abstract**: Hardware supply-chain attacks are raising significant security threats to the boot process of multiprocessor systems. This paper identifies a new, prevalent hardware supply-chain attack surface that can bypass multiprocessor secure boot due to the absence of processor-authentication mechanisms. To defend against such attacks, we present PA-Boot, the first formally verified processor-authentication protocol for secure boot in multiprocessor systems. PA-Boot is proved functionally correct and is guaranteed to detect multiple adversarial behaviors, e.g., processor replacements, man-in-the-middle attacks, and tampering with certificates. The fine-grained formalization of PA-Boot and its fully mechanized security proofs are carried out in the Isabelle/HOL theorem prover with 306 lemmas/theorems and ~7,100 LoC. Experiments on a proof-of-concept implementation indicate that PA-Boot can effectively identify boot-process attacks with a considerably minor overhead and thereby improve the security of multiprocessor systems.



## **24. Verifying Robust Unlearning: Probing Residual Knowledge in Unlearned Models**

cs.LG

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.14798v1) [paper-pdf](http://arxiv.org/pdf/2504.14798v1)

**Authors**: Hao Xuan, Xingyu Li

**Abstract**: Machine Unlearning (MUL) is crucial for privacy protection and content regulation, yet recent studies reveal that traces of forgotten information persist in unlearned models, enabling adversaries to resurface removed knowledge. Existing verification methods only confirm whether unlearning was executed, failing to detect such residual information leaks. To address this, we introduce the concept of Robust Unlearning, ensuring models are indistinguishable from retraining and resistant to adversarial recovery. To empirically evaluate whether unlearning techniques meet this security standard, we propose the Unlearning Mapping Attack (UMA), a post-unlearning verification framework that actively probes models for forgotten traces using adversarial queries. Extensive experiments on discriminative and generative tasks show that existing unlearning techniques remain vulnerable, even when passing existing verification metrics. By establishing UMA as a practical verification tool, this study sets a new standard for assessing and enhancing machine unlearning security.



## **25. Modality Unified Attack for Omni-Modality Person Re-Identification**

cs.CV

9 pages,3 figures

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2501.12761v2) [paper-pdf](http://arxiv.org/pdf/2501.12761v2)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yunfeng Ma, Yaonan Wang

**Abstract**: Deep learning based person re-identification (re-id) models have been widely employed in surveillance systems. Recent studies have demonstrated that black-box single-modality and cross-modality re-id models are vulnerable to adversarial examples (AEs), leaving the robustness of multi-modality re-id models unexplored. Due to the lack of knowledge about the specific type of model deployed in the target black-box surveillance system, we aim to generate modality unified AEs for omni-modality (single-, cross- and multi-modality) re-id models. Specifically, we propose a novel Modality Unified Attack method to train modality-specific adversarial generators to generate AEs that effectively attack different omni-modality models. A multi-modality model is adopted as the surrogate model, wherein the features of each modality are perturbed by metric disruption loss before fusion. To collapse the common features of omni-modality models, Cross Modality Simulated Disruption approach is introduced to mimic the cross-modality feature embeddings by intentionally feeding images to non-corresponding modality-specific subnetworks of the surrogate model. Moreover, Multi Modality Collaborative Disruption strategy is devised to facilitate the attacker to comprehensively corrupt the informative content of person images by leveraging a multi modality feature collaborative metric disruption loss. Extensive experiments show that our MUA method can effectively attack the omni-modality re-id models, achieving 55.9%, 24.4%, 49.0% and 62.7% mean mAP Drop Rate, respectively.



## **26. LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts**

cs.CR

23 pages, 7 figures; Accepted to FSE 2025, to be published in The  Proceedings of the ACM on Software Engineering (PACMSE)

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2401.07261v6) [paper-pdf](http://arxiv.org/pdf/2401.07261v6)

**Authors**: Shoupeng Ren, Lipeng He, Tianyu Tu, Di Wu, Jian Liu, Kui Ren, Chun Chen

**Abstract**: The exploitation of smart contract vulnerabilities in Decentralized Finance (DeFi) has resulted in financial losses exceeding 3 billion US dollars. Existing defense mechanisms primarily focus on detecting and reacting to adversarial transactions executed by attackers that target victim contracts. However, with the emergence of private transaction pools where transactions are sent directly to miners without first appearing in public mempools, current detection tools face significant challenges in identifying attack activities effectively. Based on the fact that most attack logic rely on deploying intermediate smart contracts as supporting components to the exploitation of victim contracts, novel detection methods have been proposed that focus on identifying these adversarial contracts instead of adversarial transactions. However, previous state-of-the-art approaches in this direction have failed to produce results satisfactory enough for real-world deployment. In this paper, we propose LookAhead, a new framework for detecting DeFi attacks via unveiling adversarial contracts. LookAhead leverages common attack patterns, code semantics and intrinsic characteristics found in adversarial smart contracts to train Machine Learning (ML)-based classifiers that can effectively distinguish adversarial contracts from benign ones and make timely predictions of different types of potential attacks. Experiments on our labeled datasets show that LookAhead achieves an F1-score as high as 0.8966, which represents an improvement of over 44.4% compared to the previous state-of-the-art solution, with a False Positive Rate (FPR) at only 0.16%.



## **27. Human-AI Collaboration in Cloud Security: Cognitive Hierarchy-Driven Deep Reinforcement Learning**

cs.CR

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2502.16054v2) [paper-pdf](http://arxiv.org/pdf/2502.16054v2)

**Authors**: Zahra Aref, Sheng Wei, Narayan B. Mandayam

**Abstract**: Given the complexity of multi-tenant cloud environments and the growing need for real-time threat mitigation, Security Operations Centers (SOCs) must adopt AI-driven adaptive defense mechanisms to counter Advanced Persistent Threats (APTs). However, SOC analysts face challenges in handling adaptive adversarial tactics, requiring intelligent decision-support frameworks. We propose a Cognitive Hierarchy Theory-driven Deep Q-Network (CHT-DQN) framework that models interactive decision-making between SOC analysts and AI-driven APT bots. The SOC analyst (defender) operates at cognitive level-1, anticipating attacker strategies, while the APT bot (attacker) follows a level-0 policy. By incorporating CHT into DQN, our framework enhances adaptive SOC defense using Attack Graph (AG)-based reinforcement learning. Simulation experiments across varying AG complexities show that CHT-DQN consistently achieves higher data protection and lower action discrepancies compared to standard DQN. A theoretical lower bound further confirms its superiority as AG complexity increases. A human-in-the-loop (HITL) evaluation on Amazon Mechanical Turk (MTurk) reveals that SOC analysts using CHT-DQN-derived transition probabilities align more closely with adaptive attackers, leading to better defense outcomes. Moreover, human behavior aligns with Prospect Theory (PT) and Cumulative Prospect Theory (CPT): participants are less likely to reselect failed actions and more likely to persist with successful ones. This asymmetry reflects amplified loss sensitivity and biased probability weighting -- underestimating gains after failure and overestimating continued success. Our findings highlight the potential of integrating cognitive models into deep reinforcement learning to improve real-time SOC decision-making for cloud security.



## **28. Large Language Models as Robust Data Generators in Software Analytics: Are We There Yet?**

cs.SE

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2411.10565v2) [paper-pdf](http://arxiv.org/pdf/2411.10565v2)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Large Language Model (LLM)-generated data is increasingly used in software analytics, but it is unclear how this data compares to human-written data, particularly when models are exposed to adversarial scenarios. Adversarial attacks can compromise the reliability and security of software systems, so understanding how LLM-generated data performs under these conditions, compared to human-written data, which serves as the benchmark for model performance, can provide valuable insights into whether LLM-generated data offers similar robustness and effectiveness. To address this gap, we systematically evaluate and compare the quality of human-written and LLM-generated data for fine-tuning robust pre-trained models (PTMs) in the context of adversarial attacks. We evaluate the robustness of six widely used PTMs, fine-tuned on human-written and LLM-generated data, before and after adversarial attacks. This evaluation employs nine state-of-the-art (SOTA) adversarial attack techniques across three popular software analytics tasks: clone detection, code summarization, and sentiment analysis in code review discussions. Additionally, we analyze the quality of the generated adversarial examples using eleven similarity metrics. Our findings reveal that while PTMs fine-tuned on LLM-generated data perform competitively with those fine-tuned on human-written data, they exhibit less robustness against adversarial attacks in software analytics tasks. Our study underscores the need for further exploration into enhancing the quality of LLM-generated training data to develop models that are both high-performing and capable of withstanding adversarial attacks in software analytics.



## **29. Towards Model Resistant to Transferable Adversarial Examples via Trigger Activation**

cs.CR

Accepted by IEEE TIFS 2025

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2504.14541v1) [paper-pdf](http://arxiv.org/pdf/2504.14541v1)

**Authors**: Yi Yu, Song Xia, Xun Lin, Chenqi Kong, Wenhan Yang, Shijian Lu, Yap-Peng Tan, Alex C. Kot

**Abstract**: Adversarial examples, characterized by imperceptible perturbations, pose significant threats to deep neural networks by misleading their predictions. A critical aspect of these examples is their transferability, allowing them to deceive {unseen} models in black-box scenarios. Despite the widespread exploration of defense methods, including those on transferability, they show limitations: inefficient deployment, ineffective defense, and degraded performance on clean images. In this work, we introduce a novel training paradigm aimed at enhancing robustness against transferable adversarial examples (TAEs) in a more efficient and effective way. We propose a model that exhibits random guessing behavior when presented with clean data $\boldsymbol{x}$ as input, and generates accurate predictions when with triggered data $\boldsymbol{x}+\boldsymbol{\tau}$. Importantly, the trigger $\boldsymbol{\tau}$ remains constant for all data instances. We refer to these models as \textbf{models with trigger activation}. We are surprised to find that these models exhibit certain robustness against TAEs. Through the consideration of first-order gradients, we provide a theoretical analysis of this robustness. Moreover, through the joint optimization of the learnable trigger and the model, we achieve improved robustness to transferable attacks. Extensive experiments conducted across diverse datasets, evaluating a variety of attacking methods, underscore the effectiveness and superiority of our approach.



## **30. Slice+Slice Baby: Generating Last-Level Cache Eviction Sets in the Blink of an Eye**

cs.CR

Added reference to the ID3 decision tree induction algorithm by J. R.  Quinlan in Section 5.4

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2504.11208v2) [paper-pdf](http://arxiv.org/pdf/2504.11208v2)

**Authors**: Bradley Morgan, Gal Horowitz, Sioli O'Connell, Stephan van Schaik, Chitchanok Chuengsatiansup, Daniel Genkin, Olaf Maennel, Paul Montague, Eyal Ronen, Yuval Yarom

**Abstract**: An essential step for mounting cache attacks is finding eviction sets, collections of memory locations that contend on cache space. On Intel processors, one of the main challenges for identifying contending addresses is the sliced cache design, where the processor hashes the physical address to determine where in the cache a memory location is stored. While past works have demonstrated that the hash function can be reversed, they also showed that it depends on physical address bits that the adversary does not know.   In this work, we make three main contributions to the art of finding eviction sets. We first exploit microarchitectural races to compare memory access times and identify the cache slice to which an address maps. We then use the known hash function to both reduce the error rate in our slice identification method and to reduce the work by extrapolating slice mappings to untested memory addresses. Finally, we show how to propagate information on eviction sets across different page offsets for the hitherto unexplored case of non-linear hash functions.   Our contributions allow for entire LLC eviction set generation in 0.7 seconds on the Intel i7-9850H and 1.6 seconds on the i9-10900K, both using non-linear functions. This represents a significant improvement compared to state-of-the-art techniques taking 9x and 10x longer, respectively.



## **31. Adversarial Attack for RGB-Event based Visual Object Tracking**

cs.CV

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2504.14423v1) [paper-pdf](http://arxiv.org/pdf/2504.14423v1)

**Authors**: Qiang Chen, Xiao Wang, Haowen Wang, Bo Jiang, Lin Zhu, Dawei Zhang, Yonghong Tian, Jin Tang

**Abstract**: Visual object tracking is a crucial research topic in the fields of computer vision and multi-modal fusion. Among various approaches, robust visual tracking that combines RGB frames with Event streams has attracted increasing attention from researchers. While striving for high accuracy and efficiency in tracking, it is also important to explore how to effectively conduct adversarial attacks and defenses on RGB-Event stream tracking algorithms, yet research in this area remains relatively scarce. To bridge this gap, in this paper, we propose a cross-modal adversarial attack algorithm for RGB-Event visual tracking. Because of the diverse representations of Event streams, and given that Event voxels and frames are more commonly used, this paper will focus on these two representations for an in-depth study. Specifically, for the RGB-Event voxel, we first optimize the perturbation by adversarial loss to generate RGB frame adversarial examples. For discrete Event voxel representations, we propose a two-step attack strategy, more in detail, we first inject Event voxels into the target region as initialized adversarial examples, then, conduct a gradient-guided optimization by perturbing the spatial location of the Event voxels. For the RGB-Event frame based tracking, we optimize the cross-modal universal perturbation by integrating the gradient information from multimodal data. We evaluate the proposed approach against attacks on three widely used RGB-Event Tracking datasets, i.e., COESOT, FE108, and VisEvent. Extensive experiments show that our method significantly reduces the performance of the tracker across numerous datasets in both unimodal and multimodal scenarios. The source code will be released on https://github.com/Event-AHU/Adversarial_Attack_Defense



## **32. Hydra: An Agentic Reasoning Approach for Enhancing Adversarial Robustness and Mitigating Hallucinations in Vision-Language Models**

cs.CV

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2504.14395v1) [paper-pdf](http://arxiv.org/pdf/2504.14395v1)

**Authors**: Chung-En, Yu, Hsuan-Chih, Chen, Brian Jalaian, Nathaniel D. Bastian

**Abstract**: To develop trustworthy Vision-Language Models (VLMs), it is essential to address adversarial robustness and hallucination mitigation, both of which impact factual accuracy in high-stakes applications such as defense and healthcare. Existing methods primarily focus on either adversarial defense or hallucination post-hoc correction, leaving a gap in unified robustness strategies. We introduce \textbf{Hydra}, an adaptive agentic framework that enhances plug-in VLMs through iterative reasoning, structured critiques, and cross-model verification, improving both resilience to adversarial perturbations and intrinsic model errors. Hydra employs an Action-Critique Loop, where it retrieves and critiques visual information, leveraging Chain-of-Thought (CoT) and In-Context Learning (ICL) techniques to refine outputs dynamically. Unlike static post-hoc correction methods, Hydra adapts to both adversarial manipulations and intrinsic model errors, making it robust to malicious perturbations and hallucination-related inaccuracies. We evaluate Hydra on four VLMs, three hallucination benchmarks, two adversarial attack strategies, and two adversarial defense methods, assessing performance on both clean and adversarial inputs. Results show that Hydra surpasses plug-in VLMs and state-of-the-art (SOTA) dehallucination methods, even without explicit adversarial defenses, demonstrating enhanced robustness and factual consistency. By bridging adversarial resistance and hallucination mitigation, Hydra provides a scalable, training-free solution for improving the reliability of VLMs in real-world applications.



## **33. WeiDetect: Weibull Distribution-Based Defense against Poisoning Attacks in Federated Learning for Network Intrusion Detection Systems**

cs.CR

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2504.04367v2) [paper-pdf](http://arxiv.org/pdf/2504.04367v2)

**Authors**: Sameera K. M., Vinod P., Anderson Rocha, Rafidha Rehiman K. A., Mauro Conti

**Abstract**: In the era of data expansion, ensuring data privacy has become increasingly critical, posing significant challenges to traditional AI-based applications. In addition, the increasing adoption of IoT devices has introduced significant cybersecurity challenges, making traditional Network Intrusion Detection Systems (NIDS) less effective against evolving threats, and privacy concerns and regulatory restrictions limit their deployment. Federated Learning (FL) has emerged as a promising solution, allowing decentralized model training while maintaining data privacy to solve these issues. However, despite implementing privacy-preserving technologies, FL systems remain vulnerable to adversarial attacks. Furthermore, data distribution among clients is not heterogeneous in the FL scenario. We propose WeiDetect, a two-phase, server-side defense mechanism for FL-based NIDS that detects malicious participants to address these challenges. In the first phase, local models are evaluated using a validation dataset to generate validation scores. These scores are then analyzed using a Weibull distribution, identifying and removing malicious models. We conducted experiments to evaluate the effectiveness of our approach in diverse attack settings. Our evaluation included two popular datasets, CIC-Darknet2020 and CSE-CIC-IDS2018, tested under non-IID data distributions. Our findings highlight that WeiDetect outperforms state-of-the-art defense approaches, improving higher target class recall up to 70% and enhancing the global model's F1 score by 1% to 14%.



## **34. Reason2Attack: Jailbreaking Text-to-Image Models via LLM Reasoning**

cs.CR

This paper includes model-generated content that may contain  offensive or distressing material

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2503.17987v2) [paper-pdf](http://arxiv.org/pdf/2503.17987v2)

**Authors**: Chenyu Zhang, Lanjun Wang, Yiwen Ma, Wenhui Li, An-An Liu

**Abstract**: Text-to-Image(T2I) models typically deploy safety filters to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attack methods manually design prompts for the LLM to generate adversarial prompts, which effectively bypass safety filters while producing sensitive images, exposing safety vulnerabilities of T2I models. However, due to the LLM's limited understanding of the T2I model and its safety filters, existing methods require numerous queries to achieve a successful attack, limiting their practical applicability. To address this issue, we propose Reason2Attack(R2A), which aims to enhance the LLM's reasoning capabilities in generating adversarial prompts by incorporating the jailbreaking attack into the post-training process of the LLM. Specifically, we first propose a CoT example synthesis pipeline based on Frame Semantics, which generates adversarial prompts by identifying related terms and corresponding context illustrations. Using CoT examples generated by the pipeline, we fine-tune the LLM to understand the reasoning path and format the output structure. Subsequently, we incorporate the jailbreaking attack task into the reinforcement learning process of the LLM and design an attack process reward that considers prompt length, prompt stealthiness, and prompt effectiveness, aiming to further enhance reasoning accuracy. Extensive experiments on various T2I models show that R2A achieves a better attack success ratio while requiring fewer queries than baselines. Moreover, our adversarial prompts demonstrate strong attack transferability across both open-source and commercial T2I models.



## **35. Jailbreaking as a Reward Misspecification Problem**

cs.LG

Accepted to ICLR 2025. Code:  https://github.com/zhxieml/remiss-jailbreak

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2406.14393v5) [paper-pdf](http://arxiv.org/pdf/2406.14393v5)

**Authors**: Zhihui Xie, Jiahui Gao, Lei Li, Zhenguo Li, Qi Liu, Lingpeng Kong

**Abstract**: The widespread adoption of large language models (LLMs) has raised concerns about their safety and reliability, particularly regarding their vulnerability to adversarial attacks. In this paper, we propose a novel perspective that attributes this vulnerability to reward misspecification during the alignment process. This misspecification occurs when the reward function fails to accurately capture the intended behavior, leading to misaligned model outputs. We introduce a metric ReGap to quantify the extent of reward misspecification and demonstrate its effectiveness and robustness in detecting harmful backdoor prompts. Building upon these insights, we present ReMiss, a system for automated red teaming that generates adversarial prompts in a reward-misspecified space. ReMiss achieves state-of-the-art attack success rates on the AdvBench benchmark against various target aligned LLMs while preserving the human readability of the generated prompts. Furthermore, these attacks on open-source models demonstrate high transferability to closed-source models like GPT-4o and out-of-distribution tasks from HarmBench. Detailed analysis highlights the unique advantages of the proposed reward misspecification objective compared to previous methods, offering new insights for improving LLM safety and robustness.



## **36. Rethinking Target Label Conditioning in Adversarial Attacks: A 2D Tensor-Guided Generative Approach**

cs.CV

12 pages, 4 figures

**SubmitDate**: 2025-04-19    [abs](http://arxiv.org/abs/2504.14137v1) [paper-pdf](http://arxiv.org/pdf/2504.14137v1)

**Authors**: Hangyu Liu, Bo Peng, Pengxiang Ding, Donglin Wang

**Abstract**: Compared to single-target adversarial attacks, multi-target attacks have garnered significant attention due to their ability to generate adversarial images for multiple target classes simultaneously. Existing generative approaches for multi-target attacks mainly analyze the effect of the use of target labels on noise generation from a theoretical perspective, lacking practical validation and comprehensive summarization. To address this gap, we first identify and validate that the semantic feature quality and quantity are critical factors affecting the transferability of targeted attacks: 1) Feature quality refers to the structural and detailed completeness of the implanted target features, as deficiencies may result in the loss of key discriminative information; 2) Feature quantity refers to the spatial sufficiency of the implanted target features, as inadequacy limits the victim model's attention to this feature. Based on these findings, we propose the 2D Tensor-Guided Adversarial Fusion (2D-TGAF) framework, which leverages the powerful generative capabilities of diffusion models to encode target labels into two-dimensional semantic tensors for guiding adversarial noise generation. Additionally, we design a novel masking strategy tailored for the training process, ensuring that parts of the generated noise retain complete semantic information about the target class. Extensive experiments on the standard ImageNet dataset demonstrate that 2D-TGAF consistently surpasses state-of-the-art methods in attack success rates, both on normally trained models and across various defense mechanisms.



## **37. Robust Decentralized Quantum Kernel Learning for Noisy and Adversarial Environment**

quant-ph

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13782v1) [paper-pdf](http://arxiv.org/pdf/2504.13782v1)

**Authors**: Wenxuan Ma, Kuan-Cheng Chen, Shang Yu, Mengxiang Liu, Ruilong Deng

**Abstract**: This paper proposes a general decentralized framework for quantum kernel learning (QKL). It has robustness against quantum noise and can also be designed to defend adversarial information attacks forming a robust approach named RDQKL. We analyze the impact of noise on QKL and study the robustness of decentralized QKL to the noise. By integrating robust decentralized optimization techniques, our method is able to mitigate the impact of malicious data injections across multiple nodes. Experimental results demonstrate that our approach maintains high accuracy under noisy quantum operations and effectively counter adversarial modifications, offering a promising pathway towards the future practical, scalable and secure quantum machine learning (QML).



## **38. Adversarial Hubness in Multi-Modal Retrieval**

cs.CR

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2412.14113v2) [paper-pdf](http://arxiv.org/pdf/2412.14113v2)

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a single point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries.   In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, as well as for targeted attacks on queries related to specific, attacker-chosen concepts.   We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system implemented by Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub, generated with respect to 100 randomly selected target queries, is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries), demonstrating the strong generalization capabilities of adversarial hubs. We also investigate whether techniques for mitigating natural hubness are an effective defense against adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.



## **39. Energy-Latency Attacks via Sponge Poisoning**

cs.CR

Paper accepted at Information Sciences journal; 20 pages Keywords:  energy-latency attacks, sponge attack, machine learning security, adversarial  machine learning

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2203.08147v5) [paper-pdf](http://arxiv.org/pdf/2203.08147v5)

**Authors**: Antonio Emanuele Cinà, Ambra Demontis, Battista Biggio, Fabio Roli, Marcello Pelillo

**Abstract**: Sponge examples are test-time inputs optimized to increase energy consumption and prediction latency of deep networks deployed on hardware accelerators. By increasing the fraction of neurons activated during classification, these attacks reduce sparsity in network activation patterns, worsening the performance of hardware accelerators. In this work, we present a novel training-time attack, named sponge poisoning, which aims to worsen energy consumption and prediction latency of neural networks on any test input without affecting classification accuracy. To stage this attack, we assume that the attacker can control only a few model updates during training -- a likely scenario, e.g., when model training is outsourced to an untrusted third party or distributed via federated learning. Our extensive experiments on image classification tasks show that sponge poisoning is effective, and that fine-tuning poisoned models to repair them poses prohibitive costs for most users, highlighting that tackling sponge poisoning remains an open issue.



## **40. Fairness and Robustness in Machine Unlearning**

cs.LG

5 pages

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13610v1) [paper-pdf](http://arxiv.org/pdf/2504.13610v1)

**Authors**: Khoa Tran, Simon S. Woo

**Abstract**: Machine unlearning poses the challenge of ``how to eliminate the influence of specific data from a pretrained model'' in regard to privacy concerns. While prior research on approximated unlearning has demonstrated accuracy and efficiency in time complexity, we claim that it falls short of achieving exact unlearning, and we are the first to focus on fairness and robustness in machine unlearning algorithms. Our study presents fairness Conjectures for a well-trained model, based on the variance-bias trade-off characteristic, and considers their relevance to robustness. Our Conjectures are supported by experiments conducted on the two most widely used model architectures, ResNet and ViT, demonstrating the correlation between fairness and robustness: \textit{the higher fairness-gap is, the more the model is sensitive and vulnerable}. In addition, our experiments demonstrate the vulnerability of current state-of-the-art approximated unlearning algorithms to adversarial attacks, where their unlearned models suffer a significant drop in accuracy compared to the exact-unlearned models. We claim that our fairness-gap measurement and robustness metric should be used to evaluate the unlearning algorithm. Furthermore, we demonstrate that unlearning in the intermediate and last layers is sufficient and cost-effective for time and memory complexity.



## **41. Q-FAKER: Query-free Hard Black-box Attack via Controlled Generation**

cs.CR

NAACL 2025 Findings

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13551v1) [paper-pdf](http://arxiv.org/pdf/2504.13551v1)

**Authors**: CheolWon Na, YunSeok Choi, Jee-Hyong Lee

**Abstract**: Many adversarial attack approaches are proposed to verify the vulnerability of language models. However, they require numerous queries and the information on the target model. Even black-box attack methods also require the target model's output information. They are not applicable in real-world scenarios, as in hard black-box settings where the target model is closed and inaccessible. Even the recently proposed hard black-box attacks still require many queries and demand extremely high costs for training adversarial generators. To address these challenges, we propose Q-faker (Query-free Hard Black-box Attacker), a novel and efficient method that generates adversarial examples without accessing the target model. To avoid accessing the target model, we use a surrogate model instead. The surrogate model generates adversarial sentences for a target-agnostic attack. During this process, we leverage controlled generation techniques. We evaluate our proposed method on eight datasets. Experimental results demonstrate our method's effectiveness including high transferability and the high quality of the generated adversarial examples, and prove its practical in hard black-box settings.



## **42. Few-shot Model Extraction Attacks against Sequential Recommender Systems**

cs.LG

It requires substantial modifications.The symbols in the mathematical  formulas are not explained in detail

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2411.11677v2) [paper-pdf](http://arxiv.org/pdf/2411.11677v2)

**Authors**: Hui Zhang, Fu Liu

**Abstract**: Among adversarial attacks against sequential recommender systems, model extraction attacks represent a method to attack sequential recommendation models without prior knowledge. Existing research has primarily concentrated on the adversary's execution of black-box attacks through data-free model extraction. However, a significant gap remains in the literature concerning the development of surrogate models by adversaries with access to few-shot raw data (10\% even less). That is, the challenge of how to construct a surrogate model with high functional similarity within the context of few-shot data scenarios remains an issue that requires resolution.This study addresses this gap by introducing a novel few-shot model extraction framework against sequential recommenders, which is designed to construct a superior surrogate model with the utilization of few-shot data. The proposed few-shot model extraction framework is comprised of two components: an autoregressive augmentation generation strategy and a bidirectional repair loss-facilitated model distillation procedure. Specifically, to generate synthetic data that closely approximate the distribution of raw data, autoregressive augmentation generation strategy integrates a probabilistic interaction sampler to extract inherent dependencies and a synthesis determinant signal module to characterize user behavioral patterns. Subsequently, bidirectional repair loss, which target the discrepancies between the recommendation lists, is designed as auxiliary loss to rectify erroneous predictions from surrogate models, transferring knowledge from the victim model to the surrogate model effectively. Experiments on three datasets show that the proposed few-shot model extraction framework yields superior surrogate models.



## **43. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

cs.CL

WWW'25 research track accepted

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2406.11260v3) [paper-pdf](http://arxiv.org/pdf/2406.11260v3)

**Authors**: Sungwon Park, Sungwon Han, Xing Xie, Jae-Gil Lee, Meeyoung Cha

**Abstract**: The spread of fake news harms individuals and presents a critical social challenge that must be addressed. Although numerous algorithmic and insightful features have been developed to detect fake news, many of these features can be manipulated with style-conversion attacks, especially with the emergence of advanced language models, making it more difficult to differentiate from genuine news. This study proposes adversarial style augmentation, AdStyle, designed to train a fake news detector that remains robust against various style-conversion attacks. The primary mechanism involves the strategic use of LLMs to automatically generate a diverse and coherent array of style-conversion attack prompts, enhancing the generation of particularly challenging prompts for the detector. Experiments indicate that our augmentation strategy significantly improves robustness and detection performance when evaluated on fake news benchmark datasets.



## **44. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

cs.CL

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.05050v2) [paper-pdf](http://arxiv.org/pdf/2504.05050v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.



## **45. EXAM: Exploiting Exclusive System-Level Cache in Apple M-Series SoCs for Enhanced Cache Occupancy Attacks**

cs.CR

Accepted to ACM ASIA CCS 2025

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13385v1) [paper-pdf](http://arxiv.org/pdf/2504.13385v1)

**Authors**: Tianhong Xu, Aidong Adam Ding, Yunsi Fei

**Abstract**: Cache occupancy attacks exploit the shared nature of cache hierarchies to infer a victim's activities by monitoring overall cache usage, unlike access-driven cache attacks that focus on specific cache lines or sets. There exists some prior work that target the last-level cache (LLC) of Intel processors, which is inclusive of higher-level caches, and L2 caches of ARM systems. In this paper, we target the System-Level Cache (SLC) of Apple M-series SoCs, which is exclusive to higher-level CPU caches. We address the challenges of the exclusiveness and propose a suite of SLC-cache occupancy attacks, the first of its kind, where an adversary can monitor GPU and other CPU cluster activities from their own CPU cluster. We first discover the structure of SLC in Apple M1 SOC and various policies pertaining to access and sharing through reverse engineering. We propose two attacks against websites. One is a coarse-grained fingerprinting attack, recognizing which website is accessed based on their different GPU memory access patterns monitored through the SLC occupancy channel. The other attack is a fine-grained pixel stealing attack, which precisely monitors the GPU memory usage for rendering different pixels, through the SLC occupancy channel. Third, we introduce a novel screen capturing attack which works beyond webpages, with the monitoring granularity of 57 rows of pixels (there are 1600 rows for the screen). This significantly expands the attack surface, allowing the adversary to retrieve any screen display, posing a substantial new threat to system security. Our findings reveal critical vulnerabilities in Apple's M-series SoCs and emphasize the urgent need for effective countermeasures against cache occupancy attacks in heterogeneous computing environments.



## **46. DYNAMITE: Dynamic Defense Selection for Enhancing Machine Learning-based Intrusion Detection Against Adversarial Attacks**

cs.CR

Accepted by the IEEE/ACM Workshop on the Internet of Safe Things  (SafeThings 2025)

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13301v1) [paper-pdf](http://arxiv.org/pdf/2504.13301v1)

**Authors**: Jing Chen, Onat Gungor, Zhengli Shang, Elvin Li, Tajana Rosing

**Abstract**: The rapid proliferation of the Internet of Things (IoT) has introduced substantial security vulnerabilities, highlighting the need for robust Intrusion Detection Systems (IDS). Machine learning-based intrusion detection systems (ML-IDS) have significantly improved threat detection capabilities; however, they remain highly susceptible to adversarial attacks. While numerous defense mechanisms have been proposed to enhance ML-IDS resilience, a systematic approach for selecting the most effective defense against a specific adversarial attack remains absent. To address this challenge, we propose Dynamite, a dynamic defense selection framework that enhances ML-IDS by intelligently identifying and deploying the most suitable defense using a machine learning-driven selection mechanism. Our results demonstrate that Dynamite achieves a 96.2% reduction in computational time compared to the Oracle, significantly decreasing computational overhead while preserving strong prediction performance. Dynamite also demonstrates an average F1-score improvement of 76.7% over random defense and 65.8% over the best static state-of-the-art defense.



## **47. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

cs.CR

Accepted at ICLR 2025. Updates in the v3: GPT-4o and Claude 3.5  Sonnet results, improved writing. Updates in the v2: more models (Llama3,  Phi-3, Nemotron-4-340B), jailbreak artifacts for all attacks are available,  evaluation with different judges (Llama-3-70B and Llama Guard 2), more  experiments (convergence plots, ablation on the suffix length for random  search), examples of jailbroken generation

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2404.02151v4) [paper-pdf](http://arxiv.org/pdf/2404.02151v4)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize a target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve 100% attack success rate -- according to GPT-4 as a judge -- on Vicuna-13B, Mistral-7B, Phi-3-Mini, Nemotron-4-340B, Llama-2-Chat-7B/13B/70B, Llama-3-Instruct-8B, Gemma-7B, GPT-3.5, GPT-4o, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with a 100% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings, it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). For reproducibility purposes, we provide the code, logs, and jailbreak artifacts in the JailbreakBench format at https://github.com/tml-epfl/llm-adaptive-attacks.



## **48. Chypnosis: Stealthy Secret Extraction using Undervolting-based Static Side-channel Attacks**

cs.CR

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.11633v2) [paper-pdf](http://arxiv.org/pdf/2504.11633v2)

**Authors**: Kyle Mitard, Saleh Khalaj Monfared, Fatemeh Khojasteh Dana, Shahin Tajik

**Abstract**: There is a growing class of static physical side-channel attacks that allow adversaries to extract secrets by probing the persistent state of a circuit. Techniques such as laser logic state imaging (LLSI), impedance analysis (IA), and static power analysis fall into this category. These attacks require that the targeted data remain constant for a specific duration, which often necessitates halting the circuit's clock. Some methods additionally rely on modulating the chip's supply voltage to probe the circuit. However, tampering with the clock or voltage is typically assumed to be detectable, as secure chips often deploy sensors that erase sensitive data upon detecting such anomalies. Furthermore, many secure devices use internal clock sources, making external clock control infeasible. In this work, we introduce a novel class of static side-channel attacks, called Chypnosis, that enables adversaries to freeze a chip's internal clock by inducing a hibernation state via rapid undervolting, and then extracting secrets using static side-channels. We demonstrate that, by rapidly dropping a chip's voltage below the standard nominal levels, the attacker can bypass the clock and voltage sensors and put the chip in a so-called brownout condition, in which the chip's transistors stop switching, but volatile memories (e.g., Flip-flops and SRAMs) still retain their data. We test our attack on AMD FPGAs by putting them into hibernation. We show that not only are all clock sources deactivated, but various clock and voltage sensors also fail to detect the tamper event. Afterward, we present the successful recovery of secret bits from a hibernated chip using two static attacks, namely, LLSI and IA. Finally, we discuss potential countermeasures which could be integrated into future designs.



## **49. Strategic Planning of Stealthy Backdoor Attacks in Markov Decision Processes**

eess.SY

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13276v1) [paper-pdf](http://arxiv.org/pdf/2504.13276v1)

**Authors**: Xinyi Wei, Shuo Han, Ahmed H. Hemida, Charles A. Kamhoua, Jie Fu

**Abstract**: This paper investigates backdoor attack planning in stochastic control systems modeled as Markov Decision Processes (MDPs). In a backdoor attack, the adversary provides a control policy that behaves well in the original MDP to pass the testing phase. However, when such a policy is deployed with a trigger policy, which perturbs the system dynamics at runtime, it optimizes the attacker's objective instead. To solve jointly the control policy and its trigger, we formulate the attack planning problem as a constrained optimal planning problem in an MDP with augmented state space, with the objective to maximize the attacker's total rewards in the system with an activated trigger, subject to the constraint that the control policy is near optimal in the original MDP. We then introduce a gradient-based optimization method to solve the optimal backdoor attack policy as a pair of coordinated control and trigger policies. Experimental results from a case study validate the effectiveness of our approach in achieving stealthy backdoor attacks.



## **50. Does Refusal Training in LLMs Generalize to the Past Tense?**

cs.CL

Accepted at ICLR 2025. Updates in v2 and v3: added GPT-4o, Claude 3.5  Sonnet, o1-mini, and o1-preview results. Code and jailbreak artifacts:  https://github.com/tml-epfl/llm-past-tense

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2407.11969v4) [paper-pdf](http://arxiv.org/pdf/2407.11969v4)

**Authors**: Maksym Andriushchenko, Nicolas Flammarion

**Abstract**: Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, Claude-3.5 Sonnet, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o mini, GPT-4o, o1-mini, o1-preview, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.



