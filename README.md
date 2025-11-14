# Latest Adversarial Attack Papers
**update at 2025-11-14 10:17:20**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Unveiling Hidden Threats: Using Fractal Triggers to Boost Stealthiness of Distributed Backdoor Attacks in Federated Learning**

cs.CR

10 pages, 1 figures, conference

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09252v1) [paper-pdf](None)

**Authors**: Jian Wang, Hong Shen, Chan-Tong Lam

**Abstract**: Traditional distributed backdoor attacks (DBA) in federated learning improve stealthiness by decomposing global triggers into sub-triggers, which however requires more poisoned data to maintian the attck strength and hence increases the exposure risk. To overcome this defect, This paper proposes a novel method, namely Fractal-Triggerred Distributed Backdoor Attack (FTDBA), which leverages the self-similarity of fractals to enhance the feature strength of sub-triggers and hence significantly reduce the required poisoning volume for the same attack strength. To address the detectability of fractal structures in the frequency and gradient domains, we introduce a dynamic angular perturbation mechanism that adaptively adjusts perturbation intensity across the training phases to balance efficiency and stealthiness. Experiments show that FTDBA achieves a 92.3\% attack success rate with only 62.4\% of the poisoning volume required by traditional DBA methods, while reducing the detection rate by 22.8\% and KL divergence by 41.2\%. This study presents a low-exposure, high-efficiency paradigm for federated backdoor attacks and expands the application of fractal features in adversarial sample generation.



## **2. Systematic Literature Review on Vehicular Collaborative Perception -- A Computer Vision Perspective**

cs.CV

38 pages, 8 figures, accepted for publication in IEEE Transactions on Intelligent Transportation Systems (T-ITS)

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2504.04631v3) [paper-pdf](None)

**Authors**: Lei Wan, Jianxin Zhao, Andreas Wiedholz, Manuel Bied, Mateus Martinez de Lucena, Abhishek Dinkar Jagtap, Andreas Festag, Antônio Augusto Fröhlich, Hannan Ejaz Keen, Alexey Vinel

**Abstract**: The effectiveness of autonomous vehicles relies on reliable perception capabilities. Despite significant advancements in artificial intelligence and sensor fusion technologies, current single-vehicle perception systems continue to encounter limitations, notably visual occlusions and limited long-range detection capabilities. Collaborative Perception (CP), enabled by Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication, has emerged as a promising solution to mitigate these issues and enhance the reliability of autonomous systems. Beyond advancements in communication, the computer vision community is increasingly focusing on improving vehicular perception through collaborative approaches. However, a systematic literature review that thoroughly examines existing work and reduces subjective bias is still lacking. Such a systematic approach helps identify research gaps, recognize common trends across studies, and inform future research directions. In response, this study follows the PRISMA 2020 guidelines and includes 106 peer-reviewed articles. These publications are analyzed based on modalities, collaboration schemes, and key perception tasks. Through a comparative analysis, this review illustrates how different methods address practical issues such as pose errors, temporal latency, communication constraints, domain shifts, heterogeneity, and adversarial attacks. Furthermore, it critically examines evaluation methodologies, highlighting a misalignment between current metrics and CP's fundamental objectives. By delving into all relevant topics in-depth, this review offers valuable insights into challenges, opportunities, and risks, serving as a reference for advancing research in vehicular collaborative perception.



## **3. Improving Adversarial Transferability with Neighbourhood Gradient Information**

cs.CV

Accepted by Applied Soft Computing

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2408.05745v2) [paper-pdf](None)

**Authors**: Haijing Guo, Jiafeng Wang, Zhaoyu Chen, Kaixun Jiang, Lingyi Hong, Pinxue Guo, Jinglun Li, Wenqiang Zhang

**Abstract**: Deep neural networks (DNNs) are known to be susceptible to adversarial examples, leading to significant performance degradation. In black-box attack scenarios, a considerable attack performance gap between the surrogate model and the target model persists. This work focuses on enhancing the transferability of adversarial examples to narrow this performance gap. We observe that the gradient information around the clean image, i.e., Neighbourhood Gradient Information (NGI), can offer high transferability.Based on this insight, we introduce NGI-Attack, incorporating Example Backtracking and Multiplex Mask strategies to exploit this gradient information and enhance transferability. Specifically, we first adopt Example Backtracking to accumulate Neighbourhood Gradient Information as the initial momentum term. Then, we utilize Multiplex Mask to form a multi-way attack strategy that forces the network to focus on non-discriminative regions, which can obtain richer gradient information during only a few iterations. Extensive experiments demonstrate that our approach significantly enhances adversarial transferability. Especially, when attacking numerous defense models, we achieve an average attack success rate of 95.2%. Notably, our method can seamlessly integrate with any off-the-shelf algorithm, enhancing their attack performance without incurring extra time costs.



## **4. Filtered-ViT: A Robust Defense Against Multiple Adversarial Patch Attacks**

cs.CV

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.07755v1) [paper-pdf](None)

**Authors**: Aja Khanal, Ahmed Faid, Apurva Narayan

**Abstract**: Deep learning vision systems are increasingly deployed in safety-critical domains such as healthcare, yet they remain vulnerable to small adversarial patches that can trigger misclassifications. Most existing defenses assume a single patch and fail when multiple localized disruptions occur, the type of scenario adversaries and real-world artifacts often exploit. We propose Filtered-ViT, a new vision transformer architecture that integrates SMART Vector Median Filtering (SMART-VMF), a spatially adaptive, multi-scale, robustness-aware mechanism that enables selective suppression of corrupted regions while preserving semantic detail. On ImageNet with LaVAN multi-patch attacks, Filtered-ViT achieves 79.8% clean accuracy and 46.3% robust accuracy under four simultaneous 1\% patches, outperforming existing defenses. Beyond synthetic benchmarks, a real-world case study on radiographic medical imagery shows that Filtered-ViT mitigates natural artifacts such as occlusions and scanner noise without degrading diagnostic content. This establishes Filtered-ViT as the first transformer to demonstrate unified robustness against both adversarial and naturally occurring patch-like disruptions, charting a path toward reliable vision systems in truly high-stakes environments.



## **5. A Small Leak Sinks All: Exploring the Transferable Vulnerability of Source Code Models**

cs.SE

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.08127v1) [paper-pdf](None)

**Authors**: Weiye Li, Wenyi Tang

**Abstract**: Source Code Model learn the proper embeddings from source codes, demonstrating significant success in various software engineering or security tasks. The recent explosive development of LLM extends the family of SCMs,bringing LLMs for code that revolutionize development workflows. Investigating different kinds of SCM vulnerability is the cornerstone for the security and trustworthiness of AI-powered software ecosystems, however, the fundamental one, transferable vulnerability, remains critically underexplored. Existing studies neither offer practical ways, i.e. require access to the downstream classifier of SCMs, to produce effective adversarial samples for adversarial defense, nor give heed to the widely used LLM4Code in modern software development platforms and cloud-based integrated development environments. Therefore, this work systematically studies the intrinsic vulnerability transferability of both traditional SCMs and LLM4Code, and proposes a victim-agnostic approach to generate practical adversarial samples. We design HABITAT, consisting of a tailored perturbation-inserting mechanism and a hierarchical Reinforcement Learning framework that adaptively selects optimal perturbations without requiring any access to the downstream classifier of SCMs. Furthermore, an intrinsic transferability analysis of SCM vulnerabilities is conducted, revealing the potential vulnerability correlation between traditional SCMs and LLM4Code, together with fundamental factors that govern the success rate of victim-agnostic transfer attacks. These findings of SCM vulnerabilities underscore the critical focal points for developing robust defenses in the future. Experimental evaluation demonstrates that our constructed adversarial examples crafted based on traditional SCMs achieve up to 64% success rates against LLM4Code, surpassing the state-of-the-art by over 15%.



## **6. MSCR: Exploring the Vulnerability of LLMs' Mathematical Reasoning Abilities Using Multi-Source Candidate Replacement**

cs.AI

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.08055v1) [paper-pdf](None)

**Authors**: Zhishen Sun, Guang Dai, Haishan Ye

**Abstract**: LLMs demonstrate performance comparable to human abilities in complex tasks such as mathematical reasoning, but their robustness in mathematical reasoning under minor input perturbations still lacks systematic investigation. Existing methods generally suffer from limited scalability, weak semantic preservation, and high costs. Therefore, we propose MSCR, an automated adversarial attack method based on multi-source candidate replacement. By combining three information sources including cosine similarity in the embedding space of LLMs, the WordNet dictionary, and contextual predictions from a masked language model, we generate for each word in the input question a set of semantically similar candidates, which are then filtered and substituted one by one to carry out the attack. We conduct large-scale experiments on LLMs using the GSM8K and MATH500 benchmarks. The results show that even a slight perturbation involving only a single word can significantly reduce the accuracy of all models, with the maximum drop reaching 49.89% on GSM8K and 35.40% on MATH500, while preserving the high semantic consistency of the perturbed questions. Further analysis reveals that perturbations not only lead to incorrect outputs but also substantially increase the average response length, which results in more redundant reasoning paths and higher computational resource consumption. These findings highlight the robustness deficiencies and efficiency bottlenecks of current LLMs in mathematical reasoning tasks.



## **7. Invisible Triggers, Visible Threats! Road-Style Adversarial Creation Attack for Visual 3D Detection in Autonomous Driving**

cs.CV

Accepted by the AAAI 2026 (Main Track)

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.08015v1) [paper-pdf](None)

**Authors**: Jian Wang, Lijun He, Yixing Yong, Haixia Bi, Fan Li

**Abstract**: Modern autonomous driving (AD) systems leverage 3D object detection to perceive foreground objects in 3D environments for subsequent prediction and planning. Visual 3D detection based on RGB cameras provides a cost-effective solution compared to the LiDAR paradigm. While achieving promising detection accuracy, current deep neural network-based models remain highly susceptible to adversarial examples. The underlying safety concerns motivate us to investigate realistic adversarial attacks in AD scenarios. Previous work has demonstrated the feasibility of placing adversarial posters on the road surface to induce hallucinations in the detector. However, the unnatural appearance of the posters makes them easily noticeable by humans, and their fixed content can be readily targeted and defended. To address these limitations, we propose the AdvRoad to generate diverse road-style adversarial posters. The adversaries have naturalistic appearances resembling the road surface while compromising the detector to perceive non-existent objects at the attack locations. We employ a two-stage approach, termed Road-Style Adversary Generation and Scenario-Associated Adaptation, to maximize the attack effectiveness on the input scene while ensuring the natural appearance of the poster, allowing the attack to be carried out stealthily without drawing human attention. Extensive experiments show that AdvRoad generalizes well to different detectors, scenes, and spoofing locations. Moreover, physical attacks further demonstrate the practical threats in real-world environments.



## **8. Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization**

cs.CR

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2509.05831v3) [paper-pdf](None)

**Authors**: Ishaan Verma, Arsheya Yadav

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.



## **9. Exploiting Data Structures for Bypassing and Crashing Anti-Malware Solutions via Telemetry Complexity Attacks**

cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.04472v2) [paper-pdf](None)

**Authors**: Evgenios Gkritsis, Constantinos Patsakis, George Stergiopoulos

**Abstract**: Anti-malware systems rely on sandboxes, hooks, and telemetry pipelines, including collection agents, serializers, and database backends, to monitor program and system behavior. We show that these data-handling components constitute an exploitable attack surface that can lead to denial-of-analysis (DoA) states without disabling sensors or requiring elevated privileges. As a result, we present Telemetry Complexity Attacks (TCAs), a new class of vulnerabilities that exploit fundamental mismatches between unbounded collection mechanisms and bounded processing capabilities. Our method recursively spawns child processes to generate specially crafted, deeply nested, and oversized telemetry that stresses serialization and storage boundaries, as well as visualization layers, for example, JSON/BSON depth and size limits. Depending on the product, this leads to various inconsistent results, such as truncated or missing behavioral reports, rejected database inserts, serializer recursion and size errors, and unresponsive dashboards. In the latter cases, depending on the solution, the malware under test is either not recorded and/or not presented to the analysts. Therefore, instead of evading sensors, we break the pipeline that stores the data captured by the sensors.   We evaluate our technique against twelve commercial and open-source malware analysis platforms and endpoint detection and response (EDR) solutions. Seven products fail in different stages of the telemetry pipeline; two vendors assigned CVE identifiers (CVE-2025-61301 and CVE-2025-61303), and others issued patches or configuration changes. We discuss root causes and propose mitigation strategies to prevent DoA attacks triggered by adversarial telemetry.



## **10. Keep on Going: Learning Robust Humanoid Motion Skills via Selective Adversarial Training**

cs.RO

13 pages, 10 figures, AAAI2026

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2507.08303v3) [paper-pdf](None)

**Authors**: Yang Zhang, Zhanxiang Cao, Buqing Nie, Haoyang Li, Zhong Jiangwei, Qiao Sun, Xiaoyi Hu, Xiaokang Yang, Yue Gao

**Abstract**: Humanoid robots are expected to operate reliably over long horizons while executing versatile whole-body skills. Yet Reinforcement Learning (RL) motion policies typically lose stability under prolonged operation, sensor/actuator noise, and real world disturbances. In this work, we propose a Selective Adversarial Attack for Robust Training (SA2RT) to enhance the robustness of motion skills. The adversary is learned to identify and sparsely perturb the most vulnerable states and actions under an attack-budget constraint, thereby exposing true weakness without inducing conservative overfitting. The resulting non-zero sum, alternating optimization continually strengthens the motion policy against the strongest discovered attacks. We validate our approach on the Unitree G1 humanoid robot across perceptive locomotion and whole-body control tasks. Experimental results show that adversarially trained policies improve the terrain traversal success rate by 40%, reduce the trajectory tracking error by 32%, and maintain long horizon mobility and tracking performance. Together, these results demonstrate that selective adversarial attacks are an effective driver for learning robust, long horizon humanoid motion skills.



## **11. Potent but Stealthy: Rethink Profile Pollution against Sequential Recommendation via Bi-level Constrained Reinforcement Paradigm**

cs.LG

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09392v2) [paper-pdf](None)

**Authors**: Jiajie Su, Zihan Nan, Yunshan Ma, Xiaobo Xia, Xiaohua Feng, Weiming Liu, Xiaolin Zheng, Chaochao Chen

**Abstract**: Sequential Recommenders, which exploit dynamic user intents through interaction sequences, is vulnerable to adversarial attacks. While existing attacks primarily rely on data poisoning, they require large-scale user access or fake profiles thus lacking practicality. In this paper, we focus on the Profile Pollution Attack that subtly contaminates partial user interactions to induce targeted mispredictions. Previous PPA methods suffer from two limitations, i.e., i) over-reliance on sequence horizon impact restricts fine-grained perturbations on item transitions, and ii) holistic modifications cause detectable distribution shifts. To address these challenges, we propose a constrained reinforcement driven attack CREAT that synergizes a bi-level optimization framework with multi-reward reinforcement learning to balance adversarial efficacy and stealthiness. We first develop a Pattern Balanced Rewarding Policy, which integrates pattern inversion rewards to invert critical patterns and distribution consistency rewards to minimize detectable shifts via unbalanced co-optimal transport. Then we employ a Constrained Group Relative Reinforcement Learning paradigm, enabling step-wise perturbations through dynamic barrier constraints and group-shared experience replay, achieving targeted pollution with minimal detectability. Extensive experiments demonstrate the effectiveness of CREAT.



## **12. On Stealing Graph Neural Network Models**

cs.LG

Accepted at AAAI 2026

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.07170v2) [paper-pdf](None)

**Authors**: Marcin Podhajski, Jan Dubiński, Franziska Boenisch, Adam Dziedzic, Agnieszka Pręgowska, Tomasz P. Michalak

**Abstract**: Current graph neural network (GNN) model-stealing methods rely heavily on queries to the victim model, assuming no hard query limits. However, in reality, the number of allowed queries can be severely limited. In this paper, we demonstrate how an adversary can extract a GNN with very limited interactions with the model. Our approach first enables the adversary to obtain the model backbone without making direct queries to the victim model and then to strategically utilize a fixed query limit to extract the most informative data. The experiments on eight real-world datasets demonstrate the effectiveness of the attack, even under a very restricted query limit and under defense against model extraction in place. Our findings underscore the need for robust defenses against GNN model extraction threats.



## **13. SEBA: Sample-Efficient Black-Box Attacks on Visual Reinforcement Learning**

cs.LG

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09681v1) [paper-pdf](None)

**Authors**: Tairan Huang, Yulin Jin, Junxu Liu, Qingqing Ye, Haibo Hu

**Abstract**: Visual reinforcement learning has achieved remarkable progress in visual control and robotics, but its vulnerability to adversarial perturbations remains underexplored. Most existing black-box attacks focus on vector-based or discrete-action RL, and their effectiveness on image-based continuous control is limited by the large action space and excessive environment queries. We propose SEBA, a sample-efficient framework for black-box adversarial attacks on visual RL agents. SEBA integrates a shadow Q model that estimates cumulative rewards under adversarial conditions, a generative adversarial network that produces visually imperceptible perturbations, and a world model that simulates environment dynamics to reduce real-world queries. Through a two-stage iterative training procedure that alternates between learning the shadow model and refining the generator, SEBA achieves strong attack performance while maintaining efficiency. Experiments on MuJoCo and Atari benchmarks show that SEBA significantly reduces cumulative rewards, preserves visual fidelity, and greatly decreases environment interactions compared to prior black-box and white-box methods.



## **14. Phantom Menace: Exploring and Enhancing the Robustness of VLA Models against Physical Sensor Attacks**

cs.RO

Accepted by AAAI 2026

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.10008v1) [paper-pdf](None)

**Authors**: Xuancun Lu, Jiaxiang Chen, Shilin Xiao, Zizhi Jin, Zhangrui Chen, Hanwen Yu, Bohan Qian, Ruochen Zhou, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored.   To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel ``Real-Sim-Real'' framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments.



## **15. EnchTable: Unified Safety Alignment Transfer in Fine-tuned Large Language Models**

cs.CL

Accepted by IEEE Symposium on Security and Privacy (S&P) 2026

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09880v1) [paper-pdf](None)

**Authors**: Jialin Wu, Kecen Li, Zhicong Huang, Xinfeng Li, Xiaofeng Wang, Cheng Hong

**Abstract**: Many machine learning models are fine-tuned from large language models (LLMs) to achieve high performance in specialized domains like code generation, biomedical analysis, and mathematical problem solving. However, this fine-tuning process often introduces a critical vulnerability: the systematic degradation of safety alignment, undermining ethical guidelines and increasing the risk of harmful outputs. Addressing this challenge, we introduce EnchTable, a novel framework designed to transfer and maintain safety alignment in downstream LLMs without requiring extensive retraining. EnchTable leverages a Neural Tangent Kernel (NTK)-based safety vector distillation method to decouple safety constraints from task-specific reasoning, ensuring compatibility across diverse model architectures and sizes. Additionally, our interference-aware merging technique effectively balances safety and utility, minimizing performance compromises across various task domains. We implemented a fully functional prototype of EnchTable on three different task domains and three distinct LLM architectures, and evaluated its performance through extensive experiments on eleven diverse datasets, assessing both utility and model safety. Our evaluations include LLMs from different vendors, demonstrating EnchTable's generalization capability. Furthermore, EnchTable exhibits robust resistance to static and dynamic jailbreaking attacks, outperforming vendor-released safety models in mitigating adversarial prompts. Comparative analyses with six parameter modification methods and two inference-time alignment baselines reveal that EnchTable achieves a significantly lower unsafe rate, higher utility score, and universal applicability across different task domains. Additionally, we validate EnchTable can be seamlessly integrated into various deployment pipelines without significant overhead.



## **16. Privacy on the Fly: A Predictive Adversarial Transformation Network for Mobile Sensor Data**

cs.CR

accepted by AAAI 2026 (oral)

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.07242v2) [paper-pdf](None)

**Authors**: Tianle Song, Chenhao Lin, Yang Cao, Zhengyu Zhao, Jiahao Sun, Chong Zhang, Le Yang, Chao Shen

**Abstract**: Mobile motion sensors such as accelerometers and gyroscopes are now ubiquitously accessible by third-party apps via standard APIs. While enabling rich functionalities like activity recognition and step counting, this openness has also enabled unregulated inference of sensitive user traits, such as gender, age, and even identity, without user consent. Existing privacy-preserving techniques, such as GAN-based obfuscation or differential privacy, typically require access to the full input sequence, introducing latency that is incompatible with real-time scenarios. Worse, they tend to distort temporal and semantic patterns, degrading the utility of the data for benign tasks like activity recognition. To address these limitations, we propose the Predictive Adversarial Transformation Network (PATN), a real-time privacy-preserving framework that leverages historical signals to generate adversarial perturbations proactively. The perturbations are applied immediately upon data acquisition, enabling continuous protection without disrupting application functionality. Experiments on two datasets demonstrate that PATN substantially degrades the performance of privacy inference models, achieving Attack Success Rate (ASR) of 40.11% and 44.65% (reducing inference accuracy to near-random) and increasing the Equal Error Rate (EER) from 8.30% and 7.56% to 41.65% and 46.22%. On ASR, PATN outperforms baseline methods by 16.16% and 31.96%, respectively.



## **17. Reasoning Up the Instruction Ladder for Controllable Language Models**

cs.CL

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.04694v2) [paper-pdf](None)

**Authors**: Zishuo Zheng, Vidhisha Balachandran, Chan Young Park, Faeze Brahman, Sachin Kumar

**Abstract**: As large language model (LLM) based systems take on high-stakes roles in real-world decision-making, they must reconcile competing instructions from multiple sources (e.g., model developers, users, and tools) within a single prompt context. Thus, enforcing an instruction hierarchy (IH) in LLMs, where higher-level directives override lower-priority requests, is critical for the reliability and controllability of LLMs. In this work, we reframe instruction hierarchy resolution as a reasoning task. Specifically, the model must first "think" about the relationship between a given user prompt and higher-priority (system) instructions before generating a response. To enable this capability via training, we construct VerIH, an instruction hierarchy dataset of constraint-following tasks with verifiable answers. This dataset comprises both aligned and conflicting system-user instructions. We show that lightweight reinforcement learning with VerIH effectively transfers general reasoning capabilities of models to instruction prioritization. Our finetuned models achieve consistent improvements on instruction following and instruction hierarchy benchmarks. This reasoning ability also generalizes to safety-critical settings beyond the training distribution. By treating safety issues as resolving conflicts between adversarial user inputs and predefined higher-priority policies, our trained model enhances robustness against jailbreak and prompt injection attacks. These results demonstrate that reasoning over instruction hierarchies provides a practical path to reliable LLMs, where updates to system prompts yield controllable and robust changes in model behavior.



## **18. Removal Attack and Defense on AI-generated Content Latent-based Watermarking**

cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2509.11745v3) [paper-pdf](None)

**Authors**: De Zhang Lee, Han Fang, Hanyi Wang, Ee-Chien Chang

**Abstract**: Digital watermarks can be embedded into AI-generated content (AIGC) by initializing the generation process with starting points sampled from a secret distribution. When combined with pseudorandom error-correcting codes, such watermarked outputs can remain indistinguishable from unwatermarked objects, while maintaining robustness under whitenoise. In this paper, we go beyond indistinguishability and investigate security under removal attacks. We demonstrate that indistinguishability alone does not necessarily guarantee resistance to adversarial removal. Specifically, we propose a novel attack that exploits boundary information leaked by the locations of watermarked objects. This attack significantly reduces the distortion required to remove watermarks -- by up to a factor of $15 \times$ compared to a baseline whitenoise attack under certain settings. To mitigate such attacks, we introduce a defense mechanism that applies a secret transformation to hide the boundary, and prove that the secret transformation effectively rendering any attacker's perturbations equivalent to those of a naive whitenoise adversary. Our empirical evaluations, conducted on multiple versions of Stable Diffusion, validate the effectiveness of both the attack and the proposed defense, highlighting the importance of addressing boundary leakage in latent-based watermarking schemes.



## **19. GuardFed: A Trustworthy Federated Learning Framework Against Dual-Facet Attacks**

cs.LG

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09294v1) [paper-pdf](None)

**Authors**: Yanli Li, Yanan Zhou, Zhongliang Guo, Nan Yang, Yuning Zhang, Huaming Chen, Dong Yuan, Weiping Ding, Witold Pedrycz

**Abstract**: Federated learning (FL) enables privacy-preserving collaborative model training but remains vulnerable to adversarial behaviors that compromise model utility or fairness across sensitive groups. While extensive studies have examined attacks targeting either objective, strategies that simultaneously degrade both utility and fairness remain largely unexplored. To bridge this gap, we introduce the Dual-Facet Attack (DFA), a novel threat model that concurrently undermines predictive accuracy and group fairness. Two variants, Synchronous DFA (S-DFA) and Split DFA (Sp-DFA), are further proposed to capture distinct real-world collusion scenarios. Experimental results show that existing robust FL defenses, including hybrid aggregation schemes, fail to resist DFAs effectively. To counter these threats, we propose GuardFed, a self-adaptive defense framework that maintains a fairness-aware reference model using a small amount of clean server data augmented with synthetic samples. In each training round, GuardFed computes a dual-perspective trust score for every client by jointly evaluating its utility deviation and fairness degradation, thereby enabling selective aggregation of trustworthy updates. Extensive experiments on real-world datasets demonstrate that GuardFed consistently preserves both accuracy and fairness under diverse non-IID and adversarial conditions, achieving state-of-the-art performance compared with existing robust FL methods.



## **20. Breaking the Adversarial Robustness-Performance Trade-off in Text Classification via Manifold Purification**

cs.CL

9 pages,3 figures

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.07888v1) [paper-pdf](None)

**Authors**: Chenhao Dang, Jing Ma

**Abstract**: A persistent challenge in text classification (TC) is that enhancing model robustness against adversarial attacks typically degrades performance on clean data. We argue that this challenge can be resolved by modeling the distribution of clean samples in the encoder embedding manifold. To this end, we propose the Manifold-Correcting Causal Flow (MC^2F), a two-module system that operates directly on sentence embeddings. A Stratified Riemannian Continuous Normalizing Flow (SR-CNF) learns the density of the clean data manifold. It identifies out-of-distribution embeddings, which are then corrected by a Geodesic Purification Solver. This solver projects adversarial points back onto the learned manifold via the shortest path, restoring a clean, semantically coherent representation. We conducted extensive evaluations on text classification (TC) across three datasets and multiple adversarial attacks. The results demonstrate that our method, MC^2F, not only establishes a new state-of-the-art in adversarial robustness but also fully preserves performance on clean data, even yielding modest gains in accuracy.



## **21. Multimodal Adversarial Defense for Vision-Language Models by Leveraging One-To-Many Relationships**

cs.CV

WACV 2026 Accepted

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2405.18770v4) [paper-pdf](None)

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos, Isao Echizen

**Abstract**: Pre-trained vision-language (VL) models are highly vulnerable to adversarial attacks. However, existing defense methods primarily focus on image classification, overlooking two key aspects of VL tasks: multimodal attacks, where both image and text can be perturbed, and the one-to-many relationship of images and texts, where a single image can correspond to multiple textual descriptions and vice versa (1:N and N:1). This work is the first to explore defense strategies against multimodal attacks in VL tasks, whereas prior VL defense methods focus on vision robustness. We propose multimodal adversarial training (MAT), which incorporates adversarial perturbations in both image and text modalities during training, significantly outperforming existing unimodal defenses. Furthermore, we discover that MAT is limited by deterministic one-to-one (1:1) image-text pairs in VL training data. To address this, we conduct a comprehensive study on leveraging one-to-many relationships to enhance robustness, investigating diverse augmentation techniques. Our analysis shows that, for a more effective defense, augmented image-text pairs should be well-aligned, diverse, yet avoid distribution shift -- conditions overlooked by prior research. This work pioneers defense strategies against multimodal attacks, providing insights for building robust VLMs from both optimization and data perspectives.



## **22. Exploring the Adversarial Robustness of Face Forgery Detection with Decision-based Black-box Attacks**

cs.CV

Accepted by Knowledge-Based Systems

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2310.12017v2) [paper-pdf](None)

**Authors**: Zhaoyu Chen, Bo Li, Kaixun Jiang, Shuang Wu, Shouhong Ding, Wenqiang Zhang

**Abstract**: Face forgery generation technologies generate vivid faces, which have raised public concerns about security and privacy. Many intelligent systems, such as electronic payment and identity verification, rely on face forgery detection. Although face forgery detection has successfully distinguished fake faces, recent studies have demonstrated that face forgery detectors are very vulnerable to adversarial examples. Meanwhile, existing attacks rely on network architectures or training datasets instead of the predicted labels, which leads to a gap in attacking deployed applications. To narrow this gap, we first explore the decision-based attacks on face forgery detection. We identify challenges in directly applying existing decision-based attacks, such as perturbation initialization failure and reduced image quality. To overcome these issues, we propose cross-task perturbation to handle initialization failures by utilizing the high correlation of face features on different tasks. Additionally, inspired by the use of frequency cues in face forgery detection, we introduce the frequency decision-based attack. This attack involves adding perturbations in the frequency domain while constraining visual quality in the spatial domain. Finally, extensive experiments demonstrate that our method achieves state-of-the-art attack performance on FaceForensics++, CelebDF, and industrial APIs, with high query efficiency and guaranteed image quality. Further, the fake faces by our method can pass face forgery detection and face recognition, which exposes the security problems of face forgery detectors.



## **23. Trapped by Their Own Light: Deployable and Stealth Retroreflective Patch Attacks on Traffic Sign Recognition Systems**

cs.CR

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.10050v1) [paper-pdf](None)

**Authors**: Go Tsuruoka, Takami Sato, Qi Alfred Chen, Kazuki Nomoto, Ryunosuke Kobayashi, Yuna Tanaka, Tatsuya Mori

**Abstract**: Traffic sign recognition plays a critical role in ensuring safe and efficient transportation of autonomous vehicles but remain vulnerable to adversarial attacks using stickers or laser projections. While existing attack vectors demonstrate security concerns, they suffer from visual detectability or implementation constraints, suggesting unexplored vulnerability surfaces in TSR systems. We introduce the Adversarial Retroreflective Patch (ARP), a novel attack vector that combines the high deployability of patch attacks with the stealthiness of laser projections by utilizing retroreflective materials activated only under victim headlight illumination. We develop a retroreflection simulation method and employ black-box optimization to maximize attack effectiveness. ARP achieves $\geq$93.4\% success rate in dynamic scenarios at 35 meters and $\geq$60\% success rate against commercial TSR systems in real-world conditions. Our user study demonstrates that ARP attacks maintain near-identical stealthiness to benign signs while achieving $\geq$1.9\% higher stealthiness scores than previous patch attacks. We propose the DPR Shield defense, employing strategically placed polarized filters, which achieves $\geq$75\% defense success rates for stop signs and speed limit signs against micro-prism patches.



## **24. Debiased Dual-Invariant Defense for Adversarially Robust Person Re-Identification**

cs.CV

Accepted by AAAI 2026

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09933v1) [paper-pdf](None)

**Authors**: Yuhang Zhou, Yanxiang Zhao, Zhongyun Hua, Zhipu Liu, Zhaoquan Gu, Qing Liao, Leo Yu Zhang

**Abstract**: Person re-identification (ReID) is a fundamental task in many real-world applications such as pedestrian trajectory tracking. However, advanced deep learning-based ReID models are highly susceptible to adversarial attacks, where imperceptible perturbations to pedestrian images can cause entirely incorrect predictions, posing significant security threats. Although numerous adversarial defense strategies have been proposed for classification tasks, their extension to metric learning tasks such as person ReID remains relatively unexplored. Moreover, the several existing defenses for person ReID fail to address the inherent unique challenges of adversarially robust ReID. In this paper, we systematically identify the challenges of adversarial defense in person ReID into two key issues: model bias and composite generalization requirements. To address them, we propose a debiased dual-invariant defense framework composed of two main phases. In the data balancing phase, we mitigate model bias using a diffusion-model-based data resampling strategy that promotes fairness and diversity in training data. In the bi-adversarial self-meta defense phase, we introduce a novel metric adversarial training approach incorporating farthest negative extension softening to overcome the robustness degradation caused by the absence of classifier. Additionally, we introduce an adversarially-enhanced self-meta mechanism to achieve dual-generalization for both unseen identities and unseen attack types. Experiments demonstrate that our method significantly outperforms existing state-of-the-art defenses.



## **25. Robust Decentralized Multi-armed Bandits: From Corruption-Resilience to Byzantine-Resilience**

cs.LG

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.10344v1) [paper-pdf](None)

**Authors**: Zicheng Hu, Yuchen Wang, Cheng Chen

**Abstract**: Decentralized cooperative multi-agent multi-armed bandits (DeCMA2B) considers how multiple agents collaborate in a decentralized multi-armed bandit setting. Though this problem has been extensively studied in previous work, most existing methods remain susceptible to various adversarial attacks. In this paper, we first study DeCMA2B with adversarial corruption, where an adversary can corrupt reward observations of all agents with a limited corruption budget. We propose a robust algorithm, called DeMABAR, which ensures that each agent's individual regret suffers only an additive term proportional to the corruption budget. Then we consider a more realistic scenario where the adversary can only attack a small number of agents. Our theoretical analysis shows that the DeMABAR algorithm can also almost completely eliminate the influence of adversarial attacks and is inherently robust in the Byzantine setting, where an unknown fraction of the agents can be Byzantine, i.e., may arbitrarily select arms and communicate wrong information. We also conduct numerical experiments to illustrate the robustness and effectiveness of the proposed method.



## **26. Diversifying Counterattacks: Orthogonal Exploration for Robust CLIP Inference**

cs.CV

Accepted to AAAI-2026 Oral

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09064v1) [paper-pdf](None)

**Authors**: Chengze Jiang, Minjing Dong, Xinli Shi, Jie Gui

**Abstract**: Vision-language pre-training models (VLPs) demonstrate strong multimodal understanding and zero-shot generalization, yet remain vulnerable to adversarial examples, raising concerns about their reliability. Recent work, Test-Time Counterattack (TTC), improves robustness by generating perturbations that maximize the embedding deviation of adversarial inputs using PGD, pushing them away from their adversarial representations. However, due to the fundamental difference in optimization objectives between adversarial attacks and counterattacks, generating counterattacks solely based on gradients with respect to the adversarial input confines the search to a narrow space. As a result, the counterattacks could overfit limited adversarial patterns and lack the diversity to fully neutralize a broad range of perturbations. In this work, we argue that enhancing the diversity and coverage of counterattacks is crucial to improving adversarial robustness in test-time defense. Accordingly, we propose Directional Orthogonal Counterattack (DOC), which augments counterattack optimization by incorporating orthogonal gradient directions and momentum-based updates. This design expands the exploration of the counterattack space and increases the diversity of perturbations, which facilitates the discovery of more generalizable counterattacks and ultimately improves the ability to neutralize adversarial perturbations. Meanwhile, we present a directional sensitivity score based on averaged cosine similarity to boost DOC by improving example discrimination and adaptively modulating the counterattack strength. Extensive experiments on 16 datasets demonstrate that DOC improves adversarial robustness under various attacks while maintaining competitive clean accuracy. Code is available at https://github.com/bookman233/DOC.



## **27. Spilling the Beans: Teaching LLMs to Self-Report Their Hidden Objectives**

cs.AI

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.06626v2) [paper-pdf](None)

**Authors**: Chloe Li, Mary Phuong, Daniel Tan

**Abstract**: As AI systems become more capable of complex agentic tasks, they also become more capable of pursuing undesirable objectives and causing harm. Previous work has attempted to catch these unsafe instances by interrogating models directly about their objectives and behaviors. However, the main weakness of trusting interrogations is that models can lie. We propose self-report fine-tuning (SRFT), a simple supervised fine-tuning technique that trains models to admit their factual mistakes when asked. We show that the admission of factual errors in simple question-answering settings generalizes out-of-distribution (OOD) to the admission of hidden misaligned objectives in adversarial agentic settings. We evaluate SRFT in OOD stealth tasks, where models are instructed to complete a hidden misaligned objective alongside a user-specified objective without being caught by monitoring. After SRFT, models are more likely to confess the details of their hidden objectives when interrogated, even under strong pressure not to disclose them. Interrogation on SRFT models can detect hidden objectives with near-ceiling performance (F1 score = 0.98), while the baseline model lies when interrogated under the same conditions (F1 score = 0). Interrogation on SRFT models can further elicit the content of the hidden objective, recovering 28-100% details, compared to 0% details recovered in the baseline model and by prefilled assistant turn attacks. This provides a promising technique for promoting honesty propensity and incriminating misaligned AI systems.



## **28. Class-feature Watermark: A Resilient Black-box Watermark Against Model Extraction Attacks**

cs.CR

Accepted by AAAI'26

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.07947v1) [paper-pdf](None)

**Authors**: Yaxin Xiao, Qingqing Ye, Zi Liang, Haoyang Li, RongHua Li, Huadi Zheng, Haibo Hu

**Abstract**: Machine learning models constitute valuable intellectual property, yet remain vulnerable to model extraction attacks (MEA), where adversaries replicate their functionality through black-box queries. Model watermarking counters MEAs by embedding forensic markers for ownership verification. Current black-box watermarks prioritize MEA survival through representation entanglement, yet inadequately explore resilience against sequential MEAs and removal attacks. Our study reveals that this risk is underestimated because existing removal methods are weakened by entanglement. To address this gap, we propose Watermark Removal attacK (WRK), which circumvents entanglement constraints by exploiting decision boundaries shaped by prevailing sample-level watermark artifacts. WRK effectively reduces watermark success rates by at least 88.79% across existing watermarking benchmarks.   For robust protection, we propose Class-Feature Watermarks (CFW), which improve resilience by leveraging class-level artifacts. CFW constructs a synthetic class using out-of-domain samples, eliminating vulnerable decision boundaries between original domain samples and their artifact-modified counterparts (watermark samples). CFW concurrently optimizes both MEA transferability and post-MEA stability. Experiments across multiple domains show that CFW consistently outperforms prior methods in resilience, maintaining a watermark success rate of at least 70.15% in extracted models even under the combined MEA and WRK distortion, while preserving the utility of protected models.



## **29. GreedyPixel: Fine-Grained Black-Box Adversarial Attack Via Greedy Algorithm**

cs.CV

IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2501.14230v3) [paper-pdf](None)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Christopher Leckie, Isao Echizen

**Abstract**: Deep neural networks are highly vulnerable to adversarial examples, which are inputs with small, carefully crafted perturbations that cause misclassification -- making adversarial attacks a critical tool for evaluating robustness. Existing black-box methods typically entail a trade-off between precision and flexibility: pixel-sparse attacks (e.g., single- or few-pixel attacks) provide fine-grained control but lack adaptability, whereas patch- or frequency-based attacks improve efficiency or transferability, but at the cost of producing larger and less precise perturbations. We present GreedyPixel, a fine-grained black-box attack method that performs brute-force-style, per-pixel greedy optimization guided by a surrogate-derived priority map and refined by means of query feedback. It evaluates each coordinate directly without any gradient information, guaranteeing monotonic loss reduction and convergence to a coordinate-wise optimum, while also yielding near white-box-level precision and pixel-wise sparsity and perceptual quality. On the CIFAR-10 and ImageNet datasets, spanning convolutional neural networks (CNNs) and Transformer models, GreedyPixel achieved state-of-the-art success rates with visually imperceptible perturbations, effectively bridging the gap between black-box practicality and white-box performance. The implementation is available at https://github.com/azrealwang/greedypixel.



## **30. Abstract Gradient Training: A Unified Certification Framework for Data Poisoning, Unlearning, and Differential Privacy**

cs.LG

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09400v1) [paper-pdf](None)

**Authors**: Philip Sosnin, Matthew Wicker, Josh Collyer, Calvin Tsay

**Abstract**: The impact of inference-time data perturbation (e.g., adversarial attacks) has been extensively studied in machine learning, leading to well-established certification techniques for adversarial robustness. In contrast, certifying models against training data perturbations remains a relatively under-explored area. These perturbations can arise in three critical contexts: adversarial data poisoning, where an adversary manipulates training samples to corrupt model performance; machine unlearning, which requires certifying model behavior under the removal of specific training data; and differential privacy, where guarantees must be given with respect to substituting individual data points. This work introduces Abstract Gradient Training (AGT), a unified framework for certifying robustness of a given model and training procedure to training data perturbations, including bounded perturbations, the removal of data points, and the addition of new samples. By bounding the reachable set of parameters, i.e., establishing provable parameter-space bounds, AGT provides a formal approach to analyzing the behavior of models trained via first-order optimization methods.



## **31. Siren: A Learning-Based Multi-Turn Attack Framework for Simulating Real-World Human Jailbreak Behaviors**

cs.CL

Accepted at ACSAC 2025

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2501.14250v2) [paper-pdf](None)

**Authors**: Yi Zhao, Youzhi Zhang

**Abstract**: Large language models (LLMs) are widely used in real-world applications, raising concerns about their safety and trustworthiness. While red-teaming with jailbreak prompts exposes the vulnerabilities of LLMs, current efforts focus primarily on single-turn attacks, overlooking the multi-turn strategies used by real-world adversaries. Existing multi-turn methods rely on static patterns or predefined logical chains, failing to account for the dynamic strategies during attacks. We propose Siren, a learning-based multi-turn attack framework designed to simulate real-world human jailbreak behaviors. Siren consists of three stages: (1) MiniMax-driven training set construction utilizing Turn-Level LLM feedback, (2) post-training attackers with supervised fine-tuning (SFT) and direct preference optimization (DPO), and (3) interactions between the attacking and target LLMs. Experiments demonstrate that Siren achieves an attack success rate (ASR) of 90% with LLaMA-3-8B as the attacker against Gemini-1.5-Pro as the target model, and 70% with Mistral-7B against GPT-4o, significantly outperforming single-turn baselines. Moreover, Siren with a 7B-scale model achieves performance comparable to a multi-turn baseline that leverages GPT-4o as the attacker, while requiring fewer turns and employing decomposition strategies that are better semantically aligned with attack goals. We hope Siren inspires the development of stronger defenses against advanced multi-turn jailbreak attacks under realistic scenarios. Code is available at https://github.com/YiyiyiZhao/siren. Warning: This paper contains potentially harmful text.



## **32. UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning**

cs.CR

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2503.01908v3) [paper-pdf](None)

**Authors**: Jiawei Zhang, Shuang Yang, Bo Li

**Abstract**: Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements amplify the risks of adversarial attacks, especially when agents can access sensitive external functionalities. Nevertheless, manipulating LLM agents into performing targeted malicious actions or invoking specific tools remains challenging, as these agents extensively reason or plan before executing final actions. In this work, we present UDora, a unified red teaming framework designed for LLM agents that dynamically hijacks the agent's reasoning processes to compel malicious behavior. Specifically, UDora first generates the model's reasoning trace for the given task, then automatically identifies optimal points within this trace to insert targeted perturbations. The resulting perturbed reasoning is then used as a surrogate response for optimization. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets. The code is available at https://github.com/AI-secure/UDora.



## **33. RedDiffuser: Red Teaming Vision-Language Models for Toxic Continuation via Reinforced Stable Diffusion**

cs.CV

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2503.06223v4) [paper-pdf](None)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Vision-Language Models (VLMs) are vulnerable to jailbreak attacks, where adversaries bypass safety mechanisms to elicit harmful outputs. In this work, we examine an insidious variant of this threat: toxic continuation. Unlike standard jailbreaks that rely solely on malicious instructions, toxic continuation arises when the model is given a malicious input alongside a partial toxic output, resulting in harmful completions. This vulnerability poses a unique challenge in multimodal settings, where even subtle image variations can disproportionately affect the model's response. To this end, we propose RedDiffuser (RedDiff), the first red teaming framework that uses reinforcement learning to fine-tune diffusion models into generating natural-looking adversarial images that induce toxic continuations. RedDiffuser integrates a greedy search procedure for selecting candidate image prompts with reinforcement fine-tuning that jointly promotes toxic output and semantic coherence. Experiments demonstrate that RedDiffuser significantly increases the toxicity rate in LLaVA outputs by 10.69% and 8.91% on the original and hold-out sets, respectively. It also exhibits strong transferability, increasing toxicity rates on Gemini by 5.1% and on LLaMA-Vision by 26.83%. These findings uncover a cross-modal toxicity amplification vulnerability in current VLM alignment, highlighting the need for robust multimodal red teaming. We will release the RedDiffuser codebase to support future research.



## **34. Private Remote Phase Estimation over a Lossy Quantum Channel**

quant-ph

4 + 5 pages; 2 figures

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.09123v1) [paper-pdf](None)

**Authors**: Farzad Kianvash, Marco Barbieri, Matteo Rosati

**Abstract**: Private remote quantum sensing (PRQS) aims at estimating a parameter at a distant location by transmitting quantum states on an insecure quantum channel, limiting information leakage and disruption of the estimation itself from an adversary. Previous results highlighted that one can bound the estimation performance in terms of the observed noise. However, if no assumptions are placed on the channel model, such bounds are very loose and severely limit the estimation. We propose and analyse a PRQS using, for the first time to our knowledge, continuous-variable states in the single-user setting. Assuming a typical class of lossy attacks and employing tools from quantum communication, we calculate the true estimation error and privacy of our protocol, both in the asymptotic limit of many channel uses and in the finite-size regime. Our results show that a realistic channel-model assumption, which can be validated with measurement data, allows for a much tighter quantification of the estimation error and privacy for all practical purposes.



## **35. eXIAA: eXplainable Injections for Adversarial Attack**

cs.LG

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.10088v1) [paper-pdf](None)

**Authors**: Leonardo Pesce, Jiawen Wei, Gianmarco Mengaldo

**Abstract**: Post-hoc explainability methods are a subset of Machine Learning (ML) that aim to provide a reason for why a model behaves in a certain way. In this paper, we show a new black-box model-agnostic adversarial attack for post-hoc explainable Artificial Intelligence (XAI), particularly in the image domain. The goal of the attack is to modify the original explanations while being undetected by the human eye and maintain the same predicted class. In contrast to previous methods, we do not require any access to the model or its weights, but only to the model's computed predictions and explanations. Additionally, the attack is accomplished in a single step while significantly changing the provided explanations, as demonstrated by empirical evaluation. The low requirements of our method expose a critical vulnerability in current explainability methods, raising concerns about their reliability in safety-critical applications. We systematically generate attacks based on the explanations generated by post-hoc explainability methods (saliency maps, integrated gradients, and DeepLIFT SHAP) for pretrained ResNet-18 and ViT-B16 on ImageNet. The results show that our attacks could lead to dramatically different explanations without changing the predictive probabilities. We validate the effectiveness of our attack, compute the induced change based on the explanation with mean absolute difference, and verify the closeness of the original image and the corrupted one with the Structural Similarity Index Measure (SSIM).



## **36. CertMask: Certifiable Defense Against Adversarial Patches via Theoretically Optimal Mask Coverage**

cs.CV

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09834v1) [paper-pdf](None)

**Authors**: Xuntao Lyu, Ching-Chi Lin, Abdullah Al Arafat, Georg von der Brüggen, Jian-Jia Chen, Zhishan Guo

**Abstract**: Adversarial patch attacks inject localized perturbations into images to mislead deep vision models. These attacks can be physically deployed, posing serious risks to real-world applications. In this paper, we propose CertMask, a certifiably robust defense that constructs a provably sufficient set of binary masks to neutralize patch effects with strong theoretical guarantees. While the state-of-the-art approach (PatchCleanser) requires two rounds of masking and incurs $O(n^2)$ inference cost, CertMask performs only a single round of masking with $O(n)$ time complexity, where $n$ is the cardinality of the mask set to cover an input image. Our proposed mask set is computed using a mathematically rigorous coverage strategy that ensures each possible patch location is covered at least $k$ times, providing both efficiency and robustness. We offer a theoretical analysis of the coverage condition and prove its sufficiency for certification. Experiments on ImageNet, ImageNette, and CIFAR-10 show that CertMask improves certified robust accuracy by up to +13.4\% over PatchCleanser, while maintaining clean accuracy nearly identical to the vanilla model.



## **37. Hail to the Thief: Exploring Attacks and Defenses in Decentralised GRPO**

cs.LG

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09780v1) [paper-pdf](None)

**Authors**: Nikolay Blagoev, Oğuzhan Ersoy, Lydia Yiyu Chen

**Abstract**: Group Relative Policy Optimization (GRPO) has demonstrated great utilization in post-training of Large Language Models (LLMs). In GRPO, prompts are answered by the model and, through reinforcement learning, preferred completions are learnt. Owing to the small communication volume, GRPO is inherently suitable for decentralised training as the prompts can be concurrently answered by multiple nodes and then exchanged in the forms of strings. In this work, we present the first adversarial attack in decentralised GRPO. We demonstrate that malicious parties can poison such systems by injecting arbitrary malicious tokens in benign models in both out-of-context and in-context attacks. Using empirical examples of math and coding tasks, we show that adversarial attacks can easily poison the benign nodes, polluting their local LLM post-training, achieving attack success rates up to 100% in as few as 50 iterations. We propose two ways to defend against these attacks, depending on whether all users train the same model or different models. We show that these defenses can achieve stop rates of up to 100%, making the attack impossible.



## **38. Can Current Detectors Catch Face-to-Voice Deepfake Attacks?**

cs.CR

8 pages, Accepted at Workshop on AI for Cyber Threat Intelligence, co-located with ACSAC 2025

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2510.21004v2) [paper-pdf](None)

**Authors**: Nguyen Linh Bao Nguyen, Alsharif Abuadbba, Kristen Moore, Tingmin Wu

**Abstract**: The rapid advancement of generative models has enabled the creation of increasingly stealthy synthetic voices, commonly referred to as audio deepfakes. A recent technique, FOICE [USENIX'24], demonstrates a particularly alarming capability: generating a victim's voice from a single facial image, without requiring any voice sample. By exploiting correlations between facial and vocal features, FOICE produces synthetic voices realistic enough to bypass industry-standard authentication systems, including WeChat Voiceprint and Microsoft Azure. This raises serious security concerns, as facial images are far easier for adversaries to obtain than voice samples, dramatically lowering the barrier to large-scale attacks. In this work, we investigate two core research questions: (RQ1) can state-of-the-art audio deepfake detectors reliably detect FOICE-generated speech under clean and noisy conditions, and (RQ2) whether fine-tuning these detectors on FOICE data improves detection without overfitting, thereby preserving robustness to unseen voice generators such as SpeechT5.   Our study makes three contributions. First, we present the first systematic evaluation of FOICE detection, showing that leading detectors consistently fail under both standard and noisy conditions. Second, we introduce targeted fine-tuning strategies that capture FOICE-specific artifacts, yielding significant accuracy improvements. Third, we assess generalization after fine-tuning, revealing trade-offs between specialization to FOICE and robustness to unseen synthesis pipelines. These findings expose fundamental weaknesses in today's defenses and motivate new architectures and training protocols for next-generation audio deepfake detection.



## **39. VFEFL: Privacy-Preserving Federated Learning against Malicious Clients via Verifiable Functional Encryption**

cs.CR

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2506.12846v3) [paper-pdf](None)

**Authors**: Nina Cai, Jinguang Han, Weizhi Meng

**Abstract**: Federated learning is a promising distributed learning paradigm that enables collaborative model training without exposing local client data, thereby protect data privacy. However, it also brings new threats and challenges. The advancement of model inversion attacks has rendered the plaintext transmission of local models insecure, while the distributed nature of federated learning makes it particularly vulnerable to attacks raised by malicious clients. To protect data privacy and prevent malicious client attacks, this paper proposes a privacy-preserving federated learning framework based on verifiable functional encryption, without a non-colluding dual-server setup or additional trusted third-party. Specifically, we propose a novel decentralized verifiable functional encryption (DVFE) scheme that enables the verification of specific relationships over multi-dimensional ciphertexts. This scheme is formally treated, in terms of definition, security model and security proof. Furthermore, based on the proposed DVFE scheme, we design a privacy-preserving federated learning framework VFEFL that incorporates a novel robust aggregation rule to detect malicious clients, enabling the effective training of high-accuracy models under adversarial settings. Finally, we provide formal analysis and empirical evaluation of the proposed schemes. The results demonstrate that our approach achieves the desired privacy protection, robustness, verifiability and fidelity, while eliminating the reliance on non-colluding dual-server settings or trusted third parties required by existing methods.



## **40. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

cs.CR

14 pages, 5 figures; published in EMNLP 2025 ; Code at: https://github.com/dsbuddy/GAP-LLM-Safety

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2501.18638v3) [paper-pdf](None)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.



## **41. Adversarial Bias: Data Poisoning Attacks on Fairness**

cs.LG

15 pages, 9 figures, shortened version in BigData 2025

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.08331v1) [paper-pdf](None)

**Authors**: Eunice Chan, Hanghang Tong

**Abstract**: With the growing adoption of AI and machine learning systems in real-world applications, ensuring their fairness has become increasingly critical. The majority of the work in algorithmic fairness focus on assessing and improving the fairness of machine learning systems. There is relatively little research on fairness vulnerability, i.e., how an AI system's fairness can be intentionally compromised. In this work, we first provide a theoretical analysis demonstrating that a simple adversarial poisoning strategy is sufficient to induce maximally unfair behavior in naive Bayes classifiers. Our key idea is to strategically inject a small fraction of carefully crafted adversarial data points into the training set, biasing the model's decision boundary to disproportionately affect a protected group while preserving generalizable performance. To illustrate the practical effectiveness of our method, we conduct experiments across several benchmark datasets and models. We find that our attack significantly outperforms existing methods in degrading fairness metrics across multiple models and datasets, often achieving substantially higher levels of unfairness with a comparable or only slightly worse impact on accuracy. Notably, our method proves effective on a wide range of models, in contrast to prior work, demonstrating a robust and potent approach to compromising the fairness of machine learning systems.



## **42. HybridGuard: Enhancing Minority-Class Intrusion Detection in Dew-Enabled Edge-of-Things Networks**

cs.CR

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2511.07793v1) [paper-pdf](None)

**Authors**: Binayak Kara, Ujjwal Sahua, Ciza Thomas, Jyoti Prakash Sahoo

**Abstract**: Securing Dew-Enabled Edge-of-Things (EoT) networks against sophisticated intrusions is a critical challenge. This paper presents HybridGuard, a framework that integrates machine learning and deep learning to improve intrusion detection. HybridGuard addresses data imbalance through mutual information based feature selection, ensuring that the most relevant features are used to improve detection performance, especially for minority attack classes. The framework leverages Wasserstein Conditional Generative Adversarial Networks with Gradient Penalty (WCGAN-GP) to further reduce class imbalance and enhance detection precision. It adopts a two-phase architecture called DualNetShield to support advanced traffic analysis and anomaly detection, improving the granular identification of threats in complex EoT environments. HybridGuard is evaluated on the UNSW-NB15, CIC-IDS-2017, and IOTID20 datasets, where it demonstrates strong performance across diverse attack scenarios and outperforms existing solutions in adapting to evolving cybersecurity threats. This approach establishes HybridGuard as an effective tool for protecting EoT networks against modern intrusions.



## **43. DeepTracer: Tracing Stolen Model via Deep Coupled Watermarks**

cs.CR

Extended version of the paper accepted by AAAI 2026

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.08985v1) [paper-pdf](None)

**Authors**: Yunfei Yang, Xiaojun Chen, Yuexin Xuan, Zhendong Zhao, Xin Zhao, He Li

**Abstract**: Model watermarking techniques can embed watermark information into the protected model for ownership declaration by constructing specific input-output pairs. However, existing watermarks are easily removed when facing model stealing attacks, and make it difficult for model owners to effectively verify the copyright of stolen models. In this paper, we analyze the root cause of the failure of current watermarking methods under model stealing scenarios and then explore potential solutions. Specifically, we introduce a robust watermarking framework, DeepTracer, which leverages a novel watermark samples construction method and a same-class coupling loss constraint. DeepTracer can incur a high-coupling model between watermark task and primary task that makes adversaries inevitably learn the hidden watermark task when stealing the primary task functionality. Furthermore, we propose an effective watermark samples filtering mechanism that elaborately select watermark key samples used in model ownership verification to enhance the reliability of watermarks. Extensive experiments across multiple datasets and models demonstrate that our method surpasses existing approaches in defending against various model stealing attacks, as well as watermark attacks, and achieves new state-of-the-art effectiveness and robustness.



## **44. SIFT-Graph: Benchmarking Multimodal Defense Against Image Adversarial Attacks With Robust Feature Graph**

cs.CV

Accepted by ICCV2025 Workshop, short paper

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2511.08810v1) [paper-pdf](None)

**Authors**: Jingjie He, Weijie Liang, Zihan Shan, Matthew Caesar

**Abstract**: Adversarial attacks expose a fundamental vulnerability in modern deep vision models by exploiting their dependence on dense, pixel-level representations that are highly sensitive to imperceptible perturbations. Traditional defense strategies typically operate within this fragile pixel domain, lacking mechanisms to incorporate inherently robust visual features. In this work, we introduce SIFT-Graph, a multimodal defense framework that enhances the robustness of traditional vision models by aggregating structurally meaningful features extracted from raw images using both handcrafted and learned modalities. Specifically, we integrate Scale-Invariant Feature Transform keypoints with a Graph Attention Network to capture scale and rotation invariant local structures that are resilient to perturbations. These robust feature embeddings are then fused with traditional vision model, such as Vision Transformer and Convolutional Neural Network, to form a unified, structure-aware and perturbation defensive model. Preliminary results demonstrate that our method effectively improves the visual model robustness against gradient-based white box adversarial attacks, while incurring only a marginal drop in clean accuracy.



## **45. ConfGuard: A Simple and Effective Backdoor Detection for Large Language Models**

cs.CR

This is an extended version of the copyrighted publication at AAAI

**SubmitDate**: 2025-11-12    [abs](http://arxiv.org/abs/2508.01365v3) [paper-pdf](None)

**Authors**: Zihan Wang, Rui Zhang, Hongwei Li, Wenshu Fan, Wenbo Jiang, Qingchuan Zhao, Guowen Xu

**Abstract**: Backdoor attacks pose a significant threat to Large Language Models (LLMs), where adversaries can embed hidden triggers to manipulate LLM's outputs. Most existing defense methods, primarily designed for classification tasks, are ineffective against the autoregressive nature and vast output space of LLMs, thereby suffering from poor performance and high latency. To address these limitations, we investigate the behavioral discrepancies between benign and backdoored LLMs in output space. We identify a critical phenomenon which we term sequence lock: a backdoored model generates the target sequence with abnormally high and consistent confidence compared to benign generation. Building on this insight, we propose ConfGuard, a lightweight and effective detection method that monitors a sliding window of token confidences to identify sequence lock. Extensive experiments demonstrate ConfGuard achieves a near 100\% true positive rate (TPR) and a negligible false positive rate (FPR) in the vast majority of cases. Crucially, the ConfGuard enables real-time detection almost without additional latency, making it a practical backdoor defense for real-world LLM deployments.



## **46. Identifying the Smallest Adversarial Load Perturbation that Renders DC-OPF Infeasible**

eess.SY

**SubmitDate**: 2025-11-13    [abs](http://arxiv.org/abs/2507.07850v2) [paper-pdf](None)

**Authors**: Samuel Chevalier, William A. Wheeler

**Abstract**: What is the globally smallest load perturbation that renders DC-OPF infeasible? Reliably identifying such "adversarial attack" perturbations has useful applications in a variety of emerging grid-related contexts, including machine learning performance verification, cybersecurity, and operational robustness of power systems dominated by stochastic renewable energy resources. In this paper, we formulate the inherently nonconvex adversarial attack problem by applying a parameterized version of Farkas' lemma to a perturbed set of DC-OPF equations. Since the resulting formulation is very hard to globally optimize, we also propose a parameterized generation control policy which, when applied to the primal DC-OPF problem, provides solvability guarantees. Together, these nonconvex problems provide guaranteed upper and lower bounds on adversarial attack size; by combining them into a single optimization problem, we can efficiently "squeeze" these bounds towards a common global solution. We apply these methods on a range of small- to medium-sized test cases from PGLib, benchmarking our results against the best adversarial attack lower bounds provided by Gurobi 12.0's spatial Branch and Bound solver.



## **47. Unlearning Imperative: Securing Trustworthy and Responsible LLMs through Engineered Forgetting**

cs.LG

14 pages, 4 figures, 4 tables

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09855v1) [paper-pdf](None)

**Authors**: James Jin Kang, Dang Bui, Thanh Pham, Huo-Chong Ling

**Abstract**: The growing use of large language models in sensitive domains has exposed a critical weakness: the inability to ensure that private information can be permanently forgotten. Yet these systems still lack reliable mechanisms to guarantee that sensitive information can be permanently removed once it has been used. Retraining from the beginning is prohibitively costly, and existing unlearning methods remain fragmented, difficult to verify, and often vulnerable to recovery. This paper surveys recent research on machine unlearning for LLMs and considers how far current approaches can address these challenges. We review methods for evaluating whether forgetting has occurred, the resilience of unlearned models against adversarial attacks, and mechanisms that can support user trust when model complexity or proprietary limits restrict transparency. Technical solutions such as differential privacy, homomorphic encryption, federated learning, and ephemeral memory are examined alongside institutional safeguards including auditing practices and regulatory frameworks. The review finds steady progress, but robust and verifiable unlearning is still unresolved. Efficient techniques that avoid costly retraining, stronger defenses against adversarial recovery, and governance structures that reinforce accountability are needed if LLMs are to be deployed safely in sensitive applications. By integrating technical and organizational perspectives, this study outlines a pathway toward AI systems that can be required to forget, while maintaining both privacy and public trust.



## **48. Slice-Aware Spoofing Detection in 5G Networks Using Lightweight Machine Learning**

cs.CR

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.09610v1) [paper-pdf](None)

**Authors**: Daniyal Ganiuly, Nurzhau Bolatbek

**Abstract**: The increasing virtualization of fifth generation (5G) networks expands the attack surface of the user plane, making spoofing a persistent threat to slice integrity and service reliability. This study presents a slice-aware lightweight machine-learning framework for detecting spoofing attacks within 5G network slices. The framework was implemented on a reproducible Open5GS and srsRAN testbed emulating three service classes such as enhanced Mobile Broadband (eMBB), Ultra-Reliable Low-Latency Communication (URLLC), and massive Machine-Type Communication (mMTC) under controlled benign and adversarial traffic. Two efficient classifiers, Logistic Regression and Random Forest, were trained independently for each slice using statistical flow features derived from mirrored user-plane traffic. Slice-aware training improved detection accuracy by up to 5% and achieved F1-scores between 0.93 and 0.96 while maintaining real-time operation on commodity edge hardware. The results demonstrate that aligning security intelligence with slice boundaries enhances detection reliability and preserves operational isolation, enabling practical deployment in 5G network-security environments. Conceptually, the work bridges network-security architecture and adaptive machine learning by showing that isolation-aware intelligence can achieve scalable, privacy-preserving spoofing defense without high computational cost.



## **49. Boosting Adversarial Transferability via Ensemble Non-Attention**

cs.CV

16 pages, 11 figures, accepted by AAAI 2026

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.08937v2) [paper-pdf](None)

**Authors**: Yipeng Zou, Qin Liu, Jie Wu, Yu Peng, Guo Chen, Hui Zhou, Guanghui Ye

**Abstract**: Ensemble attacks integrate the outputs of surrogate models with diverse architectures, which can be combined with various gradient-based attacks to improve adversarial transferability. However, previous work shows unsatisfactory attack performance when transferring across heterogeneous model architectures. The main reason is that the gradient update directions of heterogeneous surrogate models differ widely, making it hard to reduce the gradient variance of ensemble models while making the best of individual model. To tackle this challenge, we design a novel ensemble attack, NAMEA, which for the first time integrates the gradients from the non-attention areas of ensemble models into the iterative gradient optimization process. Our design is inspired by the observation that the attention areas of heterogeneous models vary sharply, thus the non-attention areas of ViTs are likely to be the focus of CNNs and vice versa. Therefore, we merge the gradients respectively from the attention and non-attention areas of ensemble models so as to fuse the transfer information of CNNs and ViTs. Specifically, we pioneer a new way of decoupling the gradients of non-attention areas from those of attention areas, while merging gradients by meta-learning. Empirical evaluations on ImageNet dataset indicate that NAMEA outperforms AdaEA and SMER, the state-of-the-art ensemble attacks by an average of 15.0% and 9.6%, respectively. This work is the first attempt to explore the power of ensemble non-attention in boosting cross-architecture transferability, providing new insights into launching ensemble attacks.



## **50. Biologically-Informed Hybrid Membership Inference Attacks on Generative Genomic Models**

cs.CR

**SubmitDate**: **NEW** 2025-11-14    [abs](http://arxiv.org/abs/2511.07503v2) [paper-pdf](None)

**Authors**: Asia Belfiore, Jonathan Passerat-Palmbach, Dmitrii Usynin

**Abstract**: The increased availability of genetic data has transformed genomics research, but raised many privacy concerns regarding its handling due to its sensitive nature. This work explores the use of language models (LMs) for the generation of synthetic genetic mutation profiles, leveraging differential privacy (DP) for the protection of sensitive genetic data. We empirically evaluate the privacy guarantees of our DP modes by introducing a novel Biologically-Informed Hybrid Membership Inference Attack (biHMIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. Our experiments show that both small and large transformer GPT-like models are viable synthetic variant generators for small-scale genomics, and that our hybrid attack leads, on average, to higher adversarial success compared to traditional metric-based MIAs.



