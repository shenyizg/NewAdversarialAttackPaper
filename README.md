# Latest Adversarial Attack Papers
**update at 2025-05-12 14:21:15**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Self-Supervised Federated GNSS Spoofing Detection with Opportunistic Data**

cs.CR

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2505.06171v1) [paper-pdf](http://arxiv.org/pdf/2505.06171v1)

**Authors**: Wenjie Liu, Panos Papadimitratos

**Abstract**: Global navigation satellite systems (GNSS) are vulnerable to spoofing attacks, with adversarial signals manipulating the location or time information of receivers, potentially causing severe disruptions. The task of discerning the spoofing signals from benign ones is naturally relevant for machine learning, thus recent interest in applying it for detection. While deep learning-based methods are promising, they require extensive labeled datasets, consume significant computational resources, and raise privacy concerns due to the sensitive nature of position data. This is why this paper proposes a self-supervised federated learning framework for GNSS spoofing detection. It consists of a cloud server and local mobile platforms. Each mobile platform employs a self-supervised anomaly detector using long short-term memory (LSTM) networks. Labels for training are generated locally through a spoofing-deviation prediction algorithm, ensuring privacy. Local models are trained independently, and only their parameters are uploaded to the cloud server, which aggregates them into a global model using FedAvg. The updated global model is then distributed back to the mobile platforms and trained iteratively. The evaluation shows that our self-supervised federated learning framework outperforms position-based and deep learning-based methods in detecting spoofing attacks while preserving data privacy.



## **2. Realistic Adversarial Attacks for Robustness Evaluation of Trajectory Prediction Models via Future State Perturbation**

cs.LG

20 pages, 3 figures

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2505.06134v1) [paper-pdf](http://arxiv.org/pdf/2505.06134v1)

**Authors**: Julian F. Schumann, Jeroen Hagenus, Frederik Baymler Mathiesen, Arkady Zgonnikov

**Abstract**: Trajectory prediction is a key element of autonomous vehicle systems, enabling them to anticipate and react to the movements of other road users. Evaluating the robustness of prediction models against adversarial attacks is essential to ensure their reliability in real-world traffic. However, current approaches tend to focus on perturbing the past positions of surrounding agents, which can generate unrealistic scenarios and overlook critical vulnerabilities. This limitation may result in overly optimistic assessments of model performance in real-world conditions.   In this work, we demonstrate that perturbing not just past but also future states of adversarial agents can uncover previously undetected weaknesses and thereby provide a more rigorous evaluation of model robustness. Our novel approach incorporates dynamic constraints and preserves tactical behaviors, enabling more effective and realistic adversarial attacks. We introduce new performance measures to assess the realism and impact of these adversarial trajectories. Testing our method on a state-of-the-art prediction model revealed significant increases in prediction errors and collision rates under adversarial conditions. Qualitative analysis further showed that our attacks can expose critical weaknesses, such as the inability of the model to detect potential collisions in what appear to be safe predictions. These results underscore the need for more comprehensive adversarial testing to better evaluate and improve the reliability of trajectory prediction models for autonomous vehicles.



## **3. Sparsification Under Siege: Defending Against Poisoning Attacks in Communication-Efficient Federated Learning**

cs.CR

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2505.01454v2) [paper-pdf](http://arxiv.org/pdf/2505.01454v2)

**Authors**: Zhiyong Jin, Runhua Xu, Chao Li, Yizhong Liu, Jianxin Li

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed clients while preserving data privacy, yet it faces significant challenges in communication efficiency and vulnerability to poisoning attacks. While sparsification techniques mitigate communication overhead by transmitting only critical model parameters, they inadvertently amplify security risks: adversarial clients can exploit sparse updates to evade detection and degrade model performance. Existing defense mechanisms, designed for standard FL communication scenarios, are ineffective in addressing these vulnerabilities within sparsified FL. To bridge this gap, we propose FLARE, a novel federated learning framework that integrates sparse index mask inspection and model update sign similarity analysis to detect and mitigate poisoning attacks in sparsified FL. Extensive experiments across multiple datasets and adversarial scenarios demonstrate that FLARE significantly outperforms existing defense strategies, effectively securing sparsified FL against poisoning attacks while maintaining communication efficiency.



## **4. A Taxonomy of Attacks and Defenses in Split Learning**

cs.CR

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2505.05872v1) [paper-pdf](http://arxiv.org/pdf/2505.05872v1)

**Authors**: Aqsa Shabbir, Halil İbrahim Kanpak, Alptekin Küpçü, Sinem Sav

**Abstract**: Split Learning (SL) has emerged as a promising paradigm for distributed deep learning, allowing resource-constrained clients to offload portions of their model computation to servers while maintaining collaborative learning. However, recent research has demonstrated that SL remains vulnerable to a range of privacy and security threats, including information leakage, model inversion, and adversarial attacks. While various defense mechanisms have been proposed, a systematic understanding of the attack landscape and corresponding countermeasures is still lacking. In this study, we present a comprehensive taxonomy of attacks and defenses in SL, categorizing them along three key dimensions: employed strategies, constraints, and effectiveness. Furthermore, we identify key open challenges and research gaps in SL based on our systematization, highlighting potential future directions.



## **5. Timestamp Manipulation: Timestamp-based Nakamoto-style Blockchains are Vulnerable**

cs.CR

27 pages, 6 figures

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2505.05328v2) [paper-pdf](http://arxiv.org/pdf/2505.05328v2)

**Authors**: Junjie Hu, Na Ruan

**Abstract**: We introduce two advanced attack strategies, the Unrestricted Uncle Maker (UUM) Attack and the Staircase-Unrestricted Uncle Maker (SUUM) Attack, which fundamentally threaten the security of timestamp-based Nakamoto-style blockchains by inflicting permanent systemic harm. Unlike prior work that merely enhances adversarial rewards, these attacks exploit vulnerabilities in timestamp manipulation and fork selection rules to irreversibly destabilize blockchain fairness and incentive mechanisms. Specifically, the SUUM attack enables adversaries to persistently launch attacks at zero cost, eliminating constraints on block withholding and risk-free conditions, while systematically maximizing rewards through coordinated timestamp adjustments and strategic block release.   Our analysis demonstrates that SUUM adversaries achieve disproportionate reward advantages over both UUM and the original Riskless Uncle Maker (RUM) Attack [CCS '23], with all three strategies surpassing honest mining. Crucially, SUUM's cost-free persistence allows adversaries to indefinitely drain rewards from honest participants by maintaining minimal difficulty risks through precise timestamp manipulation. This creates a self-reinforcing cycle: adversaries amplify their profits while suppressing honest returns, thereby permanently eroding the protocol's security assumptions. Through rigorous theoretical modeling and simulations, we validate how SUUM's combination of timestamp tampering, block withholding, and difficulty risk control enables unmitigated exploitation of consensus mechanisms. This work underscores the existential risks posed by timestamp-based Nakamoto-style protocols and advocates urgent countermeasures to ensure long-term stability.



## **6. Adaptive Gradient Clipping for Robust Federated Learning**

cs.LG

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2405.14432v5) [paper-pdf](http://arxiv.org/pdf/2405.14432v5)

**Authors**: Youssef Allouah, Rachid Guerraoui, Nirupam Gupta, Ahmed Jellouli, Geovani Rizk, John Stephan

**Abstract**: Robust federated learning aims to maintain reliable performance despite the presence of adversarial or misbehaving workers. While state-of-the-art (SOTA) robust distributed gradient descent (Robust-DGD) methods were proven theoretically optimal, their empirical success has often relied on pre-aggregation gradient clipping. However, existing static clipping strategies yield inconsistent results: enhancing robustness against some attacks while being ineffective or even detrimental against others. To address this limitation, we propose a principled adaptive clipping strategy, Adaptive Robust Clipping (ARC), which dynamically adjusts clipping thresholds based on the input gradients. We prove that ARC not only preserves the theoretical robustness guarantees of SOTA Robust-DGD methods but also provably improves asymptotic convergence when the model is well-initialized. Extensive experiments on benchmark image classification tasks confirm these theoretical insights, demonstrating that ARC significantly enhances robustness, particularly in highly heterogeneous and adversarial settings.



## **7. Persistence of Backdoor-based Watermarks for Neural Networks: A Comprehensive Evaluation**

cs.LG

Accepted by IEEE Transactions on Neural Networks and Learning Systems  (TNNLS)

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2501.02704v3) [paper-pdf](http://arxiv.org/pdf/2501.02704v3)

**Authors**: Anh Tu Ngo, Chuan Song Heng, Nandish Chattopadhyay, Anupam Chattopadhyay

**Abstract**: Deep Neural Networks (DNNs) have gained considerable traction in recent years due to the unparalleled results they gathered. However, the cost behind training such sophisticated models is resource intensive, resulting in many to consider DNNs to be intellectual property (IP) to model owners. In this era of cloud computing, high-performance DNNs are often deployed all over the internet so that people can access them publicly. As such, DNN watermarking schemes, especially backdoor-based watermarks, have been actively developed in recent years to preserve proprietary rights. Nonetheless, there lies much uncertainty on the robustness of existing backdoor watermark schemes, towards both adversarial attacks and unintended means such as fine-tuning neural network models. One reason for this is that no complete guarantee of robustness can be assured in the context of backdoor-based watermark. In this paper, we extensively evaluate the persistence of recent backdoor-based watermarks within neural networks in the scenario of fine-tuning, we propose/develop a novel data-driven idea to restore watermark after fine-tuning without exposing the trigger set. Our empirical results show that by solely introducing training data after fine-tuning, the watermark can be restored if model parameters do not shift dramatically during fine-tuning. Depending on the types of trigger samples used, trigger accuracy can be reinstated to up to 100%. Our study further explores how the restoration process works using loss landscape visualization, as well as the idea of introducing training data in fine-tuning stage to alleviate watermark vanishing.



## **8. Efficient Full-Stack Private Federated Deep Learning with Post-Quantum Security**

cs.CR

Accepted to IEEE TDSC

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2505.05751v1) [paper-pdf](http://arxiv.org/pdf/2505.05751v1)

**Authors**: Yiwei Zhang, Rouzbeh Behnia, Attila A. Yavuz, Reza Ebrahimi, Elisa Bertino

**Abstract**: Federated learning (FL) enables collaborative model training while preserving user data privacy by keeping data local. Despite these advantages, FL remains vulnerable to privacy attacks on user updates and model parameters during training and deployment. Secure aggregation protocols have been proposed to protect user updates by encrypting them, but these methods often incur high computational costs and are not resistant to quantum computers. Additionally, differential privacy (DP) has been used to mitigate privacy leakages, but existing methods focus on secure aggregation or DP, neglecting their potential synergies. To address these gaps, we introduce Beskar, a novel framework that provides post-quantum secure aggregation, optimizes computational overhead for FL settings, and defines a comprehensive threat model that accounts for a wide spectrum of adversaries. We also integrate DP into different stages of FL training to enhance privacy protection in diverse scenarios. Our framework provides a detailed analysis of the trade-offs between security, performance, and model accuracy, representing the first thorough examination of secure aggregation protocols combined with various DP approaches for post-quantum secure FL. Beskar aims to address the pressing privacy and security issues FL while ensuring quantum-safety and robust performance.



## **9. BM-PAW: A Profitable Mining Attack in the PoW-based Blockchain System**

cs.CR

22 pages, 6 figures

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2411.06187v2) [paper-pdf](http://arxiv.org/pdf/2411.06187v2)

**Authors**: Junjie Hu, Na Ruan

**Abstract**: Mining attacks enable an adversary to procure a disproportionately large portion of mining rewards by deviating from honest mining practices within the PoW-based blockchain system. In this paper, we demonstrate that the security vulnerabilities of PoW-based blockchain extend beyond what these mining attacks initially reveal. We introduce a novel mining strategy, named BM-PAW, which yields superior rewards for both the attacker and the targeted pool compared to the state-of-the-art mining attack, PAW. BM-PAW attackers are incentivized to offer appropriate bribe money to other targets, as they comply with the attacker's directives upon receiving payment. We further find the BM-PAW attacker can circumvent the miner's dilemma through equilibrium analysis in a two-pool BM-PAW game scenario, wherein the outcome is determined by the attacker's mining power. We finally propose practical countermeasures to mitigate these novel pool attacks.



## **10. Mitigating Evasion Attacks in Federated Learning-Based Signal Classifiers**

eess.SP

Accepted for publication in IEEE Transactions on Network Science and  Engineering. arXiv admin note: substantial text overlap with arXiv:2301.08866

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2306.04872v3) [paper-pdf](http://arxiv.org/pdf/2306.04872v3)

**Authors**: Su Wang, Rajeev Sahay, Adam Piaseczny, Christopher G. Brinton

**Abstract**: Recent interest in leveraging federated learning (FL) for radio signal classification (SC) tasks has shown promise but FL-based SC remains susceptible to model poisoning adversarial attacks. These adversarial attacks mislead the ML model training process, damaging ML models across the network and leading to lower SC performance. In this work, we seek to mitigate model poisoning adversarial attacks on FL-based SC by proposing the Underlying Server Defense of Federated Learning (USD-FL). Unlike existing server-driven defenses, USD-FL does not rely on perfect network information, i.e., knowing the quantity of adversaries, the adversarial attack architecture, or the start time of the adversarial attacks. Our proposed USD-FL methodology consists of deriving logits for devices' ML models on a reserve dataset, comparing pair-wise logits via 1-Wasserstein distance and then determining a time-varying threshold for adversarial detection. As a result, USD-FL effectively mitigates model poisoning attacks introduced in the FL network. Specifically, when baseline server-driven defenses do have perfect network information, USD-FL outperforms them by (i) improving final ML classification accuracies by at least 6%, (ii) reducing false positive adversary detection rates by at least 10%, and (iii) decreasing the total number of misclassified signals by over 8%. Moreover, when baseline defenses do not have perfect network information, we show that USD-FL achieves accuracies of approximately 74.1% and 62.5% in i.i.d. and non-i.i.d. settings, outperforming existing server-driven baselines, which achieve 52.1% and 39.2% in i.i.d. and non-i.i.d. settings, respectively.



## **11. LiteLMGuard: Seamless and Lightweight On-Device Prompt Filtering for Safeguarding Small Language Models against Quantization-induced Risks and Vulnerabilities**

cs.CR

14 pages, 18 figures, and 4 tables

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05619v1) [paper-pdf](http://arxiv.org/pdf/2505.05619v1)

**Authors**: Kalyan Nakka, Jimmy Dani, Ausmit Mondal, Nitesh Saxena

**Abstract**: The growing adoption of Large Language Models (LLMs) has influenced the development of their lighter counterparts-Small Language Models (SLMs)-to enable on-device deployment across smartphones and edge devices. These SLMs offer enhanced privacy, reduced latency, server-free functionality, and improved user experience. However, due to resource constraints of on-device environment, SLMs undergo size optimization through compression techniques like quantization, which can inadvertently introduce fairness, ethical and privacy risks. Critically, quantized SLMs may respond to harmful queries directly, without requiring adversarial manipulation, raising significant safety and trust concerns.   To address this, we propose LiteLMGuard (LLMG), an on-device prompt guard that provides real-time, prompt-level defense for quantized SLMs. Additionally, our prompt guard is designed to be model-agnostic such that it can be seamlessly integrated with any SLM, operating independently of underlying architectures. Our LLMG formalizes prompt filtering as a deep learning (DL)-based prompt answerability classification task, leveraging semantic understanding to determine whether a query should be answered by any SLM. Using our curated dataset, Answerable-or-Not, we trained and fine-tuned several DL models and selected ELECTRA as the candidate, with 97.75% answerability classification accuracy.   Our safety effectiveness evaluations demonstrate that LLMG defends against over 87% of harmful prompts, including both direct instruction and jailbreak attack strategies. We further showcase its ability to mitigate the Open Knowledge Attacks, where compromised SLMs provide unsafe responses without adversarial prompting. In terms of prompt filtering effectiveness, LLMG achieves near state-of-the-art filtering accuracy of 94%, with an average latency of 135 ms, incurring negligible overhead for users.



## **12. Towards the Worst-case Robustness of Large Language Models**

cs.LG

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2501.19040v2) [paper-pdf](http://arxiv.org/pdf/2501.19040v2)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of large language models to adversarial attacks, where adversaries craft specific input sequences to induce harmful, violent, private, or incorrect outputs. In this work, we study their worst-case robustness, i.e., whether an adversarial example exists that leads to such undesirable outputs. We upper bound the worst-case robustness using stronger white-box attacks, indicating that most current deterministic defenses achieve nearly 0\% worst-case robustness. We propose a general tight lower bound for randomized smoothing using fractional knapsack solvers or 0-1 knapsack solvers, and using them to bound the worst-case robustness of all stochastic defenses. Based on these solvers, we provide theoretical lower bounds for several previous empirical defenses. For example, we certify the robustness of a specific case, smoothing using a uniform kernel, against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.



## **13. Jailbreaking and Mitigation of Vulnerabilities in Large Language Models**

cs.CR

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2410.15236v2) [paper-pdf](http://arxiv.org/pdf/2410.15236v2)

**Authors**: Benji Peng, Keyu Chen, Qian Niu, Ziqian Bi, Ming Liu, Pohsun Feng, Tianyang Wang, Lawrence K. Q. Yan, Yizhu Wen, Yichao Zhang, Caitlyn Heqi Yin

**Abstract**: Large Language Models (LLMs) have transformed artificial intelligence by advancing natural language understanding and generation, enabling applications across fields beyond healthcare, software engineering, and conversational systems. Despite these advancements in the past few years, LLMs have shown considerable vulnerabilities, particularly to prompt injection and jailbreaking attacks. This review analyzes the state of research on these vulnerabilities and presents available defense strategies. We roughly categorize attack approaches into prompt-based, model-based, multimodal, and multilingual, covering techniques such as adversarial prompting, backdoor injections, and cross-modality exploits. We also review various defense mechanisms, including prompt filtering, transformation, alignment techniques, multi-agent defenses, and self-regulation, evaluating their strengths and shortcomings. We also discuss key metrics and benchmarks used to assess LLM safety and robustness, noting challenges like the quantification of attack success in interactive contexts and biases in existing datasets. Identifying current research gaps, we suggest future directions for resilient alignment strategies, advanced defenses against evolving attacks, automation of jailbreak detection, and consideration of ethical and societal impacts. This review emphasizes the need for continued research and cooperation within the AI community to enhance LLM security and ensure their safe deployment.



## **14. PointBA: Towards Backdoor Attacks in 3D Point Cloud**

cs.LG

Accepted by ICCV 2021

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2103.16074v4) [paper-pdf](http://arxiv.org/pdf/2103.16074v4)

**Authors**: Xinke Li, Zhirui Chen, Yue Zhao, Zekun Tong, Yabang Zhao, Andrew Lim, Joey Tianyi Zhou

**Abstract**: 3D deep learning has been increasingly more popular for a variety of tasks including many safety-critical applications. However, recently several works raise the security issues of 3D deep models. Although most of them consider adversarial attacks, we identify that backdoor attack is indeed a more serious threat to 3D deep learning systems but remains unexplored. We present the backdoor attacks in 3D point cloud with a unified framework that exploits the unique properties of 3D data and networks. In particular, we design two attack approaches on point cloud: the poison-label backdoor attack (PointPBA) and the clean-label backdoor attack (PointCBA). The first one is straightforward and effective in practice, while the latter is more sophisticated assuming there are certain data inspections. The attack algorithms are mainly motivated and developed by 1) the recent discovery of 3D adversarial samples suggesting the vulnerability of deep models under spatial transformation; 2) the proposed feature disentanglement technique that manipulates the feature of the data through optimization methods and its potential to embed a new task. Extensive experiments show the efficacy of the PointPBA with over 95% success rate across various 3D datasets and models, and the more stealthy PointCBA with around 50% success rate. Our proposed backdoor attack in 3D point cloud is expected to perform as a baseline for improving the robustness of 3D deep models.



## **15. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP**

cs.CV

ICML 2025

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05528v1) [paper-pdf](http://arxiv.org/pdf/2505.05528v1)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.



## **16. DispBench: Benchmarking Disparity Estimation to Synthetic Corruptions**

cs.CV

Accepted at CVPR 2025 Workshop on Synthetic Data for Computer Vision

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05091v1) [paper-pdf](http://arxiv.org/pdf/2505.05091v1)

**Authors**: Shashank Agnihotri, Amaan Ansari, Annika Dackermann, Fabian Rösch, Margret Keuper

**Abstract**: Deep learning (DL) has surpassed human performance on standard benchmarks, driving its widespread adoption in computer vision tasks. One such task is disparity estimation, estimating the disparity between matching pixels in stereo image pairs, which is crucial for safety-critical applications like medical surgeries and autonomous navigation. However, DL-based disparity estimation methods are highly susceptible to distribution shifts and adversarial attacks, raising concerns about their reliability and generalization. Despite these concerns, a standardized benchmark for evaluating the robustness of disparity estimation methods remains absent, hindering progress in the field.   To address this gap, we introduce DispBench, a comprehensive benchmarking tool for systematically assessing the reliability of disparity estimation methods. DispBench evaluates robustness against synthetic image corruptions such as adversarial attacks and out-of-distribution shifts caused by 2D Common Corruptions across multiple datasets and diverse corruption scenarios. We conduct the most extensive performance and robustness analysis of disparity estimation methods to date, uncovering key correlations between accuracy, reliability, and generalization. Open-source code for DispBench: https://github.com/shashankskagnihotri/benchmarking_robustness/tree/disparity_estimation/final/disparity_estimation



## **17. Integrating Communication, Sensing, and Security: Progress and Prospects of PLS in ISAC Systems**

cs.ET

IEEE COMST

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05090v1) [paper-pdf](http://arxiv.org/pdf/2505.05090v1)

**Authors**: Waqas Aman, El-Mehdi Illi, Marwa Qaraqe, Saif Al-Kuwari

**Abstract**: The sixth generation of wireless networks defined several key performance indicators (KPIs) for assessing its networks, mainly in terms of reliability, coverage, and sensing. In this regard, remarkable attention has been paid recently to the integrated sensing and communication (ISAC) paradigm as an enabler for efficiently and jointly performing communication and sensing using the same spectrum and hardware resources. On the other hand, ensuring communication and data security has been an imperative requirement for wireless networks throughout their evolution. The physical-layer security (PLS) concept paved the way to catering to the security needs in wireless networks in a sustainable way while guaranteeing theoretically secure transmissions, independently of the computational capacity of adversaries. Therefore, it is of paramount importance to consider a balanced trade-off between communication reliability, sensing, and security in future networks, such as the 5G and beyond, and the 6G. In this paper, we provide a comprehensive and system-wise review of designed secure ISAC systems from a PLS point of view. In particular, the impact of various physical-layer techniques, schemes, and wireless technologies to ensure the sensing-security trade-off is studied from the surveyed work. Furthermore, the amalgamation of PLS and ISAC is analyzed in a broader impact by considering attacks targeting data confidentiality, communication covertness, and sensing spoofing. The paper also serves as a tutorial by presenting several theoretical foundations on ISAC and PLS, which represent a practical guide for readers to develop novel secure ISAC network designs.



## **18. Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction**

cs.CL

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05084v1) [paper-pdf](http://arxiv.org/pdf/2505.05084v1)

**Authors**: Xiaowei Zhu, Yubing Ren, Yanan Cao, Xixun Lin, Fang Fang, Yangxi Li

**Abstract**: The rapid advancement of large language models has raised significant concerns regarding their potential misuse by malicious actors. As a result, developing effective detectors to mitigate these risks has become a critical priority. However, most existing detection methods focus excessively on detection accuracy, often neglecting the societal risks posed by high false positive rates (FPRs). This paper addresses this issue by leveraging Conformal Prediction (CP), which effectively constrains the upper bound of FPRs. While directly applying CP constrains FPRs, it also leads to a significant reduction in detection performance. To overcome this trade-off, this paper proposes a Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction (MCP), which both enforces the FPR constraint and improves detection performance. This paper also introduces RealDet, a high-quality dataset that spans a wide range of domains, ensuring realistic calibration and enabling superior detection performance when combined with MCP. Empirical evaluations demonstrate that MCP effectively constrains FPRs, significantly enhances detection performance, and increases robustness against adversarial attacks across multiple detectors and datasets.



## **19. Uncovering the Limitations of Model Inversion Evaluation -- Benchmarks and Connection to Type-I Adversarial Attacks**

cs.LG

Our dataset and code are available in the Supp

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.03519v2) [paper-pdf](http://arxiv.org/pdf/2505.03519v2)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information of private training data by exploiting access to machine learning models. The most common evaluation framework for MI attacks/defenses relies on an evaluation model that has been utilized to assess progress across almost all MI attacks and defenses proposed in recent years. In this paper, for the first time, we present an in-depth study of MI evaluation. Firstly, we construct the first comprehensive human-annotated dataset of MI attack samples, based on 28 setups of different MI attacks, defenses, private and public datasets. Secondly, using our dataset, we examine the accuracy of the MI evaluation framework and reveal that it suffers from a significant number of false positives. These findings raise questions about the previously reported success rates of SOTA MI attacks. Thirdly, we analyze the causes of these false positives, design controlled experiments, and discover the surprising effect of Type I adversarial features on MI evaluation, as well as adversarial transferability, highlighting a relationship between two previously distinct research areas. Our findings suggest that the performance of SOTA MI attacks has been overestimated, with the actual privacy leakage being significantly less than previously reported. In conclusion, we highlight critical limitations in the widely used MI evaluation framework and present our methods to mitigate false positive rates. We remark that prior research has shown that Type I adversarial attacks are very challenging, with no existing solution. Therefore, we urge to consider human evaluation as a primary MI evaluation framework rather than merely a supplement as in previous MI research. We also encourage further work on developing more robust and reliable automatic evaluation frameworks.



## **20. Economic Security of Multiple Shared Security Protocols**

cs.CR

21 pages, 6 figures

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.03843v2) [paper-pdf](http://arxiv.org/pdf/2505.03843v2)

**Authors**: Abhimanyu Nag, Dhruv Bodani, Abhishek Kumar

**Abstract**: As restaking protocols gain adoption across blockchain ecosystems, there is a need for Actively Validated Services (AVSs) to span multiple Shared Security Providers (SSPs). This leads to stake fragmentation which introduces new complications where an adversary may compromise an AVS by targeting its weakest SSP. In this paper, we formalize the Multiple SSP Problem and analyze two architectures : an isolated fragmented model called Model $\mathbb{M}$ and a shared unified model called Model $\mathbb{S}$, through a convex optimization and game-theoretic lens. We derive utility bounds, attack cost conditions, and market equilibrium that describes protocol security for both models. Our results show that while Model $\mathbb{M}$ offers deployment flexibility, it inherits lowest-cost attack vulnerabilities, whereas Model $\mathbb{S}$ achieves tighter security guarantees through single validator sets and aggregated slashing logic. We conclude with future directions of work including an incentive-compatible stake rebalancing allocation in restaking ecosystems.



## **21. Memory Under Siege: A Comprehensive Survey of Side-Channel Attacks on Memory**

cs.CR

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.04896v1) [paper-pdf](http://arxiv.org/pdf/2505.04896v1)

**Authors**: MD Mahady Hassan, Shanto Roy, Reza Rahaeimehr

**Abstract**: Side-channel attacks on memory (SCAM) exploit unintended data leaks from memory subsystems to infer sensitive information, posing significant threats to system security. These attacks exploit vulnerabilities in memory access patterns, cache behaviors, and other microarchitectural features to bypass traditional security measures. The purpose of this research is to examine SCAM, classify various attack techniques, and evaluate existing defense mechanisms. It guides researchers and industry professionals in improving memory security and mitigating emerging threats. We begin by identifying the major vulnerabilities in the memory system that are frequently exploited in SCAM, such as cache timing, speculative execution, \textit{Rowhammer}, and other sophisticated approaches. Next, we outline a comprehensive taxonomy that systematically classifies these attacks based on their types, target systems, attack vectors, and adversarial capabilities required to execute them. In addition, we review the current landscape of mitigation strategies, emphasizing their strengths and limitations. This work aims to provide a comprehensive overview of memory-based side-channel attacks with the goal of providing significant insights for researchers and practitioners to better understand, detect, and mitigate SCAM risks.



## **22. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

cs.LG

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2410.10572v4) [paper-pdf](http://arxiv.org/pdf/2410.10572v4)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.



## **23. Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs**

cs.CR

7 Pages, 6 Figures

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04806v1) [paper-pdf](http://arxiv.org/pdf/2505.04806v1)

**Authors**: Chetan Pathade

**Abstract**: Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security.



## **24. Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization**

cs.LG

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04578v1) [paper-pdf](http://arxiv.org/pdf/2505.04578v1)

**Authors**: Wenjun Cao

**Abstract**: Reinforcement learning (RL) fine-tuning transforms large language models while creating a vulnerability we experimentally verify: Our experiment shows that malicious RL fine-tuning dismantles safety guardrails with remarkable efficiency, requiring only 50 steps and minimal adversarial prompts, with harmful escalating from 0-2 to 7-9. This attack vector particularly threatens open-source models with parameter-level access. Existing defenses targeting supervised fine-tuning prove ineffective against RL's dynamic feedback mechanisms. We introduce Reward Neutralization, the first defense framework specifically designed against RL fine-tuning attacks, establishing concise rejection patterns that render malicious reward signals ineffective. Our approach trains models to produce minimal-information rejections that attackers cannot exploit, systematically neutralizing attempts to optimize toward harmful outputs. Experiments validate that our approach maintains low harmful scores (no greater than 2) after 200 attack steps, while standard models rapidly deteriorate. This work provides the first constructive proof that robust defense against increasingly accessible RL attacks is achievable, addressing a critical security gap for open-weight models.



## **25. REVEAL: Multi-turn Evaluation of Image-Input Harms for Vision LLM**

cs.CL

13 pages (8 main), to be published in IJCAI 2025

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04673v1) [paper-pdf](http://arxiv.org/pdf/2505.04673v1)

**Authors**: Madhur Jindal, Saurabh Deshpande

**Abstract**: Vision Large Language Models (VLLMs) represent a significant advancement in artificial intelligence by integrating image-processing capabilities with textual understanding, thereby enhancing user interactions and expanding application domains. However, their increased complexity introduces novel safety and ethical challenges, particularly in multi-modal and multi-turn conversations. Traditional safety evaluation frameworks, designed for text-based, single-turn interactions, are inadequate for addressing these complexities. To bridge this gap, we introduce the REVEAL (Responsible Evaluation of Vision-Enabled AI LLMs) Framework, a scalable and automated pipeline for evaluating image-input harms in VLLMs. REVEAL includes automated image mining, synthetic adversarial data generation, multi-turn conversational expansion using crescendo attack strategies, and comprehensive harm assessment through evaluators like GPT-4o.   We extensively evaluated five state-of-the-art VLLMs, GPT-4o, Llama-3.2, Qwen2-VL, Phi3.5V, and Pixtral, across three important harm categories: sexual harm, violence, and misinformation. Our findings reveal that multi-turn interactions result in significantly higher defect rates compared to single-turn evaluations, highlighting deeper vulnerabilities in VLLMs. Notably, GPT-4o demonstrated the most balanced performance as measured by our Safety-Usability Index (SUI) followed closely by Pixtral. Additionally, misinformation emerged as a critical area requiring enhanced contextual defenses. Llama-3.2 exhibited the highest MT defect rate ($16.55 \%$) while Qwen2-VL showed the highest MT refusal rate ($19.1 \%$).



## **26. Mitigating Many-Shot Jailbreaking**

cs.LG

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2504.09604v2) [paper-pdf](http://arxiv.org/pdf/2504.09604v2)

**Authors**: Christopher M. Ackerman, Nina Panickssery

**Abstract**: Many-shot jailbreaking (MSJ) is an adversarial technique that exploits the long context windows of modern LLMs to circumvent model safety training by including in the prompt many examples of a "fake" assistant responding inappropriately before the final request. With enough examples, the model's in-context learning abilities override its safety training, and it responds as if it were the "fake" assistant. In this work, we probe the effectiveness of different fine-tuning and input sanitization approaches on mitigating MSJ attacks, alone and in combination. We find incremental mitigation effectiveness for each, and show that the combined techniques significantly reduce the effectiveness of MSJ attacks, while retaining model performance in benign in-context learning and conversational tasks. We suggest that our approach could meaningfully ameliorate this vulnerability if incorporated into model safety post-training.



## **27. Machine Learning Cryptanalysis of a Quantum Random Number Generator**

cs.LG

Published article is at https://ieeexplore.ieee.org/document/8396276.  Related code is at  https://github.com/Nano-Neuro-Research-Lab/Machine-Learning-Cryptanalysis-of-a-Quantum-Random-Number-Generator

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/1905.02342v3) [paper-pdf](http://arxiv.org/pdf/1905.02342v3)

**Authors**: Nhan Duy Truong, Jing Yan Haw, Syed Muhamad Assad, Ping Koy Lam, Omid Kavehei

**Abstract**: Random number generators (RNGs) that are crucial for cryptographic applications have been the subject of adversarial attacks. These attacks exploit environmental information to predict generated random numbers that are supposed to be truly random and unpredictable. Though quantum random number generators (QRNGs) are based on the intrinsic indeterministic nature of quantum properties, the presence of classical noise in the measurement process compromises the integrity of a QRNG. In this paper, we develop a predictive machine learning (ML) analysis to investigate the impact of deterministic classical noise in different stages of an optical continuous variable QRNG. Our ML model successfully detects inherent correlations when the deterministic noise sources are prominent. After appropriate filtering and randomness extraction processes are introduced, our QRNG system, in turn, demonstrates its robustness against ML. We further demonstrate the robustness of our ML approach by applying it to uniformly distributed random numbers from the QRNG and a congruential RNG. Hence, our result shows that ML has potentials in benchmarking the quality of RNG devices.



## **28. Reliable Disentanglement Multi-view Learning Against View Adversarial Attacks**

cs.LG

11 pages, 11 figures, accepted by International Joint Conference on  Artificial Intelligence (IJCAI 2025)

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04046v1) [paper-pdf](http://arxiv.org/pdf/2505.04046v1)

**Authors**: Xuyang Wang, Siyuan Duan, Qizhi Li, Guiduo Duan, Yuan Sun, Dezhong Peng

**Abstract**: Recently, trustworthy multi-view learning has attracted extensive attention because evidence learning can provide reliable uncertainty estimation to enhance the credibility of multi-view predictions. Existing trusted multi-view learning methods implicitly assume that multi-view data is secure. In practice, however, in safety-sensitive applications such as autonomous driving and security monitoring, multi-view data often faces threats from adversarial perturbations, thereby deceiving or disrupting multi-view learning models. This inevitably leads to the adversarial unreliability problem (AUP) in trusted multi-view learning. To overcome this tricky problem, we propose a novel multi-view learning framework, namely Reliable Disentanglement Multi-view Learning (RDML). Specifically, we first propose evidential disentanglement learning to decompose each view into clean and adversarial parts under the guidance of corresponding evidences, which is extracted by a pretrained evidence extractor. Then, we employ the feature recalibration module to mitigate the negative impact of adversarial perturbations and extract potential informative features from them. Finally, to further ignore the irreparable adversarial interferences, a view-level evidential attention mechanism is designed. Extensive experiments on multi-view classification tasks with adversarial attacks show that our RDML outperforms the state-of-the-art multi-view learning methods by a relatively large margin.



## **29. MergeGuard: Efficient Thwarting of Trojan Attacks in Machine Learning Models**

cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.04015v1) [paper-pdf](http://arxiv.org/pdf/2505.04015v1)

**Authors**: Soheil Zibakhsh Shabgahi, Yaman Jandali, Farinaz Koushanfar

**Abstract**: This paper proposes MergeGuard, a novel methodology for mitigation of AI Trojan attacks. Trojan attacks on AI models cause inputs embedded with triggers to be misclassified to an adversary's target class, posing a significant threat to model usability trained by an untrusted third party. The core of MergeGuard is a new post-training methodology for linearizing and merging fully connected layers which we show simultaneously improves model generalizability and performance. Our Proof of Concept evaluation on Transformer models demonstrates that MergeGuard maintains model accuracy while decreasing trojan attack success rate, outperforming commonly used (post-training) Trojan mitigation by fine-tuning methodologies.



## **30. Towards Universal and Black-Box Query-Response Only Attack on LLMs with QROA**

cs.CL

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2406.02044v3) [paper-pdf](http://arxiv.org/pdf/2406.02044v3)

**Authors**: Hussein Jawad, Yassine Chenik, Nicolas J. -B. Brunel

**Abstract**: The rapid adoption of Large Language Models (LLMs) has exposed critical security and ethical vulnerabilities, particularly their susceptibility to adversarial manipulations. This paper introduces QROA, a novel black-box jailbreak method designed to identify adversarial suffixes that can bypass LLM alignment safeguards when appended to a malicious instruction. Unlike existing suffix-based jailbreak approaches, QROA does not require access to the model's logit or any other internal information. It also eliminates reliance on human-crafted templates, operating solely through the standard query-response interface of LLMs. By framing the attack as an optimization bandit problem, QROA employs a surrogate model and token level optimization to efficiently explore suffix variations. Furthermore, we propose QROA-UNV, an extension that identifies universal adversarial suffixes for individual models, enabling one-query jailbreaks across a wide range of instructions. Testing on multiple models demonstrates Attack Success Rate (ASR) greater than 80\%. These findings highlight critical vulnerabilities, emphasize the need for advanced defenses, and contribute to the development of more robust safety evaluations for secure AI deployment. The code is made public on the following link: https://github.com/qroa/QROA



## **31. Model-Targeted Data Poisoning Attacks against ITS Applications with Provable Convergence**

math.OC

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03966v1) [paper-pdf](http://arxiv.org/pdf/2505.03966v1)

**Authors**: Xin Wanga, Feilong Wang, Yuan Hong, R. Tyrrell Rockafellar, Xuegang, Ban

**Abstract**: The growing reliance of intelligent systems on data makes the systems vulnerable to data poisoning attacks. Such attacks could compromise machine learning or deep learning models by disrupting the input data. Previous studies on data poisoning attacks are subject to specific assumptions, and limited attention is given to learning models with general (equality and inequality) constraints or lacking differentiability. Such learning models are common in practice, especially in Intelligent Transportation Systems (ITS) that involve physical or domain knowledge as specific model constraints. Motivated by ITS applications, this paper formulates a model-target data poisoning attack as a bi-level optimization problem with a constrained lower-level problem, aiming to induce the model solution toward a target solution specified by the adversary by modifying the training data incrementally. As the gradient-based methods fail to solve this optimization problem, we propose to study the Lipschitz continuity property of the model solution, enabling us to calculate the semi-derivative, a one-sided directional derivative, of the solution over data. We leverage semi-derivative descent to solve the bi-level optimization problem, and establish the convergence conditions of the method to any attainable target model. The model and solution method are illustrated with a simulation of a poisoning attack on the lane change detection using SVM.



## **32. Sustainable Smart Farm Networks: Enhancing Resilience and Efficiency with Decision Theory-Guided Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03721v1) [paper-pdf](http://arxiv.org/pdf/2505.03721v1)

**Authors**: Dian Chen, Zelin Wan, Dong Sam Ha, Jin-Hee Cho

**Abstract**: Solar sensor-based monitoring systems have become a crucial agricultural innovation, advancing farm management and animal welfare through integrating sensor technology, Internet-of-Things, and edge and cloud computing. However, the resilience of these systems to cyber-attacks and their adaptability to dynamic and constrained energy supplies remain largely unexplored. To address these challenges, we propose a sustainable smart farm network designed to maintain high-quality animal monitoring under various cyber and adversarial threats, as well as fluctuating energy conditions. Our approach utilizes deep reinforcement learning (DRL) to devise optimal policies that maximize both monitoring effectiveness and energy efficiency. To overcome DRL's inherent challenge of slow convergence, we integrate transfer learning (TL) and decision theory (DT) to accelerate the learning process. By incorporating DT-guided strategies, we optimize monitoring quality and energy sustainability, significantly reducing training time while achieving comparable performance rewards. Our experimental results prove that DT-guided DRL outperforms TL-enhanced DRL models, improving system performance and reducing training runtime by 47.5%.



## **33. Adversarial Robustness of Deep Learning Models for Inland Water Body Segmentation from SAR Images**

eess.IV

21 pages, 15 figures, 2 tables

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.01884v2) [paper-pdf](http://arxiv.org/pdf/2505.01884v2)

**Authors**: Siddharth Kothari, Srinivasan Murali, Sankalp Kothari, Ujjwal Verma, Jaya Sreevalsan-Nair

**Abstract**: Inland water body segmentation from Synthetic Aperture Radar (SAR) images is an important task needed for several applications, such as flood mapping. While SAR sensors capture data in all-weather conditions as high-resolution images, differentiating water and water-like surfaces from SAR images is not straightforward. Inland water bodies, such as large river basins, have complex geometry, which adds to the challenge of segmentation. U-Net is a widely used deep learning model for land-water segmentation of SAR images. In practice, manual annotation is often used to generate the corresponding water masks as ground truth. Manual annotation of the images is prone to label noise owing to data poisoning attacks, especially due to complex geometry. In this work, we simulate manual errors in the form of adversarial attacks on the U-Net model and study the robustness of the model to human errors in annotation. Our results indicate that U-Net can tolerate a certain level of corruption before its performance drops significantly. This finding highlights the crucial role that the quality of manual annotations plays in determining the effectiveness of the segmentation model. The code and the new dataset, along with adversarial examples for robust training, are publicly available. (GitHub link - https://github.com/GVCL/IWSeg-SAR-Poison.git)



## **34. Data-Driven Falsification of Cyber-Physical Systems**

cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03863v1) [paper-pdf](http://arxiv.org/pdf/2505.03863v1)

**Authors**: Atanu Kundu, Sauvik Gon, Rajarshi Ray

**Abstract**: Cyber-Physical Systems (CPS) are abundant in safety-critical domains such as healthcare, avionics, and autonomous vehicles. Formal verification of their operational safety is, therefore, of utmost importance. In this paper, we address the falsification problem, where the focus is on searching for an unsafe execution in the system instead of proving their absence. The contribution of this paper is a framework that (a) connects the falsification of CPS with the falsification of deep neural networks (DNNs) and (b) leverages the inherent interpretability of Decision Trees for faster falsification of CPS. This is achieved by: (1) building a surrogate model of the CPS under test, either as a DNN model or a Decision Tree, (2) application of various DNN falsification tools to falsify CPS, and (3) a novel falsification algorithm guided by the explanations of safety violations of the CPS model extracted from its Decision Tree surrogate. The proposed framework has the potential to exploit a repertoire of \emph{adversarial attack} algorithms designed to falsify robustness properties of DNNs, as well as state-of-the-art falsification algorithms for DNNs. Although the presented methodology is applicable to systems that can be executed/simulated in general, we demonstrate its effectiveness, particularly in CPS. We show that our framework, implemented as a tool \textsc{FlexiFal}, can detect hard-to-find counterexamples in CPS that have linear and non-linear dynamics. Decision tree-guided falsification shows promising results in efficiently finding multiple counterexamples in the ARCH-COMP 2024 falsification benchmarks~\cite{khandait2024arch}.



## **35. ALMA: Aggregated Lipschitz Maximization Attack on Auto-encoders**

cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03646v1) [paper-pdf](http://arxiv.org/pdf/2505.03646v1)

**Authors**: Chethan Krishnamurthy Ramanaik, Arjun Roy, Eirini Ntoutsi

**Abstract**: Despite the extensive use of deep autoencoders (AEs) in critical applications, their adversarial robustness remains relatively underexplored compared to classification models. AE robustness is characterized by the Lipschitz bounds of its components. Existing robustness evaluation frameworks based on white-box attacks do not fully exploit the vulnerabilities of intermediate ill-conditioned layers in AEs. In the context of optimizing imperceptible norm-bounded additive perturbations to maximize output damage, existing methods struggle to effectively propagate adversarial loss gradients throughout the network, often converging to less effective perturbations. To address this, we propose a novel layer-conditioning-based adversarial optimization objective that effectively guides the adversarial map toward regions of local Lipschitz bounds by enhancing loss gradient information propagation during attack optimization. We demonstrate through extensive experiments on state-of-the-art AEs that our adversarial objective results in stronger attacks, outperforming existing methods in both universal and sample-specific scenarios. As a defense method against this attack, we introduce an inference-time adversarially trained defense plugin that mitigates the effects of adversarial examples.



## **36. The Adaptive Arms Race: Redefining Robustness in AI Security**

cs.AI

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2312.13435v3) [paper-pdf](http://arxiv.org/pdf/2312.13435v3)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world AI-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. Canonical robustness evaluation relies on adaptive attacks, which leverage complete knowledge of the defense and are tailored to bypass it. This work broadens the notion of adaptivity, which we employ to enhance both attacks and defenses, showing how they can benefit from mutual learning through interaction. We introduce a framework for adaptively optimizing black-box attacks and defenses under the competitive game they form. To assess robustness reliably, it is essential to evaluate against realistic and worst-case attacks. We thus enhance attacks and their evasive arsenal together using RL, apply the same principle to defenses, and evaluate them first independently and then jointly under a multi-agent perspective. We find that active defenses, those that dynamically control system responses, are an essential complement to model hardening against decision-based attacks; that these defenses can be circumvented by adaptive attacks, something that elicits defenses being adaptive too. Our findings, supported by an extensive theoretical and empirical investigation, confirm that adaptive adversaries pose a serious threat to black-box AI-based systems, rekindling the proverbial arms race. Notably, our approach outperforms the state-of-the-art black-box attacks and defenses, while bringing them together to render effective insights into the robustness of real-world deployed ML-based systems.



## **37. BadLingual: A Novel Lingual-Backdoor Attack against Large Language Models**

cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03501v1) [paper-pdf](http://arxiv.org/pdf/2505.03501v1)

**Authors**: Zihan Wang, Hongwei Li, Rui Zhang, Wenbo Jiang, Kangjie Chen, Tianwei Zhang, Qingchuan Zhao, Guowen Xu

**Abstract**: In this paper, we present a new form of backdoor attack against Large Language Models (LLMs): lingual-backdoor attacks. The key novelty of lingual-backdoor attacks is that the language itself serves as the trigger to hijack the infected LLMs to generate inflammatory speech. They enable the precise targeting of a specific language-speaking group, exacerbating racial discrimination by malicious entities. We first implement a baseline lingual-backdoor attack, which is carried out by poisoning a set of training data for specific downstream tasks through translation into the trigger language. However, this baseline attack suffers from poor task generalization and is impractical in real-world settings. To address this challenge, we design BadLingual, a novel task-agnostic lingual-backdoor, capable of triggering any downstream tasks within the chat LLMs, regardless of the specific questions of these tasks. We design a new approach using PPL-constrained Greedy Coordinate Gradient-based Search (PGCG) based adversarial training to expand the decision boundary of lingual-backdoor, thereby enhancing the generalization ability of lingual-backdoor across various tasks. We perform extensive experiments to validate the effectiveness of our proposed attacks. Specifically, the baseline attack achieves an ASR of over 90% on the specified tasks. However, its ASR reaches only 37.61% across six tasks in the task-agnostic scenario. In contrast, BadLingual brings up to 37.35% improvement over the baseline. Our study sheds light on a new perspective of vulnerabilities in LLMs with multilingual capabilities and is expected to promote future research on the potential defenses to enhance the LLMs' robustness



## **38. Mitigating Backdoor Triggered and Targeted Data Poisoning Attacks in Voice Authentication Systems**

cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03455v1) [paper-pdf](http://arxiv.org/pdf/2505.03455v1)

**Authors**: Alireza Mohammadi, Keshav Sood, Dhananjay Thiruvady, Asef Nazari

**Abstract**: Voice authentication systems remain susceptible to two major threats: backdoor triggered attacks and targeted data poisoning attacks. This dual vulnerability is critical because conventional solutions typically address each threat type separately, leaving systems exposed to adversaries who can exploit both attacks simultaneously. We propose a unified defense framework that effectively addresses both BTA and TDPA. Our framework integrates a frequency focused detection mechanism that flags covert pitch boosting and sound masking backdoor attacks in near real time, followed by a convolutional neural network that addresses TDPA. This dual layered defense approach utilizes multidimensional acoustic features to isolate anomalous signals without requiring costly model retraining. In particular, our PBSM detection mechanism can seamlessly integrate into existing voice authentication pipelines and scale effectively for large scale deployments. Experimental results on benchmark datasets and their compression with the state of the art algorithm demonstrate that our PBSM detection mechanism outperforms the state of the art. Our framework reduces attack success rates to as low as five to fifteen percent while maintaining a recall rate of up to ninety five percent in recognizing TDPA.



## **39. Robustness in AI-Generated Detection: Enhancing Resistance to Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03435v1) [paper-pdf](http://arxiv.org/pdf/2505.03435v1)

**Authors**: Sun Haoxuan, Hong Yan, Zhan Jiahui, Chen Haoxing, Lan Jun, Zhu Huijia, Wang Weiqiang, Zhang Liqing, Zhang Jianfu

**Abstract**: The rapid advancement of generative image technology has introduced significant security concerns, particularly in the domain of face generation detection. This paper investigates the vulnerabilities of current AI-generated face detection systems. Our study reveals that while existing detection methods often achieve high accuracy under standard conditions, they exhibit limited robustness against adversarial attacks. To address these challenges, we propose an approach that integrates adversarial training to mitigate the impact of adversarial examples. Furthermore, we utilize diffusion inversion and reconstruction to further enhance detection robustness. Experimental results demonstrate that minor adversarial perturbations can easily bypass existing detection systems, but our method significantly improves the robustness of these systems. Additionally, we provide an in-depth analysis of adversarial and benign examples, offering insights into the intrinsic characteristics of AI-generated content. All associated code will be made publicly available in a dedicated repository to facilitate further research and verification.



## **40. Attention-aggregated Attack for Boosting the Transferability of Facial Adversarial Examples**

cs.CV

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03383v1) [paper-pdf](http://arxiv.org/pdf/2505.03383v1)

**Authors**: Jian-Wei Li, Wen-Ze Shao

**Abstract**: Adversarial examples have revealed the vulnerability of deep learning models and raised serious concerns about information security. The transfer-based attack is a hot topic in black-box attacks that are practical to real-world scenarios where the training datasets, parameters, and structure of the target model are unknown to the attacker. However, few methods consider the particularity of class-specific deep models for fine-grained vision tasks, such as face recognition (FR), giving rise to unsatisfactory attacking performance. In this work, we first investigate what in a face exactly contributes to the embedding learning of FR models and find that both decisive and auxiliary facial features are specific to each FR model, which is quite different from the biological mechanism of human visual system. Accordingly we then propose a novel attack method named Attention-aggregated Attack (AAA) to enhance the transferability of adversarial examples against FR, which is inspired by the attention divergence and aims to destroy the facial features that are critical for the decision-making of other FR models by imitating their attentions on the clean face images. Extensive experiments conducted on various FR models validate the superiority and robust effectiveness of the proposed method over existing methods.



## **41. A Chaos Driven Metric for Backdoor Attack Detection**

cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03208v1) [paper-pdf](http://arxiv.org/pdf/2505.03208v1)

**Authors**: Hema Karnam Surendrababu, Nithin Nagaraj

**Abstract**: The advancement and adoption of Artificial Intelligence (AI) models across diverse domains have transformed the way we interact with technology. However, it is essential to recognize that while AI models have introduced remarkable advancements, they also present inherent challenges such as their vulnerability to adversarial attacks. The current work proposes a novel defense mechanism against one of the most significant attack vectors of AI models - the backdoor attack via data poisoning of training datasets. In this defense technique, an integrated approach that combines chaos theory with manifold learning is proposed. A novel metric - Precision Matrix Dependency Score (PDS) that is based on the conditional variance of Neurochaos features is formulated. The PDS metric has been successfully evaluated to distinguish poisoned samples from non-poisoned samples across diverse datasets.



## **42. Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models**

cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2503.06269v2) [paper-pdf](http://arxiv.org/pdf/2503.06269v2)

**Authors**: Thomas Winninger, Boussad Addad, Katarzyna Kapusta

**Abstract**: Traditional white-box methods for creating adversarial perturbations against LLMs typically rely only on gradient computation from the targeted model, ignoring the internal mechanisms responsible for attack success or failure. Conversely, interpretability studies that analyze these internal mechanisms lack practical applications beyond runtime interventions. We bridge this gap by introducing a novel white-box approach that leverages mechanistic interpretability techniques to craft practical adversarial inputs. Specifically, we first identify acceptance subspaces - sets of feature vectors that do not trigger the model's refusal mechanisms - then use gradient-based optimization to reroute embeddings from refusal subspaces to acceptance subspaces, effectively achieving jailbreaks. This targeted approach significantly reduces computation cost, achieving attack success rates of 80-95\% on state-of-the-art models including Gemma2, Llama3.2, and Qwen2.5 within minutes or even seconds, compared to existing techniques that often fail or require hours of computation. We believe this approach opens a new direction for both attack research and defense development. Furthermore, it showcases a practical application of mechanistic interpretability where other methods are less efficient, which highlights its utility. The code and generated datasets are available at https://github.com/Sckathach/subspace-rerouting.



## **43. PEEK: Phishing Evolution Framework for Phishing Generation and Evolving Pattern Analysis using Large Language Models**

cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2411.11389v2) [paper-pdf](http://arxiv.org/pdf/2411.11389v2)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Shuo Wang, Alsharif Abuadbba, Carsten Rudolph

**Abstract**: Phishing remains a pervasive cyber threat, as attackers craft deceptive emails to lure victims into revealing sensitive information. While Artificial Intelligence (AI), in particular, deep learning, has become a key component in defending against phishing attacks, these approaches face critical limitations. The scarcity of publicly available, diverse, and updated data, largely due to privacy concerns, constrains detection effectiveness. As phishing tactics evolve rapidly, models trained on limited, outdated data struggle to detect new, sophisticated deception strategies, leaving systems and people vulnerable to an ever-growing array of attacks. We propose the first Phishing Evolution FramEworK (PEEK) for augmenting phishing email datasets with respect to quality and diversity, and analyzing changing phishing patterns for detection to adapt to updated phishing attacks. Specifically, we integrate large language models (LLMs) into the process of adversarial training to enhance the performance of the generated dataset and leverage persuasion principles in a recurrent framework to facilitate the understanding of changing phishing strategies. PEEK raises the proportion of usable phishing samples from 21.4% to 84.8%, surpassing existing works that rely on prompting and fine-tuning LLMs. The phishing datasets provided by PEEK, with evolving phishing patterns, outperform the other two available LLM-generated phishing email datasets in improving detection robustness. PEEK phishing boosts detectors' accuracy to over 88% and reduces adversarial sensitivity by up to 70%, still maintaining 70% detection accuracy against adversarial attacks.



## **44. Adversarial Sample Generation for Anomaly Detection in Industrial Control Systems**

cs.CR

Accepted in the 1st Workshop on Modeling and Verification for Secure  and Performant Cyber-Physical Systems in conjunction with Cyber-Physical  Systems and Internet-of-Things Week, Irvine, USA, May 6-9, 2025

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03120v1) [paper-pdf](http://arxiv.org/pdf/2505.03120v1)

**Authors**: Abdul Mustafa, Muhammad Talha Khan, Muhammad Azmi Umer, Zaki Masood, Chuadhry Mujeeb Ahmed

**Abstract**: Machine learning (ML)-based intrusion detection systems (IDS) are vulnerable to adversarial attacks. It is crucial for an IDS to learn to recognize adversarial examples before malicious entities exploit them. In this paper, we generated adversarial samples using the Jacobian Saliency Map Attack (JSMA). We validate the generalization and scalability of the adversarial samples to tackle a broad range of real attacks on Industrial Control Systems (ICS). We evaluated the impact by assessing multiple attacks generated using the proposed method. The model trained with adversarial samples detected attacks with 95% accuracy on real-world attack data not used during training. The study was conducted using an operational secure water treatment (SWaT) testbed.



## **45. Adversarial Attacks in Multimodal Systems: A Practitioner's Survey**

cs.LG

Accepted in IEEE COMPSAC 2025

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03084v1) [paper-pdf](http://arxiv.org/pdf/2505.03084v1)

**Authors**: Shashank Kapoor, Sanjay Surendranath Girija, Lakshit Arora, Dipen Pradhan, Ankit Shetgaonkar, Aman Raj

**Abstract**: The introduction of multimodal models is a huge step forward in Artificial Intelligence. A single model is trained to understand multiple modalities: text, image, video, and audio. Open-source multimodal models have made these breakthroughs more accessible. However, considering the vast landscape of adversarial attacks across these modalities, these models also inherit vulnerabilities of all the modalities, and ultimately, the adversarial threat amplifies. While broad research is available on possible attacks within or across these modalities, a practitioner-focused view that outlines attack types remains absent in the multimodal world. As more Machine Learning Practitioners adopt, fine-tune, and deploy open-source models in real-world applications, it's crucial that they can view the threat landscape and take the preventive actions necessary. This paper addresses the gap by surveying adversarial attacks targeting all four modalities: text, image, video, and audio. This survey provides a view of the adversarial attack landscape and presents how multimodal adversarial threats have evolved. To the best of our knowledge, this survey is the first comprehensive summarization of the threat landscape in the multimodal world.



## **46. Large Language Models as Robust Data Generators in Software Analytics: Are We There Yet?**

cs.SE

Accepted to the AI Model/Data Track of the Evaluation and Assessment  in Software Engineering (EASE) 2025 Conference

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2411.10565v3) [paper-pdf](http://arxiv.org/pdf/2411.10565v3)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Large Language Model (LLM)-generated data is increasingly used in software analytics, but it is unclear how this data compares to human-written data, particularly when models are exposed to adversarial scenarios. Adversarial attacks can compromise the reliability and security of software systems, so understanding how LLM-generated data performs under these conditions, compared to human-written data, which serves as the benchmark for model performance, can provide valuable insights into whether LLM-generated data offers similar robustness and effectiveness. To address this gap, we systematically evaluate and compare the quality of human-written and LLM-generated data for fine-tuning robust pre-trained models (PTMs) in the context of adversarial attacks. We evaluate the robustness of six widely used PTMs, fine-tuned on human-written and LLM-generated data, before and after adversarial attacks. This evaluation employs nine state-of-the-art (SOTA) adversarial attack techniques across three popular software analytics tasks: clone detection, code summarization, and sentiment analysis in code review discussions. Additionally, we analyze the quality of the generated adversarial examples using eleven similarity metrics. Our findings reveal that while PTMs fine-tuned on LLM-generated data perform competitively with those fine-tuned on human-written data, they exhibit less robustness against adversarial attacks in software analytics tasks. Our study underscores the need for further exploration into enhancing the quality of LLM-generated training data to develop models that are both high-performing and capable of withstanding adversarial attacks in software analytics.



## **47. Adversarial Robustness Analysis of Vision-Language Models in Medical Image Segmentation**

cs.CV

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02971v1) [paper-pdf](http://arxiv.org/pdf/2505.02971v1)

**Authors**: Anjila Budathoki, Manish Dhakal

**Abstract**: Adversarial attacks have been fairly explored for computer vision and vision-language models. However, the avenue of adversarial attack for the vision language segmentation models (VLSMs) is still under-explored, especially for medical image analysis.   Thus, we have investigated the robustness of VLSMs against adversarial attacks for 2D medical images with different modalities with radiology, photography, and endoscopy. The main idea of this project was to assess the robustness of the fine-tuned VLSMs specially in the medical domain setting to address the high risk scenario.   First, we have fine-tuned pre-trained VLSMs for medical image segmentation with adapters.   Then, we have employed adversarial attacks -- projected gradient descent (PGD) and fast gradient sign method (FGSM) -- on that fine-tuned model to determine its robustness against adversaries.   We have reported models' performance decline to analyze the adversaries' impact.   The results exhibit significant drops in the DSC and IoU scores after the introduction of these adversaries. Furthermore, we also explored universal perturbation but were not able to find for the medical images.   \footnote{https://github.com/anjilab/secure-private-ai}



## **48. Constrained Adversarial Learning for Automated Software Testing: a literature review**

cs.SE

36 pages, 4 tables, 2 figures, Discover Applied Sciences journal

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2303.07546v2) [paper-pdf](http://arxiv.org/pdf/2303.07546v2)

**Authors**: João Vitorino, Tiago Dias, Tiago Fonseca, Eva Maia, Isabel Praça

**Abstract**: It is imperative to safeguard computer applications and information systems against the growing number of cyber-attacks. Automated software testing tools can be developed to quickly analyze many lines of code and detect vulnerabilities by generating function-specific testing data. This process draws similarities to the constrained adversarial examples generated by adversarial machine learning methods, so there could be significant benefits to the integration of these methods in testing tools to identify possible attack vectors. Therefore, this literature review is focused on the current state-of-the-art of constrained data generation approaches applied for adversarial learning and software testing, aiming to guide researchers and developers to enhance their software testing tools with adversarial testing methods and improve the resilience and robustness of their information systems. The found approaches were systematized, and the advantages and limitations of those specific for white-box, grey-box, and black-box testing were analyzed, identifying research gaps and opportunities to automate the testing tools with data generated by adversarial attacks.



## **49. Commitment Attacks on Ethereum's Reward Mechanism**

cs.CR

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2407.19479v2) [paper-pdf](http://arxiv.org/pdf/2407.19479v2)

**Authors**: Roozbeh Sarenche, Ertem Nusret Tas, Barnabe Monnot, Caspar Schwarz-Schilling, Bart Preneel

**Abstract**: Validators in permissionless, large-scale blockchains, such as Ethereum, are typically payoff-maximizing, rational actors. Ethereum relies on in-protocol incentives, like rewards for correct and timely votes, to induce honest behavior and secure the blockchain. However, external incentives, such as the block proposer's opportunity to capture maximal extractable value (MEV), may tempt validators to deviate from honest protocol participation.   We show a series of commitment attacks on LMD GHOST, a core part of Ethereum's consensus mechanism. We demonstrate how a single adversarial block proposer can orchestrate long-range chain reorganizations by manipulating Ethereum's reward system for timely votes. These attacks disrupt the intended balance of power between proposers and voters: by leveraging credible threats, the adversarial proposer can coerce voters from previous slots into supporting blocks that conflict with the honest chain, enabling a chain reorganization.   In response, we introduce a novel reward mechanism that restores the voters' role as a check against proposer power. Our proposed mitigation is fairer and more decentralized, not only in the context of these attacks, but also practical for implementation in Ethereum.



## **50. Robustness questions the interpretability of graph neural networks: what to do?**

cs.LG

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02566v1) [paper-pdf](http://arxiv.org/pdf/2505.02566v1)

**Authors**: Kirill Lukyanov, Georgii Sazonov, Serafim Boyarsky, Ilya Makarov

**Abstract**: Graph Neural Networks (GNNs) have become a cornerstone in graph-based data analysis, with applications in diverse domains such as bioinformatics, social networks, and recommendation systems. However, the interplay between model interpretability and robustness remains poorly understood, especially under adversarial scenarios like poisoning and evasion attacks. This paper presents a comprehensive benchmark to systematically analyze the impact of various factors on the interpretability of GNNs, including the influence of robustness-enhancing defense mechanisms.   We evaluate six GNN architectures based on GCN, SAGE, GIN, and GAT across five datasets from two distinct domains, employing four interpretability metrics: Fidelity, Stability, Consistency, and Sparsity. Our study examines how defenses against poisoning and evasion attacks, applied before and during model training, affect interpretability and highlights critical trade-offs between robustness and interpretability. The framework will be published as open source.   The results reveal significant variations in interpretability depending on the chosen defense methods and model architecture characteristics. By establishing a standardized benchmark, this work provides a foundation for developing GNNs that are both robust to adversarial threats and interpretable, facilitating trust in their deployment in sensitive applications.



