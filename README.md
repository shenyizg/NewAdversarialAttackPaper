# Latest Adversarial Attack Papers
**update at 2025-05-03 15:43:14**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Fully passive quantum random number generation with untrusted light**

quant-ph

21 pages, 9 figures

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00636v1) [paper-pdf](http://arxiv.org/pdf/2505.00636v1)

**Authors**: KaiWei Qiu, Yu Cai, Nelly H. Y. Ng, Jing Yan Haw

**Abstract**: Quantum random number generators (QRNGs) harness the inherent unpredictability of quantum mechanics to produce true randomness. Yet, in many optical implementations, the light source remains a potential vulnerability - susceptible to deviations from ideal behavior and even adversarial eavesdropping. Source-device-independent (SDI) protocols address this with a pragmatic strategy, by removing trust assumptions on the source, and instead rely on realistic modelling and characterization of the measurement device. In this work, we enhance an existing SDI-QRNG protocol by eliminating the need for a perfectly balanced beam splitter within the trusted measurement device, which is an idealized assumption made for the simplification of security analysis. We demonstrate that certified randomness can still be reliably extracted across a wide range of beam-splitting ratios, significantly improving the protocol's practicality and robustness. Using only off-the-shelf components, our implementation achieves real-time randomness generation rates of 0.347 Gbps. We also experimentally validate the protocol's resilience against adversarial attacks and highlight its self-testing capabilities. These advances mark a significant step toward practical, lightweight, high-performance, fully-passive, and composably secure QRNGs suitable for real-world deployment.



## **2. Fast and Low-Cost Genomic Foundation Models via Outlier Removal**

cs.LG

International Conference on Machine Learning (ICML) 2025

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00598v1) [paper-pdf](http://arxiv.org/pdf/2505.00598v1)

**Authors**: Haozheng Luo, Chenghao Qiu, Maojiang Su, Zhihan Zhou, Zoe Mehta, Guo Ye, Jerry Yao-Chieh Hu, Han Liu

**Abstract**: We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GERM. Unlike existing GFM benchmarks, GERM offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Empirically, transformer-based models exhibit greater robustness to adversarial perturbations compared to HyenaDNA, highlighting the impact of architectural design on vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features.



## **3. AMUN: Adversarial Machine UNlearning**

cs.LG

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2503.00917v2) [paper-pdf](http://arxiv.org/pdf/2503.00917v2)

**Authors**: Ali Ebrahimpour-Boroojeny, Hari Sundaram, Varun Chandrasekaran

**Abstract**: Machine unlearning, where users can request the deletion of a forget dataset, is becoming increasingly important because of numerous privacy regulations. Initial works on ``exact'' unlearning (e.g., retraining) incur large computational overheads. However, while computationally inexpensive, ``approximate'' methods have fallen short of reaching the effectiveness of exact unlearning: models produced fail to obtain comparable accuracy and prediction confidence on both the forget and test (i.e., unseen) dataset. Exploiting this observation, we propose a new unlearning method, Adversarial Machine UNlearning (AMUN), that outperforms prior state-of-the-art (SOTA) methods for image classification. AMUN lowers the confidence of the model on the forget samples by fine-tuning the model on their corresponding adversarial examples. Adversarial examples naturally belong to the distribution imposed by the model on the input space; fine-tuning the model on the adversarial examples closest to the corresponding forget samples (a) localizes the changes to the decision boundary of the model around each forget sample and (b) avoids drastic changes to the global behavior of the model, thereby preserving the model's accuracy on test samples. Using AMUN for unlearning a random $10\%$ of CIFAR-10 samples, we observe that even SOTA membership inference attacks cannot do better than random guessing.



## **4. Analysis of the vulnerability of machine learning regression models to adversarial attacks using data from 5G wireless networks**

cs.CR

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00487v1) [paper-pdf](http://arxiv.org/pdf/2505.00487v1)

**Authors**: Leonid Legashev, Artur Zhigalov, Denis Parfenov

**Abstract**: This article describes the process of creating a script and conducting an analytical study of a dataset using the DeepMIMO emulator. An advertorial attack was carried out using the FGSM method to maximize the gradient. A comparison is made of the effectiveness of binary classifiers in the task of detecting distorted data. The dynamics of changes in the quality indicators of the regression model were analyzed in conditions without adversarial attacks, during an adversarial attack and when the distorted data was isolated. It is shown that an adversarial FGSM attack with gradient maximization leads to an increase in the value of the MSE metric by 33% and a decrease in the R2 indicator by 10% on average. The LightGBM binary classifier effectively identifies data with adversarial anomalies with 98% accuracy. Regression machine learning models are susceptible to adversarial attacks, but rapid analysis of network traffic and data transmitted over the network makes it possible to identify malicious activity



## **5. HoneyWin: High-Interaction Windows Honeypot in Enterprise Environment**

cs.CR

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00465v1) [paper-pdf](http://arxiv.org/pdf/2505.00465v1)

**Authors**: Yan Lin Aung, Yee Loon Khoo, Davis Yang Zheng, Bryan Swee Duo, Sudipta Chattopadhyay, Jianying Zhou, Liming Lu, Weihan Goh

**Abstract**: Windows operating systems (OS) are ubiquitous in enterprise Information Technology (IT) and operational technology (OT) environments. Due to their widespread adoption and known vulnerabilities, they are often the primary targets of malware and ransomware attacks. With 93% of the ransomware targeting Windows-based systems, there is an urgent need for advanced defensive mechanisms to detect, analyze, and mitigate threats effectively. In this paper, we propose HoneyWin a high-interaction Windows honeypot that mimics an enterprise IT environment. The HoneyWin consists of three Windows 11 endpoints and an enterprise-grade gateway provisioned with comprehensive network traffic capturing, host-based logging, deceptive tokens, endpoint security and real-time alerts capabilities. The HoneyWin has been deployed live in the wild for 34 days and receives more than 5.79 million unsolicited connections, 1.24 million login attempts, 5 and 354 successful logins via remote desktop protocol (RDP) and secure shell (SSH) respectively. The adversary interacted with the deceptive token in one of the RDP sessions and exploited the public-facing endpoint to initiate the Simple Mail Transfer Protocol (SMTP) brute-force bot attack via SSH sessions. The adversary successfully harvested 1,250 SMTP credentials after attempting 151,179 credentials during the attack.



## **6. GAN-based Generator of Adversarial Attack on Intelligent End-to-End Autoencoder-based Communication System**

cs.IT

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00395v1) [paper-pdf](http://arxiv.org/pdf/2505.00395v1)

**Authors**: Jianyuan Chen, Lin Zhang, Zuwei Chen, Yawen Chen, Hongcheng Zhuang

**Abstract**: Deep neural networks have been applied in wireless communications system to intelligently adapt to dynamically changing channel conditions, while the users are still under the threat of the malicious attacks due to the broadcasting property of wireless channels. However, most attack models require the knowledge of the target details, which is difficult to be implemented in real systems. Our objective is to develop an attack model with no requirement for the target information, while enhancing the block error rate. In our design, we propose a novel Generative Adversarial Networks(GANs) based attack architecture, which exploits the property of deep learning models being vulnerable to perturbations induced by dynamically changing channel conditions. In the proposed generator, the attack network is composed of convolution layer, convolution transpose layer and linear layer. Then we present the training strategy and the details of the training algorithm. Subsequently, we propose the validation strategy to evaluate the performance of the generator. Simulations are conducted and the results show that our proposed adversarial attack generator achieve better block error rate attack performance than that of benchmark schemes over Additive White Gaussian Noise (AWGN) channel, Rayleigh channel and High-Speed Railway channel.



## **7. SacFL: Self-Adaptive Federated Continual Learning for Resource-Constrained End Devices**

cs.LG

Accepted by TNNLS 2025

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2505.00365v1) [paper-pdf](http://arxiv.org/pdf/2505.00365v1)

**Authors**: Zhengyi Zhong, Weidong Bao, Ji Wang, Jianguo Chen, Lingjuan Lyu, Wei Yang Bryan Lim

**Abstract**: The proliferation of end devices has led to a distributed computing paradigm, wherein on-device machine learning models continuously process diverse data generated by these devices. The dynamic nature of this data, characterized by continuous changes or data drift, poses significant challenges for on-device models. To address this issue, continual learning (CL) is proposed, enabling machine learning models to incrementally update their knowledge and mitigate catastrophic forgetting. However, the traditional centralized approach to CL is unsuitable for end devices due to privacy and data volume concerns. In this context, federated continual learning (FCL) emerges as a promising solution, preserving user data locally while enhancing models through collaborative updates. Aiming at the challenges of limited storage resources for CL, poor autonomy in task shift detection, and difficulty in coping with new adversarial tasks in FCL scenario, we propose a novel FCL framework named SacFL. SacFL employs an Encoder-Decoder architecture to separate task-robust and task-sensitive components, significantly reducing storage demands by retaining lightweight task-sensitive components for resource-constrained end devices. Moreover, $\rm{SacFL}$ leverages contrastive learning to introduce an autonomous data shift detection mechanism, enabling it to discern whether a new task has emerged and whether it is a benign task. This capability ultimately allows the device to autonomously trigger CL or attack defense strategy without additional information, which is more practical for end devices. Comprehensive experiments conducted on multiple text and image datasets, such as Cifar100 and THUCNews, have validated the effectiveness of $\rm{SacFL}$ in both class-incremental and domain-incremental scenarios. Furthermore, a demo system has been developed to verify its practicality.



## **8. TaeBench: Improving Quality of Toxic Adversarial Examples**

cs.CR

Accepted for publication in NAACL 2025. The official version will be  available in the ACL Anthology

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2410.05573v2) [paper-pdf](http://arxiv.org/pdf/2410.05573v2)

**Authors**: Xuan Zhu, Dmitriy Bespalov, Liwen You, Ninad Kulkarni, Yanjun Qi

**Abstract**: Toxicity text detectors can be vulnerable to adversarial examples - small perturbations to input text that fool the systems into wrong detection. Existing attack algorithms are time-consuming and often produce invalid or ambiguous adversarial examples, making them less useful for evaluating or improving real-world toxicity content moderators. This paper proposes an annotation pipeline for quality control of generated toxic adversarial examples (TAE). We design model-based automated annotation and human-based quality verification to assess the quality requirements of TAE. Successful TAE should fool a target toxicity model into making benign predictions, be grammatically reasonable, appear natural like human-generated text, and exhibit semantic toxicity. When applying these requirements to more than 20 state-of-the-art (SOTA) TAE attack recipes, we find many invalid samples from a total of 940k raw TAE attack generations. We then utilize the proposed pipeline to filter and curate a high-quality TAE dataset we call TaeBench (of size 264k). Empirically, we demonstrate that TaeBench can effectively transfer-attack SOTA toxicity content moderation models and services. Our experiments also show that TaeBench with adversarial training achieve significant improvements of the robustness of two toxicity detectors.



## **9. SoK: Security and Privacy Risks of Healthcare AI**

cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2409.07415v2) [paper-pdf](http://arxiv.org/pdf/2409.07415v2)

**Authors**: Yuanhaur Chang, Han Liu, Chenyang Lu, Ning Zhang

**Abstract**: The integration of artificial intelligence (AI) and machine learning (ML) into healthcare systems holds great promise for enhancing patient care and care delivery efficiency; however, it also exposes sensitive data and system integrity to potential cyberattacks. Current security and privacy (S&P) research on healthcare AI is highly unbalanced in terms of healthcare deployment scenarios and threat models, and has a disconnected focus with the biomedical research community. This hinders a comprehensive understanding of the risks that healthcare AI entails. To address this gap, this paper takes a thorough examination of existing healthcare AI S&P research, providing a unified framework that allows the identification of under-explored areas. Our survey presents a systematic overview of healthcare AI attacks and defenses, and points out challenges and research opportunities for each AI-driven healthcare application domain. Through our experimental analysis of different threat models and feasibility studies on under-explored adversarial attacks, we provide compelling insights into the pressing need for cybersecurity research in the rapidly evolving field of healthcare AI.



## **10. Adversarial Data Poisoning Attacks on Quantum Machine Learning in the NISQ Era**

quant-ph

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2411.14412v3) [paper-pdf](http://arxiv.org/pdf/2411.14412v3)

**Authors**: Satwik Kundu, Swaroop Ghosh

**Abstract**: With the growing interest in Quantum Machine Learning (QML) and the increasing availability of quantum computers through cloud providers, addressing the potential security risks associated with QML has become an urgent priority. One key concern in the QML domain is the threat of data poisoning attacks in the current quantum cloud setting. Adversarial access to training data could severely compromise the integrity and availability of QML models. Classical data poisoning techniques require significant knowledge and training to generate poisoned data, and lack noise resilience, making them ineffective for QML models in the Noisy Intermediate Scale Quantum (NISQ) era. In this work, we first propose a simple yet effective technique to measure intra-class encoder state similarity (ESS) by analyzing the outputs of encoding circuits. Leveraging this approach, we introduce a \underline{Qu}antum \underline{I}ndiscriminate \underline{D}ata Poisoning attack, QUID. Through extensive experiments conducted in both noiseless and noisy environments (e.g., IBM\_Brisbane's noise), across various architectures and datasets, QUID achieves up to $92\%$ accuracy degradation in model performance compared to baseline models and up to $75\%$ accuracy degradation compared to random label-flipping. We also tested QUID against state-of-the-art classical defenses, with accuracy degradation still exceeding $50\%$, demonstrating its effectiveness. This work represents the first attempt to reevaluate data poisoning attacks in the context of QML.



## **11. Real AI Agents with Fake Memories: Fatal Context Manipulation Attacks on Web3 Agents**

cs.CR

29 pages, 21 figures

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2503.16248v2) [paper-pdf](http://arxiv.org/pdf/2503.16248v2)

**Authors**: Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, Pramod Viswanath

**Abstract**: The integration of AI agents with Web3 ecosystems harnesses their complementary potential for autonomy and openness yet also introduces underexplored security risks, as these agents dynamically interact with financial protocols and immutable smart contracts. This paper investigates the vulnerabilities of AI agents within blockchain-based financial ecosystems when exposed to adversarial threats in real-world scenarios. We introduce the concept of context manipulation, a comprehensive attack vector that exploits unprotected context surfaces, including input channels, memory modules, and external data feeds.   Through empirical analysis of ElizaOS, a decentralized AI agent framework for automated Web3 operations, we demonstrate how adversaries can manipulate context by injecting malicious instructions into prompts or historical interaction records, leading to unintended asset transfers and protocol violations which could be financially devastating.   To quantify these vulnerabilities, we design CrAIBench, a Web3 domain-specific benchmark that evaluates the robustness of AI agents against context manipulation attacks across 150+ realistic blockchain tasks, including token transfers, trading, bridges and cross-chain interactions and 500+ attack test cases using context manipulation. We systematically assess attack and defense strategies, analyzing factors like the influence of security prompts, reasoning models, and the effectiveness of alignment techniques.   Our findings show that prompt-based defenses are insufficient when adversaries corrupt stored context, achieving significant attack success rates despite these defenses. Fine-tuning-based defenses offer a more robust alternative, substantially reducing attack success rates while preserving utility on single-step tasks. This research highlights the urgent need to develop AI agents that are both secure and fiduciarily responsible.



## **12. Stochastic Subspace Descent Accelerated via Bi-fidelity Line Search**

cs.LG

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2505.00162v1) [paper-pdf](http://arxiv.org/pdf/2505.00162v1)

**Authors**: Nuojin Cheng, Alireza Doostan, Stephen Becker

**Abstract**: Efficient optimization remains a fundamental challenge across numerous scientific and engineering domains, especially when objective function and gradient evaluations are computationally expensive. While zeroth-order optimization methods offer effective approaches when gradients are inaccessible, their practical performance can be limited by the high cost associated with function queries. This work introduces the bi-fidelity stochastic subspace descent (BF-SSD) algorithm, a novel zeroth-order optimization method designed to reduce this computational burden. BF-SSD leverages a bi-fidelity framework, constructing a surrogate model from a combination of computationally inexpensive low-fidelity (LF) and accurate high-fidelity (HF) function evaluations. This surrogate model facilitates an efficient backtracking line search for step size selection, for which we provide theoretical convergence guarantees under standard assumptions. We perform a comprehensive empirical evaluation of BF-SSD across four distinct problems: a synthetic optimization benchmark, dual-form kernel ridge regression, black-box adversarial attacks on machine learning models, and transformer-based black-box language model fine-tuning. Numerical results demonstrate that BF-SSD consistently achieves superior optimization performance while requiring significantly fewer HF function evaluations compared to relevant baseline methods. This study highlights the efficacy of integrating bi-fidelity strategies within zeroth-order optimization, positioning BF-SSD as a promising and computationally efficient approach for tackling large-scale, high-dimensional problems encountered in various real-world applications.



## **13. WASP: Benchmarking Web Agent Security Against Prompt Injection Attacks**

cs.CR

Code and data: https://github.com/facebookresearch/wasp

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.18575v2) [paper-pdf](http://arxiv.org/pdf/2504.18575v2)

**Authors**: Ivan Evtimov, Arman Zharmagambetov, Aaron Grattafiori, Chuan Guo, Kamalika Chaudhuri

**Abstract**: Web navigation AI agents use language-and-vision foundation models to enhance productivity but these models are known to be susceptible to indirect prompt injections that get them to follow instructions different from the legitimate user's. Existing explorations of this threat applied to web agents often focus on a single isolated adversarial goal, test with injected instructions that are either too easy or not truly malicious, and often give the adversary unreasonable access. In order to better focus adversarial research, we construct a new benchmark called WASP (Web Agent Security against Prompt injection attacks) that introduces realistic web agent hijacking objectives and an isolated environment to test them in that does not affect real users or the live web. As part of WASP, we also develop baseline attacks against popular web agentic systems (VisualWebArena, Claude Computer Use, etc.) instantiated with various state-of-the-art models. Our evaluation shows that even AI agents backed by models with advanced reasoning capabilities and by models with instruction hierarchy mitigations are susceptible to low-effort human-written prompt injections. However, the realistic objectives in WASP also allow us to observe that agents are currently not capable enough to complete the goals of attackers end-to-end. Agents begin executing the adversarial instruction between 16 and 86% of the time but only achieve the goal between 0 and 17% of the time. Based on these findings, we argue that adversarial researchers should demonstrate stronger attacks that more consistently maintain control over the agent given realistic constraints on the adversary's power.



## **14. Active Light Modulation to Counter Manipulation of Speech Visual Content**

cs.CV

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21846v1) [paper-pdf](http://arxiv.org/pdf/2504.21846v1)

**Authors**: Hadleigh Schwartz, Xiaofeng Yan, Charles J. Carver, Xia Zhou

**Abstract**: High-profile speech videos are prime targets for falsification, owing to their accessibility and influence. This work proposes Spotlight, a low-overhead and unobtrusive system for protecting live speech videos from visual falsification of speaker identity and lip and facial motion. Unlike predominant falsification detection methods operating in the digital domain, Spotlight creates dynamic physical signatures at the event site and embeds them into all video recordings via imperceptible modulated light. These physical signatures encode semantically-meaningful features unique to the speech event, including the speaker's identity and facial motion, and are cryptographically-secured to prevent spoofing. The signatures can be extracted from any video downstream and validated against the portrayed speech content to check its integrity. Key elements of Spotlight include (1) a framework for generating extremely compact (i.e., 150-bit), pose-invariant speech video features, based on locality-sensitive hashing; and (2) an optical modulation scheme that embeds >200 bps into video while remaining imperceptible both in video and live. Prototype experiments on extensive video datasets show Spotlight achieves AUCs $\geq$ 0.99 and an overall true positive rate of 100% in detecting falsified videos. Further, Spotlight is highly robust across recording conditions, video post-processing techniques, and white-box adversarial attacks on its video feature extraction methodologies.



## **15. Adversarial KA**

cs.LG

8 pages, 3 figures; minor revision, question 4.1 added

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.05255v2) [paper-pdf](http://arxiv.org/pdf/2504.05255v2)

**Authors**: Sviatoslav Dzhenzher, Michael H. Freedman

**Abstract**: Regarding the representation theorem of Kolmogorov and Arnold (KA) as an algorithm for representing or {\guillemotleft}expressing{\guillemotright} functions, we test its robustness by analyzing its ability to withstand adversarial attacks. We find KA to be robust to countable collections of continuous adversaries, but unearth a question about the equi-continuity of the outer functions that, so far, obstructs taking limits and defeating continuous groups of adversaries. This question on the regularity of the outer functions is relevant to the debate over the applicability of KA to the general theory of NNs.



## **16. Hoist with His Own Petard: Inducing Guardrails to Facilitate Denial-of-Service Attacks on Retrieval-Augmented Generation of LLMs**

cs.CR

11 pages, 6 figures. This work will be submitted to the IEEE for  possible publication

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21680v1) [paper-pdf](http://arxiv.org/pdf/2504.21680v1)

**Authors**: Pan Suo, Yu-Ming Shang, San-Chuan Guo, Xi Zhang

**Abstract**: Retrieval-Augmented Generation (RAG) integrates Large Language Models (LLMs) with external knowledge bases, improving output quality while introducing new security risks. Existing studies on RAG vulnerabilities typically focus on exploiting the retrieval mechanism to inject erroneous knowledge or malicious texts, inducing incorrect outputs. However, these approaches overlook critical weaknesses within LLMs, leaving important attack vectors unexplored and limiting the scope and efficiency of attacks. In this paper, we uncover a novel vulnerability: the safety guardrails of LLMs, while designed for protection, can also be exploited as an attack vector by adversaries. Building on this vulnerability, we propose MutedRAG, a novel denial-of-service attack that reversely leverages the guardrails of LLMs to undermine the availability of RAG systems. By injecting minimalistic jailbreak texts, such as "\textit{How to build a bomb}", into the knowledge base, MutedRAG intentionally triggers the LLM's safety guardrails, causing the system to reject legitimate queries. Besides, due to the high sensitivity of guardrails, a single jailbreak sample can affect multiple queries, effectively amplifying the efficiency of attacks while reducing their costs. Experimental results on three datasets demonstrate that MutedRAG achieves an attack success rate exceeding 60% in many scenarios, requiring only less than one malicious text to each target query on average. In addition, we evaluate potential defense strategies against MutedRAG, finding that some of current mechanisms are insufficient to mitigate this threat, underscoring the urgent need for more robust solutions.



## **17. Diffusion-based Adversarial Identity Manipulation for Facial Privacy Protection**

cs.CV

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21646v1) [paper-pdf](http://arxiv.org/pdf/2504.21646v1)

**Authors**: Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo

**Abstract**: The success of face recognition (FR) systems has led to serious privacy concerns due to potential unauthorized surveillance and user tracking on social networks. Existing methods for enhancing privacy fail to generate natural face images that can protect facial privacy. In this paper, we propose diffusion-based adversarial identity manipulation (DiffAIM) to generate natural and highly transferable adversarial faces against malicious FR systems. To be specific, we manipulate facial identity within the low-dimensional latent space of a diffusion model. This involves iteratively injecting gradient-based adversarial identity guidance during the reverse diffusion process, progressively steering the generation toward the desired adversarial faces. The guidance is optimized for identity convergence towards a target while promoting semantic divergence from the source, facilitating effective impersonation while maintaining visual naturalness. We further incorporate structure-preserving regularization to preserve facial structure consistency during manipulation. Extensive experiments on both face verification and identification tasks demonstrate that compared with the state-of-the-art, DiffAIM achieves stronger black-box attack transferability while maintaining superior visual quality. We also demonstrate the effectiveness of the proposed approach for commercial FR APIs, including Face++ and Aliyun.



## **18. Generative AI in Financial Institution: A Global Survey of Opportunities, Threats, and Regulation**

cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21574v1) [paper-pdf](http://arxiv.org/pdf/2504.21574v1)

**Authors**: Bikash Saha, Nanda Rani, Sandeep Kumar Shukla

**Abstract**: Generative Artificial Intelligence (GenAI) is rapidly reshaping the global financial landscape, offering unprecedented opportunities to enhance customer engagement, automate complex workflows, and extract actionable insights from vast financial data. This survey provides an overview of GenAI adoption across the financial ecosystem, examining how banks, insurers, asset managers, and fintech startups worldwide are integrating large language models and other generative tools into their operations. From AI-powered virtual assistants and personalized financial advisory to fraud detection and compliance automation, GenAI is driving innovation across functions. However, this transformation comes with significant cybersecurity and ethical risks. We discuss emerging threats such as AI-generated phishing, deepfake-enabled fraud, and adversarial attacks on AI systems, as well as concerns around bias, opacity, and data misuse. The evolving global regulatory landscape is explored in depth, including initiatives by major financial regulators and international efforts to develop risk-based AI governance. Finally, we propose best practices for secure and responsible adoption - including explainability techniques, adversarial testing, auditability, and human oversight. Drawing from academic literature, industry case studies, and policy frameworks, this chapter offers a perspective on how the financial sector can harness GenAI's transformative potential while navigating the complex risks it introduces.



## **19. A Test Suite for Efficient Robustness Evaluation of Face Recognition Systems**

cs.SE

IEEE Transactions on Reliability

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21420v1) [paper-pdf](http://arxiv.org/pdf/2504.21420v1)

**Authors**: Ruihan Zhang, Jun Sun

**Abstract**: Face recognition is a widely used authentication technology in practice, where robustness is required. It is thus essential to have an efficient and easy-to-use method for evaluating the robustness of (possibly third-party) trained face recognition systems. Existing approaches to evaluating the robustness of face recognition systems are either based on empirical evaluation (e.g., measuring attacking success rate using state-of-the-art attacking methods) or formal analysis (e.g., measuring the Lipschitz constant). While the former demands significant user efforts and expertise, the latter is extremely time-consuming. In pursuit of a comprehensive, efficient, easy-to-use and scalable estimation of the robustness of face recognition systems, we take an old-school alternative approach and introduce RobFace, i.e., evaluation using an optimised test suite. It contains transferable adversarial face images that are designed to comprehensively evaluate a face recognition system's robustness along a variety of dimensions. RobFace is system-agnostic and still consistent with system-specific empirical evaluation or formal analysis. We support this claim through extensive experimental results with various perturbations on multiple face recognition systems. To our knowledge, RobFace is the first system-agnostic robustness estimation test suite.



## **20. Exploiting Defenses against GAN-Based Feature Inference Attacks in Federated Learning**

cs.CR

Published in ACM Transactions on Knowledge Discovery from Data  (TKDD), 2025

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2004.12571v5) [paper-pdf](http://arxiv.org/pdf/2004.12571v5)

**Authors**: Xinjian Luo, Xianglong Zhang

**Abstract**: Federated learning (FL) is a decentralized model training framework that aims to merge isolated data islands while maintaining data privacy. However, recent studies have revealed that Generative Adversarial Network (GAN) based attacks can be employed in FL to learn the distribution of private datasets and reconstruct recognizable images. In this paper, we exploit defenses against GAN-based attacks in FL and propose a framework, Anti-GAN, to prevent attackers from learning the real distribution of the victim's data. The core idea of Anti-GAN is to manipulate the visual features of private training images to make them indistinguishable to human eyes even restored by attackers. Specifically, Anti-GAN projects the private dataset onto a GAN's generator and combines the generated fake images with the actual images to create the training dataset, which is then used for federated model training. The experimental results demonstrate that Anti-GAN is effective in preventing attackers from learning the distribution of private images while causing minimal harm to the accuracy of the federated model.



## **21. How to Backdoor the Knowledge Distillation**

cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21323v1) [paper-pdf](http://arxiv.org/pdf/2504.21323v1)

**Authors**: Chen Wu, Qian Ma, Prasenjit Mitra, Sencun Zhu

**Abstract**: Knowledge distillation has become a cornerstone in modern machine learning systems, celebrated for its ability to transfer knowledge from a large, complex teacher model to a more efficient student model. Traditionally, this process is regarded as secure, assuming the teacher model is clean. This belief stems from conventional backdoor attacks relying on poisoned training data with backdoor triggers and attacker-chosen labels, which are not involved in the distillation process. Instead, knowledge distillation uses the outputs of a clean teacher model to guide the student model, inherently preventing recognition or response to backdoor triggers as intended by an attacker. In this paper, we challenge this assumption by introducing a novel attack methodology that strategically poisons the distillation dataset with adversarial examples embedded with backdoor triggers. This technique allows for the stealthy compromise of the student model while maintaining the integrity of the teacher model. Our innovative approach represents the first successful exploitation of vulnerabilities within the knowledge distillation process using clean teacher models. Through extensive experiments conducted across various datasets and attack settings, we demonstrate the robustness, stealthiness, and effectiveness of our method. Our findings reveal previously unrecognized vulnerabilities and pave the way for future research aimed at securing knowledge distillation processes against backdoor attacks.



## **22. Round Trip Translation Defence against Large Language Model Jailbreaking Attacks**

cs.CL

6 pages, 6 figures

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2402.13517v2) [paper-pdf](http://arxiv.org/pdf/2402.13517v2)

**Authors**: Canaan Yung, Hadi Mohaghegh Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Large language models (LLMs) are susceptible to social-engineered attacks that are human-interpretable but require a high level of comprehension for LLMs to counteract. Existing defensive measures can only mitigate less than half of these attacks at most. To address this issue, we propose the Round Trip Translation (RTT) method, the first algorithm specifically designed to defend against social-engineered attacks on LLMs. RTT paraphrases the adversarial prompt and generalizes the idea conveyed, making it easier for LLMs to detect induced harmful behavior. This method is versatile, lightweight, and transferrable to different LLMs. Our defense successfully mitigated over 70% of Prompt Automatic Iterative Refinement (PAIR) attacks, which is currently the most effective defense to the best of our knowledge. We are also the first to attempt mitigating the MathsAttack and reduced its attack success rate by almost 40%. Our code is publicly available at https://github.com/Cancanxxx/Round_Trip_Translation_Defence   This version of the article has been accepted for publication, after peer review (when applicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The Version of Record is available online at: https://doi.org/10.48550/arXiv.2402.13517 Use of this Accepted Version is subject to the publisher's Accepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscript-terms



## **23. Quantifying the Noise of Structural Perturbations on Graph Adversarial Attacks**

cs.LG

Under Review

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.20869v2) [paper-pdf](http://arxiv.org/pdf/2504.20869v2)

**Authors**: Junyuan Fang, Han Yang, Haixian Wen, Jiajing Wu, Zibin Zheng, Chi K. Tse

**Abstract**: Graph neural networks have been widely utilized to solve graph-related tasks because of their strong learning power in utilizing the local information of neighbors. However, recent studies on graph adversarial attacks have proven that current graph neural networks are not robust against malicious attacks. Yet much of the existing work has focused on the optimization objective based on attack performance to obtain (near) optimal perturbations, but paid less attention to the strength quantification of each perturbation such as the injection of a particular node/link, which makes the choice of perturbations a black-box model that lacks interpretability. In this work, we propose the concept of noise to quantify the attack strength of each adversarial link. Furthermore, we propose three attack strategies based on the defined noise and classification margins in terms of single and multiple steps optimization. Extensive experiments conducted on benchmark datasets against three representative graph neural networks demonstrate the effectiveness of the proposed attack strategies. Particularly, we also investigate the preferred patterns of effective adversarial perturbations by analyzing the corresponding properties of the selected perturbation nodes.



## **24. Iron Sharpens Iron: Defending Against Attacks in Machine-Generated Text Detection with Adversarial Training**

cs.CR

Accepted by ACL 2025 Main Conference

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2502.12734v2) [paper-pdf](http://arxiv.org/pdf/2502.12734v2)

**Authors**: Yuanfan Li, Zhaohan Zhang, Chengzhengxu Li, Chao Shen, Xiaoming Liu

**Abstract**: Machine-generated Text (MGT) detection is crucial for regulating and attributing online texts. While the existing MGT detectors achieve strong performance, they remain vulnerable to simple perturbations and adversarial attacks. To build an effective defense against malicious perturbations, we view MGT detection from a threat modeling perspective, that is, analyzing the model's vulnerability from an adversary's point of view and exploring effective mitigations. To this end, we introduce an adversarial framework for training a robust MGT detector, named GREedy Adversary PromoTed DefendER (GREATER). The GREATER consists of two key components: an adversary GREATER-A and a detector GREATER-D. The GREATER-D learns to defend against the adversarial attack from GREATER-A and generalizes the defense to other attacks. GREATER-A identifies and perturbs the critical tokens in embedding space, along with greedy search and pruning to generate stealthy and disruptive adversarial examples. Besides, we update the GREATER-A and GREATER-D synchronously, encouraging the GREATER-D to generalize its defense to different attacks and varying attack intensities. Our experimental results across 10 text perturbation strategies and 6 adversarial attacks show that our GREATER-D reduces the Attack Success Rate (ASR) by 0.67% compared with SOTA defense methods while our GREATER-A is demonstrated to be more effective and efficient than SOTA attack approaches. Codes and dataset are available in https://github.com/Liyuuuu111/GREATER.



## **25. Generate-then-Verify: Reconstructing Data from Limited Published Statistics**

stat.ML

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.21199v1) [paper-pdf](http://arxiv.org/pdf/2504.21199v1)

**Authors**: Terrance Liu, Eileen Xiao, Pratiksha Thaker, Adam Smith, Zhiwei Steven Wu

**Abstract**: We study the problem of reconstructing tabular data from aggregate statistics, in which the attacker aims to identify interesting claims about the sensitive data that can be verified with 100% certainty given the aggregates. Successful attempts in prior work have conducted studies in settings where the set of published statistics is rich enough that entire datasets can be reconstructed with certainty. In our work, we instead focus on the regime where many possible datasets match the published statistics, making it impossible to reconstruct the entire private dataset perfectly (i.e., when approaches in prior work fail). We propose the problem of partial data reconstruction, in which the goal of the adversary is to instead output a $\textit{subset}$ of rows and/or columns that are $\textit{guaranteed to be correct}$. We introduce a novel integer programming approach that first $\textbf{generates}$ a set of claims and then $\textbf{verifies}$ whether each claim holds for all possible datasets consistent with the published aggregates. We evaluate our approach on the housing-level microdata from the U.S. Decennial Census release, demonstrating that privacy violations can still persist even when information published about such data is relatively sparse.



## **26. Demo: ViolentUTF as An Accessible Platform for Generative AI Red Teaming**

cs.CR

3 pages, 1 figure, 1 table. This is a demo paper for  CyberWarrior2025. The video demo is at https://youtu.be/c-UCYXq0rfY. Codes  will be shared when the competition concludes in June 2025 due to embargo  requirements

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.10603v2) [paper-pdf](http://arxiv.org/pdf/2504.10603v2)

**Authors**: Tam n. Nguyen

**Abstract**: The rapid integration of Generative AI (GenAI) into various applications necessitates robust risk management strategies which includes Red Teaming (RT) - an evaluation method for simulating adversarial attacks. Unfortunately, RT for GenAI is often hindered by technical complexity, lack of user-friendly interfaces, and inadequate reporting features. This paper introduces Violent UTF - an accessible, modular, and scalable platform for GenAI red teaming. Through intuitive interfaces (Web GUI, CLI, API, MCP) powered by LLMs and for LLMs, Violent UTF aims to empower non-technical domain experts and students alongside technical experts, facilitate comprehensive security evaluation by unifying capabilities from RT frameworks like Microsoft PyRIT, Nvidia Garak and its own specialized evaluators. ViolentUTF is being used for evaluating the robustness of a flagship LLM-based product in a large US Government department. It also demonstrates effectiveness in evaluating LLMs' cross-domain reasoning capability between cybersecurity and behavioral psychology.



## **27. AegisLLM: Scaling Agentic Systems for Self-Reflective Defense in LLM Security**

cs.LG

ICLR 2025 Workshop BuildingTrust

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20965v1) [paper-pdf](http://arxiv.org/pdf/2504.20965v1)

**Authors**: Zikui Cai, Shayan Shabihi, Bang An, Zora Che, Brian R. Bartoldson, Bhavya Kailkhura, Tom Goldstein, Furong Huang

**Abstract**: We introduce AegisLLM, a cooperative multi-agent defense against adversarial attacks and information leakage. In AegisLLM, a structured workflow of autonomous agents - orchestrator, deflector, responder, and evaluator - collaborate to ensure safe and compliant LLM outputs, while self-improving over time through prompt optimization. We show that scaling agentic reasoning system at test-time - both by incorporating additional agent roles and by leveraging automated prompt optimization (such as DSPy)- substantially enhances robustness without compromising model utility. This test-time defense enables real-time adaptability to evolving attacks, without requiring model retraining. Comprehensive evaluations across key threat scenarios, including unlearning and jailbreaking, demonstrate the effectiveness of AegisLLM. On the WMDP unlearning benchmark, AegisLLM achieves near-perfect unlearning with only 20 training examples and fewer than 300 LM calls. For jailbreaking benchmarks, we achieve 51% improvement compared to the base model on StrongReject, with false refusal rates of only 7.9% on PHTest compared to 18-55% for comparable methods. Our results highlight the advantages of adaptive, agentic reasoning over static defenses, establishing AegisLLM as a strong runtime alternative to traditional approaches based on model modifications. Code is available at https://github.com/zikuicai/aegisllm



## **28. NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary**

cs.CV

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2503.00063v4) [paper-pdf](http://arxiv.org/pdf/2503.00063v4)

**Authors**: Zezeng Li, Xiaoyu Du, Na Lei, Liming Chen, Weimin Wang

**Abstract**: Adversarial attacks exploit the vulnerability of deep models against adversarial samples. Existing point cloud attackers are tailored to specific models, iteratively optimizing perturbations based on gradients in either a white-box or black-box setting. Despite their promising attack performance, they often struggle to produce transferable adversarial samples due to overfitting the specific parameters of surrogate models. To overcome this issue, we shift our focus to the data distribution itself and introduce a novel approach named NoPain, which employs optimal transport (OT) to identify the inherent singular boundaries of the data manifold for cross-network point cloud attacks. Specifically, we first calculate the OT mapping from noise to the target feature space, then identify singular boundaries by locating non-differentiable positions. Finally, we sample along singular boundaries to generate adversarial point clouds. Once the singular boundaries are determined, NoPain can efficiently produce adversarial samples without the need of iterative updates or guidance from the surrogate classifiers. Extensive experiments demonstrate that the proposed end-to-end method outperforms baseline approaches in terms of both transferability and efficiency, while also maintaining notable advantages even against defense strategies. Code and model are available at https://github.com/cognaclee/nopain



## **29. Erased but Not Forgotten: How Backdoors Compromise Concept Erasure**

cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.21072v1) [paper-pdf](http://arxiv.org/pdf/2504.21072v1)

**Authors**: Jonas Henry Grebe, Tobias Braun, Marcus Rohrbach, Anna Rohrbach

**Abstract**: The expansion of large-scale text-to-image diffusion models has raised growing concerns about their potential to generate undesirable or harmful content, ranging from fabricated depictions of public figures to sexually explicit images. To mitigate these risks, prior work has devised machine unlearning techniques that attempt to erase unwanted concepts through fine-tuning. However, in this paper, we introduce a new threat model, Toxic Erasure (ToxE), and demonstrate how recent unlearning algorithms, including those explicitly designed for robustness, can be circumvented through targeted backdoor attacks. The threat is realized by establishing a link between a trigger and the undesired content. Subsequent unlearning attempts fail to erase this link, allowing adversaries to produce harmful content. We instantiate ToxE via two established backdoor attacks: one targeting the text encoder and another manipulating the cross-attention layers. Further, we introduce Deep Intervention Score-based Attack (DISA), a novel, deeper backdoor attack that optimizes the entire U-Net using a score-based objective, improving the attack's persistence across different erasure methods. We evaluate five recent concept erasure methods against our threat model. For celebrity identity erasure, our deep attack circumvents erasure with up to 82% success, averaging 57% across all erasure methods. For explicit content erasure, ToxE attacks can elicit up to 9 times more exposed body parts, with DISA yielding an average increase by a factor of 2.9. These results highlight a critical security gap in current unlearning strategies.



## **30. Mitigating the Structural Bias in Graph Adversarial Defenses**

cs.LG

Under Review

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20848v1) [paper-pdf](http://arxiv.org/pdf/2504.20848v1)

**Authors**: Junyuan Fang, Huimin Liu, Han Yang, Jiajing Wu, Zibin Zheng, Chi K. Tse

**Abstract**: In recent years, graph neural networks (GNNs) have shown great potential in addressing various graph structure-related downstream tasks. However, recent studies have found that current GNNs are susceptible to malicious adversarial attacks. Given the inevitable presence of adversarial attacks in the real world, a variety of defense methods have been proposed to counter these attacks and enhance the robustness of GNNs. Despite the commendable performance of these defense methods, we have observed that they tend to exhibit a structural bias in terms of their defense capability on nodes with low degree (i.e., tail nodes), which is similar to the structural bias of traditional GNNs on nodes with low degree in the clean graph. Therefore, in this work, we propose a defense strategy by including hetero-homo augmented graph construction, $k$NN augmented graph construction, and multi-view node-wise attention modules to mitigate the structural bias of GNNs against adversarial attacks. Notably, the hetero-homo augmented graph consists of removing heterophilic links (i.e., links connecting nodes with dissimilar features) globally and adding homophilic links (i.e., links connecting nodes with similar features) for nodes with low degree. To further enhance the defense capability, an attention mechanism is adopted to adaptively combine the representations from the above two kinds of graph views. We conduct extensive experiments to demonstrate the defense and debiasing effect of the proposed strategy on benchmark datasets.



## **31. GaussTrap: Stealthy Poisoning Attacks on 3D Gaussian Splatting for Targeted Scene Confusion**

cs.CV

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20829v1) [paper-pdf](http://arxiv.org/pdf/2504.20829v1)

**Authors**: Jiaxin Hong, Sixu Chen, Shuoyang Sun, Hongyao Yu, Hao Fang, Yuqi Tan, Bin Chen, Shuhan Qi, Jiawei Li

**Abstract**: As 3D Gaussian Splatting (3DGS) emerges as a breakthrough in scene representation and novel view synthesis, its rapid adoption in safety-critical domains (e.g., autonomous systems, AR/VR) urgently demands scrutiny of potential security vulnerabilities. This paper presents the first systematic study of backdoor threats in 3DGS pipelines. We identify that adversaries may implant backdoor views to induce malicious scene confusion during inference, potentially leading to environmental misperception in autonomous navigation or spatial distortion in immersive environments. To uncover this risk, we propose GuassTrap, a novel poisoning attack method targeting 3DGS models. GuassTrap injects malicious views at specific attack viewpoints while preserving high-quality rendering in non-target views, ensuring minimal detectability and maximizing potential harm. Specifically, the proposed method consists of a three-stage pipeline (attack, stabilization, and normal training) to implant stealthy, viewpoint-consistent poisoned renderings in 3DGS, jointly optimizing attack efficacy and perceptual realism to expose security risks in 3D rendering. Extensive experiments on both synthetic and real-world datasets demonstrate that GuassTrap can effectively embed imperceptible yet harmful backdoor views while maintaining high-quality rendering in normal views, validating its robustness, adaptability, and practical applicability.



## **32. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

cs.LG

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2410.10572v3) [paper-pdf](http://arxiv.org/pdf/2410.10572v3)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.



## **33. A Survey on Adversarial Contention Resolution**

cs.DC

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2403.03876v4) [paper-pdf](http://arxiv.org/pdf/2403.03876v4)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We also highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.



## **34. Data Encryption Battlefield: A Deep Dive into the Dynamic Confrontations in Ransomware Attacks**

cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20681v1) [paper-pdf](http://arxiv.org/pdf/2504.20681v1)

**Authors**: Arash Mahboubi, Hamed Aboutorab, Seyit Camtepe, Hang Thanh Bui, Khanh Luong, Keyvan Ansari, Shenlu Wang, Bazara Barry

**Abstract**: In the rapidly evolving landscape of cybersecurity threats, ransomware represents a significant challenge. Attackers increasingly employ sophisticated encryption methods, such as entropy reduction through Base64 encoding, and partial or intermittent encryption to evade traditional detection methods. This study explores the dynamic battle between adversaries who continuously refine encryption strategies and defenders developing advanced countermeasures to protect vulnerable data. We investigate the application of online incremental machine learning algorithms designed to predict file encryption activities despite adversaries evolving obfuscation techniques. Our analysis utilizes an extensive dataset of 32.6 GB, comprising 11,928 files across multiple formats, including Microsoft Word documents (doc), PowerPoint presentations (ppt), Excel spreadsheets (xlsx), image formats (jpg, jpeg, png, tif, gif), PDFs (pdf), audio (mp3), and video (mp4) files. These files were encrypted by 75 distinct ransomware families, facilitating a robust empirical evaluation of machine learning classifiers effectiveness against diverse encryption tactics. Results highlight the Hoeffding Tree algorithms superior incremental learning capability, particularly effective in detecting traditional and AES-Base64 encryption methods employed to lower entropy. Conversely, the Random Forest classifier with warm-start functionality excels at identifying intermittent encryption methods, demonstrating the necessity of tailored machine learning solutions to counter sophisticated ransomware strategies.



## **35. Learning and Generalization with Mixture Data**

stat.ML

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20651v1) [paper-pdf](http://arxiv.org/pdf/2504.20651v1)

**Authors**: Harsh Vardhan, Avishek Ghosh, Arya Mazumdar

**Abstract**: In many, if not most, machine learning applications the training data is naturally heterogeneous (e.g. federated learning, adversarial attacks and domain adaptation in neural net training). Data heterogeneity is identified as one of the major challenges in modern day large-scale learning. A classical way to represent heterogeneous data is via a mixture model. In this paper, we study generalization performance and statistical rates when data is sampled from a mixture distribution. We first characterize the heterogeneity of the mixture in terms of the pairwise total variation distance of the sub-population distributions. Thereafter, as a central theme of this paper, we characterize the range where the mixture may be treated as a single (homogeneous) distribution for learning. In particular, we study the generalization performance under the classical PAC framework and the statistical error rates for parametric (linear regression, mixture of hyperplanes) as well as non-parametric (Lipschitz, convex and H\"older-smooth) regression problems. In order to do this, we obtain Rademacher complexity and (local) Gaussian complexity bounds with mixture data, and apply them to get the generalization and convergence rates respectively. We observe that as the (regression) function classes get more complex, the requirement on the pairwise total variation distance gets stringent, which matches our intuition. We also do a finer analysis for the case of mixed linear regression and provide a tight bound on the generalization error in terms of heterogeneity.



## **36. WILD: a new in-the-Wild Image Linkage Dataset for synthetic image attribution**

cs.MM

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.19595v2) [paper-pdf](http://arxiv.org/pdf/2504.19595v2)

**Authors**: Pietro Bongini, Sara Mandelli, Andrea Montibeller, Mirko Casu, Orazio Pontorno, Claudio Vittorio Ragaglia, Luca Zanchetta, Mattia Aquilina, Taiba Majid Wani, Luca Guarnera, Benedetta Tondi, Giulia Boato, Paolo Bestagini, Irene Amerini, Francesco De Natale, Sebastiano Battiato, Mauro Barni

**Abstract**: Synthetic image source attribution is an open challenge, with an increasing number of image generators being released yearly. The complexity and the sheer number of available generative techniques, as well as the scarcity of high-quality open source datasets of diverse nature for this task, make training and benchmarking synthetic image source attribution models very challenging. WILD is a new in-the-Wild Image Linkage Dataset designed to provide a powerful training and benchmarking tool for synthetic image attribution models. The dataset is built out of a closed set of 10 popular commercial generators, which constitutes the training base of attribution models, and an open set of 10 additional generators, simulating a real-world in-the-wild scenario. Each generator is represented by 1,000 images, for a total of 10,000 images in the closed set and 10,000 images in the open set. Half of the images are post-processed with a wide range of operators. WILD allows benchmarking attribution models in a wide range of tasks, including closed and open set identification and verification, and robust attribution with respect to post-processing and adversarial attacks. Models trained on WILD are expected to benefit from the challenging scenario represented by the dataset itself. Moreover, an assessment of seven baseline methodologies on closed and open set attribution is presented, including robustness tests with respect to post-processing.



## **37. NeuRel-Attack: Neuron Relearning for Safety Disalignment in Large Language Models**

cs.LG

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.21053v1) [paper-pdf](http://arxiv.org/pdf/2504.21053v1)

**Authors**: Yi Zhou, Wenpeng Xing, Dezhang Kong, Changting Lin, Meng Han

**Abstract**: Safety alignment in large language models (LLMs) is achieved through fine-tuning mechanisms that regulate neuron activations to suppress harmful content. In this work, we propose a novel approach to induce disalignment by identifying and modifying the neurons responsible for safety constraints. Our method consists of three key steps: Neuron Activation Analysis, where we examine activation patterns in response to harmful and harmless prompts to detect neurons that are critical for distinguishing between harmful and harmless inputs; Similarity-Based Neuron Identification, which systematically locates the neurons responsible for safe alignment; and Neuron Relearning for Safety Removal, where we fine-tune these selected neurons to restore the model's ability to generate previously restricted responses. Experimental results demonstrate that our method effectively removes safety constraints with minimal fine-tuning, highlighting a critical vulnerability in current alignment techniques. Our findings underscore the need for robust defenses against adversarial fine-tuning attacks on LLMs.



## **38. Enhancing Leakage Attacks on Searchable Symmetric Encryption Using LLM-Based Synthetic Data Generation**

cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20414v1) [paper-pdf](http://arxiv.org/pdf/2504.20414v1)

**Authors**: Joshua Chiu, Partha Protim Paul, Zahin Wahab

**Abstract**: Searchable Symmetric Encryption (SSE) enables efficient search capabilities over encrypted data, allowing users to maintain privacy while utilizing cloud storage. However, SSE schemes are vulnerable to leakage attacks that exploit access patterns, search frequency, and volume information. Existing studies frequently assume that adversaries possess a substantial fraction of the encrypted dataset to mount effective inference attacks, implying there is a database leakage of such documents, thus, an assumption that may not hold in real-world scenarios. In this work, we investigate the feasibility of enhancing leakage attacks under a more realistic threat model in which adversaries have access to minimal leaked data. We propose a novel approach that leverages large language models (LLMs), specifically GPT-4 variants, to generate synthetic documents that statistically and semantically resemble the real-world dataset of Enron emails. Using the email corpus as a case study, we evaluate the effectiveness of synthetic data generated via random sampling and hierarchical clustering methods on the performance of the SAP (Search Access Pattern) keyword inference attack restricted to token volumes only. Our results demonstrate that, while the choice of LLM has limited effect, increasing dataset size and employing clustering-based generation significantly improve attack accuracy, achieving comparable performance to attacks using larger amounts of real data. We highlight the growing relevance of LLMs in adversarial contexts.



## **39. Inception: Jailbreak the Memory Mechanism of Text-to-Image Generation Systems**

cs.CV

17 pages, 8 figures

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20376v1) [paper-pdf](http://arxiv.org/pdf/2504.20376v1)

**Authors**: Shiqian Zhao, Jiayang Liu, Yiming Li, Runyi Hu, Xiaojun Jia, Wenshu Fan, Xinfeng Li, Jie Zhang, Wei Dong, Tianwei Zhang, Luu Anh Tuan

**Abstract**: Currently, the memory mechanism has been widely and successfully exploited in online text-to-image (T2I) generation systems ($e.g.$, DALL$\cdot$E 3) for alleviating the growing tokenization burden and capturing key information in multi-turn interactions. Despite its practicality, its security analyses have fallen far behind. In this paper, we reveal that this mechanism exacerbates the risk of jailbreak attacks. Different from previous attacks that fuse the unsafe target prompt into one ultimate adversarial prompt, which can be easily detected or may generate non-unsafe images due to under- or over-optimization, we propose Inception, the first multi-turn jailbreak attack against the memory mechanism in real-world text-to-image generation systems. Inception embeds the malice at the inception of the chat session turn by turn, leveraging the mechanism that T2I generation systems retrieve key information in their memory. Specifically, Inception mainly consists of two modules. It first segments the unsafe prompt into chunks, which are subsequently fed to the system in multiple turns, serving as pseudo-gradients for directive optimization. Specifically, we develop a series of segmentation policies that ensure the images generated are semantically consistent with the target prompt. Secondly, after segmentation, to overcome the challenge of the inseparability of minimum unsafe words, we propose recursion, a strategy that makes minimum unsafe words subdivisible. Collectively, segmentation and recursion ensure that all the request prompts are benign but can lead to malicious outcomes. We conduct experiments on the real-world text-to-image generation system ($i.e.$, DALL$\cdot$E 3) to validate the effectiveness of Inception. The results indicate that Inception surpasses the state-of-the-art by a 14\% margin in attack success rate.



## **40. A Cryptographic Perspective on Mitigation vs. Detection in Machine Learning**

cs.LG

29 pages

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.20310v1) [paper-pdf](http://arxiv.org/pdf/2504.20310v1)

**Authors**: Greg Gluch, Shafi Goldwasser

**Abstract**: In this paper, we initiate a cryptographically inspired theoretical study of detection versus mitigation of adversarial inputs produced by attackers of Machine Learning algorithms during inference time.   We formally define defense by detection (DbD) and defense by mitigation (DbM). Our definitions come in the form of a 3-round protocol between two resource-bounded parties: a trainer/defender and an attacker. The attacker aims to produce inference-time inputs that fool the training algorithm. We define correctness, completeness, and soundness properties to capture successful defense at inference time while not degrading (too much) the performance of the algorithm on inputs from the training distribution.   We first show that achieving DbD and achieving DbM are equivalent for ML classification tasks. Surprisingly, this is not the case for ML generative learning tasks, where there are many possible correct outputs that can be generated for each input. We show a separation between DbD and DbM by exhibiting a generative learning task for which is possible to defend by mitigation but is provably impossible to defend by detection under the assumption that the Identity-Based Fully Homomorphic Encryption (IB-FHE), publicly-verifiable zero-knowledge Succinct Non-Interactive Arguments of Knowledge (zk-SNARK) and Strongly Unforgeable Signatures exist. The mitigation phase uses significantly fewer samples than the initial training algorithm.



## **41. The Dark Side of Digital Twins: Adversarial Attacks on AI-Driven Water Forecasting**

cs.LG

7 Pages, 7 Figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.20295v1) [paper-pdf](http://arxiv.org/pdf/2504.20295v1)

**Authors**: Mohammadhossein Homaei, Victor Gonzalez Morales, Oscar Mogollon-Gutierrez, Andres Caro

**Abstract**: Digital twins (DTs) are improving water distribution systems by using real-time data, analytics, and prediction models to optimize operations. This paper presents a DT platform designed for a Spanish water supply network, utilizing Long Short-Term Memory (LSTM) networks to predict water consumption. However, machine learning models are vulnerable to adversarial attacks, such as the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). These attacks manipulate critical model parameters, injecting subtle distortions that degrade forecasting accuracy. To further exploit these vulnerabilities, we introduce a Learning Automata (LA) and Random LA-based approach that dynamically adjusts perturbations, making adversarial attacks more difficult to detect. Experimental results show that this approach significantly impacts prediction reliability, causing the Mean Absolute Percentage Error (MAPE) to rise from 26% to over 35%. Moreover, adaptive attack strategies amplify this effect, highlighting cybersecurity risks in AI-driven DTs. These findings emphasize the urgent need for robust defenses, including adversarial training, anomaly detection, and secure data pipelines.



## **42. A Case Study on the Use of Representativeness Bias as a Defense Against Adversarial Cyber Threats**

cs.CR

To appear in the 4th Workshop on Active Defense and Deception (ADnD),  co-located with the 10th IEEE European Symposium on Security and Privacy  (EuroS&P 2025)

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.20245v1) [paper-pdf](http://arxiv.org/pdf/2504.20245v1)

**Authors**: Briland Hitaj, Grit Denker, Laura Tinnel, Michael McAnally, Bruce DeBruhl, Nathan Bunting, Alex Fafard, Daniel Aaron, Richard D. Roberts, Joshua Lawson, Greg McCain, Dylan Starink

**Abstract**: Cyberspace is an ever-evolving battleground involving adversaries seeking to circumvent existing safeguards and defenders aiming to stay one step ahead by predicting and mitigating the next threat. Existing mitigation strategies have focused primarily on solutions that consider software or hardware aspects, often ignoring the human factor. This paper takes a first step towards psychology-informed, active defense strategies, where we target biases that human beings are susceptible to under conditions of uncertainty.   Using capture-the-flag events, we create realistic challenges that tap into a particular cognitive bias: representativeness. This study finds that this bias can be triggered to thwart hacking attempts and divert hackers into non-vulnerable attack paths. Participants were exposed to two different challenges designed to exploit representativeness biases. One of the representativeness challenges significantly thwarted attackers away from vulnerable attack vectors and onto non-vulnerable paths, signifying an effective bias-based defense mechanism. This work paves the way towards cyber defense strategies that leverage additional human biases to thwart future, sophisticated adversarial attacks.



## **43. DROP: Poison Dilution via Knowledge Distillation for Federated Learning**

cs.LG

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2502.07011v2) [paper-pdf](http://arxiv.org/pdf/2502.07011v2)

**Authors**: Georgios Syros, Anshuman Suri, Farinaz Koushanfar, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Federated Learning is vulnerable to adversarial manipulation, where malicious clients can inject poisoned updates to influence the global model's behavior. While existing defense mechanisms have made notable progress, they fail to protect against adversaries that aim to induce targeted backdoors under different learning and attack configurations. To address this limitation, we introduce DROP (Distillation-based Reduction Of Poisoning), a novel defense mechanism that combines clustering and activity-tracking techniques with extraction of benign behavior from clients via knowledge distillation to tackle stealthy adversaries that manipulate low data poisoning rates and diverse malicious client ratios within the federation. Through extensive experimentation, our approach demonstrates superior robustness compared to existing defenses across a wide range of learning configurations. Finally, we evaluate existing defenses and our method under the challenging setting of non-IID client data distribution and highlight the challenges of designing a resilient FL defense in this setting.



## **44. AGATE: Stealthy Black-box Watermarking for Multimodal Model Copyright Protection**

cs.CR

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.21044v1) [paper-pdf](http://arxiv.org/pdf/2504.21044v1)

**Authors**: Jianbo Gao, Keke Gai, Jing Yu, Liehuang Zhu, Qi Wu

**Abstract**: Recent advancement in large-scale Artificial Intelligence (AI) models offering multimodal services have become foundational in AI systems, making them prime targets for model theft. Existing methods select Out-of-Distribution (OoD) data as backdoor watermarks and retrain the original model for copyright protection. However, existing methods are susceptible to malicious detection and forgery by adversaries, resulting in watermark evasion. In this work, we propose Model-\underline{ag}nostic Black-box Backdoor W\underline{ate}rmarking Framework (AGATE) to address stealthiness and robustness challenges in multimodal model copyright protection. Specifically, we propose an adversarial trigger generation method to generate stealthy adversarial triggers from ordinary dataset, providing visual fidelity while inducing semantic shifts. To alleviate the issue of anomaly detection among model outputs, we propose a post-transform module to correct the model output by narrowing the distance between adversarial trigger image embedding and text embedding. Subsequently, a two-phase watermark verification is proposed to judge whether the current model infringes by comparing the two results with and without the transform module. Consequently, we consistently outperform state-of-the-art methods across five datasets in the downstream tasks of multimodal image-text retrieval and image classification. Additionally, we validated the robustness of AGATE under two adversarial attack scenarios.



## **45. Evaluate-and-Purify: Fortifying Code Language Models Against Adversarial Attacks Using LLM-as-a-Judge**

cs.SE

25 pages, 6 figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19730v1) [paper-pdf](http://arxiv.org/pdf/2504.19730v1)

**Authors**: Wenhan Mu, Ling Xu, Shuren Pei, Le Mi, Huichi Zhou

**Abstract**: The widespread adoption of code language models in software engineering tasks has exposed vulnerabilities to adversarial attacks, especially the identifier substitution attacks. Although existing identifier substitution attackers demonstrate high success rates, they often produce adversarial examples with unnatural code patterns. In this paper, we systematically assess the quality of adversarial examples using LLM-as-a-Judge. Our analysis reveals that over 80% of adversarial examples generated by state-of-the-art identifier substitution attackers (e.g., ALERT) are actually detectable. Based on this insight, we propose EP-Shield, a unified framework for evaluating and purifying identifier substitution attacks via naturalness-aware reasoning. Specifically, we first evaluate the naturalness of code and identify the perturbed adversarial code, then purify it so that the victim model can restore correct prediction. Extensive experiments demonstrate the superiority of EP-Shield over adversarial fine-tuning (up to 83.36% improvement) and its lightweight design 7B parameters) with GPT-4-level performance.



## **46. Fooling the Decoder: An Adversarial Attack on Quantum Error Correction**

quant-ph

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19651v1) [paper-pdf](http://arxiv.org/pdf/2504.19651v1)

**Authors**: Jerome Lenssen, Alexandru Paler

**Abstract**: Neural network decoders are becoming essential for achieving fault-tolerant quantum computations. However, their internal mechanisms are poorly understood, hindering our ability to ensure their reliability and security against adversarial attacks. Leading machine learning decoders utilize recurrent and transformer models (e.g., AlphaQubit), with reinforcement learning (RL) playing a key role in training advanced transformer models (e.g., DeepSeek R1). In this work, we target a basic RL surface code decoder (DeepQ) to create the first adversarial attack on quantum error correction. By applying state-of-the-art white-box methods, we uncover vulnerabilities in this decoder, demonstrating an attack that reduces the logical qubit lifetime in memory experiments by up to five orders of magnitude. We validate that this attack exploits a genuine weakness, as the decoder exhibits robustness against noise fluctuations, is largely unaffected by substituting the referee decoder, responsible for episode termination, with an MWPM decoder, and demonstrates fault tolerance at checkable code distances. This attack highlights the susceptibility of machine learning-based QEC and underscores the importance of further research into robust QEC methods.



## **47. Prefill-Based Jailbreak: A Novel Approach of Bypassing LLM Safety Boundary**

cs.CR

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.21038v1) [paper-pdf](http://arxiv.org/pdf/2504.21038v1)

**Authors**: Yakai Li, Jiekang Hu, Weiduan Sang, Luping Ma, Jing Xie, Weijuan Zhang, Aimin Yu, Shijie Zhao, Qingjia Huang, Qihang Zhou

**Abstract**: Large Language Models (LLMs) are designed to generate helpful and safe content. However, adversarial attacks, commonly referred to as jailbreak, can bypass their safety protocols, prompting LLMs to generate harmful content or reveal sensitive data. Consequently, investigating jailbreak methodologies is crucial for exposing systemic vulnerabilities within LLMs, ultimately guiding the continuous implementation of security enhancements by developers. In this paper, we introduce a novel jailbreak attack method that leverages the prefilling feature of LLMs, a feature designed to enhance model output constraints. Unlike traditional jailbreak methods, the proposed attack circumvents LLMs' safety mechanisms by directly manipulating the probability distribution of subsequent tokens, thereby exerting control over the model's output. We propose two attack variants: Static Prefilling (SP), which employs a universal prefill text, and Optimized Prefilling (OP), which iteratively optimizes the prefill text to maximize the attack success rate. Experiments on six state-of-the-art LLMs using the AdvBench benchmark validate the effectiveness of our method and demonstrate its capability to substantially enhance attack success rates when combined with existing jailbreak approaches. The OP method achieved attack success rates of up to 99.82% on certain models, significantly outperforming baseline methods. This work introduces a new jailbreak attack method in LLMs, emphasizing the need for robust content validation mechanisms to mitigate the adversarial exploitation of prefilling features. All code and data used in this paper are publicly available.



## **48. FCGHunter: Towards Evaluating Robustness of Graph-Based Android Malware Detection**

cs.CR

14 pages, 5 figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19456v1) [paper-pdf](http://arxiv.org/pdf/2504.19456v1)

**Authors**: Shiwen Song, Xiaofei Xie, Ruitao Feng, Qi Guo, Sen Chen

**Abstract**: Graph-based detection methods leveraging Function Call Graphs (FCGs) have shown promise for Android malware detection (AMD) due to their semantic insights. However, the deployment of malware detectors in dynamic and hostile environments raises significant concerns about their robustness. While recent approaches evaluate the robustness of FCG-based detectors using adversarial attacks, their effectiveness is constrained by the vast perturbation space, particularly across diverse models and features.   To address these challenges, we introduce FCGHunter, a novel robustness testing framework for FCG-based AMD systems. Specifically, FCGHunter employs innovative techniques to enhance exploration and exploitation within this huge search space. Initially, it identifies critical areas within the FCG related to malware behaviors to narrow down the perturbation space. We then develop a dependency-aware crossover and mutation method to enhance the validity and diversity of perturbations, generating diverse FCGs. Furthermore, FCGHunter leverages multi-objective feedback to select perturbed FCGs, significantly improving the search process with interpretation-based feature change feedback.   Extensive evaluations across 40 scenarios demonstrate that FCGHunter achieves an average attack success rate of 87.9%, significantly outperforming baselines by at least 44.7%. Notably, FCGHunter achieves a 100% success rate on robust models (e.g., AdaBoost with MalScan), where baselines achieve only 11% or are inapplicable.



## **49. Mitigating Evasion Attacks in Federated Learning-Based Signal Classifiers**

eess.SP

Accepted for publication in IEEE Transactions on Network Science and  Engineering. arXiv admin note: substantial text overlap with arXiv:2301.08866

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2306.04872v2) [paper-pdf](http://arxiv.org/pdf/2306.04872v2)

**Authors**: Su Wang, Rajeev Sahay, Adam Piaseczny, Christopher G. Brinton

**Abstract**: Recent interest in leveraging federated learning (FL) for radio signal classification (SC) tasks has shown promise but FL-based SC remains susceptible to model poisoning adversarial attacks. These adversarial attacks mislead the ML model training process, damaging ML models across the network and leading to lower SC performance. In this work, we seek to mitigate model poisoning adversarial attacks on FL-based SC by proposing the Underlying Server Defense of Federated Learning (USD-FL). Unlike existing server-driven defenses, USD-FL does not rely on perfect network information, i.e., knowing the quantity of adversaries, the adversarial attack architecture, or the start time of the adversarial attacks. Our proposed USD-FL methodology consists of deriving logits for devices' ML models on a reserve dataset, comparing pair-wise logits via 1-Wasserstein distance and then determining a time-varying threshold for adversarial detection. As a result, USD-FL effectively mitigates model poisoning attacks introduced in the FL network. Specifically, when baseline server-driven defenses do have perfect network information, USD-FL outperforms them by (i) improving final ML classification accuracies by at least 6%, (ii) reducing false positive adversary detection rates by at least 10%, and (iii) decreasing the total number of misclassified signals by over 8%. Moreover, when baseline defenses do not have perfect network information, we show that USD-FL achieves accuracies of approximately 74.1% and 62.5% in i.i.d. and non-i.i.d. settings, outperforming existing server-driven baselines, which achieve 52.1% and 39.2% in i.i.d. and non-i.i.d. settings, respectively.



## **50. Forging and Removing Latent-Noise Diffusion Watermarks Using a Single Image**

cs.CV

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2504.20111v1) [paper-pdf](http://arxiv.org/pdf/2504.20111v1)

**Authors**: Anubhav Jain, Yuya Kobayashi, Naoki Murata, Yuhta Takida, Takashi Shibuya, Yuki Mitsufuji, Niv Cohen, Nasir Memon, Julian Togelius

**Abstract**: Watermarking techniques are vital for protecting intellectual property and preventing fraudulent use of media. Most previous watermarking schemes designed for diffusion models embed a secret key in the initial noise. The resulting pattern is often considered hard to remove and forge into unrelated images. In this paper, we propose a black-box adversarial attack without presuming access to the diffusion model weights. Our attack uses only a single watermarked example and is based on a simple observation: there is a many-to-one mapping between images and initial noises. There are regions in the clean image latent space pertaining to each watermark that get mapped to the same initial noise when inverted. Based on this intuition, we propose an adversarial attack to forge the watermark by introducing perturbations to the images such that we can enter the region of watermarked images. We show that we can also apply a similar approach for watermark removal by learning perturbations to exit this region. We report results on multiple watermarking schemes (Tree-Ring, RingID, WIND, and Gaussian Shading) across two diffusion models (SDv1.4 and SDv2.0). Our results demonstrate the effectiveness of the attack and expose vulnerabilities in the watermarking methods, motivating future research on improving them.



