# Latest Adversarial Attack Papers
**update at 2025-05-16 16:51:10**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. The Ephemeral Threat: Assessing the Security of Algorithmic Trading Systems powered by Deep Learning**

cs.CR

To appear at ACM CODASPY 2025. 12 pages

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.10430v1) [paper-pdf](http://arxiv.org/pdf/2505.10430v1)

**Authors**: Advije Rizvani, Giovanni Apruzzese, Pavel Laskov

**Abstract**: We study the security of stock price forecasting using Deep Learning (DL) in computational finance. Despite abundant prior research on the vulnerability of DL to adversarial perturbations, such work has hitherto hardly addressed practical adversarial threat models in the context of DL-powered algorithmic trading systems (ATS). Specifically, we investigate the vulnerability of ATS to adversarial perturbations launched by a realistically constrained attacker. We first show that existing literature has paid limited attention to DL security in the financial domain, which is naturally attractive for adversaries. Then, we formalize the concept of ephemeral perturbations (EP), which can be used to stage a novel type of attack tailored for DL-based ATS. Finally, we carry out an end-to-end evaluation of our EP against a profitable ATS. Our results reveal that the introduction of small changes to the input stock prices not only (i) induces the DL model to behave incorrectly but also (ii) leads the whole ATS to make suboptimal buy/sell decisions, resulting in a worse financial performance of the targeted ATS.



## **2. A Unified and Scalable Membership Inference Method for Visual Self-supervised Encoder via Part-aware Capability**

cs.CV

An extension of our ACM CCS2024 conference paper (arXiv:2404.02462).  We show the impacts of scaling from both data and model aspects on membership  inference for self-supervised visual encoders

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.10351v1) [paper-pdf](http://arxiv.org/pdf/2505.10351v1)

**Authors**: Jie Zhu, Jirong Zha, Ding Li, Leye Wang

**Abstract**: Self-supervised learning shows promise in harnessing extensive unlabeled data, but it also confronts significant privacy concerns, especially in vision. In this paper, we perform membership inference on visual self-supervised models in a more realistic setting: self-supervised training method and details are unknown for an adversary when attacking as he usually faces a black-box system in practice. In this setting, considering that self-supervised model could be trained by completely different self-supervised paradigms, e.g., masked image modeling and contrastive learning, with complex training details, we propose a unified membership inference method called PartCrop. It is motivated by the shared part-aware capability among models and stronger part response on the training data. Specifically, PartCrop crops parts of objects in an image to query responses within the image in representation space. We conduct extensive attacks on self-supervised models with different training protocols and structures using three widely used image datasets. The results verify the effectiveness and generalization of PartCrop. Moreover, to defend against PartCrop, we evaluate two common approaches, i.e., early stop and differential privacy, and propose a tailored method called shrinking crop scale range. The defense experiments indicate that all of them are effective. Finally, besides prototype testing on toy visual encoders and small-scale image datasets, we quantitatively study the impacts of scaling from both data and model aspects in a realistic scenario and propose a scalable PartCrop-v2 by introducing two structural improvements to PartCrop. Our code is at https://github.com/JiePKU/PartCrop.



## **3. DAPPER: A Performance-Attack-Resilient Tracker for RowHammer Defense**

cs.CR

The initial version of this paper was submitted to MICRO 2024 on  April 18, 2024. The final version was presented at HPCA 2025  (https://hpca-conf.org/2025) and is 16 pages long, including references

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2501.18857v2) [paper-pdf](http://arxiv.org/pdf/2501.18857v2)

**Authors**: Jeonghyun Woo, Prashant J. Nair

**Abstract**: RowHammer vulnerabilities pose a significant threat to modern DRAM-based systems, where rapid activation of DRAM rows can induce bit-flips in neighboring rows. To mitigate this, state-of-the-art host-side RowHammer mitigations typically rely on shared counters or tracking structures. While these optimizations benefit benign applications, they are vulnerable to Performance Attacks (Perf-Attacks), where adversaries exploit shared structures to reduce DRAM bandwidth for co-running benign applications by increasing DRAM accesses for RowHammer counters or triggering repetitive refreshes required for the early reset of structures, significantly degrading performance.   In this paper, we propose secure hashing mechanisms to thwart adversarial attempts to capture the mapping of shared structures. We propose DAPPER, a novel low-cost tracker resilient to Perf-Attacks even at ultra-low RowHammer thresholds. We first present a secure hashing template in the form of DAPPER-S. We then develop DAPPER-H, an enhanced version of DAPPER-S, incorporating double-hashing, novel reset strategies, and mitigative refresh techniques. Our security analysis demonstrates the effectiveness of DAPPER-H against both RowHammer and Perf-Attacks. Experiments with 57 workloads from SPEC2006, SPEC2017, TPC, Hadoop, MediaBench, and YCSB show that, even at an ultra-low RowHammer threshold of 500, DAPPER-H incurs only a 0.9% slowdown in the presence of Perf-Attacks while using only 96KB of SRAM per 32GB of DRAM memory.



## **4. Scaling Laws for Black box Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2411.16782v2) [paper-pdf](http://arxiv.org/pdf/2411.16782v2)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.



## **5. Sybil-based Virtual Data Poisoning Attacks in Federated Learning**

cs.CR

7 pages, 6 figures, accepted by IEEE Codit 2025

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.09983v1) [paper-pdf](http://arxiv.org/pdf/2505.09983v1)

**Authors**: Changxun Zhu, Qilong Wu, Lingjuan Lyu, Shibei Xue

**Abstract**: Federated learning is vulnerable to poisoning attacks by malicious adversaries. Existing methods often involve high costs to achieve effective attacks. To address this challenge, we propose a sybil-based virtual data poisoning attack, where a malicious client generates sybil nodes to amplify the poisoning model's impact. To reduce neural network computational complexity, we develop a virtual data generation method based on gradient matching. We also design three schemes for target model acquisition, applicable to online local, online global, and offline scenarios. In simulation, our method outperforms other attack algorithms since our method can obtain a global target model under non-independent uniformly distributed data.



## **6. Adversarial Attack on Large Language Models using Exponentiated Gradient Descent**

cs.LG

Accepted to International Joint Conference on Neural Networks (IJCNN)  2025

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09820v1) [paper-pdf](http://arxiv.org/pdf/2505.09820v1)

**Authors**: Sajib Biswas, Mao Nishino, Samuel Jacob Chacko, Xiuwen Liu

**Abstract**: As Large Language Models (LLMs) are widely used, understanding them systematically is key to improving their safety and realizing their full potential. Although many models are aligned using techniques such as reinforcement learning from human feedback (RLHF), they are still vulnerable to jailbreaking attacks. Some of the existing adversarial attack methods search for discrete tokens that may jailbreak a target model while others try to optimize the continuous space represented by the tokens of the model's vocabulary. While techniques based on the discrete space may prove to be inefficient, optimization of continuous token embeddings requires projections to produce discrete tokens, which might render them ineffective. To fully utilize the constraints and the structures of the space, we develop an intrinsic optimization technique using exponentiated gradient descent with the Bregman projection method to ensure that the optimized one-hot encoding always stays within the probability simplex. We prove the convergence of the technique and implement an efficient algorithm that is effective in jailbreaking several widely used LLMs. We demonstrate the efficacy of the proposed technique using five open-source LLMs on four openly available datasets. The results show that the technique achieves a higher success rate with great efficiency compared to three other state-of-the-art jailbreaking techniques. The source code for our implementation is available at: https://github.com/sbamit/Exponentiated-Gradient-Descent-LLM-Attack



## **7. Self-Consuming Generative Models with Adversarially Curated Data**

cs.LG

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09768v1) [paper-pdf](http://arxiv.org/pdf/2505.09768v1)

**Authors**: Xiukun Wei, Xueru Zhang

**Abstract**: Recent advances in generative models have made it increasingly difficult to distinguish real data from model-generated synthetic data. Using synthetic data for successive training of future model generations creates "self-consuming loops", which may lead to model collapse or training instability. Furthermore, synthetic data is often subject to human feedback and curated by users based on their preferences. Ferbach et al. (2024) recently showed that when data is curated according to user preferences, the self-consuming retraining loop drives the model to converge toward a distribution that optimizes those preferences. However, in practice, data curation is often noisy or adversarially manipulated. For example, competing platforms may recruit malicious users to adversarially curate data and disrupt rival models. In this paper, we study how generative models evolve under self-consuming retraining loops with noisy and adversarially curated data. We theoretically analyze the impact of such noisy data curation on generative models and identify conditions for the robustness of the retraining process. Building on this analysis, we design attack algorithms for competitive adversarial scenarios, where a platform with a limited budget employs malicious users to misalign a rival's model from actual user preferences. Experiments on both synthetic and real-world datasets demonstrate the effectiveness of the proposed algorithms.



## **8. Adversarial Suffix Filtering: a Defense Pipeline for LLMs**

cs.LG

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09602v1) [paper-pdf](http://arxiv.org/pdf/2505.09602v1)

**Authors**: David Khachaturov, Robert Mullins

**Abstract**: Large Language Models (LLMs) are increasingly embedded in autonomous systems and public-facing environments, yet they remain susceptible to jailbreak vulnerabilities that may undermine their security and trustworthiness. Adversarial suffixes are considered to be the current state-of-the-art jailbreak, consistently outperforming simpler methods and frequently succeeding even in black-box settings. Existing defenses rely on access to the internal architecture of models limiting diverse deployment, increase memory and computation footprints dramatically, or can be bypassed with simple prompt engineering methods. We introduce $\textbf{Adversarial Suffix Filtering}$ (ASF), a lightweight novel model-agnostic defensive pipeline designed to protect LLMs against adversarial suffix attacks. ASF functions as an input preprocessor and sanitizer that detects and filters adversarially crafted suffixes in prompts, effectively neutralizing malicious injections. We demonstrate that ASF provides comprehensive defense capabilities across both black-box and white-box attack settings, reducing the attack efficacy of state-of-the-art adversarial suffix generation methods to below 4%, while only minimally affecting the target model's capabilities in non-adversarial scenarios.



## **9. I Know What You Said: Unveiling Hardware Cache Side-Channels in Local Large Language Model Inference**

cs.CR

Submitted for review in January 22, 2025

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.06738v2) [paper-pdf](http://arxiv.org/pdf/2505.06738v2)

**Authors**: Zibo Gao, Junjie Hu, Feng Guo, Yixin Zhang, Yinglong Han, Siyuan Liu, Haiyang Li, Zhiqiang Lv

**Abstract**: Large Language Models (LLMs) that can be deployed locally have recently gained popularity for privacy-sensitive tasks, with companies such as Meta, Google, and Intel playing significant roles in their development. However, the security of local LLMs through the lens of hardware cache side-channels remains unexplored. In this paper, we unveil novel side-channel vulnerabilities in local LLM inference: token value and token position leakage, which can expose both the victim's input and output text, thereby compromising user privacy. Specifically, we found that adversaries can infer the token values from the cache access patterns of the token embedding operation, and deduce the token positions from the timing of autoregressive decoding phases. To demonstrate the potential of these leaks, we design a novel eavesdropping attack framework targeting both open-source and proprietary LLM inference systems. The attack framework does not directly interact with the victim's LLM and can be executed without privilege.   We evaluate the attack on a range of practical local LLM deployments (e.g., Llama, Falcon, and Gemma), and the results show that our attack achieves promising accuracy. The restored output and input text have an average edit distance of 5.2% and 17.3% to the ground truth, respectively. Furthermore, the reconstructed texts achieve average cosine similarity scores of 98.7% (input) and 98.0% (output).



## **10. Evaluating the Robustness of Adversarial Defenses in Malware Detection Systems**

cs.CR

Submitted to IEEE Transactions on Information Forensics and Security  (T-IFS), 13 pages, 4 figures

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09342v1) [paper-pdf](http://arxiv.org/pdf/2505.09342v1)

**Authors**: Mostafa Jafari, Alireza Shameli-Sendi

**Abstract**: Machine learning is a key tool for Android malware detection, effectively identifying malicious patterns in apps. However, ML-based detectors are vulnerable to evasion attacks, where small, crafted changes bypass detection. Despite progress in adversarial defenses, the lack of comprehensive evaluation frameworks in binary-constrained domains limits understanding of their robustness. We introduce two key contributions. First, Prioritized Binary Rounding, a technique to convert continuous perturbations into binary feature spaces while preserving high attack success and low perturbation size. Second, the sigma-binary attack, a novel adversarial method for binary domains, designed to achieve attack goals with minimal feature changes. Experiments on the Malscan dataset show that sigma-binary outperforms existing attacks and exposes key vulnerabilities in state-of-the-art defenses. Defenses equipped with adversary detectors, such as KDE, DLA, DNN+, and ICNN, exhibit significant brittleness, with attack success rates exceeding 90% using fewer than 10 feature modifications and reaching 100% with just 20. Adversarially trained defenses, including AT-rFGSM-k, AT-MaxMA, improves robustness under small budgets but remains vulnerable to unrestricted perturbations, with attack success rates of 99.45% and 96.62%, respectively. Although PAD-SMA demonstrates strong robustness against state-of-the-art gradient-based adversarial attacks by maintaining an attack success rate below 16.55%, the sigma-binary attack significantly outperforms these methods, achieving a 94.56% success rate under unrestricted perturbations. These findings highlight the critical need for precise method like sigma-binary to expose hidden vulnerabilities in existing defenses and support the development of more resilient malware detection systems.



## **11. Beyond Optimal Fault Tolerance**

cs.DC

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2501.06044v5) [paper-pdf](http://arxiv.org/pdf/2501.06044v5)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal "recoverable fault-tolerance" achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic "recovery procedure" that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.



## **12. Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction**

cs.CL

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.05084v2) [paper-pdf](http://arxiv.org/pdf/2505.05084v2)

**Authors**: Xiaowei Zhu, Yubing Ren, Yanan Cao, Xixun Lin, Fang Fang, Yangxi Li

**Abstract**: The rapid advancement of large language models has raised significant concerns regarding their potential misuse by malicious actors. As a result, developing effective detectors to mitigate these risks has become a critical priority. However, most existing detection methods focus excessively on detection accuracy, often neglecting the societal risks posed by high false positive rates (FPRs). This paper addresses this issue by leveraging Conformal Prediction (CP), which effectively constrains the upper bound of FPRs. While directly applying CP constrains FPRs, it also leads to a significant reduction in detection performance. To overcome this trade-off, this paper proposes a Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction (MCP), which both enforces the FPR constraint and improves detection performance. This paper also introduces RealDet, a high-quality dataset that spans a wide range of domains, ensuring realistic calibration and enabling superior detection performance when combined with MCP. Empirical evaluations demonstrate that MCP effectively constrains FPRs, significantly enhances detection performance, and increases robustness against adversarial attacks across multiple detectors and datasets.



## **13. BridgePure: Limited Protection Leakage Can Break Black-Box Data Protection**

cs.LG

29 pages,18 figures

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2412.21061v2) [paper-pdf](http://arxiv.org/pdf/2412.21061v2)

**Authors**: Yihan Wang, Yiwei Lu, Xiao-Shan Gao, Gautam Kamath, Yaoliang Yu

**Abstract**: Availability attacks, or unlearnable examples, are defensive techniques that allow data owners to modify their datasets in ways that prevent unauthorized machine learning models from learning effectively while maintaining the data's intended functionality. It has led to the release of popular black-box tools (e.g., APIs) for users to upload personal data and receive protected counterparts. In this work, we show that such black-box protections can be substantially compromised if a small set of unprotected in-distribution data is available. Specifically, we propose a novel threat model of protection leakage, where an adversary can (1) easily acquire (unprotected, protected) pairs by querying the black-box protections with a small unprotected dataset; and (2) train a diffusion bridge model to build a mapping between unprotected and protected data. This mapping, termed BridgePure, can effectively remove the protection from any previously unseen data within the same distribution. BridgePure demonstrates superior purification performance on classification and style mimicry tasks, exposing critical vulnerabilities in black-box data protection. We suggest that practitioners implement multi-level countermeasures to mitigate such risks.



## **14. SAFE-SiP: Secure Authentication Framework for System-in-Package Using Multi-party Computation**

cs.CR

Accepted for GLSVLSI 2025, New Orleans, LA, USA

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.09002v1) [paper-pdf](http://arxiv.org/pdf/2505.09002v1)

**Authors**: Ishraq Tashdid, Tasnuva Farheen, Sazadur Rahman

**Abstract**: The emergence of chiplet-based heterogeneous integration is transforming the semiconductor, AI, and high-performance computing industries by enabling modular designs and improved scalability. However, assembling chiplets from multiple vendors after fabrication introduces a complex supply chain that raises serious security concerns, including counterfeiting, overproduction, and unauthorized access. Current solutions often depend on dedicated security chiplets or changes to the timing flow, which assume a trusted SiP integrator. This assumption can expose chiplet signatures to other vendors and create new attack surfaces. This work addresses those vulnerabilities using Multi-party Computation (MPC), which enables zero-trust authentication without disclosing sensitive information to any party. We present SAFE-SiP, a scalable authentication framework that garbles chiplet signatures and uses MPC for verifying integrity, effectively blocking unauthorized access and adversarial inference. SAFE-SiP removes the need for a dedicated security chiplet and ensures secure authentication, even in untrusted integration scenarios. We evaluated SAFE-SiP on five RISC-V-based System-in-Package (SiP) designs. Experimental results show that SAFE-SiP incurs minimal power overhead, an average area overhead of only 3.05%, and maintains a computational complexity of 2^192, offering a highly efficient and scalable security solution.



## **15. Towards Adaptive Meta-Gradient Adversarial Examples for Visual Tracking**

cs.CV

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08999v1) [paper-pdf](http://arxiv.org/pdf/2505.08999v1)

**Authors**: Wei-Long Tian, Peng Gao, Xiao Liu, Long Xu, Hamido Fujita, Hanan Aljuai, Mao-Li Wang

**Abstract**: In recent years, visual tracking methods based on convolutional neural networks and Transformers have achieved remarkable performance and have been successfully applied in fields such as autonomous driving. However, the numerous security issues exposed by deep learning models have gradually affected the reliable application of visual tracking methods in real-world scenarios. Therefore, how to reveal the security vulnerabilities of existing visual trackers through effective adversarial attacks has become a critical problem that needs to be addressed. To this end, we propose an adaptive meta-gradient adversarial attack (AMGA) method for visual tracking. This method integrates multi-model ensembles and meta-learning strategies, combining momentum mechanisms and Gaussian smoothing, which can significantly enhance the transferability and attack effectiveness of adversarial examples. AMGA randomly selects models from a large model repository, constructs diverse tracking scenarios, and iteratively performs both white- and black-box adversarial attacks in each scenario, optimizing the gradient directions of each model. This paradigm minimizes the gap between white- and black-box adversarial attacks, thus achieving excellent attack performance in black-box scenarios. Extensive experimental results on large-scale datasets such as OTB2015, LaSOT, and GOT-10k demonstrate that AMGA significantly improves the attack performance, transferability, and deception of adversarial examples. Codes and data are available at https://github.com/pgao-lab/AMGA.



## **16. On the Impact of Uncertainty and Calibration on Likelihood-Ratio Membership Inference Attacks**

cs.IT

16 pages, 23 figures

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2402.10686v4) [paper-pdf](http://arxiv.org/pdf/2402.10686v4)

**Authors**: Meiyi Zhu, Caili Guo, Chunyan Feng, Osvaldo Simeone

**Abstract**: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in which an adaptive prediction set is produced as in conformal prediction. We derive bounds on the advantage of an MIA adversary with the aim of offering insights into the impact of uncertainty and calibration on the effectiveness of MIAs. Simulation results demonstrate that the derived analytical bounds predict well the effectiveness of MIAs.



## **17. DFA-CON: A Contrastive Learning Approach for Detecting Copyright Infringement in DeepFake Art**

cs.CV

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08552v1) [paper-pdf](http://arxiv.org/pdf/2505.08552v1)

**Authors**: Haroon Wahab, Hassan Ugail, Irfan Mehmood

**Abstract**: Recent proliferation of generative AI tools for visual content creation-particularly in the context of visual artworks-has raised serious concerns about copyright infringement and forgery. The large-scale datasets used to train these models often contain a mixture of copyrighted and non-copyrighted artworks. Given the tendency of generative models to memorize training patterns, they are susceptible to varying degrees of copyright violation. Building on the recently proposed DeepfakeArt Challenge benchmark, this work introduces DFA-CON, a contrastive learning framework designed to detect copyright-infringing or forged AI-generated art. DFA-CON learns a discriminative representation space, posing affinity among original artworks and their forged counterparts within a contrastive learning framework. The model is trained across multiple attack types, including inpainting, style transfer, adversarial perturbation, and cutmix. Evaluation results demonstrate robust detection performance across most attack types, outperforming recent pretrained foundation models. Code and model checkpoints will be released publicly upon acceptance.



## **18. Minimax rates of convergence for nonparametric regression under adversarial attacks**

math.ST

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2410.09402v2) [paper-pdf](http://arxiv.org/pdf/2410.09402v2)

**Authors**: Jingfu Peng, Yuhong Yang

**Abstract**: Recent research shows the susceptibility of machine learning models to adversarial attacks, wherein minor but maliciously chosen perturbations of the input can significantly degrade model performance. In this paper, we theoretically analyse the limits of robustness against such adversarial attacks in a nonparametric regression setting, by examining the minimax rates of convergence in an adversarial sup-norm. Our work reveals that the minimax rate under adversarial attacks in the input is the same as sum of two terms: one represents the minimax rate in the standard setting without adversarial attacks, and the other reflects the maximum deviation of the true regression function value within the target function class when subjected to the input perturbations. The optimal rates under the adversarial setup can be achieved by an adversarial plug-in procedure constructed from a minimax optimal estimator in the corresponding standard setting. Two specific examples are given to illustrate the established minimax results.



## **19. Quantum Support Vector Regression for Robust Anomaly Detection**

quant-ph

Submitted to IEEE International Conference on Quantum Computing and  Engineering (QCE) 2025

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.01012v2) [paper-pdf](http://arxiv.org/pdf/2505.01012v2)

**Authors**: Kilian Tscharke, Maximilian Wendlinger, Sebastian Issel, Pascal Debus

**Abstract**: Anomaly Detection (AD) is critical in data analysis, particularly within the domain of IT security. In recent years, Machine Learning (ML) algorithms have emerged as a powerful tool for AD in large-scale data. In this study, we explore the potential of quantum ML approaches, specifically quantum kernel methods, for the application to robust AD. We build upon previous work on Quantum Support Vector Regression (QSVR) for semisupervised AD by conducting a comprehensive benchmark on IBM quantum hardware using eleven datasets. Our results demonstrate that QSVR achieves strong classification performance and even outperforms the noiseless simulation on two of these datasets. Moreover, we investigate the influence of - in the NISQ-era inevitable - quantum noise on the performance of the QSVR. Our findings reveal that the model exhibits robustness to depolarizing, phase damping, phase flip, and bit flip noise, while amplitude damping and miscalibration noise prove to be more disruptive. Finally, we explore the domain of Quantum Adversarial Machine Learning and demonstrate that QSVR is highly vulnerable to adversarial attacks and that noise does not improve the adversarial robustness of the model.



## **20. SHAP-based Explanations are Sensitive to Feature Representation**

cs.LG

Accepted to ACM FAccT 2025

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08345v1) [paper-pdf](http://arxiv.org/pdf/2505.08345v1)

**Authors**: Hyunseung Hwang, Andrew Bell, Joao Fonseca, Venetia Pliatsika, Julia Stoyanovich, Steven Euijong Whang

**Abstract**: Local feature-based explanations are a key component of the XAI toolkit. These explanations compute feature importance values relative to an ``interpretable'' feature representation. In tabular data, feature values themselves are often considered interpretable. This paper examines the impact of data engineering choices on local feature-based explanations. We demonstrate that simple, common data engineering techniques, such as representing age with a histogram or encoding race in a specific way, can manipulate feature importance as determined by popular methods like SHAP. Notably, the sensitivity of explanations to feature representation can be exploited by adversaries to obscure issues like discrimination. While the intuition behind these results is straightforward, their systematic exploration has been lacking. Previous work has focused on adversarial attacks on feature-based explainers by biasing data or manipulating models. To the best of our knowledge, this is the first study demonstrating that explainers can be misled by standard, seemingly innocuous data engineering techniques.



## **21. Robustness Analysis against Adversarial Patch Attacks in Fully Unmanned Stores**

cs.CR

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08835v1) [paper-pdf](http://arxiv.org/pdf/2505.08835v1)

**Authors**: Hyunsik Na, Wonho Lee, Seungdeok Roh, Sohee Park, Daeseon Choi

**Abstract**: The advent of convenient and efficient fully unmanned stores equipped with artificial intelligence-based automated checkout systems marks a new era in retail. However, these systems have inherent artificial intelligence security vulnerabilities, which are exploited via adversarial patch attacks, particularly in physical environments. This study demonstrated that adversarial patches can severely disrupt object detection models used in unmanned stores, leading to issues such as theft, inventory discrepancies, and interference. We investigated three types of adversarial patch attacks -- Hiding, Creating, and Altering attacks -- and highlighted their effectiveness. We also introduce the novel color histogram similarity loss function by leveraging attacker knowledge of the color information of a target class object. Besides the traditional confusion-matrix-based attack success rate, we introduce a new bounding-boxes-based metric to analyze the practical impact of these attacks. Starting with attacks on object detection models trained on snack and fruit datasets in a digital environment, we evaluated the effectiveness of adversarial patches in a physical testbed that mimicked a real unmanned store with RGB cameras and realistic conditions. Furthermore, we assessed the robustness of these attacks in black-box scenarios, demonstrating that shadow attacks can enhance success rates of attacks even without direct access to model parameters. Our study underscores the necessity for robust defense strategies to protect unmanned stores from adversarial threats. Highlighting the limitations of the current defense mechanisms in real-time detection systems and discussing various proactive measures, we provide insights into improving the robustness of object detection models and fortifying unmanned retail environments against these attacks.



## **22. Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs**

cs.CR

7 Pages, 6 Figures

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.04806v2) [paper-pdf](http://arxiv.org/pdf/2505.04806v2)

**Authors**: Chetan Pathade

**Abstract**: Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security.



## **23. Removing Watermarks with Partial Regeneration using Semantic Information**

cs.CV

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08234v1) [paper-pdf](http://arxiv.org/pdf/2505.08234v1)

**Authors**: Krti Tallam, John Kevin Cava, Caleb Geniesse, N. Benjamin Erichson, Michael W. Mahoney

**Abstract**: As AI-generated imagery becomes ubiquitous, invisible watermarks have emerged as a primary line of defense for copyright and provenance. The newest watermarking schemes embed semantic signals - content-aware patterns that are designed to survive common image manipulations - yet their true robustness against adaptive adversaries remains under-explored. We expose a previously unreported vulnerability and introduce SemanticRegen, a three-stage, label-free attack that erases state-of-the-art semantic and invisible watermarks while leaving an image's apparent meaning intact. Our pipeline (i) uses a vision-language model to obtain fine-grained captions, (ii) extracts foreground masks with zero-shot segmentation, and (iii) inpaints only the background via an LLM-guided diffusion model, thereby preserving salient objects and style cues. Evaluated on 1,000 prompts across four watermarking systems - TreeRing, StegaStamp, StableSig, and DWT/DCT - SemanticRegen is the only method to defeat the semantic TreeRing watermark (p = 0.10 > 0.05) and reduces bit-accuracy below 0.75 for the remaining schemes, all while maintaining high perceptual quality (masked SSIM = 0.94 +/- 0.01). We further introduce masked SSIM (mSSIM) to quantify fidelity within foreground regions, showing that our attack achieves up to 12 percent higher mSSIM than prior diffusion-based attackers. These results highlight an urgent gap between current watermark defenses and the capabilities of adaptive, semantics-aware adversaries, underscoring the need for watermarking algorithms that are resilient to content-preserving regenerative attacks.



## **24. AI and Generative AI Transforming Disaster Management: A Survey of Damage Assessment and Response Techniques**

cs.CY

Accepted in IEEE Compsac 2025

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08202v1) [paper-pdf](http://arxiv.org/pdf/2505.08202v1)

**Authors**: Aman Raj, Lakshit Arora, Sanjay Surendranath Girija, Shashank Kapoor, Dipen Pradhan, Ankit Shetgaonkar

**Abstract**: Natural disasters, including earthquakes, wildfires and cyclones, bear a huge risk on human lives as well as infrastructure assets. An effective response to disaster depends on the ability to rapidly and efficiently assess the intensity of damage. Artificial Intelligence (AI) and Generative Artificial Intelligence (GenAI) presents a breakthrough solution, capable of combining knowledge from multiple types and sources of data, simulating realistic scenarios of disaster, and identifying emerging trends at a speed previously unimaginable. In this paper, we present a comprehensive review on the prospects of AI and GenAI in damage assessment for various natural disasters, highlighting both its strengths and limitations. We talk about its application to multimodal data such as text, image, video, and audio, and also cover major issues of data privacy, security, and ethical use of the technology during crises. The paper also recognizes the threat of Generative AI misuse, in the form of dissemination of misinformation and for adversarial attacks. Finally, we outline avenues of future research, emphasizing the need for secure, reliable, and ethical Generative AI systems for disaster management in general. We believe that this work represents the first comprehensive survey of Gen-AI techniques being used in the field of Disaster Assessment and Response.



## **25. FlippedRAG: Black-Box Opinion Manipulation Adversarial Attacks to Retrieval-Augmented Generation Models**

cs.IR

arXiv admin note: text overlap with arXiv:2407.13757

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2501.02968v3) [paper-pdf](http://arxiv.org/pdf/2501.02968v3)

**Authors**: Zhuo Chen, Jiawei Liu, Yuyang Gong, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu, Xiaofeng Wang

**Abstract**: Retrieval-Augmented Generation (RAG) enriches LLMs by dynamically retrieving external knowledge, reducing hallucinations and satisfying real-time information needs. While existing research mainly targets RAG's performance and efficiency, emerging studies highlight critical security concerns. Yet, current adversarial approaches remain limited, mostly addressing white-box scenarios or heuristic black-box attacks without fully investigating vulnerabilities in the retrieval phase. Additionally, prior works mainly focus on factoid QA tasks, their attacks lack complexity and can be easily corrected by advanced LLMs. In this paper, we investigate a more realistic and critical threat scenario: adversarial attacks intended for opinion manipulation against black-box RAG models, particularly on controversial topics. Specifically, we propose FlippedRAG, a transfer-based adversarial attack against black-box RAG systems. We first demonstrate that the underlying retriever of a black-box RAG system can be reverse-engineered, enabling us to train a surrogate retriever. Leveraging the surrogate retriever, we further craft target poisoning triggers, altering vary few documents to effectively manipulate both retrieval and subsequent generation. Extensive empirical results show that FlippedRAG substantially outperforms baseline methods, improving the average attack success rate by 16.7%. FlippedRAG achieves on average a 50% directional shift in the opinion polarity of RAG-generated responses, ultimately causing a notable 20% shift in user cognition. Furthermore, we evaluate the performance of several potential defensive measures, concluding that existing mitigation strategies remain insufficient against such sophisticated manipulation attacks. These results highlight an urgent need for developing innovative defensive solutions to ensure the security and trustworthiness of RAG systems.



## **26. PoisonCatcher: Revealing and Identifying LDP Poisoning Attacks in IIoT**

cs.CR

14 pages,7 figures, 2 tables

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2412.15704v2) [paper-pdf](http://arxiv.org/pdf/2412.15704v2)

**Authors**: Lisha Shuai, Shaofeng Tan, Nan Zhang, Jiamin Zhang, Min Zhang, Xiaolong Yang

**Abstract**: Local Differential Privacy (LDP), a robust privacy-protection model, is widely adopted in the Industrial Internet of Things (IIoT) due to its lightweight, decentralized, and scalable. However, its perturbation-based privacy-protection mechanism hinders distinguishing between any two data, thereby facilitating LDP poisoning attacks. The exposed physical-layer vulnerabilities and resource-constrained prevalent at the IIoT edge not only facilitate such attacks but also render existing LDP poisoning defenses, all of which are deployed at the edge and rely on ample resources, impractical.   This work proposes a LDP poisoning defense for IIoT in the resource-rich aggregator. We first reveal key poisoning attack modes occurring within the LDP-utilized IIoT data-collection process, detailing how IIoT vulnerabilities enable attacks, and then formulate a general attack model and derive the poisoned data's indistinguishability. This work subsequently analyzes the poisoning impacts on aggregated data based on industrial process correlation, revealing the distortion of statistical query results' temporal similarity and the resulting disruption of inter-attribute correlation, and uncovering the intriguing paradox that adversaries' attempts to stabilize their poisoning actions for stealth are difficult to maintain. Given these findings, we propose PoisonCatcher, a solution for identifying poisoned data, which includes time-series detectors based on temporal similarity, attribute correlation, and pattern stability metrics to detect poisoned attributes, and a latent-bias feature miner for identifying poisons. Experiments on the real-world dataset indicate that PoisonCatcher successfully identifies poisoned data, demonstrating robust identification capabilities with F2 scores above 90.7\% under various attack settings.



## **27. Sharp Gaussian approximations for Decentralized Federated Learning**

stat.ML

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.08125v1) [paper-pdf](http://arxiv.org/pdf/2505.08125v1)

**Authors**: Soham Bonnerjee, Sayar Karmakar, Wei Biao Wu

**Abstract**: Federated Learning has gained traction in privacy-sensitive collaborative environments, with local SGD emerging as a key optimization method in decentralized settings. While its convergence properties are well-studied, asymptotic statistical guarantees beyond convergence remain limited. In this paper, we present two generalized Gaussian approximation results for local SGD and explore their implications. First, we prove a Berry-Esseen theorem for the final local SGD iterates, enabling valid multiplier bootstrap procedures. Second, motivated by robustness considerations, we introduce two distinct time-uniform Gaussian approximations for the entire trajectory of local SGD. The time-uniform approximations support Gaussian bootstrap-based tests for detecting adversarial attacks. Extensive simulations are provided to support our theoretical results.



## **28. LiteLMGuard: Seamless and Lightweight On-Device Prompt Filtering for Safeguarding Small Language Models against Quantization-induced Risks and Vulnerabilities**

cs.CR

14 pages, 18 figures, and 4 tables

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.05619v2) [paper-pdf](http://arxiv.org/pdf/2505.05619v2)

**Authors**: Kalyan Nakka, Jimmy Dani, Ausmit Mondal, Nitesh Saxena

**Abstract**: The growing adoption of Large Language Models (LLMs) has influenced the development of their lighter counterparts-Small Language Models (SLMs)-to enable on-device deployment across smartphones and edge devices. These SLMs offer enhanced privacy, reduced latency, server-free functionality, and improved user experience. However, due to resource constraints of on-device environment, SLMs undergo size optimization through compression techniques like quantization, which can inadvertently introduce fairness, ethical and privacy risks. Critically, quantized SLMs may respond to harmful queries directly, without requiring adversarial manipulation, raising significant safety and trust concerns.   To address this, we propose LiteLMGuard (LLMG), an on-device prompt guard that provides real-time, prompt-level defense for quantized SLMs. Additionally, our prompt guard is designed to be model-agnostic such that it can be seamlessly integrated with any SLM, operating independently of underlying architectures. Our LLMG formalizes prompt filtering as a deep learning (DL)-based prompt answerability classification task, leveraging semantic understanding to determine whether a query should be answered by any SLM. Using our curated dataset, Answerable-or-Not, we trained and fine-tuned several DL models and selected ELECTRA as the candidate, with 97.75% answerability classification accuracy.   Our safety effectiveness evaluations demonstrate that LLMG defends against over 87% of harmful prompts, including both direct instruction and jailbreak attack strategies. We further showcase its ability to mitigate the Open Knowledge Attacks, where compromised SLMs provide unsafe responses without adversarial prompting. In terms of prompt filtering effectiveness, LLMG achieves near state-of-the-art filtering accuracy of 94%, with an average latency of 135 ms, incurring negligible overhead for users.



## **29. Dynamical Low-Rank Compression of Neural Networks with Robustness under Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.08022v1) [paper-pdf](http://arxiv.org/pdf/2505.08022v1)

**Authors**: Steffen Schotthöfer, H. Lexie Yang, Stefan Schnake

**Abstract**: Deployment of neural networks on resource-constrained devices demands models that are both compact and robust to adversarial inputs. However, compression and adversarial robustness often conflict. In this work, we introduce a dynamical low-rank training scheme enhanced with a novel spectral regularizer that controls the condition number of the low-rank core in each layer. This approach mitigates the sensitivity of compressed models to adversarial perturbations without sacrificing clean accuracy. The method is model- and data-agnostic, computationally efficient, and supports rank adaptivity to automatically compress the network at hand. Extensive experiments across standard architectures, datasets, and adversarial attacks show the regularized networks can achieve over 94% compression while recovering or improving adversarial accuracy relative to uncompressed baselines.



## **30. SCA: Improve Semantic Consistent in Unrestricted Adversarial Attacks via DDPM Inversion**

cs.CV

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2410.02240v6) [paper-pdf](http://arxiv.org/pdf/2410.02240v6)

**Authors**: Zihao Pan, Lifeng Chen, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Systems based on deep neural networks are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often result in substantial semantic distortions in the denoised output and suffer from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes a Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our code can be found at https://github.com/Pan-Zihao/SCA.



## **31. Must Read: A Systematic Survey of Computational Persuasion**

cs.CL

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07775v1) [paper-pdf](http://arxiv.org/pdf/2505.07775v1)

**Authors**: Nimet Beyza Bozdag, Shuhaib Mehri, Xiaocheng Yang, Hyeonjeong Ha, Zirui Cheng, Esin Durmus, Jiaxuan You, Heng Ji, Gokhan Tur, Dilek Hakkani-Tür

**Abstract**: Persuasion is a fundamental aspect of communication, influencing decision-making across diverse contexts, from everyday conversations to high-stakes scenarios such as politics, marketing, and law. The rise of conversational AI systems has significantly expanded the scope of persuasion, introducing both opportunities and risks. AI-driven persuasion can be leveraged for beneficial applications, but also poses threats through manipulation and unethical influence. Moreover, AI systems are not only persuaders, but also susceptible to persuasion, making them vulnerable to adversarial attacks and bias reinforcement. Despite rapid advancements in AI-generated persuasive content, our understanding of what makes persuasion effective remains limited due to its inherently subjective and context-dependent nature. In this survey, we provide a comprehensive overview of computational persuasion, structured around three key perspectives: (1) AI as a Persuader, which explores AI-generated persuasive content and its applications; (2) AI as a Persuadee, which examines AI's susceptibility to influence and manipulation; and (3) AI as a Persuasion Judge, which analyzes AI's role in evaluating persuasive strategies, detecting manipulation, and ensuring ethical persuasion. We introduce a taxonomy for computational persuasion research and discuss key challenges, including evaluating persuasiveness, mitigating manipulative persuasion, and developing responsible AI-driven persuasive systems. Our survey outlines future research directions to enhance the safety, fairness, and effectiveness of AI-powered persuasion while addressing the risks posed by increasingly capable language models.



## **32. Trial and Trust: Addressing Byzantine Attacks with Comprehensive Defense Strategy**

cs.LG

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07614v1) [paper-pdf](http://arxiv.org/pdf/2505.07614v1)

**Authors**: Gleb Molodtsov, Daniil Medyakov, Sergey Skorik, Nikolas Khachaturov, Shahane Tigranyan, Vladimir Aletov, Aram Avetisyan, Martin Takáč, Aleksandr Beznosikov

**Abstract**: Recent advancements in machine learning have improved performance while also increasing computational demands. While federated and distributed setups address these issues, their structure is vulnerable to malicious influences. In this paper, we address a specific threat, Byzantine attacks, where compromised clients inject adversarial updates to derail global convergence. We combine the trust scores concept with trial function methodology to dynamically filter outliers. Our methods address the critical limitations of previous approaches, allowing functionality even when Byzantine nodes are in the majority. Moreover, our algorithms adapt to widely used scaled methods like Adam and RMSProp, as well as practical scenarios, including local training and partial participation. We validate the robustness of our methods by conducting extensive experiments on both synthetic and real ECG data collected from medical institutions. Furthermore, we provide a broad theoretical analysis of our algorithms and their extensions to aforementioned practical setups. The convergence guarantees of our methods are comparable to those of classical algorithms developed without Byzantine interference.



## **33. SecReEvalBench: A Multi-turned Security Resilience Evaluation Benchmark for Large Language Models**

cs.CR

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07584v1) [paper-pdf](http://arxiv.org/pdf/2505.07584v1)

**Authors**: Huining Cui, Wei Liu

**Abstract**: The increasing deployment of large language models in security-sensitive domains necessitates rigorous evaluation of their resilience against adversarial prompt-based attacks. While previous benchmarks have focused on security evaluations with limited and predefined attack domains, such as cybersecurity attacks, they often lack a comprehensive assessment of intent-driven adversarial prompts and the consideration of real-life scenario-based multi-turn attacks. To address this gap, we present SecReEvalBench, the Security Resilience Evaluation Benchmark, which defines four novel metrics: Prompt Attack Resilience Score, Prompt Attack Refusal Logic Score, Chain-Based Attack Resilience Score and Chain-Based Attack Rejection Time Score. Moreover, SecReEvalBench employs six questioning sequences for model assessment: one-off attack, successive attack, successive reverse attack, alternative attack, sequential ascending attack with escalating threat levels and sequential descending attack with diminishing threat levels. In addition, we introduce a dataset customized for the benchmark, which incorporates both neutral and malicious prompts, categorised across seven security domains and sixteen attack techniques. In applying this benchmark, we systematically evaluate five state-of-the-art open-weighted large language models, Llama 3.1, Gemma 2, Mistral v0.3, DeepSeek-R1 and Qwen 3. Our findings offer critical insights into the strengths and weaknesses of modern large language models in defending against evolving adversarial threats. The SecReEvalBench dataset is publicly available at https://kaggle.com/datasets/5a7ee22cf9dab6c93b55a73f630f6c9b42e936351b0ae98fbae6ddaca7fe248d, which provides a groundwork for advancing research in large language model security.



## **34. GRADA: Graph-based Reranker against Adversarial Documents Attack**

cs.IR

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07546v1) [paper-pdf](http://arxiv.org/pdf/2505.07546v1)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.



## **35. Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems**

cs.CR

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2311.11796v2) [paper-pdf](http://arxiv.org/pdf/2311.11796v2)

**Authors**: Guangjing Wang, Ce Zhou, Yuanda Wang, Bocheng Chen, Hanqing Guo, Qiben Yan

**Abstract**: As Artificial Intelligence (AI) systems increasingly underpin critical applications, from autonomous vehicles to biometric authentication, their vulnerability to transferable attacks presents a growing concern. These attacks, designed to generalize across instances, domains, models, tasks, modalities, or even hardware platforms, pose severe risks to security, privacy, and system integrity. This survey delivers the first comprehensive review of transferable attacks across seven major categories, including evasion, backdoor, data poisoning, model stealing, model inversion, membership inference, and side-channel attacks. We introduce a unified six-dimensional taxonomy: cross-instance, cross-domain, cross-modality, cross-model, cross-task, and cross-hardware, which systematically captures the diverse transfer pathways of adversarial strategies. Through this framework, we examine both the underlying mechanics and practical implications of transferable attacks on AI systems. Furthermore, we review cutting-edge methods for enhancing attack transferability, organized around data augmentation and optimization strategies. By consolidating fragmented research and identifying critical future directions, this work provides a foundational roadmap for understanding, evaluating, and defending against transferable threats in real-world AI systems.



## **36. Decentralized Adversarial Training over Graphs**

cs.LG

arXiv admin note: text overlap with arXiv:2303.01936

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2303.13326v3) [paper-pdf](http://arxiv.org/pdf/2303.13326v3)

**Authors**: Ying Cao, Elsa Rizk, Stefan Vlaski, Ali H. Sayed

**Abstract**: The vulnerability of machine learning models to adversarial attacks has been attracting considerable attention in recent years. Most existing studies focus on the behavior of stand-alone single-agent learners. In comparison, this work studies adversarial training over graphs, where individual agents are subjected to perturbations of varied strength levels across space. It is expected that interactions by linked agents, and the heterogeneity of the attack models that are possible over the graph, can help enhance robustness in view of the coordination power of the group. Using a min-max formulation of distributed learning, we develop a decentralized adversarial training framework for multi-agent systems. Specifically, we devise two decentralized adversarial training algorithms by relying on two popular decentralized learning strategies--diffusion and consensus. We analyze the convergence properties of the proposed framework for strongly-convex, convex, and non-convex environments, and illustrate the enhanced robustness to adversarial attacks.



## **37. AttackBench: Evaluating Gradient-based Attacks for Adversarial Examples**

cs.LG

Paper accepted at AAAI2025. Project page and leaderboard:  https://attackbench.github.io

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2404.19460v3) [paper-pdf](http://arxiv.org/pdf/2404.19460v3)

**Authors**: Antonio Emanuele Cinà, Jérôme Rony, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Ismail Ben Ayed, Fabio Roli

**Abstract**: Adversarial examples are typically optimized with gradient-based attacks. While novel attacks are continuously proposed, each is shown to outperform its predecessors using different experimental setups, hyperparameter settings, and number of forward and backward calls to the target models. This provides overly-optimistic and even biased evaluations that may unfairly favor one particular attack over the others. In this work, we aim to overcome these limitations by proposing AttackBench, i.e., the first evaluation framework that enables a fair comparison among different attacks. To this end, we first propose a categorization of gradient-based attacks, identifying their main components and differences. We then introduce our framework, which evaluates their effectiveness and efficiency. We measure these characteristics by (i) defining an optimality metric that quantifies how close an attack is to the optimal solution, and (ii) limiting the number of forward and backward queries to the model, such that all attacks are compared within a given maximum query budget. Our extensive experimental analysis compares more than $100$ attack implementations with a total of over $800$ different configurations against CIFAR-10 and ImageNet models, highlighting that only very few attacks outperform all the competing approaches. Within this analysis, we shed light on several implementation issues that prevent many attacks from finding better solutions or running at all. We release AttackBench as a publicly-available benchmark, aiming to continuously update it to include and evaluate novel gradient-based attacks for optimizing adversarial examples.



## **38. No Query, No Access**

cs.CL

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07258v1) [paper-pdf](http://arxiv.org/pdf/2505.07258v1)

**Authors**: Wenqiang Wang, Siyuan Liang, Yangshijie Zhang, Xiaojun Jia, Hao Lin, Xiaochun Cao

**Abstract**: Textual adversarial attacks mislead NLP models, including Large Language Models (LLMs), by subtly modifying text. While effective, existing attacks often require knowledge of the victim model, extensive queries, or access to training data, limiting real-world feasibility. To overcome these constraints, we introduce the \textbf{Victim Data-based Adversarial Attack (VDBA)}, which operates using only victim texts. To prevent access to the victim model, we create a shadow dataset with publicly available pre-trained models and clustering methods as a foundation for developing substitute models. To address the low attack success rate (ASR) due to insufficient information feedback, we propose the hierarchical substitution model design, generating substitute models to mitigate the failure of a single substitute model at the decision boundary.   Concurrently, we use diverse adversarial example generation, employing various attack methods to generate and select the adversarial example with better similarity and attack effectiveness. Experiments on the Emotion and SST5 datasets show that VDBA outperforms state-of-the-art methods, achieving an ASR improvement of 52.08\% while significantly reducing attack queries to 0. More importantly, we discover that VDBA poses a significant threat to LLMs such as Qwen2 and the GPT family, and achieves the highest ASR of 45.99% even without access to the API, confirming that advanced NLP models still face serious security risks. Our codes can be found at https://anonymous.4open.science/r/VDBA-Victim-Data-based-Adversarial-Attack-36EC/



## **39. Approximate Energetic Resilience of Nonlinear Systems under Partial Loss of Control Authority**

math.OC

20 pages, 3 figures, 1 table

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2502.07603v2) [paper-pdf](http://arxiv.org/pdf/2502.07603v2)

**Authors**: Ram Padmanabhan, Melkior Ornik

**Abstract**: In this paper, we quantify the resilience of nonlinear dynamical systems by studying the increased energy used by all inputs of a system that suffers a partial loss of control authority, either through actuator malfunctions or through adversarial attacks. To quantify the maximal increase in energy, we introduce the notion of an energetic resilience metric. Prior work in this particular setting does not consider general nonlinear dynamical systems. In developing this framework, we first consider the special case of linear driftless systems and recall the energies in the control signal in the nominal and malfunctioning systems. Using these energies, we derive a bound on the energetic resilience metric. For general nonlinear systems, we first obtain a condition on the mean value of the control signal in both the nominal and malfunctioning systems, which allows us to approximate the energy in the control. We then obtain a worst-case approximation of this energy for the malfunctioning system, over all malfunctioning inputs. Assuming this approximation is exact, we derive bounds on the energetic resilience metric when control authority is lost over one actuator. A set of simulation examples demonstrate that the metric is useful in quantifying the resilience of the system without significant conservatism, despite the approximations used in obtaining control energies for nonlinear systems.



## **40. AugMixCloak: A Defense against Membership Inference Attacks via Image Transformation**

cs.LG

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2505.07149v1) [paper-pdf](http://arxiv.org/pdf/2505.07149v1)

**Authors**: Heqing Ren, Chao Feng, Alberto Huertas, Burkhard Stiller

**Abstract**: Traditional machine learning (ML) raises serious privacy concerns, while federated learning (FL) mitigates the risk of data leakage by keeping data on local devices. However, the training process of FL can still leak sensitive information, which adversaries may exploit to infer private data. One of the most prominent threats is the membership inference attack (MIA), where the adversary aims to determine whether a particular data record was part of the training set.   This paper addresses this problem through a two-stage defense called AugMixCloak. The core idea is to apply data augmentation and principal component analysis (PCA)-based information fusion to query images, which are detected by perceptual hashing (pHash) as either identical to or highly similar to images in the training set. Experimental results show that AugMixCloak successfully defends against both binary classifier-based MIA and metric-based MIA across five datasets and various decentralized FL (DFL) topologies. Compared with regularization-based defenses, AugMixCloak demonstrates stronger protection. Compared with confidence score masking, AugMixCloak exhibits better generalization.



## **41. Unleashing the potential of prompt engineering for large language models**

cs.CL

v6 - Metadata updated (title, journal ref, DOI). PDF identical to v5  (original submission). Please cite the peer-reviewed Version of Record in  "Patterns" (DOI: 10.1016/j.patter.2025.101260)

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2310.14735v6) [paper-pdf](http://arxiv.org/pdf/2310.14735v6)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review delves into the pivotal role of prompt engineering in unleashing the capabilities of Large Language Models (LLMs). The development of Artificial Intelligence (AI), from its inception in the 1950s to the emergence of advanced neural networks and deep learning architectures, has made a breakthrough in LLMs, with models such as GPT-4o and Claude-3, and in Vision-Language Models (VLMs), with models such as CLIP and ALIGN. Prompt engineering is the process of structuring inputs, which has emerged as a crucial technique to maximize the utility and accuracy of these models. This paper explores both foundational and advanced methodologies of prompt engineering, including techniques such as self-consistency, chain-of-thought, and generated knowledge, which significantly enhance model performance. Additionally, it examines the prompt method of VLMs through innovative approaches such as Context Optimization (CoOp), Conditional Context Optimization (CoCoOp), and Multimodal Prompt Learning (MaPLe). Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is also addressed through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review also reflects the essential role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.



## **42. IM-BERT: Enhancing Robustness of BERT through the Implicit Euler Method**

cs.CL

Accepted to EMNLP 2024 Main

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2505.06889v1) [paper-pdf](http://arxiv.org/pdf/2505.06889v1)

**Authors**: Mihyeon Kim, Juhyoung Park, Youngbin Kim

**Abstract**: Pre-trained Language Models (PLMs) have achieved remarkable performance on diverse NLP tasks through pre-training and fine-tuning. However, fine-tuning the model with a large number of parameters on limited downstream datasets often leads to vulnerability to adversarial attacks, causing overfitting of the model on standard datasets.   To address these issues, we propose IM-BERT from the perspective of a dynamic system by conceptualizing a layer of BERT as a solution of Ordinary Differential Equations (ODEs). Under the situation of initial value perturbation, we analyze the numerical stability of two main numerical ODE solvers: the explicit and implicit Euler approaches.   Based on these analyses, we introduce a numerically robust IM-connection incorporating BERT's layers. This strategy enhances the robustness of PLMs against adversarial attacks, even in low-resource scenarios, without introducing additional parameters or adversarial training strategies.   Experimental results on the adversarial GLUE (AdvGLUE) dataset validate the robustness of IM-BERT under various conditions. Compared to the original BERT, IM-BERT exhibits a performance improvement of approximately 8.3\%p on the AdvGLUE dataset. Furthermore, in low-resource scenarios, IM-BERT outperforms BERT by achieving 5.9\%p higher accuracy.



## **43. DP-TRAE: A Dual-Phase Merging Transferable Reversible Adversarial Example for Image Privacy Protection**

cs.CR

12 pages, 5 figures

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2505.06860v1) [paper-pdf](http://arxiv.org/pdf/2505.06860v1)

**Authors**: Xia Du, Jiajie Zhu, Jizhe Zhou, Chi-man Pun, Zheng Lin, Cong Wu, Zhe Chen, Jun Luo

**Abstract**: In the field of digital security, Reversible Adversarial Examples (RAE) combine adversarial attacks with reversible data hiding techniques to effectively protect sensitive data and prevent unauthorized analysis by malicious Deep Neural Networks (DNNs). However, existing RAE techniques primarily focus on white-box attacks, lacking a comprehensive evaluation of their effectiveness in black-box scenarios. This limitation impedes their broader deployment in complex, dynamic environments. Further more, traditional black-box attacks are often characterized by poor transferability and high query costs, significantly limiting their practical applicability. To address these challenges, we propose the Dual-Phase Merging Transferable Reversible Attack method, which generates highly transferable initial adversarial perturbations in a white-box model and employs a memory augmented black-box strategy to effectively mislead target mod els. Experimental results demonstrate the superiority of our approach, achieving a 99.0% attack success rate and 100% recovery rate in black-box scenarios, highlighting its robustness in privacy protection. Moreover, we successfully implemented a black-box attack on a commercial model, further substantiating the potential of this approach for practical use.



## **44. Endless Jailbreaks with Bijection Learning**

cs.CL

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2410.01294v3) [paper-pdf](http://arxiv.org/pdf/2410.01294v3)

**Authors**: Brian R. Y. Huang, Maximilian Li, Leonard Tang

**Abstract**: Despite extensive safety measures, LLMs are vulnerable to adversarial inputs, or jailbreaks, which can elicit unsafe behaviors. In this work, we introduce bijection learning, a powerful attack algorithm which automatically fuzzes LLMs for safety vulnerabilities using randomly-generated encodings whose complexity can be tightly controlled. We leverage in-context learning to teach models bijective encodings, pass encoded queries to the model to bypass built-in safety mechanisms, and finally decode responses back into English. Our attack is extremely effective on a wide range of frontier language models. Moreover, by controlling complexity parameters such as number of key-value mappings in the encodings, we find a close relationship between the capability level of the attacked LLM and the average complexity of the most effective bijection attacks. Our work highlights that new vulnerabilities in frontier models can emerge with scale: more capable models are more severely jailbroken by bijection attacks.



## **45. Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving**

cs.RO

Accepted in the 36th IEEE Intelligent Vehicles Symposium (IV 2025)

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06740v1) [paper-pdf](http://arxiv.org/pdf/2505.06740v1)

**Authors**: Ahmed Abouelazm, Mianzhi Liu, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner

**Abstract**: Accurate prediction of surrounding road users' trajectories is essential for safe and efficient autonomous driving. While deep learning models have improved performance, challenges remain in preventing off-road predictions and ensuring kinematic feasibility. Existing methods incorporate road-awareness modules and enforce kinematic constraints but lack plausibility guarantees and often introduce trade-offs in complexity and flexibility. This paper proposes a novel framework that formulates trajectory prediction as a constrained regression guided by permissible driving directions and their boundaries. Using the agent's current state and an HD map, our approach defines the valid boundaries and ensures on-road predictions by training the network to learn superimposed paths between left and right boundary polylines. To guarantee feasibility, the model predicts acceleration profiles that determine the vehicle's travel distance along these paths while adhering to kinematic constraints. We evaluate our approach on the Argoverse-2 dataset against the HPTR baseline. Our approach shows a slight decrease in benchmark metrics compared to HPTR but notably improves final displacement error and eliminates infeasible trajectories. Moreover, the proposed approach has superior generalization to less prevalent maneuvers and unseen out-of-distribution scenarios, reducing the off-road rate under adversarial attacks from 66\% to just 1\%. These results highlight the effectiveness of our approach in generating feasible and robust predictions.



## **46. Practical Reasoning Interruption Attacks on Reasoning Large Language Models**

cs.CR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06643v1) [paper-pdf](http://arxiv.org/pdf/2505.06643v1)

**Authors**: Yu Cui, Cong Zuo

**Abstract**: Reasoning large language models (RLLMs) have demonstrated outstanding performance across a variety of tasks, yet they also expose numerous security vulnerabilities. Most of these vulnerabilities have centered on the generation of unsafe content. However, recent work has identified a distinct "thinking-stopped" vulnerability in DeepSeek-R1: under adversarial prompts, the model's reasoning process ceases at the system level and produces an empty final answer. Building upon this vulnerability, researchers developed a novel prompt injection attack, termed reasoning interruption attack, and also offered an initial analysis of its root cause. Through extensive experiments, we verify the previous analyses, correct key errors based on three experimental findings, and present a more rigorous explanation of the fundamental causes driving the vulnerability. Moreover, existing attacks typically require over 2,000 tokens, impose significant overhead, reduce practicality, and are easily detected. To overcome these limitations, we propose the first practical reasoning interruption attack. It succeeds with just 109 tokens by exploiting our newly uncovered "reasoning token overflow" (RTO) effect to overwrite the model's final answer, forcing it to return an invalid response. Experimental results demonstrate that our proposed attack is highly effective. Furthermore, we discover that the method for triggering RTO differs between the official DeepSeek-R1 release and common unofficial deployments. As a broadened application of RTO, we also construct a novel jailbreak attack that enables the transfer of unsafe content within the reasoning tokens into final answer, thereby exposing it to the user. Our work carries significant implications for enhancing the security of RLLMs.



## **47. TAROT: Towards Essentially Domain-Invariant Robustness with Theoretical Justification**

cs.AI

Accepted in CVPR 2025 (19 pages, 7 figures)

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06580v1) [paper-pdf](http://arxiv.org/pdf/2505.06580v1)

**Authors**: Dongyoon Yang, Jihu Lee, Yongdai Kim

**Abstract**: Robust domain adaptation against adversarial attacks is a critical research area that aims to develop models capable of maintaining consistent performance across diverse and challenging domains. In this paper, we derive a new generalization bound for robust risk on the target domain using a novel divergence measure specifically designed for robust domain adaptation. Building upon this, we propose a new algorithm named TAROT, which is designed to enhance both domain adaptability and robustness. Through extensive experiments, TAROT not only surpasses state-of-the-art methods in accuracy and robustness but also significantly enhances domain generalization and scalability by effectively learning domain-invariant features. In particular, TAROT achieves superior performance on the challenging DomainNet dataset, demonstrating its ability to learn domain-invariant representations that generalize well across different domains, including unseen ones. These results highlight the broader applicability of our approach in real-world domain adaptation scenarios.



## **48. Towards Robust Recommendation: A Review and an Adversarial Robustness Evaluation Library**

cs.IR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2404.17844v2) [paper-pdf](http://arxiv.org/pdf/2404.17844v2)

**Authors**: Lei Cheng, Xiaowen Huang, Jitao Sang, Jian Yu

**Abstract**: Recently, recommender system has achieved significant success. However, due to the openness of recommender systems, they remain vulnerable to malicious attacks. Additionally, natural noise in training data and issues such as data sparsity can also degrade the performance of recommender systems. Therefore, enhancing the robustness of recommender systems has become an increasingly important research topic. In this survey, we provide a comprehensive overview of the robustness of recommender systems. Based on our investigation, we categorize the robustness of recommender systems into adversarial robustness and non-adversarial robustness. In the adversarial robustness, we introduce the fundamental principles and classical methods of recommender system adversarial attacks and defenses. In the non-adversarial robustness, we analyze non-adversarial robustness from the perspectives of data sparsity, natural noise, and data imbalance. Additionally, we summarize commonly used datasets and evaluation metrics for evaluating the robustness of recommender systems. Finally, we also discuss the current challenges in the field of recommender system robustness and potential future research directions. Additionally, to facilitate fair and efficient evaluation of attack and defense methods in adversarial robustness, we propose an adversarial robustness evaluation library--ShillingREC, and we conduct evaluations of basic attack models and recommendation models. ShillingREC project is released at https://github.com/chengleileilei/ShillingREC.



## **49. An In-kernel Forensics Engine for Investigating Evasive Attacks**

cs.CR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06498v1) [paper-pdf](http://arxiv.org/pdf/2505.06498v1)

**Authors**: Javad Zhandi, Lalchandra Rampersaud, Amin Kharraz

**Abstract**: Over the years, adversarial attempts against critical services have become more effective and sophisticated in launching low-profile attacks. This trend has always been concerning. However, an even more alarming trend is the increasing difficulty of collecting relevant evidence about these attacks and the involved threat actors in the early stages before significant damage is done. This issue puts defenders at a significant disadvantage, as it becomes exceedingly difficult to understand the attack details and formulate an appropriate response. Developing robust forensics tools to collect evidence about modern threats has never been easy. One main challenge is to provide a robust trade-off between achieving sufficient visibility while leaving minimal detectable artifacts. This paper will introduce LASE, an open-source Low-Artifact Forensics Engine to perform threat analysis and forensics in Windows operating system. LASE augments current analysis tools by providing detailed, system-wide monitoring capabilities while minimizing detectable artifacts. We designed multiple deployment scenarios, showing LASE's potential in evidence gathering and threat reasoning in a real-world setting. By making LASE and its execution trace data available to the broader research community, this work encourages further exploration in the field by reducing the engineering costs for threat analysis and building a longitudinal behavioral analysis catalog for diverse security domains.



## **50. Fun-tuning: Characterizing the Vulnerability of Proprietary LLMs to Optimization-based Prompt Injection Attacks via the Fine-Tuning Interface**

cs.CR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2501.09798v2) [paper-pdf](http://arxiv.org/pdf/2501.09798v2)

**Authors**: Andrey Labunets, Nishit V. Pandya, Ashish Hooda, Xiaohan Fu, Earlence Fernandes

**Abstract**: We surface a new threat to closed-weight Large Language Models (LLMs) that enables an attacker to compute optimization-based prompt injections. Specifically, we characterize how an attacker can leverage the loss-like information returned from the remote fine-tuning interface to guide the search for adversarial prompts. The fine-tuning interface is hosted by an LLM vendor and allows developers to fine-tune LLMs for their tasks, thus providing utility, but also exposes enough information for an attacker to compute adversarial prompts. Through an experimental analysis, we characterize the loss-like values returned by the Gemini fine-tuning API and demonstrate that they provide a useful signal for discrete optimization of adversarial prompts using a greedy search algorithm. Using the PurpleLlama prompt injection benchmark, we demonstrate attack success rates between 65% and 82% on Google's Gemini family of LLMs. These attacks exploit the classic utility-security tradeoff - the fine-tuning interface provides a useful feature for developers but also exposes the LLMs to powerful attacks.



