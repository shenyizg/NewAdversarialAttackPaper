# Latest Adversarial Attack Papers
**update at 2025-12-23 09:37:34**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Adversarial Robustness of Vision in Open Foundation Models**

cs.CV

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17902v1) [paper-pdf](https://arxiv.org/pdf/2512.17902v1)

**Authors**: Jonathon Fox, William J Buchanan, Pavlos Papadopoulos

**Abstract**: With the increase in deep learning, it becomes increasingly difficult to understand the model in which AI systems can identify objects. Thus, an adversary could aim to modify an image by adding unseen elements, which will confuse the AI in its recognition of an entity. This paper thus investigates the adversarial robustness of LLaVA-1.5-13B and Meta's Llama 3.2 Vision-8B-2. These are tested for untargeted PGD (Projected Gradient Descent) against the visual input modality, and empirically evaluated on the Visual Question Answering (VQA) v2 dataset subset. The results of these adversarial attacks are then quantified using the standard VQA accuracy metric. This evaluation is then compared with the accuracy degradation (accuracy drop) of LLaVA and Llama 3.2 Vision. A key finding is that Llama 3.2 Vision, despite a lower baseline accuracy in this setup, exhibited a smaller drop in performance under attack compared to LLaVA, particularly at higher perturbation levels. Overall, the findings confirm that the vision modality represents a viable attack vector for degrading the performance of contemporary open-weight VLMs, including Meta's Llama 3.2 Vision. Furthermore, they highlight that adversarial robustness does not necessarily correlate directly with standard benchmark performance and may be influenced by underlying architectural and training factors.



## **2. STAR: Semantic-Traffic Alignment and Retrieval for Zero-Shot HTTPS Website Fingerprinting**

cs.CR

Accepted by IEEE INFOCOM 2026. Camera-ready version

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17667v1) [paper-pdf](https://arxiv.org/pdf/2512.17667v1)

**Authors**: Yifei Cheng, Yujia Zhu, Baiyang Li, Xinhao Deng, Yitong Cai, Yaochen Ren, Qingyun Liu

**Abstract**: Modern HTTPS mechanisms such as Encrypted Client Hello (ECH) and encrypted DNS improve privacy but remain vulnerable to website fingerprinting (WF) attacks, where adversaries infer visited sites from encrypted traffic patterns. Existing WF methods rely on supervised learning with site-specific labeled traces, which limits scalability and fails to handle previously unseen websites. We address these limitations by reformulating WF as a zero-shot cross-modal retrieval problem and introducing STAR. STAR learns a joint embedding space for encrypted traffic traces and crawl-time logic profiles using a dual-encoder architecture. Trained on 150K automatically collected traffic-logic pairs with contrastive and consistency objectives and structure-aware augmentation, STAR retrieves the most semantically aligned profile for a trace without requiring target-side traffic during training. Experiments on 1,600 unseen websites show that STAR achieves 87.9 percent top-1 accuracy and 0.963 AUC in open-world detection, outperforming supervised and few-shot baselines. Adding an adapter with only four labeled traces per site further boosts top-5 accuracy to 98.8 percent. Our analysis reveals intrinsic semantic-traffic alignment in modern web protocols, identifying semantic leakage as the dominant privacy risk in encrypted HTTPS traffic. We release STAR's datasets and code to support reproducibility and future research.



## **3. SCAR: Semantic Cardiac Adversarial Representation via Spatiotemporal Manifold Optimization in ECG**

eess.SP

13 pages, 5 figures

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17423v1) [paper-pdf](https://arxiv.org/pdf/2512.17423v1)

**Authors**: Shunbo Jia, Caizhi Liao

**Abstract**: Deep learning models for Electrocardiogram (ECG) analysis have achieved expert-level performance but remain vulnerable to adversarial attacks. However, applying Universal Adversarial Perturbations (UAP) to ECG signals presents a unique challenge: standard imperceptible noise constraints (e.g., 10 uV) fail to generate effective universal attacks due to the high inter-subject variability of cardiac waveforms. Furthermore, traditional "invisible" attacks are easily dismissed by clinicians as technical artifacts, failing to compromise the human-in-the-loop diagnostic pipeline. In this study, we propose SCAR (Semantic Cardiac Adversarial Representation), a novel UAP framework tailored to bypass the clinical "Human Firewall." Unlike traditional approaches, SCAR integrates spatiotemporal smoothing (W=25, approx. 50ms), spectral consistency (<15 Hz), and anatomical amplitude constraints (<0.2 mV) directly into the gradient optimization manifold.   Results: We benchmarked SCAR against a rigorous baseline (Standard Universal DeepFool with post-hoc physiological filtering). While the baseline suffers a performance collapse (~16% success rate on transfer tasks), SCAR maintains robust transferability (58.09% on ResNet) and achieves 82.46% success on the source model. Crucially, clinical analysis reveals an emergent targeted behavior: SCAR specifically converges to forging Myocardial Infarction features (90.2% misdiagnosis) by mathematically reconstructing pathological ST-segment elevations. Finally, we demonstrate that SCAR serves a dual purpose: it not only functions as a robust data augmentation strategy for Hybrid Adversarial Training, offering optimal clinical defense, but also provides effective educational samples for training clinicians to recognize low-cost, AI-targeted semantic forgeries.



## **4. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

cs.LG

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17367v1) [paper-pdf](https://arxiv.org/pdf/2512.17367v1)

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.



## **5. Practical Framework for Privacy-Preserving and Byzantine-robust Federated Learning**

cs.CR

Accepted for publication in IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17254v1) [paper-pdf](https://arxiv.org/pdf/2512.17254v1)

**Authors**: Baolei Zhang, Minghong Fang, Zhuqing Liu, Biao Yi, Peizhao Zhou, Yuan Wang, Tong Li, Zheli Liu

**Abstract**: Federated Learning (FL) allows multiple clients to collaboratively train a model without sharing their private data. However, FL is vulnerable to Byzantine attacks, where adversaries manipulate client models to compromise the federated model, and privacy inference attacks, where adversaries exploit client models to infer private data. Existing defenses against both backdoor and privacy inference attacks introduce significant computational and communication overhead, creating a gap between theory and practice. To address this, we propose ABBR, a practical framework for Byzantine-robust and privacy-preserving FL. We are the first to utilize dimensionality reduction to speed up the private computation of complex filtering rules in privacy-preserving FL. Additionally, we analyze the accuracy loss of vector-wise filtering in low-dimensional space and introduce an adaptive tuning strategy to minimize the impact of malicious models that bypass filtering on the global model. We implement ABBR with state-of-the-art Byzantine-robust aggregation rules and evaluate it on public datasets, showing that it runs significantly faster, has minimal communication overhead, and maintains nearly the same Byzantine-resilience as the baselines.



## **6. Biosecurity-Aware AI: Agentic Risk Auditing of Soft Prompt Attacks on ESM-Based Variant Predictors**

cs.CR

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17146v1) [paper-pdf](https://arxiv.org/pdf/2512.17146v1)

**Authors**: Huixin Zhan

**Abstract**: Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated remarkable success in variant effect prediction. However, their security and robustness under adversarial manipulation remain largely unexplored. To address this gap, we introduce the Secure Agentic Genomic Evaluator (SAGE), an agentic framework for auditing the adversarial vulnerabilities of GFMs. SAGE functions through an interpretable and automated risk auditing loop. It injects soft prompt perturbations, monitors model behavior across training checkpoints, computes risk metrics such as AUROC and AUPR, and generates structured reports with large language model-based narrative explanations. This agentic process enables continuous evaluation of embedding-space robustness without modifying the underlying model. Using SAGE, we find that even state-of-the-art GFMs like ESM2 are sensitive to targeted soft prompt attacks, resulting in measurable performance degradation. These findings reveal critical and previously hidden vulnerabilities in genomic foundation models, showing the importance of agentic risk auditing in securing biomedical applications such as clinical variant interpretation.



## **7. Adversarial VR: An Open-Source Testbed for Evaluating Adversarial Robustness of VR Cybersickness Detection and Mitigation**

cs.CR

Published in the 2025 IEEE International Symposium on Mixed and Augmented Reality Adjunct (ISMAR-Adjunct)

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.17029v1) [paper-pdf](https://arxiv.org/pdf/2512.17029v1)

**Authors**: Istiak Ahmed, Ripan Kumar Kundu, Khaza Anuarul Hoque

**Abstract**: Deep learning (DL)-based automated cybersickness detection methods, along with adaptive mitigation techniques, can enhance user comfort and interaction. However, recent studies show that these DL-based systems are susceptible to adversarial attacks; small perturbations to sensor inputs can degrade model performance, trigger incorrect mitigation, and disrupt the user's immersive experience (UIX). Additionally, there is a lack of dedicated open-source testbeds that evaluate the robustness of these systems under adversarial conditions, limiting the ability to assess their real-world effectiveness. To address this gap, this paper introduces Adversarial-VR, a novel real-time VR testbed for evaluating DL-based cybersickness detection and mitigation strategies under adversarial conditions. Developed in Unity, the testbed integrates two state-of-the-art (SOTA) DL models: DeepTCN and Transformer, which are trained on the open-source MazeSick dataset, for real-time cybersickness severity detection and applies a dynamic visual tunneling mechanism that adjusts the field-of-view based on model outputs. To assess robustness, we incorporate three SOTA adversarial attacks: MI-FGSM, PGD, and C&W, which successfully prevent cybersickness mitigation by fooling DL-based cybersickness models' outcomes. We implement these attacks using a testbed with a custom-built VR Maze simulation and an HTC Vive Pro Eye headset, and we open-source our implementation for widespread adoption by VR developers and researchers. Results show that these adversarial attacks are capable of successfully fooling the system. For instance, the C&W attack results in a $5.94x decrease in accuracy for the Transformer-based cybersickness model compared to the accuracy without the attack.



## **8. PrivateXR: Defending Privacy Attacks in Extended Reality Through Explainable AI-Guided Differential Privacy**

cs.CR

Published in the IEEE ISMAR 2025 conference

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16851v1) [paper-pdf](https://arxiv.org/pdf/2512.16851v1)

**Authors**: Ripan Kumar Kundu, Istiak Ahmed, Khaza Anuarul Hoque

**Abstract**: The convergence of artificial AI and XR technologies (AI XR) promises innovative applications across many domains. However, the sensitive nature of data (e.g., eye-tracking) used in these systems raises significant privacy concerns, as adversaries can exploit these data and models to infer and leak personal information through membership inference attacks (MIA) and re-identification (RDA) with a high success rate. Researchers have proposed various techniques to mitigate such privacy attacks, including differential privacy (DP). However, AI XR datasets often contain numerous features, and applying DP uniformly can introduce unnecessary noise to less relevant features, degrade model accuracy, and increase inference time, limiting real-time XR deployment. Motivated by this, we propose a novel framework combining explainable AI (XAI) and DP-enabled privacy-preserving mechanisms to defend against privacy attacks. Specifically, we leverage post-hoc explanations to identify the most influential features in AI XR models and selectively apply DP to those features during inference. We evaluate our XAI-guided DP approach on three state-of-the-art AI XR models and three datasets: cybersickness, emotion, and activity classification. Our results show that the proposed method reduces MIA and RDA success rates by up to 43% and 39%, respectively, for cybersickness tasks while preserving model utility with up to 97% accuracy using Transformer models. Furthermore, it improves inference time by up to ~2x compared to traditional DP approaches. To demonstrate practicality, we deploy the XAI-guided DP AI XR models on an HTC VIVE Pro headset and develop a user interface (UI), namely PrivateXR, allowing users to adjust privacy levels (e.g., low, medium, high) while receiving real-time task predictions, protecting user privacy during XR gameplay.



## **9. Misspecified Crame-Rao Bound for AoA Estimation at a ULA under a Spoofing Attack**

eess.SP

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16735v1) [paper-pdf](https://arxiv.org/pdf/2512.16735v1)

**Authors**: Sotiris Skaperas, Arsenia Chorti

**Abstract**: A framework is presented for analyzing the impact of active attacks to location-based physical layer authentication (PLA) using the machinery of misspecified Cramér--Rao bound (MCRB). In this work, we focus on the MCRB in the angle-of-arrival (AoA) based authentication of a single antenna user when the verifier posseses an $M$ antenna element uniform linear array (ULA), assuming deterministic pilot signals; in our system model the presence of a spoofing adversary with an arbitrary number $L$ of antenna elements is assumed. We obtain a closed-form expression for the MCRB and demonstrate that the attack introduces in it a penalty term compared to the classic CRB, which does not depend on the signal-to-noise ratio (SNR) but on the adversary's location, the array geometry and the attacker precoding vector.



## **10. Dual-View Inference Attack: Machine Unlearning Amplifies Privacy Exposure**

cs.LG

Accepeted by AAAI2026

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16126v1) [paper-pdf](https://arxiv.org/pdf/2512.16126v1)

**Authors**: Lulu Xue, Shengshan Hu, Linqiang Qian, Peijin Guo, Yechao Zhang, Minghui Li, Yanjun Zhang, Dayong Ye, Leo Yu Zhang

**Abstract**: Machine unlearning is a newly popularized technique for removing specific training data from a trained model, enabling it to comply with data deletion requests. While it protects the rights of users requesting unlearning, it also introduces new privacy risks. Prior works have primarily focused on the privacy of data that has been unlearned, while the risks to retained data remain largely unexplored. To address this gap, we focus on the privacy risks of retained data and, for the first time, reveal the vulnerabilities introduced by machine unlearning under the dual-view setting, where an adversary can query both the original and the unlearned models. From an information-theoretic perspective, we introduce the concept of {privacy knowledge gain} and demonstrate that the dual-view setting allows adversaries to obtain more information than querying either model alone, thereby amplifying privacy leakage. To effectively demonstrate this threat, we propose DVIA, a Dual-View Inference Attack, which extracts membership information on retained data using black-box queries to both models. DVIA eliminates the need to train an attack model and employs a lightweight likelihood ratio inference module for efficient inference. Experiments across different datasets and model architectures validate the effectiveness of DVIA and highlight the privacy risks inherent in the dual-view setting.



## **11. Autoencoder-based Denoising Defense against Adversarial Attacks on Object Detection**

cs.CR

7 pages, 2 figures

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16123v1) [paper-pdf](https://arxiv.org/pdf/2512.16123v1)

**Authors**: Min Geun Song, Gang Min Kim, Woonmin Kim, Yongsik Kim, Jeonghyun Sim, Sangbeom Park, Huy Kang Kim

**Abstract**: Deep learning-based object detection models play a critical role in real-world applications such as autonomous driving and security surveillance systems, yet they remain vulnerable to adversarial examples. In this work, we propose an autoencoder-based denoising defense to recover object detection performance degraded by adversarial perturbations. We conduct adversarial attacks using Perlin noise on vehicle-related images from the COCO dataset, apply a single-layer convolutional autoencoder to remove the perturbations, and evaluate detection performance using YOLOv5. Our experiments demonstrate that adversarial attacks reduce bbox mAP from 0.2890 to 0.1640, representing a 43.3% performance degradation. After applying the proposed autoencoder defense, bbox mAP improves to 0.1700 (3.7% recovery) and bbox mAP@50 increases from 0.2780 to 0.3080 (10.8% improvement). These results indicate that autoencoder-based denoising can provide partial defense against adversarial attacks without requiring model retraining.



## **12. From Risk to Resilience: Towards Assessing and Mitigating the Risk of Data Reconstruction Attacks in Federated Learning**

cs.LG

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15460v1) [paper-pdf](https://arxiv.org/pdf/2512.15460v1)

**Authors**: Xiangrui Xu, Zhize Li, Yufei Han, Bin Wang, Jiqiang Liu, Wei Wang

**Abstract**: Data Reconstruction Attacks (DRA) pose a significant threat to Federated Learning (FL) systems by enabling adversaries to infer sensitive training data from local clients. Despite extensive research, the question of how to characterize and assess the risk of DRAs in FL systems remains unresolved due to the lack of a theoretically-grounded risk quantification framework. In this work, we address this gap by introducing Invertibility Loss (InvLoss) to quantify the maximum achievable effectiveness of DRAs for a given data instance and FL model. We derive a tight and computable upper bound for InvLoss and explore its implications from three perspectives. First, we show that DRA risk is governed by the spectral properties of the Jacobian matrix of exchanged model updates or feature embeddings, providing a unified explanation for the effectiveness of defense methods. Second, we develop InvRE, an InvLoss-based DRA risk estimator that offers attack method-agnostic, comprehensive risk evaluation across data instances and model architectures. Third, we propose two adaptive noise perturbation defenses that enhance FL privacy without harming classification accuracy. Extensive experiments on real-world datasets validate our framework, demonstrating its potential for systematic DRA risk evaluation and mitigation in FL systems.



## **13. Talking to the Airgap: Exploiting Radio-Less Embedded Devices as Radio Receivers**

cs.CR

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15387v1) [paper-pdf](https://arxiv.org/pdf/2512.15387v1)

**Authors**: Paul Staat, Daniel Davidovich, Christof Paar

**Abstract**: Intelligent electronics are deeply embedded in critical infrastructures and must remain reliable, particularly against deliberate attacks. To minimize risks and impede remote compromise, sensitive systems can be physically isolated from external networks, forming an airgap. Yet, airgaps can still be infiltrated by capable adversaries gaining code execution. Prior research has shown that attackers can then attempt to wirelessly exfiltrate data across the airgap by exploiting unintended radio emissions. In this work, we demonstrate reversal of this link: malicious code execution on embedded devices can enable wireless infiltration of airgapped systems without any hardware modification. In contrast to previous infiltration methods that depend on dedicated sensors (e.g., microphones, LEDs, or temperature sensors) or require strict line-of-sight, we show that unmodified, sensor-less embedded devices can inadvertently act as radio receivers. This phenomenon stems from parasitic RF sensitivity in PCB traces and on-chip analog-to-digital converters (ADCs), allowing external transmissions to be received and decoded entirely in software.   Across twelve commercially available embedded devices and two custom prototypes, we observe repeatable reception in the 300-1000 MHz range, with detectable signal power as low as 1 mW. To this end, we propose a systematic methodology to identify device configurations that foster such radio sensitivities and comprehensively evaluate their feasibility for wireless data reception. Exploiting these sensitivities, we demonstrate successful data reception over tens of meters, even in non-line-of-sight conditions and show that the reception sensitivities accommodate data rates of up to 100 kbps. Our findings reveal a previously unexplored command-and-control vector for air-gapped systems while challenging assumptions about their inherent isolation. [shortened]



## **14. Bounty Hunter: Autonomous, Comprehensive Emulation of Multi-Faceted Adversaries**

cs.CR

15 pages, 9 figures

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15275v1) [paper-pdf](https://arxiv.org/pdf/2512.15275v1)

**Authors**: Louis Hackländer-Jansen, Rafael Uetz, Martin Henze

**Abstract**: Adversary emulation is an essential procedure for cybersecurity assessments such as evaluating an organization's security posture or facilitating structured training and research in dedicated environments. To allow for systematic and time-efficient assessments, several approaches from academia and industry have worked towards the automation of adversarial actions. However, they exhibit significant limitations regarding autonomy, tactics coverage, and real-world applicability. Consequently, adversary emulation remains a predominantly manual task requiring substantial human effort and security expertise - even amidst the rise of Large Language Models. In this paper, we present Bounty Hunter, an automated adversary emulation method, designed and implemented as an open-source plugin for the popular adversary emulation platform Caldera, that enables autonomous emulation of adversaries with multi-faceted behavior while providing a wide coverage of tactics. To this end, it realizes diverse adversarial behavior, such as different levels of detectability and varying attack paths across repeated emulations. By autonomously compromising a simulated enterprise network, Bounty Hunter showcases its ability to achieve given objectives without prior knowledge of its target, including pre-compromise, initial compromise, and post-compromise attack tactics. Overall, Bounty Hunter facilitates autonomous, comprehensive, and multi-faceted adversary emulation to help researchers and practitioners in performing realistic and time-efficient security assessments, training exercises, and intrusion detection research.



## **15. An Efficient Gradient-Based Inference Attack for Federated Learning**

cs.LG

This paper was supported by the TRUMPET project, funded by the European Union under Grant Agreement No. 101070038

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15143v1) [paper-pdf](https://arxiv.org/pdf/2512.15143v1)

**Authors**: Pablo Montaña-Fernández, Ines Ortega-Fernandez

**Abstract**: Federated Learning is a machine learning setting that reduces direct data exposure, improving the privacy guarantees of machine learning models. Yet, the exchange of model updates between the participants and the aggregator can still leak sensitive information. In this work, we present a new gradient-based membership inference attack for federated learning scenarios that exploits the temporal evolution of last-layer gradients across multiple federated rounds. Our method uses the shadow technique to learn round-wise gradient patterns of the training records, requiring no access to the private dataset, and is designed to consider both semi-honest and malicious adversaries (aggregators or data owners). Beyond membership inference, we also provide a natural extension of the proposed attack to discrete attribute inference by contrasting gradient responses under alternative attribute hypotheses. The proposed attacks are model-agnostic, and therefore applicable to any gradient-based model and can be applied to both classification and regression settings. We evaluate the attack on CIFAR-100 and Purchase100 datasets for membership inference and on Breast Cancer Wisconsin for attribute inference. Our findings reveal strong attack performance and comparable computational and memory overhead in membership inference when compared to another attack from the literature. The obtained results emphasize that multi-round federated learning can increase the vulnerability to inference attacks, that aggregators pose a more substantial threat than data owners, and that attack performance is strongly influenced by the nature of the training dataset, with richer, high-dimensional data leading to stronger leakage than simpler tabular data.



## **16. Quantifying Return on Security Controls in LLM Systems**

cs.CR

13 pages, 9 figures, 3 tables

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15081v1) [paper-pdf](https://arxiv.org/pdf/2512.15081v1)

**Authors**: Richard Helder Moulton, Austin O'Brien, John D. Hastings

**Abstract**: Although large language models (LLMs) are increasingly used in security-critical workflows, practitioners lack quantitative guidance on which safeguards are worth deploying. This paper introduces a decision-oriented framework and reproducible methodology that together quantify residual risk, convert adversarial probe outcomes into financial risk estimates and return-on-control (RoC) metrics, and enable monetary comparison of layered defenses for LLM-based systems. A retrieval-augmented generation (RAG) service is instantiated using the DeepSeek-R1 model over a corpus containing synthetic personally identifiable information (PII), and subjected to automated attacks with Garak across five vulnerability classes: PII leakage, latent context injection, prompt injection, adversarial attack generation, and divergence. For each (vulnerability, control) pair, attack success probabilities are estimated via Laplace's Rule of Succession and combined with loss triangle distributions, calibrated from public breach-cost data, in 10,000-run Monte Carlo simulations to produce loss exceedance curves and expected losses. Three widely used mitigations, attribute-based access control (ABAC); named entity recognition (NER) redaction using Microsoft Presidio; and NeMo Guardrails, are then compared to a baseline RAG configuration. The baseline system exhibits very high attack success rates (>= 0.98 for PII, latent injection, and prompt injection), yielding a total simulated expected loss of $313k per attack scenario. ABAC collapses success probabilities for PII and prompt-related attacks to near zero and reduces the total expected loss by ~94%, achieving an RoC of 9.83. NER redaction likewise eliminates PII leakage and attains an RoC of 5.97, while NeMo Guardrails provides only marginal benefit (RoC of 0.05).



## **17. Cloud Security Leveraging AI: A Fusion-Based AISOC for Malware and Log Behaviour Detection**

cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14935v1) [paper-pdf](https://arxiv.org/pdf/2512.14935v1)

**Authors**: Nnamdi Philip Okonkwo, Lubna Luxmi Dhirani

**Abstract**: Cloud Security Operations Center (SOC) enable cloud governance, risk and compliance by providing insights visibility and control. Cloud SOC triages high-volume, heterogeneous telemetry from elastic, short-lived resources while staying within tight budgets. In this research, we implement an AI-Augmented Security Operations Center (AISOC) on AWS that combines cloud-native instrumentation with ML-based detection. The architecture uses three Amazon EC2 instances: Attacker, Defender, and Monitoring. We simulate a reverse-shell intrusion with Metasploit, and Filebeat forwards Defender logs to an Elasticsearch and Kibana stack for analysis. We train two classifiers, a malware detector built on a public dataset and a log-anomaly detector trained on synthetically augmented logs that include adversarial variants. We calibrate and fuse the scores to produce multi-modal threat intelligence and triage activity into NORMAL, SUSPICIOUS, and HIGH\_CONFIDENCE\_ATTACK. On held-out tests the fusion achieves strong macro-F1 (up to 1.00) under controlled conditions, though performance will vary in noisier and more diverse environments. These results indicate that simple, calibrated fusion can enhance cloud SOC capabilities in constrained, cost-sensitive setups.



## **18. PerProb: Indirectly Evaluating Memorization in Large Language Models**

cs.CR

Accepted at APSEC 2025

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14600v1) [paper-pdf](https://arxiv.org/pdf/2512.14600v1)

**Authors**: Yihan Liao, Jacky Keung, Xiaoxue Ma, Jingyu Zhang, Yicheng Sun

**Abstract**: The rapid advancement of Large Language Models (LLMs) has been driven by extensive datasets that may contain sensitive information, raising serious privacy concerns. One notable threat is the Membership Inference Attack (MIA), where adversaries infer whether a specific sample was used in model training. However, the true impact of MIA on LLMs remains unclear due to inconsistent findings and the lack of standardized evaluation methods, further complicated by the undisclosed nature of many LLM training sets. To address these limitations, we propose PerProb, a unified, label-free framework for indirectly assessing LLM memorization vulnerabilities. PerProb evaluates changes in perplexity and average log probability between data generated by victim and adversary models, enabling an indirect estimation of training-induced memory. Compared with prior MIA methods that rely on member/non-member labels or internal access, PerProb is independent of model and task, and applicable in both black-box and white-box settings. Through a systematic classification of MIA into four attack patterns, we evaluate PerProb's effectiveness across five datasets, revealing varying memory behaviors and privacy risks among LLMs. Additionally, we assess mitigation strategies, including knowledge distillation, early stopping, and differential privacy, demonstrating their effectiveness in reducing data leakage. Our findings offer a practical and generalizable framework for evaluating and improving LLM privacy.



## **19. Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space**

cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14448v1) [paper-pdf](https://arxiv.org/pdf/2512.14448v1)

**Authors**: Xingfu Zhou, Pengfei Wang

**Abstract**: Large Language Model (LLM) agents relying on external retrieval are increasingly deployed in high-stakes environments. While existing adversarial attacks primarily focus on content falsification or instruction injection, we identify a novel, process-oriented attack surface: the agent's reasoning style. We propose Reasoning-Style Poisoning (RSP), a paradigm that manipulates how agents process information rather than what they process. We introduce Generative Style Injection (GSI), an attack method that rewrites retrieved documents into pathological tones--specifically "analysis paralysis" or "cognitive haste"--without altering underlying facts or using explicit triggers. To quantify these shifts, we develop the Reasoning Style Vector (RSV), a metric tracking Verification depth, Self-confidence, and Attention focus. Experiments on HotpotQA and FEVER using ReAct, Reflection, and Tree of Thoughts (ToT) architectures reveal that GSI significantly degrades performance. It increases reasoning steps by up to 4.4 times or induces premature errors, successfully bypassing state-of-the-art content filters. Finally, we propose RSP-M, a lightweight runtime monitor that calculates RSV metrics in real-time and triggers alerts when values exceed safety thresholds. Our work demonstrates that reasoning style is a distinct, exploitable vulnerability, necessitating process-level defenses beyond static content analysis.



## **20. Mimicking Human Visual Development for Learning Robust Image Representations**

cs.CV

Accepted to ICVGIP 2025

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14360v1) [paper-pdf](https://arxiv.org/pdf/2512.14360v1)

**Authors**: Ankita Raj, Kaashika Prajaapat, Tapan Kumar Gandhi, Chetan Arora

**Abstract**: The human visual system is remarkably adept at adapting to changes in the input distribution; a capability modern convolutional neural networks (CNNs) still struggle to match. Drawing inspiration from the developmental trajectory of human vision, we propose a progressive blurring curriculum to improve the generalization and robustness of CNNs. Human infants are born with poor visual acuity, gradually refining their ability to perceive fine details. Mimicking this process, we begin training CNNs on highly blurred images during the initial epochs and progressively reduce the blur as training advances. This approach encourages the network to prioritize global structures over high-frequency artifacts, improving robustness against distribution shifts and noisy inputs. Challenging prior claims that blurring in the initial training epochs imposes a stimulus deficit and irreversibly harms model performance, we reveal that early-stage blurring enhances generalization with minimal impact on in-domain accuracy. Our experiments demonstrate that the proposed curriculum reduces mean corruption error (mCE) by up to 8.30% on CIFAR-10-C and 4.43% on ImageNet-100-C datasets, compared to standard training without blurring. Unlike static blur-based augmentation, which applies blurred images randomly throughout training, our method follows a structured progression, yielding consistent gains across various datasets. Furthermore, our approach complements other augmentation techniques, such as CutMix and MixUp, and enhances both natural and adversarial robustness against common attack methods. Code is available at https://github.com/rajankita/Visual_Acuity_Curriculum.



## **21. Optimizing the Adversarial Perturbation with a Momentum-based Adaptive Matrix**

cs.LG

IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14188v1) [paper-pdf](https://arxiv.org/pdf/2512.14188v1)

**Authors**: Wei Tao, Sheng Long, Xin Liu, Wei Li, Qing Tao

**Abstract**: Generating adversarial examples (AEs) can be formulated as an optimization problem. Among various optimization-based attacks, the gradient-based PGD and the momentum-based MI-FGSM have garnered considerable interest. However, all these attacks use the sign function to scale their perturbations, which raises several theoretical concerns from the point of view of optimization. In this paper, we first reveal that PGD is actually a specific reformulation of the projected gradient method using only the current gradient to determine its step-size. Further, we show that when we utilize a conventional adaptive matrix with the accumulated gradients to scale the perturbation, PGD becomes AdaGrad. Motivated by this analysis, we present a novel momentum-based attack AdaMI, in which the perturbation is optimized with an interesting momentum-based adaptive matrix. AdaMI is proved to attain optimal convergence for convex problems, indicating that it addresses the non-convergence issue of MI-FGSM, thereby ensuring stability of the optimization process. The experiments demonstrate that the proposed momentum-based adaptive matrix can serve as a general and effective technique to boost adversarial transferability over the state-of-the-art methods across different networks while maintaining better stability and imperceptibility.



## **22. On Improving Deep Active Learning with Formal Verification**

cs.LG

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14170v1) [paper-pdf](https://arxiv.org/pdf/2512.14170v1)

**Authors**: Jonathan Spiegelman, Guy Amir, Guy Katz

**Abstract**: Deep Active Learning (DAL) aims to reduce labeling costs in neural-network training by prioritizing the most informative unlabeled samples for annotation. Beyond selecting which samples to label, several DAL approaches further enhance data efficiency by augmenting the training set with synthetic inputs that do not require additional manual labeling. In this work, we investigate how augmenting the training data with adversarial inputs that violate robustness constraints can improve DAL performance. We show that adversarial examples generated via formal verification contribute substantially more than those produced by standard, gradient-based attacks. We apply this extension to multiple modern DAL techniques, as well as to a new technique that we propose, and show that it yields significant improvements in model generalization across standard benchmarks.



## **23. MURIM: Multidimensional Reputation-based Incentive Mechanism for Federated Learning**

cs.AI

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13955v1) [paper-pdf](https://arxiv.org/pdf/2512.13955v1)

**Authors**: Sindhuja Madabushi, Dawood Wasif, Jin-Hee Cho

**Abstract**: Federated Learning (FL) has emerged as a leading privacy-preserving machine learning paradigm, enabling participants to share model updates instead of raw data. However, FL continues to face key challenges, including weak client incentives, privacy risks, and resource constraints. Assessing client reliability is essential for fair incentive allocation and ensuring that each client's data contributes meaningfully to the global model. To this end, we propose MURIM, a MUlti-dimensional Reputation-based Incentive Mechanism that jointly considers client reliability, privacy, resource capacity, and fairness while preventing malicious or unreliable clients from earning undeserved rewards. MURIM allocates incentives based on client contribution, latency, and reputation, supported by a reliability verification module. Extensive experiments on MNIST, FMNIST, and ADULT Income datasets demonstrate that MURIM achieves up to 18% improvement in fairness metrics, reduces privacy attack success rates by 5-9%, and improves robustness against poisoning and noisy-gradient attacks by up to 85% compared to state-of-the-art baselines. Overall, MURIM effectively mitigates adversarial threats, promotes fair and truthful participation, and preserves stable model convergence across heterogeneous and dynamic federated settings.



## **24. Bilevel Optimization for Covert Memory Tampering in Heterogeneous Multi-Agent Architectures (XAMT)**

cs.CR

10 pages, 5 figures, 4 tables. Conference-style paper (IEEEtran). Proposes unified bilevel optimization framework for covert memory poisoning attacks in heterogeneous multi-agent systems (MARL + RAG)

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.15790v1) [paper-pdf](https://arxiv.org/pdf/2512.15790v1)

**Authors**: Akhil Sharma, Shaikh Yaser Arafat, Jai Kumar Sharma, Ken Huang

**Abstract**: The increasing operational reliance on complex Multi-Agent Systems (MAS) across safety-critical domains necessitates rigorous adversarial robustness assessment. Modern MAS are inherently heterogeneous, integrating conventional Multi-Agent Reinforcement Learning (MARL) with emerging Large Language Model (LLM) agent architectures utilizing Retrieval-Augmented Generation (RAG). A critical shared vulnerability is reliance on centralized memory components: the shared Experience Replay (ER) buffer in MARL and the external Knowledge Base (K) in RAG agents. This paper proposes XAMT (Bilevel Optimization for Covert Memory Tampering in Heterogeneous Multi-Agent Architectures), a novel framework that formalizes attack generation as a bilevel optimization problem. The Upper Level minimizes perturbation magnitude (delta) to enforce covertness while maximizing system behavior divergence toward an adversary-defined target (Lower Level). We provide rigorous mathematical instantiations for CTDE MARL algorithms and RAG-based LLM agents, demonstrating that bilevel optimization uniquely crafts stealthy, minimal-perturbation poisons evading detection heuristics. Comprehensive experimental protocols utilize SMAC and SafeRAG benchmarks to quantify effectiveness at sub-percent poison rates (less than or equal to 1 percent in MARL, less than or equal to 0.1 percent in RAG). XAMT defines a new unified class of training-time threats essential for developing intrinsically secure MAS, with implications for trust, formal verification, and defensive strategies prioritizing intrinsic safety over perimeter-based detection.



## **25. Evaluating Adversarial Attacks on Federated Learning for Temperature Forecasting**

cs.LG

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.13207v2) [paper-pdf](https://arxiv.org/pdf/2512.13207v2)

**Authors**: Karina Chichifoi, Fabio Merizzi, Michele Colajanni

**Abstract**: Deep learning and federated learning (FL) are becoming powerful partners for next-generation weather forecasting. Deep learning enables high-resolution spatiotemporal forecasts that can surpass traditional numerical models, while FL allows institutions in different locations to collaboratively train models without sharing raw data, addressing efficiency and security concerns. While FL has shown promise across heterogeneous regions, its distributed nature introduces new vulnerabilities. In particular, data poisoning attacks, in which compromised clients inject manipulated training data, can degrade performance or introduce systematic biases. These threats are amplified by spatial dependencies in meteorological data, allowing localized perturbations to influence broader regions through global model aggregation. In this study, we investigate how adversarial clients distort federated surface temperature forecasts trained on the Copernicus European Regional ReAnalysis (CERRA) dataset. We simulate geographically distributed clients and evaluate patch-based and global biasing attacks on regional temperature forecasts. Our results show that even a small fraction of poisoned clients can mislead predictions across large, spatially connected areas. A global temperature bias attack from a single compromised client shifts predictions by up to -1.7 K, while coordinated patch attacks more than triple the mean squared error and produce persistent regional anomalies exceeding +3.5 K. Finally, we assess trimmed mean aggregation as a defense mechanism, showing that it successfully defends against global bias attacks (2-13% degradation) but fails against patch attacks (281-603% amplification), exposing limitations of outlier-based defenses for spatially correlated data.



## **26. The Eminence in Shadow: Exploiting Feature Boundary Ambiguity for Robust Backdoor Attacks**

cs.LG

Accepted by KDD2026 Cycle 1 Research Track

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.10402v2) [paper-pdf](https://arxiv.org/pdf/2512.10402v2)

**Authors**: Zhou Feng, Jiahao Chen, Chunyi Zhou, Yuwen Pu, Tianyu Du, Jinbao Li, Jianhai Chen, Shouling Ji

**Abstract**: Deep neural networks (DNNs) underpin critical applications yet remain vulnerable to backdoor attacks, typically reliant on heuristic brute-force methods. Despite significant empirical advancements in backdoor research, the lack of rigorous theoretical analysis limits understanding of underlying mechanisms, constraining attack predictability and adaptability. Therefore, we provide a theoretical analysis targeting backdoor attacks, focusing on how sparse decision boundaries enable disproportionate model manipulation. Based on this finding, we derive a closed-form, ambiguous boundary region, wherein negligible relabeled samples induce substantial misclassification. Influence function analysis further quantifies significant parameter shifts caused by these margin samples, with minimal impact on clean accuracy, formally grounding why such low poison rates suffice for efficacious attacks. Leveraging these insights, we propose Eminence, an explainable and robust black-box backdoor framework with provable theoretical guarantees and inherent stealth properties. Eminence optimizes a universal, visually subtle trigger that strategically exploits vulnerable decision boundaries and effectively achieves robust misclassification with exceptionally low poison rates (< 0.1%, compared to SOTA methods typically requiring > 1%). Comprehensive experiments validate our theoretical discussions and demonstrate the effectiveness of Eminence, confirming an exponential relationship between margin poisoning and adversarial boundary manipulation. Eminence maintains > 90% attack success rate, exhibits negligible clean-accuracy loss, and demonstrates high transferability across diverse models, datasets and scenarios.



## **27. CNFinBench: A Benchmark for Safety and Compliance of Large Language Models in Finance**

cs.CE

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.09506v2) [paper-pdf](https://arxiv.org/pdf/2512.09506v2)

**Authors**: Jinru Ding, Chao Ding, Wenrao Pang, Boyi Xiao, Zhiqiang Liu, Pengcheng Chen, Jiayuan Chen, Tiantian Yuan, Junming Guan, Yidong Jiang, Dawei Cheng, Jie Xu

**Abstract**: Large language models (LLMs) are increasingly deployed across the financial sector for tasks like investment research and algorithmic trading. Their high-stakes nature demands rigorous evaluation of models' safety and regulatory alignment. However, there is a significant gap between evaluation capabilities and safety requirements. Current financial benchmarks mainly focus on textbook-style question answering and numerical problem-solving, failing to simulate the open-ended scenarios where safety risks typically manifest. To close these gaps, we introduce CNFinBench, a benchmark structured around a Capability-Compliance-Safety triad encompassing 15 subtasks. For Capability Q&As, we introduce a novel business-vertical taxonomy aligned with core financial domains like banking operations, which allows institutions to assess model readiness for deployment in operational scenarios. For Compliance and Risk Control Q&As, we embed regulatory requirements within realistic business scenarios to ensure models are evaluated under practical, scenario-driven conditions. For Safety Q&As, we uniquely incorporate structured bias and fairness auditing, a dimension overlooked by other holistic financial benchmarks, and introduce the first multi-turn adversarial dialogue task to systematically expose compliance decay under sustained, context-aware attacks. Accordingly, we propose the Harmful Instruction Compliance Score (HICS) to quantify models' consistency in resisting harmful instructions across multi-turn dialogues. Experiments on 21 models across all subtasks reveal a persistent gap between capability and compliance: models achieve an average score of 61.0 on capability tasks but drop to 34.2 on compliance and risk-control evaluations. In multi-turn adversarial dialogue tests, most LLMs attain only partial resistance, demonstrating that refusal alone is insufficient without cited, verifiable reasoning.



## **28. Developing Distance-Aware, and Evident Uncertainty Quantification in Dynamic Physics-Constrained Neural Networks for Robust Bearing Degradation Estimation**

cs.LG

Under review at Structural health Monitoring - SAGE

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.08499v2) [paper-pdf](https://arxiv.org/pdf/2512.08499v2)

**Authors**: Waleed Razzaq, Yun-Bo Zhao

**Abstract**: Accurate and uncertainty-aware degradation estimation is essential for predictive maintenance in safety-critical systems like rotating machinery with rolling-element bearings. Many existing uncertainty methods lack confidence calibration, are costly to run, are not distance-aware, and fail to generalize under out-of-distribution data. We introduce two distance-aware uncertainty methods for deterministic physics-guided neural networks: PG-SNGP, based on Spectral Normalization Gaussian Process, and PG-SNER, based on Deep Evidential Regression. We apply spectral normalization to the hidden layers so the network preserves distances from input to latent space. PG-SNGP replaces the final dense layer with a Gaussian Process layer for distance-sensitive uncertainty, while PG-SNER outputs Normal Inverse Gamma parameters to model uncertainty in a coherent probabilistic form. We assess performance using standard accuracy metrics and a new distance-aware metric based on the Pearson Correlation Coefficient, which measures how well predicted uncertainty tracks the distance between test and training samples. We also design a dynamic weighting scheme in the loss to balance data fidelity and physical consistency. We test our methods on rolling-element bearing degradation using the PRONOSTIA, XJTU-SY and HUST datasets and compare them with Monte Carlo and Deep Ensemble PGNNs. Results show that PG-SNGP and PG-SNER improve prediction accuracy, generalize reliably under OOD conditions, and remain robust to adversarial attacks and noise.



## **29. SEA: Spectral Edge Attack**

cs.LG

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.08964v2) [paper-pdf](https://arxiv.org/pdf/2512.08964v2)

**Authors**: Yongyu Wang

**Abstract**: Graph based machine learning algorithms occupy an important position in today AI landscape. The ability of graph topology to represent complex data structures is both the key strength of graph algorithms and a source of their vulnerability. In other words, attacking or perturbing a graph can severely degrade the performance of graph-based methods. For the attack methods, the greatest challenge is achieving strong attack effectiveness while remaining undetected. To address this problem, this paper proposes a new attack model that employs spectral adversarial robustness evaluation to quantitatively analyze the vulnerability of each edge in a graph under attack. By precisely targeting the weakest links, the proposed approach achieves the maximum attack impact with minimal perturbation. Experimental results demonstrate the effectiveness of the proposed method.



## **30. Tuning for Two Adversaries: Enhancing the Robustness Against Transfer and Query-Based Attacks using Hyperparameter Tuning**

cs.LG

To appear in the Proceedings of the AAAI Conference on Artificial Intelligence (AAAI) 2026

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2511.13654v2) [paper-pdf](https://arxiv.org/pdf/2511.13654v2)

**Authors**: Pascal Zimmer, Ghassan Karame

**Abstract**: In this paper, we present the first detailed analysis of how training hyperparameters -- such as learning rate, weight decay, momentum, and batch size -- influence robustness against both transfer-based and query-based attacks. Supported by theory and experiments, our study spans a variety of practical deployment settings, including centralized training, ensemble learning, and distributed training. We uncover a striking dichotomy: for transfer-based attacks, decreasing the learning rate significantly enhances robustness by up to $64\%$. In contrast, for query-based attacks, increasing the learning rate consistently leads to improved robustness by up to $28\%$ across various settings and data distributions. Leveraging these findings, we explore -- for the first time -- the training hyperparameter space to jointly enhance robustness against both transfer-based and query-based attacks. Our results reveal that distributed models benefit the most from hyperparameter tuning, achieving a remarkable tradeoff by simultaneously mitigating both attack types more effectively than other training setups.



## **31. Phantom Menace: Exploring and Enhancing the Robustness of VLA Models Against Physical Sensor Attacks**

cs.RO

Accepted by AAAI 2026 main track

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2511.10008v2) [paper-pdf](https://arxiv.org/pdf/2511.10008v2)

**Authors**: Xuancun Lu, Jiaxiang Chen, Shilin Xiao, Zizhi Jin, Zhangrui Chen, Hanwen Yu, Bohan Qian, Ruochen Zhou, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored. To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel "Real-Sim-Real" framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments.



## **32. Biologically-Informed Hybrid Membership Inference Attacks on Generative Genomic Models**

cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2511.07503v3) [paper-pdf](https://arxiv.org/pdf/2511.07503v3)

**Authors**: Asia Belfiore, Jonathan Passerat-Palmbach, Dmitrii Usynin

**Abstract**: The increased availability of genetic data has transformed genomics research, but raised many privacy concerns regarding its handling due to its sensitive nature. This work explores the use of language models (LMs) for the generation of synthetic genetic mutation profiles, leveraging differential privacy (DP) for the protection of sensitive genetic data. We empirically evaluate the privacy guarantees of our DP modes by introducing a novel Biologically-Informed Hybrid Membership Inference Attack (biHMIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. Our experiments show that both small and large transformer GPT-like models are viable synthetic variant generators for small-scale genomics, and that our hybrid attack leads, on average, to higher adversarial success compared to traditional metric-based MIAs.



## **33. Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems**

cs.SD

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2509.07677v3) [paper-pdf](https://arxiv.org/pdf/2509.07677v3)

**Authors**: Kamel Kamel, Hridoy Sankar Dutta, Keshav Sood, Sunil Aryal

**Abstract**: Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial samples that sound authentic while deceiving CMs. We conducted a comprehensive evaluation of our attack against state-of-the-art (SOTA) models across multiple tasks, under simulated real-world conditions. SMIA achieved a strong attack success rate (ASR) of at least 82% against combined VAS/CM systems, at least 97.5% against standalone speaker verification systems, and 100% against countermeasures. These findings conclusively demonstrate that current security postures are insufficient against adaptive adversarial attacks. This work highlights the urgent need for a paradigm shift toward next-generation defenses that employ dynamic, context-aware frameworks capable of evolving with the threat landscape.



## **34. Larger Scale Offers Better Security in the Nakamoto-style Blockchain**

cs.CR

22 pages, 4 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2509.05708v3) [paper-pdf](https://arxiv.org/pdf/2509.05708v3)

**Authors**: Junjie Hu

**Abstract**: Traditional security models for Nakamoto-style blockchains assume instantaneous synchronization among malicious nodes, which overestimate adversarial coordination capability. We revisit these existing models and propose two more realistic security models. First, we propose the static delay model. This model first incorporates adversarial communication delay. It quantifies how the delay constrains the effective growth rate of private chains and yields a closed-form expression for the security threshold. Second, we propose the dynamic delay model that further captures the decay of adversarial corruption capability and the total adversarial delay window. Theoretical analysis shows that private attacks remain optimal under both models. Finally, we prove that large-scale Nakamoto-style blockchains offer better security. This result provided a theoretical foundation for optimizing consensus protocols and assessing the robustness of large-scale blockchains.



## **35. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

cs.LG

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2508.17361v2) [paper-pdf](https://arxiv.org/pdf/2508.17361v2)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective against basic and reasoning models, but are also transferable across model families (OpenAI, Anthropic, Google), and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.



## **36. Exact Verification of Graph Neural Networks with Incremental Constraint Solving**

cs.LG

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2508.09320v2) [paper-pdf](https://arxiv.org/pdf/2508.09320v2)

**Authors**: Minghao Liu, Chia-Hsuan Lu, Marta Kwiatkowska

**Abstract**: Graph neural networks (GNNs) are increasingly employed in high-stakes applications, such as fraud detection or healthcare, but are susceptible to adversarial attacks. A number of techniques have been proposed to provide adversarial robustness guarantees, but support for commonly used aggregation functions in message-passing GNNs is lacking. In this paper, we develop an exact (sound and complete) verification method for GNNs to compute guarantees against attribute and structural perturbations that involve edge addition or deletion, subject to budget constraints. Our method employs constraint solving with bound tightening, and iteratively solves a sequence of relaxed constraint satisfaction problems while relying on incremental solving capabilities of solvers to improve efficiency. We implement GNNev, a versatile exact verifier for message-passing neural networks, which supports three aggregation functions, sum, max and mean, with the latter two considered here for the first time. Extensive experimental evaluation of GNNev on real-world fraud datasets (Amazon and Yelp) and biochemical datasets (MUTAG and ENZYMES) demonstrates its usability and effectiveness, as well as superior performance for node classification and competitiveness on graph classification compared to existing exact verification tools on sum-aggregated GNNs.



## **37. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

cs.CL

Published in NeurIPS 2025

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2507.06489v3) [paper-pdf](https://arxiv.org/pdf/2507.06489v3)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.



## **38. Breaking the Bulkhead: Demystifying Cross-Namespace Reference Vulnerabilities in Kubernetes Operators**

cs.CR

18 pages. Accepted by Network and Distributed System Security (NDSS) Symposium 2026. Some information has been omitted from this preprint version due to ethical considerations. The final published version differs from this version

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2507.03387v3) [paper-pdf](https://arxiv.org/pdf/2507.03387v3)

**Authors**: Andong Chen, Ziyi Guo, Zhaoxuan Jin, Zhenyuan Li, Yan Chen

**Abstract**: Kubernetes Operators, automated tools designed to manage application lifecycles within Kubernetes clusters, extend the functionalities of Kubernetes, and reduce the operational burden on human engineers. While Operators significantly simplify DevOps workflows, they introduce new security risks. In particular, Kubernetes enforces namespace isolation to separate workloads and limit user access, ensuring that users can only interact with resources within their authorized namespaces. However, Kubernetes Operators often demand elevated privileges and may interact with resources across multiple namespaces. This introduces a new class of vulnerabilities, the Cross-Namespace Reference Vulnerability. The root cause lies in the mismatch between the declared scope of resources and the implemented scope of the Operator logic, resulting in Kubernetes being unable to properly isolate the namespace. Leveraging such vulnerability, an adversary with limited access to a single authorized namespace may exploit the Operator to perform operations affecting other unauthorized namespaces, causing Privilege Escalation and further impacts.   To the best of our knowledge, this paper is the first to systematically investigate Kubernetes Operator attacks. We present Cross-Namespace Reference Vulnerability with two strategies, demonstrating how an attacker can bypass namespace isolation. Through large-scale measurements, we found that over 14% of Operators in the wild are potentially vulnerable. Our findings have been reported to the relevant developers, resulting in 8 confirmations and 7 CVEs by the time of submission, affecting vendors including Red Hat and NVIDIA, highlighting the critical need for enhanced security practices in Kubernetes Operators. To mitigate it, we open-source the static analysis suite and propose concrete mitigation to benefit the ecosystem.



## **39. Multimodal Representation Learning and Fusion**

cs.LG

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2506.20494v2) [paper-pdf](https://arxiv.org/pdf/2506.20494v2)

**Authors**: Qihang Jin, Enze Ge, Yuhang Xie, Hongying Luo, Junhao Song, Ziqian Bi, Chia Xin Liang, Jibin Guan, Joe Yeong, Xinyuan Song, Junfeng Hao

**Abstract**: Multi-modal learning is a fast growing area in artificial intelligence. It tries to help machines understand complex things by combining information from different sources, like images, text, and audio. By using the strengths of each modality, multi-modal learning allows AI systems to build stronger and richer internal representations. These help machines better interpretation, reasoning, and making decisions in real-life situations. This field includes core techniques such as representation learning (to get shared features from different data types), alignment methods (to match information across modalities), and fusion strategies (to combine them by deep learning models). Although there has been good progress, some major problems still remain. Like dealing with different data formats, missing or incomplete inputs, and defending against adversarial attacks. Researchers now are exploring new methods, such as unsupervised or semi-supervised learning, AutoML tools, to make models more efficient and easier to scale. And also more attention on designing better evaluation metrics or building shared benchmarks, make it easier to compare model performance across tasks and domains. As the field continues to grow, multi-modal learning is expected to improve many areas: computer vision, natural language processing, speech recognition, and healthcare. In the future, it may help to build AI systems that can understand the world in a way more like humans, flexible, context aware, and able to deal with real-world complexity.



## **40. Benchmarking Gaslighting Negation Attacks Against Reasoning Models**

cs.CV

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2506.09677v2) [paper-pdf](https://arxiv.org/pdf/2506.09677v2)

**Authors**: Bin Zhu, Hailong Yin, Jingjing Chen, Yu-Gang Jiang

**Abstract**: Recent advances in reasoning-centric models promise improved robustness through mechanisms such as chain-of-thought prompting and test-time scaling. However, their ability to withstand gaslighting negation attacks-adversarial prompts that confidently deny correct answers-remains underexplored. In this paper, we conduct a systematic evaluation of three state-of-the-art reasoning models, i.e., OpenAI's o4-mini, Claude-3.7-Sonnet and Gemini-2.5-Flash, across three multimodal benchmarks: MMMU, MathVista, and CharXiv. Our evaluation reveals significant accuracy drops (25-29% on average) following gaslighting negation attacks, indicating that even top-tier reasoning models struggle to preserve correct answers under manipulative user feedback. Built upon the insights of the evaluation and to further probe this vulnerability, we introduce GaslightingBench-R, a new diagnostic benchmark specifically designed to evaluate reasoning models' susceptibility to defend their belief under gaslighting negation attacks. Constructed by filtering and curating 1,025 challenging samples from the existing benchmarks, GaslightingBench-R induces even more dramatic failures, with accuracy drops exceeding 53% on average. Our findings highlight a fundamental gap between step-by-step reasoning and resistance to adversarial manipulation, calling for new robustness strategies that safeguard reasoning models against gaslighting negation attacks.



## **41. MoAPT: Mixture of Adversarial Prompt Tuning for Vision-Language Models**

cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2505.17509v2) [paper-pdf](https://arxiv.org/pdf/2505.17509v2)

**Authors**: Shiji Zhao, Qihui Zhu, Shukun Xiong, Shouwei Ruan, Maoxun Yuan, Jialing Tao, Jiexi Liu, Ranjie Duan, Jie Zhang, Jie Zhang, Xingxing Wei

**Abstract**: Large pre-trained Vision Language Models (VLMs) demonstrate excellent generalization capabilities but remain highly susceptible to adversarial examples, posing potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which ultimately results in overfitting. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts yields greater robustness improvements than simply extending the length of a single prompt. Building on this observation, we propose an adversarial tuning method named \textbf{Mixture of Adversarial Prompt Tuning (MoAPT)} to enhance the generalization against various adversarial attacks for VLMs. MoAPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the adversarial images to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific mixture text features aligning with different adversarial image features. Extensive experiments across 11 datasets under different settings show that our method can achieve better adversarial robustness than state-of-the-art approaches.



## **42. Improving Graph Neural Network Training, Defense and Hypergraph Partitioning via Adversarial Robustness Evaluation**

cs.LG

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2412.14738v9) [paper-pdf](https://arxiv.org/pdf/2412.14738v9)

**Authors**: Yongyu Wang

**Abstract**: Graph Neural Networks (GNNs) are a highly effective neural network architecture for processing graph-structured data. Unlike traditional neural networks that rely solely on the features of the data as input, GNNs leverage both the graph structure, which represents the relationships between data points, and the feature matrix of the data to optimize their feature representation. This unique capability enables GNNs to achieve superior performance across various tasks. However, it also makes GNNs more susceptible to noise and adversarial attacks from both the graph structure and data features, which can significantly increase the training difficulty and degrade their performance. Similarly, a hypergraph is a highly complex structure, and partitioning a hypergraph is a challenging task. This paper leverages spectral adversarial robustness evaluation to effectively address key challenges in complex-graph algorithms. By using spectral adversarial robustness evaluation to distinguish robust nodes from non-robust ones and treating them differently, we propose a training-set construction strategy that improves the training quality of GNNs. In addition, we develop algorithms to enhance both the adversarial robustness of GNNs and the performance of hypergraph partitioning. Experimental results show that this series of methods is highly effective.



## **43. Scaling Laws for Black box Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2411.16782v4) [paper-pdf](https://arxiv.org/pdf/2411.16782v4)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Jun Zhu, Yinpeng Dong

**Abstract**: Adversarial examples exhibit cross-model transferability, enabling threatening black-box attacks on commercial models. Model ensembling, which attacks multiple surrogate models, is a known strategy to improve this transferability. However, prior studies typically use small, fixed ensembles, which leaves open an intriguing question of whether scaling the number of surrogate models can further improve black-box attacks. In this work, we conduct the first large-scale empirical study of this question. We show that by resolving gradient conflict with advanced optimizers, we discover a robust and universal log-linear scaling law through both theoretical analysis and empirical evaluations: the Attack Success Rate (ASR) scales linearly with the logarithm of the ensemble size $T$. We rigorously verify this law across standard classifiers, SOTA defenses, and MLLMs, and find that scaling distills robust, semantic features of the target class. Consequently, we apply this fundamental insight to benchmark SOTA MLLMs. This reveals both the attack's devastating power and a clear robustness hierarchy: we achieve 80\%+ transfer attack success rate on proprietary models like GPT-4o, while also highlighting the exceptional resilience of Claude-3.5-Sonnet. Our findings urge a shift in focus for robustness evaluation: from designing intricate algorithms on small ensembles to understanding the principled and powerful threat of scaling.



## **44. Toward Robust and Accurate Adversarial Camouflage Generation against Vehicle Detectors**

cs.CV

14 pages. arXiv admin note: substantial text overlap with arXiv:2402.15853

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2411.10029v2) [paper-pdf](https://arxiv.org/pdf/2411.10029v2)

**Authors**: Jiawei Zhou, Linye Lyu, Daojing He, Yu Li

**Abstract**: Adversarial camouflage is a widely used physical attack against vehicle detectors for its superiority in multi-view attack performance. One promising approach involves using differentiable neural renderers to facilitate adversarial camouflage optimization through gradient back-propagation. However, existing methods often struggle to capture environmental characteristics during the rendering process or produce adversarial textures that can precisely map to the target vehicle. Moreover, these approaches neglect diverse weather conditions, reducing the efficacy of generated camouflage across varying weather scenarios. To tackle these challenges, we propose a robust and accurate camouflage generation method, namely RAUCA. The core of RAUCA is a novel neural rendering component, End-to-End Neural Renderer Plus (E2E-NRP), which can accurately optimize and project vehicle textures and render images with environmental characteristics such as lighting and weather. In addition, we integrate a multi-weather dataset for camouflage generation, leveraging the E2E-NRP to enhance the attack robustness. Experimental results on six popular object detectors show that RAUCA-final outperforms existing methods in both simulation and real-world settings.



## **45. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2407.20836v5) [paper-pdf](https://arxiv.org/pdf/2407.20836v5)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g., transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we demonstrate that adversarial attacks pose a real threat to AIGI detectors. FPBA can deliver successful black-box attacks across various detectors, generators, defense methods, and even evade cross-generator and compressed image detection, which are crucial real-world detection scenarios. Our code is available at https://github.com/onotoa/fpba.



## **46. Over-parameterization and Adversarial Robustness in Neural Networks: An Overview and Empirical Analysis**

cs.LG

Submitted to Discover AI

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2406.10090v2) [paper-pdf](https://arxiv.org/pdf/2406.10090v2)

**Authors**: Srishti Gupta, Zhang Chen, Luca Demetrio, Xiaoyi Feng, Zhaoqiang Xia, Antonio Emanuele Cinà, Maura Pintor, Luca Oneto, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: Thanks to their extensive capacity, over-parameterized neural networks exhibit superior predictive capabilities and generalization. However, having a large parameter space is considered one of the main suspects of the neural networks' vulnerability to adversarial example -- input samples crafted ad-hoc to induce a desired misclassification. Relevant literature has claimed contradictory remarks in support of and against the robustness of over-parameterized networks. These contradictory findings might be due to the failure of the attack employed to evaluate the networks' robustness. Previous research has demonstrated that depending on the considered model, the algorithm employed to generate adversarial examples may not function properly, leading to overestimating the model's robustness. In this work, we empirically study the robustness of over-parameterized networks against adversarial examples. However, unlike the previous works, we also evaluate the considered attack's reliability to support the results' veracity. Our results show that over-parameterized networks are robust against adversarial attacks as opposed to their under-parameterized counterparts.



## **47. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

cs.CV

Accepted by NDSS 2026

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2312.04960v5) [paper-pdf](https://arxiv.org/pdf/2312.04960v5)

**Authors**: Xiaoyun Xu, Shujian Yu, Zhuoran Liu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) have emerged as a fundamental architecture and serve as the backbone of modern vision-language models. Despite their impressive performance, ViTs exhibit notable vulnerability to evasion attacks, necessitating the development of specialized Adversarial Training (AT) strategies tailored to their unique architecture. While a direct solution might involve applying existing AT methods to ViTs, our analysis reveals significant incompatibilities, particularly with state-of-the-art (SOTA) approaches such as Generalist (CVPR 2023) and DBAT (USENIX Security 2024). This paper presents a systematic investigation of adversarial robustness in ViTs and provides a novel theoretical Mutual Information (MI) analysis in its autoencoder-based self-supervised pre-training. Specifically, we show that MI between the adversarial example and its latent representation in ViT-based autoencoders should be constrained via derived MI bounds. Building on this insight, we propose a self-supervised AT method, MIMIR, that employs an MI penalty to facilitate adversarial pre-training by masked image modeling with autoencoders. Extensive experiments on CIFAR-10, Tiny-ImageNet, and ImageNet-1K show that MIMIR can consistently provide improved natural and robust accuracy, where MIMIR outperforms SOTA AT results on ImageNet-1K. Notably, MIMIR demonstrates superior robustness against unforeseen attacks and common corruption data and can also withstand adaptive attacks where the adversary possesses full knowledge of the defense mechanism. Our code and trained models are publicly available at: https://github.com/xiaoyunxxy/MIMIR.



## **48. Enigma: Application-Layer Privacy for Quantum Optimization on Untrusted Computers**

quant-ph

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2311.13546v2) [paper-pdf](https://arxiv.org/pdf/2311.13546v2)

**Authors**: Ramin Ayanzadeh, Ahmad Mousavi, Amirhossein Basareh, Narges Alavisamani, Kazem Taram

**Abstract**: The Early Fault-Tolerant (EFT) era is emerging, where modest Quantum Error Correction (QEC) can enable quantum utility before full-scale fault tolerance. Quantum optimization is a leading candidate for early applications, but protecting these workloads is critical since they will run on expensive cloud services where providers could learn sensitive problem details. Experience with classical computing systems has shown that treating security as an afterthought can lead to significant vulnerabilities. Thus, we must address the security implications of quantum computing before widespread adoption. However, current Secure Quantum Computing (SQC) approaches, although theoretically promising, are impractical in the EFT era: blind quantum computing requires large-scale quantum networks, and quantum homomorphic encryption depends on full QEC.   We propose application-specific SQC, a principle that applies obfuscation at the application layer to enable practical deployment while remaining agnostic to algorithms, computing models, and hardware architectures. We present Enigma, the first realization of this principle for quantum optimization. Enigma integrates three complementary obfuscations: ValueGuard scrambles coefficients, StructureCamouflage inserts decoys, and TopologyTrimmer prunes variables. These techniques guarantee recovery of original solutions, and their stochastic nature resists repository-matching attacks. Evaluated against seven state-of-the-art AI models across five representative graph families, even combined adversaries, under a conservatively strong attacker model, identify the correct problem within their top five guesses in only 4.4% of cases. The protections come at the cost of problem size and T-gate counts increasing by averages of 1.07x and 1.13x, respectively, with both obfuscation and decoding completing within seconds for large-scale problems.



## **49. Why Does Little Robustness Help? A Further Step Towards Understanding Adversarial Transferability**

cs.LG

IEEE Symposium on Security and Privacy (Oakland) 2024; Extended version; Fix an proof error of Theorem 1

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2307.07873v8) [paper-pdf](https://arxiv.org/pdf/2307.07873v8)

**Authors**: Yechao Zhang, Shengshan Hu, Leo Yu Zhang, Junyu Shi, Minghui Li, Xiaogeng Liu, Wei Wan, Hai Jin

**Abstract**: Adversarial examples (AEs) for DNNs have been shown to be transferable: AEs that successfully fool white-box surrogate models can also deceive other black-box models with different architectures. Although a bunch of empirical studies have provided guidance on generating highly transferable AEs, many of these findings lack explanations and even lead to inconsistent advice. In this paper, we take a further step towards understanding adversarial transferability, with a particular focus on surrogate aspects. Starting from the intriguing little robustness phenomenon, where models adversarially trained with mildly perturbed adversarial samples can serve as better surrogates, we attribute it to a trade-off between two predominant factors: model smoothness and gradient similarity. Our investigations focus on their joint effects, rather than their separate correlations with transferability. Through a series of theoretical and empirical analyses, we conjecture that the data distribution shift in adversarial training explains the degradation of gradient similarity. Building on these insights, we explore the impacts of data augmentation and gradient regularization on transferability and identify that the trade-off generally exists in the various training mechanisms, thus building a comprehensive blueprint for the regulation mechanism behind transferability. Finally, we provide a general route for constructing better surrogates to boost transferability which optimizes both model smoothness and gradient similarity simultaneously, e.g., the combination of input gradient regularization and sharpness-aware minimization (SAM), validated by extensive experiments. In summary, we call for attention to the united impacts of these two factors for launching effective transfer attacks, rather than optimizing one while ignoring the other, and emphasize the crucial role of manipulating surrogate models.



## **50. A First Order Meta Stackelberg Method for Robust Federated Learning (Technical Report)**

cs.CR

This submission is a technical report for "A First Order Meta Stackelberg Method for Robust Federated Learning" (arXiv:2306.13800). We later submitted a full paper, "Meta Stackelberg Game: Robust Federated Learning Against Adaptive and Mixed Poisoning Attacks" (arXiv:2410.17431), which fully incorporates this report in its Appendix. To avoid duplication, we withdraw this submission

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2306.13273v3) [paper-pdf](https://arxiv.org/pdf/2306.13273v3)

**Authors**: Henger Li, Tianyi Xu, Tao Li, Yunian Pan, Quanyan Zhu, Zizhan Zheng

**Abstract**: Recent research efforts indicate that federated learning (FL) systems are vulnerable to a variety of security breaches. While numerous defense strategies have been suggested, they are mainly designed to counter specific attack patterns and lack adaptability, rendering them less effective when facing uncertain or adaptive threats. This work models adversarial FL as a Bayesian Stackelberg Markov game (BSMG) between the defender and the attacker to address the lack of adaptability to uncertain adaptive attacks. We further devise an effective meta-learning technique to solve for the Stackelberg equilibrium, leading to a resilient and adaptable defense. The experiment results suggest that our meta-Stackelberg learning approach excels in combating intense model poisoning and backdoor attacks of indeterminate types.



