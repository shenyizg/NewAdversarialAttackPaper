# Latest Adversarial Attack Papers
**update at 2025-09-08 17:25:12**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. On Evaluating the Poisoning Robustness of Federated Learning under Local Differential Privacy**

cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05265v1) [paper-pdf](http://arxiv.org/pdf/2509.05265v1)

**Authors**: Zijian Wang, Wei Tong, Tingxuan Han, Haoyu Chen, Tianling Zhang, Yunlong Mao, Sheng Zhong

**Abstract**: Federated learning (FL) combined with local differential privacy (LDP) enables privacy-preserving model training across decentralized data sources. However, the decentralized data-management paradigm leaves LDPFL vulnerable to participants with malicious intent. The robustness of LDPFL protocols, particularly against model poisoning attacks (MPA), where adversaries inject malicious updates to disrupt global model convergence, remains insufficiently studied. In this paper, we propose a novel and extensible model poisoning attack framework tailored for LDPFL settings. Our approach is driven by the objective of maximizing the global training loss while adhering to local privacy constraints. To counter robust aggregation mechanisms such as Multi-Krum and trimmed mean, we develop adaptive attacks that embed carefully crafted constraints into a reverse training process, enabling evasion of these defenses. We evaluate our framework across three representative LDPFL protocols, three benchmark datasets, and two types of deep neural networks. Additionally, we investigate the influence of data heterogeneity and privacy budgets on attack effectiveness. Experimental results demonstrate that our adaptive attacks can significantly degrade the performance of the global model, revealing critical vulnerabilities and highlighting the need for more robust LDPFL defense strategies against MPA. Our code is available at https://github.com/ZiJW/LDPFL-Attack



## **2. On Hyperparameters and Backdoor-Resistance in Horizontal Federated Learning**

cs.CR

To appear in the Proceedings of the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05192v1) [paper-pdf](http://arxiv.org/pdf/2509.05192v1)

**Authors**: Simon Lachnit, Ghassan Karame

**Abstract**: Horizontal Federated Learning (HFL) is particularly vulnerable to backdoor attacks as adversaries can easily manipulate both the training data and processes to execute sophisticated attacks. In this work, we study the impact of training hyperparameters on the effectiveness of backdoor attacks and defenses in HFL. More specifically, we show both analytically and by means of measurements that the choice of hyperparameters by benign clients does not only influence model accuracy but also significantly impacts backdoor attack success. This stands in sharp contrast with the multitude of contributions in the area of HFL security, which often rely on custom ad-hoc hyperparameter choices for benign clients$\unicode{x2013}$leading to more pronounced backdoor attack strength and diminished impact of defenses. Our results indicate that properly tuning benign clients' hyperparameters$\unicode{x2013}$such as learning rate, batch size, and number of local epochs$\unicode{x2013}$can significantly curb the effectiveness of backdoor attacks, regardless of the malicious clients' settings. We support this claim with an extensive robustness evaluation of state-of-the-art attack-defense combinations, showing that carefully chosen hyperparameters yield across-the-board improvements in robustness without sacrificing main task accuracy. For example, we show that the 50%-lifespan of the strong A3FL attack can be reduced by 98.6%, respectively$\unicode{x2013}$all without using any defense and while incurring only a 2.9 percentage points drop in clean task accuracy.



## **3. Jamming Smarter, Not Harder: Exploiting O-RAN Y1 RAN Analytics for Efficient Interference**

cs.CR

8 pages, 7 figures

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05161v1) [paper-pdf](http://arxiv.org/pdf/2509.05161v1)

**Authors**: Abiodun Ganiyu, Dara Ron, Syed Rafiul Hussain, Vijay K Shah

**Abstract**: The Y1 interface in O-RAN enables the sharing of RAN Analytics Information (RAI) between the near-RT RIC and authorized Y1 consumers, which may be internal applications within the operator's trusted domain or external systems accessing data through a secure exposure function. While this visibility enhances network optimization and enables advanced services, it also introduces a potential security risk -- a malicious or compromised Y1 consumer could misuse analytics to facilitate targeted interference. In this work, we demonstrate how an adversary can exploit the Y1 interface to launch selective jamming attacks by passively monitoring downlink metrics. We propose and evaluate two Y1-aided jamming strategies: a clustering-based jammer leveraging DBSCAN for traffic profiling and a threshold-based jammer. These are compared against two baselines strategies -- always-on jammer and random jammer -- on an over-the-air LTE/5G O-RAN testbed. Experimental results show that in unconstrained jamming budget scenarios, the threshold-based jammer can closely replicate the disruption caused by always-on jamming while reducing transmission time by 27\%. Under constrained jamming budgets, the clustering-based jammer proves most effective, causing up to an 18.1\% bitrate drop while remaining active only 25\% of the time. These findings reveal a critical trade-off between jamming stealthiness and efficiency, and illustrate how exposure of RAN analytics via the Y1 interface can enable highly targeted, low-overhead attacks, raising important security considerations for both civilian and mission-critical O-RAN deployments.



## **4. Robust Experts: the Effect of Adversarial Training on CNNs with Sparse Mixture-of-Experts Layers**

cs.CV

Accepted for publication at the STREAM workshop at ICCV 2025

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05086v1) [paper-pdf](http://arxiv.org/pdf/2509.05086v1)

**Authors**: Svetlana Pavlitska, Haixi Fan, Konstantin Ditschuneit, J. Marius Zöllner

**Abstract**: Robustifying convolutional neural networks (CNNs) against adversarial attacks remains challenging and often requires resource-intensive countermeasures. We explore the use of sparse mixture-of-experts (MoE) layers to improve robustness by replacing selected residual blocks or convolutional layers, thereby increasing model capacity without additional inference cost. On ResNet architectures trained on CIFAR-100, we find that inserting a single MoE layer in the deeper stages leads to consistent improvements in robustness under PGD and AutoPGD attacks when combined with adversarial training. Furthermore, we discover that when switch loss is used for balancing, it causes routing to collapse onto a small set of overused experts, thereby concentrating adversarial training on these paths and inadvertently making them more robust. As a result, some individual experts outperform the gated MoE model in robustness, suggesting that robust subpaths emerge through specialization. Our code is available at https://github.com/KASTEL-MobilityLab/robust-sparse-moes.



## **5. Adversarial Augmentation and Active Sampling for Robust Cyber Anomaly Detection**

cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04999v1) [paper-pdf](http://arxiv.org/pdf/2509.04999v1)

**Authors**: Sidahmed Benabderrahmane, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) present a considerable challenge to cybersecurity due to their stealthy, long-duration nature. Traditional supervised learning methods typically require large amounts of labeled data, which is often scarce in real-world scenarios. This paper introduces a novel approach that combines AutoEncoders for anomaly detection with active learning to iteratively enhance APT detection. By selectively querying an oracle for labels on uncertain or ambiguous samples, our method reduces labeling costs while improving detection accuracy, enabling the model to effectively learn with minimal data and reduce reliance on extensive manual labeling. We present a comprehensive formulation of the Attention Adversarial Dual AutoEncoder-based anomaly detection framework and demonstrate how the active learning loop progressively enhances the model's performance. The framework is evaluated on real-world, imbalanced provenance trace data from the DARPA Transparent Computing program, where APT-like attacks account for just 0.004\% of the data. The datasets, which cover multiple operating systems including Android, Linux, BSD, and Windows, are tested in two attack scenarios. The results show substantial improvements in detection rates during active learning, outperforming existing methods.



## **6. Training a Perceptual Model for Evaluating Auditory Similarity in Music Adversarial Attack**

cs.SD

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04985v1) [paper-pdf](http://arxiv.org/pdf/2509.04985v1)

**Authors**: Yuxuan Liu, Rui Sang, Peihong Zhang, Zhixin Li, Shengchen Li

**Abstract**: Music Information Retrieval (MIR) systems are highly vulnerable to adversarial attacks that are often imperceptible to humans, primarily due to a misalignment between model feature spaces and human auditory perception. Existing defenses and perceptual metrics frequently fail to adequately capture these auditory nuances, a limitation supported by our initial listening tests showing low correlation between common metrics and human judgments. To bridge this gap, we introduce Perceptually-Aligned MERT Transformer (PAMT), a novel framework for learning robust, perceptually-aligned music representations. Our core innovation lies in the psychoacoustically-conditioned sequential contrastive transformer, a lightweight projection head built atop a frozen MERT encoder. PAMT achieves a Spearman correlation coefficient of 0.65 with subjective scores, outperforming existing perceptual metrics. Our approach also achieves an average of 9.15\% improvement in robust accuracy on challenging MIR tasks, including Cover Song Identification and Music Genre Classification, under diverse perceptual adversarial attacks. This work pioneers architecturally-integrated psychoacoustic conditioning, yielding representations significantly more aligned with human perception and robust against music adversarial attacks.



## **7. MAIA: An Inpainting-Based Approach for Music Adversarial Attacks**

cs.SD

Accepted at ISMIR2025

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04980v1) [paper-pdf](http://arxiv.org/pdf/2509.04980v1)

**Authors**: Yuxuan Liu, Peihong Zhang, Rui Sang, Zhixin Li, Shengchen Li

**Abstract**: Music adversarial attacks have garnered significant interest in the field of Music Information Retrieval (MIR). In this paper, we present Music Adversarial Inpainting Attack (MAIA), a novel adversarial attack framework that supports both white-box and black-box attack scenarios. MAIA begins with an importance analysis to identify critical audio segments, which are then targeted for modification. Utilizing generative inpainting models, these segments are reconstructed with guidance from the output of the attacked model, ensuring subtle and effective adversarial perturbations. We evaluate MAIA on multiple MIR tasks, demonstrating high attack success rates in both white-box and black-box settings while maintaining minimal perceptual distortion. Additionally, subjective listening tests confirm the high audio fidelity of the adversarial samples. Our findings highlight vulnerabilities in current MIR systems and emphasize the need for more robust and secure models.



## **8. RINSER: Accurate API Prediction Using Masked Language Models**

cs.CY

16 pages, 8 figures

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04887v1) [paper-pdf](http://arxiv.org/pdf/2509.04887v1)

**Authors**: Muhammad Ejaz Ahmed, Christopher Cody, Muhammad Ikram, Sean Lamont, Alsharif Abuadbba, Seyit Camtepe, Surya Nepal, Muhammad Ali Kaafar

**Abstract**: Malware authors commonly use obfuscation to hide API identities in binary files, making analysis difficult and time-consuming for a human expert to understand the behavior and intent of the program. Automatic API prediction tools are necessary to efficiently analyze unknown binaries, facilitating rapid malware triage while reducing the workload on human analysts. In this paper, we present RINSER (AccuRate API predictioN using maSked languagE model leaRning), an automated framework for predicting Windows API (WinAPI) function names. RINSER introduces the novel concept of API codeprints, a set of API-relevant assembly instructions, and supports x86 PE binaries. RINSER relies on BERT's masked language model (LM) to predict API names at scale, achieving 85.77% accuracy for normal binaries and 82.88% accuracy for stripped binaries. We evaluate RINSER on a large dataset of 4.7M API codeprints from 11,098 malware binaries, covering 4,123 unique Windows APIs, making it the largest publicly available dataset of this type. RINSER successfully discovered 65 obfuscated Windows APIs related to C2 communication, spying, and evasion in our dataset, which the commercial disassembler IDA failed to identify. Furthermore, we compared RINSER against three state-of-the-art approaches, showing over 20% higher prediction accuracy. We also demonstrated RINSER's resilience to adversarial attacks, including instruction randomization and code displacement, with a performance drop of no more than 3%.



## **9. PersonaTeaming: Exploring How Introducing Personas Can Improve Automated AI Red-Teaming**

cs.AI

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.03728v2) [paper-pdf](http://arxiv.org/pdf/2509.03728v2)

**Authors**: Wesley Hanwen Deng, Sunnie S. Y. Kim, Akshita Jha, Ken Holstein, Motahhare Eslami, Lauren Wilcox, Leon A Gatys

**Abstract**: Recent developments in AI governance and safety research have called for red-teaming methods that can effectively surface potential risks posed by AI models. Many of these calls have emphasized how the identities and backgrounds of red-teamers can shape their red-teaming strategies, and thus the kinds of risks they are likely to uncover. While automated red-teaming approaches promise to complement human red-teaming by enabling larger-scale exploration of model behavior, current approaches do not consider the role of identity. As an initial step towards incorporating people's background and identities in automated red-teaming, we develop and evaluate a novel method, PersonaTeaming, that introduces personas in the adversarial prompt generation process to explore a wider spectrum of adversarial strategies. In particular, we first introduce a methodology for mutating prompts based on either "red-teaming expert" personas or "regular AI user" personas. We then develop a dynamic persona-generating algorithm that automatically generates various persona types adaptive to different seed prompts. In addition, we develop a set of new metrics to explicitly measure the "mutation distance" to complement existing diversity measurements of adversarial prompts. Our experiments show promising improvements (up to 144.1%) in the attack success rates of adversarial prompts through persona mutation, while maintaining prompt diversity, compared to RainbowPlus, a state-of-the-art automated red-teaming method. We discuss the strengths and limitations of different persona types and mutation methods, shedding light on future opportunities to explore complementarities between automated and human red-teaming approaches.



## **10. Adversarial Hubness in Multi-Modal Retrieval**

cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2412.14113v3) [paper-pdf](http://arxiv.org/pdf/2412.14113v3)

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries.   In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, and also for targeted attacks on queries related to specific, attacker-chosen concepts.   We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system implemented by Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub, generated using 100 random queries, is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries), demonstrating the strong generalization capabilities of adversarial hubs. We also investigate whether techniques for mitigating natural hubness can also mitigate adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.



## **11. Breaking to Build: A Threat Model of Prompt-Based Attacks for Securing LLMs**

cs.CL

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04615v1) [paper-pdf](http://arxiv.org/pdf/2509.04615v1)

**Authors**: Brennen Hill, Surendra Parla, Venkata Abhijeeth Balabhadruni, Atharv Prajod Padmalayam, Sujay Chandra Shekara Sharma

**Abstract**: The proliferation of Large Language Models (LLMs) has introduced critical security challenges, where adversarial actors can manipulate input prompts to cause significant harm and circumvent safety alignments. These prompt-based attacks exploit vulnerabilities in a model's design, training, and contextual understanding, leading to intellectual property theft, misinformation generation, and erosion of user trust. A systematic understanding of these attack vectors is the foundational step toward developing robust countermeasures. This paper presents a comprehensive literature survey of prompt-based attack methodologies, categorizing them to provide a clear threat model. By detailing the mechanisms and impacts of these exploits, this survey aims to inform the research community's efforts in building the next generation of secure LLMs that are inherently resistant to unauthorized distillation, fine-tuning, and editing.



## **12. Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing**

cs.LG

Published at ICLR 2025

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2412.13341v2) [paper-pdf](http://arxiv.org/pdf/2412.13341v2)

**Authors**: Keltin Grimes, Marco Christiani, David Shriver, Marissa Connor

**Abstract**: Model editing methods modify specific behaviors of Large Language Models by altering a small, targeted set of network weights and require very little data and compute. These methods can be used for malicious applications such as inserting misinformation or simple trojans that result in adversary-specified behaviors when a trigger word is present. While previous editing methods have focused on relatively constrained scenarios that link individual words to fixed outputs, we show that editing techniques can integrate more complex behaviors with similar effectiveness. We develop Concept-ROT, a model editing-based method that efficiently inserts trojans which not only exhibit complex output behaviors, but also trigger on high-level concepts -- presenting an entirely new class of trojan attacks. Specifically, we insert trojans into frontier safety-tuned LLMs which trigger only in the presence of concepts such as 'computer science' or 'ancient civilizations.' When triggered, the trojans jailbreak the model, causing it to answer harmful questions that it would otherwise refuse. Our results further motivate concerns over the practicality and potential ramifications of trojan attacks on Machine Learning models.



## **13. DisPatch: Disarming Adversarial Patches in Object Detection with Diffusion Models**

cs.CV

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04597v1) [paper-pdf](http://arxiv.org/pdf/2509.04597v1)

**Authors**: Jin Ma, Mohammed Aldeen, Christopher Salas, Feng Luo, Mashrur Chowdhury, Mert Pesé, Long Cheng

**Abstract**: Object detection is fundamental to various real-world applications, such as security monitoring and surveillance video analysis. Despite their advancements, state-of-theart object detectors are still vulnerable to adversarial patch attacks, which can be easily applied to real-world objects to either conceal actual items or create non-existent ones, leading to severe consequences. Given the current diversity of adversarial patch attacks and potential unknown threats, an ideal defense method should be effective, generalizable, and robust against adaptive attacks. In this work, we introduce DISPATCH, the first diffusion-based defense framework for object detection. Unlike previous works that aim to "detect and remove" adversarial patches, DISPATCH adopts a "regenerate and rectify" strategy, leveraging generative models to disarm attack effects while preserving the integrity of the input image. Specifically, we utilize the in-distribution generative power of diffusion models to regenerate the entire image, aligning it with benign data. A rectification process is then employed to identify and replace adversarial regions with their regenerated benign counterparts. DISPATCH is attack-agnostic and requires no prior knowledge of the existing patches. Extensive experiments across multiple detectors and attacks demonstrate that DISPATCH consistently outperforms state-of-the-art defenses on both hiding attacks and creating attacks, achieving the best overall mAP.5 score of 89.3% on hiding attacks, and lowering the attack success rate to 24.8% on untargeted creating attacks. Moreover, it maintains strong robustness against adaptive attacks, making it a practical and reliable defense for object detection systems.



## **14. Manipulating Transformer-Based Models: Controllability, Steerability, and Robust Interventions**

cs.CL

13 pages

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04549v1) [paper-pdf](http://arxiv.org/pdf/2509.04549v1)

**Authors**: Faruk Alpay, Taylan Alpay

**Abstract**: Transformer-based language models excel in NLP tasks, but fine-grained control remains challenging. This paper explores methods for manipulating transformer models through principled interventions at three levels: prompts, activations, and weights. We formalize controllable text generation as an optimization problem addressable via prompt engineering, parameter-efficient fine-tuning, model editing, and reinforcement learning. We introduce a unified framework encompassing prompt-level steering, activation interventions, and weight-space edits. We analyze robustness and safety implications, including adversarial attacks and alignment mitigations. Theoretically, we show minimal weight updates can achieve targeted behavior changes with limited side-effects. Empirically, we demonstrate >90% success in sentiment control and factual edits while preserving base performance, though generalization-specificity trade-offs exist. We discuss ethical dual-use risks and the need for rigorous evaluation. This work lays groundwork for designing controllable and robust language models.



## **15. LADSG: Label-Anonymized Distillation and Similar Gradient Substitution for Label Privacy in Vertical Federated Learning**

cs.CR

20 pages, 8 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2506.06742v3) [paper-pdf](http://arxiv.org/pdf/2506.06742v3)

**Authors**: Zeyu Yan, Yanfei Yao, Xuanbing Wen, Shixiong Zhang, Juli Zhang, Kai Fan

**Abstract**: Vertical Federated Learning (VFL) has emerged as a promising paradigm for collaborative model training across distributed feature spaces, which enables privacy-preserving learning without sharing raw data. However, recent studies have confirmed the feasibility of label inference attacks by internal adversaries. By strategically exploiting gradient vectors and semantic embeddings, attackers-through passive, active, or direct attacks-can accurately reconstruct private labels, leading to catastrophic data leakage. Existing defenses, which typically address isolated leakage vectors or are designed for specific types of attacks, remain vulnerable to emerging hybrid attacks that exploit multiple pathways simultaneously. To bridge this gap, we propose Label-Anonymized Defense with Substitution Gradient (LADSG), a unified and lightweight defense framework for VFL. LADSG first anonymizes true labels via soft distillation to reduce semantic exposure, then generates semantically-aligned substitute gradients to disrupt gradient-based leakage, and finally filters anomalous updates through gradient norm detection. It is scalable and compatible with standard VFL pipelines. Extensive experiments on six real-world datasets show that LADSG reduces the success rates of all three types of label inference attacks by 30-60% with minimal computational overhead, demonstrating its practical effectiveness.



## **16. An Automated, Scalable Machine Learning Model Inversion Assessment Pipeline**

cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04214v1) [paper-pdf](http://arxiv.org/pdf/2509.04214v1)

**Authors**: Tyler Shumaker, Jessica Carpenter, David Saranchak, Nathaniel D. Bastian

**Abstract**: Machine learning (ML) models have the potential to transform military battlefields, presenting a large external pressure to rapidly incorporate them into operational settings. However, it is well-established that these ML models are vulnerable to a number of adversarial attacks throughout the model deployment pipeline that threaten to negate battlefield advantage. One broad category is privacy attacks (such as model inversion) where an adversary can reverse engineer information from the model, such as the sensitive data used in its training. The ability to quantify the risk of model inversion attacks (MIAs) is not well studied, and there is a lack of automated developmental test and evaluation (DT&E) tools and metrics to quantify the effectiveness of privacy loss of the MIA. The current DT&E process is difficult because ML model inversions can be hard for a human to interpret, subjective when they are interpretable, and difficult to quantify in terms of inversion quality. Additionally, scaling the DT&E process is challenging due to many ML model architectures and data modalities that need to be assessed. In this work, we present a novel DT&E tool that quantifies the risk of data privacy loss from MIAs and introduces four adversarial risk dimensions to quantify privacy loss. Our DT&E pipeline combines inversion with vision language models (VLMs) to improve effectiveness while enabling scalable analysis. We demonstrate effectiveness using multiple MIA techniques and VLMs configured for zero-shot classification and image captioning. We benchmark the pipeline using several state-of-the-art MIAs in the computer vision domain with an image classification task that is typical in military applications. In general, our innovative pipeline extends the current model inversion DT&E capabilities by improving the effectiveness and scalability of the privacy loss analysis in an automated fashion.



## **17. MUNBa: Machine Unlearning via Nash Bargaining**

cs.CV

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2411.15537v4) [paper-pdf](http://arxiv.org/pdf/2411.15537v4)

**Authors**: Jing Wu, Mehrtash Harandi

**Abstract**: Machine Unlearning (MU) aims to selectively erase harmful behaviors from models while retaining the overall utility of the model. As a multi-task learning problem, MU involves balancing objectives related to forgetting specific concepts/data and preserving general performance. A naive integration of these forgetting and preserving objectives can lead to gradient conflicts and dominance, impeding MU algorithms from reaching optimal solutions. To address the gradient conflict and dominance issue, we reformulate MU as a two-player cooperative game, where the two players, namely, the forgetting player and the preservation player, contribute via their gradient proposals to maximize their overall gain and balance their contributions. To this end, inspired by the Nash bargaining theory, we derive a closed-form solution to guide the model toward the Pareto stationary point. Our formulation of MU guarantees an equilibrium solution, where any deviation from the final state would lead to a reduction in the overall objectives for both players, ensuring optimality in each objective. We evaluate our algorithm's effectiveness on a diverse set of tasks across image classification and image generation. Extensive experiments with ResNet, vision-language model CLIP, and text-to-image diffusion models demonstrate that our method outperforms state-of-the-art MU algorithms, achieving a better trade-off between forgetting and preserving. Our results also highlight improvements in forgetting precision, preservation of generalization, and robustness against adversarial attacks.



## **18. ICSLure: A Very High Interaction Honeynet for PLC-based Industrial Control Systems**

cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04080v1) [paper-pdf](http://arxiv.org/pdf/2509.04080v1)

**Authors**: Francesco Aurelio Pironti, Angelo Furfaro, Francesco Blefari, Carmelo Felicetti, Matteo Lupinacci, Francesco Romeo

**Abstract**: The security of Industrial Control Systems (ICSs) is critical to ensuring the safety of industrial processes and personnel. The rapid adoption of Industrial Internet of Things (IIoT) technologies has expanded system functionality but also increased the attack surface, exposing ICSs to a growing range of cyber threats. Honeypots provide a means to detect and analyze such threats by emulating target systems and capturing attacker behavior. However, traditional ICS honeypots, often limited to software-based simulations of a single Programmable Logic Controller (PLC), lack the realism required to engage sophisticated adversaries. In this work, we introduce a modular honeynet framework named ICSLure. The framework has been designed to emulate realistic ICS environments. Our approach integrates physical PLCs interacting with live data sources via industrial protocols such as Modbus and Profinet RTU, along with virtualized network components including routers, switches, and Remote Terminal Units (RTUs). The system incorporates comprehensive monitoring capabilities to collect detailed logs of attacker interactions. We demonstrate that our framework enables coherent and high-fidelity emulation of real-world industrial plants. This high-interaction environment significantly enhances the quality of threat data collected and supports advanced analysis of ICS-specific attack strategies, contributing to more effective detection and mitigation techniques.



## **19. System Identification from Partial Observations under Adversarial Attacks**

math.OC

8 pages, 3 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2504.00244v3) [paper-pdf](http://arxiv.org/pdf/2504.00244v3)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper is concerned with the partially observed linear system identification, where the goal is to obtain reasonably accurate estimation of the balanced truncation of the true system up to order $k$ from output measurements. We consider the challenging case of system identification under adversarial attacks, where the probability of having an attack at each time is $\Theta(1/k)$ while the value of the attack is arbitrary. We first show that the $\ell_1$-norm estimator exactly identifies the true Markov parameter matrix for nilpotent systems under any type of attack. We then build on this result to extend it to general systems and show that the estimation error exponentially decays as $k$ grows. The estimated balanced truncation model accordingly shows an exponentially decaying error for the identification of the true system up to a similarity transformation. This work is the first to provide the input-output analysis of the system with partial observations under arbitrary attacks.



## **20. NeuroBreak: Unveil Internal Jailbreak Mechanisms in Large Language Models**

cs.CR

12 pages, 9 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.03985v1) [paper-pdf](http://arxiv.org/pdf/2509.03985v1)

**Authors**: Chuhan Zhang, Ye Zhang, Bowen Shi, Yuyou Gan, Tianyu Du, Shouling Ji, Dazhan Deng, Yingcai Wu

**Abstract**: In deployment and application, large language models (LLMs) typically undergo safety alignment to prevent illegal and unethical outputs. However, the continuous advancement of jailbreak attack techniques, designed to bypass safety mechanisms with adversarial prompts, has placed increasing pressure on the security defenses of LLMs. Strengthening resistance to jailbreak attacks requires an in-depth understanding of the security mechanisms and vulnerabilities of LLMs. However, the vast number of parameters and complex structure of LLMs make analyzing security weaknesses from an internal perspective a challenging task. This paper presents NeuroBreak, a top-down jailbreak analysis system designed to analyze neuron-level safety mechanisms and mitigate vulnerabilities. We carefully design system requirements through collaboration with three experts in the field of AI security. The system provides a comprehensive analysis of various jailbreak attack methods. By incorporating layer-wise representation probing analysis, NeuroBreak offers a novel perspective on the model's decision-making process throughout its generation steps. Furthermore, the system supports the analysis of critical neurons from both semantic and functional perspectives, facilitating a deeper exploration of security mechanisms. We conduct quantitative evaluations and case studies to verify the effectiveness of our system, offering mechanistic insights for developing next-generation defense strategies against evolving jailbreak attacks.



## **21. Towards Robust Graph Structural Learning Beyond Homophily via Preserving Neighbor Similarity**

cs.LG

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2401.09754v2) [paper-pdf](http://arxiv.org/pdf/2401.09754v2)

**Authors**: Yulin Zhu, Yuni Lai, Xing Ai, Wai Lun LO, Gaolei Li, Jianhua Li, Di Tang, Xingxing Zhang, Mengpei Yang, Kai Zhou

**Abstract**: Despite the tremendous success of graph-based learning systems in handling structural data, it has been widely investigated that they are fragile to adversarial attacks on homophilic graph data, where adversaries maliciously modify the semantic and topology information of the raw graph data to degrade the predictive performances. Motivated by this, a series of robust models are crafted to enhance the adversarial robustness of graph-based learning systems on homophilic graphs. However, the security of graph-based learning systems on heterophilic graphs remains a mystery to us. To bridge this gap, in this paper, we start to explore the vulnerability of graph-based learning systems regardless of the homophily degree, and theoretically prove that the update of the negative classification loss is negatively correlated with the pairwise similarities based on the powered aggregated neighbor features. The theoretical finding inspires us to craft a novel robust graph structural learning strategy that serves as a useful graph mining module in a robust model that incorporates a dual-kNN graph constructions pipeline to supervise the neighbor-similarity-preserved propagation, where the graph convolutional layer adaptively smooths or discriminates the features of node pairs according to their affluent local structures. In this way, the proposed methods can mine the ``better" topology of the raw graph data under diverse graph homophily and achieve more reliable data management on homophilic and heterophilic graphs.



## **22. Generative AI for Physical-Layer Authentication**

eess.SP

10 pages, 3 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2504.18175v2) [paper-pdf](http://arxiv.org/pdf/2504.18175v2)

**Authors**: Rui Meng, Xiqi Cheng, Song Gao, Xiaodong Xu, Chen Dong, Guoshun Nan, Xiaofeng Tao, Ping Zhang, Tony Q. S. Quek

**Abstract**: In recent years, Artificial Intelligence (AI)-driven Physical-Layer Authentication (PLA), which focuses on achieving endogenous security and intelligent identity authentication, has attracted considerable interest. When compared with Discriminative AI (DAI), Generative AI (GAI) offers several advantages, such as fingerprint data augmentation, fingerprint denoising and reconstruction, and protection against adversarial attacks. Inspired by these innovations, this paper provides a systematic exploration of GAI's integration into PLA frameworks. We commence with a review of representative authentication techniques, emphasizing PLA's inherent strengths. Following this, we revisit four typical GAI models and contrast the limitations of DAI with the potential of GAI in addressing PLA challenges, including insufficient fingerprint data, environment noises and inferences, perturbations in fingerprint data, and complex tasks. Specifically, we delve into providing GAI-enhanced methods for PLA across the fingerprint collection, model training, and performance optimization phases in detail. Moreover, we present a case study that combines fingerprint extrapolation and Generative Diffusion Model (GDM) to illustrate the superiority of GAI in bolstering the reliability of PLA. Additionally, we outline potential future research directions for GAI-based PLA.



## **23. Learning an Adversarial World Model for Automated Curriculum Generation in MARL**

cs.LG

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03771v1) [paper-pdf](http://arxiv.org/pdf/2509.03771v1)

**Authors**: Brennen Hill

**Abstract**: World models that infer and predict environmental dynamics are foundational to embodied intelligence. However, their potential is often limited by the finite complexity and implicit biases of hand-crafted training environments. To develop truly generalizable and robust agents, we need environments that scale in complexity alongside the agents learning within them. In this work, we reframe the challenge of environment generation as the problem of learning a goal-conditioned, generative world model. We propose a system where a generative **Attacker** agent learns an implicit world model to synthesize increasingly difficult challenges for a team of cooperative **Defender** agents. The Attacker's objective is not passive prediction, but active, goal-driven interaction: it models and generates world states (i.e., configurations of enemy units) specifically to exploit the Defenders' weaknesses. Concurrently, the embodied Defender team learns a cooperative policy to overcome these generated worlds. This co-evolutionary dynamic creates a self-scaling curriculum where the world model continuously adapts to challenge the decision-making policy of the agents, providing an effectively infinite stream of novel and relevant training scenarios. We demonstrate that this framework leads to the emergence of complex behaviors, such as the world model learning to generate flanking and shielding formations, and the defenders learning coordinated focus-fire and spreading tactics. Our findings position adversarial co-evolution as a powerful method for learning instrumental world models that drive agents toward greater strategic depth and robustness.



## **24. ANNIE: Be Careful of Your Robots**

cs.AI

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03383v1) [paper-pdf](http://arxiv.org/pdf/2509.03383v1)

**Authors**: Yiyang Huang, Zixuan Wang, Zishen Wan, Yapeng Tian, Haobo Xu, Yinhe Han, Yiming Gan

**Abstract**: The integration of vision-language-action (VLA) models into embodied AI (EAI) robots is rapidly advancing their ability to perform complex, long-horizon tasks in humancentric environments. However, EAI systems introduce critical security risks: a compromised VLA model can directly translate adversarial perturbations on sensory input into unsafe physical actions. Traditional safety definitions and methodologies from the machine learning community are no longer sufficient. EAI systems raise new questions, such as what constitutes safety, how to measure it, and how to design effective attack and defense mechanisms in physically grounded, interactive settings. In this work, we present the first systematic study of adversarial safety attacks on embodied AI systems, grounded in ISO standards for human-robot interactions. We (1) formalize a principled taxonomy of safety violations (critical, dangerous, risky) based on physical constraints such as separation distance, velocity, and collision boundaries; (2) introduce ANNIEBench, a benchmark of nine safety-critical scenarios with 2,400 video-action sequences for evaluating embodied safety; and (3) ANNIE-Attack, a task-aware adversarial framework with an attack leader model that decomposes long-horizon goals into frame-level perturbations. Our evaluation across representative EAI models shows attack success rates exceeding 50% across all safety categories. We further demonstrate sparse and adaptive attack strategies and validate the real-world impact through physical robot experiments. These results expose a previously underexplored but highly consequential attack surface in embodied AI systems, highlighting the urgent need for security-driven defenses in the physical AI era. Code is available at https://github.com/RLCLab/Annie.



## **25. On the MIA Vulnerability Gap Between Private GANs and Diffusion Models**

cs.LG

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03341v1) [paper-pdf](http://arxiv.org/pdf/2509.03341v1)

**Authors**: Ilana Sebag, Jean-Yves Franceschi, Alain Rakotomamonjy, Alexandre Allauzen, Jamal Atif

**Abstract**: Generative Adversarial Networks (GANs) and diffusion models have emerged as leading approaches for high-quality image synthesis. While both can be trained under differential privacy (DP) to protect sensitive data, their sensitivity to membership inference attacks (MIAs), a key threat to data confidentiality, remains poorly understood. In this work, we present the first unified theoretical and empirical analysis of the privacy risks faced by differentially private generative models. We begin by showing, through a stability-based analysis, that GANs exhibit fundamentally lower sensitivity to data perturbations than diffusion models, suggesting a structural advantage in resisting MIAs. We then validate this insight with a comprehensive empirical study using a standardized MIA pipeline to evaluate privacy leakage across datasets and privacy budgets. Our results consistently reveal a marked privacy robustness gap in favor of GANs, even in strong DP regimes, highlighting that model type alone can critically shape privacy leakage.



## **26. Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models**

cs.CL

Presented at EMNLP 2025

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2410.20940v2) [paper-pdf](http://arxiv.org/pdf/2410.20940v2)

**Authors**: Piotr Przybyła, Euan McGill, Horacio Saggion

**Abstract**: Large language models have many beneficial applications, but can they also be used to attack content-filtering algorithms in social media platforms? We investigate the challenge of generating adversarial examples to test the robustness of text classification algorithms detecting low-credibility content, including propaganda, false claims, rumours and hyperpartisan news. We focus on simulation of content moderation by setting realistic limits on the number of queries an attacker is allowed to attempt. Within our solution (TREPAT), initial rephrasings are generated by large language models with prompts inspired by meaning-preserving NLP tasks, such as text simplification and style transfer. Subsequently, these modifications are decomposed into small changes, applied through beam search procedure, until the victim classifier changes its decision. We perform (1) quantitative evaluation using various prompts, models and query limits, (2) targeted manual assessment of the generated text and (3) qualitative linguistic analysis. The results confirm the superiority of our approach in the constrained scenario, especially in case of long input text (news articles), where exhaustive search is not feasible.



## **27. Efficient and Secure Sleepy Model for BFT Consensus**

cs.DC

Accepted to ESORICS 2025, 20 pages, 7 figures

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03145v1) [paper-pdf](http://arxiv.org/pdf/2509.03145v1)

**Authors**: Pengkun Ren, Hai Dong, Zahir Tari, Pengcheng Zhang

**Abstract**: Byzantine Fault Tolerant (BFT) consensus protocols for dynamically available systems face a critical challenge: balancing latency and security in fluctuating node participation. Existing solutions often require multiple rounds of voting per decision, leading to high latency or limited resilience to adversarial behavior. This paper presents a BFT protocol integrating a pre-commit mechanism with publicly verifiable secret sharing (PVSS) into message transmission. By binding users' identities to their messages through PVSS, our approach reduces communication rounds. Compared to other state-of-the-art methods, our protocol typically requires only four network delays (4$\Delta$) in common scenarios while being resilient to up to 1/2 adversarial participants. This integration enhances the efficiency and security of the protocol without compromising integrity. Theoretical analysis demonstrates the robustness of the protocol against Byzantine attacks. Experimental evaluations show that, compared to traditional BFT protocols, our protocol significantly prevents fork occurrences and improves chain stability. Furthermore, compared to longest-chain protocol, our protocol maintains stability and lower latency in scenarios with moderate participation fluctuations.



## **28. Similarity between Units of Natural Language: The Transition from Coarse to Fine Estimation**

cs.CL

PhD thesis

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2210.14275v3) [paper-pdf](http://arxiv.org/pdf/2210.14275v3)

**Authors**: Wenchuan Mu

**Abstract**: Capturing the similarities between human language units is crucial for explaining how humans associate different objects, and therefore its computation has received extensive attention, research, and applications. With the ever-increasing amount of information around us, calculating similarity becomes increasingly complex, especially in many cases, such as legal or medical affairs, measuring similarity requires extra care and precision, as small acts within a language unit can have significant real-world effects. My research goal in this thesis is to develop regression models that account for similarities between language units in a more refined way.   Computation of similarity has come a long way, but approaches to debugging the measures are often based on continually fitting human judgment values. To this end, my goal is to develop an algorithm that precisely catches loopholes in a similarity calculation. Furthermore, most methods have vague definitions of the similarities they compute and are often difficult to interpret. The proposed framework addresses both shortcomings. It constantly improves the model through catching different loopholes. In addition, every refinement of the model provides a reasonable explanation. The regression model introduced in this thesis is called progressively refined similarity computation, which combines attack testing with adversarial training. The similarity regression model of this thesis achieves state-of-the-art performance in handling edge cases.



## **29. EverTracer: Hunting Stolen Large Language Models via Stealthy and Robust Probabilistic Fingerprint**

cs.CR

Accepted by EMNLP2025 Main

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03058v1) [paper-pdf](http://arxiv.org/pdf/2509.03058v1)

**Authors**: Zhenhua Xu, Meng Han, Wenpeng Xing

**Abstract**: The proliferation of large language models (LLMs) has intensified concerns over model theft and license violations, necessitating robust and stealthy ownership verification. Existing fingerprinting methods either require impractical white-box access or introduce detectable statistical anomalies. We propose EverTracer, a novel gray-box fingerprinting framework that ensures stealthy and robust model provenance tracing. EverTracer is the first to repurpose Membership Inference Attacks (MIAs) for defensive use, embedding ownership signals via memorization instead of artificial trigger-output overfitting. It consists of Fingerprint Injection, which fine-tunes the model on any natural language data without detectable artifacts, and Verification, which leverages calibrated probability variation signal to distinguish fingerprinted models. This approach remains robust against adaptive adversaries, including input level modification, and model-level modifications. Extensive experiments across architectures demonstrate EverTracer's state-of-the-art effectiveness, stealthness, and resilience, establishing it as a practical solution for securing LLM intellectual property. Our code and data are publicly available at https://github.com/Xuzhenhua55/EverTracer.



## **30. When and Where do Data Poisons Attack Textual Inversion?**

cs.CR

Accepted to ICCV 2025

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2507.10578v4) [paper-pdf](http://arxiv.org/pdf/2507.10578v4)

**Authors**: Jeremy Styborski, Mingzhi Lyu, Jiayou Lu, Nupur Kapur, Adams Kong

**Abstract**: Poisoning attacks pose significant challenges to the robustness of diffusion models (DMs). In this paper, we systematically analyze when and where poisoning attacks textual inversion (TI), a widely used personalization technique for DMs. We first introduce Semantic Sensitivity Maps, a novel method for visualizing the influence of poisoning on text embeddings. Second, we identify and experimentally verify that DMs exhibit non-uniform learning behavior across timesteps, focusing on lower-noise samples. Poisoning attacks inherit this bias and inject adversarial signals predominantly at lower timesteps. Lastly, we observe that adversarial signals distract learning away from relevant concept regions within training data, corrupting the TI process. Based on these insights, we propose Safe-Zone Training (SZT), a novel defense mechanism comprised of 3 key components: (1) JPEG compression to weaken high-frequency poison signals, (2) restriction to high timesteps during TI training to avoid adversarial signals at lower timesteps, and (3) loss masking to constrain learning to relevant regions. Extensive experiments across multiple poisoning methods demonstrate that SZT greatly enhances the robustness of TI against all poisoning attacks, improving generative quality beyond prior published defenses. Code: www.github.com/JStyborski/Diff_Lab Data: www.github.com/JStyborski/NC10



## **31. Network-Level Prompt and Trait Leakage in Local Research Agents**

cs.CR

Code available at https://github.com/umass-aisec/wra

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2508.20282v2) [paper-pdf](http://arxiv.org/pdf/2508.20282v2)

**Authors**: Hyejun Jeong, Mohammadreza Teymoorianfard, Abhinav Kumar, Amir Houmansadr, Eugene Bagdasarian

**Abstract**: We show that Web and Research Agents (WRAs) -- language model-based systems that investigate complex topics on the Internet -- are vulnerable to inference attacks by passive network adversaries such as ISPs. These agents could be deployed locally by organizations and individuals for privacy, legal, or financial purposes. Unlike sporadic web browsing by humans, WRAs visit $70{-}140$ domains with distinguishable timing correlations, enabling unique fingerprinting attacks.   Specifically, we demonstrate a novel prompt and user trait leakage attack against WRAs that only leverages their network-level metadata (i.e., visited IP addresses and their timings). We start by building a new dataset of WRA traces based on user search queries and queries generated by synthetic personas. We define a behavioral metric (called OBELS) to comprehensively assess similarity between original and inferred prompts, showing that our attack recovers over 73% of the functional and domain knowledge of user prompts. Extending to a multi-session setting, we recover up to 19 of 32 latent traits with high accuracy. Our attack remains effective under partial observability and noisy conditions. Finally, we discuss mitigation strategies that constrain domain diversity or obfuscate traces, showing negligible utility impact while reducing attack effectiveness by an average of 29%.



## **32. See No Evil: Adversarial Attacks Against Linguistic-Visual Association in Referring Multi-Object Tracking Systems**

cs.CV

12 pages, 1 figure, 3 tables

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.02028v2) [paper-pdf](http://arxiv.org/pdf/2509.02028v2)

**Authors**: Halima Bouzidi, Haoyu Liu, Mohammad Abdullah Al Faruque

**Abstract**: Language-vision understanding has driven the development of advanced perception systems, most notably the emerging paradigm of Referring Multi-Object Tracking (RMOT). By leveraging natural-language queries, RMOT systems can selectively track objects that satisfy a given semantic description, guided through Transformer-based spatial-temporal reasoning modules. End-to-End (E2E) RMOT models further unify feature extraction, temporal memory, and spatial reasoning within a Transformer backbone, enabling long-range spatial-temporal modeling over fused textual-visual representations. Despite these advances, the reliability and robustness of RMOT remain underexplored. In this paper, we examine the security implications of RMOT systems from a design-logic perspective, identifying adversarial vulnerabilities that compromise both the linguistic-visual referring and track-object matching components. Additionally, we uncover a novel vulnerability in advanced RMOT models employing FIFO-based memory, whereby targeted and consistent attacks on their spatial-temporal reasoning introduce errors that persist within the history buffer over multiple subsequent frames. We present VEIL, a novel adversarial framework designed to disrupt the unified referring-matching mechanisms of RMOT models. We show that carefully crafted digital and physical perturbations can corrupt the tracking logic reliability, inducing track ID switches and terminations. We conduct comprehensive evaluations using the Refer-KITTI dataset to validate the effectiveness of VEIL and demonstrate the urgent need for security-aware RMOT designs for critical large-scale applications.



## **33. Near-Optimal Stability for Distributed Transaction Processing in Blockchain Sharding**

cs.DC

13 pages, 1 figure, accepted for publication in Proceedings of the  27th International Symposium on Stabilization, Safety, and Security of  Distributed Systems (SSS 2025)

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02421v1) [paper-pdf](http://arxiv.org/pdf/2509.02421v1)

**Authors**: Ramesh Adhikari, Costas Busch, Dariusz R. Kowalski

**Abstract**: In blockchain sharding, $n$ processing nodes are divided into $s$ shards, and each shard processes transactions in parallel. A key challenge in such a system is to ensure system stability for any ``tractable'' pattern of generated transactions; this is modeled by an adversary generating transactions with a certain rate of at most $\rho$ and burstiness $b$. This model captures worst-case scenarios and even some attacks on transactions' processing, e.g., DoS. A stable system ensures bounded transaction queue sizes and bounded transaction latency. It is known that the absolute upper bound on the maximum injection rate for which any scheduler could guarantee bounded queues and latency of transactions is $\max\left\{ \frac{2}{k+1}, \frac{2}{ \left\lfloor\sqrt{2s}\right\rfloor}\right\}$, where $k$ is the maximum number of shards that each transaction accesses. Here, we first provide a single leader scheduler that guarantees stability under injection rate $\rho \leq \max\left\{ \frac{1}{16k}, \frac{1}{16\lceil \sqrt{s} \rceil}\right\}$. Moreover, we also give a distributed scheduler with multiple leaders that guarantees stability under injection rate $\rho \leq \frac{1}{16c_1 \log D \log s}\max\left\{ \frac{1}{k}, \frac{1}{\lceil \sqrt{s} \rceil} \right\}$, where $c_1$ is some positive constant and $D$ is the diameter of shard graph $G_s$. This bound is within a poly-log factor from the optimal injection rate, and significantly improves the best previous known result for the distributed setting by Adhikari et al., SPAA 2024.



## **34. A Survey: Towards Privacy and Security in Mobile Large Language Models**

cs.CR

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02411v1) [paper-pdf](http://arxiv.org/pdf/2509.02411v1)

**Authors**: Honghui Xu, Kaiyang Li, Wei Chen, Danyang Zheng, Zhiyuan Li, Zhipeng Cai

**Abstract**: Mobile Large Language Models (LLMs) are revolutionizing diverse fields such as healthcare, finance, and education with their ability to perform advanced natural language processing tasks on-the-go. However, the deployment of these models in mobile and edge environments introduces significant challenges related to privacy and security due to their resource-intensive nature and the sensitivity of the data they process. This survey provides a comprehensive overview of privacy and security issues associated with mobile LLMs, systematically categorizing existing solutions such as differential privacy, federated learning, and prompt encryption. Furthermore, we analyze vulnerabilities unique to mobile LLMs, including adversarial attacks, membership inference, and side-channel attacks, offering an in-depth comparison of their effectiveness and limitations. Despite recent advancements, mobile LLMs face unique hurdles in achieving robust security while maintaining efficiency in resource-constrained environments. To bridge this gap, we propose potential applications, discuss open challenges, and suggest future research directions, paving the way for the development of trustworthy, privacy-compliant, and scalable mobile LLM systems.



## **35. Enhancing Security in Multi-Robot Systems through Co-Observation Planning, Reachability Analysis, and Network Flow**

cs.RO

12 pages, 6 figures, submitted to IEEE Transactions on Control of  Network Systems

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2403.13266v2) [paper-pdf](http://arxiv.org/pdf/2403.13266v2)

**Authors**: Ziqi Yang, Roberto Tron

**Abstract**: This paper addresses security challenges in multi-robot systems (MRS) where adversaries may compromise robot control, risking unauthorized access to forbidden areas. We propose a novel multi-robot optimal planning algorithm that integrates mutual observations and introduces reachability constraints for enhanced security. This ensures that, even with adversarial movements, compromised robots cannot breach forbidden regions without missing scheduled co-observations. The reachability constraint uses ellipsoidal over-approximation for efficient intersection checking and gradient computation. To enhance system resilience and tackle feasibility challenges, we also introduce sub-teams. These cohesive units replace individual robot assignments along each route, enabling redundant robots to deviate for co-observations across different trajectories, securing multiple sub-teams without requiring modifications. We formulate the cross-trajectory co-observation plan by solving a network flow coverage problem on the checkpoint graph generated from the original unsecured MRS trajectories, providing the same security guarantees against plan-deviation attacks. We demonstrate the effectiveness and robustness of our proposed algorithm, which significantly strengthens the security of multi-robot systems in the face of adversarial threats.



## **36. Enhancing Reliability in LLM-Integrated Robotic Systems: A Unified Approach to Security and Safety**

cs.RO

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02163v1) [paper-pdf](http://arxiv.org/pdf/2509.02163v1)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Bräunl, Jin B. Hong

**Abstract**: Integrating large language models (LLMs) into robotic systems has revolutionised embodied artificial intelligence, enabling advanced decision-making and adaptability. However, ensuring reliability, encompassing both security against adversarial attacks and safety in complex environments, remains a critical challenge. To address this, we propose a unified framework that mitigates prompt injection attacks while enforcing operational safety through robust validation mechanisms. Our approach combines prompt assembling, state management, and safety validation, evaluated using both performance and security metrics. Experiments show a 30.8% improvement under injection attacks and up to a 325% improvement in complex environment settings under adversarial conditions compared to baseline scenarios. This work bridges the gap between safety and security in LLM-based robotic systems, offering actionable insights for deploying reliable LLM-integrated mobile robots in real-world settings. The framework is open-sourced with simulation and physical deployment demos at https://llmeyesim.vercel.app/



## **37. Targeted Physical Evasion Attacks in the Near-Infrared Domain**

cs.CR

To appear in the Proceedings of the Network and Distributed Systems  Security Symposium (NDSS) 2026

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02042v1) [paper-pdf](http://arxiv.org/pdf/2509.02042v1)

**Authors**: Pascal Zimmer, Simon Lachnit, Alexander Jan Zielinski, Ghassan Karame

**Abstract**: A number of attacks rely on infrared light sources or heat-absorbing material to imperceptibly fool systems into misinterpreting visual input in various image recognition applications. However, almost all existing approaches can only mount untargeted attacks and require heavy optimizations due to the use-case-specific constraints, such as location and shape. In this paper, we propose a novel, stealthy, and cost-effective attack to generate both targeted and untargeted adversarial infrared perturbations. By projecting perturbations from a transparent film onto the target object with an off-the-shelf infrared flashlight, our approach is the first to reliably mount laser-free targeted attacks in the infrared domain. Extensive experiments on traffic signs in the digital and physical domains show that our approach is robust and yields higher attack success rates in various attack scenarios across bright lighting conditions, distances, and angles compared to prior work. Equally important, our attack is highly cost-effective, requiring less than US\$50 and a few tens of seconds for deployment. Finally, we propose a novel segmentation-based detection that thwarts our attack with an F1-score of up to 99%.



## **38. Adversarial Attacks and Defenses in Multivariate Time-Series Forecasting for Smart and Connected Infrastructures**

cs.LG

18 pages, 34 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2408.14875v2) [paper-pdf](http://arxiv.org/pdf/2408.14875v2)

**Authors**: Pooja Krishan, Rohan Mohapatra, Sanchari Das, Saptarshi Sengupta

**Abstract**: The emergence of deep learning models has revolutionized various industries over the last decade, leading to a surge in connected devices and infrastructures. However, these models can be tricked into making incorrect predictions with high confidence, leading to disastrous failures and security concerns. To this end, we explore the impact of adversarial attacks on multivariate time-series forecasting and investigate methods to counter them. Specifically, we employ untargeted white-box attacks, namely the Fast Gradient Sign Method (FGSM) and the Basic Iterative Method (BIM), to poison the inputs to the training process, effectively misleading the model. We also illustrate the subtle modifications to the inputs after the attack, which makes detecting the attack using the naked eye quite difficult. Having demonstrated the feasibility of these attacks, we develop robust models through adversarial training and model hardening. We are among the first to showcase the transferability of these attacks and defenses by extrapolating our work from the benchmark electricity data to a larger, 10-year real-world data used for predicting the time-to-failure of hard disks. Our experimental results confirm that the attacks and defenses achieve the desired security thresholds, leading to a 72.41% and 94.81% decrease in RMSE for the electricity and hard disk datasets respectively after implementing the adversarial defenses.



## **39. Evaluating the Defense Potential of Machine Unlearning against Membership Inference Attacks**

cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2508.16150v2) [paper-pdf](http://arxiv.org/pdf/2508.16150v2)

**Authors**: Aristeidis Sidiropoulos, Christos Chrysanthos Nikolaidis, Theodoros Tsiolakis, Nikolaos Pavlidis, Vasilis Perifanis, Pavlos S. Efraimidis

**Abstract**: Membership Inference Attacks (MIAs) pose a significant privacy risk, as they enable adversaries to determine whether a specific data point was included in the training dataset of a model. While Machine Unlearning is primarily designed as a privacy mechanism to efficiently remove private data from a machine learning model without the need for full retraining, its impact on the susceptibility of models to MIA remains an open question. In this study, we systematically assess the vulnerability of models to MIA after applying state-of-art Machine Unlearning algorithms. Our analysis spans four diverse datasets (two from the image domain and two in tabular format), exploring how different unlearning approaches influence the exposure of models to membership inference. The findings highlight that while Machine Unlearning is not inherently a countermeasure against MIA, the unlearning algorithm and data characteristics can significantly affect a model's vulnerability. This work provides essential insights into the interplay between Machine Unlearning and MIAs, offering guidance for the design of privacy-preserving machine learning systems.



## **40. Addressing Key Challenges of Adversarial Attacks and Defenses in the Tabular Domain: A Methodological Framework for Coherence and Consistency**

cs.LG

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2412.07326v3) [paper-pdf](http://arxiv.org/pdf/2412.07326v3)

**Authors**: Yael Itzhakev, Amit Giloni, Yuval Elovici, Asaf Shabtai

**Abstract**: Machine learning models trained on tabular data are vulnerable to adversarial attacks, even in realistic scenarios where attackers only have access to the model's outputs. Since tabular data contains complex interdependencies among features, it presents a unique challenge for adversarial samples which must maintain coherence and respect these interdependencies to remain indistinguishable from benign data. Moreover, existing attack evaluation metrics-such as the success rate, perturbation magnitude, and query count-fail to account for this challenge. To address those gaps, we propose a technique for perturbing dependent features while preserving sample coherence. In addition, we introduce Class-Specific Anomaly Detection (CSAD), an effective novel anomaly detection approach, along with concrete metrics for assessing the quality of tabular adversarial attacks. CSAD evaluates adversarial samples relative to their predicted class distribution, rather than a broad benign distribution. It ensures that subtle adversarial perturbations, which may appear coherent in other classes, are correctly identified as anomalies. We integrate SHAP explainability techniques to detect inconsistencies in model decision-making, extending CSAD for SHAP-based anomaly detection. Our evaluation incorporates both anomaly detection rates with SHAP-based assessments to provide a more comprehensive measure of adversarial sample quality. We evaluate various attack strategies, examining black-box query-based and transferability-based gradient attacks across four target models. Experiments on benchmark tabular datasets reveal key differences in the attacker's risk and effort and attack quality, offering insights into the strengths, limitations, and trade-offs faced by attackers and defenders. Our findings lay the groundwork for future research on adversarial attacks and defense development in the tabular domain.



## **41. SoK: Cybersecurity Assessment of Humanoid Ecosystem**

cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2508.17481v2) [paper-pdf](http://arxiv.org/pdf/2508.17481v2)

**Authors**: Priyanka Prakash Surve, Asaf Shabtai, Yuval Elovici

**Abstract**: Humanoids are progressing toward practical deployment across healthcare, industrial, defense, and service sectors. While typically considered cyber-physical systems (CPSs), their dependence on traditional networked software stacks (e.g., Linux operating systems), robot operating system (ROS) middleware, and over-the-air update channels, creates a distinct security profile that exposes them to vulnerabilities conventional CPS models do not fully address. Prior studies have mainly examined specific threats, such as LiDAR spoofing or adversarial machine learning (AML). This narrow focus overlooks how an attack targeting one component can cascade harm throughout the robot's interconnected systems. We address this gap through a systematization of knowledge (SoK) that takes a comprehensive approach, consolidating fragmented research from robotics, CPS, and network security domains. We introduce a seven-layer security model for humanoid robots, organizing 39 known attacks and 35 defenses across the humanoid ecosystem-from hardware to human-robot interaction. Building on this security model, we develop a quantitative 39x35 attack-defense matrix with risk-weighted scoring, validated through Monte Carlo analysis. We demonstrate our method by evaluating three real-world robots: Pepper, G1 EDU, and Digit. The scoring analysis revealed varying security maturity levels, with scores ranging from 39.9% to 79.5% across the platforms. This work introduces a structured, evidence-based assessment method that enables systematic security evaluation, supports cross-platform benchmarking, and guides prioritization of security investments in humanoid robotics.



## **42. Privacy-preserving authentication for military 5G networks**

cs.CR

To appear in Proc. IEEE Military Commun. Conf. (MILCOM), (Los  Angeles, CA), Oct. 2025

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01470v1) [paper-pdf](http://arxiv.org/pdf/2509.01470v1)

**Authors**: I. D. Lutz, A. M. Hill, M. C. Valenti

**Abstract**: As 5G networks gain traction in defense applications, ensuring the privacy and integrity of the Authentication and Key Agreement (AKA) protocol is critical. While 5G AKA improves upon previous generations by concealing subscriber identities, it remains vulnerable to replay-based synchronization and linkability threats under realistic adversary models. This paper provides a unified analysis of the standardized 5G AKA flow, identifying several vulnerabilities and highlighting how each exploits protocol behavior to compromise user privacy. To address these risks, we present five lightweight mitigation strategies. We demonstrate through prototype implementation and testing that these enhancements strengthen resilience against linkability attacks with minimal computational and signaling overhead. Among the solutions studied, those introducing a UE-generated nonce emerge as the most promising, effectively neutralizing the identified tracking and correlation attacks with negligible additional overhead. Integrating this extension as an optional feature to the standard 5G AKA protocol offers a backward-compatible, low-overhead path toward a more privacy-preserving authentication framework for both commercial and military 5G deployments.



## **43. LLMHoney: A Real-Time SSH Honeypot with Large Language Model-Driven Dynamic Response Generation**

cs.CR

7 Pages

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01463v1) [paper-pdf](http://arxiv.org/pdf/2509.01463v1)

**Authors**: Pranjay Malhotra

**Abstract**: Cybersecurity honeypots are deception tools for engaging attackers and gather intelligence, but traditional low or medium-interaction honeypots often rely on static, pre-scripted interactions that can be easily identified by skilled adversaries. This Report presents LLMHoney, an SSH honeypot that leverages Large Language Models (LLMs) to generate realistic, dynamic command outputs in real time. LLMHoney integrates a dictionary-based virtual file system to handle common commands with low latency while using LLMs for novel inputs, achieving a balance between authenticity and performance. We implemented LLMHoney using open-source LLMs and evaluated it on a testbed with 138 representative Linux commands. We report comprehensive metrics including accuracy (exact-match, Cosine Similarity, Jaro-Winkler Similarity, Levenshtein Similarity and BLEU score), response latency and memory overhead. We evaluate LLMHoney using multiple LLM backends ranging from 0.36B to 3.8B parameters, including both open-source models and a proprietary model(Gemini). Our experiments compare 13 different LLM variants; results show that Gemini-2.0 and moderately-sized models Qwen2.5:1.5B and Phi3:3.8B provide the most reliable and accurate responses, with mean latencies around 3 seconds, whereas smaller models often produce incorrect or out-of-character outputs. We also discuss how LLM integration improves honeypot realism and adaptability compared to traditional honeypots, as well as challenges such as occasional hallucinated outputs and increased resource usage. Our findings demonstrate that LLM-driven honeypots are a promising approach to enhance attacker engagement and collect richer threat intelligence.



## **44. An Automated Attack Investigation Approach Leveraging Threat-Knowledge-Augmented Large Language Models**

cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01271v1) [paper-pdf](http://arxiv.org/pdf/2509.01271v1)

**Authors**: Rujie Dai, Peizhuo Lv, Yujiang Gui, Qiujian Lv, Yuanyuan Qiao, Yan Wang, Degang Sun, Weiqing Huang, Yingjiu Li, XiaoFeng Wang

**Abstract**: Advanced Persistent Threats (APTs) are prolonged, stealthy intrusions by skilled adversaries that compromise high-value systems to steal data or disrupt operations. Reconstructing complete attack chains from massive, heterogeneous logs is essential for effective attack investigation, yet existing methods suffer from poor platform generality, limited generalization to evolving tactics, and an inability to produce analyst-ready reports. Large Language Models (LLMs) offer strong semantic understanding and summarization capabilities, but in this domain they struggle to capture the long-range, cross-log dependencies critical for accurate reconstruction.   To solve these problems, we present an LLM-empowered attack investigation framework augmented with a dynamically adaptable Kill-Chain-aligned threat knowledge base. We organizes attack-relevant behaviors into stage-aware knowledge units enriched with semantic annotations, enabling the LLM to iteratively retrieve relevant intelligence, perform causal reasoning, and progressively expand the investigation context. This process reconstructs multi-phase attack scenarios and generates coherent, human-readable investigation reports. Evaluated on 15 attack scenarios spanning single-host and multi-host environments across Windows and Linux (over 4.3M log events, 7.2 GB of data), the system achieves an average True Positive Rate (TPR) of 97.1% and an average False Positive Rate (FPR) of 0.2%, significantly outperforming the SOTA method ATLAS, which achieves an average TPR of 79.2% and an average FPR of 29.1%.



## **45. Geometric origin of adversarial vulnerability in deep learning**

cs.LG

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01235v1) [paper-pdf](http://arxiv.org/pdf/2509.01235v1)

**Authors**: Yixiong Ren, Wenkang Du, Jianhui Zhou, Haiping Huang

**Abstract**: How to balance training accuracy and adversarial robustness has become a challenge since the birth of deep learning. Here, we introduce a geometry-aware deep learning framework that leverages layer-wise local training to sculpt the internal representations of deep neural networks. This framework promotes intra-class compactness and inter-class separation in feature space, leading to manifold smoothness and adversarial robustness against white or black box attacks. The performance can be explained by an energy model with Hebbian coupling between elements of the hidden representation. Our results thus shed light on the physics of learning in the direction of alignment between biological and artificial intelligence systems. Using the current framework, the deep network can assimilate new information into existing knowledge structures while reducing representation interference.



## **46. Crosstalk Attacks and Defence in a Shared Quantum Computing Environment**

quant-ph

13 pages, 7 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2402.02753v2) [paper-pdf](http://arxiv.org/pdf/2402.02753v2)

**Authors**: Benjamin Harper, Behnam Tonekaboni, Bahar Goldozian, Martin Sevior, Muhammad Usman

**Abstract**: Quantum computing has the potential to provide solutions to problems that are intractable on classical computers, but the accuracy of the current generation of quantum computers suffer from the impact of noise or errors such as leakage, crosstalk, dephasing, and amplitude damping among others. As the access to quantum computers is almost exclusively in a shared environment through cloud-based services, it is possible that an adversary can exploit crosstalk noise to disrupt quantum computations on nearby qubits, even carefully designing quantum circuits to purposely lead to wrong answers. In this paper, we analyze the extent and characteristics of crosstalk noise through tomography conducted on IBM Quantum computers, leading to an enhanced crosstalk simulation model. Our results indicate that crosstalk noise is a significant source of errors on IBM quantum hardware, making crosstalk based attack a viable threat to quantum computing in a shared environment. Based on our crosstalk simulator benchmarked against IBM hardware, we assess the impact of crosstalk attacks and develop strategies for mitigating crosstalk effects. Through a systematic set of simulations, we assess the effectiveness of three crosstalk attack mitigation strategies, namely circuit separation, qubit allocation optimization via reinforcement learning, and the use of spectator qubits, and show that they all overcome crosstalk attacks with varying degrees of success and help to secure quantum computing in a shared platform.



## **47. Clone What You Can't Steal: Black-Box LLM Replication via Logit Leakage and Distillation**

cs.CR

8 pages. Accepted for publication in the proceedings of 7th IEEE  International Conference on Trust, Privacy and Security in Intelligent  Systems, and Applications (IEEE TPS 2025)

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00973v1) [paper-pdf](http://arxiv.org/pdf/2509.00973v1)

**Authors**: Kanchon Gharami, Hansaka Aluvihare, Shafika Showkat Moni, Berker Peköz

**Abstract**: Large Language Models (LLMs) are increasingly deployed in mission-critical systems, facilitating tasks such as satellite operations, command-and-control, military decision support, and cyber defense. Many of these systems are accessed through application programming interfaces (APIs). When such APIs lack robust access controls, they can expose full or top-k logits, creating a significant and often overlooked attack surface. Prior art has mainly focused on reconstructing the output projection layer or distilling surface-level behaviors. However, regenerating a black-box model under tight query constraints remains underexplored. We address that gap by introducing a constrained replication pipeline that transforms partial logit leakage into a functional deployable substitute model clone. Our two-stage approach (i) reconstructs the output projection matrix by collecting top-k logits from under 10k black-box queries via singular value decomposition (SVD) over the logits, then (ii) distills the remaining architecture into compact student models with varying transformer depths, trained on an open source dataset. A 6-layer student recreates 97.6% of the 6-layer teacher model's hidden-state geometry, with only a 7.31% perplexity increase, and a 7.58 Negative Log-Likelihood (NLL). A 4-layer variant achieves 17.1% faster inference and 18.1% parameter reduction with comparable performance. The entire attack completes in under 24 graphics processing unit (GPU) hours and avoids triggering API rate-limit defenses. These results demonstrate how quickly a cost-limited adversary can clone an LLM, underscoring the urgent need for hardened inference APIs and secure on-premise defense deployments.



## **48. FusionCounting: Robust visible-infrared image fusion guided by crowd counting via multi-task learning**

cs.CV

11 pages, 9 figures

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2508.20817v2) [paper-pdf](http://arxiv.org/pdf/2508.20817v2)

**Authors**: He Li, Xinyu Liu, Weihang Kong, Xingchen Zhang

**Abstract**: Visible and infrared image fusion (VIF) is an important multimedia task in computer vision. Most VIF methods focus primarily on optimizing fused image quality. Recent studies have begun incorporating downstream tasks, such as semantic segmentation and object detection, to provide semantic guidance for VIF. However, semantic segmentation requires extensive annotations, while object detection, despite reducing annotation efforts compared with segmentation, faces challenges in highly crowded scenes due to overlapping bounding boxes and occlusion. Moreover, although RGB-T crowd counting has gained increasing attention in recent years, no studies have integrated VIF and crowd counting into a unified framework. To address these challenges, we propose FusionCounting, a novel multi-task learning framework that integrates crowd counting into the VIF process. Crowd counting provides a direct quantitative measure of population density with minimal annotation, making it particularly suitable for dense scenes. Our framework leverages both input images and population density information in a mutually beneficial multi-task design. To accelerate convergence and balance tasks contributions, we introduce a dynamic loss function weighting strategy. Furthermore, we incorporate adversarial training to enhance the robustness of both VIF and crowd counting, improving the model's stability and resilience to adversarial attacks. Experimental results on public datasets demonstrate that FusionCounting not only enhances image fusion quality but also achieves superior crowd counting performance.



## **49. Redesigning Traffic Signs to Mitigate Machine-Learning Patch Attacks**

cs.CR

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2402.04660v3) [paper-pdf](http://arxiv.org/pdf/2402.04660v3)

**Authors**: Tsufit Shua, Liron David, Mahmood Sharif

**Abstract**: Traffic-Sign Recognition (TSR) is a critical safety component for autonomous driving. Unfortunately, however, past work has highlighted the vulnerability of TSR models to physical-world attacks, through low-cost, easily deployable adversarial patches leading to misclassification. To mitigate these threats, most defenses focus on altering the training process or modifying the inference procedure. Still, while these approaches improve adversarial robustness, TSR remains susceptible to attacks attaining substantial success rates.   To further the adversarial robustness of TSR, this work offers a novel approach that redefines traffic-sign designs to create signs that promote robustness while remaining interpretable to humans. Our framework takes three inputs: (1) A traffic-sign standard along with modifiable features and associated constraints; (2) A state-of-the-art adversarial training method; and (3) A function for efficiently synthesizing realistic traffic-sign images. Using these user-defined inputs, the framework emits an optimized traffic-sign standard such that traffic signs generated per this standard enable training TSR models with increased adversarial robustness.   We evaluate the effectiveness of our framework via a concrete implementation, where we allow modifying the pictograms (i.e., symbols) and colors of traffic signs. The results show substantial improvements in robustness -- with gains of up to 16.33%--24.58% in robust accuracy over state-of-the-art methods -- while benign accuracy is even improved. Importantly, a user study also confirms that the redesigned traffic signs remain easily recognizable and to human observers. Overall, the results highlight that carefully redesigning traffic signs can significantly enhance TSR system robustness without compromising human interpretability.



## **50. Sequential Difference Maximization: Generating Adversarial Examples via Multi-Stage Optimization**

cs.CV

5 pages, 2 figures, 5 tables, CIKM 2025

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00826v1) [paper-pdf](http://arxiv.org/pdf/2509.00826v1)

**Authors**: Xinlei Liu, Tao Hu, Peng Yi, Weitao Han, Jichao Xie, Baolin Li

**Abstract**: Efficient adversarial attack methods are critical for assessing the robustness of computer vision models. In this paper, we reconstruct the optimization objective for generating adversarial examples as "maximizing the difference between the non-true labels' probability upper bound and the true label's probability," and propose a gradient-based attack method termed Sequential Difference Maximization (SDM). SDM establishes a three-layer optimization framework of "cycle-stage-step." The processes between cycles and between iterative steps are respectively identical, while optimization stages differ in terms of loss functions: in the initial stage, the negative probability of the true label is used as the loss function to compress the solution space; in subsequent stages, we introduce the Directional Probability Difference Ratio (DPDR) loss function to gradually increase the non-true labels' probability upper bound by compressing the irrelevant labels' probabilities. Experiments demonstrate that compared with previous SOTA methods, SDM not only exhibits stronger attack performance but also achieves higher attack cost-effectiveness. Additionally, SDM can be combined with adversarial training methods to enhance their defensive effects. The code is available at https://github.com/X-L-Liu/SDM.



