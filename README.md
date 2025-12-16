# Latest Adversarial Attack Papers
**update at 2025-12-16 18:37:38**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. REVERB-FL: Server-Side Adversarial and Reserve-Enhanced Federated Learning for Robust Audio Classification**

eess.AS

13 pages, 4 figures

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13647v1) [paper-pdf](https://arxiv.org/pdf/2512.13647v1)

**Authors**: Sathwika Peechara, Rajeev Sahay

**Abstract**: Federated learning (FL) enables a privacy-preserving training paradigm for audio classification but is highly sensitive to client heterogeneity and poisoning attacks, where adversarially compromised clients can bias the global model and hinder the performance of audio classifiers. To mitigate the effects of model poisoning for audio signal classification, we present REVERB-FL, a lightweight, server-side defense that couples a small reserve set (approximately 5%) with pre- and post-aggregation retraining and adversarial training. After each local training round, the server refines the global model on the reserve set with either clean or additional adversarially perturbed data, thereby counteracting non-IID drift and mitigating potential model poisoning without adding substantial client-side cost or altering the aggregation process. We theoretically demonstrate the feasibility of our framework, showing faster convergence and a reduced steady-state error relative to baseline federated averaging. We validate our framework on two open-source audio classification datasets with varying IID and Dirichlet non-IID partitions and demonstrate that REVERB-FL mitigates global model poisoning under multiple designs of local data poisoning.



## **2. Async Control: Stress-testing Asynchronous Control Measures for LLM Agents**

cs.LG

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13526v1) [paper-pdf](https://arxiv.org/pdf/2512.13526v1)

**Authors**: Asa Cooper Stickland, Jan Michelfeit, Arathi Mani, Charlie Griffin, Ollie Matthews, Tomek Korbak, Rogan Inglis, Oliver Makins, Alan Cooney

**Abstract**: LLM-based software engineering agents are increasingly used in real-world development tasks, often with access to sensitive data or security-critical codebases. Such agents could intentionally sabotage these codebases if they were misaligned. We investigate asynchronous monitoring, in which a monitoring system reviews agent actions after the fact. Unlike synchronous monitoring, this approach does not impose runtime latency, while still attempting to disrupt attacks before irreversible harm occurs. We treat monitor development as an adversarial game between a blue team (who design monitors) and a red team (who create sabotaging agents). We attempt to set the game rules such that they upper bound the sabotage potential of an agent based on Claude 4.1 Opus. To ground this game in a realistic, high-stakes deployment scenario, we develop a suite of 5 diverse software engineering environments that simulate tasks that an agent might perform within an AI developer's internal infrastructure. Over the course of the game, we develop an ensemble monitor that achieves a 6% false negative rate at 1% false positive rate on a held out test environment. Then, we estimate risk of sabotage at deployment time by extrapolating from our monitor's false negative rate. We describe one simple model for this extrapolation, present a sensitivity analysis, and describe situations in which the model would be invalid. Code is available at: https://github.com/UKGovernmentBEIS/async-control.



## **3. An $H_2$-norm approach to performance analysis of networked control systems under multiplicative routing transformations**

eess.SY

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13504v1) [paper-pdf](https://arxiv.org/pdf/2512.13504v1)

**Authors**: Ruslan Seifullaev, André M. H. Teixeira

**Abstract**: This paper investigates the performance of networked control systems subject to multiplicative routing transformations that alter measurement pathways without directly injecting signals. Such transformations, arising from faults or adversarial actions, modify the feedback structure and can degrade performance while remaining stealthy. An $H_2$-norm framework is proposed to quantify the impact of these transformations by evaluating the ratio between the steady-state energies of performance and residual outputs. Equivalent linear matrix inequality (LMI) formulations are derived for computational assessment, and analytical upper bounds are established to estimate the worst-case degradation. The results provide structural insight into how routing manipulations influence closed-loop behavior and reveal conditions for stealthy multiplicative attacks.



## **4. Behavior-Aware and Generalizable Defense Against Black-Box Adversarial Attacks for ML-Based IDS**

cs.CR

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13501v1) [paper-pdf](https://arxiv.org/pdf/2512.13501v1)

**Authors**: Sabrine Ennaji, Elhadj Benkhelifa, Luigi Vincenzo Mancini

**Abstract**: Machine learning based intrusion detection systems are increasingly targeted by black box adversarial attacks, where attackers craft evasive inputs using indirect feedback such as binary outputs or behavioral signals like response time and resource usage. While several defenses have been proposed, including input transformation, adversarial training, and surrogate detection, they often fall short in practice. Most are tailored to specific attack types, require internal model access, or rely on static mechanisms that fail to generalize across evolving attack strategies. Furthermore, defenses such as input transformation can degrade intrusion detection system performance, making them unsuitable for real time deployment.   To address these limitations, we propose Adaptive Feature Poisoning, a lightweight and proactive defense mechanism designed specifically for realistic black box scenarios. Adaptive Feature Poisoning assumes that probing can occur silently and continuously, and introduces dynamic and context aware perturbations to selected traffic features, corrupting the attacker feedback loop without impacting detection capabilities. The method leverages traffic profiling, change point detection, and adaptive scaling to selectively perturb features that an attacker is likely exploiting, based on observed deviations.   We evaluate Adaptive Feature Poisoning against multiple realistic adversarial attack strategies, including silent probing, transferability based attacks, and decision boundary based attacks. The results demonstrate its ability to confuse attackers, degrade attack effectiveness, and preserve detection performance. By offering a generalizable, attack agnostic, and undetectable defense, Adaptive Feature Poisoning represents a significant step toward practical and robust adversarial resilience in machine learning based intrusion detection systems.



## **5. On the Effectiveness of Membership Inference in Targeted Data Extraction from Large Language Models**

cs.LG

Accepted to IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) 2026

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13352v1) [paper-pdf](https://arxiv.org/pdf/2512.13352v1)

**Authors**: Ali Al Sahili, Ali Chehab, Razane Tajeddine

**Abstract**: Large Language Models (LLMs) are prone to memorizing training data, which poses serious privacy risks. Two of the most prominent concerns are training data extraction and Membership Inference Attacks (MIAs). Prior research has shown that these threats are interconnected: adversaries can extract training data from an LLM by querying the model to generate a large volume of text and subsequently applying MIAs to verify whether a particular data point was included in the training set. In this study, we integrate multiple MIA techniques into the data extraction pipeline to systematically benchmark their effectiveness. We then compare their performance in this integrated setting against results from conventional MIA benchmarks, allowing us to evaluate their practical utility in real-world extraction scenarios.



## **6. Evaluating Adversarial Attacks on Federated Learning for Temperature Forecasting**

cs.LG

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13207v1) [paper-pdf](https://arxiv.org/pdf/2512.13207v1)

**Authors**: Karina Chichifoi, Fabio Merizzi, Michele Colajanni

**Abstract**: Deep learning and federated learning (FL) are becoming powerful partners for next-generation weather forecasting. Deep learning enables high-resolution spatiotemporal forecasts that can surpass traditional numerical models, while FL allows institutions in different locations to collaboratively train models without sharing raw data, addressing efficiency and security concerns. While FL has shown promise across heterogeneous regions, its distributed nature introduces new vulnerabilities. In particular, data poisoning attacks, in which compromised clients inject manipulated training data, can degrade performance or introduce systematic biases. These threats are amplified by spatial dependencies in meteorological data, allowing localized perturbations to influence broader regions through global model aggregation. In this study, we investigate how adversarial clients distort federated surface temperature forecasts trained on the Copernicus European Regional ReAnalysis (CERRA) dataset. We simulate geographically distributed clients and evaluate patch-based and global biasing attacks on regional temperature forecasts. Our results show that even a small fraction of poisoned clients can mislead predictions across large, spatially connected areas. A global temperature bias attack from a single compromised client shifts predictions by up to -1.7 K, while coordinated patch attacks more than triple the mean squared error and produce persistent regional anomalies exceeding +3.5 K. Finally, we assess trimmed mean aggregation as a defense mechanism, showing that it successfully defends against global bias attacks (2-13\% degradation) but fails against patch attacks (281-603\% amplification), exposing limitations of outlier-based defenses for spatially correlated data.



## **7. Less Is More: Sparse and Cooperative Perturbation for Point Cloud Attacks**

cs.CR

Accepted by AAAI'2026 (Oral)

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13119v1) [paper-pdf](https://arxiv.org/pdf/2512.13119v1)

**Authors**: Keke Tang, Tianyu Hao, Xiaofei Wang, Weilong Peng, Denghui Zhang, Peican Zhu, Zhihong Tian

**Abstract**: Most adversarial attacks on point clouds perturb a large number of points, causing widespread geometric changes and limiting applicability in real-world scenarios. While recent works explore sparse attacks by modifying only a few points, such approaches often struggle to maintain effectiveness due to the limited influence of individual perturbations. In this paper, we propose SCP, a sparse and cooperative perturbation framework that selects and leverages a compact subset of points whose joint perturbations produce amplified adversarial effects. Specifically, SCP identifies the subset where the misclassification loss is locally convex with respect to their joint perturbations, determined by checking the positivedefiniteness of the corresponding Hessian block. The selected subset is then optimized to generate high-impact adversarial examples with minimal modifications. Extensive experiments show that SCP achieves 100% attack success rates, surpassing state-of-the-art sparse attacks, and delivers superior imperceptibility to dense attacks with far fewer modifications.



## **8. Calibrating Uncertainty for Zero-Shot Adversarial CLIP**

cs.CV

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12997v1) [paper-pdf](https://arxiv.org/pdf/2512.12997v1)

**Authors**: Wenjing lu, Zerui Tao, Dongping Zhang, Yuning Qiu, Yang Yang, Qibin Zhao

**Abstract**: CLIP delivers strong zero-shot classification but remains highly vulnerable to adversarial attacks. Previous work of adversarial fine-tuning largely focuses on matching the predicted logits between clean and adversarial examples, which overlooks uncertainty calibration and may degrade the zero-shot generalization. A common expectation in reliable uncertainty estimation is that predictive uncertainty should increase as inputs become more difficult or shift away from the training distribution. However, we frequently observe the opposite in the adversarial setting: perturbations not only degrade accuracy but also suppress uncertainty, leading to severe miscalibration and unreliable over-confidence. This overlooked phenomenon highlights a critical reliability gap beyond robustness. To bridge this gap, we propose a novel adversarial fine-tuning objective for CLIP considering both prediction accuracy and uncertainty alignments. By reparameterizing the output of CLIP as the concentration parameter of a Dirichlet distribution, we propose a unified representation that captures relative semantic structure and the magnitude of predictive confidence. Our objective aligns these distributions holistically under perturbations, moving beyond single-logit anchoring and restoring calibrated uncertainty. Experiments on multiple zero-shot classification benchmarks demonstrate that our approach effectively restores calibrated uncertainty and achieves competitive adversarial robustness while maintaining clean accuracy.



## **9. Cisco Integrated AI Security and Safety Framework Report**

cs.CR

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12921v1) [paper-pdf](https://arxiv.org/pdf/2512.12921v1)

**Authors**: Amy Chang, Tiffany Saade, Sanket Mendapara, Adam Swanda, Ankit Garg

**Abstract**: Artificial intelligence (AI) systems are being readily and rapidly adopted, increasingly permeating critical domains: from consumer platforms and enterprise software to networked systems with embedded agents. While this has unlocked potential for human productivity gains, the attack surface has expanded accordingly: threats now span content safety failures (e.g., harmful or deceptive outputs), model and data integrity compromise (e.g., poisoning, supply-chain tampering), runtime manipulations (e.g., prompt injection, tool and agent misuse), and ecosystem risks (e.g., orchestration abuse, multi-agent collusion). Existing frameworks such as MITRE ATLAS, National Institute of Standards and Technology (NIST) AI 100-2 Adversarial Machine Learning (AML) taxonomy, and OWASP Top 10s for Large Language Models (LLMs) and Agentic AI Applications provide valuable viewpoints, but each covers only slices of this multi-dimensional space.   This paper presents Cisco's Integrated AI Security and Safety Framework ("AI Security Framework"), a unified, lifecycle-aware taxonomy and operationalization framework that can be used to classify, integrate, and operationalize the full range of AI risks. It integrates AI security and AI safety across modalities, agents, pipelines, and the broader ecosystem. The AI Security Framework is designed to be practical for threat identification, red-teaming, risk prioritization, and it is comprehensive in scope and can be extensible to emerging deployments in multimodal contexts, humanoids, wearables, and sensory infrastructures. We analyze gaps in prevailing frameworks, discuss design principles for our framework, and demonstrate how the taxonomy provides structure for understanding how modern AI systems fail, how adversaries exploit these failures, and how organizations can build defenses across the AI lifecycle that evolve alongside capability advancements.



## **10. CTIGuardian: A Few-Shot Framework for Mitigating Privacy Leakage in Fine-Tuned LLMs**

cs.CR

Accepted at the 18th Cybersecurity Experimentation and Test Workshop (CSET), in conjunction with ACSAC 2025

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12914v1) [paper-pdf](https://arxiv.org/pdf/2512.12914v1)

**Authors**: Shashie Dilhara Batan Arachchige, Benjamin Zi Hao Zhao, Hassan Jameel Asghar, Dinusha Vatsalan, Dali Kaafar

**Abstract**: Large Language Models (LLMs) are often fine-tuned to adapt their general-purpose knowledge to specific tasks and domains such as cyber threat intelligence (CTI). Fine-tuning is mostly done through proprietary datasets that may contain sensitive information. Owners expect their fine-tuned model to not inadvertently leak this information to potentially adversarial end users. Using CTI as a use case, we demonstrate that data-extraction attacks can recover sensitive information from fine-tuned models on CTI reports, underscoring the need for mitigation. Retraining the full model to eliminate this leakage is computationally expensive and impractical. We propose an alternative approach, which we call privacy alignment, inspired by safety alignment in LLMs. Just like safety alignment teaches the model to abide by safety constraints through a few examples, we enforce privacy alignment through few-shot supervision, integrating a privacy classifier and a privacy redactor, both handled by the same underlying LLM. We evaluate our system, called CTIGuardian, using GPT-4o mini and Mistral-7B Instruct models, benchmarking against Presidio, a named entity recognition (NER) baseline. Results show that CTIGuardian provides a better privacy-utility trade-off than NER based models. While we demonstrate its effectiveness on a CTI use case, the framework is generic enough to be applicable to other sensitive domains.



## **11. PRIVEE: Privacy-Preserving Vertical Federated Learning Against Feature Inference Attacks**

cs.LG

12 pages, 3 figures

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.12840v1) [paper-pdf](https://arxiv.org/pdf/2512.12840v1)

**Authors**: Sindhuja Madabushi, Ahmad Faraz Khan, Haider Ali, Ananthram Swami, Rui Ning, Hongyi Wu, Jin-Hee Cho

**Abstract**: Vertical Federated Learning (VFL) enables collaborative model training across organizations that share common user samples but hold disjoint feature spaces. Despite its potential, VFL is susceptible to feature inference attacks, in which adversarial parties exploit shared confidence scores (i.e., prediction probabilities) during inference to reconstruct private input features of other participants. To counter this threat, we propose PRIVEE (PRIvacy-preserving Vertical fEderated lEarning), a novel defense mechanism named after the French word privée, meaning "private." PRIVEE obfuscates confidence scores while preserving critical properties such as relative ranking and inter-score distances. Rather than exposing raw scores, PRIVEE shares only the transformed representations, mitigating the risk of reconstruction attacks without degrading model prediction accuracy. Extensive experiments show that PRIVEE achieves a threefold improvement in privacy protection compared to state-of-the-art defenses, while preserving full predictive performance against advanced feature inference attacks.



## **12. GradID: Adversarial Detection via Intrinsic Dimensionality of Gradients**

cs.LG

16 pages, 8 figures

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.12827v1) [paper-pdf](https://arxiv.org/pdf/2512.12827v1)

**Authors**: Mohammad Mahdi Razmjoo, Mohammad Mahdi Sharifian, Saeed Bagheri Shouraki

**Abstract**: Despite their remarkable performance, deep neural networks exhibit a critical vulnerability: small, often imperceptible, adversarial perturbations can lead to drastically altered model predictions. Given the stringent reliability demands of applications such as medical diagnosis and autonomous driving, robust detection of such adversarial attacks is paramount. In this paper, we investigate the geometric properties of a model's input loss landscape. We analyze the Intrinsic Dimensionality (ID) of the model's gradient parameters, which quantifies the minimal number of coordinates required to describe the data points on their underlying manifold. We reveal a distinct and consistent difference in the ID for natural and adversarial data, which forms the basis of our proposed detection method. We validate our approach across two distinct operational scenarios. First, in a batch-wise context for identifying malicious data groups, our method demonstrates high efficacy on datasets like MNIST and SVHN. Second, in the critical individual-sample setting, we establish new state-of-the-art results on challenging benchmarks such as CIFAR-10 and MS COCO. Our detector significantly surpasses existing methods against a wide array of attacks, including CW and AutoAttack, achieving detection rates consistently above 92\% on CIFAR-10. The results underscore the robustness of our geometric approach, highlighting that intrinsic dimensionality is a powerful fingerprint for adversarial detection across diverse datasets and attack strategies.



## **13. Spectral Sentinel: Scalable Byzantine-Robust Decentralized Federated Learning via Sketched Random Matrix Theory on Blockchain**

cs.LG

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.12617v1) [paper-pdf](https://arxiv.org/pdf/2512.12617v1)

**Authors**: Animesh Mishra

**Abstract**: Decentralized federated learning (DFL) enables collaborative model training without centralized trust, but it remains vulnerable to Byzantine clients that poison gradients under heterogeneous (Non-IID) data. Existing defenses face a scalability trilemma: distance-based filtering (e.g., Krum) can reject legitimate Non-IID updates, geometric-median methods incur prohibitive $O(n^2 d)$ cost, and many certified defenses are evaluated only on models below 100M parameters. We propose Spectral Sentinel, a Byzantine detection and aggregation framework that leverages a random-matrix-theoretic signature: honest Non-IID gradients produce covariance eigenspectra whose bulk follows the Marchenko-Pastur law, while Byzantine perturbations induce detectable tail anomalies. Our algorithm combines Frequent Directions sketching with data-dependent MP tracking, enabling detection on models up to 1.5B parameters using $O(k^2)$ memory with $k \ll d$. Under a $(σ,f)$ threat model with coordinate-wise honest variance bounded by $σ^2$ and $f < 1/2$ adversaries, we prove $(ε,δ)$-Byzantine resilience with convergence rate $O(σf / \sqrt{T} + f^2 / T)$, and we provide a matching information-theoretic lower bound $Ω(σf / \sqrt{T})$, establishing minimax optimality. We implement the full system with blockchain integration on Polygon networks and validate it across 144 attack-aggregator configurations, achieving 78.4 percent average accuracy versus 48-63 percent for baseline methods.



## **14. Unveiling Malicious Logic: Towards a Statement-Level Taxonomy and Dataset for Securing Python Packages**

cs.CR

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.12559v1) [paper-pdf](https://arxiv.org/pdf/2512.12559v1)

**Authors**: Ahmed Ryan, Junaid Mansur Ifti, Md Erfan, Akond Ashfaque Ur Rahman, Md Rayhanur Rahman

**Abstract**: The widespread adoption of open-source ecosystems enables developers to integrate third-party packages, but also exposes them to malicious packages crafted to execute harmful behavior via public repositories such as PyPI. Existing datasets (e.g., pypi-malregistry, DataDog, OpenSSF, MalwareBench) label packages as malicious or benign at the package level, but do not specify which statements implement malicious behavior. This coarse granularity limits research and practice: models cannot be trained to localize malicious code, detectors cannot justify alerts with code-level evidence, and analysts cannot systematically study recurring malicious indicators or attack chains. To address this gap, we construct a statement-level dataset of 370 malicious Python packages (833 files, 90,527 lines) with 2,962 labeled occurrences of malicious indicators. From these annotations, we derive a fine-grained taxonomy of 47 malicious indicators across 7 types that capture how adversarial behavior is implemented in code, and we apply sequential pattern mining to uncover recurring indicator sequences that characterize common attack workflows. Our contribution enables explainable, behavior-centric detection and supports both semantic-aware model training and practical heuristics for strengthening software supply-chain defenses.



## **15. Keep the Lights On, Keep the Lengths in Check: Plug-In Adversarial Detection for Time-Series LLMs in Energy Forecasting**

cs.CR

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12154v1) [paper-pdf](https://arxiv.org/pdf/2512.12154v1)

**Authors**: Hua Ma, Ruoxi Sun, Minhui Xue, Xingliang Yuan, Carsten Rudolph, Surya Nepal, Ling Liu

**Abstract**: Accurate time-series forecasting is increasingly critical for planning and operations in low-carbon power systems. Emerging time-series large language models (TS-LLMs) now deliver this capability at scale, requiring no task-specific retraining, and are quickly becoming essential components within the Internet-of-Energy (IoE) ecosystem. However, their real-world deployment is complicated by a critical vulnerability: adversarial examples (AEs). Detecting these AEs is challenging because (i) adversarial perturbations are optimized across the entire input sequence and exploit global temporal dependencies, which renders local detection methods ineffective, and (ii) unlike traditional forecasting models with fixed input dimensions, TS-LLMs accept sequences of variable length, increasing variability that complicates detection. To address these challenges, we propose a plug-in detection framework that capitalizes on the TS-LLM's own variable-length input capability. Our method uses sampling-induced divergence as a detection signal. Given an input sequence, we generate multiple shortened variants and detect AEs by measuring the consistency of their forecasts: Benign sequences tend to produce stable predictions under sampling, whereas adversarial sequences show low forecast similarity, because perturbations optimized for a full-length sequence do not transfer reliably to shorter, differently-structured subsamples. We evaluate our approach on three representative TS-LLMs (TimeGPT, TimesFM, and TimeLLM) across three energy datasets: ETTh2 (Electricity Transformer Temperature), NI (Hourly Energy Consumption), and Consumption (Hourly Electricity Consumption and Production). Empirical results confirm strong and robust detection performance across both black-box and white-box attack scenarios, highlighting its practicality as a reliable safeguard for TS-LLM forecasting in real-world energy systems.



## **16. BRIDG-ICS: AI-Grounded Knowledge Graphs for Intelligent Threat Analytics in Industry~5.0 Cyber-Physical Systems**

cs.CR

44 Pages, To be published in Springer Cybersecurity Journal

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12112v1) [paper-pdf](https://arxiv.org/pdf/2512.12112v1)

**Authors**: Padmeswari Nandiya, Ahmad Mohsin, Ahmed Ibrahim, Iqbal H. Sarker, Helge Janicke

**Abstract**: Industry 5.0's increasing integration of IT and OT systems is transforming industrial operations but also expanding the cyber-physical attack surface. Industrial Control Systems (ICS) face escalating security challenges as traditional siloed defences fail to provide coherent, cross-domain threat insights. We present BRIDG-ICS (BRIDge for Industrial Control Systems), an AI-driven Knowledge Graph (KG) framework for context-aware threat analysis and quantitative assessment of cyber resilience in smart manufacturing environments. BRIDG-ICS fuses heterogeneous industrial and cybersecurity data into an integrated Industrial Security Knowledge Graph linking assets, vulnerabilities, and adversarial behaviours with probabilistic risk metrics (e.g. exploit likelihood, attack cost). This unified graph representation enables multi-stage attack path simulation using graph-analytic techniques. To enrich the graph's semantic depth, the framework leverages Large Language Models (LLMs): domain-specific LLMs extract cybersecurity entities, predict relationships, and translate natural-language threat descriptions into structured graph triples, thereby populating the knowledge graph with missing associations and latent risk indicators. This unified AI-enriched KG supports multi-hop, causality-aware threat reasoning, improving visibility into complex attack chains and guiding data-driven mitigation. In simulated industrial scenarios, BRIDG-ICS scales well, reduces potential attack exposure, and can enhance cyber-physical system resilience in Industry 5.0 settings.



## **17. CLOAK: Contrastive Guidance for Latent Diffusion-Based Data Obfuscation**

cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.12086v1) [paper-pdf](https://arxiv.org/pdf/2512.12086v1)

**Authors**: Xin Yang, Omid Ardakanian

**Abstract**: Data obfuscation is a promising technique for mitigating attribute inference attacks by semi-trusted parties with access to time-series data emitted by sensors. Recent advances leverage conditional generative models together with adversarial training or mutual information-based regularization to balance data privacy and utility. However, these methods often require modifying the downstream task, struggle to achieve a satisfactory privacy-utility trade-off, or are computationally intensive, making them impractical for deployment on resource-constrained mobile IoT devices. We propose Cloak, a novel data obfuscation framework based on latent diffusion models. In contrast to prior work, we employ contrastive learning to extract disentangled representations, which guide the latent diffusion process to retain useful information while concealing private information. This approach enables users with diverse privacy needs to navigate the privacy-utility trade-off with minimal retraining. Extensive experiments on four public time-series datasets, spanning multiple sensing modalities, and a dataset of facial images demonstrate that Cloak consistently outperforms state-of-the-art obfuscation techniques and is well-suited for deployment in resource-constrained settings.



## **18. Adversarial Attacks Against Deep Learning-Based Radio Frequency Fingerprint Identification**

cs.CR

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.12002v1) [paper-pdf](https://arxiv.org/pdf/2512.12002v1)

**Authors**: Jie Ma, Junqing Zhang, Guanxiong Shen, Alan Marshall, Chip-Hong Chang

**Abstract**: Radio frequency fingerprint identification (RFFI) is an emerging technique for the lightweight authentication of wireless Internet of things (IoT) devices. RFFI exploits deep learning models to extract hardware impairments to uniquely identify wireless devices. Recent studies show deep learning-based RFFI is vulnerable to adversarial attacks. However, effective adversarial attacks against different types of RFFI classifiers have not yet been explored. In this paper, we carried out a comprehensive investigations into different adversarial attack methods on RFFI systems using various deep learning models. Three specific algorithms, fast gradient sign method (FGSM), projected gradient descent (PGD), and universal adversarial perturbation (UAP), were analyzed. The attacks were launched to LoRa-RFFI and the experimental results showed the generated perturbations were effective against convolutional neural networks (CNNs), long short-term memory (LSTM) networks, and gated recurrent units (GRU). We further used UAP to launch practical attacks. Special factors were considered for the wireless context, including implementing real-time attacks, the effectiveness of the attacks over a period of time, etc. Our experimental evaluation demonstrated that UAP can successfully launch adversarial attacks against the RFFI, achieving a success rate of 81.7% when the adversary almost has no prior knowledge of the victim RFFI systems.



## **19. Super Suffixes: Bypassing Text Generation Alignment and Guard Models Simultaneously**

cs.CR

13 pages, 5 Figures

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11783v1) [paper-pdf](https://arxiv.org/pdf/2512.11783v1)

**Authors**: Andrew Adiletta, Kathryn Adiletta, Kemal Derya, Berk Sunar

**Abstract**: The rapid deployment of Large Language Models (LLMs) has created an urgent need for enhanced security and privacy measures in Machine Learning (ML). LLMs are increasingly being used to process untrusted text inputs and even generate executable code, often while having access to sensitive system controls. To address these security concerns, several companies have introduced guard models, which are smaller, specialized models designed to protect text generation models from adversarial or malicious inputs. In this work, we advance the study of adversarial inputs by introducing Super Suffixes, suffixes capable of overriding multiple alignment objectives across various models with different tokenization schemes. We demonstrate their effectiveness, along with our joint optimization technique, by successfully bypassing the protection mechanisms of Llama Prompt Guard 2 on five different text generation models for malicious text and code generation. To the best of our knowledge, this is the first work to reveal that Llama Prompt Guard 2 can be compromised through joint optimization.   Additionally, by analyzing the changing similarity of a model's internal state to specific concept directions during token sequence processing, we propose an effective and lightweight method to detect Super Suffix attacks. We show that the cosine similarity between the residual stream and certain concept directions serves as a distinctive fingerprint of model intent. Our proposed countermeasure, DeltaGuard, significantly improves the detection of malicious prompts generated through Super Suffixes. It increases the non-benign classification rate to nearly 100%, making DeltaGuard a valuable addition to the guard model stack and enhancing robustness against adversarial prompt attacks.



## **20. Smudged Fingerprints: A Systematic Evaluation of the Robustness of AI Image Fingerprints**

cs.CV

This work has been accepted for publication in the 4th IEEE Conference on Secure and Trustworthy Machine Learning (IEEE SaTML 2026). The final version will be available on IEEE Xplore

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11771v1) [paper-pdf](https://arxiv.org/pdf/2512.11771v1)

**Authors**: Kai Yao, Marc Juarez

**Abstract**: Model fingerprint detection techniques have emerged as a promising approach for attributing AI-generated images to their source models, but their robustness under adversarial conditions remains largely unexplored. We present the first systematic security evaluation of these techniques, formalizing threat models that encompass both white- and black-box access and two attack goals: fingerprint removal, which erases identifying traces to evade attribution, and fingerprint forgery, which seeks to cause misattribution to a target model. We implement five attack strategies and evaluate 14 representative fingerprinting methods across RGB, frequency, and learned-feature domains on 12 state-of-the-art image generators. Our experiments reveal a pronounced gap between clean and adversarial performance. Removal attacks are highly effective, often achieving success rates above 80% in white-box settings and over 50% under constrained black-box access. While forgery is more challenging than removal, its success significantly varies across targeted models. We also identify a utility-robustness trade-off: methods with the highest attribution accuracy are often vulnerable to attacks. Although some techniques exhibit robustness in specific settings, none achieves high robustness and accuracy across all evaluated threat models. These findings highlight the need for techniques balancing robustness and accuracy, and identify the most promising approaches for advancing this goal.



## **21. SpectralKrum: A Spectral-Geometric Defense Against Byzantine Attacks in Federated Learning**

cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11760v1) [paper-pdf](https://arxiv.org/pdf/2512.11760v1)

**Authors**: Aditya Tripathi, Karan Sharma, Rahul Mishra, Tapas Kumar Maiti

**Abstract**: Federated Learning (FL) distributes model training across clients who retain their data locally, but this architecture exposes a fundamental vulnerability: Byzantine clients can inject arbitrarily corrupted updates that degrade or subvert the global model. While robust aggregation methods (including Krum, Bulyan, and coordinate-wise defenses) offer theoretical guarantees under idealized assumptions, their effectiveness erodes substantially when client data distributions are heterogeneous (non-IID) and adversaries can observe or approximate the defense mechanism.   This paper introduces SpectralKrum, a defense that fuses spectral subspace estimation with geometric neighbor-based selection. The core insight is that benign optimization trajectories, despite per-client heterogeneity, concentrate near a low-dimensional manifold that can be estimated from historical aggregates. SpectralKrum projects incoming updates into this learned subspace, applies Krum selection in compressed coordinates, and filters candidates whose orthogonal residual energy exceeds a data-driven threshold. The method requires no auxiliary data, operates entirely on model updates, and preserves FL privacy properties.   We evaluate SpectralKrum against eight robust baselines across seven attack scenarios on CIFAR-10 with Dirichlet-distributed non-IID partitions (alpha = 0.1). Experiments spanning over 56,000 training rounds show that SpectralKrum is competitive against directional and subspace-aware attacks (adaptive-steer, buffer-drift), but offers limited advantage under label-flip and min-max attacks where malicious updates remain spectrally indistinguishable from benign ones.



## **22. CLINIC: Evaluating Multilingual Trustworthiness in Language Models for Healthcare**

cs.CL

49 pages, 31 figures

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11437v1) [paper-pdf](https://arxiv.org/pdf/2512.11437v1)

**Authors**: Akash Ghosh, Srivarshinee Sridhar, Raghav Kaushik Ravi, Muhsin Muhsin, Sriparna Saha, Chirag Agarwal

**Abstract**: Integrating language models (LMs) in healthcare systems holds great promise for improving medical workflows and decision-making. However, a critical barrier to their real-world adoption is the lack of reliable evaluation of their trustworthiness, especially in multilingual healthcare settings. Existing LMs are predominantly trained in high-resource languages, making them ill-equipped to handle the complexity and diversity of healthcare queries in mid- and low-resource languages, posing significant challenges for deploying them in global healthcare contexts where linguistic diversity is key. In this work, we present CLINIC, a Comprehensive Multilingual Benchmark to evaluate the trustworthiness of language models in healthcare. CLINIC systematically benchmarks LMs across five key dimensions of trustworthiness: truthfulness, fairness, safety, robustness, and privacy, operationalized through 18 diverse tasks, spanning 15 languages (covering all the major continents), and encompassing a wide array of critical healthcare topics like disease conditions, preventive actions, diagnostic tests, treatments, surgeries, and medications. Our extensive evaluation reveals that LMs struggle with factual correctness, demonstrate bias across demographic and linguistic groups, and are susceptible to privacy breaches and adversarial attacks. By highlighting these shortcomings, CLINIC lays the foundation for enhancing the global reach and safety of LMs in healthcare across diverse languages.



## **23. Attacking and Securing Community Detection: A Game-Theoretic Framework**

cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11359v1) [paper-pdf](https://arxiv.org/pdf/2512.11359v1)

**Authors**: Yifan Niu, Aochuan Chen, Tingyang Xu, Jia Li

**Abstract**: It has been demonstrated that adversarial graphs, i.e., graphs with imperceptible perturbations, can cause deep graph models to fail on classification tasks. In this work, we extend the concept of adversarial graphs to the community detection problem, which is more challenging. We propose novel attack and defense techniques for community detection problem, with the objective of hiding targeted individuals from detection models and enhancing the robustness of community detection models, respectively. These techniques have many applications in real-world scenarios, for example, protecting personal privacy in social networks and understanding camouflage patterns in transaction networks. To simulate interactive attack and defense behaviors, we further propose a game-theoretic framework, called CD-GAME. One player is a graph attacker, while the other player is a Rayleigh Quotient defender. The CD-GAME models the mutual influence and feedback mechanisms between the attacker and the defender, revealing the dynamic evolutionary process of the game. Both players dynamically update their strategies until they reach the Nash equilibrium. Extensive experiments demonstrate the effectiveness of our proposed attack and defense methods, and both outperform existing baselines by a significant margin. Furthermore, CD-GAME provides valuable insights for understanding interactive attack and defense scenarios in community detection problems. We found that in traditional single-step attack or defense, attacker tends to employ strategies that are most effective, but are easily detected and countered by defender. When the interactive game reaches a Nash equilibrium, attacker adopts more imperceptible strategies that can still achieve satisfactory attack effectiveness even after defense.



## **24. Empirical evaluation of the Frank-Wolfe methods for constructing white-box adversarial attacks**

cs.LG

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10936v1) [paper-pdf](https://arxiv.org/pdf/2512.10936v1)

**Authors**: Kristina Korotkova, Aleksandr Katrutsa

**Abstract**: The construction of adversarial attacks for neural networks appears to be a crucial challenge for their deployment in various services. To estimate the adversarial robustness of a neural network, a fast and efficient approach is needed to construct adversarial attacks. Since the formalization of adversarial attack construction involves solving a specific optimization problem, we consider the problem of constructing an efficient and effective adversarial attack from a numerical optimization perspective. Specifically, we suggest utilizing advanced projection-free methods, known as modified Frank-Wolfe methods, to construct white-box adversarial attacks on the given input data. We perform a theoretical and numerical evaluation of these methods and compare them with standard approaches based on projection operations or geometrical intuition. Numerical experiments are performed on the MNIST and CIFAR-10 datasets, utilizing a multiclass logistic regression model, the convolutional neural networks (CNNs), and the Vision Transformer (ViT).



## **25. Adaptive Intrusion Detection System Leveraging Dynamic Neural Models with Adversarial Learning for 5G/6G Networks**

cs.CR

7 pages,3 figures, 2 Table. Neha and T. Bhatia "Adaptive Intrusion Detection System Leveraging Dynamic Neural Models with Adversarial Learning for 5G/6G Networks" (2025) 103-107

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.10637v2) [paper-pdf](https://arxiv.org/pdf/2512.10637v2)

**Authors**: Neha, Tarunpreet Bhatia

**Abstract**: Intrusion Detection Systems (IDS) are critical components in safeguarding 5G/6G networks from both internal and external cyber threats. While traditional IDS approaches rely heavily on signature-based methods, they struggle to detect novel and evolving attacks. This paper presents an advanced IDS framework that leverages adversarial training and dynamic neural networks in 5G/6G networks to enhance network security by providing robust, real-time threat detection and response capabilities. Unlike conventional models, which require costly retraining to update knowledge, the proposed framework integrates incremental learning algorithms, reducing the need for frequent retraining. Adversarial training is used to fortify the IDS against poisoned data. By using fewer features and incorporating statistical properties, the system can efficiently detect potential threats. Extensive evaluations using the NSL- KDD dataset demonstrate that the proposed approach provides better accuracy of 82.33% for multiclass classification of various network attacks while resisting dataset poisoning. This research highlights the potential of adversarial-trained, dynamic neural networks for building resilient IDS solutions.



## **26. Authority Backdoor: A Certifiable Backdoor Mechanism for Authoring DNNs**

cs.CR

Accepted to AAAI 2026 (Main Track). Code is available at: https://github.com/PlayerYangh/Authority-Trigger

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10600v1) [paper-pdf](https://arxiv.org/pdf/2512.10600v1)

**Authors**: Han Yang, Shaofeng Li, Tian Dong, Xiangyu Xu, Guangchi Liu, Zhen Ling

**Abstract**: Deep Neural Networks (DNNs), as valuable intellectual property, face unauthorized use. Existing protections, such as digital watermarking, are largely passive; they provide only post-hoc ownership verification and cannot actively prevent the illicit use of a stolen model. This work proposes a proactive protection scheme, dubbed ``Authority Backdoor," which embeds access constraints directly into the model. In particular, the scheme utilizes a backdoor learning framework to intrinsically lock a model's utility, such that it performs normally only in the presence of a specific trigger (e.g., a hardware fingerprint). But in its absence, the DNN's performance degrades to be useless. To further enhance the security of the proposed authority scheme, the certifiable robustness is integrated to prevent an adaptive attacker from removing the implanted backdoor. The resulting framework establishes a secure authority mechanism for DNNs, combining access control with certifiable robustness against adversarial attacks. Extensive experiments on diverse architectures and datasets validate the effectiveness and certifiable robustness of the proposed framework.



## **27. T-ADD: Enhancing DOA Estimation Robustness Against Adversarial Attacks**

eess.SP

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10496v1) [paper-pdf](https://arxiv.org/pdf/2512.10496v1)

**Authors**: Shilian Zheng, Xiaoxiang Wu, Luxin Zhang, Keqiang Yue, Peihan Qi, Zhijin Zhao

**Abstract**: Deep learning has achieved remarkable success in direction-of-arrival (DOA) estimation. However, recent studies have shown that adversarial perturbations can severely compromise the performance of such models. To address this vulnerability, we propose Transformer-based Adversarial Defense for DOA estimation (T-ADD), a transformer-based defense method designed to counter adversarial attacks. To achieve a balance between robustness and estimation accuracy, we formulate the adversarial defense as a joint reconstruction task and introduce a tailored joint loss function. Experimental results demonstrate that, compared with three state-of-the-art adversarial defense methods, the proposed T-ADD significantly mitigates the adverse effects of widely used adversarial attacks, leading to notable improvements in the adversarial robustness of the DOA model.



## **28. Stealth and Evasion in Rogue AP Attacks: An Analysis of Modern Detection and Bypass Techniques**

cs.CR

5 pages, 3 figures, experimental paper

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10470v1) [paper-pdf](https://arxiv.org/pdf/2512.10470v1)

**Authors**: Kaleb Bacztub, Braden Vester, Matteo Hodge, Liulseged Abate

**Abstract**: Wireless networks act as the backbone of modern digital connectivity, making them a primary target for cyber adversaries. Rogue Access Point attacks, specifically the Evil Twin variant, enable attackers to clone legitimate wireless network identifiers to deceive users into connecting. Once a connection is established, the adversary can intercept traffic and harvest sensitive credentials. While modern defensive architectures often employ Network Intrusion Detection Systems (NIDS) to identify malicious activity, the effectiveness of these systems against Layer 2 wireless threats remains a subject of critical inquiry. This project aimed to design a stealth-capable Rogue AP and evaluate its detectability against Suricata, an open-source NIDS/IPS. The methodology initially focused on a hardware-based deployment using Raspberry Pi platforms but transitioned to a virtualized environment due to severe system compatibility issues. Using Wifipumpkin3, the research team successfully deployed a captive portal that harvested user credentials from connected devices. However, the Suricata NIDS failed to flag the attack, highlighting a significant blind spot in traditional intrusion detection regarding wireless management frame attacks. This paper details the construction of the attack, the evasion techniques employed, and the limitations of current NIDS solutions in detecting localized wireless threats



## **29. When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection**

cs.AI

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.10449v2) [paper-pdf](https://arxiv.org/pdf/2512.10449v2)

**Authors**: Devanshu Sahoo, Manish Prasad, Vasudev Majhi, Jahnvi Singh, Vinay Chamola, Yash Sinha, Murari Mandal, Dhruv Kumar

**Abstract**: The landscape of scientific peer review is rapidly evolving with the integration of Large Language Models (LLMs). This shift is driven by two parallel trends: the widespread individual adoption of LLMs by reviewers to manage workload (the "Lazy Reviewer" hypothesis) and the formal institutional deployment of AI-powered assessment systems by conferences like AAAI and Stanford's Agents4Science. This study investigates the robustness of these "LLM-as-a-Judge" systems (both illicit and sanctioned) to adversarial PDF manipulation. Unlike general jailbreaks, we focus on a distinct incentive: flipping "Reject" decisions to "Accept," for which we develop a novel evaluation metric which we term as WAVS (Weighted Adversarial Vulnerability Score). We curated a dataset of 200 scientific papers and adapted 15 domain-specific attack strategies to this task, evaluating them across 13 Language Models, including GPT-5, Claude Haiku, and DeepSeek. Our results demonstrate that obfuscation strategies like "Maximum Mark Magyk" successfully manipulate scores, achieving alarming decision flip rates even in large-scale models. We will release our complete dataset and injection framework to facilitate more research on this topic.



## **30. Differential Privacy for Secure Machine Learning in Healthcare IoT-Cloud Systems**

cs.CR

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10426v1) [paper-pdf](https://arxiv.org/pdf/2512.10426v1)

**Authors**: N Mangala, Murtaza Rangwala, S Aishwarya, B Eswara Reddy, Rajkumar Buyya, KR Venugopal, SS Iyengar, LM Patnaik

**Abstract**: Healthcare has become exceptionally sophisticated, as wearables and connected medical devices are revolutionising remote patient monitoring, emergency response, medication management, diagnosis, and predictive and prescriptive analytics. Internet of Things and Cloud computing integrated systems (IoT-Cloud) facilitate sensing, automation, and processing for these healthcare applications. While real-time response is crucial for alleviating patient emergencies, protecting patient privacy is extremely important in data-driven healthcare. In this paper, we propose a multi-layer IoT, Edge and Cloud architecture to enhance the speed of response for emergency healthcare by distributing tasks based on response criticality and permanence of storage. Privacy of patient data is assured by proposing a Differential Privacy framework across several machine learning models such as K-means, Logistic Regression, Random Forest and Naive Bayes. We establish a comprehensive threat model identifying three adversary classes and evaluate Laplace, Gaussian, and hybrid noise mechanisms across varying privacy budgets, with supervised algorithms achieving up to 86% accuracy. The proposed hybrid Laplace-Gaussian noise mechanism with adaptive budget allocation provides a balanced approach, offering moderate tails and better privacy-utility trade-offs for both low and high dimension datasets. At the practical threshold of $\varepsilon = 5.0$, supervised algorithms achieve 82-84% accuracy while reducing attribute inference attacks by up to 18% and data reconstruction correlation by 70%. Blockchain security further ensures trusted communication through time-stamping, traceability, and immutability for analytics applications. Edge computing demonstrates 8$\times$ latency reduction for emergency scenarios, validating the hierarchical architecture for time-critical operations.



## **31. Phishing Email Detection Using Large Language Models**

cs.CR

7 pages

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.10104v2) [paper-pdf](https://arxiv.org/pdf/2512.10104v2)

**Authors**: Najmul Hasan, Prashanth BusiReddyGari, Haitao Zhao, Yihao Ren, Jinsheng Xu, Shaohu Zhang

**Abstract**: Email phishing is one of the most prevalent and globally consequential vectors of cyber intrusion. As systems increasingly deploy Large Language Models (LLMs) applications, these systems face evolving phishing email threats that exploit their fundamental architectures. Current LLMs require substantial hardening before deployment in email security systems, particularly against coordinated multi-vector attacks that exploit architectural vulnerabilities. This paper proposes LLMPEA, an LLM-based framework to detect phishing email attacks across multiple attack vectors, including prompt injection, text refinement, and multilingual attacks. We evaluate three frontier LLMs (e.g., GPT-4o, Claude Sonnet 4, and Grok-3) and comprehensive prompting design to assess their feasibility, robustness, and limitations against phishing email attacks. Our empirical analysis reveals that LLMs can detect the phishing email over 90% accuracy while we also highlight that LLM-based phishing email detection systems could be exploited by adversarial attack, prompt injection, and multilingual attacks. Our findings provide critical insights for LLM-based phishing detection in real-world settings where attackers exploit multiple vulnerabilities in combination.



## **32. Optimization-Guided Diffusion for Interactive Scene Generation**

cs.CV

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.07661v2) [paper-pdf](https://arxiv.org/pdf/2512.07661v2)

**Authors**: Shihao Li, Naisheng Ye, Tianyu Li, Kashyap Chitta, Tuo An, Peng Su, Boyang Wang, Haiou Liu, Chen Lv, Hongyang Li

**Abstract**: Realistic and diverse multi-agent driving scenes are crucial for evaluating autonomous vehicles, but safety-critical events which are essential for this task are rare and underrepresented in driving datasets. Data-driven scene generation offers a low-cost alternative by synthesizing complex traffic behaviors from existing driving logs. However, existing models often lack controllability or yield samples that violate physical or social constraints, limiting their usability. We present OMEGA, an optimization-guided, training-free framework that enforces structural consistency and interaction awareness during diffusion-based sampling from a scene generation model. OMEGA re-anchors each reverse diffusion step via constrained optimization, steering the generation towards physically plausible and behaviorally coherent trajectories. Building on this framework, we formulate ego-attacker interactions as a game-theoretic optimization in the distribution space, approximating Nash equilibria to generate realistic, safety-critical adversarial scenarios. Experiments on nuPlan and Waymo show that OMEGA improves generation realism, consistency, and controllability, increasing the ratio of physically and behaviorally valid scenes from 32.35% to 72.27% for free exploration capabilities, and from 11% to 80% for controllability-focused generation. Our approach can also generate $5\times$ more near-collision frames with a time-to-collision under three seconds while maintaining the overall scene realism.



## **33. Defense That Attacks: How Robust Models Become Better Attackers**

cs.CV

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.02830v3) [paper-pdf](https://arxiv.org/pdf/2512.02830v3)

**Authors**: Mohamed Awad, Mahmoud Akrm, Walid Gomaa

**Abstract**: Deep learning has achieved great success in computer vision, but remains vulnerable to adversarial attacks. Adversarial training is the leading defense designed to improve model robustness. However, its effect on the transferability of attacks is underexplored. In this work, we ask whether adversarial training unintentionally increases the transferability of adversarial examples. To answer this, we trained a diverse zoo of 36 models, including CNNs and ViTs, and conducted comprehensive transferability experiments. Our results reveal a clear paradox: adversarially trained (AT) models produce perturbations that transfer more effectively than those from standard models, which introduce a new ecosystem risk. To enable reproducibility and further study, we release all models, code, and experimental scripts. Furthermore, we argue that robustness evaluations should assess not only the resistance of a model to transferred attacks but also its propensity to produce transferable adversarial examples.



## **34. SA$^{2}$GFM: Enhancing Robust Graph Foundation Models with Structure-Aware Semantic Augmentation**

cs.LG

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.07857v2) [paper-pdf](https://arxiv.org/pdf/2512.07857v2)

**Authors**: Junhua Shi, Qingyun Sun, Haonan Yuan, Xingcheng Fu

**Abstract**: We present Graph Foundation Models (GFMs) which have made significant progress in various tasks, but their robustness against domain noise, structural perturbations, and adversarial attacks remains underexplored. A key limitation is the insufficient modeling of hierarchical structural semantics, which are crucial for generalization. In this paper, we propose SA$^{2}$GFM, a robust GFM framework that improves domain-adaptive representations through Structure-Aware Semantic Augmentation. First, we encode hierarchical structural priors by transforming entropy-based encoding trees into structure-aware textual prompts for feature augmentation. The enhanced inputs are processed by a self-supervised Information Bottleneck mechanism that distills robust, transferable representations via structure-guided compression. To address negative transfer in cross-domain adaptation, we introduce an expert adaptive routing mechanism, combining a mixture-of-experts architecture with a null expert design. For efficient downstream adaptation, we propose a fine-tuning module that optimizes hierarchical structures through joint intra- and inter-community structure learning. Extensive experiments demonstrate that SA$^{2}$GFM outperforms 9 state-of-the-art baselines in terms of effectiveness and robustness against random noise and adversarial perturbations for node and graph classification.



## **35. ATAC: Augmentation-Based Test-Time Adversarial Correction for CLIP**

cs.CV

16 pages

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2511.17362v2) [paper-pdf](https://arxiv.org/pdf/2511.17362v2)

**Authors**: Linxiang Su, András Balogh

**Abstract**: Despite its remarkable success in zero-shot image-text matching, CLIP remains highly vulnerable to adversarial perturbations on images. As adversarial fine-tuning is prohibitively costly, recent works explore various test-time defense strategies; however, these approaches still exhibit limited robustness. In this work, we revisit this problem and propose a simple yet effective strategy: Augmentation-based Test-time Adversarial Correction (ATAC). Our method operates directly in the embedding space of CLIP, calculating augmentation-induced drift vectors to infer a semantic recovery direction and correcting the embedding based on the angular consistency of these latent drifts. Across a wide range of benchmarks, ATAC consistently achieves remarkably high robustness, surpassing that of previous state-of-the-art methods by nearly 50\% on average, all while requiring minimal computational overhead. Furthermore, ATAC retains state-of-the-art robustness in unconventional and extreme settings and even achieves nontrivial robustness against adaptive attacks. Our results demonstrate that ATAC is an efficient method in a novel paradigm for test-time adversarial defenses in the embedding space of CLIP.



## **36. Physically Realistic Sequence-Level Adversarial Clothing for Robust Human-Detection Evasion**

cs.CV

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2511.16020v2) [paper-pdf](https://arxiv.org/pdf/2511.16020v2)

**Authors**: Dingkun Zhou, Patrick P. K. Chan, Hengxu Wu, Shikang Zheng, Ruiqi Huang, Yuanjie Zhao

**Abstract**: Deep neural networks used for human detection are highly vulnerable to adversarial manipulation, creating safety and privacy risks in real surveillance environments. Wearable attacks offer a realistic threat model, yet existing approaches usually optimize textures frame by frame and therefore fail to maintain concealment across long video sequences with motion, pose changes, and garment deformation. In this work, a sequence-level optimization framework is introduced to generate natural, printable adversarial textures for shirts, trousers, and hats that remain effective throughout entire walking videos in both digital and physical settings. Product images are first mapped to UV space and converted into a compact palette and control-point parameterization, with ICC locking to keep all colors printable. A physically based human-garment pipeline is then employed to simulate motion, multi-angle camera viewpoints, cloth dynamics, and illumination variation. An expectation-over-transformation objective with temporal weighting is used to optimize the control points so that detection confidence is minimized across whole sequences. Extensive experiments demonstrate strong and stable concealment, high robustness to viewpoint changes, and superior cross-model transferability. Physical garments produced with sublimation printing achieve reliable suppression under indoor and outdoor recordings, confirming real-world feasibility.



## **37. Cyber-Resilient Data-Driven Event-Triggered Secure Control for Autonomous Vehicles Under False Data Injection Attacks**

eess.SY

14 pages, 8 figures

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2511.15925v2) [paper-pdf](https://arxiv.org/pdf/2511.15925v2)

**Authors**: Yashar Mousavi, Mahsa Tavasoli, Ibrahim Beklan Kucukdemiral, Umit Cali, Abdolhossein Sarrafzadeh, Ali Karimoddini, Afef Fekih

**Abstract**: This paper proposes a cyber-resilient secure control framework for autonomous vehicles (AVs) subject to false data injection (FDI) threats as actuator attacks. The framework integrates data-driven modeling, event-triggered communication, and fractional-order sliding mode control (FSMC) to enhance the resilience against adversarial interventions. A dynamic model decomposition (DMD)-based methodology is employed to extract the lateral dynamics from real-world data, eliminating the reliance on conventional mechanistic modeling. To optimize communication efficiency, an event-triggered transmission scheme is designed to reduce the redundant transmissions while ensuring system stability. Furthermore, an extended state observer (ESO) is developed for real-time estimation and mitigation of actuator attack effects. Theoretical stability analysis, conducted using Lyapunov methods and linear matrix inequality (LMI) formulations, guarantees exponential error convergence. Extensive simulations validate the proposed event-triggered secure control framework, demonstrating substantial improvements in attack mitigation, communication efficiency, and lateral tracking performance. The results show that the framework effectively counteracts actuator attacks while optimizing communication-resource utilization, making it highly suitable for safety-critical AV applications.



## **38. BreakFun: Jailbreaking LLMs via Schema Exploitation**

cs.CR

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2510.17904v2) [paper-pdf](https://arxiv.org/pdf/2510.17904v2)

**Authors**: Amirkia Rafiei Oskooei, Mehmet S. Aktas

**Abstract**: The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.



## **39. DGTEN: A Robust Deep Gaussian based Graph Neural Network for Dynamic Trust Evaluation with Uncertainty-Quantification Support**

cs.LG

15 pages, 6 figures, 5 tables

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2510.07620v2) [paper-pdf](https://arxiv.org/pdf/2510.07620v2)

**Authors**: Muhammad Usman, Yugyung Lee

**Abstract**: Dynamic trust evaluation in large, rapidly evolving graphs demands models that capture changing relationships, express calibrated confidence, and resist adversarial manipulation. DGTEN (Deep Gaussian-Based Trust Evaluation Network) introduces a unified graph-based framework that does all three by combining uncertainty-aware message passing, expressive temporal modeling, and built-in defenses against trust-targeted attacks. It represents nodes and edges as Gaussian distributions so that both semantic signals and epistemic uncertainty propagate through the graph neural network, enabling risk-aware trust decisions rather than overconfident guesses. To track how trust evolves, it layers hybrid absolute-Gaussian-hourglass positional encoding with Kolmogorov-Arnold network-based unbiased multi-head attention, then applies an ordinary differential equation-based residual learning module to jointly model abrupt shifts and smooth trends. Robust adaptive ensemble coefficient analysis prunes or down-weights suspicious interactions using complementary cosine and Jaccard similarity, curbing reputation laundering, sabotage, and on-off attacks. On two signed Bitcoin trust networks, DGTEN delivers standout gains where it matters most: in single-timeslot prediction on Bitcoin-OTC, it improves MCC by +12.34% over the best dynamic baseline; in the cold-start scenario on Bitcoin-Alpha, it achieves a +25.00% MCC improvement, the largest across all tasks and datasets; while under adversarial on-off attacks, it surpasses the baseline by up to +10.23% MCC. These results endorse the unified DGTEN framework.



## **40. Stealthy Yet Effective: Distribution-Preserving Backdoor Attacks on Graph Classification**

cs.LG

Accepted by NeurIPS 2025

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2509.26032v2) [paper-pdf](https://arxiv.org/pdf/2509.26032v2)

**Authors**: Xiaobao Wang, Ruoxiao Sun, Yujun Zhang, Bingdao Feng, Dongxiao He, Luzhi Wang, Di Jin

**Abstract**: Graph Neural Networks (GNNs) have demonstrated strong performance across tasks such as node classification, link prediction, and graph classification, but remain vulnerable to backdoor attacks that implant imperceptible triggers during training to control predictions. While node-level attacks exploit local message passing, graph-level attacks face the harder challenge of manipulating global representations while maintaining stealth. We identify two main sources of anomaly in existing graph classification backdoor methods: structural deviation from rare subgraph triggers and semantic deviation caused by label flipping, both of which make poisoned graphs easily detectable by anomaly detection models. To address this, we propose DPSBA, a clean-label backdoor framework that learns in-distribution triggers via adversarial training guided by anomaly-aware discriminators. DPSBA effectively suppresses both structural and semantic anomalies, achieving high attack success while significantly improving stealth. Extensive experiments on real-world datasets validate that DPSBA achieves a superior balance between effectiveness and detectability compared to state-of-the-art baselines.



## **41. LLMs Encode Harmfulness and Refusal Separately**

cs.CL

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2507.11878v4) [paper-pdf](https://arxiv.org/pdf/2507.11878v4)

**Authors**: Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, Weiyan Shi

**Abstract**: LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety.



## **42. DATABench: Evaluating Dataset Auditing in Deep Learning from an Adversarial Perspective**

cs.CR

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2507.05622v3) [paper-pdf](https://arxiv.org/pdf/2507.05622v3)

**Authors**: Shuo Shao, Yiming Li, Mengren Zheng, Zhiyang Hu, Yukun Chen, Boheng Li, Yu He, Junfeng Guo, Dacheng Tao, Zhan Qin

**Abstract**: The widespread application of Deep Learning across diverse domains hinges critically on the quality and composition of training datasets. However, the common lack of disclosure regarding their usage raises significant privacy and copyright concerns. Dataset auditing techniques, which aim to determine if a specific dataset was used to train a given suspicious model, provide promising solutions to addressing these transparency gaps. While prior work has developed various auditing methods, their resilience against dedicated adversarial attacks remains largely unexplored. To bridge the gap, this paper initiates a comprehensive study evaluating dataset auditing from an adversarial perspective. We start with introducing a novel taxonomy, classifying existing methods based on their reliance on internal features (IF) (inherent to the data) versus external features (EF) (artificially introduced for auditing). Subsequently, we formulate two primary attack types: evasion attacks, designed to conceal the use of a dataset, and forgery attacks, intending to falsely implicate an unused dataset. Building on the understanding of existing methods and attack objectives, we further propose systematic attack strategies: decoupling, removal, and detection for evasion; adversarial example-based methods for forgery. These formulations and strategies lead to our new benchmark, DATABench, comprising 17 evasion attacks, 5 forgery attacks, and 9 representative auditing methods. Extensive evaluations using DATABench reveal that none of the evaluated auditing methods are sufficiently robust or distinctive under adversarial settings. These findings underscore the urgent need for developing a more secure and reliable dataset auditing method capable of withstanding sophisticated adversarial manipulation. Code is available in https://github.com/shaoshuo-ss/DATABench.



## **43. Meta Policy Switching for Secure UAV Deconfliction in Adversarial Airspace**

cs.LG

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2506.21127v3) [paper-pdf](https://arxiv.org/pdf/2506.21127v3)

**Authors**: Deepak Kumar Panda, Weisi Guo

**Abstract**: Autonomous UAV navigation using reinforcement learning (RL) is vulnerable to adversarial attacks that manipulate sensor inputs, potentially leading to unsafe behavior and mission failure. Although robust RL methods provide partial protection, they often struggle to generalize to unseen or out-of-distribution (OOD) attacks due to their reliance on fixed perturbation settings. To address this limitation, we propose a meta-policy switching framework in which a meta-level polic dynamically selects among multiple robust policies to counter unknown adversarial shifts. At the core of this framework lies a discounted Thompson sampling (DTS) mechanism that formulates policy selection as a multi-armed bandit problem, thereby minimizing value distribution shifts via self-induced adversarial observations. We first construct a diverse ensemble of action-robust policies trained under varying perturbation intensities. The DTS-based meta-policy then adaptively selects among these policies online, optimizing resilience against self-induced, piecewise-stationary attacks. Theoretical analysis shows that the DTS mechanism minimizes expected regret, ensuring adaptive robustness to OOD attacks and exhibiting emergent antifragile behavior under uncertainty. Extensive simulations in complex 3D obstacle environments under both white-box (Projected Gradient Descent) and black-box (GPS spoofing) attacks demonstrate significantly improved navigation efficiency and higher conflict free trajectory rates compared to standard robust and vanilla RL baselines, highlighting the practical security and dependability benefits of the proposed approach.



## **44. A Flat Minima Perspective on Understanding Augmentations and Model Robustness**

cs.LG

In Proceedings of the Association for the Advancement of Artificial Intelligence (AAAI) 2026, Singapore

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2505.24592v3) [paper-pdf](https://arxiv.org/pdf/2505.24592v3)

**Authors**: Weebum Yoo, Sung Whan Yoon

**Abstract**: Model robustness indicates a model's capability to generalize well on unforeseen distributional shifts, including data corruptions and adversarial attacks. Data augmentation is one of the most prevalent and effective ways to enhance robustness. Despite the great success of the diverse augmentations in different fields, a unified theoretical understanding of their efficacy in improving model robustness is lacking. We theoretically reveal a general condition for label-preserving augmentations to bring robustness to diverse distribution shifts through the lens of flat minima and generalization bound, which de facto turns out to be strongly correlated with robustness against different distribution shifts in practice. Unlike most earlier works, our theoretical framework accommodates all the label-preserving augmentations and is not limited to particular distribution shifts. We substantiate our theories through different simulations on the existing common corruption and adversarial robustness benchmarks based on the CIFAR and ImageNet datasets.



## **45. Quantum Support Vector Regression for Robust Anomaly Detection**

quant-ph

Accepted to International Conference on Agents and Artificial Intelligence (ICAART) 2026

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2505.01012v3) [paper-pdf](https://arxiv.org/pdf/2505.01012v3)

**Authors**: Kilian Tscharke, Maximilian Wendlinger, Sebastian Issel, Pascal Debus

**Abstract**: Anomaly Detection (AD) is critical in data analysis, particularly within the domain of IT security. In this study, we explore the potential of Quantum Machine Learning for application to AD with special focus on the robustness to noise and adversarial attacks. We build upon previous work on Quantum Support Vector Regression (QSVR) for semisupervised AD by conducting a comprehensive benchmark on IBM quantum hardware using eleven datasets. Our results demonstrate that QSVR achieves strong classification performance and even outperforms the noiseless simulation on two of these datasets. Moreover, we investigate the influence of - in the NISQ-era inevitable - quantum noise on the performance of the QSVR. Our findings reveal that the model exhibits robustness to depolarizing, phase damping, phase flip, and bit flip noise, while amplitude damping and miscalibration noise prove to be more disruptive. Finally, we explore the domain of Quantum Adversarial Machine Learning by demonstrating that QSVR is highly vulnerable to adversarial attacks, with neither quantum noise nor adversarial training improving the model's robustness against such attacks.



## **46. Towards Backdoor Stealthiness in Model Parameter Space**

cs.CR

accepted by CCS 2025

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2501.05928v3) [paper-pdf](https://arxiv.org/pdf/2501.05928v3)

**Authors**: Xiaoyun Xu, Zhuoran Liu, Stefanos Koffas, Stjepan Picek

**Abstract**: Recent research on backdoor stealthiness focuses mainly on indistinguishable triggers in input space and inseparable backdoor representations in feature space, aiming to circumvent backdoor defenses that examine these respective spaces. However, existing backdoor attacks are typically designed to resist a specific type of backdoor defense without considering the diverse range of defense mechanisms. Based on this observation, we pose a natural question: Are current backdoor attacks truly a real-world threat when facing diverse practical defenses?   To answer this question, we examine 12 common backdoor attacks that focus on input-space or feature-space stealthiness and 17 diverse representative defenses. Surprisingly, we reveal a critical blind spot: Backdoor attacks designed to be stealthy in input and feature spaces can be mitigated by examining backdoored models in parameter space. To investigate the underlying causes behind this common vulnerability, we study the characteristics of backdoor attacks in the parameter space. Notably, we find that input- and feature-space attacks introduce prominent backdoor-related neurons in parameter space, which are not thoroughly considered by current backdoor attacks. Taking comprehensive stealthiness into account, we propose a novel supply-chain attack called Grond. Grond limits the parameter changes by a simple yet effective module, Adversarial Backdoor Injection (ABI), which adaptively increases the parameter-space stealthiness during the backdoor injection. Extensive experiments demonstrate that Grond outperforms all 12 backdoor attacks against state-of-the-art (including adaptive) defenses on CIFAR-10, GTSRB, and a subset of ImageNet. In addition, we show that ABI consistently improves the effectiveness of common backdoor attacks.



## **47. FlippedRAG: Black-Box Opinion Manipulation Adversarial Attacks to Retrieval-Augmented Generation Models**

cs.IR

arXiv admin note: text overlap with arXiv:2407.13757

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2501.02968v5) [paper-pdf](https://arxiv.org/pdf/2501.02968v5)

**Authors**: Zhuo Chen, Yuyang Gong, Jiawei Liu, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu

**Abstract**: Retrieval-Augmented Generation (RAG) enriches LLMs by dynamically retrieving external knowledge, reducing hallucinations and satisfying real-time information needs. While existing research mainly targets RAG's performance and efficiency, emerging studies highlight critical security concerns. Yet, current adversarial approaches remain limited, mostly addressing white-box scenarios or heuristic black-box attacks without fully investigating vulnerabilities in the retrieval phase. Additionally, prior works mainly focus on factoid Q&A tasks, their attacks lack complexity and can be easily corrected by advanced LLMs. In this paper, we investigate a more realistic and critical threat scenario: adversarial attacks intended for opinion manipulation against black-box RAG models, particularly on controversial topics. Specifically, we propose FlippedRAG, a transfer-based adversarial attack against black-box RAG systems. We first demonstrate that the underlying retriever of a black-box RAG system can be reverse-engineered, enabling us to train a surrogate retriever. Leveraging the surrogate retriever, we further craft target poisoning triggers, altering vary few documents to effectively manipulate both retrieval and subsequent generation. Extensive empirical results show that FlippedRAG substantially outperforms baseline methods, improving the average attack success rate by 16.7%. FlippedRAG achieves on average a 50% directional shift in the opinion polarity of RAG-generated responses, ultimately causing a notable 20% shift in user cognition. Furthermore, we evaluate the performance of several potential defensive measures, concluding that existing mitigation strategies remain insufficient against such sophisticated manipulation attacks. These results highlight an urgent need for developing innovative defensive solutions to ensure the security and trustworthiness of RAG systems.



## **48. Defending Collaborative Filtering Recommenders via Adversarial Robustness Based Edge Reweighting**

cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2412.10850v2) [paper-pdf](https://arxiv.org/pdf/2412.10850v2)

**Authors**: Yongyu Wang

**Abstract**: User based collaborative filtering (CF) relies on a user and user similarity graph, making it vulnerable to profile injection (shilling) attacks that manipulate neighborhood relations to promote (push) or demote (nuke) target items. In this work, we propose an adversarial robustness based edge reweighting defense for CF. We first assign each user and user edge a non robustness score via spectral adversarial robustness evaluation, which quantifies the edge sensitivity to adversarial perturbations. We then attenuate the influence of non robust edges by reweighting similarities during prediction. Extensive experiments demonstrate that the proposed method effectively defends against various types of attacks.



## **49. Certifying Robustness of Graph Convolutional Networks for Node Perturbation with Polyhedra Abstract Interpretation**

cs.LG

Author preprint, published at Data Mining and Knowledge Discovery. Published version available at: https://link.springer.com/article/10.1007/s10618-025-01180-w

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2405.08645v2) [paper-pdf](https://arxiv.org/pdf/2405.08645v2)

**Authors**: Boqi Chen, Kristóf Marussy, Oszkár Semeráth, Gunter Mussbacher, Dániel Varró

**Abstract**: Graph convolutional neural networks (GCNs) are powerful tools for learning graph-based knowledge representations from training data. However, they are vulnerable to small perturbations in the input graph, which makes them susceptible to input faults or adversarial attacks. This poses a significant problem for GCNs intended to be used in critical applications, which need to provide certifiably robust services even in the presence of adversarial perturbations. We propose an improved GCN robustness certification technique for node classification in the presence of node feature perturbations. We introduce a novel polyhedra-based abstract interpretation approach to tackle specific challenges of graph data and provide tight upper and lower bounds for the robustness of the GCN. Experiments show that our approach simultaneously improves the tightness of robustness bounds as well as the runtime performance of certification. Moreover, our method can be used during training to further improve the robustness of GCNs.



## **50. We Can Always Catch You: Detecting Adversarial Patched Objects WITH or WITHOUT Signature**

cs.CV

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2106.05261v3) [paper-pdf](https://arxiv.org/pdf/2106.05261v3)

**Authors**: Jiachun Li, Jianan Feng, Jianjun Huang, Bin Liang

**Abstract**: Recently, object detection has proven vulnerable to adversarial patch attacks. The attackers holding a specially crafted patch can hide themselves from state-of-the-art detectors, e.g., YOLO, even in the physical world. This attack can bring serious security threats, such as escaping from surveillance cameras. How to effectively detect this kind of adversarial examples to catch potential attacks has become an important problem. In this paper, we propose two detection methods: the signature-based method and the signature-independent method. First, we identify two signatures of existing adversarial patches that can be utilized to precisely locate patches within adversarial examples. By employing the signatures, a fast signature-based method is developed to detect the adversarial objects. Second, we present a robust signature-independent method based on the \textit{content semantics consistency} of model outputs. Adversarial objects violate this consistency, appearing locally but disappearing globally, while benign ones remain consistently present. The experiments demonstrate that two proposed methods can effectively detect attacks both in the digital and physical world. These methods each offer distinct advantage. Specifically, the signature-based method is capable of real-time detection, while the signature-independent method can detect unknown adversarial patch attacks and makes defense-aware attacks almost impossible to perform.



