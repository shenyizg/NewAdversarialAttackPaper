# Traditional Deep Learning Models - Digital Attack
**update at 2026-01-25 10:36:50**

Sorted by classifier confidence (high to low).

## **1. VTarbel: Targeted Label Attack with Minimal Knowledge on Detector-enhanced Vertical Federated Learning**

cs.CR

Accepted by ACM Transactions on Sensor Networks (TOSN)

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2507.14625v2) [paper-pdf](https://arxiv.org/pdf/2507.14625v2)

**Confidence**: 0.95

**Authors**: Juntao Tan, Anran Li, Quanchao Liu, Peng Ran, Lan Zhang

**Abstract**: Vertical federated learning (VFL) enables multiple parties with disjoint features to collaboratively train models without sharing raw data. While privacy vulnerabilities of VFL are extensively-studied, its security threats-particularly targeted label attacks-remain underexplored. In such attacks, a passive party perturbs inputs at inference to force misclassification into adversary-chosen labels. Existing methods rely on unrealistic assumptions (e.g., accessing VFL-model's outputs) and ignore anomaly detectors deployed in real-world systems. To bridge this gap, we introduce VTarbel, a two-stage, minimal-knowledge attack framework explicitly designed to evade detector-enhanced VFL inference. During the preparation stage, the attacker selects a minimal set of high-expressiveness samples (via maximum mean discrepancy), submits them through VFL protocol to collect predicted labels, and uses these pseudo-labels to train estimated detector and surrogate model on local features. In attack stage, these models guide gradient-based perturbations of remaining samples, crafting adversarial instances that induce targeted misclassifications and evade detection. We implement VTarbel and evaluate it against four model architectures, seven multimodal datasets, and two anomaly detectors. Across all settings, VTarbel outperforms four state-of-the-art baselines, evades detection, and retains effective against three representative privacy-preserving defenses. These results reveal critical security blind spots in current VFL deployments and underscore urgent need for robust, attack-aware defenses.



## **2. Power to the Clients: Federated Learning in a Dictatorship Setting**

cs.LG

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2510.22149v2) [paper-pdf](https://arxiv.org/pdf/2510.22149v2)

**Confidence**: 0.95

**Authors**: Mohammadsajad Alipour, Mohammad Mohammadi Amiri

**Abstract**: Federated learning (FL) has emerged as a promising paradigm for decentralized model training, enabling multiple clients to collaboratively learn a shared model without exchanging their local data. However, the decentralized nature of FL also introduces vulnerabilities, as malicious clients can compromise or manipulate the training process. In this work, we introduce dictator clients, a novel, well-defined, and analytically tractable class of malicious participants capable of entirely erasing the contributions of all other clients from the server model, while preserving their own. We propose concrete attack strategies that empower such clients and systematically analyze their effects on the learning process. Furthermore, we explore complex scenarios involving multiple dictator clients, including cases where they collaborate, act independently, or form an alliance in order to ultimately betray one another. For each of these settings, we provide a theoretical analysis of their impact on the global model's convergence. Our theoretical algorithms and findings about the complex scenarios including multiple dictator clients are further supported by empirical evaluations on both computer vision and natural language processing benchmarks.



## **3. BREPS: Bounding-Box Robustness Evaluation of Promptable Segmentation**

cs.CV

Accepted by AAAI2026

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15123v1) [paper-pdf](https://arxiv.org/pdf/2601.15123v1)

**Confidence**: 0.95

**Authors**: Andrey Moskalenko, Danil Kuznetsov, Irina Dudko, Anastasiia Iasakova, Nikita Boldyrev, Denis Shepelev, Andrei Spiridonov, Andrey Kuznetsov, Vlad Shakhuro

**Abstract**: Promptable segmentation models such as SAM have established a powerful paradigm, enabling strong generalization to unseen objects and domains with minimal user input, including points, bounding boxes, and text prompts. Among these, bounding boxes stand out as particularly effective, often outperforming points while significantly reducing annotation costs. However, current training and evaluation protocols typically rely on synthetic prompts generated through simple heuristics, offering limited insight into real-world robustness. In this paper, we investigate the robustness of promptable segmentation models to natural variations in bounding box prompts. First, we conduct a controlled user study and collect thousands of real bounding box annotations. Our analysis reveals substantial variability in segmentation quality across users for the same model and instance, indicating that SAM-like models are highly sensitive to natural prompt noise. Then, since exhaustive testing of all possible user inputs is computationally prohibitive, we reformulate robustness evaluation as a white-box optimization problem over the bounding box prompt space. We introduce BREPS, a method for generating adversarial bounding boxes that minimize or maximize segmentation error while adhering to naturalness constraints. Finally, we benchmark state-of-the-art models across 10 datasets, spanning everyday scenes to medical imaging. Code - https://github.com/emb-ai/BREPS.



## **4. Diffusion-Driven Deceptive Patches: Adversarial Manipulation and Forensic Detection in Facial Identity Verification**

cs.CV

This manuscript is a preprint. A revised version of this work has been accepted for publication in the Springer Nature book Artificial Intelligence-Driven Forensics. This version includes one additional figure for completeness

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.09806v1) [paper-pdf](https://arxiv.org/pdf/2601.09806v1)

**Confidence**: 0.95

**Authors**: Shahrzad Sayyafzadeh, Hongmei Chi, Shonda Bernadin

**Abstract**: This work presents an end-to-end pipeline for generating, refining, and evaluating adversarial patches to compromise facial biometric systems, with applications in forensic analysis and security testing. We utilize FGSM to generate adversarial noise targeting an identity classifier and employ a diffusion model with reverse diffusion to enhance imperceptibility through Gaussian smoothing and adaptive brightness correction, thereby facilitating synthetic adversarial patch evasion. The refined patch is applied to facial images to test its ability to evade recognition systems while maintaining natural visual characteristics. A Vision Transformer (ViT)-GPT2 model generates captions to provide a semantic description of a person's identity for adversarial images, supporting forensic interpretation and documentation for identity evasion and recognition attacks. The pipeline evaluates changes in identity classification, captioning results, and vulnerabilities in facial identity verification and expression recognition under adversarial conditions. We further demonstrate effective detection and analysis of adversarial patches and adversarial samples using perceptual hashing and segmentation, achieving an SSIM of 0.95.



## **5. DiMEx: Breaking the Cold Start Barrier in Data-Free Model Extraction via Latent Diffusion Priors**

cs.LG

8 pages, 3 figures, 4 tables

**SubmitDate**: 2026-01-10    [abs](http://arxiv.org/abs/2601.01688v2) [paper-pdf](https://arxiv.org/pdf/2601.01688v2)

**Confidence**: 0.95

**Authors**: Yash Thesia, Meera Suthar

**Abstract**: Model stealing attacks pose an existential threat to Machine Learning as a Service (MLaaS), allowing adversaries to replicate proprietary models for a fraction of their training cost. While Data-Free Model Extraction (DFME) has emerged as a stealthy vector, it remains fundamentally constrained by the "Cold Start" problem: GAN-based adversaries waste thousands of queries converging from random noise to meaningful data. We propose DiMEx, a framework that weaponizes the rich semantic priors of pre-trained Latent Diffusion Models to bypass this initialization barrier entirely. By employing Random Embedding Bayesian Optimization (REMBO) within the generator's latent space, DiMEx synthesizes high-fidelity queries immediately, achieving 52.1 percent agreement on SVHN with just 2,000 queries - outperforming state-of-the-art GAN baselines by over 16 percent. To counter this highly semantic threat, we introduce the Hybrid Stateful Ensemble (HSE) defense, which identifies the unique "optimization trajectory" of latent-space attacks. Our results demonstrate that while DiMEx evades static distribution detectors, HSE exploits this temporal signature to suppress attack success rates to 21.6 percent with negligible latency.



## **6. ZQBA: Zero Query Black-box Adversarial Attack**

cs.CV

Accepted in ICAART 2026 Conference

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2510.00769v2) [paper-pdf](https://arxiv.org/pdf/2510.00769v2)

**Confidence**: 0.95

**Authors**: Joana C. Costa, Tiago Roxo, Hugo Proença, Pedro R. M. Inácio

**Abstract**: Current black-box adversarial attacks either require multiple queries or diffusion models to produce adversarial samples that can impair the target model performance. However, these methods require training a surrogate loss or diffusion models to produce adversarial samples, which limits their applicability in real-world settings. Thus, we propose a Zero Query Black-box Adversarial (ZQBA) attack that exploits the representations of Deep Neural Networks (DNNs) to fool other networks. Instead of requiring thousands of queries to produce deceiving adversarial samples, we use the feature maps obtained from a DNN and add them to clean images to impair the classification of a target model. The results suggest that ZQBA can transfer the adversarial samples to different models and across various datasets, namely CIFAR and Tiny ImageNet. The experiments also show that ZQBA is more effective than state-of-the-art black-box attacks with a single query, while maintaining the imperceptibility of perturbations, evaluated both quantitatively (SSIM) and qualitatively, emphasizing the vulnerabilities of employing DNNs in real-world contexts. All the source code is available at https://github.com/Joana-Cabral/ZQBA.



## **7. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2407.20836v5) [paper-pdf](https://arxiv.org/pdf/2407.20836v5)

**Confidence**: 0.95

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g., transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we demonstrate that adversarial attacks pose a real threat to AIGI detectors. FPBA can deliver successful black-box attacks across various detectors, generators, defense methods, and even evade cross-generator and compressed image detection, which are crucial real-world detection scenarios. Our code is available at https://github.com/onotoa/fpba.



## **8. DiffProtect: Generate Adversarial Examples with Diffusion Models for Facial Privacy Protection**

cs.CV

Code is at https://github.com/joellliu/DiffProtect/

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2305.13625v4) [paper-pdf](https://arxiv.org/pdf/2305.13625v4)

**Confidence**: 0.95

**Authors**: Jiang Liu, Chun Pong Lau, Zhongliang Guo, Yuxiang Guo, Zhaoyang Wang, Rama Chellappa

**Abstract**: The increasingly pervasive facial recognition (FR) systems raise serious concerns about personal privacy, especially for billions of users who have publicly shared their photos on social media. Several attempts have been made to protect individuals from being identified by unauthorized FR systems utilizing adversarial attacks to generate encrypted face images. However, existing methods suffer from poor visual quality or low attack success rates, which limit their utility. Recently, diffusion models have achieved tremendous success in image generation. In this work, we ask: can diffusion models be used to generate adversarial examples to improve both visual quality and attack performance? We propose DiffProtect, which utilizes a diffusion autoencoder to generate semantically meaningful perturbations on FR systems. Extensive experiments demonstrate that DiffProtect produces more natural-looking encrypted images than state-of-the-art methods while achieving significantly higher attack success rates, e.g., 24.5% and 25.1% absolute improvements on the CelebA-HQ and FFHQ datasets.



## **9. MORE: Multi-Objective Adversarial Attacks on Speech Recognition**

eess.AS

19 pages

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2601.01852v2) [paper-pdf](https://arxiv.org/pdf/2601.01852v2)

**Confidence**: 0.95

**Authors**: Xiaoxue Gao, Zexin Li, Yiming Chen, Nancy F. Chen

**Abstract**: The emergence of large-scale automatic speech recognition (ASR) models such as Whisper has greatly expanded their adoption across diverse real-world applications. Ensuring robustness against even minor input perturbations is therefore critical for maintaining reliable performance in real-time environments. While prior work has mainly examined accuracy degradation under adversarial attacks, robustness with respect to efficiency remains largely unexplored. This narrow focus provides only a partial understanding of ASR model vulnerabilities. To address this gap, we conduct a comprehensive study of ASR robustness under multiple attack scenarios. We introduce MORE, a multi-objective repetitive doubling encouragement attack, which jointly degrades recognition accuracy and inference efficiency through a hierarchical staged repulsion-anchoring mechanism. Specifically, we reformulate multi-objective adversarial optimization into a hierarchical framework that sequentially achieves the dual objectives. To further amplify effectiveness, we propose a novel repetitive encouragement doubling objective (REDO) that induces duplicative text generation by maintaining accuracy degradation and periodically doubling the predicted sequence length. Overall, MORE compels ASR models to produce incorrect transcriptions at a substantially higher computational cost, triggered by a single adversarial input. Experiments show that MORE consistently yields significantly longer transcriptions while maintaining high word error rates compared to existing baselines, underscoring its effectiveness in multi-objective adversarial attack.



## **10. Instruct2Attack: Language-Guided Semantic Adversarial Attacks**

cs.CV

under submission, code coming soon

**SubmitDate**: 2023-11-27    [abs](http://arxiv.org/abs/2311.15551v1) [paper-pdf](https://arxiv.org/pdf/2311.15551v1)

**Confidence**: 0.95

**Authors**: Jiang Liu, Chen Wei, Yuxiang Guo, Heng Yu, Alan Yuille, Soheil Feizi, Chun Pong Lau, Rama Chellappa

**Abstract**: We propose Instruct2Attack (I2A), a language-guided semantic attack that generates semantically meaningful perturbations according to free-form language instructions. We make use of state-of-the-art latent diffusion models, where we adversarially guide the reverse diffusion process to search for an adversarial latent code conditioned on the input image and text instruction. Compared to existing noise-based and semantic attacks, I2A generates more natural and diverse adversarial examples while providing better controllability and interpretability. We further automate the attack process with GPT-4 to generate diverse image-specific text instructions. We show that I2A can successfully break state-of-the-art deep neural networks even under strong adversarial defenses, and demonstrate great transferability among a variety of network architectures.



## **11. Cyberattack Detection in Virtualized Microgrids Using LightGBM and Knowledge-Distilled Classifiers**

eess.SY

12 pages

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2601.03495v1) [paper-pdf](https://arxiv.org/pdf/2601.03495v1)

**Confidence**: 0.95

**Authors**: Osasumwen Cedric Ogiesoba-Eguakun, Suman Rath

**Abstract**: Modern microgrids depend on distributed sensing and communication interfaces, making them increasingly vulnerable to cyber physical disturbances that threaten operational continuity and equipment safety. In this work, a complete virtual microgrid was designed and implemented in MATLAB/Simulink, integrating heterogeneous renewable sources and secondary controller layers. A structured cyberattack framework was developed using MGLib to inject adversarial signals directly into the secondary control pathways. Multiple attack classes were emulated, including ramp, sinusoidal, additive, coordinated stealth, and denial of service behaviors. The virtual environment was used to generate labeled datasets under both normal and attack conditions. The datasets trained Light Gradient Boosting Machine (LightGBM) models to perform two functions: detecting the presence of an intrusion (binary) and distinguishing among attack types (multiclass). The multiclass model attained 99.72% accuracy and a 99.62% F1 score, while the binary model attained 94.8% accuracy and a 94.3% F1 score. A knowledge-distillation step reduced the size of the multiclass model, allowing faster predictions with only a small drop in performance. Real-time tests showed a processing delay of about 54 to 67 ms per 1000 samples, demonstrating suitability for CPU-based edge deployment in microgrid controllers. The results confirm that lightweight machine learning based intrusion detection methods can provide fast, accurate, and efficient cyberattack detection without relying on complex deep learning models. Key contributions include: (1) development of a complete MATLAB-based virtual microgrid, (2) structured attack injection at the control layer, (3) creation of multiclass labeled datasets, and (4) design of low-cost AI models suitable for practical microgrid cybersecurity.



## **12. Engineering Attack Vectors and Detecting Anomalies in Additive Manufacturing**

cs.CR

This paper has been accepted to EAI SmartSP 2025. This is the preprint version

**SubmitDate**: 2026-01-01    [abs](http://arxiv.org/abs/2601.00384v1) [paper-pdf](https://arxiv.org/pdf/2601.00384v1)

**Confidence**: 0.95

**Authors**: Md Mahbub Hasan, Marcus Sternhagen, Krishna Chandra Roy

**Abstract**: Additive manufacturing (AM) is rapidly integrating into critical sectors such as aerospace, automotive, and healthcare. However, this cyber-physical convergence introduces new attack surfaces, especially at the interface between computer-aided design (CAD) and machine execution layers. In this work, we investigate targeted cyberattacks on two widely used fused deposition modeling (FDM) systems, Creality's flagship model K1 Max, and Ender 3. Our threat model is a multi-layered Man-in-the-Middle (MitM) intrusion, where the adversary intercepts and manipulates G-code files during upload from the user interface to the printer firmware. The MitM intrusion chain enables several stealthy sabotage scenarios. These attacks remain undetectable by conventional slicer software or runtime interfaces, resulting in structurally defective yet externally plausible printed parts. To counter these stealthy threats, we propose an unsupervised Intrusion Detection System (IDS) that analyzes structured machine logs generated during live printing. Our defense mechanism uses a frozen Transformer-based encoder (a BERT variant) to extract semantic representations of system behavior, followed by a contrastively trained projection head that learns anomaly-sensitive embeddings. Later, a clustering-based approach and a self-attention autoencoder are used for classification. Experimental results demonstrate that our approach effectively distinguishes between benign and compromised executions.



## **13. PHANTOM: Physics-Aware Adversarial Attacks against Federated Learning-Coordinated EV Charging Management System**

cs.ET

**SubmitDate**: 2025-12-26    [abs](http://arxiv.org/abs/2512.22381v1) [paper-pdf](https://arxiv.org/pdf/2512.22381v1)

**Confidence**: 0.95

**Authors**: Mohammad Zakaria Haider, Amit Kumar Podder, Prabin Mali, Aranya Chakrabortty, Sumit Paudyal, Mohammad Ashiqur Rahman

**Abstract**: The rapid deployment of electric vehicle charging stations (EVCS) within distribution networks necessitates intelligent and adaptive control to maintain the grid's resilience and reliability. In this work, we propose PHANTOM, a physics-aware adversarial network that is trained and optimized through a multi-agent reinforcement learning model. PHANTOM integrates a physics-informed neural network (PINN) enabled by federated learning (FL) that functions as a digital twin of EVCS-integrated systems, ensuring physically consistent modeling of operational dynamics and constraints. Building on this digital twin, we construct a multi-agent RL environment that utilizes deep Q-networks (DQN) and soft actor-critic (SAC) methods to derive adversarial false data injection (FDI) strategies capable of bypassing conventional detection mechanisms. To examine the broader grid-level consequences, a transmission and distribution (T and D) dual simulation platform is developed, allowing us to capture cascading interactions between EVCS disturbances at the distribution level and the operations of the bulk transmission system. Results demonstrate how learned attack policies disrupt load balancing and induce voltage instabilities that propagate across T and D boundaries. These findings highlight the critical need for physics-aware cybersecurity to ensure the resilience of large-scale vehicle-grid integration.



## **14. Satellite Cybersecurity Across Orbital Altitudes: Analyzing Ground-Based Threats to LEO, MEO, and GEO**

cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.21367v1) [paper-pdf](https://arxiv.org/pdf/2512.21367v1)

**Confidence**: 0.95

**Authors**: Mark Ballard, Guanqun Song, Ting Zhu

**Abstract**: The rapid proliferation of satellite constellations, particularly in Low Earth Orbit (LEO), has fundamentally altered the global space infrastructure, shifting the risk landscape from purely kinetic collisions to complex cyber-physical threats. While traditional safety frameworks focus on debris mitigation, ground-based adversaries increasingly exploit radio-frequency links, supply chain vulnerabilities, and software update pathways to degrade space assets. This paper presents a comparative analysis of satellite cybersecurity across LEO, Medium Earth Orbit (MEO), and Geostationary Earth Orbit (GEO) regimes. By synthesizing data from 60 publicly documented security incidents with key vulnerability proxies--including Telemetry, Tracking, and Command (TT&C) anomalies, encryption weaknesses, and environmental stressors--we characterize how orbital altitude dictates attack feasibility and impact. Our evaluation reveals distinct threat profiles: GEO systems are predominantly targeted via high-frequency uplink exposure, whereas LEO constellations face unique risks stemming from limited power budgets, hardware constraints, and susceptibility to thermal and radiation-induced faults. We further bridge the gap between security and sustainability, arguing that unmitigated cyber vulnerabilities accelerate hardware obsolescence and debris accumulation, undermining efforts toward carbon-neutral space operations. The results demonstrate that weak encryption and command path irregularities are the most consistent predictors of adversarial success across all orbits.



## **15. What You Read Isn't What You Hear: Linguistic Sensitivity in Deepfake Speech Detection**

cs.LG

15 pages, 2 fogures

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17513v1) [paper-pdf](https://arxiv.org/pdf/2505.17513v1)

**Confidence**: 0.95

**Authors**: Binh Nguyen, Shuji Shi, Ryan Ofman, Thai Le

**Abstract**: Recent advances in text-to-speech technologies have enabled realistic voice generation, fueling audio-based deepfake attacks such as fraud and impersonation. While audio anti-spoofing systems are critical for detecting such threats, prior work has predominantly focused on acoustic-level perturbations, leaving the impact of linguistic variation largely unexplored. In this paper, we investigate the linguistic sensitivity of both open-source and commercial anti-spoofing detectors by introducing transcript-level adversarial attacks. Our extensive evaluation reveals that even minor linguistic perturbations can significantly degrade detection accuracy: attack success rates surpass 60% on several open-source detector-voice pairs, and notably one commercial detection accuracy drops from 100% on synthetic audio to just 32%. Through a comprehensive feature attribution analysis, we identify that both linguistic complexity and model-level audio embedding similarity contribute strongly to detector vulnerability. We further demonstrate the real-world risk via a case study replicating the Brad Pitt audio deepfake scam, using transcript adversarial attacks to completely bypass commercial detectors. These results highlight the need to move beyond purely acoustic defenses and account for linguistic variation in the design of robust anti-spoofing systems. All source code will be publicly available.



## **16. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

cs.CR

(Last update!, a constructive comment from arxiv led to this latest update ) Stochastic investment models and a Bayesian approach to better modeling of uncertainty : adversarial machine learning or Stochastic market. arXiv admin note: substantial text overlap with arXiv:2402.05967 (see this link to the paper by : Orson Mengara)

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2406.10719v5) [paper-pdf](https://arxiv.org/pdf/2406.10719v5)

**Confidence**: 0.95

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.



## **17. SoK: Challenges in Tabular Membership Inference Attacks**

cs.LG

This paper is currently under review for the EuroS&P conference

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2601.15874v1) [paper-pdf](https://arxiv.org/pdf/2601.15874v1)

**Confidence**: 0.95

**Authors**: Cristina Pêra, Tânia Carvalho, Maxime Cordy, Luís Antunes

**Abstract**: Membership Inference Attacks (MIAs) are currently a dominant approach for evaluating privacy in machine learning applications. Despite their significance in identifying records belonging to the training dataset, several concerns remain unexplored, particularly with regard to tabular data. In this paper, first, we provide an extensive review and analysis of MIAs considering two main learning paradigms: centralized and federated learning. We extend and refine the taxonomy for both. Second, we demonstrate the efficacy of MIAs in tabular data using several attack strategies, also including defenses. Furthermore, in a federated learning scenario, we consider the threat posed by an outsider adversary, which is often neglected. Third, we demonstrate the high vulnerability of single-outs (records with a unique signature) to MIAs. Lastly, we explore how MIAs transfer across model architectures. Our results point towards a general poor performance of these attacks in tabular data which contrasts with previous state-of-the-art. Notably, even attacks with limited attack performance can still successfully expose a large portion of single-outs. Moreover, our findings suggest that using different surrogate models makes MIAs more effective.



## **18. Deep Leakage with Generative Flow Matching Denoiser**

cs.CV

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.15049v1) [paper-pdf](https://arxiv.org/pdf/2601.15049v1)

**Confidence**: 0.95

**Authors**: Isaac Baglin, Xiatian Zhu, Simon Hadfield

**Abstract**: Federated Learning (FL) has emerged as a powerful paradigm for decentralized model training, yet it remains vulnerable to deep leakage (DL) attacks that reconstruct private client data from shared model updates. While prior DL methods have demonstrated varying levels of success, they often suffer from instability, limited fidelity, or poor robustness under realistic FL settings. We introduce a new DL attack that integrates a generative Flow Matching (FM) prior into the reconstruction process. By guiding optimization toward the distribution of realistic images (represented by a flow matching foundation model), our method enhances reconstruction fidelity without requiring knowledge of the private data. Extensive experiments on multiple datasets and target models demonstrate that our approach consistently outperforms state-of-the-art attacks across pixel-level, perceptual, and feature-based similarity metrics. Crucially, the method remains effective across different training epochs, larger client batch sizes, and under common defenses such as noise injection, clipping, and sparsification. Our findings call for the development of new defense strategies that explicitly account for adversaries equipped with powerful generative priors.



## **19. Beyond Denial-of-Service: The Puppeteer's Attack for Fine-Grained Control in Ranking-Based Federated Learning**

cs.LG

12 pages. To appear in The Web Conference 2026

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.14687v1) [paper-pdf](https://arxiv.org/pdf/2601.14687v1)

**Confidence**: 0.95

**Authors**: Zhihao Chen, Zirui Gong, Jianting Ning, Yanjun Zhang, Leo Yu Zhang

**Abstract**: Federated Rank Learning (FRL) is a promising Federated Learning (FL) paradigm designed to be resilient against model poisoning attacks due to its discrete, ranking-based update mechanism. Unlike traditional FL methods that rely on model updates, FRL leverages discrete rankings as a communication parameter between clients and the server. This approach significantly reduces communication costs and limits an adversary's ability to scale or optimize malicious updates in the continuous space, thereby enhancing its robustness. This makes FRL particularly appealing for applications where system security and data privacy are crucial, such as web-based auction and bidding platforms. While FRL substantially reduces the attack surface, we demonstrate that it remains vulnerable to a new class of local model poisoning attack, i.e., fine-grained control attacks. We introduce the Edge Control Attack (ECA), the first fine-grained control attack tailored to ranking-based FL frameworks. Unlike conventional denial-of-service (DoS) attacks that cause conspicuous disruptions, ECA enables an adversary to precisely degrade a competitor's accuracy to any target level while maintaining a normal-looking convergence trajectory, thereby avoiding detection. ECA operates in two stages: (i) identifying and manipulating Ascending and Descending Edges to align the global model with the target model, and (ii) widening the selection boundary gap to stabilize the global model at the target accuracy. Extensive experiments across seven benchmark datasets and nine Byzantine-robust aggregation rules (AGRs) show that ECA achieves fine-grained accuracy control with an average error of only 0.224%, outperforming the baseline by up to 17x. Our findings highlight the need for stronger defenses against advanced poisoning attacks. Our code is available at: https://github.com/Chenzh0205/ECA



## **20. Gradient Structure Estimation under Label-Only Oracles via Spectral Sensitivity**

cs.LG

**SubmitDate**: 2026-01-17    [abs](http://arxiv.org/abs/2601.14300v1) [paper-pdf](https://arxiv.org/pdf/2601.14300v1)

**Confidence**: 0.95

**Authors**: Jun Liu, Leo Yu Zhang, Fengpeng Li, Isao Echizen, Jiantao Zhou

**Abstract**: Hard-label black-box settings, where only top-1 predicted labels are observable, pose a fundamentally constrained yet practically important feedback model for understanding model behavior. A central challenge in this regime is whether meaningful gradient information can be recovered from such discrete responses. In this work, we develop a unified theoretical perspective showing that a wide range of existing sign-flipping hard-label attacks can be interpreted as implicitly approximating the sign of the true loss gradient. This observation reframes hard-label attacks from heuristic search procedures into instances of gradient sign recovery under extremely limited feedback. Motivated by this first-principles understanding, we propose a new attack framework that combines a zero-query frequency-domain initialization with a Pattern-Driven Optimization (PDO) strategy. We establish theoretical guarantees demonstrating that, under mild assumptions, our initialization achieves higher expected cosine similarity to the true gradient sign compared to random baselines, while the proposed PDO procedure attains substantially lower query complexity than existing structured search approaches. We empirically validate our framework through extensive experiments on CIFAR-10, ImageNet, and ObjectNet, covering standard and adversarially trained models, commercial APIs, and CLIP-based models. The results show that our method consistently surpasses SOTA hard-label attacks in both attack success rate and query efficiency, particularly in low-query regimes. Beyond image classification, our approach generalizes effectively to corrupted data, biomedical datasets, and dense prediction tasks. Notably, it also successfully circumvents Blacklight, a SOTA stateful defense, resulting in a $0\%$ detection rate. Our code will be released publicly soon at https://github.com/csjunjun/DPAttack.git.



## **21. Adversarial Attacks on Medical Hyperspectral Imaging Exploiting Spectral-Spatial Dependencies and Multiscale Features**

cs.CV

**SubmitDate**: 2026-01-11    [abs](http://arxiv.org/abs/2601.07056v1) [paper-pdf](https://arxiv.org/pdf/2601.07056v1)

**Confidence**: 0.95

**Authors**: Yunrui Gu, Zhenzhe Gao, Cong Kong, Zhaoxia Yin

**Abstract**: Medical hyperspectral imaging (HSI) enables accurate disease diagnosis by capturing rich spectral-spatial tissue information, but recent advances in deep learning have exposed its vulnerability to adversarial attacks. In this work, we identify two fundamental causes of this fragility: the reliance on local pixel dependencies for preserving tissue structure and the dependence on multiscale spectral-spatial representations for hierarchical feature encoding. Building on these insights, we propose a targeted adversarial attack framework for medical HSI, consisting of a Local Pixel Dependency Attack that exploits spatial correlations among neighboring pixels, and a Multiscale Information Attack that perturbs features across hierarchical spectral-spatial scales. Experiments on the Brain and MDC datasets demonstrate that our attacks significantly degrade classification performance, especially in tumor regions, while remaining visually imperceptible. Compared with existing methods, our approach reveals the unique vulnerabilities of medical HSI models and underscores the need for robust, structure-aware defenses in clinical applications.



## **22. Sparse Neural Approximations for Bilevel Adversarial Problems in Power Grids**

eess.SY

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.06187v1) [paper-pdf](https://arxiv.org/pdf/2512.06187v1)

**Confidence**: 0.95

**Authors**: Young-ho Cho, Harsha Nagarajan, Deepjyoti Deka, Hao Zhu

**Abstract**: The adversarial worst-case load shedding (AWLS) problem is pivotal for identifying critical contingencies under line outages. It is naturally cast as a bilevel program: the upper level simulates an attacker determining worst-case line failures, and the lower level corresponds to the defender's generator redispatch operations. Conventional techniques using optimality conditions render the bilevel, mixed-integer formulation computationally prohibitive due to the combinatorial number of topologies and the nonconvexity of AC power flow constraints. To address these challenges, we develop a novel single-level optimal value-function (OVF) reformulation and further leverage a data-driven neural network (NN) surrogate of the follower's optimal value. To ensure physical realizability, we embed the trained surrogate in a physics-constrained NN (PCNN) formulation that couples the OVF inequality with (relaxed) AC feasibility, yielding a mixed-integer convex model amenable to off-the-shelf solvers. To achieve scalability, we learn a sparse, area-partitioned NN via spectral clustering; the resulting block-sparse architecture scales essentially linearly with system size while preserving accuracy. Notably, our approach produces near-optimal worst-case failures and generalizes across loading conditions and unseen topologies, enabling rapid online recomputation. Numerical experiments on the IEEE 14- and 118-bus systems demonstrate the method's scalability and solution quality for large-scale contingency analysis, with an average optimality gap of 5.8% compared to conventional methods, while maintaining computation times under one minute.



## **23. Towards Trustworthy Wi-Fi Sensing: Systematic Evaluation of Deep Learning Model Robustness to Adversarial Attacks**

cs.LG

19 pages, 8 figures, 7 tables

**SubmitDate**: 2025-11-25    [abs](http://arxiv.org/abs/2511.20456v1) [paper-pdf](https://arxiv.org/pdf/2511.20456v1)

**Confidence**: 0.95

**Authors**: Shreevanth Krishnaa Gopalakrishnan, Stephen Hailes

**Abstract**: Machine learning has become integral to Channel State Information (CSI)-based human sensing systems and is expected to power applications such as device-free activity recognition and identity detection in future cellular and Wi-Fi generations. However, these systems rely on models whose decisions can be subtly perturbed, raising concerns for security and reliability in ubiquitous sensing. Quantifying and understanding the robustness of such models, defined as their ability to maintain accurate predictions under adversarial perturbations, is therefore critical before wireless sensing can be safely deployed in real-world environments.   This work presents a systematic evaluation of the robustness of CSI deep learning models under diverse threat models (white-box, black-box/transfer, and universal perturbations) and varying degrees of attack realism. We establish a framework to compare compact temporal autoencoder models with larger deep architectures across three public datasets, quantifying how model scale, training regime, and physical constraints influence robustness. Our experiments show that smaller models, while efficient and equally performant on clean data, are markedly less robust. We further confirm that physically realizable signal-space perturbations, designed to be feasible in real wireless channels, significantly reduce attack success compared to unconstrained feature-space attacks. Adversarial training mitigates these vulnerabilities, improving mean robust accuracy with only moderate degradation in clean performance across both model classes. As wireless sensing advances towards reliable, cross-domain operation, these findings provide quantitative baselines for robustness estimation and inform design principles for secure and trustworthy human-centered sensing systems.



## **24. Algorithmic detection of false data injection attacks in cyber-physical systems**

math.OC

13 pages, 6 figures

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2511.18588v1) [paper-pdf](https://arxiv.org/pdf/2511.18588v1)

**Confidence**: 0.95

**Authors**: Souvik Das, Avishek Ghosh, Debasish Chatterjee

**Abstract**: This article introduces an anomaly detection based algorithm (AD-CPS) to detect false data injection attacks that fall under the category of data deception/integrity attacks, but with arbitrary information structure, in cyber-physical systems (CPSs) modeled as stochastic linear time-invariant systems. The core idea of this data-driven algorithm is based on the fact that an honest state (one not compromised by adversaries) generated by the CPS should concentrate near its weighted empirical mean of the immediate past samples. As the first theoretical result, we provide non-asymptotic guarantees on the false positive error incurred by the algorithm for attacks that are 2-step honest, referring to adversaries that act intermittently rather than successively. Moreover, we establish that for adversaries possessing a certain minimum energy, the false negative error incurred by AD-CPS is low. Extensive experiments were conducted on partially observed stochastic LTI systems to demonstrate these properties and to quantitatively compare AD-CPS with an optimal CUSUM-based test.



## **25. A Novel and Practical Universal Adversarial Perturbations against Deep Reinforcement Learning based Intrusion Detection Systems**

cs.CR

13 pages, 7 Figures,

**SubmitDate**: 2025-11-22    [abs](http://arxiv.org/abs/2511.18223v1) [paper-pdf](https://arxiv.org/pdf/2511.18223v1)

**Confidence**: 0.95

**Authors**: H. Zhang, L. Zhang, G. Epiphaniou, C. Maple

**Abstract**: Intrusion Detection Systems (IDS) play a vital role in defending modern cyber physical systems against increasingly sophisticated cyber threats. Deep Reinforcement Learning-based IDS, have shown promise due to their adaptive and generalization capabilities. However, recent studies reveal their vulnerability to adversarial attacks, including Universal Adversarial Perturbations (UAPs), which can deceive models with a single, input-agnostic perturbation. In this work, we propose a novel UAP attack against Deep Reinforcement Learning (DRL)-based IDS under the domain-specific constraints derived from network data rules and feature relationships. To the best of our knowledge, there is no existing study that has explored UAP generation for the DRL-based IDS. In addition, this is the first work that focuses on developing a UAP against a DRL-based IDS under realistic domain constraints based on not only the basic domain rules but also mathematical relations between the features. Furthermore, we enhance the evasion performance of the proposed UAP, by introducing a customized loss function based on the Pearson Correlation Coefficient, and we denote it as Customized UAP. To the best of our knowledge, this is also the first work using the PCC value in the UAP generation, even in the broader context. Four additional established UAP baselines are implemented for a comprehensive comparison. Experimental results demonstrate that our proposed Customized UAP outperforms two input-dependent attacks including Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), and four UAP baselines, highlighting its effectiveness for real-world adversarial scenarios.



## **26. BadPatches: Routing-aware Backdoor Attacks on Vision Mixture of Experts**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2505.01811v3) [paper-pdf](https://arxiv.org/pdf/2505.01811v3)

**Confidence**: 0.95

**Authors**: Cedric Chan, Jona te Lintelo, Stjepan Picek

**Abstract**: Mixture of Experts (MoE) architectures have gained popularity for reducing computational costs in deep neural networks by activating only a subset of parameters during inference. While this efficiency makes MoE attractive for vision tasks, the patch-based processing in vision models introduces new methods for adversaries to perform backdoor attacks. In this work, we investigate the vulnerability of vision MoE models for image classification, specifically the patch-based MoE (pMoE) models and MoE-based vision transformers, against backdoor attacks. We propose a novel routing-aware trigger application method BadPatches, which is designed for patch-based processing in vision MoE models. BadPatches applies triggers on image patches rather than on the entire image. We show that BadPatches achieves high attack success rates (ASRs) with lower poisoning rates than routing-agnostic triggers and is successful at poisoning rates as low as 0.01% with an ASR above 80% on pMoE. Moreover, BadPatches is still effective when an adversary does not have complete knowledge of the patch routing configuration of the considered models. Next, we explore how trigger design affects pMoE patch routing. Finally, we investigate fine-pruning as a defense. Results show that only the fine-tuning stage of fine-pruning removes the backdoor from the model.



## **27. Boosting Adversarial Transferability with Low-Cost Optimization via Maximin Expected Flatness**

cs.CV

Accepted by IEEE T-IFS

**SubmitDate**: 2026-01-14    [abs](http://arxiv.org/abs/2405.16181v3) [paper-pdf](https://arxiv.org/pdf/2405.16181v3)

**Confidence**: 0.95

**Authors**: Chunlin Qiu, Ang Li, Yiheng Duan, Shenyi Zhang, Yuanjie Zhang, Lingchen Zhao, Qian Wang

**Abstract**: Transfer-based attacks craft adversarial examples on white-box surrogate models and directly deploy them against black-box target models, offering model-agnostic and query-free threat scenarios. While flatness-enhanced methods have recently emerged to improve transferability by enhancing the loss surface flatness of adversarial examples, their divergent flatness definitions and heuristic attack designs suffer from unexamined optimization limitations and missing theoretical foundation, thus constraining their effectiveness and efficiency. This work exposes the severely imbalanced exploitation-exploration dynamics in flatness optimization, establishing the first theoretical foundation for flatness-based transferability and proposing a principled framework to overcome these optimization pitfalls. Specifically, we systematically unify fragmented flatness definitions across existing methods, revealing their imbalanced optimization limitations in over-exploration of sensitivity peaks or over-exploitation of local plateaus. To resolve these issues, we rigorously formalize average-case flatness and transferability gaps, proving that enhancing zeroth-order average-case flatness minimizes cross-model discrepancies. Building on this theory, we design a Maximin Expected Flatness (MEF) attack that enhances zeroth-order average-case flatness while balancing flatness exploration and exploitation. Extensive evaluations across 22 models and 24 current transfer-based attacks demonstrate MEF's superiority: it surpasses the state-of-the-art PGN attack by 4% in attack success rate at half the computational cost and achieves 8% higher success rate under the same budget. When combined with input augmentation, MEF attains 15% additional gains against defense-equipped models, establishing new robustness benchmarks. Our code is available at https://github.com/SignedQiu/MEFAttack.



## **28. A Measurement of Genuine Tor Traces for Realistic Website Fingerprinting**

cs.CR

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2404.07892v2) [paper-pdf](https://arxiv.org/pdf/2404.07892v2)

**Confidence**: 0.95

**Authors**: Rob Jansen, Ryan Wails, Aaron Johnson

**Abstract**: Website fingerprinting (WF) is a dangerous attack on web privacy because it enables an adversary to predict the website a user is visiting, despite the use of encryption, VPNs, or anonymizing networks such as Tor. Previous WF work almost exclusively uses synthetic datasets to evaluate the performance and estimate the feasibility of WF attacks despite evidence that synthetic data misrepresents the real world. In this paper we present GTT23, the first WF dataset of genuine Tor traces, which we obtain through a large-scale measurement of the Tor network and which is intended especially for WF. It represents real Tor user behavior better than any existing WF dataset, is larger than any existing WF dataset by at least an order of magnitude, and will help ground the future study of realistic WF attacks and defenses. In a detailed evaluation, we survey 28 WF datasets published since 2008 and compare their characteristics to those of GTT23. We discover common deficiencies of synthetic datasets that make them inferior to GTT23 for drawing meaningful conclusions about the effectiveness of WF attacks directed at real Tor users. We have made GTT23 available to promote reproducible research and to help inspire new directions for future work.



## **29. PINNsFailureRegion Localization and Refinement through White-box AdversarialAttack**

cs.LG

**SubmitDate**: 2025-11-23    [abs](http://arxiv.org/abs/2310.11789v2) [paper-pdf](https://arxiv.org/pdf/2310.11789v2)

**Confidence**: 0.95

**Authors**: Shengzhu Shi, Yao Li, Zhichang Guo, Boying Wu, Yang Zhao

**Abstract**: Physics-informed neural networks (PINNs) have shown great promise in solving partial differential equations (PDEs). However, vanilla PINNs often face challenges when solving complex PDEs, especially those involving multi-scale behaviors or solutions with sharp or oscillatory characteristics. To precisely and adaptively locate the critical regions that fail in the solving process we propose a sampling strategy grounded in white-box adversarial attacks, referred to as WbAR. WbAR search for failure regions in the direction of the loss gradient, thus directly locating the most critical positions. WbAR generates adversarial samples in a random walk manner and iteratively refines PINNs to guide the model's focus towards dynamically updated critical regions during training. We implement WbAR to the elliptic equation with multi-scale coefficients, Poisson equation with multi-peak solutions, high-dimensional Poisson equations, and Burgers equation with sharp solutions. The results demonstrate that WbAR can effectively locate and reduce failure regions. Moreover, WbAR is suitable for solving complex PDEs, since locating failure regions through adversarial attacks is independent of the size of failure regions or the complexity of the distribution.



## **30. Memory Backdoor Attacks on Neural Networks**

cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2411.14516v2) [paper-pdf](https://arxiv.org/pdf/2411.14516v2)

**Confidence**: 0.90

**Authors**: Eden Luzon, Guy Amit, Roy Weiss, Torsten Kraub, Alexandra Dmitrienko, Yisroel Mirsky

**Abstract**: Neural networks are often trained on proprietary datasets, making them attractive attack targets. We present a novel dataset extraction method leveraging an innovative training time backdoor attack, allowing a malicious federated learning server to systematically and deterministically extract complete client training samples through a simple indexing process. Unlike prior techniques, our approach guarantees exact data recovery rather than probabilistic reconstructions or hallucinations, provides precise control over which samples are memorized and how many, and shows high capacity and robustness. Infected models output data samples when they receive a patternbased index trigger, enabling systematic extraction of meaningful patches from each clients local data without disrupting global model utility. To address small model output sizes, we extract patches and then recombined them. The attack requires only a minor modification to the training code that can easily evade detection during client-side verification. Hence, this vulnerability represents a realistic FL supply-chain threat, where a malicious server can distribute modified training code to clients and later recover private data from their updates. Evaluations across classifiers, segmentation models, and large language models demonstrate that thousands of sensitive training samples can be recovered from client models with minimal impact on task performance, and a clients entire dataset can be stolen after multiple FL rounds. For instance, a medical segmentation dataset can be extracted with only a 3 percent utility drop. These findings expose a critical privacy vulnerability in FL systems, emphasizing the need for stronger integrity and transparency in distributed training pipelines.



## **31. Backdoor Attacks on Multi-modal Contrastive Learning**

cs.LG

**SubmitDate**: 2026-01-16    [abs](http://arxiv.org/abs/2601.11006v1) [paper-pdf](https://arxiv.org/pdf/2601.11006v1)

**Confidence**: 0.85

**Authors**: Simi D Kuniyilh, Rita Machacy

**Abstract**: Contrastive learning has become a leading self- supervised approach to representation learning across domains, including vision, multimodal settings, graphs, and federated learning. However, recent studies have shown that contrastive learning is susceptible to backdoor and data poisoning attacks. In these attacks, adversaries can manipulate pretraining data or model updates to insert hidden malicious behavior. This paper offers a thorough and comparative review of backdoor attacks in contrastive learning. It analyzes threat models, attack methods, target domains, and available defenses. We summarize recent advancements in this area, underline the specific vulnerabilities inherent to contrastive learning, and discuss the challenges and future research directions. Our findings have significant implications for the secure deployment of systems in industrial and distributed environments.



## **32. Abusing the Internet of Medical Things: Evaluating Threat Models and Forensic Readiness for Multi-Vector Attacks on Connected Healthcare Devices**

cs.CR

In review at IEEE Euro S&P 2026

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2601.12593v1) [paper-pdf](https://arxiv.org/pdf/2601.12593v1)

**Confidence**: 0.85

**Authors**: Isabel Straw, Akhil Polamarasetty, Mustafa Jaafar

**Abstract**: Individuals experiencing interpersonal violence (IPV), who depend on medical devices, represent a uniquely vulnerable population as healthcare technologies become increasingly connected. Despite rapid growth in MedTech innovation and "health-at-home" ecosystems, the intersection of MedTech cybersecurity and technology-facilitated abuse remains critically under-examined. IPV survivors who rely on therapeutic devices encounter a qualitatively different threat environment from the external, technically sophisticated adversaries typically modeled in MedTech cybersecurity research. We address this gap through two complementary methods: (1) the development of hazard-integrated threat models that fuse Cyber physical system security modeling with tech-abuse frameworks, and (2) an immersive simulation with practitioners, deploying a live version of our model, identifying gaps in digital forensic practice.   Our hazard-integrated CIA threat models map exploits to acute and chronic biological effects, uncovering (i) Integrity attack pathways that facilitate "Medical gaslighting" and "Munchausen-by-IoMT", (ii) Availability attacks that create life-critical and sub-acute harms (glycaemic emergencies, blindness, mood destabilization), and (iii) Confidentiality threats arising from MedTech advertisements (geolocation tracking from BLE broadcasts). Our simulation demonstrates that these attack surfaces are unlikely to be detected in practice: participants overlooked MedTech, misclassified reproductive and assistive technologies, and lacked awareness of BLE broadcast artifacts. Our findings show that MedTech cybersecurity in IPV contexts requires integrated threat modeling and improved forensic capabilities for detecting, preserving and interpreting harms arising from compromised patient-technology ecosystems.



## **33. Social Engineering Attacks: A Systemisation of Knowledge on People Against Humans**

cs.CR

10 pages, 6 Figures, 3 Tables

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2601.04215v1) [paper-pdf](https://arxiv.org/pdf/2601.04215v1)

**Confidence**: 0.85

**Authors**: Scott Thomson, Michael Bewong, Arash Mahboubi, Tanveer Zia

**Abstract**: Our systematisation of knowledge on Social Engineering Attacks (SEAs), identifies the human, organisational, and adversarial dimensions of cyber threats. It addresses the growing risks posed by SEAs, highly relevant in the context physical cyber places, such as travellers at airports and residents in smart cities, and synthesizes findings from peer reviewed studies, industry and government reports to inform effective countermeasures that can be embedded into future smart city strategies. SEAs increasingly sidestep technical controls by weaponising leaked personal data and behavioural cues, an urgency underscored by the Optus, Medibank and now Qantas (2025) mega breaches that placed millions of personal records in criminals' hands. Our review surfaces three critical dimensions: (i) human factors of knowledge, abilities and behaviours (KAB) (ii) organisational culture and informal norms that shape those behaviours and (iii) attacker motivations, techniques and return on investment calculations. Our contributions are threefold: (1) TriLayer Systematisation: to the best of our knowledge, we are the first to unify KAB metrics, cultural drivers and attacker economics into a single analytical lens, enabling practitioners to see how vulnerabilities, norms and threat incentives coevolve. (2) Risk Weighted HAISQ Meta analysis: By normalising and ranking HAISQ scores across recent field studies, we reveal persistent high risk clusters (Internet and Social Media use) and propose impact weightings that make the instrument predictive rather than descriptive. (3) Adaptive 'Segment and Simulate' Training Blueprint: Building on clustering evidence, we outline a differentiated programme that matches low, medium, high risk user cohorts to experiential learning packages including phishing simulations, gamified challenges and realtime feedback thereby aligning effort with measured exposure.



## **34. To See or Not to See -- Fingerprinting Devices in Adversarial Environments Amid Advanced Machine Learning**

cs.CR

10 pages, 4 figures

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2504.08264v2) [paper-pdf](https://arxiv.org/pdf/2504.08264v2)

**Confidence**: 0.85

**Authors**: Justin Feng, Amirmohammad Haddad, Nader Sehatbakhsh

**Abstract**: The increasing use of the Internet of Things raises security concerns. To address this, device fingerprinting is often employed to authenticate devices, detect adversaries, and identify eavesdroppers in an environment. This requires the ability to discern between legitimate and malicious devices which is achieved by analyzing the unique physical and/or operational characteristics of IoT devices. In the era of the latest progress in machine learning, particularly generative models, it is crucial to methodically examine the current studies in device fingerprinting. This involves explaining their approaches and underscoring their limitations when faced with adversaries armed with these ML tools. To systematically analyze existing methods, we propose a generic, yet simplified, model for device fingerprinting. Additionally, we thoroughly investigate existing methods to authenticate devices and detect eavesdropping, using our proposed model. We further study trends and similarities between works in authentication and eavesdropping detection and present the existing threats and attacks in these domains. Finally, we discuss future directions in fingerprinting based on these trends to develop more secure IoT fingerprinting schemes.



## **35. NoisyHate: Mining Online Human-Written Perturbations for Realistic Robustness Benchmarking of Content Moderation Models**

cs.LG

Accepted to International AAAI Conference on Web and Social Media (ICWSM 2025)

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2303.10430v2) [paper-pdf](https://arxiv.org/pdf/2303.10430v2)

**Confidence**: 0.85

**Authors**: Yiran Ye, Thai Le, Dongwon Lee

**Abstract**: Online texts with toxic content are a clear threat to the users on social media in particular and society in general. Although many platforms have adopted various measures (e.g., machine learning-based hate-speech detection systems) to diminish their effect, toxic content writers have also attempted to evade such measures by using cleverly modified toxic words, so-called human-written text perturbations. Therefore, to help build automatic detection tools to recognize those perturbations, prior methods have developed sophisticated techniques to generate diverse adversarial samples. However, we note that these ``algorithms"-generated perturbations do not necessarily capture all the traits of ``human"-written perturbations. Therefore, in this paper, we introduce a novel, high-quality dataset of human-written perturbations, named as NoisyHate, that was created from real-life perturbations that are both written and verified by human-in-the-loop. We show that perturbations in NoisyHate have different characteristics than prior algorithm-generated toxic datasets show, and thus can be in particular useful to help develop better toxic speech detection solutions. We thoroughly validate NoisyHate against state-of-the-art language models, such as BERT and RoBERTa, and black box APIs, such as Perspective API, on two tasks, such as perturbation normalization and understanding.



## **36. Serverless AI Security: Attack Surface Analysis and Runtime Protection Mechanisms for FaaS-Based Machine Learning**

cs.CR

17 Pages, 2 Figures, 4 Tables

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.11664v1) [paper-pdf](https://arxiv.org/pdf/2601.11664v1)

**Confidence**: 0.85

**Authors**: Chetan Pathade, Vinod Dhimam, Sheheryar Ahmad, Ilsa Lareb

**Abstract**: Serverless computing has achieved widespread adoption, with over 70% of AWS organizations using serverless solutions [1]. Meanwhile, machine learning inference workloads increasingly migrate to Function-as-a-Service (FaaS) platforms for their scalability and cost-efficiency [2], [3], [4]. However, this convergence introduces critical security challenges, with recent reports showing a 220% increase in AI/ML vulnerabilities [5] and serverless computing's fragmented architecture raises new security concerns distinct from traditional cloud deployments [6], [7]. This paper presents the first comprehensive security analysis of machine learning workloads in serverless environments. We systematically characterize the attack surface across five categories: function-level vulnerabilities (cold start exploitation, dependency poisoning), model-specific threats (API-based extraction, adversarial inputs), infrastructure attacks (cross-function contamination, privilege escalation), supply chain risks (malicious layers, backdoored libraries), and IAM complexity (ephemeral nature, serverless functions). Through empirical assessments across AWS Lambda, Azure Functions, and Google Cloud Functions, we demonstrate real-world attack scenarios and quantify their security impact. We propose Serverless AI Shield (SAS), a multi-layered defense framework providing pre-deployment validation, runtime monitoring, and post-execution forensics. Our evaluation shows SAS achieves 94% detection rates while maintaining performance overhead below 9% for inference latency. We release an open-source security toolkit to enable practitioners to assess and harden their serverless AI deployments, advancing the field toward more resilient cloud-native machine learning systems.



## **37. Constraint-Guided Prediction Refinement via Deterministic Diffusion Trajectories**

cs.AI

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2506.12911v2) [paper-pdf](https://arxiv.org/pdf/2506.12911v2)

**Confidence**: 0.85

**Authors**: Pantelis Dogoulis, Fabien Bernier, Félix Fourreau, Karim Tit, Maxime Cordy

**Abstract**: Many real-world machine learning tasks require outputs that satisfy hard constraints, such as physical conservation laws, structured dependencies in graphs, or column-level relationships in tabular data. Existing approaches rely either on domain-specific architectures and losses or on strong assumptions on the constraint space, restricting their applicability to linear or convex constraints. We propose a general-purpose framework for constraint-aware refinement that leverages denoising diffusion implicit models (DDIMs). Starting from a coarse prediction, our method iteratively refines it through a deterministic diffusion trajectory guided by a learned prior and augmented by constraint gradient corrections. The approach accommodates a wide class of non-convex and nonlinear equality constraints and can be applied post hoc to any base model. We demonstrate the method in two representative domains: constrained adversarial attack generation on tabular data with column-level dependencies and in AC power flow prediction under Kirchhoff's laws. Across both settings, our diffusion-guided refinement improves both constraint satisfaction and performance while remaining lightweight and model-agnostic.



