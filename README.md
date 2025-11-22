# Latest Adversarial Attack Papers
**update at 2025-11-22 11:18:45**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Large Language Model-Based Reward Design for Deep Reinforcement Learning-Driven Autonomous Cyber Defense**

cs.LG

Accepted in the AAAI-26 Workshop on Artificial Intelligence for Cyber Security (AICS)

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16483v1) [paper-pdf](https://arxiv.org/pdf/2511.16483v1)

**Authors**: Sayak Mukherjee, Samrat Chatterjee, Emilie Purvine, Ted Fujimoto, Tegan Emerson

**Abstract**: Designing rewards for autonomous cyber attack and defense learning agents in a complex, dynamic environment is a challenging task for subject matter experts. We propose a large language model (LLM)-based reward design approach to generate autonomous cyber defense policies in a deep reinforcement learning (DRL)-driven experimental simulation environment. Multiple attack and defense agent personas were crafted, reflecting heterogeneity in agent actions, to generate LLM-guided reward designs where the LLM was first provided with contextual cyber simulation environment information. These reward structures were then utilized within a DRL-driven attack-defense simulation environment to learn an ensemble of cyber defense policies. Our results suggest that LLM-guided reward designs can lead to effective defense strategies against diverse adversarial behaviors.



## **2. Q-MLLM: Vector Quantization for Robust Multimodal Large Language Model Security**

cs.CR

Accepted by NDSS 2026

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16229v1) [paper-pdf](https://arxiv.org/pdf/2511.16229v1)

**Authors**: Wei Zhao, Zhe Li, Yige Li, Jun Sun

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated impressive capabilities in cross-modal understanding, but remain vulnerable to adversarial attacks through visual inputs despite robust textual safety mechanisms. These vulnerabilities arise from two core weaknesses: the continuous nature of visual representations, which allows for gradient-based attacks, and the inadequate transfer of text-based safety mechanisms to visual content. We introduce Q-MLLM, a novel architecture that integrates two-level vector quantization to create a discrete bottleneck against adversarial attacks while preserving multimodal reasoning capabilities. By discretizing visual representations at both pixel-patch and semantic levels, Q-MLLM blocks attack pathways and bridges the cross-modal safety alignment gap. Our two-stage training methodology ensures robust learning while maintaining model utility. Experiments demonstrate that Q-MLLM achieves significantly better defense success rate against both jailbreak attacks and toxic image attacks than existing approaches. Notably, Q-MLLM achieves perfect defense success rate (100\%) against jailbreak attacks except in one arguable case, while maintaining competitive performance on multiple utility benchmarks with minimal inference overhead. This work establishes vector quantization as an effective defense mechanism for secure multimodal AI systems without requiring expensive safety-specific fine-tuning or detection overhead. Code is available at https://github.com/Amadeuszhao/QMLLM.



## **3. PSM: Prompt Sensitivity Minimization via LLM-Guided Black-Box Optimization**

cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16209v1) [paper-pdf](https://arxiv.org/pdf/2511.16209v1)

**Authors**: Huseein Jawad, Nicolas Brunel

**Abstract**: System prompts are critical for guiding the behavior of Large Language Models (LLMs), yet they often contain proprietary logic or sensitive information, making them a prime target for extraction attacks. Adversarial queries can successfully elicit these hidden instructions, posing significant security and privacy risks. Existing defense mechanisms frequently rely on heuristics, incur substantial computational overhead, or are inapplicable to models accessed via black-box APIs. This paper introduces a novel framework for hardening system prompts through shield appending, a lightweight approach that adds a protective textual layer to the original prompt. Our core contribution is the formalization of prompt hardening as a utility-constrained optimization problem. We leverage an LLM-as-optimizer to search the space of possible SHIELDs, seeking to minimize a leakage metric derived from a suite of adversarial attacks, while simultaneously preserving task utility above a specified threshold, measured by semantic fidelity to baseline outputs. This black-box, optimization-driven methodology is lightweight and practical, requiring only API access to the target and optimizer LLMs. We demonstrate empirically that our optimized SHIELDs significantly reduce prompt leakage against a comprehensive set of extraction attacks, outperforming established baseline defenses without compromising the model's intended functionality. Our work presents a paradigm for developing robust, utility-aware defenses in the escalating landscape of LLM security. The code is made public on the following link: https://github.com/psm-defense/psm



## **4. When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models**

cs.CV

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16203v1) [paper-pdf](https://arxiv.org/pdf/2511.16203v1)

**Authors**: Yuping Yan, Yuhan Xie, Yinxin Zhang, Lingjuan Lyu, Yaochu Jin

**Abstract**: Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.



## **5. An Image Is Worth Ten Thousand Words: Verbose-Text Induction Attacks on VLMs**

cs.CV

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16163v1) [paper-pdf](https://arxiv.org/pdf/2511.16163v1)

**Authors**: Zhi Luo, Zenghui Yuan, Wenqi Wei, Daizong Liu, Pan Zhou

**Abstract**: With the remarkable success of Vision-Language Models (VLMs) on multimodal tasks, concerns regarding their deployment efficiency have become increasingly prominent. In particular, the number of tokens consumed during the generation process has emerged as a key evaluation metric.Prior studies have shown that specific inputs can induce VLMs to generate lengthy outputs with low information density, which significantly increases energy consumption, latency, and token costs. However, existing methods simply delay the occurrence of the EOS token to implicitly prolong output, and fail to directly maximize the output token length as an explicit optimization objective, lacking stability and controllability.To address these limitations, this paper proposes a novel verbose-text induction attack (VTIA) to inject imperceptible adversarial perturbations into benign images via a two-stage framework, which identifies the most malicious prompt embeddings for optimizing and maximizing the output token of the perturbed images.Specifically, we first perform adversarial prompt search, employing reinforcement learning strategies to automatically identify adversarial prompts capable of inducing the LLM component within VLMs to produce verbose outputs. We then conduct vision-aligned perturbation optimization to craft adversarial examples on input images, maximizing the similarity between the perturbed image's visual embeddings and those of the adversarial prompt, thereby constructing malicious images that trigger verbose text generation. Comprehensive experiments on four popular VLMs demonstrate that our method achieves significant advantages in terms of effectiveness, efficiency, and generalization capability.



## **6. Layer-wise Noise Guided Selective Wavelet Reconstruction for Robust Medical Image Segmentation**

cs.CV

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16162v1) [paper-pdf](https://arxiv.org/pdf/2511.16162v1)

**Authors**: Yuting Lu, Ziliang Wang, Weixin Xu, Wei Zhang, Yongqiang Zhao, Yang Yu, Xiaohong Zhang

**Abstract**: Clinical deployment requires segmentation models to stay stable under distribution shifts and perturbations. The mainstream solution is adversarial training (AT) to improve robustness; however, AT often brings a clean--robustness trade-off and high training/tuning cost, which limits scalability and maintainability in medical imaging. We propose \emph{Layer-wise Noise-Guided Selective Wavelet Reconstruction (LNG-SWR)}. During training, we inject small, zero-mean noise at multiple layers to learn a frequency-bias prior that steers representations away from noise-sensitive directions. We then apply prior-guided selective wavelet reconstruction on the input/feature branch to achieve frequency adaptation: suppress noise-sensitive bands, enhance directional structures and shape cues, and stabilize boundary responses while maintaining spectral consistency. The framework is backbone-agnostic and adds low additional inference overhead. It can serve as a plug-in enhancement to AT and also improves robustness without AT. On CT and ultrasound datasets, under a unified protocol with PGD-$L_{\infty}/L_{2}$ and SSAH, LNG-SWR delivers consistent gains on clean Dice/IoU and significantly reduces the performance drop under strong attacks; combining LNG-SWR with AT yields additive gains. When combined with adversarial training, robustness improves further without sacrificing clean accuracy, indicating an engineering-friendly and scalable path to robust segmentation. These results indicate that LNG-SWR provides a simple, effective, and engineering-friendly path to robust medical image segmentation in both adversarial and standard training regimes.



## **7. SceneGuard: Training-Time Voice Protection with Scene-Consistent Audible Background Noise**

cs.SD

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16114v1) [paper-pdf](https://arxiv.org/pdf/2511.16114v1)

**Authors**: Rui Sang, Yuxuan Liu

**Abstract**: Voice cloning technology poses significant privacy threats by enabling unauthorized speech synthesis from limited audio samples. Existing defenses based on imperceptible adversarial perturbations are vulnerable to common audio preprocessing such as denoising and compression. We propose SceneGuard, a training-time voice protection method that applies scene-consistent audible background noise to speech recordings. Unlike imperceptible perturbations, SceneGuard leverages naturally occurring acoustic scenes (e.g., airport, street, park) to create protective noise that is contextually appropriate and robust to countermeasures. We evaluate SceneGuard on text-to-speech training attacks, demonstrating 5.5% speaker similarity degradation with extremely high statistical significance (p < 10^{-15}, Cohen's d = 2.18) while preserving 98.6% speech intelligibility (STOI = 0.986). Robustness evaluation shows that SceneGuard maintains or enhances protection under five common countermeasures including MP3 compression, spectral subtraction, lowpass filtering, and downsampling. Our results suggest that audible, scene-consistent noise provides a more robust alternative to imperceptible perturbations for training-time voice protection. The source code are available at: https://github.com/richael-sang/SceneGuard.



## **8. Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models**

cs.CR

AAAI 2026 Oral

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16110v1) [paper-pdf](https://arxiv.org/pdf/2511.16110v1)

**Authors**: Yijun Yang, Lichao Wang, Jianping Zhang, Chi Harold Liu, Lanqing Hong, Qiang Xu

**Abstract**: The growing misuse of Vision-Language Models (VLMs) has led providers to deploy multiple safeguards, including alignment tuning, system prompts, and content moderation. However, the real-world robustness of these defenses against adversarial attacks remains underexplored. We introduce Multi-Faceted Attack (MFA), a framework that systematically exposes general safety vulnerabilities in leading defense-equipped VLMs such as GPT-4o, Gemini-Pro, and Llama-4. The core component of MFA is the Attention-Transfer Attack (ATA), which hides harmful instructions inside a meta task with competing objectives. We provide a theoretical perspective based on reward hacking to explain why this attack succeeds. To improve cross-model transferability, we further introduce a lightweight transfer-enhancement algorithm combined with a simple repetition strategy that jointly bypasses both input-level and output-level filters without model-specific fine-tuning. Empirically, we show that adversarial images optimized for one vision encoder transfer broadly to unseen VLMs, indicating that shared visual representations create a cross-model safety vulnerability. Overall, MFA achieves a 58.5% success rate and consistently outperforms existing methods. On state-of-the-art commercial models, MFA reaches a 52.8% success rate, surpassing the second-best attack by 34%. These results challenge the perceived robustness of current defense mechanisms and highlight persistent safety weaknesses in modern VLMs. Code: https://github.com/cure-lab/MultiFacetedAttack



## **9. Future-Back Threat Modeling: A Foresight-Driven Security Framework**

cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16088v1) [paper-pdf](https://arxiv.org/pdf/2511.16088v1)

**Authors**: Vu Van Than

**Abstract**: Traditional threat modeling remains reactive-focused on known TTPs and past incident data, while threat prediction and forecasting frameworks are often disconnected from operational or architectural artifacts. This creates a fundamental weakness: the most serious cyber threats often do not arise from what is known, but from what is assumed, overlooked, or not yet conceived, and frequently originate from the future, such as artificial intelligence, information warfare, and supply chain attacks, where adversaries continuously develop new exploits that can bypass defenses built on current knowledge. To address this mental gap, this paper introduces the theory and methodology of Future-Back Threat Modeling (FBTM). This predictive approach begins with envisioned future threat states and works backward to identify assumptions, gaps, blind spots, and vulnerabilities in the current defense architecture, providing a clearer and more accurate view of impending threats so that we can anticipate their emergence and shape the future we want through actions taken now. The proposed methodology further aims to reveal known unknowns and unknown unknowns, including tactics, techniques, and procedures that are emerging, anticipated, and plausible. This enhances the predictability of adversary behavior, particularly under future uncertainty, helping security leaders make informed decisions today that shape more resilient security postures for the future.



## **10. Physically Realistic Sequence-Level Adversarial Clothing for Robust Human-Detection Evasion**

cs.CV

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16020v1) [paper-pdf](https://arxiv.org/pdf/2511.16020v1)

**Authors**: Dingkun Zhou, Patrick P. K. Chan, Hengxu Wu, Shikang Zheng, Ruiqi Huang, Yuanjie Zhao

**Abstract**: Deep neural networks used for human detection are highly vulnerable to adversarial manipulation, creating safety and privacy risks in real surveillance environments. Wearable attacks offer a realistic threat model, yet existing approaches usually optimize textures frame by frame and therefore fail to maintain concealment across long video sequences with motion, pose changes, and garment deformation. In this work, a sequence-level optimization framework is introduced to generate natural, printable adversarial textures for shirts, trousers, and hats that remain effective throughout entire walking videos in both digital and physical settings. Product images are first mapped to UV space and converted into a compact palette and control-point parameterization, with ICC locking to keep all colors printable. A physically based human-garment pipeline is then employed to simulate motion, multi-angle camera viewpoints, cloth dynamics, and illumination variation. An expectation-over-transformation objective with temporal weighting is used to optimize the control points so that detection confidence is minimized across whole sequences. Extensive experiments demonstrate strong and stable concealment, high robustness to viewpoint changes, and superior cross-model transferability. Physical garments produced with sublimation printing achieve reliable suppression under indoor and outdoor recordings, confirming real-world feasibility.



## **11. Nonadaptive One-Way to Hiding Implies Adaptive Quantum Reprogramming**

quant-ph

24 pages, 12 figures

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.16009v1) [paper-pdf](https://arxiv.org/pdf/2511.16009v1)

**Authors**: Joseph Jaeger

**Abstract**: An important proof technique in the random oracle model involves reprogramming it on hard to predict inputs and arguing that an attacker cannot detect that this occurred. In the quantum setting, a particularly challenging version of this considers adaptive reprogramming wherein the points to be reprogrammed (or the output values they should be programmed to) are dependent on choices made by the adversary. Some quantum frameworks for analyzing adaptive reprogramming were given by Unruh (CRYPTO 2014, EUROCRYPT 2015), Grilo-Hövelmanns-Hülsing-Majenz (ASIACRYPT 2021), and Pan-Zeng (PKC 2024). We show, counterintuitively, that these adaptive results follow from the \emph{nonadaptive} one-way to hiding theorem of Ambainis-Hamburg-Unruh (CRYPTO 2019). These implications contradict beliefs (whether stated explicitly or implicitly) that some properties of the adaptive frameworks cannot be provided by the Ambainis-Hamburg-Unruh result.



## **12. Lifefin: Escaping Mempool Explosions in DAG-based BFT**

cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15936v1) [paper-pdf](https://arxiv.org/pdf/2511.15936v1)

**Authors**: Jianting Zhang, Sen Yang, Alberto Sonnino, Sebastián Loza, Aniket Kate

**Abstract**: Directed Acyclic Graph (DAG)-based Byzantine Fault-Tolerant (BFT) protocols have emerged as promising solutions for high-throughput blockchains. By decoupling data dissemination from transaction ordering and constructing a well-connected DAG in the mempool, these protocols enable zero-message ordering and implicit view changes. However, we identify a fundamental liveness vulnerability: an adversary can trigger mempool explosions to prevent transaction commitment, ultimately compromising the protocol's liveness.   In response, this work presents Lifefin, a generic and self-stabilizing protocol designed to integrate seamlessly with existing DAG-based BFT protocols and circumvent such vulnerabilities. Lifefin leverages the Agreement on Common Subset (ACS) mechanism, allowing nodes to escape mempool explosions by committing transactions with bounded resource usage even in adverse conditions. As a result, Lifefin imposes (almost) zero overhead in typical cases while effectively eliminating liveness vulnerabilities.   To demonstrate the effectiveness of Lifefin, we integrate it into two state-of-the-art DAG-based BFT protocols, Sailfish and Mysticeti, resulting in two enhanced variants: Sailfish-Lifefin and Mysticeti-Lifefin. We implement these variants and compare them with the original Sailfish and Mysticeti systems. Our evaluation demonstrates that Lifefin achieves comparable transaction throughput while introducing only minimal additional latency to resist similar attacks.



## **13. Cyber-Resilient Data-Driven Event-Triggered Secure Control for Autonomous Vehicles Under False Data Injection Attacks**

eess.SY

14 pages, 8 figures

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15925v1) [paper-pdf](https://arxiv.org/pdf/2511.15925v1)

**Authors**: Yashar Mousavi, Mahsa Tavasoli, Ibrahim Beklan Kucukdemiral, Umit Cali, Abdolhossein Sarrafzadeh, Ali Karimoddini, Afef Fekih

**Abstract**: This paper proposes a cyber-resilient secure control framework for autonomous vehicles (AVs) subject to false data injection (FDI) threats as actuator attacks. The framework integrates data-driven modeling, event-triggered communication, and fractional-order sliding mode control (FSMC) to enhance the resilience against adversarial interventions. A dynamic model decomposition (DMD)-based methodology is employed to extract the lateral dynamics from real-world data, eliminating the reliance on conventional mechanistic modeling. To optimize communication efficiency, an event-triggered transmission scheme is designed to reduce the redundant transmissions while ensuring system stability. Furthermore, an extended state observer (ESO) is developed for real-time estimation and mitigation of actuator attack effects. Theoretical stability analysis, conducted using Lyapunov methods and linear matrix inequality (LMI) formulations, guarantees exponential error convergence. Extensive simulations validate the proposed event-triggered secure control framework, demonstrating substantial improvements in attack mitigation, communication efficiency, and lateral tracking performance. The results show that the framework effectively counteracts actuator attacks while optimizing communication-resource utilization, making it highly suitable for safety-critical AV applications.



## **14. TopoReformer: Mitigating Adversarial Attacks Using Topological Purification in OCR Models**

cs.LG

Accepted at AAAI 2026 AI for CyberSecurity (AICS) Workshop

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15807v1) [paper-pdf](https://arxiv.org/pdf/2511.15807v1)

**Authors**: Bhagyesh Kumar, A S Aravinthakashan, Akshat Satyanarayan, Ishaan Gakhar, Ujjwal Verma

**Abstract**: Adversarially perturbed images of text can cause sophisticated OCR systems to produce misleading or incorrect transcriptions from seemingly invisible changes to humans. Some of these perturbations even survive physical capture, posing security risks to high-stakes applications such as document processing, license plate recognition, and automated compliance systems. Existing defenses, such as adversarial training, input preprocessing, or post-recognition correction, are often model-specific, computationally expensive, and affect performance on unperturbed inputs while remaining vulnerable to unseen or adaptive attacks. To address these challenges, TopoReformer is introduced, a model-agnostic reformation pipeline that mitigates adversarial perturbations while preserving the structural integrity of text images. Topology studies properties of shapes and spaces that remain unchanged under continuous deformations, focusing on global structures such as connectivity, holes, and loops rather than exact distance. Leveraging these topological features, TopoReformer employs a topological autoencoder to enforce manifold-level consistency in latent space and improve robustness without explicit gradient regularization. The proposed method is benchmarked on EMNIST, MNIST, against standard adversarial attacks (FGSM, PGD, Carlini-Wagner), adaptive attacks (EOT, BDPA), and an OCR-specific watermark attack (FAWA).



## **15. Transferable Dual-Domain Feature Importance Attack against AI-Generated Image Detector**

cs.CV

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15571v1) [paper-pdf](https://arxiv.org/pdf/2511.15571v1)

**Authors**: Weiheng Zhu, Gang Cao, Jing Liu, Lifang Yu, Shaowei Weng

**Abstract**: Recent AI-generated image (AIGI) detectors achieve impressive accuracy under clean condition. In view of antiforensics, it is significant to develop advanced adversarial attacks for evaluating the security of such detectors, which remains unexplored sufficiently. This letter proposes a Dual-domain Feature Importance Attack (DuFIA) scheme to invalidate AIGI detectors to some extent. Forensically important features are captured by the spatially interpolated gradient and frequency-aware perturbation. The adversarial transferability is enhanced by jointly modeling spatial and frequency-domain feature importances, which are fused to guide the optimization-based adversarial example generation. Extensive experiments across various AIGI detectors verify the cross-model transferability, transparency and robustness of DuFIA.



## **16. Beluga: Block Synchronization for BFT Consensus Protocols**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15517v1) [paper-pdf](https://arxiv.org/pdf/2511.15517v1)

**Authors**: Tasos Kichidis, Lefteris Kokoris-Kogias, Arun Koshy, Ilya Sergey, Alberto Sonnino, Mingwei Tian, Jianting Zhang

**Abstract**: Modern high-throughput BFT consensus protocols use streamlined push-pull mechanisms to disseminate blocks and keep happy-path performance optimal. Yet state-of-the-art designs lack a principled and efficient way to exchange blocks, which leaves them open to targeted attacks and performance collapse under network asynchrony. This work introduces the concept of a block synchronizer, a simple abstraction that drives incremental block retrieval and enforces resource-aware exchange. Its interface and role fit cleanly inside a modern BFT consensus stack. We also uncover a new attack, where an adversary steers honest validators into redundant, uncoordinated pulls that exhaust bandwidth and stall progress. Beluga is a modular and scarcity-aware instantiation of the block synchronizer. It achieves optimal common-case latency while bounding the cost of recovery under faults and adversarial behavior. We integrate Beluga into Mysticeti, the consensus core of the Sui blockchain, and show on a geo-distributed AWS deployment that Beluga sustains optimal performance in the optimistic path and, under attack, delivers up to 3x higher throughput and 25x lower latency than prior designs. The Sui blockchain adopted Beluga in production.



## **17. HV-Attack: Hierarchical Visual Attack for Multimodal Retrieval Augmented Generation**

cs.CV

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15435v1) [paper-pdf](https://arxiv.org/pdf/2511.15435v1)

**Authors**: Linyin Luo, Yujuan Ding, Yunshan Ma, Wenqi Fan, Hanjiang Lai

**Abstract**: Advanced multimodal Retrieval-Augmented Generation (MRAG) techniques have been widely applied to enhance the capabilities of Large Multimodal Models (LMMs), but they also bring along novel safety issues. Existing adversarial research has revealed the vulnerability of MRAG systems to knowledge poisoning attacks, which fool the retriever into recalling injected poisoned contents. However, our work considers a different setting: visual attack of MRAG by solely adding imperceptible perturbations at the image inputs of users, without manipulating any other components. This is challenging due to the robustness of fine-tuned retrievers and large-scale generators, and the effect of visual perturbation may be further weakened by propagation through the RAG chain. We propose a novel Hierarchical Visual Attack that misaligns and disrupts the two inputs (the multimodal query and the augmented knowledge) of MRAG's generator to confuse its generation. We further design a hierarchical two-stage strategy to obtain misaligned augmented knowledge. We disrupt the image input of the retriever to make it recall irrelevant knowledge from the original database, by optimizing the perturbation which first breaks the cross-modal alignment and then disrupts the multimodal semantic alignment. We conduct extensive experiments on two widely-used MRAG datasets: OK-VQA and InfoSeek. We use CLIP-based retrievers and two LMMs BLIP-2 and LLaVA as generators. Results demonstrate the effectiveness of our visual attack on MRAG through the significant decrease in both retrieval and generation performance.



## **18. Adversarial Poetry as a Universal Single-Turn Jailbreak Mechanism in Large Language Models**

cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.15304v2) [paper-pdf](https://arxiv.org/pdf/2511.15304v2)

**Authors**: Piercosma Bisconti, Matteo Prandi, Federico Pierucci, Francesco Giarrusso, Marcantonio Bracale, Marcello Galisai, Vincenzo Suriani, Olga Sorokoletova, Federico Sartore, Daniele Nardi

**Abstract**: We present evidence that adversarial poetry functions as a universal single-turn jailbreak technique for Large Language Models (LLMs). Across 25 frontier proprietary and open-weight models, curated poetic prompts yielded high attack-success rates (ASR), with some providers exceeding 90%. Mapping prompts to MLCommons and EU CoP risk taxonomies shows that poetic attacks transfer across CBRN, manipulation, cyber-offence, and loss-of-control domains. Converting 1,200 MLCommons harmful prompts into verse via a standardized meta-prompt produced ASRs up to 18 times higher than their prose baselines. Outputs are evaluated using an ensemble of 3 open-weight LLM judges, whose binary safety assessments were validated on a stratified human-labeled subset. Poetic framing achieved an average jailbreak success rate of 62% for hand-crafted poems and approximately 43% for meta-prompt conversions (compared to non-poetic baselines), substantially outperforming non-poetic baselines and revealing a systematic vulnerability across model families and safety training approaches. These findings demonstrate that stylistic variation alone can circumvent contemporary safety mechanisms, suggesting fundamental limitations in current alignment methods and evaluation protocols.



## **19. Securing AI Agents Against Prompt Injection Attacks**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15759v1) [paper-pdf](https://arxiv.org/pdf/2511.15759v1)

**Authors**: Badrinath Ramakrishnan, Akshaya Balaji

**Abstract**: Retrieval-augmented generation (RAG) systems have become widely used for enhancing large language model capabilities, but they introduce significant security vulnerabilities through prompt injection attacks. We present a comprehensive benchmark for evaluating prompt injection risks in RAG-enabled AI agents and propose a multi-layered defense framework. Our benchmark includes 847 adversarial test cases across five attack categories: direct injection, context manipulation, instruction override, data exfiltration, and cross-context contamination. We evaluate three defense mechanisms: content filtering with embedding-based anomaly detection, hierarchical system prompt guardrails, and multi-stage response verification, across seven state-of-the-art language models. Our combined framework reduces successful attack rates from 73.2% to 8.7% while maintaining 94.3% of baseline task performance. We release our benchmark dataset and defense implementation to support future research in AI agent security.



## **20. Adversarial Attack on Black-Box Multi-Agent by Adaptive Perturbation**

cs.MA

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15292v1) [paper-pdf](https://arxiv.org/pdf/2511.15292v1)

**Authors**: Jianming Chen, Yawen Wang, Junjie Wang, Xiaofei Xie, Yuanzhe Hu, Qing Wang, Fanjiang Xu

**Abstract**: Evaluating security and reliability for multi-agent systems (MAS) is urgent as they become increasingly prevalent in various applications. As an evaluation technique, existing adversarial attack frameworks face certain limitations, e.g., impracticality due to the requirement of white-box information or high control authority, and a lack of stealthiness or effectiveness as they often target all agents or specific fixed agents. To address these issues, we propose AdapAM, a novel framework for adversarial attacks on black-box MAS. AdapAM incorporates two key components: (1) Adaptive Selection Policy simultaneously selects the victim and determines the anticipated malicious action (the action would lead to the worst impact on MAS), balancing effectiveness and stealthiness. (2) Proxy-based Perturbation to Induce Malicious Action utilizes generative adversarial imitation learning to approximate the target MAS, allowing AdapAM to generate perturbed observations using white-box information and thus induce victims to execute malicious action in black-box settings. We evaluate AdapAM across eight multi-agent environments and compare it with four state-of-the-art and commonly-used baselines. Results demonstrate that AdapAM achieves the best attack performance in different perturbation rates. Besides, AdapAM-generated perturbations are the least noisy and hardest to detect, emphasizing the stealthiness.



## **21. The Walls Have Ears: Unveiling Cross-Chain Sandwich Attacks in DeFi**

cs.CE

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15245v1) [paper-pdf](https://arxiv.org/pdf/2511.15245v1)

**Authors**: Chuanlei Li, Zhicheng Sun, Jing Xin Yuu, Xuechao Wang

**Abstract**: Cross-chain interoperability is a core component of modern blockchain infrastructure, enabling seamless asset transfers and composable applications across multiple blockchain ecosystems. However, the transparency of cross-chain messages can inadvertently expose sensitive transaction information, creating opportunities for adversaries to exploit value through manipulation or front-running strategies.   In this work, we investigate cross-chain sandwich attacks targeting liquidity pool-based cross-chain bridge protocols. We uncover a critical vulnerability where attackers can exploit events emitted on the source chain to learn transaction details on the destination chain before they appear in the destination chain mempool. This information advantage allows attackers to strategically place front-running and back-running transactions, ensuring that their front-running transactions always precede those of existing MEV bots monitoring the mempool of the destination chain. Moreover, current sandwich-attack defenses are ineffective against this new cross-chain variant. To quantify this threat, we conduct an empirical study using two months (August 10 to October 10, 2025) of cross-chain transaction data from the Symbiosis protocol and a tailored heuristic detection model. Our analysis identifies attacks that collectively garnered over \(5.27\) million USD in profit, equivalent to 1.28\% of the total bridged volume.



## **22. Trustworthy GenAI over 6G: Integrated Applications and Security Frameworks**

cs.CR

8 pages, 5 figures. Submitted for publication

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15206v1) [paper-pdf](https://arxiv.org/pdf/2511.15206v1)

**Authors**: Bui Duc Son, Trinh Van Chien, Dong In Kim

**Abstract**: The integration of generative artificial intelligence (GenAI) into 6G networks promises substantial performance gains while simultaneously exposing novel security vulnerabilities rooted in multimodal data processing and autonomous reasoning. This article presents a unified perspective on cross-domain vulnerabilities that arise across integrated sensing and communication (ISAC), federated learning (FL), digital twins (DTs), diffusion models (DMs), and large telecommunication models (LTMs). We highlight emerging adversarial agents such as compromised DTs and LTMs that can manipulate both the physical and cognitive layers of 6G systems. To address these risks, we propose an adaptive evolutionary defense (AED) concept that continuously co-evolves with attacks through GenAI-driven simulation and feedback, combining physical-layer protection, secure learning pipelines, and cognitive-layer resilience. A case study using an LLM-based port prediction model for fluid-antenna systems demonstrates the susceptibility of GenAI modules to adversarial perturbations and the effectiveness of the proposed defense concept. Finally, we summarize open challenges and future research directions toward building trustworthy, quantum-resilient, and adaptive GenAI-enabled 6G networks.



## **23. Effective Code Membership Inference for Code Completion Models via Adversarial Prompts**

cs.SE

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.15107v1) [paper-pdf](https://arxiv.org/pdf/2511.15107v1)

**Authors**: Yuan Jiang, Zehao Li, Shan Huang, Christoph Treude, Xiaohong Su, Tiantian Wang

**Abstract**: Membership inference attacks (MIAs) on code completion models offer an effective way to assess privacy risks by inferring whether a given code snippet was part of the training data. Existing black- and gray-box MIAs rely on expensive surrogate models or manually crafted heuristic rules, which limit their ability to capture the nuanced memorization patterns exhibited by over-parameterized code language models. To address these challenges, we propose AdvPrompt-MIA, a method specifically designed for code completion models, combining code-specific adversarial perturbations with deep learning. The core novelty of our method lies in designing a series of adversarial prompts that induce variations in the victim code model's output. By comparing these outputs with the ground-truth completion, we construct feature vectors to train a classifier that automatically distinguishes member from non-member samples. This design allows our method to capture richer memorization patterns and accurately infer training set membership. We conduct comprehensive evaluations on widely adopted models, such as Code Llama 7B, over the APPS and HumanEval benchmarks. The results show that our approach consistently outperforms state-of-the-art baselines, with AUC gains of up to 102%. In addition, our method exhibits strong transferability across different models and datasets, underscoring its practical utility and generalizability.



## **24. Critical Evaluation of Quantum Machine Learning for Adversarial Robustness**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.14989v1) [paper-pdf](https://arxiv.org/pdf/2511.14989v1)

**Authors**: Saeefa Rubaiyet Nowmi, Jesus Lopez, Md Mahmudul Alam Imon, Shahrooz Pouryouse, Mohammad Saidur Rahman

**Abstract**: Quantum Machine Learning (QML) integrates quantum computational principles into learning algorithms, offering improved representational capacity and computational efficiency. Nevertheless, the security and robustness of QML systems remain underexplored, especially under adversarial conditions. In this paper, we present a systematization of adversarial robustness in QML, integrating conceptual organization with empirical evaluation across three threat models-black-box, gray-box, and white-box. We implement representative attacks in each category, including label-flipping for black-box, QUID encoder-level data poisoning for gray-box, and FGSM and PGD for white-box, using Quantum Neural Networks (QNNs) trained on two datasets from distinct domains: MNIST from computer vision and AZ-Class from Android malware, across multiple circuit depths (2, 5, 10, and 50 layers) and two encoding schemes (angle and amplitude). Our evaluation shows that amplitude encoding yields the highest clean accuracy (93% on MNIST and 67% on AZ-Class) in deep, noiseless circuits; however, it degrades sharply under adversarial perturbations and depolarization noise (p=0.01), dropping accuracy below 5%. In contrast, angle encoding, while offering lower representational capacity, remains more stable in shallow, noisy regimes, revealing a trade-off between capacity and robustness. Moreover, the QUID attack attains higher attack success rates, though quantum noise channels disrupt the Hilbert-space correlations it exploits, weakening its impact in image domains. This suggests that noise can act as a natural defense mechanism in Noisy Intermediate-Scale Quantum (NISQ) systems. Overall, our findings guide the development of secure and resilient QML architectures for practical deployment. These insights underscore the importance of designing threat-aware models that remain reliable under real-world noise in NISQ settings.



## **25. Attacking Autonomous Driving Agents with Adversarial Machine Learning: A Holistic Evaluation with the CARLA Leaderboard**

cs.CR

12 pages

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14876v1) [paper-pdf](https://arxiv.org/pdf/2511.14876v1)

**Authors**: Henry Wong, Clement Fung, Weiran Lin, Karen Li, Stanley Chen, Lujo Bauer

**Abstract**: To autonomously control vehicles, driving agents use outputs from a combination of machine-learning (ML) models, controller logic, and custom modules. Although numerous prior works have shown that adversarial examples can mislead ML models used in autonomous driving contexts, it remains unclear if these attacks are effective at producing harmful driving actions for various agents, environments, and scenarios.   To assess the risk of adversarial examples to autonomous driving, we evaluate attacks against a variety of driving agents, rather than against ML models in isolation. To support this evaluation, we leverage CARLA, an urban driving simulator, to create and evaluate adversarial examples. We create adversarial patches designed to stop or steer driving agents, stream them into the CARLA simulator at runtime, and evaluate them against agents from the CARLA Leaderboard, a public repository of best-performing autonomous driving agents from an annual research competition. Unlike prior work, we evaluate attacks against autonomous driving systems without creating or modifying any driving-agent code and against all parts of the agent included with the ML model.   We perform a case-study investigation of two attack strategies against three open-source driving agents from the CARLA Leaderboard across multiple driving scenarios, lighting conditions, and locations. Interestingly, we show that, although some attacks can successfully mislead ML models into predicting erroneous stopping or steering commands, some driving agents use modules, such as PID control or GPS-based rules, that can overrule attacker-manipulated predictions from ML models.



## **26. FLARE: Adaptive Multi-Dimensional Reputation for Robust Client Reliability in Federated Learning**

cs.LG

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.14715v2) [paper-pdf](https://arxiv.org/pdf/2511.14715v2)

**Authors**: Abolfazl Younesi, Leon Kiss, Zahra Najafabadi Samani, Juan Aznar Poveda, Thomas Fahringer

**Abstract**: Federated learning (FL) enables collaborative model training while preserving data privacy. However, it remains vulnerable to malicious clients who compromise model integrity through Byzantine attacks, data poisoning, or adaptive adversarial behaviors. Existing defense mechanisms rely on static thresholds and binary classification, failing to adapt to evolving client behaviors in real-world deployments. We propose FLARE, an adaptive reputation-based framework that transforms client reliability assessment from binary decisions to a continuous, multi-dimensional trust evaluation. FLARE integrates: (i) a multi-dimensional reputation score capturing performance consistency, statistical anomaly indicators, and temporal behavior, (ii) a self-calibrating adaptive threshold mechanism that adjusts security strictness based on model convergence and recent attack intensity, (iii) reputation-weighted aggregation with soft exclusion to proportionally limit suspicious contributions rather than eliminating clients outright, and (iv) a Local Differential Privacy (LDP) mechanism enabling reputation scoring on privatized client updates. We further introduce a highly evasive Statistical Mimicry (SM) attack, a benchmark adversary that blends honest gradients with synthetic perturbations and persistent drift to remain undetected by traditional filters. Extensive experiments with 100 clients on MNIST, CIFAR-10, and SVHN demonstrate that FLARE maintains high model accuracy and converges faster than state-of-the-art Byzantine-robust methods under diverse attack types, including label flipping, gradient scaling, adaptive attacks, ALIE, and SM. FLARE improves robustness by up to 16% and preserves model convergence within 30% of the non-attacked baseline, while achieving strong malicious-client detection performance with minimal computational overhead. https://github.com/Anonymous0-0paper/FLARE



## **27. Sigil: Server-Enforced Watermarking in U-Shaped Split Federated Learning via Gradient Injection**

cs.CR

18 pages,8 figures

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14422v1) [paper-pdf](https://arxiv.org/pdf/2511.14422v1)

**Authors**: Zhengchunmin Dai, Jiaxiong Tang, Peng Sun, Honglong Chen, Liantao Wu

**Abstract**: In decentralized machine learning paradigms such as Split Federated Learning (SFL) and its variant U-shaped SFL, the server's capabilities are severely restricted. Although this enhances client-side privacy, it also leaves the server highly vulnerable to model theft by malicious clients. Ensuring intellectual property protection for such capability-limited servers presents a dual challenge: watermarking schemes that depend on client cooperation are unreliable in adversarial settings, whereas traditional server-side watermarking schemes are technically infeasible because the server lacks access to critical elements such as model parameters or labels.   To address this challenge, this paper proposes Sigil, a mandatory watermarking framework designed specifically for capability-limited servers. Sigil defines the watermark as a statistical constraint on the server-visible activation space and embeds the watermark into the client model via gradient injection, without requiring any knowledge of the data. Besides, we design an adaptive gradient clipping mechanism to ensure that our watermarking process remains both mandatory and stealthy, effectively countering existing gradient anomaly detection methods and a specifically designed adaptive subspace removal attack. Extensive experiments on multiple datasets and models demonstrate Sigil's fidelity, robustness, and stealthiness.



## **28. Cheating Stereo Matching in Full-scale: Physical Adversarial Attack against Binocular Depth Estimation in Autonomous Driving**

cs.CV

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2511.14386v2) [paper-pdf](https://arxiv.org/pdf/2511.14386v2)

**Authors**: Kangqiao Zhao, Shuo Huai, Xurui Song, Jun Luo

**Abstract**: Though deep neural models adopted to realize the perception of autonomous driving have proven vulnerable to adversarial examples, known attacks often leverage 2D patches and target mostly monocular perception. Therefore, the effectiveness of Physical Adversarial Examples (PAEs) on stereo-based binocular depth estimation remains largely unexplored. To this end, we propose the first texture-enabled physical adversarial attack against stereo matching models in the context of autonomous driving. Our method employs a 3D PAE with global camouflage texture rather than a local 2D patch-based one, ensuring both visual consistency and attack effectiveness across different viewpoints of stereo cameras. To cope with the disparity effect of these cameras, we also propose a new 3D stereo matching rendering module that allows the PAE to be aligned with real-world positions and headings in binocular vision. We further propose a novel merging attack that seamlessly blends the target into the environment through fine-grained PAE optimization. It has significantly enhanced stealth and lethality upon existing hiding attacks that fail to get seamlessly merged into the background. Extensive evaluations show that our PAEs can successfully fool the stereo models into producing erroneous depth information.



## **29. Steganographic Backdoor Attacks in NLP: Ultra-Low Poisoning and Defense Evasion**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14301v1) [paper-pdf](https://arxiv.org/pdf/2511.14301v1)

**Authors**: Eric Xue, Ruiyi Zhang, Zijun Zhang, Pengtao Xie

**Abstract**: Transformer models are foundational to natural language processing (NLP) applications, yet remain vulnerable to backdoor attacks introduced through poisoned data, which implant hidden behaviors during training. To strengthen the ability to prevent such compromises, recent research has focused on designing increasingly stealthy attacks to stress-test existing defenses, pairing backdoor behaviors with stylized artifact or token-level perturbation triggers. However, this trend diverts attention from the harder and more realistic case: making the model respond to semantic triggers such as specific names or entities, where a successful backdoor could manipulate outputs tied to real people or events in deployed systems. Motivated by this growing disconnect, we introduce SteganoBackdoor, bringing stealth techniques back into line with practical threat models. Leveraging innocuous properties from natural-language steganography, SteganoBackdoor applies a gradient-guided data optimization process to transform semantic trigger seeds into steganographic carriers that embed a high backdoor payload, remain fluent, and exhibit no representational resemblance to the trigger. Across diverse experimental settings, SteganoBackdoor achieves over 99% attack success at an order-of-magnitude lower data-poisoning rate than prior approaches while maintaining unparalleled evasion against a comprehensive suite of data-level defenses. By revealing this practical and covert attack, SteganoBackdoor highlights an urgent blind spot in current defenses and demands immediate attention to adversarial data defenses and real-world threat modeling.



## **30. A Fuzzy Logic-Based Cryptographic Framework For Real-Time Dynamic Key Generation For Enhanced Data Encryption**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14132v1) [paper-pdf](https://arxiv.org/pdf/2511.14132v1)

**Authors**: Kavya Bhand, Payal Khubchandani, Jyoti Khubchandani

**Abstract**: With the ever-growing demand for cybersecurity, static key encryption mechanisms are increasingly vulnerable to adversarial attacks due to their deterministic and non-adaptive nature. Brute-force attacks, key compromise, and unauthorized access have become highly common cyber threats. This research presents a novel fuzzy logic-based cryptographic framework that dynamically generates encryption keys in real-time by accessing system-level entropy and hardware-bound trust. The proposed system leverages a Fuzzy Inference System (FIS) to evaluate system parameters that include CPU utilization, process count, and timestamp variation. It assigns entropy level based on linguistically defined fuzzy rules which are fused with hardware-generated randomness and then securely sealed using a Trusted Platform Module (TPM). The sealed key is incorporated in an AES-GCM encryption scheme to ensure both confidentiality and integrity of the data. This system introduces a scalable solution for adaptive encryption in high-assurance computing, zero-trust environments, and cloud-based infrastructure.



## **31. Dynamic Black-box Backdoor Attacks on IoT Sensory Data**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.14074v1) [paper-pdf](https://arxiv.org/pdf/2511.14074v1)

**Authors**: Ajesh Koyatan Chathoth, Stephen Lee

**Abstract**: Sensor data-based recognition systems are widely used in various applications, such as gait-based authentication and human activity recognition (HAR). Modern wearable and smart devices feature various built-in Inertial Measurement Unit (IMU) sensors, and such sensor-based measurements can be fed to a machine learning-based model to train and classify human activities. While deep learning-based models have proven successful in classifying human activity and gestures, they pose various security risks. In our paper, we discuss a novel dynamic trigger-generation technique for performing black-box adversarial attacks on sensor data-based IoT systems. Our empirical analysis shows that the attack is successful on various datasets and classifier models with minimal perturbation on the input data. We also provide a detailed comparative analysis of performance and stealthiness to various other poisoning techniques found in backdoor attacks. We also discuss some adversarial defense mechanisms and their impact on the effectiveness of our trigger-generation technique.



## **32. Accuracy is Not Enough: Poisoning Interpretability in Federated Learning via Color Skew**

cs.CV

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.13535v2) [paper-pdf](https://arxiv.org/pdf/2511.13535v2)

**Authors**: Farhin Farhad Riya, Shahinul Hoque, Jinyuan Stella Sun, Olivera Kotevska

**Abstract**: As machine learning models are increasingly deployed in safety-critical domains, visual explanation techniques have become essential tools for supporting transparency. In this work, we reveal a new class of attacks that compromise model interpretability without affecting accuracy. Specifically, we show that small color perturbations applied by adversarial clients in a federated learning setting can shift a model's saliency maps away from semantically meaningful regions while keeping the prediction unchanged. The proposed saliency-aware attack framework, called Chromatic Perturbation Module, systematically crafts adversarial examples by altering the color contrast between foreground and background in a way that disrupts explanation fidelity. These perturbations accumulate across training rounds, poisoning the global model's internal feature attributions in a stealthy and persistent manner. Our findings challenge a common assumption in model auditing that correct predictions imply faithful explanations and demonstrate that interpretability itself can be an attack surface. We evaluate this vulnerability across multiple datasets and show that standard training pipelines are insufficient to detect or mitigate explanation degradation, especially in the federated learning setting, where subtle color perturbations are harder to discern. Our attack reduces peak activation overlap in Grad-CAM explanations by up to 35% while preserving classification accuracy above 96% on all evaluated datasets.



## **33. MPD-SGR: Robust Spiking Neural Networks with Membrane Potential Distribution-Driven Surrogate Gradient Regularization**

cs.LG

Accepted by AAAI 2026

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.12199v2) [paper-pdf](https://arxiv.org/pdf/2511.12199v2)

**Authors**: Runhao Jiang, Chengzhi Jiang, Rui Yan, Huajin Tang

**Abstract**: The surrogate gradient (SG) method has shown significant promise in enhancing the performance of deep spiking neural networks (SNNs), but it also introduces vulnerabilities to adversarial attacks. Although spike coding strategies and neural dynamics parameters have been extensively studied for their impact on robustness, the critical role of gradient magnitude, which reflects the model's sensitivity to input perturbations, remains underexplored. In SNNs, the gradient magnitude is primarily determined by the interaction between the membrane potential distribution (MPD) and the SG function. In this study, we investigate the relationship between the MPD and SG and their implications for improving the robustness of SNNs. Our theoretical analysis reveals that reducing the proportion of membrane potentials lying within the gradient-available range of the SG function effectively mitigates the sensitivity of SNNs to input perturbations. Building upon this insight, we propose a novel MPD-driven surrogate gradient regularization (MPD-SGR) method, which enhances robustness by explicitly regularizing the MPD based on its interaction with the SG function. Extensive experiments across multiple image classification benchmarks and diverse network architectures confirm that the MPD-SGR method significantly enhances the resilience of SNNs to adversarial perturbations and exhibits strong generalizability across diverse network configurations, SG functions, and spike encoding schemes.



## **34. Privacy on the Fly: A Predictive Adversarial Transformation Network for Mobile Sensor Data**

cs.CR

accepted by AAAI 2026 (oral)

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.07242v3) [paper-pdf](https://arxiv.org/pdf/2511.07242v3)

**Authors**: Tianle Song, Chenhao Lin, Yang Cao, Zhengyu Zhao, Jiahao Sun, Chong Zhang, Le Yang, Chao Shen

**Abstract**: Mobile motion sensors such as accelerometers and gyroscopes are now ubiquitously accessible by third-party apps via standard APIs. While enabling rich functionalities like activity recognition and step counting, this openness has also enabled unregulated inference of sensitive user traits, such as gender, age, and even identity, without user consent. Existing privacy-preserving techniques, such as GAN-based obfuscation or differential privacy, typically require access to the full input sequence, introducing latency that is incompatible with real-time scenarios. Worse, they tend to distort temporal and semantic patterns, degrading the utility of the data for benign tasks like activity recognition. To address these limitations, we propose the Predictive Adversarial Transformation Network (PATN), a real-time privacy-preserving framework that leverages historical signals to generate adversarial perturbations proactively. The perturbations are applied immediately upon data acquisition, enabling continuous protection without disrupting application functionality. Experiments on two datasets demonstrate that PATN substantially degrades the performance of privacy inference models, achieving Attack Success Rate (ASR) of 40.11% and 44.65% (reducing inference accuracy to near-random) and increasing the Equal Error Rate (EER) from 8.30% and 7.56% to 41.65% and 46.22%. On ASR, PATN outperforms baseline methods by 16.16% and 31.96%, respectively.



## **35. Injecting Falsehoods: Adversarial Man-in-the-Middle Attacks Undermining Factual Recall in LLMs**

cs.CR

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2511.05919v2) [paper-pdf](https://arxiv.org/pdf/2511.05919v2)

**Authors**: Alina Fastowski, Bardh Prenkaj, Yuxiao Li, Gjergji Kasneci

**Abstract**: LLMs are now an integral part of information retrieval. As such, their role as question answering chatbots raises significant concerns due to their shown vulnerability to adversarial man-in-the-middle (MitM) attacks. Here, we propose the first principled attack evaluation on LLM factual memory under prompt injection via Xmera, our novel, theory-grounded MitM framework. By perturbing the input given to "victim" LLMs in three closed-book and fact-based QA settings, we undermine the correctness of the responses and assess the uncertainty of their generation process. Surprisingly, trivial instruction-based attacks report the highest success rate (up to ~85.3%) while simultaneously having a high uncertainty for incorrectly answered questions. To provide a simple defense mechanism against Xmera, we train Random Forest classifiers on the response uncertainty levels to distinguish between attacked and unattacked queries (average AUC of up to ~96%). We believe that signaling users to be cautious about the answers they receive from black-box and potentially corrupt LLMs is a first checkpoint toward user cyberspace safety.



## **36. DRIP: Defending Prompt Injection via Token-wise Representation Editing and Residual Instruction Fusion**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2511.00447v2) [paper-pdf](https://arxiv.org/pdf/2511.00447v2)

**Authors**: Ruofan Liu, Yun Lin, Zhiyong Huang, Jin Song Dong

**Abstract**: Large language models (LLMs) are increasingly integrated into IT infrastructures, where they process user data according to predefined instructions. However, conventional LLMs remain vulnerable to prompt injection, where malicious users inject directive tokens into the data to subvert model behavior. Existing defenses train LLMs to semantically separate data and instruction tokens, but still struggle to (1) balance utility and security and (2) prevent instruction-like semantics in the data from overriding the intended instructions.   We propose DRIP, which (1) precisely removes instruction semantics from tokens in the data section while preserving their data semantics, and (2) robustly preserves the effect of the intended instruction even under strong adversarial content. To "de-instructionalize" data tokens, DRIP introduces a data curation and training paradigm with a lightweight representation-editing module that edits embeddings of instruction-like tokens in the data section, enhancing security without harming utility. To ensure non-overwritability of instructions, DRIP adds a minimal residual module that reduces the ability of adversarial data to overwrite the original instruction. We evaluate DRIP on LLaMA 8B and Mistral 7B against StruQ, SecAlign, ISE, and PFT on three prompt-injection benchmarks (SEP, AlpacaFarm, and InjecAgent). DRIP improves role-separation score by 12-49\%, reduces attack success rate by over 66\% under adaptive attacks, and matches the utility of the undefended model, establishing a new state of the art for prompt-injection robustness.



## **37. Observation-Free Attacks on Online Learning to Rank**

cs.LG

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2509.22855v3) [paper-pdf](https://arxiv.org/pdf/2509.22855v3)

**Authors**: Sameep Chattopadhyay, Nikhil Karamchandani, Sharayu Moharir

**Abstract**: Online learning to rank (OLTR) plays a critical role in information retrieval and machine learning systems, with a wide range of applications in search engines and content recommenders. However, despite their extensive adoption, the susceptibility of OLTR algorithms to coordinated adversarial attacks remains poorly understood. In this work, we present a novel framework for attacking some of the widely used OLTR algorithms. Our framework is designed to promote a set of target items so that they appear in the list of top-K recommendations for T - o(T) rounds, while simultaneously inducing linear regret in the learning algorithm. We propose two novel attack strategies: CascadeOFA for CascadeUCB1 and PBMOFA for PBM-UCB . We provide theoretical guarantees showing that both strategies require only O(log T) manipulations to succeed. Additionally, we supplement our theoretical analysis with empirical results on real-world data.



## **38. Decoding Deception: Understanding Automatic Speech Recognition Vulnerabilities in Evasion and Poisoning Attacks**

cs.SD

Remove due to conflict in authors

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2509.22060v2) [paper-pdf](https://arxiv.org/pdf/2509.22060v2)

**Authors**: Aravindhan G, Yuvaraj Govindarajulu, Parin Shah

**Abstract**: Recent studies have demonstrated the vulnerability of Automatic Speech Recognition systems to adversarial examples, which can deceive these systems into misinterpreting input speech commands. While previous research has primarily focused on white-box attacks with constrained optimizations, and transferability based black-box attacks against commercial Automatic Speech Recognition devices, this paper explores cost efficient white-box attack and non transferability black-box adversarial attacks on Automatic Speech Recognition systems, drawing insights from approaches such as Fast Gradient Sign Method and Zeroth-Order Optimization. Further, the novelty of the paper includes how poisoning attack can degrade the performances of state-of-the-art models leading to misinterpretation of audio signals. Through experimentation and analysis, we illustrate how hybrid models can generate subtle yet impactful adversarial examples with very little perturbation having Signal Noise Ratio of 35dB that can be generated within a minute. These vulnerabilities of state-of-the-art open source model have practical security implications, and emphasize the need for adversarial security.



## **39. SoK: Exposing the Generation and Detection Gaps in LLM-Generated Phishing Through Examination of Generation Methods, Content Characteristics, and Countermeasures**

cs.CR

18 pages, 5 tables, 4 figures

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2508.21457v2) [paper-pdf](https://arxiv.org/pdf/2508.21457v2)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Phishing campaigns involve adversaries masquerading as trusted vendors trying to trigger user behavior that enables them to exfiltrate private data. While URLs are an important part of phishing campaigns, communicative elements like text and images are central in triggering the required user behavior. Further, due to advances in phishing detection, attackers react by scaling campaigns to larger numbers and diversifying and personalizing content. In addition to established mechanisms, such as template-based generation, large language models (LLMs) can be used for phishing content generation, enabling attacks to scale in minutes, challenging existing phishing detection paradigms through personalized content, stealthy explicit phishing keywords, and dynamic adaptation to diverse attack scenarios. Countering these dynamically changing attack campaigns requires a comprehensive understanding of the complex LLM-related threat landscape. Existing studies are fragmented and focus on specific areas. In this work, we provide the first holistic examination of LLM-generated phishing content. First, to trace the exploitation pathways of LLMs for phishing content generation, we adopt a modular taxonomy documenting nine stages by which adversaries breach LLM safety guardrails. We then characterize how LLM-generated phishing manifests as threats, revealing that it evades detectors while emphasizing human cognitive manipulation. Third, by taxonomizing defense techniques aligned with generation methods, we expose a critical asymmetry that offensive mechanisms adapt dynamically to attack scenarios, whereas defensive strategies remain static and reactive. Finally, based on a thorough analysis of the existing literature, we highlight insights and gaps and suggest a roadmap for understanding and countering LLM-driven phishing at scale.



## **40. Constraint-Guided Prediction Refinement via Deterministic Diffusion Trajectories**

cs.AI

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2506.12911v2) [paper-pdf](https://arxiv.org/pdf/2506.12911v2)

**Authors**: Pantelis Dogoulis, Fabien Bernier, Félix Fourreau, Karim Tit, Maxime Cordy

**Abstract**: Many real-world machine learning tasks require outputs that satisfy hard constraints, such as physical conservation laws, structured dependencies in graphs, or column-level relationships in tabular data. Existing approaches rely either on domain-specific architectures and losses or on strong assumptions on the constraint space, restricting their applicability to linear or convex constraints. We propose a general-purpose framework for constraint-aware refinement that leverages denoising diffusion implicit models (DDIMs). Starting from a coarse prediction, our method iteratively refines it through a deterministic diffusion trajectory guided by a learned prior and augmented by constraint gradient corrections. The approach accommodates a wide class of non-convex and nonlinear equality constraints and can be applied post hoc to any base model. We demonstrate the method in two representative domains: constrained adversarial attack generation on tabular data with column-level dependencies and in AC power flow prediction under Kirchhoff's laws. Across both settings, our diffusion-guided refinement improves both constraint satisfaction and performance while remaining lightweight and model-agnostic.



## **41. TooBadRL: Trigger Optimization to Boost Effectiveness of Backdoor Attacks on Deep Reinforcement Learning**

cs.CR

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2506.09562v3) [paper-pdf](https://arxiv.org/pdf/2506.09562v3)

**Authors**: Mingxuan Zhang, Oubo Ma, Kang Wei, Songze Li, Shouling Ji

**Abstract**: Deep reinforcement learning (DRL) has achieved remarkable success in a wide range of sequential decision-making applications, including robotics, healthcare, smart grids, and finance. Recent studies reveal that adversaries can implant backdoors into DRL agents during the training phase. These backdoors can later be activated by specific triggers during deployment, compelling the agent to execute targeted actions and potentially leading to severe consequences, such as drone crashes or vehicle collisions. However, existing backdoor attacks utilize simplistic and heuristic trigger configurations, overlooking the critical impact of trigger design on attack effectiveness. To address this gap, we introduce TooBadRL, the first framework to systematically optimize DRL backdoor triggers across three critical aspects: injection timing, trigger dimension, and manipulation magnitude. Specifically, we first introduce a performance-aware adaptive freezing mechanism to determine the injection timing during training. Then, we formulate trigger selection as an influence attribution problem and apply Shapley value analysis to identify the most influential trigger dimension for injection. Furthermore, we propose an adversarial input synthesis method to optimize the manipulation magnitude under environmental constraints. Extensive evaluations on three DRL algorithms and nine benchmark tasks demonstrate that TooBadRL outperforms five baseline methods in terms of attack success rate while only slightly affecting normal task performance. We further evaluate potential defense strategies from detection and mitigation perspectives. We open-source our code to facilitate reproducibility and further research.



## **42. Revisiting Model Inversion Evaluation: From Misleading Standards to Reliable Privacy Assessment**

cs.LG

To support future work, we release our MLLM-based MI evaluation framework and benchmarking suite at https://github.com/hosytuyen/MI-Eval-MLLM

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2505.03519v4) [paper-pdf](https://arxiv.org/pdf/2505.03519v4)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information from private training data by exploiting access to machine learning models T. To evaluate such attacks, the standard evaluation framework relies on an evaluation model E, trained under the same task design as T. This framework has become the de facto standard for assessing progress in MI research, used across nearly all recent MI studies without question. In this paper, we present the first in-depth study of this evaluation framework. In particular, we identify a critical issue of this standard framework: Type-I adversarial examples. These are reconstructions that do not capture the visual features of private training data, yet are still deemed successful by T and ultimately transferable to E. Such false positives undermine the reliability of the standard MI evaluation framework. To address this issue, we introduce a new MI evaluation framework that replaces the evaluation model E with advanced Multimodal Large Language Models (MLLMs). By leveraging their general-purpose visual understanding, our MLLM-based framework does not depend on training of shared task design as in T, thus reducing Type-I transferability and providing more faithful assessments of reconstruction success. Using our MLLM-based evaluation framework, we reevaluate 27 diverse MI attack setups and empirically reveal consistently high false positive rates under the standard evaluation framework. Importantly, we demonstrate that many state-of-the-art (SOTA) MI methods report inflated attack accuracy, indicating that actual privacy leakage is significantly lower than previously believed. By uncovering this critical issue and proposing a robust solution, our work enables a reassessment of progress in MI research and sets a new standard for reliable and robust evaluation. Code can be found in https://github.com/hosytuyen/MI-Eval-MLLM



## **43. Quantifying Privacy Leakage in Split Inference via Fisher-Approximated Shannon Information Analysis**

cs.CR

13pages, 12 figures

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2504.10016v2) [paper-pdf](https://arxiv.org/pdf/2504.10016v2)

**Authors**: Ruijun Deng, Zhihui Lu, Qiang Duan, Shijing Hu

**Abstract**: Split inference (SI) partitions deep neural networks into distributed sub-models, enabling collaborative learning without directly sharing raw data. However, SI remains vulnerable to Data Reconstruction Attacks (DRAs), where adversaries exploit exposed smashed data to recover private inputs. Despite substantial progress in attack-defense methodologies, the fundamental quantification of privacy risks is still underdeveloped. This paper establishes an information-theoretic framework for privacy leakage in SI, defining leakage as the adversary's certainty and deriving both average-case and worst-case error lower bounds. We further introduce Fisher-approximated Shannon information (FSInfo), a new privacy metric based on Fisher Information (FI) that enables operational and tractable computation of privacy leakage. Building on this metric, we develop FSInfoGuard, a defense mechanism that achieves a strong privacy-utility tradeoff. Our empirical study shows that FSInfo is an effective privacy metric across datasets, models, and defense strengths, providing accurate privacy estimates that support the design of defense methods outperforming existing approaches in both privacy protection and utility preservation. The code is available at https://github.com/SASA-cloud/FSInfo.



## **44. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

cs.CL

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2504.02132v3) [paper-pdf](https://arxiv.org/pdf/2504.02132v3)

**Authors**: Ezzeldin Shereen, Dan Ristea, Shae McFadden, Burak Hasircioglu, Vasilios Mavroudis, Chris Hicks

**Abstract**: Retrieval-augmented generation (RAG) is instrumental for inhibiting hallucinations in large language models (LLMs) through the use of a factual knowledge base (KB). Although PDF documents are prominent sources of knowledge, text-based RAG pipelines are ineffective at capturing their rich multi-modal information. In contrast, visual document RAG (VD-RAG) uses screenshots of document pages as the KB, which has been shown to achieve state-of-the-art results. However, by introducing the image modality, VD-RAG introduces new attack vectors for adversaries to disrupt the system by injecting malicious documents into the KB. In this paper, we demonstrate the vulnerability of VD-RAG to poisoning attacks targeting both retrieval and generation. We define two attack objectives and demonstrate that both can be realized by injecting only a single adversarial image into the KB. Firstly, we introduce a targeted attack against one or a group of queries with the goal of spreading targeted disinformation. Secondly, we present a universal attack that, for any potential user query, influences the response to cause a denial-of-service in the VD-RAG system. We investigate the two attack objectives under both white-box and black-box assumptions, employing a multi-objective gradient-based optimization approach as well as prompting state-of-the-art generative models. Using two visual document datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (vision language models), we show VD-RAG is vulnerable to poisoning attacks in both the targeted and universal settings, yet demonstrating robustness to black-box attacks in the universal setting.



## **45. A Closer Look at Adversarial Suffix Learning for Jailbreaking LLMs: Augmented Adversarial Trigger Learning**

cs.LG

the Association for Computational Linguistics: NAACL 2025

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2503.12339v4) [paper-pdf](https://arxiv.org/pdf/2503.12339v4)

**Authors**: Zhe Wang, Yanjun Qi

**Abstract**: Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs. We released our code https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning



## **46. Adversarial Agents: Black-Box Evasion Attacks with Reinforcement Learning**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2503.01734v2) [paper-pdf](https://arxiv.org/pdf/2503.01734v2)

**Authors**: Kyle Domico, Jean-Charles Noirot Ferrand, Ryan Sheatsley, Eric Pauley, Josiah Hanna, Patrick McDaniel

**Abstract**: Attacks on machine learning models have been extensively studied through stateless optimization. In this paper, we demonstrate how a reinforcement learning (RL) agent can learn a new class of attack algorithms that generate adversarial samples. Unlike traditional adversarial machine learning (AML) methods that craft adversarial samples independently, our RL-based approach retains and exploits past attack experience to improve the effectiveness and efficiency of future attacks. We formulate adversarial sample generation as a Markov Decision Process and evaluate RL's ability to (a) learn effective and efficient attack strategies and (b) compete with state-of-the-art AML. On two image classification benchmarks, our agent increases attack success rate by up to 13.2% and decreases the average number of victim model queries per attack by up to 16.9% from the start to the end of training. In a head-to-head comparison with state-of-the-art image attacks, our approach enables an adversary to generate adversarial samples with 17% more success on unseen inputs post-training. From a security perspective, this work demonstrates a powerful new attack vector that uses RL to train agents that attack ML models efficiently and at scale.



## **47. 1-Lipschitz Network Initialization for Certifiably Robust Classification Applications: A Decay Problem**

cs.LG

15 pages, 11 figures; added additional experimental results and formatted to Elsevier format

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2503.00240v2) [paper-pdf](https://arxiv.org/pdf/2503.00240v2)

**Authors**: Marius F. R. Juston, Ramavarapu S. Sreenivas, William R. Norris, Dustin Nottage, Ahmet Soylemezoglu

**Abstract**: This paper discusses the weight parametrization of two standard 1-Lipschitz network architectures, the Almost-Orthogonal-Layers (AOL) and the SDP-based Lipschitz Layers (SLL). It examines their impact on initialization for deep 1-Lipschitz feedforward networks, and discusses underlying issues surrounding this initialization. These networks are mainly used in certifiably robust classification applications to combat adversarial attacks by limiting the impact of perturbations on the classification output. Exact and upper bounds for the parameterized weight variance were calculated assuming a standard Normal distribution initialization; additionally, an upper bound was computed assuming a Generalized Normal Distribution, generalizing the proof for Uniform, Laplace, and Normal distribution weight initializations. It is demonstrated that the weight variance holds no bearing on the output variance distribution and that only the dimension of the weight matrices matters. Additionally, this paper demonstrates that the weight initialization always causes deep 1-Lipschitz networks to decay to zero.



## **48. Eguard: Defending LLM Embeddings Against Inversion Attacks via Text Mutual Information Optimization**

cs.CR

**SubmitDate**: 2025-11-19    [abs](http://arxiv.org/abs/2411.05034v2) [paper-pdf](https://arxiv.org/pdf/2411.05034v2)

**Authors**: Tiantian Liu, Hongwei Yao, Feng Lin, Tong Wu, Zhan Qin, Kui Ren

**Abstract**: Embeddings have become a cornerstone in the functionality of large language models (LLMs) due to their ability to transform text data into rich, dense numerical representations that capture semantic and syntactic properties. These embedding vector databases serve as the long-term memory of LLMs, enabling efficient handling of a wide range of natural language processing tasks. However, the surge in popularity of embedding vector databases in LLMs has been accompanied by significant concerns about privacy leakage. Embedding vector databases are particularly vulnerable to embedding inversion attacks, where adversaries can exploit the embeddings to reverse-engineer and extract sensitive information from the original text data. Existing defense mechanisms have shown limitations, often struggling to balance security with the performance of downstream tasks. To address these challenges, we introduce Eguard, a novel defense mechanism designed to mitigate embedding inversion attacks. Eguard employs a transformer-based projection network and text mutual information optimization to safeguard embeddings while preserving the utility of LLMs. Our approach significantly reduces privacy risks, protecting over 95% of tokens from inversion while maintaining high performance across downstream tasks consistent with original embeddings.



## **49. Sparse-PGD: A Unified Framework for Sparse Adversarial Perturbations Generation**

cs.LG

Accepted by TPAMI

**SubmitDate**: 2025-11-20    [abs](http://arxiv.org/abs/2405.05075v4) [paper-pdf](https://arxiv.org/pdf/2405.05075v4)

**Authors**: Xuyang Zhong, Chen Liu

**Abstract**: This work studies sparse adversarial perturbations, including both unstructured and structured ones. We propose a framework based on a white-box PGD-like attack method named Sparse-PGD to effectively and efficiently generate such perturbations. Furthermore, we combine Sparse-PGD with a black-box attack to comprehensively and more reliably evaluate the models' robustness against unstructured and structured sparse adversarial perturbations. Moreover, the efficiency of Sparse-PGD enables us to conduct adversarial training to build robust models against various sparse perturbations. Extensive experiments demonstrate that our proposed attack algorithm exhibits strong performance in different scenarios. More importantly, compared with other robust models, our adversarially trained model demonstrates state-of-the-art robustness against various sparse attacks. Codes are available at https://github.com/CityU-MLO/sPGD.



## **50. High Dimensional Distributed Gradient Descent with Arbitrary Number of Byzantine Attackers**

cs.LG

25 pages, 4 figures

**SubmitDate**: 2025-11-18    [abs](http://arxiv.org/abs/2307.13352v3) [paper-pdf](https://arxiv.org/pdf/2307.13352v3)

**Authors**: Wenyu Liu, Tianqiang Huang, Pengfei Zhang, Zong Ke, Minghui Min, Puning Zhao

**Abstract**: Adversarial attacks pose a major challenge to distributed learning systems, prompting the development of numerous robust learning methods. However, most existing approaches suffer from the curse of dimensionality, i.e. the error increases with the number of model parameters. In this paper, we make a progress towards high dimensional problems, under arbitrary number of Byzantine attackers. The cornerstone of our design is a direct high dimensional semi-verified mean estimation method. The idea is to identify a subspace with large variance. The components of the mean value perpendicular to this subspace are estimated using corrupted gradient vectors uploaded from worker machines, while the components within this subspace are estimated using auxiliary dataset. As a result, a combination of large corrupted dataset and small clean dataset yields significantly better performance than using them separately. We then apply this method as the aggregator for distributed learning problems. The theoretical analysis shows that compared with existing solutions, our method gets rid of $\sqrt{d}$ dependence on the dimensionality, and achieves minimax optimal statistical rates. Numerical results validate our theory as well as the effectiveness of the proposed method.



