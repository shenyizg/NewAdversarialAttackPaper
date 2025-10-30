# Latest Adversarial Attack Papers
**update at 2025-10-30 15:03:12**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Distribution System Reconfiguration to Mitigate Load Altering Attacks via Stackelberg Games**

eess.SY

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2407.07065v6) [paper-pdf](http://arxiv.org/pdf/2407.07065v6)

**Authors**: Sajjad Maleki, E. Veronica Belmaga, Charalambos Konstantinou, Subhash Lakshminarayana

**Abstract**: The widespread integration of IoT-controllable devices (e.g., smart EV charging stations and heat pumps) into modern power systems enhances capabilities but introduces critical cybersecurity risks. Specifically, these devices are susceptible to load-altering attacks (LAAs) that can compromise power system safety. This paper quantifies the impact of LAAs on nodal voltage constraint violations in distribution networks (DNs). We first present closed-form expressions to analytically characterize LAA effects and quantify the minimum number of compromised devices for a successful LAA. Based on these insights, we propose a reactive defense mechanism that mitigates LAAs through DN reconfiguration. To address strategic adversaries, we then formulate defense strategies using a non-cooperative sequential game, which models the knowledgeable and strategic attacker, accounting for the worst-case scenario and enabling the reactive defender to devise an efficient and robust defense. Further, our formulation also accounts for uncertainties in attack localization. A novel Bayesian optimization approach is introduced to compute the Stackelberg equilibrium, significantly reducing computational burden efficiently. The game-theoretic strategy effectively mitigates the attack's impact while ensuring minimal system reconfiguration.



## **2. Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation**

cs.SD

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2507.17937v3) [paper-pdf](http://arxiv.org/pdf/2507.17937v3)

**Authors**: Jaechul Roh, Zachary Novack, Yuefeng Peng, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Amir Houmansadr

**Abstract**: Generative AI systems for music and video commonly use text-based filters to prevent the regurgitation of copyrighted material. We expose a fundamental flaw in this approach by introducing Adversarial PhoneTic Prompting (APT), a novel attack that bypasses these safeguards by exploiting phonetic memorization. The APT attack replaces iconic lyrics with homophonic but semantically unrelated alternatives (e.g., "mom's spaghetti" becomes "Bob's confetti"), preserving acoustic structure while altering meaning; we identify high-fidelity phonetic matches using CMU pronouncing dictionary. We demonstrate that leading Lyrics-to-Song (L2S) models like SUNO and YuE regenerate songs with striking melodic and rhythmic similarity to their copyrighted originals when prompted with these altered lyrics. More surprisingly, this vulnerability extends across modalities. When prompted with phonetically modified lyrics from a song, a Text-to-Video (T2V) model like Veo 3 reconstructs visual scenes from the original music video-including specific settings and character archetypes-despite the absence of any visual cues in the prompt. Our findings reveal that models memorize deep, structural patterns tied to acoustics, not just verbatim text. This phonetic-to-visual leakage represents a critical vulnerability in transcript-conditioned generative models, rendering simple copyright filters ineffective and raising urgent concerns about the secure deployment of multimodal AI systems. Demo examples are available at our project page (https://jrohsc.github.io/music_attack/).



## **3. NetEcho: From Real-World Streaming Side-Channels to Full LLM Conversation Recovery**

cs.CR

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.25472v1) [paper-pdf](http://arxiv.org/pdf/2510.25472v1)

**Authors**: Zheng Zhang, Guanlong Wu, Sen Deng, Shuai Wang, Yinqian Zhang

**Abstract**: In the rapidly expanding landscape of Large Language Model (LLM) applications, real-time output streaming has become the dominant interaction paradigm. While this enhances user experience, recent research reveals that it exposes a non-trivial attack surface through network side-channels. Adversaries can exploit patterns in encrypted traffic to infer sensitive information and reconstruct private conversations. In response, LLM providers and third-party services are deploying defenses such as traffic padding and obfuscation to mitigate these vulnerabilities.   This paper starts by presenting a systematic analysis of contemporary side-channel defenses in mainstream LLM applications, with a focus on services from vendors like OpenAI and DeepSeek. We identify and examine seven representative deployment scenarios, each incorporating active/passive mitigation techniques. Despite these enhanced security measures, our investigation uncovers significant residual information that remains vulnerable to leakage within the network traffic.   Building on this discovery, we introduce NetEcho, a novel, LLM-based framework that comprehensively unleashes the network side-channel risks of today's LLM applications. NetEcho is designed to recover entire conversations -- including both user prompts and LLM responses -- directly from encrypted network traffic. It features a deliberate design that ensures high-fidelity text recovery, transferability across different deployment scenarios, and moderate operational cost. In our evaluations on medical and legal applications built upon leading models like DeepSeek-v3 and GPT-4o, NetEcho can recover avg $\sim$70\% information of each conversation, demonstrating a critical limitation in current defense mechanisms. We conclude by discussing the implications of our findings and proposing future directions for augmenting network traffic security.



## **4. Timestamp Manipulation: Timestamp-based Nakamoto-style Blockchains are Vulnerable**

cs.CR

25 pages, 6 figures

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2505.05328v5) [paper-pdf](http://arxiv.org/pdf/2505.05328v5)

**Authors**: Junjie Hu, Sisi Duan

**Abstract**: Nakamoto consensus are the most widely adopted decentralized consensus mechanism in cryptocurrency systems. Since it was proposed in 2008, many studies have focused on analyzing its security. Most of them focus on maximizing the profit of the adversary. Examples include the selfish mining attack [FC '14] and the recent riskless uncle maker (RUM) attack [CCS '23]. In this work, we introduce the Staircase-Unrestricted Uncle Maker (SUUM), the first block withholding attack targeting the timestamp-based Nakamoto-style blockchain. Through block withholding, timestamp manipulation, and difficulty risk control, SUUM adversaries are capable of launching persistent attacks with zero cost and minimal difficulty risk characteristics, indefinitely exploiting rewards from honest participants. This creates a self-reinforcing cycle that threatens the security of blockchains. We conduct a comprehensive and systematic evaluation of SUUM, including the attack conditions, its impact on blockchains, and the difficulty risks. Finally, we further discuss four feasible mitigation measures against SUUM.



## **5. TextCrafter: Optimization-Calibrated Noise for Defending Against Text Embedding Inversion**

cs.CR

More sufficient and convincing experiments are needed

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2509.17302v2) [paper-pdf](http://arxiv.org/pdf/2509.17302v2)

**Authors**: Duoxun Tang, Xinhang Jiang, Jiajun Niu

**Abstract**: Text embedding inversion attacks reconstruct original sentences from latent representations, posing severe privacy threats in collaborative inference and edge computing. We propose TextCrafter, an optimization-based adversarial perturbation mechanism that combines RL learned, geometry aware noise injection orthogonal to user embeddings with cluster priors and PII signal guidance to suppress inversion while preserving task utility. Unlike prior defenses either non learnable or agnostic to perturbation direction, TextCrafter provides a directional protective policy that balances privacy and utility. Under strong privacy setting, TextCrafter maintains 70 percentage classification accuracy on four datasets and consistently outperforms Gaussian/LDP baselines across lower privacy budgets, demonstrating a superior privacy utility trade off.



## **6. A Unified Bilevel Model for Adversarial Learning and A Case Study**

cs.LG

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.25121v1) [paper-pdf](http://arxiv.org/pdf/2510.25121v1)

**Authors**: Yutong Zheng, Qingna Li

**Abstract**: Adversarial learning has been attracting more and more attention thanks to the fast development of machine learning and artificial intelligence. However, due to the complicated structure of most machine learning models, the mechanism of adversarial attacks is not well interpreted. How to measure the effect of attack is still not quite clear. In this paper, we propose a unified bilevel model for adversarial learning. We further investigate the adversarial attack in clustering models and interpret it from data perturbation point of view. We reveal that when the data perturbation is relatively small, the clustering model is robust, whereas if it is relatively large, the clustering result changes, which leads to an attack. To measure the effect of attacks for clustering models, we analyse the well-definedness of the so-called $\delta$-measure, which can be used in the proposed bilevel model for adversarial learning of clustering models.



## **7. An Adversarial-Driven Experimental Study on Deep Learning for RF Fingerprinting**

cs.CR

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2507.14109v2) [paper-pdf](http://arxiv.org/pdf/2507.14109v2)

**Authors**: Xinyu Cao, Bimal Adhikari, Shangqing Zhao, Jingxian Wu, Yanjun Pan

**Abstract**: Radio frequency (RF) fingerprinting, which extracts unique hardware imperfections of radio devices, has emerged as a promising physical-layer device identification mechanism in zero trust architectures and beyond 5G networks. In particular, deep learning (DL) methods have demonstrated state-of-the-art performance in this domain. However, existing approaches have primarily focused on enhancing system robustness against temporal and spatial variations in wireless environments, while the security vulnerabilities of these DL-based approaches have often been overlooked. In this work, we systematically investigate the security risks of DL-based RF fingerprinting systems through an adversarial-driven experimental analysis. We observe a consistent misclassification behavior for DL models under domain shifts, where a device is frequently misclassified as another specific one. Our analysis based on extensive real-world experiments demonstrates that this behavior can be exploited as an effective backdoor to enable external attackers to intrude into the system. Furthermore, we show that training DL models on raw received signals causes the models to entangle RF fingerprints with environmental and signal-pattern features, creating additional attack vectors that cannot be mitigated solely through post-processing security methods such as confidence thresholds.



## **8. Jailbreak Transferability Emerges from Shared Representations**

cs.LG

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2506.12913v2) [paper-pdf](http://arxiv.org/pdf/2506.12913v2)

**Authors**: Rico Angell, Jannik Brinkmann, He He

**Abstract**: Jailbreak transferability is the surprising phenomenon when an adversarial attack compromising one model also elicits harmful responses from other models. Despite widespread demonstrations, there is little consensus on why transfer is possible: is it a quirk of safety training, an artifact of model families, or a more fundamental property of representation learning? We present evidence that transferability emerges from shared representations rather than incidental flaws. Across 20 open-weight models and 33 jailbreak attacks, we find two factors that systematically shape transfer: (1) representational similarity under benign prompts, and (2) the strength of the jailbreak on the source model. To move beyond correlation, we show that deliberately increasing similarity through benign only distillation causally increases transfer. Our qualitative analyses reveal systematic transferability patterns across different types of jailbreaks. For example, persona-style jailbreaks transfer far more often than cipher-based prompts, consistent with the idea that natural-language attacks exploit models' shared representation space, whereas cipher-based attacks rely on idiosyncratic quirks that do not generalize. Together, these results reframe jailbreak transfer as a consequence of representation alignment rather than a fragile byproduct of safety training.



## **9. Cybersecurity AI Benchmark (CAIBench): A Meta-Benchmark for Evaluating Cybersecurity AI Agents**

cs.CR

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.24317v1) [paper-pdf](http://arxiv.org/pdf/2510.24317v1)

**Authors**: María Sanz-Gómez, Víctor Mayoral-Vilches, Francesco Balassone, Luis Javier Navarrete-Lozano, Cristóbal R. J. Veas Chavez, Maite del Mundo de Torres

**Abstract**: Cybersecurity spans multiple interconnected domains, complicating the development of meaningful, labor-relevant benchmarks. Existing benchmarks assess isolated skills rather than integrated performance. We find that pre-trained knowledge of cybersecurity in LLMs does not imply attack and defense abilities, revealing a gap between knowledge and capability. To address this limitation, we present the Cybersecurity AI Benchmark (CAIBench), a modular meta-benchmark framework that allows evaluating LLM models and agents across offensive and defensive cybersecurity domains, taking a step towards meaningfully measuring their labor-relevance. CAIBench integrates five evaluation categories, covering over 10,000 instances: Jeopardy-style CTFs, Attack and Defense CTFs, Cyber Range exercises, knowledge benchmarks, and privacy assessments. Key novel contributions include systematic simultaneous offensive-defensive evaluation, robotics-focused cybersecurity challenges (RCTF2), and privacy-preserving performance assessment (CyberPII-Bench). Evaluation of state-of-the-art AI models reveals saturation on security knowledge metrics (~70\% success) but substantial degradation in multi-step adversarial (A\&D) scenarios (20-40\% success), or worse in robotic targets (22\% success). The combination of framework scaffolding and LLM model choice significantly impacts performance; we find that proper matches improve up to 2.6$\times$ variance in Attack and Defense CTFs. These results demonstrate a pronounced gap between conceptual knowledge and adaptive capability, emphasizing the need for a meta-benchmark.



## **10. MixAT: Combining Continuous and Discrete Adversarial Training for LLMs**

cs.LG

Published at 39th Conference on Neural Information Processing Systems  (NeurIPS 2025)

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2505.16947v2) [paper-pdf](http://arxiv.org/pdf/2505.16947v2)

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: Despite recent efforts in Large Language Model (LLM) safety and alignment, current adversarial attacks on frontier LLMs can still consistently force harmful generations. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. At the same time, despite their effectiveness and generalization capabilities, training with continuous perturbations does not always capture the full spectrum of vulnerabilities exploited by discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.



## **11. Is It Certainly a Deepfake? Reliability Analysis in Detection & Generation Ecosystem**

cs.AI

Accepted for publication at the ICCV 2025 workshop - STREAM

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2509.17550v3) [paper-pdf](http://arxiv.org/pdf/2509.17550v3)

**Authors**: Neslihan Kose, Anthony Rhodes, Umur Aybars Ciftci, Ilke Demir

**Abstract**: As generative models are advancing in quality and quantity for creating synthetic content, deepfakes begin to cause online mistrust. Deepfake detectors are proposed to counter this effect, however, misuse of detectors claiming fake content as real or vice versa further fuels this misinformation problem. We present the first comprehensive uncertainty analysis of deepfake detectors, systematically investigating how generative artifacts influence prediction confidence. As reflected in detectors' responses, deepfake generators also contribute to this uncertainty as their generative residues vary, so we cross the uncertainty analysis of deepfake detectors and generators. Based on our observations, the uncertainty manifold holds enough consistent information to leverage uncertainty for deepfake source detection. Our approach leverages Bayesian Neural Networks and Monte Carlo dropout to quantify both aleatoric and epistemic uncertainties across diverse detector architectures. We evaluate uncertainty on two datasets with nine generators, with four blind and two biological detectors, compare different uncertainty methods, explore region- and pixel-based uncertainty, and conduct ablation studies. We conduct and analyze binary real/fake, multi-class real/fake, source detection, and leave-one-out experiments between the generator/detector combinations to share their generalization capability, model calibration, uncertainty, and robustness against adversarial attacks. We further introduce uncertainty maps that localize prediction confidence at the pixel level, revealing distinct patterns correlated with generator-specific artifacts. Our analysis provides critical insights for deploying reliable deepfake detection systems and establishes uncertainty quantification as a fundamental requirement for trustworthy synthetic media detection.



## **12. Vanish into Thin Air: Cross-prompt Universal Adversarial Attacks for SAM2**

cs.CV

Accepted by NeurIPS 2025

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.24195v1) [paper-pdf](http://arxiv.org/pdf/2510.24195v1)

**Authors**: Ziqi Zhou, Yifan Hu, Yufei Song, Zijing Li, Shengshan Hu, Leo Yu Zhang, Dezhong Yao, Long Zheng, Hai Jin

**Abstract**: Recent studies reveal the vulnerability of the image segmentation foundation model SAM to adversarial examples. Its successor, SAM2, has attracted significant attention due to its strong generalization capability in video segmentation. However, its robustness remains unexplored, and it is unclear whether existing attacks on SAM can be directly transferred to SAM2. In this paper, we first analyze the performance gap of existing attacks between SAM and SAM2 and highlight two key challenges arising from their architectural differences: directional guidance from the prompt and semantic entanglement across consecutive frames. To address these issues, we propose UAP-SAM2, the first cross-prompt universal adversarial attack against SAM2 driven by dual semantic deviation. For cross-prompt transferability, we begin by designing a target-scanning strategy that divides each frame into k regions, each randomly assigned a prompt, to reduce prompt dependency during optimization. For effectiveness, we design a dual semantic deviation framework that optimizes a UAP by distorting the semantics within the current frame and disrupting the semantic consistency across consecutive frames. Extensive experiments on six datasets across two segmentation tasks demonstrate the effectiveness of the proposed method for SAM2. The comparative results show that UAP-SAM2 significantly outperforms state-of-the-art (SOTA) attacks by a large margin.



## **13. Untargeted Jailbreak Attack**

cs.CR

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.02999v2) [paper-pdf](http://arxiv.org/pdf/2510.02999v2)

**Authors**: Xinzhe Huang, Wenjing Hu, Tianhang Zheng, Kedong Xiu, Xiaojun Jia, Di Wang, Zhan Qin, Kui Ren

**Abstract**: Existing gradient-based jailbreak attacks on Large Language Models (LLMs), such as Greedy Coordinate Gradient (GCG) and COLD-Attack, typically optimize adversarial suffixes to align the LLM output with a predefined target response. However, by restricting the optimization objective as inducing a predefined target, these methods inherently constrain the adversarial search space, which limit their overall attack efficacy. Furthermore, existing methods typically require a large number of optimization iterations to fulfill the large gap between the fixed target and the original model response, resulting in low attack efficiency.   To overcome the limitations of targeted jailbreak attacks, we propose the first gradient-based untargeted jailbreak attack (UJA), aiming to elicit an unsafe response without enforcing any predefined patterns. Specifically, we formulate an untargeted attack objective to maximize the unsafety probability of the LLM response, which can be quantified using a judge model. Since the objective is non-differentiable, we further decompose it into two differentiable sub-objectives for optimizing an optimal harmful response and the corresponding adversarial prompt, with a theoretical analysis to validate the decomposition. In contrast to targeted jailbreak attacks, UJA's unrestricted objective significantly expands the search space, enabling a more flexible and efficient exploration of LLM vulnerabilities.Extensive evaluations demonstrate that UJA can achieve over 80% attack success rates against recent safety-aligned LLMs with only 100 optimization iterations, outperforming the state-of-the-art gradient-based attacks such as I-GCG and COLD-Attack by over 20%.



## **14. Learning to Attack: Uncovering Privacy Risks in Sequential Data Releases**

cs.CR

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.24807v1) [paper-pdf](http://arxiv.org/pdf/2510.24807v1)

**Authors**: Ziyao Cui, Minxing Zhang, Jian Pei

**Abstract**: Privacy concerns have become increasingly critical in modern AI and data science applications, where sensitive information is collected, analyzed, and shared across diverse domains such as healthcare, finance, and mobility. While prior research has focused on protecting privacy in a single data release, many real-world systems operate under sequential or continuous data publishing, where the same or related data are released over time. Such sequential disclosures introduce new vulnerabilities, as temporal correlations across releases may enable adversaries to infer sensitive information that remains hidden in any individual release. In this paper, we investigate whether an attacker can compromise privacy in sequential data releases by exploiting dependencies between consecutive publications, even when each individual release satisfies standard privacy guarantees. To this end, we propose a novel attack model that captures these sequential dependencies by integrating a Hidden Markov Model with a reinforcement learning-based bi-directional inference mechanism. This enables the attacker to leverage both earlier and later observations in the sequence to infer private information. We instantiate our framework in the context of trajectory data, demonstrating how an adversary can recover sensitive locations from sequential mobility datasets. Extensive experiments on Geolife, Porto Taxi, and SynMob datasets show that our model consistently outperforms baseline approaches that treat each release independently. The results reveal a fundamental privacy risk inherent to sequential data publishing, where individually protected releases can collectively leak sensitive information when analyzed temporally. These findings underscore the need for new privacy-preserving frameworks that explicitly model temporal dependencies, such as time-aware differential privacy or sequential data obfuscation strategies.



## **15. Enhancing CLIP Robustness via Cross-Modality Alignment**

cs.CV

NeurIPS 2025 Spotlight

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.24038v1) [paper-pdf](http://arxiv.org/pdf/2510.24038v1)

**Authors**: Xingyu Zhu, Beier Zhu, Shuo Wang, Kesen Zhao, Hanwang Zhang

**Abstract**: Vision-language models (VLMs) such as CLIP demonstrate strong generalization in zero-shot classification but remain highly vulnerable to adversarial perturbations. Existing methods primarily focus on adversarial fine-tuning or prompt optimization; they often overlook the gaps in CLIP's encoded features, which is shown as the text and image features lie far apart from each other. This misalignment is significantly amplified under adversarial perturbations, leading to severe degradation in classification performance. To address this problem, we propose Cross-modality Alignment, dubbed COLA, an optimal transport-based framework that explicitly addresses adversarial misalignment by restoring both global image-text alignment and local structural consistency in the feature space. (1) COLA first projects adversarial image embeddings onto a subspace spanned by class text features, effectively filtering out non-semantic distortions while preserving discriminative information. (2) It then models images and texts as discrete distributions over multiple augmented views and refines their alignment via OT, with the subspace projection seamlessly integrated into the cost computation. This design ensures stable cross-modal alignment even under adversarial conditions. COLA is training-free and compatible with existing fine-tuned models. Extensive evaluations across 14 zero-shot classification benchmarks demonstrate the effectiveness of COLA, especially with an average improvement of 6.7% on ImageNet and its variants under PGD adversarial attacks, while maintaining high accuracy on clean samples.



## **16. A Volumetric Privacy Measure for Dynamical Systems With Bounded Disturbance**

eess.SY

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2501.02893v5) [paper-pdf](http://arxiv.org/pdf/2501.02893v5)

**Authors**: Chuanghong Weng, Ehsan Nekouei

**Abstract**: In this paper, we first present a volumetric privacy measure for dynamical systems with bounded disturbances, wherein the states of the system contain private information and an adversary with access to sensor measurements attempts to infer the set of potential values of the private information. Under the proposed privacy measure, the volume of the uncertainty set of the adversary given the sensor measurements is considered as the privacy level of the system. We next characteristic the time evolution of the proposed privacy measure and study its properties for a particular system with both public and private states, where a set containing the public state is shared as the observation. Approximate set-membership estimation techniques are developed to compute the private-state uncertainty set, and the properties of the privacy measure are analyzed, demonstrating that the uncertainty reduction of the adversary is bounded by the information gain from the observation set. Furthermore, an optimization-based privacy filter design problem is formulated, employing randomization and linear programming to enhance the privacy level. The effectiveness of the proposed approach is validated through a production-inventory case study. Results show that the optimal privacy filter significantly improves robustness against inference attacks and outperforms two baseline mechanisms based on additive noise and quantization.



## **17. Modeling Object Attention in Mobile AR for Intrinsic Cognitive Security**

cs.HC

Conference Paper, 5 pages. Published at the 2025 ACM the  International Symposium on Theory, Algorithmic Foundations, and Protocol  Design for Mobile Networks and Mobile Computing (MobiHoc)

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.24004v1) [paper-pdf](http://arxiv.org/pdf/2510.24004v1)

**Authors**: Shane Dirksen, Radha Kumaran, You-Jin Kim, Yilin Wang, Tobias Höllerer

**Abstract**: We study attention in mobile Augmented Reality (AR) using object recall as a proxy outcome. We observe that the ability to recall an object (physical or virtual) that was encountered in a mobile AR experience depends on many possible impact factors and attributes, with some objects being readily recalled while others are not, and some people recalling objects overall much better or worse than others. This opens up a potential cognitive attack in which adversaries might create conditions that make an AR user not recall certain potentially mission-critical objects. We explore whether a calibrated predictor of object recall can help shield against such cognitive attacks. We pool data from four mobile AR studies (with a total of 1,152 object recall probes) and fit a Partial Least Squares Structural Equation Model (PLS-SEM) with formative Object, Scene, and User State composites predicting recall, also benchmarking against Random Forest and multilayer perceptron classifiers. PLS-SEM attains the best F1 score in three of four studies. Additionally, path estimates identify lighting, augmentation density, AR registration stability, cognitive load, and AR familiarity as primary drivers. The model outputs per-object recall probabilities that can drive interface adjustments when predicted recall falls. Overall, PLS-SEM provides competitive accuracy with interpretable levers for design and evaluation in mobile AR.



## **18. Fortytwo: Swarm Inference with Peer-Ranked Consensus**

cs.LG

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.24801v1) [paper-pdf](http://arxiv.org/pdf/2510.24801v1)

**Authors**: Vladyslav Larin, Ihor Naumenko, Aleksei Ivashov, Ivan Nikitin, Alexander Firsov

**Abstract**: As centralized AI hits compute ceilings and diminishing returns from ever-larger training runs, meeting demand requires an inference layer that scales horizontally in both capacity and capability. We present Fortytwo, a novel protocol that leverages swarm intelligence principles and distributed pairwise ranking consensus to achieve superior performance in AI inference. Our approach reimagines collaboration among AI nodes using swarm inference: a peer-ranked, reputation-weighted consensus across heterogeneous models that surfaces the highest-quality responses. Using pairwise ranking with a custom Bradley-Terry-style aggregation model, we demonstrate that swarm inference substantially outperforms majority voting, achieving 85.90% on GPQA Diamond versus 68.69% for majority voting with the same model set - an improvement of +17.21 percentage points (approximately +25.1% relative). The protocol incorporates on-chain reputation so node influence adapts to demonstrated accuracy over time, yielding a meritocratic consensus that filters low-quality or malicious participants. To resist Sybil attacks, Fortytwo employs proof-of-capability in its consensus: nodes must successfully complete calibration/test requests and stake reputation to enter ranking rounds, making multi-identity attacks economically unattractive while preserving openness. Across six challenging benchmarks, including GPQA Diamond, LiveCodeBench, and AIME, our evaluation indicates higher accuracy and strong resilience to adversarial and noisy free-form prompting (e.g., prompt-injection degradation of only 0.12% versus 6.20% for a monolithic single-model baseline), while retaining practical deployability. Together, these results establish a foundation for decentralized AI systems - democratizing access to high-quality inference through collective intelligence without sacrificing reliability or security.



## **19. Secure Control of Connected and Autonomous Electrified Vehicles Under Adversarial Cyber-Attacks**

eess.SY

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.23922v1) [paper-pdf](http://arxiv.org/pdf/2510.23922v1)

**Authors**: Shashank Dhananjay Vyas, Satadru Dey

**Abstract**: Connected and Autonomous Electrified Vehicles (CAEV) is the solution to the future smart mobility having benefits of efficient traffic flow and cleaner environmental impact. Although CAEV has advantages they are still susceptible to adversarial cyber attacks due to their autonomous electric operation and the involved connectivity. To alleviate this issue, we propose a secure control architecture of CAEV. Particularly, we design an additional control input using Reinforcement Learning (RL) to be applied to the vehicle powertrain along with the input commanded by the battery. We present simulation case studies to demonstrate the potential of the proposed approach in keeping the CAEV platoon operating safely without collisions by curbing the effect of adversarial attacks.



## **20. Securing Transfer-Learned Networks with Reverse Homomorphic Encryption**

cs.CR

added protection via RHE and black box attacks

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2505.14323v2) [paper-pdf](http://arxiv.org/pdf/2505.14323v2)

**Authors**: Robert Allison, Tomasz Maciążek, Henry Bourne

**Abstract**: The growing body of literature on training-data reconstruction attacks raises significant concerns about deploying neural network classifiers trained on sensitive data. However, differentially private (DP) training (e.g. using DP-SGD) can defend against such attacks with large training datasets causing only minimal loss of network utility. Folklore, heuristics, and (albeit pessimistic) DP bounds suggest this fails for networks trained with small per-class datasets, yet to the best of our knowledge the literature offers no compelling evidence. We directly demonstrate this vulnerability by significantly extending reconstruction attack capabilities under a realistic adversary threat model for few-shot transfer learned image classifiers. We design new white-box and black-box attacks and find that DP-SGD is unable to defend against these without significant classifier utility loss. To address this, we propose a novel homomorphic encryption (HE) method that protects training data without degrading model's accuracy. Conventional HE secures model's input data and requires costly homomorphic implementation of the entire classifier. In contrast, our new scheme is computationally efficient and protects training data rather than input data. This is achieved by means of a simple role-reversal where classifier input data is unencrypted but transfer-learned weights are encrypted. Classifier outputs remain encrypted, thus preventing both white-box and black-box (and any other) training-data reconstruction attacks. Under this new scheme only a trusted party with a private decryption key can obtain the classifier class decisions.



## **21. Apollo: A Posteriori Label-Only Membership Inference Attack Towards Machine Unlearning**

cs.LG

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2506.09923v2) [paper-pdf](http://arxiv.org/pdf/2506.09923v2)

**Authors**: Liou Tang, James Joshi, Ashish Kundu

**Abstract**: Machine Unlearning (MU) aims to update Machine Learning (ML) models following requests to remove training samples and their influences on a trained model efficiently without retraining the original ML model from scratch. While MU itself has been employed to provide privacy protection and regulatory compliance, it can also increase the attack surface of the model. Existing privacy inference attacks towards MU that aim to infer properties of the unlearned set rely on the weaker threat model that assumes the attacker has access to both the unlearned model and the original model, limiting their feasibility toward real-life scenarios. We propose a novel privacy attack, A Posteriori Label-Only Membership Inference Attack towards MU, Apollo, that infers whether a data sample has been unlearned, following a strict threat model where an adversary has access to the label-output of the unlearned model only. We demonstrate that our proposed attack, while requiring less access to the target model compared to previous attacks, can achieve relatively high precision on the membership status of the unlearned samples.



## **22. UNDREAM: Bridging Differentiable Rendering and Photorealistic Simulation for End-to-end Adversarial Attacks**

cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.16923v2) [paper-pdf](http://arxiv.org/pdf/2510.16923v2)

**Authors**: Mansi Phute, Matthew Hull, Haoran Wang, Alec Helbling, ShengYun Peng, Willian Lunardi, Martin Andreoni, Wenke Lee, Duen Horng Chau

**Abstract**: Deep learning models deployed in safety critical applications like autonomous driving use simulations to test their robustness against adversarial attacks in realistic conditions. However, these simulations are non-differentiable, forcing researchers to create attacks that do not integrate simulation environmental factors, reducing attack success. To address this limitation, we introduce UNDREAM, the first software framework that bridges the gap between photorealistic simulators and differentiable renderers to enable end-to-end optimization of adversarial perturbations on any 3D objects. UNDREAM enables manipulation of the environment by offering complete control over weather, lighting, backgrounds, camera angles, trajectories, and realistic human and object movements, thereby allowing the creation of diverse scenes. We showcase a wide array of distinct physically plausible adversarial objects that UNDREAM enables researchers to swiftly explore in different configurable environments. This combination of photorealistic simulation and differentiable optimization opens new avenues for advancing research of physical adversarial attacks.



## **23. On the Stability of Graph Convolutional Neural Networks: A Probabilistic Perspective**

cs.LG

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2506.01213v4) [paper-pdf](http://arxiv.org/pdf/2506.01213v4)

**Authors**: Ning Zhang, Henry Kenlay, Li Zhang, Mihai Cucuringu, Xiaowen Dong

**Abstract**: Graph convolutional neural networks (GCNNs) have emerged as powerful tools for analyzing graph-structured data, achieving remarkable success across diverse applications. However, the theoretical understanding of the stability of these models, i.e., their sensitivity to small changes in the graph structure, remains in rather limited settings, hampering the development and deployment of robust and trustworthy models in practice. To fill this gap, we study how perturbations in the graph topology affect GCNN outputs and propose a novel formulation for analyzing model stability. Unlike prior studies that focus only on worst-case perturbations, our distribution-aware formulation characterizes output perturbations across a broad range of input data. This way, our framework enables, for the first time, a probabilistic perspective on the interplay between the statistical properties of the node data and perturbations in the graph topology. We conduct extensive experiments to validate our theoretical findings and demonstrate their benefits over existing baselines, in terms of both representation stability and adversarial attacks on downstream tasks. Our results demonstrate the practical significance of the proposed formulation and highlight the importance of incorporating data distribution into stability analysis.



## **24. Authentication Against Insecure Bootstrapping for 5G Networks: Feasibility, Resiliency, and Transitional Solutions in Post-Quantum Era**

cs.CR

17 pages, 3 tables, 6 figures

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.23457v1) [paper-pdf](http://arxiv.org/pdf/2510.23457v1)

**Authors**: Saleh Darzi, Mirza Masfiqur Rahman, Imtiaz Karim, Rouzbeh Behnia, Attila A Yavuz, Elisa Bertino

**Abstract**: The 5G protocol lacks a robust base station authentication mechanism during the initial bootstrapping phase, leaving it susceptible to threats such as fake base station attacks. Conventional solutions, including digital signatures based on Public Key Infrastructures (PKIs) and identity-based signatures, are inadequate against quantum-capable adversaries. While integrating NIST's Post-Quantum Cryptography (PQC) standards is a leading approach for quantum resistance, their suitability for 5G base station authentication remains unexplored. Moreover, current solutions are predominantly centralized and lack security features such as distributed authentication. This work presents, to our knowledge, the first comprehensive network-level performance characterization of integrating NIST-PQC standards and conventional digital signatures (including threshold and identity-based schemes) into 5G base station authentication. Our findings reveal significant feasibility concerns, with direct PQC adoption hindered by protocol constraints and large signature sizes. We also highlight the performance limitations of conventional methods due to the overhead of certificate chains. To mitigate these challenges, we propose BORG, a transitional authentication solution based on a Hierarchical Identity-Based Threshold Signature scheme with a Fail-Stop property. BORG offers post-mortem post-quantum forgery detection and distributed trust via threshold and compact signatures, well-suited for 5G's stringent requirements. Our performance analysis underscores an important warning on the infeasibility of direct PQC integration and positions BORG as an effective transitional solution toward future quantum-resilient 5G authentication.



## **25. Generalization Bounds for Robust Contrastive Learning: From Theory to Practice**

cs.LG

13 pages, 1 figure, 4 tables

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2311.09671v2) [paper-pdf](http://arxiv.org/pdf/2311.09671v2)

**Authors**: Ngoc N. Tran, Lam Tran, Hoang Phan, Anh Bui, Tung Pham, Toan Tran, Dinh Phung, Trung Le

**Abstract**: Contrastive Learning first extracts features from unlabeled data, followed by linear probing with labeled data. Adversarial Contrastive Learning (ACL) integrates Adversarial Training into the first phase to enhance feature robustness against attacks in the probing phase. While ACL has shown strong empirical results, its theoretical understanding remains limited. Furthermore, while a fair amount of theoretical works analyze how the unsupervised loss can support the supervised loss in the probing phase, none has examined its role to the robust supervised loss. To fill this gap, our work develops rigorous theories to identify which components in the unsupervised training can help improve the robust supervised loss. Specifically, besides the adversarial contrastive loss, we reveal that the benign one, along with a global divergence between benign and adversarial examples can also improve robustness. Proper experiments are conducted to justify our findings.



## **26. Attention! Your Vision Language Model Could Be Maliciously Manipulated**

cs.CV

NeurIPS 2025

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2505.19911v2) [paper-pdf](http://arxiv.org/pdf/2505.19911v2)

**Authors**: Xiaosen Wang, Shaokang Wang, Zhijin Ge, Yuyang Luo, Shudong Zhang

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable success in understanding complex real-world scenarios and supporting data-driven decision-making processes. However, VLMs exhibit significant vulnerability against adversarial examples, either text or image, which can lead to various adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In this work, we empirically and theoretically demonstrate that VLMs are particularly susceptible to image-based adversarial examples, where imperceptible perturbations can precisely manipulate each output token. To this end, we propose a novel attack called Vision-language model Manipulation Attack (VMA), which integrates first-order and second-order momentum optimization techniques with a differentiable transformation mechanism to effectively optimize the adversarial perturbation. Notably, VMA can be a double-edged sword: it can be leveraged to implement various attacks, such as jailbreaking, hijacking, privacy breaches, Denial-of-Service, and the generation of sponge examples, etc, while simultaneously enabling the injection of watermarks for copyright protection. Extensive empirical evaluations substantiate the efficacy and generalizability of VMA across diverse scenarios and datasets. Code is available at https://github.com/Trustworthy-AI-Group/VMA.



## **27. Exploring Semantic-constrained Adversarial Example with Instruction Uncertainty Reduction**

cs.AI

NeurIPS 2025

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.22981v1) [paper-pdf](http://arxiv.org/pdf/2510.22981v1)

**Authors**: Jin Hu, Jiakai Wang, Linna Jing, Haolin Li, Haodong Liu, Haotong Qin, Aishan Liu, Ke Xu, Xianglong Liu

**Abstract**: Recently, semantically constrained adversarial examples (SemanticAE), which are directly generated from natural language instructions, have become a promising avenue for future research due to their flexible attacking forms. To generate SemanticAEs, current methods fall short of satisfactory attacking ability as the key underlying factors of semantic uncertainty in human instructions, such as referring diversity, descriptive incompleteness, and boundary ambiguity, have not been fully investigated. To tackle the issues, this paper develops a multi-dimensional instruction uncertainty reduction (InSUR) framework to generate more satisfactory SemanticAE, i.e., transferable, adaptive, and effective. Specifically, in the dimension of the sampling method, we propose the residual-driven attacking direction stabilization to alleviate the unstable adversarial optimization caused by the diversity of language references. By coarsely predicting the language-guided sampling process, the optimization process will be stabilized by the designed ResAdv-DDIM sampler, therefore releasing the transferable and robust adversarial capability of multi-step diffusion models. In task modeling, we propose the context-encoded attacking scenario constraint to supplement the missing knowledge from incomplete human instructions. Guidance masking and renderer integration are proposed to regulate the constraints of 2D/3D SemanticAE, activating stronger scenario-adapted attacks. Moreover, in the dimension of generator evaluation, we propose the semantic-abstracted attacking evaluation enhancement by clarifying the evaluation boundary, facilitating the development of more effective SemanticAE generators. Extensive experiments demonstrate the superiority of the transfer attack performance of InSUR. Moreover, we realize the reference-free generation of semantically constrained 3D adversarial examples for the first time.



## **28. PersonaTeaming: Exploring How Introducing Personas Can Improve Automated AI Red-Teaming**

cs.AI

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2509.03728v3) [paper-pdf](http://arxiv.org/pdf/2509.03728v3)

**Authors**: Wesley Hanwen Deng, Sunnie S. Y. Kim, Akshita Jha, Ken Holstein, Motahhare Eslami, Lauren Wilcox, Leon A Gatys

**Abstract**: Recent developments in AI governance and safety research have called for red-teaming methods that can effectively surface potential risks posed by AI models. Many of these calls have emphasized how the identities and backgrounds of red-teamers can shape their red-teaming strategies, and thus the kinds of risks they are likely to uncover. While automated red-teaming approaches promise to complement human red-teaming by enabling larger-scale exploration of model behavior, current approaches do not consider the role of identity. As an initial step towards incorporating people's background and identities in automated red-teaming, we develop and evaluate a novel method, PersonaTeaming, that introduces personas in the adversarial prompt generation process to explore a wider spectrum of adversarial strategies. In particular, we first introduce a methodology for mutating prompts based on either "red-teaming expert" personas or "regular AI user" personas. We then develop a dynamic persona-generating algorithm that automatically generates various persona types adaptive to different seed prompts. In addition, we develop a set of new metrics to explicitly measure the "mutation distance" to complement existing diversity measurements of adversarial prompts. Our experiments show promising improvements (up to 144.1%) in the attack success rates of adversarial prompts through persona mutation, while maintaining prompt diversity, compared to RainbowPlus, a state-of-the-art automated red-teaming method. We discuss the strengths and limitations of different persona types and mutation methods, shedding light on future opportunities to explore complementarities between automated and human red-teaming approaches.



## **29. CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents**

cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.22963v1) [paper-pdf](http://arxiv.org/pdf/2510.22963v1)

**Authors**: Zesen Liu, Zhixiang Zhang, Yuchong Xie, Dongdong She

**Abstract**: LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to 80% attack success and 98% preference flips, while remaining highly stealthy and transferable. Case studies in VSCode Cline and Ollama confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections.



## **30. Your Compiler is Backdooring Your Model: Understanding and Exploiting Compilation Inconsistency Vulnerabilities in Deep Learning Compilers**

cs.CR

This paper is accepted to IEEE S&P 2026, the code is available at  https://github.com/SeekingDream/DLCompilerAttack

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2509.11173v3) [paper-pdf](http://arxiv.org/pdf/2509.11173v3)

**Authors**: Simin Chen, Jinjun Peng, Yixin He, Junfeng Yang, Baishakhi Ray

**Abstract**: Deep learning (DL) compilers are core infrastructure in modern DL systems, offering flexibility and scalability beyond vendor-specific libraries. This work uncovers a fundamental vulnerability in their design: can an official, unmodified compiler alter a model's semantics during compilation and introduce hidden backdoors? We study both adversarial and natural settings. In the adversarial case, we craft benign models where triggers have no effect pre-compilation but become effective backdoors after compilation. Tested on six models, three commercial compilers, and two hardware platforms, our attack yields 100% success on triggered inputs while preserving normal accuracy and remaining undetected by state-of-the-art detectors. The attack generalizes across compilers, hardware, and floating-point settings. In the natural setting, we analyze the top 100 HuggingFace models (including one with 220M+ downloads) and find natural triggers in 31 models. This shows that compilers can introduce risks even without adversarial manipulation.   Our results reveal an overlooked threat: unmodified DL compilers can silently alter model semantics. To our knowledge, this is the first work to expose inherent security risks in DL compiler design, opening a new direction for secure and trustworthy ML.



## **31. QuantumShield: Multilayer Fortification for Quantum Federated Learning**

cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.22945v1) [paper-pdf](http://arxiv.org/pdf/2510.22945v1)

**Authors**: Dev Gurung, Shiva Raj Pokhrel

**Abstract**: In this paper, we propose a groundbreaking quantum-secure federated learning (QFL) framework designed to safeguard distributed learning systems against the emerging threat of quantum-enabled adversaries. As classical cryptographic methods become increasingly vulnerable to quantum attacks, our framework establishes a resilient security architecture that remains robust even in the presence of quantum-capable attackers. We integrate and rigorously evaluate advanced quantum and post-quantum protocols including Quantum Key Distribution (QKD), Quantum Teleportation, Key Encapsulation Mechanisms (KEM) and Post-Quantum Cryptography (PQC) to fortify the QFL process against both classical and quantum threats. These mechanisms are systematically analyzed and implemented to demonstrate their seamless interoperability within a secure and scalable QFL ecosystem. Through comprehensive theoretical modeling and experimental validation, this work provides a detailed security and performance assessment of the proposed framework. Our findings lay a strong foundation for next-generation federated learning systems that are inherently secure in the quantum era.



## **32. Is Your Prompt Poisoning Code? Defect Induction Rates and Security Mitigation Strategies**

cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.22944v1) [paper-pdf](http://arxiv.org/pdf/2510.22944v1)

**Authors**: Bin Wang, YiLu Zhong, MiDi Wan, WenJie Yu, YuanBing Ouyang, Yenan Huang, Hui Li

**Abstract**: Large language models (LLMs) have become indispensable for automated code generation, yet the quality and security of their outputs remain a critical concern. Existing studies predominantly concentrate on adversarial attacks or inherent flaws within the models. However, a more prevalent yet underexplored issue concerns how the quality of a benign but poorly formulated prompt affects the security of the generated code. To investigate this, we first propose an evaluation framework for prompt quality encompassing three key dimensions: goal clarity, information completeness, and logical consistency. Based on this framework, we construct and publicly release CWE-BENCH-PYTHON, a large-scale benchmark dataset containing tasks with prompts categorized into four distinct levels of normativity (L0-L3). Extensive experiments on multiple state-of-the-art LLMs reveal a clear correlation: as prompt normativity decreases, the likelihood of generating insecure code consistently and markedly increases. Furthermore, we demonstrate that advanced prompting techniques, such as Chain-of-Thought and Self-Correction, effectively mitigate the security risks introduced by low-quality prompts, substantially improving code safety. Our findings highlight that enhancing the quality of user prompts constitutes a critical and effective strategy for strengthening the security of AI-generated code.



## **33. Self-Calibrated Consistency can Fight Back for Adversarial Robustness in Vision-Language Models**

cs.CV

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22785v1) [paper-pdf](http://arxiv.org/pdf/2510.22785v1)

**Authors**: Jiaxiang Liu, Jiawei Du, Xiao Liu, Prayag Tiwari, Mingkun Xu

**Abstract**: Pre-trained vision-language models (VLMs) such as CLIP have demonstrated strong zero-shot capabilities across diverse domains, yet remain highly vulnerable to adversarial perturbations that disrupt image-text alignment and compromise reliability. Existing defenses typically rely on adversarial fine-tuning with labeled data, limiting their applicability in zero-shot settings. In this work, we identify two key weaknesses of current CLIP adversarial attacks -- lack of semantic guidance and vulnerability to view variations -- collectively termed semantic and viewpoint fragility. To address these challenges, we propose Self-Calibrated Consistency (SCC), an effective test-time defense. SCC consists of two complementary modules: Semantic consistency, which leverages soft pseudo-labels from counterattack warm-up and multi-view predictions to regularize cross-modal alignment and separate the target embedding from confusable negatives; and Spatial consistency, aligning perturbed visual predictions via augmented views to stabilize inference under adversarial perturbations. Together, these modules form a plug-and-play inference strategy. Extensive experiments on 22 benchmarks under diverse attack settings show that SCC consistently improves the zero-shot robustness of CLIP while maintaining accuracy, and can be seamlessly integrated with other VLMs for further gains. These findings highlight the great potential of establishing an adversarially robust paradigm from CLIP, with implications extending to broader vision-language domains such as BioMedCLIP.



## **34. SpoofTrackBench: Interpretable AI for Spoof-Aware UAV Tracking and Benchmarking**

cs.CR

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22726v1) [paper-pdf](http://arxiv.org/pdf/2510.22726v1)

**Authors**: Van Le, Tan Le

**Abstract**: SpoofTrackBench is a reproducible, modular benchmark for evaluating adversarial robustness in real-time localization and tracking (RTLS) systems under radar spoofing. Leveraging the Hampton University Skyler Radar Sensor dataset, we simulate drift, ghost, and mirror-type spoofing attacks and evaluate tracker performance using both Joint Probabilistic Data Association (JPDA) and Global Nearest Neighbor (GNN) architectures. Our framework separates clean and spoofed detection streams, visualizes spoof-induced trajectory divergence, and quantifies assignment errors via direct drift-from-truth metrics. Clustering overlays, injection-aware timelines, and scenario-adaptive visualizations enable interpretability across spoof types and configurations. Evaluation figures and logs are auto-exported for reproducible comparison. SpoofTrackBench sets a new standard for open, ethical benchmarking of spoof-aware tracking pipelines, enabling rigorous cross-architecture analysis and community validation.



## **35. Measuring the (Un)Faithfulness of Concept-Based Explanations**

cs.LG

Pre-print

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2504.10833v2) [paper-pdf](http://arxiv.org/pdf/2504.10833v2)

**Authors**: Shubham Kumar, Narendra Ahuja

**Abstract**: Post-hoc, unsupervised concept-based explanation methods (U-CBEMs) translate a vision model's internal reasoning into human-understandable concepts, leading to interpretable explanations. However, we find that many state-of-the-art (SOTA) U-CBEMs are not faithful: their concepts seem interpretable but fail to reproduce the model's predictions. We argue that this deficiency has gone unnoticed due to fragmented evaluation - each paper proposes its own faithfulness measure, with no measure-over-measure comparison or broad benchmarking. We close this gap by (i) organizing prior metrics in a unified framework, discussing their limitations, and identifying desiderata for a faithfulness measure; (ii) introducing the Surrogate Faithfulness (SURF) measure, which quantifies faithfulness via the predictive loss of a surrogate that maps explanations to the model's outputs; and (iii) delivering the first comprehensive U-CBEM faithfulness benchmark across diverse tasks and architectures. In a controlled setting, SURF outperforms prior faithfulness measures in measure-over-measure comparisons, and applying SURF to SOTA U-CBEMs reveals that many visually appealing U-CBEMs are surprisingly unfaithful. We demonstrate SURF applicability in two downstream settings - (i) faithfulness versus the number of concepts used in the explanation and (ii) U-CBEM robustness to adversarial attacks - underscoring SURF's value as a reliable faithfulness measure. Code to be released.



## **36. If You Want to Be Robust, Be Wary of Initialization**

cs.LG

Accepted at NeurIPS 2024

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22652v1) [paper-pdf](http://arxiv.org/pdf/2510.22652v1)

**Authors**: Sofiane Ennadir, Johannes F. Lutzeyer, Michalis Vazirgiannis, El Houcine Bergou

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable performance across a spectrum of graph-related tasks, however concerns persist regarding their vulnerability to adversarial perturbations. While prevailing defense strategies focus primarily on pre-processing techniques and adaptive message-passing schemes, this study delves into an under-explored dimension: the impact of weight initialization and associated hyper-parameters, such as training epochs, on a model's robustness. We introduce a theoretical framework bridging the connection between initialization strategies and a network's resilience to adversarial perturbations. Our analysis reveals a direct relationship between initial weights, number of training epochs and the model's vulnerability, offering new insights into adversarial robustness beyond conventional defense mechanisms. While our primary focus is on GNNs, we extend our theoretical framework, providing a general upper-bound applicable to Deep Neural Networks. Extensive experiments, spanning diverse models and real-world datasets subjected to various adversarial attacks, validate our findings. We illustrate that selecting appropriate initialization not only ensures performance on clean datasets but also enhances model robustness against adversarial perturbations, with observed gaps of up to 50\% compared to alternative initialization approaches.



## **37. Enhancing Graph Classification Robustness with Singular Pooling**

cs.LG

Accepted at Neurips 2025

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22643v1) [paper-pdf](http://arxiv.org/pdf/2510.22643v1)

**Authors**: Sofiane Ennadir, Oleg Smirnov, Yassine Abbahaddou, Lele Cao, Johannes F. Lutzeyer

**Abstract**: Graph Neural Networks (GNNs) have achieved strong performance across a range of graph representation learning tasks, yet their adversarial robustness in graph classification remains underexplored compared to node classification. While most existing defenses focus on the message-passing component, this work investigates the overlooked role of pooling operations in shaping robustness. We present a theoretical analysis of standard flat pooling methods (sum, average and max), deriving upper bounds on their adversarial risk and identifying their vulnerabilities under different attack scenarios and graph structures. Motivated by these insights, we propose \textit{Robust Singular Pooling (RS-Pool)}, a novel pooling strategy that leverages the dominant singular vector of the node embedding matrix to construct a robust graph-level representation. We theoretically investigate the robustness of RS-Pool and interpret the resulting bound leading to improved understanding of our proposed pooling operator. While our analysis centers on Graph Convolutional Networks (GCNs), RS-Pool is model-agnostic and can be implemented efficiently via power iteration. Empirical results on real-world benchmarks show that RS-Pool provides better robustness than the considered pooling methods when subject to state-of-the-art adversarial attacks while maintaining competitive clean accuracy. Our code is publicly available at:\href{https://github.com/king/rs-pool}{https://github.com/king/rs-pool}.



## **38. Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing**

eess.AS

Accepted to IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2504.05657v2) [paper-pdf](http://arxiv.org/pdf/2504.05657v2)

**Authors**: Tianchi Liu, Duc-Tuan Truong, Rohan Kumar Das, Kong Aik Lee, Haizhou Li

**Abstract**: Speech foundation models have significantly advanced various speech-related tasks by providing exceptional representation capabilities. However, their high-dimensional output features often create a mismatch with downstream task models, which typically require lower-dimensional inputs. A common solution is to apply a dimensionality reduction (DR) layer, but this approach increases parameter overhead, computational costs, and risks losing valuable information. To address these issues, we propose Nested Res2Net (Nes2Net), a lightweight back-end architecture designed to directly process high-dimensional features without DR layers. The nested structure enhances multi-scale feature extraction, improves feature interaction, and preserves high-dimensional information. We first validate Nes2Net on CtrSVDD, a singing voice deepfake detection dataset, and report a 22% performance improvement and an 87% back-end computational cost reduction over the state-of-the-art baseline. Additionally, extensive testing across four diverse datasets: ASVspoof 2021, ASVspoof 5, PartialSpoof, and In-the-Wild, covering fully spoofed speech, adversarial attacks, partial spoofing, and real-world scenarios, consistently highlights Nes2Net's superior robustness and generalization capabilities. The code package and pre-trained models are available at https://github.com/Liu-Tianchi/Nes2Net.



## **39. Sentra-Guard: A Multilingual Human-AI Framework for Real-Time Defense Against Adversarial LLM Jailbreaks**

cs.CR

11 pages, 5 figures. Preprint version under review in the area of  Artificial Intelligence (cs.AI)

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22628v1) [paper-pdf](http://arxiv.org/pdf/2510.22628v1)

**Authors**: Md. Mehedi Hasan, Ziaur Rahman, Rafid Mostafiz, Md. Abir Hossain

**Abstract**: This paper presents a real-time modular defense system named Sentra-Guard. The system detects and mitigates jailbreak and prompt injection attacks targeting large language models (LLMs). The framework uses a hybrid architecture with FAISS-indexed SBERT embedding representations that capture the semantic meaning of prompts, combined with fine-tuned transformer classifiers, which are machine learning models specialized for distinguishing between benign and adversarial language inputs. It identifies adversarial prompts in both direct and obfuscated attack vectors. A core innovation is the classifier-retriever fusion module, which dynamically computes context-aware risk scores that estimate how likely a prompt is to be adversarial based on its content and context. The framework ensures multilingual resilience with a language-agnostic preprocessing layer. This component automatically translates non-English prompts into English for semantic evaluation, enabling consistent detection across over 100 languages. The system includes a HITL feedback loop, where decisions made by the automated system are reviewed by human experts for continual learning and rapid adaptation under adversarial pressure. Sentra-Guard maintains an evolving dual-labeled knowledge base of benign and malicious prompts, enhancing detection reliability and reducing false positives. Evaluation results show a 99.96% detection rate (AUC = 1.00, F1 = 1.00) and an attack success rate (ASR) of only 0.004%. This outperforms leading baselines such as LlamaGuard-2 (1.3%) and OpenAI Moderation (3.7%). Unlike black-box approaches, Sentra-Guard is transparent, fine-tunable, and compatible with diverse LLM backends. Its modular design supports scalable deployment in both commercial and open-source environments. The system establishes a new state-of-the-art in adversarial LLM defense.



## **40. Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents**

cs.CR

Julia Bazinska and Max Mathys contributed equally

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22620v1) [paper-pdf](http://arxiv.org/pdf/2510.22620v1)

**Authors**: Julia Bazinska, Max Mathys, Francesco Casucci, Mateo Rojas-Carulla, Xander Davies, Alexandra Souly, Niklas Pfister

**Abstract**: AI agents powered by large language models (LLMs) are being deployed at scale, yet we lack a systematic understanding of how the choice of backbone LLM affects agent security. The non-deterministic sequential nature of AI agents complicates security modeling, while the integration of traditional software with AI components entangles novel LLM vulnerabilities with conventional security risks. Existing frameworks only partially address these challenges as they either capture specific vulnerabilities only or require modeling of complete agents. To address these limitations, we introduce threat snapshots: a framework that isolates specific states in an agent's execution flow where LLM vulnerabilities manifest, enabling the systematic identification and categorization of security risks that propagate from the LLM to the agent level. We apply this framework to construct the $\operatorname{b}^3$ benchmark, a security benchmark based on 194331 unique crowdsourced adversarial attacks. We then evaluate 31 popular LLMs with it, revealing, among other insights, that enhanced reasoning capabilities improve security, while model size does not correlate with security. We release our benchmark, dataset, and evaluation code to facilitate widespread adoption by LLM providers and practitioners, offering guidance for agent developers and incentivizing model developers to prioritize backbone security improvements.



## **41. Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks**

cs.CV

Accepted for Oral Presentation at SafeMM-AI @ ICCV 2025 (Spotlight)

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2502.09110v3) [paper-pdf](http://arxiv.org/pdf/2502.09110v3)

**Authors**: Eylon Mizrahi, Raz Lapid, Moshe Sipper

**Abstract**: Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations that can significantly degrade model performance. Conventional defense mechanisms predominantly focus on either enhancing model robustness or detecting adversarial inputs independently. In this work, we propose an Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN) to uncover adversarial behavior within auxiliary feature representations, without the need for adversarial examples. U-CAN is embedded within selected intermediate layers of the target model. These auxiliary networks, comprising projection layers and ArcFace-based linear layers, refine feature representations to more effectively distinguish between benign and adversarial inputs. Comprehensive experiments across multiple datasets (CIFAR-10, Mammals, and a subset of ImageNet) and architectures (ResNet-50, VGG-16, and ViT) demonstrate that our method surpasses existing unsupervised adversarial detection techniques, achieving superior F1 scores against four distinct attack methods. The proposed framework provides a scalable and effective solution for enhancing the security and reliability of deep learning systems.



## **42. FAARM: Firmware Attestation and Authentication Framework for Mali GPUs**

cs.CR

10 pages, 8 figures. Preprint version under review in the area of  Computer Security (cs.CR)

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22566v1) [paper-pdf](http://arxiv.org/pdf/2510.22566v1)

**Authors**: Md. Mehedi Hasan

**Abstract**: Recent work has revealed MOLE, the first practical attack to compromise GPU Trusted Execution Environments (TEEs), by injecting malicious firmware into the embedded Microcontroller Unit (MCU) of Arm Mali GPUs. By exploiting the absence of cryptographic verification during initialization, adversaries with kernel privileges can bypass memory protections, exfiltrate sensitive data at over 40 MB/s, and tamper with inference results, all with negligible runtime overhead. This attack surface affects commodity mobile SoCs and cloud accelerators, exposing a critical firmware-level trust gap in existing GPU TEE designs. To address this gap, this paper presents FAARM, a lightweight Firmware Attestation and Authentication framework that prevents MOLE-style firmware subversion. FAARM integrates digital signature verification at the EL3 secure monitor using vendor-signed firmware bundles and an on-device public key anchor. At boot, EL3 verifies firmware integrity and authenticity, enforces version checks, and locks the firmware region, eliminating both pre-verification and time-of-check-to-time-of-use (TOCTOU) attack vectors. We implement FAARM as a software-only prototype on a Mali GPU testbed, using a Google Colab-based emulation framework that models the firmware signing process, the EL1 to EL3 load path, and secure memory configuration. FAARM reliably detects and blocks malicious firmware injections, rejecting tampered images before use and denying overwrite attempts after attestation. Firmware verification incurs only 1.34 ms latency on average, demonstrating that strong security can be achieved with negligible overhead. FAARM thus closes a fundamental gap in shim-based GPU TEEs, providing a practical, deployable defense that raises the security baseline for both mobile and cloud GPU deployments.



## **43. Cross-Paradigm Graph Backdoor Attacks with Promptable Subgraph Triggers**

cs.CR

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22555v1) [paper-pdf](http://arxiv.org/pdf/2510.22555v1)

**Authors**: Dongyi Liu, Jiangtong Li, Dawei Cheng, Changjun Jiang

**Abstract**: Graph Neural Networks(GNNs) are vulnerable to backdoor attacks, where adversaries implant malicious triggers to manipulate model predictions.   Existing trigger generators are often simplistic in structure and overly reliant on specific features, confining them to a single graph learning paradigm, such as graph supervised learning, graph contrastive learning, or graph prompt learning.   This specialized design, which aligns the trigger with one learning objective, results in poor transferability when applied to other learning paradigms.   For instance, triggers generated for the graph supervised learning paradigm perform poorly when tested within graph contrastive learning or graph prompt learning environments.   Furthermore, these simple generators often fail to utilize complex structural information or node diversity within the graph data.   These constraints limit the attack success rates of such methods in general testing scenarios.   Therefore, to address these limitations, we propose Cross-Paradigm Graph Backdoor Attacks with Promptable Subgraph Triggers(CP-GBA), a new transferable graph backdoor attack that employs graph prompt learning(GPL) to train a set of universal subgraph triggers.   First, we distill a compact yet expressive trigger set from target graphs, which is structured as a queryable repository, by jointly enforcing class-awareness, feature richness, and structural fidelity.   Second, we conduct the first exploration of the theoretical transferability of GPL to train these triggers under prompt-based objectives, enabling effective generalization to diverse and unseen test-time paradigms.   Extensive experiments across multiple real-world datasets and defense scenarios show that CP-GBA achieves state-of-the-art attack success rates.



## **44. Security of Gradient Tracking Algorithms Against Malicious Agents**

eess.SY

under review

**SubmitDate**: 2025-10-25    [abs](http://arxiv.org/abs/2505.14473v2) [paper-pdf](http://arxiv.org/pdf/2505.14473v2)

**Authors**: Sribalaji C. Anand, Alexander J Gallo, Nicola Bastianello

**Abstract**: Consensus algorithms are fundamental to multi-agent distributed optimization, and their security under adversarial conditions is an active area of research. While prior works primarily establish conditions for successful global consensus under attack, little is known about system behavior when these conditions are violated. This paper addresses this gap by investigating the robustness of the Wang--Elia algorithm, which is a robust to noise version of gradient tracking algorithm, in the presence of malicious agents. We consider a network of agents collaboratively minimizing a global cost function, where a subset of agents may transmit faulty information to disrupt consensus. To quantify resilience, we formulate a security metric as an optimization problem, which is rooted in centralized attack detection literature. We provide a tractable reformulation of the optimization problem, and derive conditions under which the metric becomes unbounded, identifying undetectable attack signals that reveal inherent vulnerabilities. To facilitate design and analysis, we propose a well-posed variant of the metric and propose design methods to enhance network robustness against stealthy adversarial attacks. Numerical examples demonstrate the effectiveness of the proposed framework to enhance the resilience of multi-agent distributed optimization.



## **45. Adapting Noise-Driven PUF and AI for Secure WBG ICS: A Proof-of-Concept Study**

cs.CR

**SubmitDate**: 2025-10-25    [abs](http://arxiv.org/abs/2510.22283v1) [paper-pdf](http://arxiv.org/pdf/2510.22283v1)

**Authors**: Devon A. Kelly, Christiana Chamon

**Abstract**: Wide-bandgap (WBG) technologies offer unprecedented improvements in power system efficiency, size, and performance, but also introduce unique sensor corruption and cybersecurity risks in industrial control systems (ICS), particularly due to high-frequency noise and sophisticated cyber-physical threats. This proof-of-concept (PoC) study demonstrates the adaptation of a noise-driven physically unclonable function (PUF) and machine learning (ML)-assisted anomaly detection framework to the demanding environment of WBG-based ICS sensor pathways. By extracting entropy from unavoidable WBG switching noise (up to 100 kHz) as a PUF source, and simultaneously using this noise as a real-time threat indicator, the proposed system unites hardware-level authentication and anomaly detection. Our approach integrates hybrid machine learning (ML) models with adaptive Bayesian filtering, providing robust and low-latency detection capabilities resilient to both natural electromagnetic interference (EMI) and active adversarial manipulation. Through detailed simulations of WBG modules under benign and attack scenarios--including EMI injection, signal tampering, and node impersonation--we achieve 95% detection accuracy and sub-millisecond processing latency. These results demonstrate the feasibility of physics-driven, dual-use noise exploitation as a scalable ICS defense primitive. Our findings lay the groundwork for next-generation security strategies that leverage inherent device characteristics, bridging hardware and artificial intelligence (AI) for enhanced protection of critical ICS infrastructure.



## **46. SecureLearn -- An Attack-agnostic Defense for Multiclass Machine Learning Against Data Poisoning Attacks**

cs.CR

**SubmitDate**: 2025-10-25    [abs](http://arxiv.org/abs/2510.22274v1) [paper-pdf](http://arxiv.org/pdf/2510.22274v1)

**Authors**: Anum Paracha, Junaid Arshad, Mohamed Ben Farah, Khalid Ismail

**Abstract**: Data poisoning attacks are a potential threat to machine learning (ML) models, aiming to manipulate training datasets to disrupt their performance. Existing defenses are mostly designed to mitigate specific poisoning attacks or are aligned with particular ML algorithms. Furthermore, most defenses are developed to secure deep neural networks or binary classifiers. However, traditional multiclass classifiers need attention to be secure from data poisoning attacks, as these models are significant in developing multi-modal applications. Therefore, this paper proposes SecureLearn, a two-layer attack-agnostic defense to defend multiclass models from poisoning attacks. It comprises two components of data sanitization and a new feature-oriented adversarial training. To ascertain the effectiveness of SecureLearn, we proposed a 3D evaluation matrix with three orthogonal dimensions: data poisoning attack, data sanitization and adversarial training. Benchmarking SecureLearn in a 3D matrix, a detailed analysis is conducted at different poisoning levels (10%-20%), particularly analysing accuracy, recall, F1-score, detection and correction rates, and false discovery rate. The experimentation is conducted for four ML algorithms, namely Random Forest (RF), Decision Tree (DT), Gaussian Naive Bayes (GNB) and Multilayer Perceptron (MLP), trained with three public datasets, against three poisoning attacks and compared with two existing mitigations. Our results highlight that SecureLearn is effective against the provided attacks. SecureLearn has strengthened resilience and adversarial robustness of traditional multiclass models and neural networks, confirming its generalization beyond algorithm-specific defenses. It consistently maintained accuracy above 90%, recall and F1-score above 75%. For neural networks, SecureLearn achieved 97% recall and F1-score against all selected poisoning attacks.



## **47. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

cs.CV

NeurIPS 2025. Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-10-25    [abs](http://arxiv.org/abs/2503.10635v2) [paper-pdf](http://arxiv.org/pdf/2503.10635v2)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against closed-source commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial black-box LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we propose to refine semantic clarity by encoding explicit semantic details within local regions, thus ensuring the capture of finer-grained features and inter-model transferability, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective baseline: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. While the naive source-target matching method has been utilized before in the literature, we are the first to provide a tight analysis, which establishes a close connection between perturbation optimization and semantics. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5/3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods with lower $\ell_1/\ell_2$ perturbations.



## **48. Dual-Flow: Transferable Multi-Target, Instance-Agnostic Attacks via In-the-wild Cascading Flow Optimization**

cs.CV

Accepted at NeurIPS 2025

**SubmitDate**: 2025-10-25    [abs](http://arxiv.org/abs/2502.02096v3) [paper-pdf](http://arxiv.org/pdf/2502.02096v3)

**Authors**: Yixiao Chen, Shikun Sun, Jianshu Li, Ruoyu Li, Zhe Li, Junliang Xing

**Abstract**: Adversarial attacks are widely used to evaluate model robustness, and in black-box scenarios, the transferability of these attacks becomes crucial. Existing generator-based attacks have excellent generalization and transferability due to their instance-agnostic nature. However, when training generators for multi-target tasks, the success rate of transfer attacks is relatively low due to the limitations of the model's capacity. To address these challenges, we propose a novel Dual-Flow framework for multi-target instance-agnostic adversarial attacks, utilizing Cascading Distribution Shift Training to develop an adversarial velocity function. Extensive experiments demonstrate that Dual-Flow significantly improves transferability over previous multi-target generative attacks. For example, it increases the success rate from Inception-v3 to ResNet-152 by 34.58\%. Furthermore, our attack method shows substantially stronger robustness against defense mechanisms, such as adversarially trained models. The code of Dual-Flow is available at: $\href{https://github.com/Chyxx/Dual-Flow}{https://github.com/Chyxx/Dual-Flow}$.



## **49. Jailbreak Mimicry: Automated Discovery of Narrative-Based Jailbreaks for Large Language Models**

cs.CR

18 pages, 5 figures

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.22085v1) [paper-pdf](http://arxiv.org/pdf/2510.22085v1)

**Authors**: Pavlos Ntais

**Abstract**: Large language models (LLMs) remain vulnerable to sophisticated prompt engineering attacks that exploit contextual framing to bypass safety mechanisms, posing significant risks in cybersecurity applications. We introduce Jailbreak Mimicry, a systematic methodology for training compact attacker models to automatically generate narrative-based jailbreak prompts in a one-shot manner. Our approach transforms adversarial prompt discovery from manual craftsmanship into a reproducible scientific process, enabling proactive vulnerability assessment in AI-driven security systems. Developed for the OpenAI GPT-OSS-20B Red-Teaming Challenge, we use parameter-efficient fine-tuning (LoRA) on Mistral-7B with a curated dataset derived from AdvBench, achieving an 81.0% Attack Success Rate (ASR) against GPT-OSS-20B on a held-out test set of 200 items. Cross-model evaluation reveals significant variation in vulnerability patterns: our attacks achieve 66.5% ASR against GPT-4, 79.5% on Llama-3 and 33.0% against Gemini 2.5 Flash, demonstrating both broad applicability and model-specific defensive strengths in cybersecurity contexts. This represents a 54x improvement over direct prompting (1.5% ASR) and demonstrates systematic vulnerabilities in current safety alignment approaches. Our analysis reveals that technical domains (Cybersecurity: 93% ASR) and deception-based attacks (Fraud: 87.8% ASR) are particularly vulnerable, highlighting threats to AI-integrated threat detection, malware analysis, and secure systems, while physical harm categories show greater resistance (55.6% ASR). We employ automated harmfulness evaluation using Claude Sonnet 4, cross-validated with human expert assessment, ensuring reliable and scalable evaluation for cybersecurity red-teaming. Finally, we analyze failure mechanisms and discuss defensive strategies to mitigate these vulnerabilities in AI for cybersecurity.



## **50. Toward Understanding the Transferability of Adversarial Suffixes in Large Language Models**

cs.CL

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.22014v1) [paper-pdf](http://arxiv.org/pdf/2510.22014v1)

**Authors**: Sarah Ball, Niki Hasrati, Alexander Robey, Avi Schwarzschild, Frauke Kreuter, Zico Kolter, Andrej Risteski

**Abstract**: Discrete optimization-based jailbreaking attacks on large language models aim to generate short, nonsensical suffixes that, when appended onto input prompts, elicit disallowed content. Notably, these suffixes are often transferable -- succeeding on prompts and models for which they were never optimized. And yet, despite the fact that transferability is surprising and empirically well-established, the field lacks a rigorous analysis of when and why transfer occurs. To fill this gap, we identify three statistical properties that strongly correlate with transfer success across numerous experimental settings: (1) how much a prompt without a suffix activates a model's internal refusal direction, (2) how strongly a suffix induces a push away from this direction, and (3) how large these shifts are in directions orthogonal to refusal. On the other hand, we find that prompt semantic similarity only weakly correlates with transfer success. These findings lead to a more fine-grained understanding of transferability, which we use in interventional experiments to showcase how our statistical analysis can translate into practical improvements in attack success.



