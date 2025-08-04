# Latest Adversarial Attack Papers
**update at 2025-08-04 09:18:57**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. LeakyCLIP: Extracting Training Data from CLIP**

cs.CR

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00756v1) [paper-pdf](http://arxiv.org/pdf/2508.00756v1)

**Authors**: Yunhao Chen, Shujie Wang, Xin Wang, Xingjun Ma

**Abstract**: Understanding the memorization and privacy leakage risks in Contrastive Language--Image Pretraining (CLIP) is critical for ensuring the security of multimodal models. Recent studies have demonstrated the feasibility of extracting sensitive training examples from diffusion models, with conditional diffusion models exhibiting a stronger tendency to memorize and leak information. In this work, we investigate data memorization and extraction risks in CLIP through the lens of CLIP inversion, a process that aims to reconstruct training images from text prompts. To this end, we introduce \textbf{LeakyCLIP}, a novel attack framework designed to achieve high-quality, semantically accurate image reconstruction from CLIP embeddings. We identify three key challenges in CLIP inversion: 1) non-robust features, 2) limited visual semantics in text embeddings, and 3) low reconstruction fidelity. To address these challenges, LeakyCLIP employs 1) adversarial fine-tuning to enhance optimization smoothness, 2) linear transformation-based embedding alignment, and 3) Stable Diffusion-based refinement to improve fidelity. Empirical results demonstrate the superiority of LeakyCLIP, achieving over 358% improvement in Structural Similarity Index Measure (SSIM) for ViT-B-16 compared to baseline methods on LAION-2B subset. Furthermore, we uncover a pervasive leakage risk, showing that training data membership can even be successfully inferred from the metrics of low-fidelity reconstructions. Our work introduces a practical method for CLIP inversion while offering novel insights into the nature and scope of privacy risks in multimodal models.



## **2. Criticality-Based Dynamic Topology Optimization for Enhancing Aerial-Marine Swarm Resilience**

cs.NI

Submit to INFOCOM 2026

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00688v1) [paper-pdf](http://arxiv.org/pdf/2508.00688v1)

**Authors**: Ruiyang Huang, Haocheng Wang, Yixuan Shen, Ning Gao, Qiang Ni, Shi Jin, Yifan Wu

**Abstract**: Heterogeneous marine-aerial swarm networks encounter substantial difficulties due to targeted communication disruptions and structural weaknesses in adversarial environments. This paper proposes a two-step framework to strengthen the network's resilience. Specifically, our framework combines the node prioritization based on criticality with multi-objective topology optimization. First, we design a three-layer architecture to represent structural, communication, and task dependencies of the swarm networks. Then, we introduce the SurBi-Ranking method, which utilizes graph convolutional networks, to dynamically evaluate and rank the criticality of nodes and edges in real time. Next, we apply the NSGA-III algorithm to optimize the network topology, aiming to balance communication efficiency, global connectivity, and mission success rate. Experiments demonstrate that compared to traditional methods like K-Shell, our SurBi-Ranking method identifies critical nodes and edges with greater accuracy, as deliberate attacks on these components cause more significant connectivity degradation. Furthermore, our optimization approach, when prioritizing SurBi-Ranked critical components under attack, reduces the natural connectivity degradation by around 30%, achieves higher mission success rates, and incurs lower communication reconfiguration costs, ensuring sustained connectivity and mission effectiveness across multi-phase operations.



## **3. Revisiting Adversarial Patch Defenses on Object Detectors: Unified Evaluation, Large-Scale Dataset, and New Insights**

cs.CV

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00649v1) [paper-pdf](http://arxiv.org/pdf/2508.00649v1)

**Authors**: Junhao Zheng, Jiahao Sun, Chenhao Lin, Zhengyu Zhao, Chen Ma, Chong Zhang, Cong Wang, Qian Wang, Chao Shen

**Abstract**: Developing reliable defenses against patch attacks on object detectors has attracted increasing interest. However, we identify that existing defense evaluations lack a unified and comprehensive framework, resulting in inconsistent and incomplete assessments of current methods. To address this issue, we revisit 11 representative defenses and present the first patch defense benchmark, involving 2 attack goals, 13 patch attacks, 11 object detectors, and 4 diverse metrics. This leads to the large-scale adversarial patch dataset with 94 types of patches and 94,000 images. Our comprehensive analyses reveal new insights: (1) The difficulty in defending against naturalistic patches lies in the data distribution, rather than the commonly believed high frequencies. Our new dataset with diverse patch distributions can be used to improve existing defenses by 15.09% AP@0.5. (2) The average precision of the attacked object, rather than the commonly pursued patch detection accuracy, shows high consistency with defense performance. (3) Adaptive attacks can substantially bypass existing defenses, and defenses with complex/stochastic models or universal patch properties are relatively robust. We hope that our analyses will serve as guidance on properly evaluating patch attacks/defenses and advancing their design. Code and dataset are available at https://github.com/Gandolfczjh/APDE, where we will keep integrating new attacks/defenses.



## **4. SwarnRaft: Leveraging Consensus for Robust Drone Swarm Coordination in GNSS-Degraded Environments**

cs.DC

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00622v1) [paper-pdf](http://arxiv.org/pdf/2508.00622v1)

**Authors**: Kapel Dev, Yash Madhwal, Sofia Shevelo, Pavel Osinenko, Yury Yanovich

**Abstract**: Unmanned aerial vehicle (UAV) swarms are increasingly used in critical applications such as aerial mapping, environmental monitoring, and autonomous delivery. However, the reliability of these systems is highly dependent on uninterrupted access to the Global Navigation Satellite Systems (GNSS) signals, which can be disrupted in real-world scenarios due to interference, environmental conditions, or adversarial attacks, causing disorientation, collision risks, and mission failure. This paper proposes SwarnRaft, a blockchain-inspired positioning and consensus framework for maintaining coordination and data integrity in UAV swarms operating under GNSS-denied conditions. SwarnRaft leverages the Raft consensus algorithm to enable distributed drones (nodes) to agree on state updates such as location and heading, even in the absence of GNSS signals for one or more nodes. In our prototype, each node uses GNSS and local sensing, and communicates over WiFi in a simulated swarm. Upon signal loss, consensus is used to reconstruct or verify the position of the failed node based on its last known state and trajectory. Our system demonstrates robustness in maintaining swarm coherence and fault tolerance through a lightweight, scalable communication model. This work offers a practical and secure foundation for decentralized drone operation in unpredictable environments.



## **5. LeakSealer: A Semisupervised Defense for LLMs Against Prompt Injection and Leakage Attacks**

cs.CR

22 pages, preprint

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00602v1) [paper-pdf](http://arxiv.org/pdf/2508.00602v1)

**Authors**: Francesco Panebianco, Stefano Bonfanti, Francesco Trovò, Michele Carminati

**Abstract**: The generalization capabilities of Large Language Models (LLMs) have led to their widespread deployment across various applications. However, this increased adoption has introduced several security threats, notably in the forms of jailbreaking and data leakage attacks. Additionally, Retrieval Augmented Generation (RAG), while enhancing context-awareness in LLM responses, has inadvertently introduced vulnerabilities that can result in the leakage of sensitive information. Our contributions are twofold. First, we introduce a methodology to analyze historical interaction data from an LLM system, enabling the generation of usage maps categorized by topics (including adversarial interactions). This approach further provides forensic insights for tracking the evolution of jailbreaking attack patterns. Second, we propose LeakSealer, a model-agnostic framework that combines static analysis for forensic insights with dynamic defenses in a Human-In-The-Loop (HITL) pipeline. This technique identifies topic groups and detects anomalous patterns, allowing for proactive defense mechanisms. We empirically evaluate LeakSealer under two scenarios: (1) jailbreak attempts, employing a public benchmark dataset, and (2) PII leakage, supported by a curated dataset of labeled LLM interactions. In the static setting, LeakSealer achieves the highest precision and recall on the ToxicChat dataset when identifying prompt injection. In the dynamic setting, PII leakage detection achieves an AUPRC of $0.97$, significantly outperforming baselines such as Llama Guard.



## **6. ExclaveFL: Providing Transparency to Federated Learning using Exclaves**

cs.CR

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2412.10537v2) [paper-pdf](http://arxiv.org/pdf/2412.10537v2)

**Authors**: Jinnan Guo, Kapil Vaswani, Andrew Paverd, Peter Pietzuch

**Abstract**: In federated learning (FL), data providers jointly train a model without disclosing their training data. Despite its inherent privacy benefits, a malicious data provider can simply deviate from the correct training protocol without being detected, potentially compromising the trained model. While current solutions have explored the use of trusted execution environments (TEEs) to combat such attacks, they usually assume side-channel attacks against the TEEs are out of scope. However, such side-channel attacks can undermine the security properties of TEE-based FL frameworks, not by extracting the FL data, but by leaking keys that allow the adversary to impersonate as the TEE whilst deviating arbitrarily from the correct training protocol.   We describe ExclaveFL, an FL platform that provides end-to-end integrity and transparency, even in the presence of side-channel attacks on TEEs. We propose a new paradigm in which existing TEEs are used as exclaves -- integrity-protected execution environments that do not contain any secrets, making them immune to side-channel attacks. Whereas previous approaches attest the TEE itself and bind this attestation to a key held by the TEE, ExclaveFL attests individual data transformations at runtime. These runtime attestations form an attested dataflow graph, which can be checked to ensure the FL training job satisfies claims, such as deviations from the correct computation. We implement ExclaveFL by extending the popular NVFlare FL framework to use exclaves, and show experimentally that ExclaveFL introduces less than 10% overhead compared to the same FL framework without TEEs, whilst providing stronger security guarantees.



## **7. Wukong Framework for Not Safe For Work Detection in Text-to-Image systems**

cs.CV

Under review

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00591v1) [paper-pdf](http://arxiv.org/pdf/2508.00591v1)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long

**Abstract**: Text-to-Image (T2I) generation is a popular AI-generated content (AIGC) technology enabling diverse and creative image synthesis. However, some outputs may contain Not Safe For Work (NSFW) content (e.g., violence), violating community guidelines. Detecting NSFW content efficiently and accurately, known as external safeguarding, is essential. Existing external safeguards fall into two types: text filters, which analyze user prompts but overlook T2I model-specific variations and are prone to adversarial attacks; and image filters, which analyze final generated images but are computationally costly and introduce latency. Diffusion models, the foundation of modern T2I systems like Stable Diffusion, generate images through iterative denoising using a U-Net architecture with ResNet and Transformer blocks. We observe that: (1) early denoising steps define the semantic layout of the image, and (2) cross-attention layers in U-Net are crucial for aligning text and image regions. Based on these insights, we propose Wukong, a transformer-based NSFW detection framework that leverages intermediate outputs from early denoising steps and reuses U-Net's pre-trained cross-attention parameters. Wukong operates within the diffusion process, enabling early detection without waiting for full image generation. We also introduce a new dataset containing prompts, seeds, and image-specific NSFW labels, and evaluate Wukong on this and two public benchmarks. Results show that Wukong significantly outperforms text-based safeguards and achieves comparable accuracy of image filters, while offering much greater efficiency.



## **8. Activation-Guided Local Editing for Jailbreaking Attacks**

cs.CR

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00555v1) [paper-pdf](http://arxiv.org/pdf/2508.00555v1)

**Authors**: Jiecong Wang, Haoran Li, Hao Peng, Ziqian Zeng, Zihao Wang, Haohua Du, Zhengtao Yu

**Abstract**: Jailbreaking is an essential adversarial technique for red-teaming these models to uncover and patch security flaws. However, existing jailbreak methods face significant drawbacks. Token-level jailbreak attacks often produce incoherent or unreadable inputs and exhibit poor transferability, while prompt-level attacks lack scalability and rely heavily on manual effort and human ingenuity. We propose a concise and effective two-stage framework that combines the advantages of these approaches. The first stage performs a scenario-based generation of context and rephrases the original malicious query to obscure its harmful intent. The second stage then utilizes information from the model's hidden states to guide fine-grained edits, effectively steering the model's internal representation of the input from a malicious toward a benign one. Extensive experiments demonstrate that this method achieves state-of-the-art Attack Success Rate, with gains of up to 37.74% over the strongest baseline, and exhibits excellent transferability to black-box models. Our analysis further demonstrates that AGILE maintains substantial effectiveness against prominent defense mechanisms, highlighting the limitations of current safeguards and providing valuable insights for future defense development. Our code is available at https://github.com/yunsaijc/AGILE.



## **9. CyGATE: Game-Theoretic Cyber Attack-Defense Engine for Patch Strategy Optimization**

cs.CR

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00478v1) [paper-pdf](http://arxiv.org/pdf/2508.00478v1)

**Authors**: Yuning Jiang, Nay Oo, Qiaoran Meng, Lu Lin, Dusit Niyato, Zehui Xiong, Hoon Wei Lim, Biplab Sikdar

**Abstract**: Modern cyber attacks unfold through multiple stages, requiring defenders to dynamically prioritize mitigations under uncertainty. While game-theoretic models capture attacker-defender interactions, existing approaches often rely on static assumptions and lack integration with real-time threat intelligence, limiting their adaptability. This paper presents CyGATE, a game-theoretic framework modeling attacker-defender interactions, using large language models (LLMs) with retrieval-augmented generation (RAG) to enhance tactic selection and patch prioritization. Applied to a two-agent scenario, CyGATE frames cyber conflicts as a partially observable stochastic game (POSG) across Cyber Kill Chain stages. Both agents use belief states to navigate uncertainty, with the attacker adapting tactics and the defender re-prioritizing patches based on evolving risks and observed adversary behavior. The framework's flexible architecture enables extension to multi-agent scenarios involving coordinated attackers, collaborative defenders, or complex enterprise environments with multiple stakeholders. Evaluated in a dynamic patch scheduling scenario, CyGATE effectively prioritizes high-risk vulnerabilities, enhancing adaptability through dynamic threat integration, strategic foresight by anticipating attacker moves under uncertainty, and efficiency by optimizing resource use.



## **10. System Identification from Partial Observations under Adversarial Attacks**

math.OC

8 pages, 3 figures

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2504.00244v2) [paper-pdf](http://arxiv.org/pdf/2504.00244v2)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper is concerned with the partially observed linear system identification, where the goal is to obtain reasonably accurate estimation of the balanced truncation of the true system up to order $k$ from output measurements. We consider the challenging case of system identification under adversarial attacks, where the probability of having an attack at each time is $\Theta(1/k)$ while the value of the attack is arbitrary. We first show that the $\ell_1$-norm estimator exactly identifies the true Markov parameter matrix for nilpotent systems under any type of attack. We then build on this result to extend it to general systems and show that the estimation error exponentially decays as $k$ grows. The estimated balanced truncation model accordingly shows an exponentially decaying error for the identification of the true system up to a similarity transformation. This work is the first to provide the input-output analysis of the system with partial observations under arbitrary attacks.



## **11. Quantum Key-Recovery Attacks on FBC Algorithm**

quant-ph

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00448v1) [paper-pdf](http://arxiv.org/pdf/2508.00448v1)

**Authors**: Yan-Ying Zhu, Bin-Bin Cai, Fei Gao, Song Lin

**Abstract**: With the advancement of quantum computing, symmetric cryptography faces new challenges from quantum attacks. These attacks are typically classified into two models: Q1 (classical queries) and Q2 (quantum superposition queries). In this context, we present a comprehensive security analysis of the FBC algorithm considering quantum adversaries with different query capabilities. In the Q2 model, we first design 4-round polynomial-time quantum distinguishers for FBC-F and FBC-KF structures, and then perform $r(r>6)$-round quantum key-recovery attacks. Our attacks require $O(2^{(2n(r-6)+3n)/2})$ quantum queries, reducing the time complexity by a factor of $2^{4.5n}$ compared with quantum brute-force search, where $n$ denotes the subkey length. Moreover, we give a new 6-round polynomial-time quantum distinguisher for FBC-FK structure. Based on this, we construct an $r(r>6)$-round quantum key-recovery attack with complexity $O(2^{n(r-6)})$. Considering an adversary with classical queries and quantum computing capabilities, we demonstrate low-data quantum key-recovery attacks on FBC-KF/FK structures in the Q1 model. These attacks require only a constant number of plaintext-ciphertext pairs, then use the Grover algorithm to search the intermediate states, thereby recovering all keys in $O(2^{n/2})$ time.



## **12. Adaptive Branch Specialization in Spectral-Spatial Graph Neural Networks for Certified Robustness**

cs.LG

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2505.08320v3) [paper-pdf](http://arxiv.org/pdf/2505.08320v3)

**Authors**: Yoonhyuk Choi, Jiho Choi, Chong-Kwon Kim

**Abstract**: Recent Graph Neural Networks (GNNs) combine spectral-spatial architectures for enhanced representation learning. However, limited attention has been paid to certified robustness, particularly regarding training strategies and underlying rationale. In this paper, we explicitly specialize each branch: the spectral network is trained to withstand l0 edge flips and capture homophilic structures, while the spatial part is designed to resist linf feature perturbations and heterophilic patterns. A context-aware gating network adaptively fuses the two representations, dynamically routing each node's prediction to the more reliable branch. This specialized adversarial training scheme uses branch-specific inner maximization (structure vs feature attacks) and a unified alignment objective. We provide theoretical guarantees: (i) expressivity of the gating mechanism beyond 1-WL, (ii) spectral-spatial frequency bias, and (iii) certified robustness with trade-off. Empirically, SpecSphere attains state-of-the-art node classification accuracy and offers tighter certified robustness on real-world benchmarks.



## **13. Preliminary Investigation into Uncertainty-Aware Attack Stage Classification**

cs.CR

Proceedings for SPAIML2025 workshop, 26/10/2025 Bologna Italy,  co-located with ECAI2025

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00368v1) [paper-pdf](http://arxiv.org/pdf/2508.00368v1)

**Authors**: Alessandro Gaudenzi, Lorenzo Nodari, Lance Kaplan, Alessandra Russo, Murat Sensoy, Federico Cerutti

**Abstract**: Advanced Persistent Threats (APTs) represent a significant challenge in cybersecurity due to their prolonged, multi-stage nature and the sophistication of their operators. Traditional detection systems typically focus on identifying malicious activity in binary terms (benign or malicious) without accounting for the progression of an attack. However, effective response strategies depend on accurate inference of the attack's current stage, as countermeasures must be tailored to whether an adversary is in the early reconnaissance phase or actively conducting exploitation or exfiltration. This work addresses the problem of attack stage inference under uncertainty, with a focus on robustness to out-of-distribution (OOD) inputs. We propose a classification approach based on Evidential Deep Learning (EDL), which models predictive uncertainty by outputting parameters of a Dirichlet distribution over possible stages. This allows the system not only to predict the most likely stage of an attack but also to indicate when it is uncertain or the input lies outside the training distribution. Preliminary experiments in a simulated environment demonstrate that the proposed model can accurately infer the stage of an attack with calibrated confidence while effectively detecting OOD inputs, which may indicate changes in the attackers' tactics. These results support the feasibility of deploying uncertainty-aware models for staged threat detection in dynamic and adversarial environments.



## **14. Boosting Adversarial Transferability with Low-Cost Optimization via Maximin Expected Flatness**

cs.CV

The original NCS method has been revised and renamed as MEF. A  theoretical proof of the relationship between flatness and transferability is  added

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2405.16181v2) [paper-pdf](http://arxiv.org/pdf/2405.16181v2)

**Authors**: Chunlin Qiu, Ang Li, Yiheng Duan, Shenyi Zhang, Yuanjie Zhang, Lingchen Zhao, Qian Wang

**Abstract**: Transfer-based attacks craft adversarial examples on white-box surrogate models and directly deploy them against black-box target models, offering model-agnostic and query-free threat scenarios. While flatness-enhanced methods have recently emerged to improve transferability by enhancing the loss surface flatness of adversarial examples, their divergent flatness definitions and heuristic attack designs suffer from unexamined optimization limitations and missing theoretical foundation, thus constraining their effectiveness and efficiency. This work exposes the severely imbalanced exploitation-exploration dynamics in flatness optimization, establishing the first theoretical foundation for flatness-based transferability and proposing a principled framework to overcome these optimization pitfalls. Specifically, we systematically unify fragmented flatness definitions across existing methods, revealing their imbalanced optimization limitations in over-exploration of sensitivity peaks or over-exploitation of local plateaus. To resolve these issues, we rigorously formalize average-case flatness and transferability gaps, proving that enhancing zeroth-order average-case flatness minimizes cross-model discrepancies. Building on this theory, we design a Maximin Expected Flatness (MEF) attack that enhances zeroth-order average-case flatness while balancing flatness exploration and exploitation. Extensive evaluations across 22 models and 24 current transfer-based attacks demonstrate MEF's superiority: it surpasses the state-of-the-art PGN attack by 4% in attack success rate at half the computational cost and achieves 8% higher success rate under the same budget. When combined with input augmentation, MEF attains 15% additional gains against defense-equipped models, establishing new robustness benchmarks. Our code is available at https://github.com/SignedQiu/MEFAttack.



## **15. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

cs.RO

ICCV camera ready; Github:  https://github.com/William-wAng618/roboticAttack Homepage:  https://vlaattacker.github.io/

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2411.13587v4) [paper-pdf](http://arxiv.org/pdf/2411.13587v4)

**Authors**: Taowen Wang, Cheng Han, James Chenhao Liang, Wenhao Yang, Dongfang Liu, Luna Xinyu Zhang, Qifan Wang, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. Despite their significant capabilities, VLA models introduce new attack surfaces. This paper systematically evaluates their robustness. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce two untargeted attack objectives that leverage spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, we advance both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for continuously developing robust defense strategies prior to physical-world deployments.



## **16. LLMs Encode Harmfulness and Refusal Separately**

cs.CL

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.11878v2) [paper-pdf](http://arxiv.org/pdf/2507.11878v2)

**Authors**: Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, Weiyan Shi

**Abstract**: LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety



## **17. DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing**

cs.CV

Accepted to ICCV 2025

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2504.17894v2) [paper-pdf](http://arxiv.org/pdf/2504.17894v2)

**Authors**: Aniruddha Bala, Rohit Chowdhury, Rohan Jaiswal, Siddharth Roheda

**Abstract**: Advancements in diffusion models have enabled effortless image editing via text prompts, raising concerns about image security. Attackers with access to user images can exploit these tools for malicious edits. Recent defenses attempt to protect images by adding a limited noise in the pixel space to disrupt the functioning of diffusion-based editing models. However, the adversarial noise added by previous methods is easily noticeable to the human eye. Moreover, most of these methods are not robust to purification techniques like JPEG compression under a feasible pixel budget. We propose a novel optimization approach that introduces adversarial perturbations directly in the frequency domain by modifying the Discrete Cosine Transform (DCT) coefficients of the input image. By leveraging the JPEG pipeline, our method generates adversarial images that effectively prevent malicious image editing. Extensive experiments across a variety of tasks and datasets demonstrate that our approach introduces fewer visual artifacts while maintaining similar levels of edit protection and robustness to noise purification techniques.



## **18. Beyond Optimal Fault Tolerance**

cs.DC

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2501.06044v7) [paper-pdf](http://arxiv.org/pdf/2501.06044v7)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal "recoverable fault-tolerance" achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic "recovery procedure" that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.



## **19. Graph Representation-based Model Poisoning on Federated Large Language Models**

cs.CR

7 pages, 5 figures (Submitted to IEEE Communication Magazine)

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.01694v2) [paper-pdf](http://arxiv.org/pdf/2507.01694v2)

**Authors**: Hanlin Cai, Haofan Dong, Houtianfu Wang, Kai Li, Ozgur B. Akan

**Abstract**: Federated large language models (FedLLMs) enable powerful generative capabilities within wireless networks while preserving data privacy. Nonetheless, FedLLMs remain vulnerable to model poisoning attacks. This article first reviews recent advancements in model poisoning techniques and existing defense mechanisms for FedLLMs, underscoring critical limitations, especially when dealing with non-IID textual data distributions. Current defense strategies predominantly employ distance or similarity-based outlier detection mechanisms, relying on the assumption that malicious updates markedly differ from benign statistical patterns. However, this assumption becomes inadequate against adaptive adversaries targeting billion-parameter LLMs. The article further investigates graph representation-based model poisoning (GRMP), an emerging attack paradigm that exploits higher-order correlations among benign client gradients to craft malicious updates indistinguishable from legitimate ones. GRMP can effectively circumvent advanced defense systems, causing substantial degradation in model accuracy and overall performance. Moreover, the article outlines a forward-looking research roadmap that emphasizes the necessity of graph-aware secure aggregation methods, specialized vulnerability metrics tailored for FedLLMs, and evaluation frameworks to enhance the robustness of federated language model deployments.



## **20. Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application**

cs.CR

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2503.06989v2) [paper-pdf](http://arxiv.org/pdf/2503.06989v2)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Zonghao Ying, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Quanchen Zou

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.



## **21. Scalable and Precise Patch Robustness Certification for Deep Learning Models with Top-k Predictions**

cs.LG

accepted by QRS 2025

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.23335v1) [paper-pdf](http://arxiv.org/pdf/2507.23335v1)

**Authors**: Qilin Zhou, Haipeng Wang, Zhengyuan Wei, W. K. Chan

**Abstract**: Patch robustness certification is an emerging verification approach for defending against adversarial patch attacks with provable guarantees for deep learning systems. Certified recovery techniques guarantee the prediction of the sole true label of a certified sample. However, existing techniques, if applicable to top-k predictions, commonly conduct pairwise comparisons on those votes between labels, failing to certify the sole true label within the top k prediction labels precisely due to the inflation on the number of votes controlled by the attacker (i.e., attack budget); yet enumerating all combinations of vote allocation suffers from the combinatorial explosion problem. We propose CostCert, a novel, scalable, and precise voting-based certified recovery defender. CostCert verifies the true label of a sample within the top k predictions without pairwise comparisons and combinatorial explosion through a novel design: whether the attack budget on the sample is infeasible to cover the smallest total additional votes on top of the votes uncontrollable by the attacker to exclude the true labels from the top k prediction labels. Experiments show that CostCert significantly outperforms the current state-of-the-art defender PatchGuard, such as retaining up to 57.3% in certified accuracy when the patch size is 96, whereas PatchGuard has already dropped to zero.



## **22. Fine-Grained Privacy Extraction from Retrieval-Augmented Generation Systems via Knowledge Asymmetry Exploitation**

cs.CR

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.23229v1) [paper-pdf](http://arxiv.org/pdf/2507.23229v1)

**Authors**: Yufei Chen, Yao Wang, Haibin Zhang, Tao Gu

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge bases, but this advancement introduces significant privacy risks. Existing privacy attacks on RAG systems can trigger data leakage but often fail to accurately isolate knowledge-base-derived sentences within mixed responses. They also lack robustness when applied across multiple domains. This paper addresses these challenges by presenting a novel black-box attack framework that exploits knowledge asymmetry between RAG and standard LLMs to achieve fine-grained privacy extraction across heterogeneous knowledge landscapes. We propose a chain-of-thought reasoning strategy that creates adaptive prompts to steer RAG systems away from sensitive content. Specifically, we first decompose adversarial queries to maximize information disparity and then apply a semantic relationship scoring to resolve lexical and syntactic ambiguities. We finally train a neural network on these feature scores to precisely identify sentences containing private information. Unlike prior work, our framework generalizes to unseen domains through iterative refinement without pre-defined knowledge. Experimental results show that we achieve over 91% privacy extraction rate in single-domain and 83% in multi-domain scenarios, reducing sensitive sentence exposure by over 65% in case studies. This work bridges the gap between attack and defense in RAG systems, enabling precise extraction of private information while providing a foundation for adaptive mitigation.



## **23. Adversarial-Guided Diffusion for Multimodal LLM Attacks**

cs.CV

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.23202v1) [paper-pdf](http://arxiv.org/pdf/2507.23202v1)

**Authors**: Chengwei Xia, Fan Ma, Ruijie Quan, Kun Zhan, Yi Yang

**Abstract**: This paper addresses the challenge of generating adversarial image using a diffusion model to deceive multimodal large language models (MLLMs) into generating the targeted responses, while avoiding significant distortion of the clean image. To address the above challenges, we propose an adversarial-guided diffusion (AGD) approach for adversarial attack MLLMs. We introduce adversarial-guided noise to ensure attack efficacy. A key observation in our design is that, unlike most traditional adversarial attacks which embed high-frequency perturbations directly into the clean image, AGD injects target semantics into the noise component of the reverse diffusion. Since the added noise in a diffusion model spans the entire frequency spectrum, the adversarial signal embedded within it also inherits this full-spectrum property. Importantly, during reverse diffusion, the adversarial image is formed as a linear combination of the clean image and the noise. Thus, when applying defenses such as a simple low-pass filtering, which act independently on each component, the adversarial image within the noise component is less likely to be suppressed, as it is not confined to the high-frequency band. This makes AGD inherently robust to variety defenses. Extensive experiments demonstrate that our AGD outperforms state-of-the-art methods in attack performance as well as in model robustness to some defenses.



## **24. AUV-Fusion: Cross-Modal Adversarial Fusion of User Interactions and Visual Perturbations Against VARS**

cs.IR

14 pages,6 figures

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22880v1) [paper-pdf](http://arxiv.org/pdf/2507.22880v1)

**Authors**: Hai Ling, Tianchi Wang, Xiaohao Liu, Zhulin Tao, Lifang Yang, Xianglin Huang

**Abstract**: Modern Visual-Aware Recommender Systems (VARS) exploit the integration of user interaction data and visual features to deliver personalized recommendations with high precision. However, their robustness against adversarial attacks remains largely underexplored, posing significant risks to system reliability and security. Existing attack strategies suffer from notable limitations: shilling attacks are costly and detectable, and visual-only perturbations often fail to align with user preferences. To address these challenges, we propose AUV-Fusion, a cross-modal adversarial attack framework that adopts high-order user preference modeling and cross-modal adversary generation. Specifically, we obtain robust user embeddings through multi-hop user-item interactions and transform them via an MLP into semantically aligned perturbations. These perturbations are injected onto the latent space of a pre-trained VAE within the diffusion model. By synergistically integrating genuine user interaction data with visually plausible perturbations, AUV-Fusion eliminates the need for injecting fake user profiles and effectively mitigates the challenge of insufficient user preference extraction inherent in traditional visual-only attacks. Comprehensive evaluations on diverse VARS architectures and real-world datasets demonstrate that AUV-Fusion significantly enhances the exposure of target (cold-start) items compared to conventional baseline methods. Moreover, AUV-Fusion maintains exceptional stealth under rigorous scrutiny.



## **25. Curvature Dynamic Black-box Attack: revisiting adversarial robustness via dynamic curvature estimation**

cs.LG

This article contains several flaws

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2505.19194v2) [paper-pdf](http://arxiv.org/pdf/2505.19194v2)

**Authors**: Peiran Sun

**Abstract**: Adversarial attack reveals the vulnerability of deep learning models. For about a decade, countless attack and defense methods have been proposed, leading to robustified classifiers and better understanding of models. Among these methods, curvature-based approaches have attracted attention because it is assumed that high curvature may give rise to rough decision boundary. However, the most commonly used \textit{curvature} is the curvature of loss function, scores or other parameters from within the model as opposed to decision boundary curvature, since the former can be relatively easily formed using second order derivative. In this paper, we propose a new query-efficient method, dynamic curvature estimation(DCE), to estimate the decision boundary curvature in a black-box setting. Our approach is based on CGBA, a black-box adversarial attack. By performing DCE on a wide range of classifiers, we discovered, statistically, a connection between decision boundary curvature and adversarial robustness. We also propose a new attack method, curvature dynamic black-box attack(CDBA) with improved performance using the dynamically estimated curvature.



## **26. DISTIL: Data-Free Inversion of Suspicious Trojan Inputs via Latent Diffusion**

cs.CV

ICCV 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22813v1) [paper-pdf](http://arxiv.org/pdf/2507.22813v1)

**Authors**: Hossein Mirzaei, Zeinab Taghavi, Sepehr Rezaee, Masoud Hadi, Moein Madadi, Mackenzie W. Mathis

**Abstract**: Deep neural networks have demonstrated remarkable success across numerous tasks, yet they remain vulnerable to Trojan (backdoor) attacks, raising serious concerns about their safety in real-world mission-critical applications. A common countermeasure is trigger inversion -- reconstructing malicious "shortcut" patterns (triggers) inserted by an adversary during training. Current trigger-inversion methods typically search the full pixel space under specific assumptions but offer no assurances that the estimated trigger is more than an adversarial perturbation that flips the model output. Here, we propose a data-free, zero-shot trigger-inversion strategy that restricts the search space while avoiding strong assumptions on trigger appearance. Specifically, we incorporate a diffusion-based generator guided by the target classifier; through iterative generation, we produce candidate triggers that align with the internal representations the model relies on for malicious behavior. Empirical evaluations, both quantitative and qualitative, show that our approach reconstructs triggers that effectively distinguish clean versus Trojaned models. DISTIL surpasses alternative methods by high margins, achieving up to 7.1% higher accuracy on the BackdoorBench dataset and a 9.4% improvement on trojaned object detection model scanning, offering a promising new direction for reliable backdoor defense without reliance on extensive data or strong prior assumptions about triggers. The code is available at https://github.com/AdaptiveMotorControlLab/DISTIL.



## **27. Cryptanalysis of LC-MUME: A Lightweight Certificateless Multi-User Matchmaking Encryption for Mobile Devices**

cs.CR

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22674v1) [paper-pdf](http://arxiv.org/pdf/2507.22674v1)

**Authors**: Ramprasad Sarkar

**Abstract**: Yang et al. proposed a lightweight certificateless multiuser matchmaking encryption (LC-MUME) scheme for mobile devices, published in IEEE Transactions on Information Forensics and Security (TIFS) (DOI: 10.1109/TIFS.2023.3321961). Their construction aims to reduce computational and communication overhead within a one-to-many certificateless cryptographic framework. The authors claim that their scheme satisfies existential unforgeability under chosen-message attacks (EUF-CMA) in the random oracle model. However, our cryptanalytic study demonstrates that the scheme fails to meet this critical security requirement. In particular, we show that a Type-I adversary can successfully forge a valid ciphertext without possessing the complete private key of the sender. Both theoretical analysis and practical implementation confirm that this attack can be mounted with minimal computational cost. To address these weaknesses, we propose a modification strategy to strengthen the security of matchmaking encryption schemes in mobile computing environments.



## **28. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

cs.AI

Accepted at VecDB @ ICML 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2504.04858v3) [paper-pdf](http://arxiv.org/pdf/2504.04858v3)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.



## **29. Diffusion-based Adversarial Identity Manipulation for Facial Privacy Protection**

cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2504.21646v3) [paper-pdf](http://arxiv.org/pdf/2504.21646v3)

**Authors**: Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo

**Abstract**: The success of face recognition (FR) systems has led to serious privacy concerns due to potential unauthorized surveillance and user tracking on social networks. Existing methods for enhancing privacy fail to generate natural face images that can protect facial privacy. In this paper, we propose diffusion-based adversarial identity manipulation (DiffAIM) to generate natural and highly transferable adversarial faces against malicious FR systems. To be specific, we manipulate facial identity within the low-dimensional latent space of a diffusion model. This involves iteratively injecting gradient-based adversarial identity guidance during the reverse diffusion process, progressively steering the generation toward the desired adversarial faces. The guidance is optimized for identity convergence towards a target while promoting semantic divergence from the source, facilitating effective impersonation while maintaining visual naturalness. We further incorporate structure-preserving regularization to preserve facial structure consistency during manipulation. Extensive experiments on both face verification and identification tasks demonstrate that compared with the state-of-the-art, DiffAIM achieves stronger black-box attack transferability while maintaining superior visual quality. We also demonstrate the effectiveness of the proposed approach for commercial FR APIs, including Face++ and Aliyun.



## **30. Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs**

cs.CL

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22564v1) [paper-pdf](http://arxiv.org/pdf/2507.22564v1)

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.



## **31. Ownership Verification of DNN Models Using White-Box Adversarial Attacks with Specified Probability Manipulation**

cs.LG

Accepted to EUSIPCO 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2505.17579v3) [paper-pdf](http://arxiv.org/pdf/2505.17579v3)

**Authors**: Teruki Sano, Minoru Kuribayashi, Masao Sakai, Shuji Isobe, Eisuke Koizumi

**Abstract**: In this paper, we propose a novel framework for ownership verification of deep neural network (DNN) models for image classification tasks. It allows verification of model identity by both the rightful owner and third party without presenting the original model. We assume a gray-box scenario where an unauthorized user owns a model that is illegally copied from the original model, provides services in a cloud environment, and the user throws images and receives the classification results as a probability distribution of output classes. The framework applies a white-box adversarial attack to align the output probability of a specific class to a designated value. Due to the knowledge of original model, it enables the owner to generate such adversarial examples. We propose a simple but effective adversarial attack method based on the iterative Fast Gradient Sign Method (FGSM) by introducing control parameters. Experimental results confirm the effectiveness of the identification of DNN models using adversarial attack.



## **32. RCR-AF: Enhancing Model Generalization via Rademacher Complexity Reduction Activation Function**

cs.LG

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22446v1) [paper-pdf](http://arxiv.org/pdf/2507.22446v1)

**Authors**: Yunrui Yu, Kafeng Wang, Hang Su, Jun Zhu

**Abstract**: Despite their widespread success, deep neural networks remain critically vulnerable to adversarial attacks, posing significant risks in safety-sensitive applications. This paper investigates activation functions as a crucial yet underexplored component for enhancing model robustness. We propose a Rademacher Complexity Reduction Activation Function (RCR-AF), a novel activation function designed to improve both generalization and adversarial resilience. RCR-AF uniquely combines the advantages of GELU (including smoothness, gradient stability, and negative information retention) with ReLU's desirable monotonicity, while simultaneously controlling both model sparsity and capacity through built-in clipping mechanisms governed by two hyperparameters, $\alpha$ and $\gamma$. Our theoretical analysis, grounded in Rademacher complexity, demonstrates that these parameters directly modulate the model's Rademacher complexity, offering a principled approach to enhance robustness. Comprehensive empirical evaluations show that RCR-AF consistently outperforms widely-used alternatives (ReLU, GELU, and Swish) in both clean accuracy under standard training and in adversarial robustness within adversarial training paradigms.



## **33. Theoretical Analysis of Relative Errors in Gradient Computations for Adversarial Attacks with CE Loss**

cs.LG

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22428v1) [paper-pdf](http://arxiv.org/pdf/2507.22428v1)

**Authors**: Yunrui Yu, Hang Su, Cheng-zhong Xu, Zhizhong Su, Jun Zhu

**Abstract**: Gradient-based adversarial attacks using the Cross-Entropy (CE) loss often suffer from overestimation due to relative errors in gradient computation induced by floating-point arithmetic. This paper provides a rigorous theoretical analysis of these errors, conducting the first comprehensive study of floating-point computation errors in gradient-based attacks across four distinct scenarios: (i) unsuccessful untargeted attacks, (ii) successful untargeted attacks, (iii) unsuccessful targeted attacks, and (iv) successful targeted attacks. We establish theoretical foundations characterizing the behavior of relative numerical errors under different attack conditions, revealing previously unknown patterns in gradient computation instability, and identify floating-point underflow and rounding as key contributors. Building on this insight, we propose the Theoretical MIFPE (T-MIFPE) loss function, which incorporates an optimal scaling factor $T = t^*$ to minimize the impact of floating-point errors, thereby enhancing the accuracy of gradient computation in adversarial attacks. Extensive experiments on the MNIST, CIFAR-10, and CIFAR-100 datasets demonstrate that T-MIFPE outperforms existing loss functions, including CE, C\&W, DLR, and MIFPE, in terms of attack potency and robustness evaluation accuracy.



## **34. Benchmarking Fraud Detectors on Private Graph Data**

cs.CR

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22347v1) [paper-pdf](http://arxiv.org/pdf/2507.22347v1)

**Authors**: Alexander Goldberg, Giulia Fanti, Nihar Shah, Zhiwei Steven Wu

**Abstract**: We introduce the novel problem of benchmarking fraud detectors on private graph-structured data. Currently, many types of fraud are managed in part by automated detection algorithms that operate over graphs. We consider the scenario where a data holder wishes to outsource development of fraud detectors to third parties (e.g., vendors or researchers). The third parties submit their fraud detectors to the data holder, who evaluates these algorithms on a private dataset and then publicly communicates the results. We propose a realistic privacy attack on this system that allows an adversary to de-anonymize individuals' data based only on the evaluation results. In simulations of a privacy-sensitive benchmark for facial recognition algorithms by the National Institute of Standards and Technology (NIST), our attack achieves near perfect accuracy in identifying whether individuals' data is present in a private dataset, with a True Positive Rate of 0.98 at a False Positive Rate of 0.00. We then study how to benchmark algorithms while satisfying a formal differential privacy (DP) guarantee. We empirically evaluate two classes of solutions: subsample-and-aggregate and DP synthetic graph data. We demonstrate through extensive experiments that current approaches do not provide utility when guaranteeing DP. Our results indicate that the error arising from DP trades off between bias from distorting graph structure and variance from adding random noise. Current methods lie on different points along this bias-variance trade-off, but more complex methods tend to require high-variance noise addition, undermining utility.



## **35. Resilient State Recovery using Prior Measurement Support Information**

math.OC

To be published in SIAM Journal on Control and Optimization

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22340v1) [paper-pdf](http://arxiv.org/pdf/2507.22340v1)

**Authors**: Yu Zheng, Olugbenga Moses Anubi, Warren E. Dixon

**Abstract**: Resilient state recovery of cyber-physical systems has attracted much research attention due to the unique challenges posed by the tight coupling between communication, computation, and the underlying physics of such systems. By modeling attacks as additive adversary signals to a sparse subset of measurements, this resilient recovery problem can be formulated as an error correction problem. To achieve exact state recovery, most existing results require less than $50\%$ of the measurement nodes to be compromised, which limits the resiliency of the estimators. In this paper, we show that observer resiliency can be further improved by incorporating data-driven prior information. We provide an analytical bridge between the precision of prior information and the resiliency of the estimator. By quantifying the relationship between the estimation error of the weighted $\ell_1$ observer and the precision of the support prior. This quantified relationship provides guidance for the estimator's weight design to achieve optimal resiliency. Several numerical simulations and an application case study are presented to validate the theoretical claims.



## **36. Can adversarial attacks by large language models be attributed?**

cs.AI

22 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2411.08003v3) [paper-pdf](http://arxiv.org/pdf/2411.08003v3)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.



## **37. Persistent Backdoor Attacks in Continual Learning**

cs.LG

19 pages, 20 figures, 6 tables

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2409.13864v3) [paper-pdf](http://arxiv.org/pdf/2409.13864v3)

**Authors**: Zhen Guo, Abhinav Kumar, Reza Tourani

**Abstract**: Backdoor attacks pose a significant threat to neural networks, enabling adversaries to manipulate model outputs on specific inputs, often with devastating consequences, especially in critical applications. While backdoor attacks have been studied in various contexts, little attention has been given to their practicality and persistence in continual learning, particularly in understanding how the continual updates to model parameters, as new data distributions are learned and integrated, impact the effectiveness of these attacks over time. To address this gap, we introduce two persistent backdoor attacks-Blind Task Backdoor and Latent Task Backdoor-each leveraging minimal adversarial influence. Our blind task backdoor subtly alters the loss computation without direct control over the training process, while the latent task backdoor influences only a single task's training, with all other tasks trained benignly. We evaluate these attacks under various configurations, demonstrating their efficacy with static, dynamic, physical, and semantic triggers. Our results show that both attacks consistently achieve high success rates across different continual learning algorithms, while effectively evading state-of-the-art defenses, such as SentiNet and I-BAU.



## **38. Teach Me to Trick: Exploring Adversarial Transferability via Knowledge Distillation**

cs.LG

10 pages, 4 figures

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21992v1) [paper-pdf](http://arxiv.org/pdf/2507.21992v1)

**Authors**: Siddhartha Pradhan, Shikshya Shiwakoti, Neha Bathuri

**Abstract**: We investigate whether knowledge distillation (KD) from multiple heterogeneous teacher models can enhance the generation of transferable adversarial examples. A lightweight student model is trained using two KD strategies: curriculum-based switching and joint optimization, with ResNet50 and DenseNet-161 as teachers. The trained student is then used to generate adversarial examples using FG, FGS, and PGD attacks, which are evaluated against a black-box target model (GoogLeNet). Our results show that student models distilled from multiple teachers achieve attack success rates comparable to ensemble-based baselines, while reducing adversarial example generation time by up to a factor of six. An ablation study further reveals that lower temperature settings and the inclusion of hard-label supervision significantly enhance transferability. These findings suggest that KD can serve not only as a model compression technique but also as a powerful tool for improving the efficiency and effectiveness of black-box adversarial attacks.



## **39. ZIUM: Zero-Shot Intent-Aware Adversarial Attack on Unlearned Models**

cs.CV

Accepted to ICCV2025

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21985v1) [paper-pdf](http://arxiv.org/pdf/2507.21985v1)

**Authors**: Hyun Jun Yook, Ga San Jhun, Jae Hyun Cho, Min Jeon, Donghyun Kim, Tae Hyung Kim, Youn Kyu Lee

**Abstract**: Machine unlearning (MU) removes specific data points or concepts from deep learning models to enhance privacy and prevent sensitive content generation. Adversarial prompts can exploit unlearned models to generate content containing removed concepts, posing a significant security risk. However, existing adversarial attack methods still face challenges in generating content that aligns with an attacker's intent while incurring high computational costs to identify successful prompts. To address these challenges, we propose ZIUM, a Zero-shot Intent-aware adversarial attack on Unlearned Models, which enables the flexible customization of target attack images to reflect an attacker's intent. Additionally, ZIUM supports zero-shot adversarial attacks without requiring further optimization for previously attacked unlearned concepts. The evaluation across various MU scenarios demonstrated ZIUM's effectiveness in successfully customizing content based on user-intent prompts while achieving a superior attack success rate compared to existing methods. Moreover, its zero-shot adversarial attack significantly reduces the attack time for previously attacked unlearned concepts.



## **40. Anyone Can Jailbreak: Prompt-Based Attacks on LLMs and T2Is**

cs.CV

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21820v1) [paper-pdf](http://arxiv.org/pdf/2507.21820v1)

**Authors**: Ahmed B Mustafa, Zihan Ye, Yang Lu, Michael P Pound, Shreyank N Gowda

**Abstract**: Despite significant advancements in alignment and content moderation, large language models (LLMs) and text-to-image (T2I) systems remain vulnerable to prompt-based attacks known as jailbreaks. Unlike traditional adversarial examples requiring expert knowledge, many of today's jailbreaks are low-effort, high-impact crafted by everyday users with nothing more than cleverly worded prompts. This paper presents a systems-style investigation into how non-experts reliably circumvent safety mechanisms through techniques such as multi-turn narrative escalation, lexical camouflage, implication chaining, fictional impersonation, and subtle semantic edits. We propose a unified taxonomy of prompt-level jailbreak strategies spanning both text-output and T2I models, grounded in empirical case studies across popular APIs. Our analysis reveals that every stage of the moderation pipeline, from input filtering to output validation, can be bypassed with accessible strategies. We conclude by highlighting the urgent need for context-aware defenses that reflect the ease with which these jailbreaks can be reproduced in real-world settings.



## **41. Adversarial Defence without Adversarial Defence: Enhancing Language Model Robustness via Instance-level Principal Component Removal**

cs.CL

This paper was accepted with an A-decision to Transactions of the  Association for Computational Linguistics. This version is the  pre-publication version prior to MIT Press production

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21750v1) [paper-pdf](http://arxiv.org/pdf/2507.21750v1)

**Authors**: Yang Wang, Chenghao Xiao, Yizhi Li, Stuart E. Middleton, Noura Al Moubayed, Chenghua Lin

**Abstract**: Pre-trained language models (PLMs) have driven substantial progress in natural language processing but remain vulnerable to adversarial attacks, raising concerns about their robustness in real-world applications. Previous studies have sought to mitigate the impact of adversarial attacks by introducing adversarial perturbations into the training process, either implicitly or explicitly. While both strategies enhance robustness, they often incur high computational costs. In this work, we propose a simple yet effective add-on module that enhances the adversarial robustness of PLMs by removing instance-level principal components, without relying on conventional adversarial defences or perturbing the original training data. Our approach transforms the embedding space to approximate Gaussian properties, thereby reducing its susceptibility to adversarial perturbations while preserving semantic relationships. This transformation aligns embedding distributions in a way that minimises the impact of adversarial noise on decision boundaries, enhancing robustness without requiring adversarial examples or costly training-time augmentation. Evaluations on eight benchmark datasets show that our approach improves adversarial robustness while maintaining comparable before-attack accuracy to baselines, achieving a balanced trade-off between robustness and generalisation.



## **42. Defending Against Unforeseen Failure Modes with Latent Adversarial Training**

cs.CR

See also followup work at arXiv:2407.15549

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2403.05030v6) [paper-pdf](http://arxiv.org/pdf/2403.05030v6)

**Authors**: Stephen Casper, Lennart Schulze, Oam Patel, Dylan Hadfield-Menell

**Abstract**: Despite extensive diagnostics and debugging by developers, AI systems sometimes exhibit harmful unintended behaviors. Finding and fixing these is challenging because the attack surface is so large -- it is not tractable to exhaustively search for inputs that may elicit harmful behaviors. Red-teaming and adversarial training (AT) are commonly used to improve robustness, however, they empirically struggle to fix failure modes that differ from the attacks used during training. In this work, we utilize latent adversarial training (LAT) to defend against vulnerabilities without leveraging knowledge of what they are or using inputs that elicit them. LAT makes use of the compressed, abstract, and structured latent representations of concepts that the network actually uses for prediction. Here, we use it to defend against failure modes without examples that elicit them. Specifically, we use LAT to remove backdoors and defend against held-out classes of adversarial attacks. We show in image classification, text classification, and text generation tasks that LAT usually improves both robustness to novel attacks and performance on clean data relative to AT. This suggests that LAT can be a promising tool for defending against failure modes that are not explicitly identified by developers.



## **43. Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

cs.LG

Code at https://github.com/aengusl/latent-adversarial-training.  Models at https://huggingface.co/LLM-LAT

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2407.15549v3) [paper-pdf](http://arxiv.org/pdf/2407.15549v3)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.



## **44. PRISM: Programmatic Reasoning with Image Sequence Manipulation for LVLM Jailbreaking**

cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21540v1) [paper-pdf](http://arxiv.org/pdf/2507.21540v1)

**Authors**: Quanchen Zou, Zonghao Ying, Moyang Chen, Wenzhuo Xu, Yisong Xiao, Yakai Li, Deyue Zhang, Dongdong Yang, Zhao Liu, Xiangzheng Zhang

**Abstract**: The increasing sophistication of large vision-language models (LVLMs) has been accompanied by advances in safety alignment mechanisms designed to prevent harmful content generation. However, these defenses remain vulnerable to sophisticated adversarial attacks. Existing jailbreak methods typically rely on direct and semantically explicit prompts, overlooking subtle vulnerabilities in how LVLMs compose information over multiple reasoning steps. In this paper, we propose a novel and effective jailbreak framework inspired by Return-Oriented Programming (ROP) techniques from software security. Our approach decomposes a harmful instruction into a sequence of individually benign visual gadgets. A carefully engineered textual prompt directs the sequence of inputs, prompting the model to integrate the benign visual gadgets through its reasoning process to produce a coherent and harmful output. This makes the malicious intent emergent and difficult to detect from any single component. We validate our method through extensive experiments on established benchmarks including SafeBench and MM-SafetyBench, targeting popular LVLMs. Results show that our approach consistently and substantially outperforms existing baselines on state-of-the-art models, achieving near-perfect attack success rates (over 0.90 on SafeBench) and improving ASR by up to 0.39. Our findings reveal a critical and underexplored vulnerability that exploits the compositional reasoning abilities of LVLMs, highlighting the urgent need for defenses that secure the entire reasoning process.



## **45. Can We End the Cat-and-Mouse Game? Simulating Self-Evolving Phishing Attacks with LLMs and Genetic Algorithms**

cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21538v1) [paper-pdf](http://arxiv.org/pdf/2507.21538v1)

**Authors**: Seiji Sato, Tetsushi Ohki, Masakatsu Nishigaki

**Abstract**: Anticipating emerging attack methodologies is crucial for proactive cybersecurity. Recent advances in Large Language Models (LLMs) have enabled the automated generation of phishing messages and accelerated research into potential attack techniques. However, predicting future threats remains challenging due to reliance on existing training data. To address this limitation, we propose a novel framework that integrates LLM-based phishing attack simulations with a genetic algorithm in a psychological context, enabling phishing strategies to evolve dynamically through adversarial interactions with simulated victims. Through simulations using Llama 3.1, we demonstrate that (1) self-evolving phishing strategies employ increasingly sophisticated psychological manipulation techniques, surpassing naive LLM-generated attacks, (2) variations in a victim's prior knowledge significantly influence the evolution of attack strategies, and (3) adversarial interactions between evolving attacks and adaptive defenses create a cat-and-mouse dynamic, revealing an inherent asymmetry in cybersecurity -- attackers continuously refine their methods, whereas defenders struggle to comprehensively counter all evolving threats. Our approach provides a scalable, cost-effective method for analyzing the evolution of phishing strategies and defenses, offering insights into future social engineering threats and underscoring the necessity of proactive cybersecurity measures.



## **46. NCCR: to Evaluate the Robustness of Neural Networks and Adversarial Examples**

cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21483v1) [paper-pdf](http://arxiv.org/pdf/2507.21483v1)

**Authors**: Pu Shi

**Abstract**: Neural networks have received a lot of attention recently, and related security issues have come with it. Many studies have shown that neural networks are vulnerable to adversarial examples that have been artificially perturbed with modification, which is too small to be distinguishable by human perception. Different attacks and defenses have been proposed to solve these problems, but there is little research on evaluating the robustness of neural networks and their inputs. In this work, we propose a metric called the neuron cover change rate (NCCR) to measure the ability of deep learning models to resist attacks and the stability of adversarial examples. NCCR monitors alterations in the output of specifically chosen neurons when the input is perturbed, and networks with a smaller degree of variation are considered to be more robust. The results of the experiment on image recognition and the speaker recognition model show that our metrics can provide a good assessment of the robustness of neural networks or their inputs. It can also be used to detect whether an input is adversarial or not, as adversarial examples are always less robust.



## **47. PAR-AdvGAN: Improving Adversarial Attack Capability with Progressive Auto-Regression AdvGAN**

cs.LG

Best paper award of ECML-PKDD 2025

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2502.12207v2) [paper-pdf](http://arxiv.org/pdf/2502.12207v2)

**Authors**: Jiayu Zhang, Zhiyu Zhu, Xinyi Wang, Silin Liao, Zhibo Jin, Flora D. Salim, Huaming Chen

**Abstract**: Deep neural networks have demonstrated remarkable performance across various domains. However, they are vulnerable to adversarial examples, which can lead to erroneous predictions. Generative Adversarial Networks (GANs) can leverage the generators and discriminators model to quickly produce high-quality adversarial examples. Since both modules train in a competitive and simultaneous manner, GAN-based algorithms like AdvGAN can generate adversarial examples with better transferability compared to traditional methods. However, the generation of perturbations is usually limited to a single iteration, preventing these examples from fully exploiting the potential of the methods. To tackle this issue, we introduce a novel approach named Progressive Auto-Regression AdvGAN (PAR-AdvGAN). It incorporates an auto-regressive iteration mechanism within a progressive generation network to craft adversarial examples with enhanced attack capability. We thoroughly evaluate our PAR-AdvGAN method with a large-scale experiment, demonstrating its superior performance over various state-of-the-art black-box adversarial attacks, as well as the original AdvGAN.Moreover, PAR-AdvGAN significantly accelerates the adversarial example generation, i.e., achieving the speeds of up to 335.5 frames per second on Inception-v3 model, outperforming the gradient-based transferable attack algorithms. Our code is available at: https://github.com/LMBTough/PAR



## **48. Cascading and Proxy Membership Inference Attacks**

cs.CR

Our code is available at: https://github.com/zealscott/MIA

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21412v1) [paper-pdf](http://arxiv.org/pdf/2507.21412v1)

**Authors**: Yuntao Du, Jiacheng Li, Yuetian Chen, Kaiyuan Zhang, Zhizhen Yuan, Hanshen Xiao, Bruno Ribeiro, Ninghui Li

**Abstract**: A Membership Inference Attack (MIA) assesses how much a trained machine learning model reveals about its training data by determining whether specific query instances were included in the dataset. We classify existing MIAs into adaptive or non-adaptive, depending on whether the adversary is allowed to train shadow models on membership queries. In the adaptive setting, where the adversary can train shadow models after accessing query instances, we highlight the importance of exploiting membership dependencies between instances and propose an attack-agnostic framework called Cascading Membership Inference Attack (CMIA), which incorporates membership dependencies via conditional shadow training to boost membership inference performance.   In the non-adaptive setting, where the adversary is restricted to training shadow models before obtaining membership queries, we introduce Proxy Membership Inference Attack (PMIA). PMIA employs a proxy selection strategy that identifies samples with similar behaviors to the query instance and uses their behaviors in shadow models to perform a membership posterior odds test for membership inference. We provide theoretical analyses for both attacks, and extensive experimental results demonstrate that CMIA and PMIA substantially outperform existing MIAs in both settings, particularly in the low false-positive regime, which is crucial for evaluating privacy risks.



## **49. FedStrategist: A Meta-Learning Framework for Adaptive and Robust Aggregation in Federated Learning**

cs.LG

24 pages, 8 figures. This work is intended for a journal submission

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.14322v2) [paper-pdf](http://arxiv.org/pdf/2507.14322v2)

**Authors**: Md Rafid Haque, Abu Raihan Mostofa Kamal, Md. Azam Hossain

**Abstract**: Federated Learning (FL) offers a paradigm for privacy-preserving collaborative AI, but its decentralized nature creates significant vulnerabilities to model poisoning attacks. While numerous static defenses exist, their effectiveness is highly context-dependent, often failing against adaptive adversaries or in heterogeneous data environments. This paper introduces FedStrategist, a novel meta-learning framework that reframes robust aggregation as a real-time, cost-aware control problem. We design a lightweight contextual bandit agent that dynamically selects the optimal aggregation rule from an arsenal of defenses based on real-time diagnostic metrics. Through comprehensive experiments, we demonstrate that no single static rule is universally optimal. We show that our adaptive agent successfully learns superior policies across diverse scenarios, including a ``Krum-favorable" environment and against a sophisticated "stealth" adversary designed to neutralize specific diagnostic signals. Critically, we analyze the paradoxical scenario where a non-robust baseline achieves high but compromised accuracy, and demonstrate that our agent learns a conservative policy to prioritize model integrity. Furthermore, we prove the agent's policy is controllable via a single "risk tolerance" parameter, allowing practitioners to explicitly manage the trade-off between performance and security. Our work provides a new, practical, and analyzable approach to creating resilient and intelligent decentralized AI systems.



## **50. Radio Adversarial Attacks on EMG-based Gesture Recognition Networks**

cs.CR

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.21387v1) [paper-pdf](http://arxiv.org/pdf/2507.21387v1)

**Authors**: Hongyi Xie

**Abstract**: Surface electromyography (EMG) enables non-invasive human-computer interaction in rehabilitation, prosthetics, and virtual reality. While deep learning models achieve over 97% classification accuracy, their vulnerability to adversarial attacks remains largely unexplored in the physical domain. We present ERa Attack, the first radio frequency (RF) adversarial method targeting EMG devices through intentional electromagnetic interference (IEMI). Using low-power software-defined radio transmitters, attackers inject optimized RF perturbations to mislead downstream models. Our approach bridges digital and physical domains: we generate adversarial perturbations using Projected Gradient Descent, extract 50-150 Hz components via inverse STFT, and employ synchronization-free strategies (constant spectrum noise or narrowband modulation). Perturbations, constrained to 1-10% of signal amplitude, are amplitude-modulated onto 433 MHz carriers. Experiments on the Myo Dataset (7 gestures, 350 samples) demonstrate significant impact: at 1 meter and 0 dBm transmission power, classification accuracy drops from 97.8% to 58.3%, with 41.7% misclassification rate and 25.6% targeted attack success rate. Attack effectiveness decreases exponentially with distance, recovering to 85% accuracy at 3 meters. Increasing power to 10 dBm reduces accuracy by an additional 15% at 1 meter. This work pioneers RF-based adversarial attacks on EMG recognition systems, revealing critical vulnerabilities in safety-critical applications. We quantify attack effectiveness across different perturbation modes and distances, and propose defenses including hardware shielding, spectrum monitoring, and adversarial training. Our findings inform the design of robust EMG systems against electromagnetic threats.



