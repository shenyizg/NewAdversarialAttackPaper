# LLM / MLLM (Language Models) - Other
**update at 2026-01-25 10:36:50**

Sorted by classifier confidence (high to low).

## **1. KinGuard: Hierarchical Kinship-Aware Fingerprinting to Defend Against Large Language Model Stealing**

cs.CR

Accepted by ICASSP2026

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.12986v2) [paper-pdf](https://arxiv.org/pdf/2601.12986v2)

**Confidence**: 0.95

**Authors**: Zhenhua Xu, Xiaoning Tian, Wenjun Zeng, Wenpeng Xing, Tianliang Lu, Gaolei Li, Chaochao Chen, Meng Han

**Abstract**: Protecting the intellectual property of large language models requires robust ownership verification. Conventional backdoor fingerprinting, however, is flawed by a stealth-robustness paradox: to be robust, these methods force models to memorize fixed responses to high-perplexity triggers, but this targeted overfitting creates detectable statistical artifacts. We resolve this paradox with KinGuard, a framework that embeds a private knowledge corpus built on structured kinship narratives. Instead of memorizing superficial triggers, the model internalizes this knowledge via incremental pre-training, and ownership is verified by probing its conceptual understanding. Extensive experiments demonstrate KinGuard's superior effectiveness, stealth, and resilience against a battery of attacks including fine-tuning, input perturbation, and model merging. Our work establishes knowledge-based embedding as a practical and secure paradigm for model fingerprinting.



## **2. DNF: Dual-Layer Nested Fingerprinting for Large Language Model Intellectual Property Protection**

cs.CR

Accepted by ICASSP2026

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2601.08223v3) [paper-pdf](https://arxiv.org/pdf/2601.08223v3)

**Confidence**: 0.95

**Authors**: Zhenhua Xu, Yiran Zhao, Mengting Zhong, Dezhang Kong, Changting Lin, Tong Qiao, Meng Han

**Abstract**: The rapid growth of large language models raises pressing concerns about intellectual property protection under black-box deployment. Existing backdoor-based fingerprints either rely on rare tokens -- leading to high-perplexity inputs susceptible to filtering -- or use fixed trigger-response mappings that are brittle to leakage and post-hoc adaptation. We propose \textsc{Dual-Layer Nested Fingerprinting} (DNF), a black-box method that embeds a hierarchical backdoor by coupling domain-specific stylistic cues with implicit semantic triggers. Across Mistral-7B, LLaMA-3-8B-Instruct, and Falcon3-7B-Instruct, DNF achieves perfect fingerprint activation while preserving downstream utility. Compared with existing methods, it uses lower-perplexity triggers, remains undetectable under fingerprint detection attacks, and is relatively robust to incremental fine-tuning and model merging. These results position DNF as a practical, stealthy, and resilient solution for LLM ownership verification and intellectual property protection.



## **3. PrivTune: Efficient and Privacy-Preserving Fine-Tuning of Large Language Models via Device-Cloud Collaboration**

cs.CR

Accepted at IEEE INFOCOM 2026 (full version). Update the cited references

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2512.08809v3) [paper-pdf](https://arxiv.org/pdf/2512.08809v3)

**Confidence**: 0.95

**Authors**: Yi Liu, Weixiang Han, Chengjun Cai, Xingliang Yuan, Cong Wang

**Abstract**: With the rise of large language models, service providers offer language models as a service, enabling users to fine-tune customized models via uploaded private datasets. However, this raises concerns about sensitive data leakage. Prior methods, relying on differential privacy within device-cloud collaboration frameworks, struggle to balance privacy and utility, exposing users to inference attacks or degrading fine-tuning performance. To address this, we propose PrivTune, an efficient and privacy-preserving fine-tuning framework via Split Learning (SL). The key idea of PrivTune is to inject crafted noise into token representations from the SL bottom model, making each token resemble the $n$-hop indirect neighbors. PrivTune formulates this as an optimization problem to compute the optimal noise vector, aligning with defense-utility goals. On this basis, it then adjusts the parameters (i.e., mean) of the $d_χ$-Privacy noise distribution to align with the optimization direction and scales the noise according to token importance to minimize distortion. Experiments on five datasets (covering both classification and generation tasks) against three embedding inversion and three attribute inference attacks show that, using RoBERTa on the Stanford Sentiment Treebank dataset, PrivTune reduces the attack success rate to 10% with only a 3.33% drop in utility performance, outperforming state-of-the-art baselines.



## **4. A Visual Semantic Adaptive Watermark grounded by Prefix-Tuning for Large Vision-Language Model**

cs.CV

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07291v1) [paper-pdf](https://arxiv.org/pdf/2601.07291v1)

**Confidence**: 0.95

**Authors**: Qi Zheng, Shuliang Liu, Yu Huang, Sihang Jia, Jungang Li, Lyuhao Chen, Junhao Chen, Hanqian Li, Aiwei Liu, Yibo Yan, Xuming Hu

**Abstract**: Watermarking has emerged as a pivotal solution for content traceability and intellectual property protection in Large Vision-Language Models (LVLMs). However, vision-agnostic watermarks introduce visually irrelevant tokens and disrupt visual grounding by enforcing indiscriminate pseudo-random biases, while some semantic-aware methods incur prohibitive inference latency due to rejection sampling. In this paper, we propose the VIsual Semantic Adaptive Watermark (VISA-Mark), a novel framework that embeds detectable signals while strictly preserving visual fidelity. Our approach employs a lightweight, efficiently trained prefix-tuner to extract dynamic Visual-Evidence Weights, which quantify the evidentiary support for candidate tokens based on the visual input. These weights guide an adaptive vocabulary partitioning and logits perturbation mechanism, concentrating watermark strength specifically on visually-supported tokens. By actively aligning the watermark with visual evidence, VISA-Mark effectively maintains visual fidelity. Empirical results confirm that VISA-Mark outperforms conventional methods with a 7.8% improvement in visual consistency (Chair-I) and superior semantic fidelity. The framework maintains highly competitive detection accuracy (96.88% AUC) and robust attack resilience (99.3%) without sacrificing inference efficiency, effectively establishing a new standard for reliability-preserving multimodal watermarking.



## **5. Large Language Models for Detecting Cyberattacks on Smart Grid Protective Relays**

cs.CR

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2601.04443v1) [paper-pdf](https://arxiv.org/pdf/2601.04443v1)

**Confidence**: 0.95

**Authors**: Ahmad Mohammad Saber, Saeed Jafari, Zhengmao Ouyang, Paul Budnarain, Amr Youssef, Deepa Kundur

**Abstract**: This paper presents a large language model (LLM)-based framework for detecting cyberattacks on transformer current differential relays (TCDRs), which, if undetected, may trigger false tripping of critical transformers. The proposed approach adapts and fine-tunes compact LLMs such as DistilBERT to distinguish cyberattacks from actual faults using textualized multidimensional TCDR current measurements recorded before and after tripping. Our results demonstrate that DistilBERT detects 97.6% of cyberattacks without compromising TCDR dependability and achieves inference latency below 6 ms on a commercial workstation. Additional evaluations confirm the framework's robustness under combined time-synchronization and false-data-injection attacks, resilience to measurement noise, and stability across prompt formulation variants. Furthermore, GPT-2 and DistilBERT+LoRA achieve comparable performance, highlighting the potential of LLMs for enhancing smart grid cybersecurity. We provide the full dataset used in this study for reproducibility.



## **6. Multimodal Adversarial Defense for Vision-Language Models by Leveraging One-To-Many Relationships**

cs.CV

WACV 2026 Accepted. Code available at https://github.com/CyberAgentAILab/multimodal-adversarial-training

**SubmitDate**: 2026-01-05    [abs](http://arxiv.org/abs/2405.18770v6) [paper-pdf](https://arxiv.org/pdf/2405.18770v6)

**Confidence**: 0.95

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos, Isao Echizen

**Abstract**: Pre-trained vision-language (VL) models are highly vulnerable to adversarial attacks. However, existing defense methods primarily focus on image classification, overlooking two key aspects of VL tasks: multimodal attacks, where both image and text can be perturbed, and the one-to-many relationship of images and texts, where a single image can correspond to multiple textual descriptions and vice versa (1:N and N:1). This work is the first to explore defense strategies against multimodal attacks in VL tasks, whereas prior VL defense methods focus on vision robustness. We propose multimodal adversarial training (MAT), which incorporates adversarial perturbations in both image and text modalities during training, significantly outperforming existing unimodal defenses. Furthermore, we discover that MAT is limited by deterministic one-to-one (1:1) image-text pairs in VL training data. To address this, we conduct a comprehensive study on leveraging one-to-many relationships to enhance robustness, investigating diverse augmentation techniques. Our analysis shows that, for a more effective defense, augmented image-text pairs should be well-aligned, diverse, yet avoid distribution shift -- conditions overlooked by prior research. This work pioneers defense strategies against multimodal attacks, providing insights for building robust VLMs from both optimization and data perspectives. Our code is publicly available at https://github.com/CyberAgentAILab/multimodal-adversarial-training.



## **7. Who Can See Through You? Adversarial Shielding Against VLM-Based Attribute Inference Attacks**

cs.CV

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18264v1) [paper-pdf](https://arxiv.org/pdf/2512.18264v1)

**Confidence**: 0.95

**Authors**: Yucheng Fan, Jiawei Chen, Yu Tian, Zhaoxia Yin

**Abstract**: As vision-language models (VLMs) become widely adopted, VLM-based attribute inference attacks have emerged as a serious privacy concern, enabling adversaries to infer private attributes from images shared on social media. This escalating threat calls for dedicated protection methods to safeguard user privacy. However, existing methods often degrade the visual quality of images or interfere with vision-based functions on social media, thereby failing to achieve a desirable balance between privacy protection and user experience. To address this challenge, we propose a novel protection method that jointly optimizes privacy suppression and utility preservation under a visual consistency constraint. While our method is conceptually effective, fair comparisons between methods remain challenging due to the lack of publicly available evaluation datasets. To fill this gap, we introduce VPI-COCO, a publicly available benchmark comprising 522 images with hierarchically structured privacy questions and corresponding non-private counterparts, enabling fine-grained and joint evaluation of protection methods in terms of privacy preservation and user experience. Building upon this benchmark, experiments on multiple VLMs demonstrate that our method effectively reduces PAR below 25%, keeps NPAR above 88%, maintains high visual consistency, and generalizes well to unseen and paraphrased privacy questions, demonstrating its strong practical applicability for real-world VLM deployments.



## **8. MoAPT: Mixture of Adversarial Prompt Tuning for Vision-Language Models**

cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2505.17509v2) [paper-pdf](https://arxiv.org/pdf/2505.17509v2)

**Confidence**: 0.95

**Authors**: Shiji Zhao, Qihui Zhu, Shukun Xiong, Shouwei Ruan, Maoxun Yuan, Jialing Tao, Jiexi Liu, Ranjie Duan, Jie Zhang, Jie Zhang, Xingxing Wei

**Abstract**: Large pre-trained Vision Language Models (VLMs) demonstrate excellent generalization capabilities but remain highly susceptible to adversarial examples, posing potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which ultimately results in overfitting. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts yields greater robustness improvements than simply extending the length of a single prompt. Building on this observation, we propose an adversarial tuning method named \textbf{Mixture of Adversarial Prompt Tuning (MoAPT)} to enhance the generalization against various adversarial attacks for VLMs. MoAPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the adversarial images to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific mixture text features aligning with different adversarial image features. Extensive experiments across 11 datasets under different settings show that our method can achieve better adversarial robustness than state-of-the-art approaches.



## **9. Proxy Robustness in Vision Language Models is Effortlessly Transferable**

cs.CV

**SubmitDate**: 2026-01-19    [abs](http://arxiv.org/abs/2601.12865v1) [paper-pdf](https://arxiv.org/pdf/2601.12865v1)

**Confidence**: 0.95

**Authors**: Xiaowei Fu, Fuxiang Huang, Lei Zhang

**Abstract**: As a pivotal technique for improving the defense of deep models, adversarial robustness transfer via distillation has demonstrated remarkable success in conventional image classification tasks. However, this paradigm encounters critical challenges when applied to vision-language models (VLM) (e.g., CLIP): constructing adversarially robust teacher for large-scale multi-modal models demands prohibitively high computational resources. We bridge this gap by revealing an interesting phenomenon: vanilla CLIP (without adversarial training) exhibits intrinsic defensive capabilities against adversarial examples generated by another CLIP with different architectures. We formally define this as proxy adversarial robustness, and naturally propose a Heterogeneous Proxy Transfer (HPT) framework that establishes cross-architectural robustness distillation channels between CLIP variants, effortlessly enabling the VLM robustness transfer from proxy to target models. Yet, such proxy transfer paradigm easily induces severe overfitting, leading to a sharp degradation in zero-shot natural generalization. To resolve that, we design Generalization-Pivot Decoupling (GPD) by leveraging the difference in learning rate scheduling. This decouples the proxy transfer process into a generalization-anchored warm-up that maintains generalization and a generalization-pulled HPT that promotes adversarial robustness, to achieve an equilibrium between natural generalization and adversarial robustness. Extensive experiments on 15 zero-shot datasets demonstrate the effectiveness of our HPT-GPD method. The code is available at the website of github.com/fxw13/HPT-GPD.



## **10. SoK: Privacy-aware LLM in Healthcare: Threat Model, Privacy Techniques, Challenges and Recommendations**

cs.CR

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.10004v1) [paper-pdf](https://arxiv.org/pdf/2601.10004v1)

**Confidence**: 0.95

**Authors**: Mohoshin Ara Tahera, Karamveer Singh Sidhu, Shuvalaxmi Dass, Sajal Saha

**Abstract**: Large Language Models (LLMs) are increasingly adopted in healthcare to support clinical decision-making, summarize electronic health records (EHRs), and enhance patient care. However, this integration introduces significant privacy and security challenges, driven by the sensitivity of clinical data and the high-stakes nature of medical workflows. These risks become even more pronounced across heterogeneous deployment environments, ranging from small on-premise hospital systems to regional health networks, each with unique resource limitations and regulatory demands. This Systematization of Knowledge (SoK) examines the evolving threat landscape across the three core LLM phases: Data preprocessing, Fine-tuning, and Inference within realistic healthcare settings. We present a detailed threat model that characterizes adversaries, capabilities, and attack surfaces at each phase, and we systematize how existing privacy-preserving techniques (PPTs) attempt to mitigate these vulnerabilities. While existing defenses show promise, our analysis identifies persistent limitations in securing sensitive clinical data across diverse operational tiers. We conclude with phase-aware recommendations and future research directions aimed at strengthening privacy guarantees for LLMs in regulated environments. This work provides a foundation for understanding the intersection of LLMs, threats, and privacy in healthcare, offering a roadmap toward more robust and clinically trustworthy AI systems.



## **11. Rethinking On-Device LLM Reasoning: Why Analogical Mapping Outperforms Abstract Thinking for IoT DDoS Detection**

cs.CR

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2601.14343v1) [paper-pdf](https://arxiv.org/pdf/2601.14343v1)

**Confidence**: 0.90

**Authors**: William Pan, Guiran Liu, Binrong Zhu, Qun Wang, Yingzhou Lu, Beiyu Lin, Rose Qingyang Hu

**Abstract**: The rapid expansion of IoT deployments has intensified cybersecurity threats, notably Distributed Denial of Service (DDoS) attacks, characterized by increasingly sophisticated patterns. Leveraging Generative AI through On-Device Large Language Models (ODLLMs) provides a viable solution for real-time threat detection at the network edge, though limited computational resources present challenges for smaller ODLLMs. This paper introduces a novel detection framework that integrates Chain-of-Thought (CoT) reasoning with Retrieval-Augmented Generation (RAG), tailored specifically for IoT edge environments. We systematically evaluate compact ODLLMs, including LLaMA 3.2 (1B, 3B) and Gemma 3 (1B, 4B), using structured prompting and exemplar-driven reasoning strategies. Experimental results demonstrate substantial performance improvements with few-shot prompting, achieving macro-average F1 scores as high as 0.85. Our findings highlight the significant advantages of incorporating exemplar-based reasoning, underscoring that CoT and RAG approaches markedly enhance small ODLLMs' capabilities in accurately classifying complex network attacks under stringent resource constraints.



## **12. Multifaceted Evaluation of Audio-Visual Capability for MLLMs: Effectiveness, Efficiency, Generalizability and Robustness**

cs.MM

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.16936v1) [paper-pdf](https://arxiv.org/pdf/2504.16936v1)

**Confidence**: 0.90

**Authors**: Yusheng Zhao, Junyu Luo, Xiao Luo, Weizhi Zhang, Zhiping Xiao, Wei Ju, Philip S. Yu, Ming Zhang

**Abstract**: Multi-modal large language models (MLLMs) have recently achieved great success in processing and understanding information from diverse modalities (e.g., text, audio, and visual signals). Despite their growing popularity, there remains a lack of comprehensive evaluation measuring the audio-visual capabilities of these models, especially in diverse scenarios (e.g., distribution shifts and adversarial attacks). In this paper, we present a multifaceted evaluation of the audio-visual capability of MLLMs, focusing on four key dimensions: effectiveness, efficiency, generalizability, and robustness. Through extensive experiments, we find that MLLMs exhibit strong zero-shot and few-shot generalization abilities, enabling them to achieve great performance with limited data. However, their success relies heavily on the vision modality, which impairs performance when visual input is corrupted or missing. Additionally, while MLLMs are susceptible to adversarial samples, they demonstrate greater robustness compared to traditional models. The experimental results and our findings provide insights into the audio-visual capabilities of MLLMs, highlighting areas for improvement and offering guidance for future research.



## **13. PhishLumos: An Adaptive Multi-Agent System for Proactive Phishing Campaign Mitigation**

cs.CR

Accepted for publication at IEEE ICC 2026

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2509.21772v2) [paper-pdf](https://arxiv.org/pdf/2509.21772v2)

**Confidence**: 0.85

**Authors**: Daiki Chiba, Hiroki Nakano, Takashi Koide

**Abstract**: Phishing attacks are a significant societal threat, disproportionately harming vulnerable populations and eroding trust in essential digital services. Current defenses are often reactive, failing against modern evasive tactics like cloaking that conceal malicious content. To address this, we introduce PhishLumos, an adaptive multi-agent system that proactively mitigates entire attack campaigns. It confronts a core cybersecurity imbalance: attackers can easily scale operations, while defense remains an intensive expert task. Instead of being blocked by evasion, PhishLumos treats it as a critical signal to investigate the underlying infrastructure. Its Large Language Model (LLM)-powered agents uncover shared hosting, certificates, and domain registration patterns. On real-world data, our system identified 100% of campaigns in the median case, over a week before their confirmation by cybersecurity experts. PhishLumos demonstrates a practical shift from reactive URL blocking to proactive campaign mitigation, protecting users before they are harmed and making the digital world safer for all.



## **14. Large AI Model-Enabled Secure Communications in Low-Altitude Wireless Networks: Concepts, Perspectives and Case Study**

cs.NI

This paper has been accepted to IEEE Communications Magazine

**SubmitDate**: 2026-01-20    [abs](http://arxiv.org/abs/2508.00256v2) [paper-pdf](https://arxiv.org/pdf/2508.00256v2)

**Confidence**: 0.85

**Authors**: Chuang Zhang, Geng Sun, Yijing Lin, Weijie Yuan, Sinem Coleri, Dusit Niyato

**Abstract**: Low-altitude wireless networks (LAWNs) have the potential to revolutionize communications by supporting a range of applications, including urban parcel delivery, aerial inspections and air taxis. However, compared with traditional wireless networks, LAWNs face unique security challenges due to low-altitude operations, frequent mobility and reliance on unlicensed spectrum, making it more vulnerable to some malicious attacks. In this paper, we investigate some large artificial intelligence model (LAM)-enabled solutions for secure communications in LAWNs. Specifically, we first explore the amplified security risks and important limitations of traditional AI methods in LAWNs. Then, we introduce the basic concepts of LAMs and delve into the role of LAMs in addressing these challenges. To demonstrate the practical benefits of LAMs for secure communications in LAWNs, we propose a novel LAM-based optimization framework that leverages large language models (LLMs) to generate enhanced state features on top of handcrafted representations, and to design intrinsic rewards accordingly, thereby improving reinforcement learning performance for secure communication tasks. Through a typical case study, simulation results validate the effectiveness of the proposed framework. Finally, we outline future directions for integrating LAMs into secure LAWN applications.



## **15. Adversarial Defense in Vision-Language Models: An Overview**

cs.CV

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2601.12443v1) [paper-pdf](https://arxiv.org/pdf/2601.12443v1)

**Confidence**: 0.85

**Authors**: Xiaowei Fu, Lei Zhang

**Abstract**: The widespread use of Vision Language Models (VLMs, e.g. CLIP) has raised concerns about their vulnerability to sophisticated and imperceptible adversarial attacks. These attacks could compromise model performance and system security in cross-modal tasks. To address this challenge, three main defense paradigms have been proposed: Training-time Defense, Test-time Adaptation Defense, and Training-free Defense. Training-time Defense involves modifying the training process, typically through adversarial fine-tuning to improve the robustness to adversarial examples. While effective, this approach requires substantial computational resources and may not generalize across all adversarial attacks. Test-time Adaptation Defense focuses on adapting the model at inference time by updating its parameters to handle unlabeled adversarial examples, offering flexibility but often at the cost of increased complexity and computational overhead. Training-free Defense avoids modifying the model itself, instead focusing on altering the adversarial inputs or their feature embeddings, which enforces input perturbations to mitigate the impact of attacks without additional training. This survey reviews the latest advancements in adversarial defense strategies for VLMs, highlighting the strengths and limitations of such approaches and discussing ongoing challenges in enhancing the robustness of VLMs.



## **16. Cisco Integrated AI Security and Safety Framework Report**

cs.CR

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12921v1) [paper-pdf](https://arxiv.org/pdf/2512.12921v1)

**Confidence**: 0.85

**Authors**: Amy Chang, Tiffany Saade, Sanket Mendapara, Adam Swanda, Ankit Garg

**Abstract**: Artificial intelligence (AI) systems are being readily and rapidly adopted, increasingly permeating critical domains: from consumer platforms and enterprise software to networked systems with embedded agents. While this has unlocked potential for human productivity gains, the attack surface has expanded accordingly: threats now span content safety failures (e.g., harmful or deceptive outputs), model and data integrity compromise (e.g., poisoning, supply-chain tampering), runtime manipulations (e.g., prompt injection, tool and agent misuse), and ecosystem risks (e.g., orchestration abuse, multi-agent collusion). Existing frameworks such as MITRE ATLAS, National Institute of Standards and Technology (NIST) AI 100-2 Adversarial Machine Learning (AML) taxonomy, and OWASP Top 10s for Large Language Models (LLMs) and Agentic AI Applications provide valuable viewpoints, but each covers only slices of this multi-dimensional space.   This paper presents Cisco's Integrated AI Security and Safety Framework ("AI Security Framework"), a unified, lifecycle-aware taxonomy and operationalization framework that can be used to classify, integrate, and operationalize the full range of AI risks. It integrates AI security and AI safety across modalities, agents, pipelines, and the broader ecosystem. The AI Security Framework is designed to be practical for threat identification, red-teaming, risk prioritization, and it is comprehensive in scope and can be extensible to emerging deployments in multimodal contexts, humanoids, wearables, and sensory infrastructures. We analyze gaps in prevailing frameworks, discuss design principles for our framework, and demonstrate how the taxonomy provides structure for understanding how modern AI systems fail, how adversaries exploit these failures, and how organizations can build defenses across the AI lifecycle that evolve alongside capability advancements.



## **17. BRIDG-ICS: AI-Grounded Knowledge Graphs for Intelligent Threat Analytics in Industry~5.0 Cyber-Physical Systems**

cs.CR

44 Pages, To be published in Springer Cybersecurity Journal

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12112v1) [paper-pdf](https://arxiv.org/pdf/2512.12112v1)

**Confidence**: 0.85

**Authors**: Padmeswari Nandiya, Ahmad Mohsin, Ahmed Ibrahim, Iqbal H. Sarker, Helge Janicke

**Abstract**: Industry 5.0's increasing integration of IT and OT systems is transforming industrial operations but also expanding the cyber-physical attack surface. Industrial Control Systems (ICS) face escalating security challenges as traditional siloed defences fail to provide coherent, cross-domain threat insights. We present BRIDG-ICS (BRIDge for Industrial Control Systems), an AI-driven Knowledge Graph (KG) framework for context-aware threat analysis and quantitative assessment of cyber resilience in smart manufacturing environments. BRIDG-ICS fuses heterogeneous industrial and cybersecurity data into an integrated Industrial Security Knowledge Graph linking assets, vulnerabilities, and adversarial behaviours with probabilistic risk metrics (e.g. exploit likelihood, attack cost). This unified graph representation enables multi-stage attack path simulation using graph-analytic techniques. To enrich the graph's semantic depth, the framework leverages Large Language Models (LLMs): domain-specific LLMs extract cybersecurity entities, predict relationships, and translate natural-language threat descriptions into structured graph triples, thereby populating the knowledge graph with missing associations and latent risk indicators. This unified AI-enriched KG supports multi-hop, causality-aware threat reasoning, improving visibility into complex attack chains and guiding data-driven mitigation. In simulated industrial scenarios, BRIDG-ICS scales well, reduces potential attack exposure, and can enhance cyber-physical system resilience in Industry 5.0 settings.



## **18. Enhancing Cloud Network Resilience via a Robust LLM-Empowered Multi-Agent Reinforcement Learning Framework**

cs.CR

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07122v1) [paper-pdf](https://arxiv.org/pdf/2601.07122v1)

**Confidence**: 0.85

**Authors**: Yixiao Peng, Hao Hu, Feiyang Li, Xinye Cao, Yingchang Jiang, Jipeng Tang, Guoshun Nan, Yuling Liu

**Abstract**: While virtualization and resource pooling empower cloud networks with structural flexibility and elastic scalability, they inevitably expand the attack surface and challenge cyber resilience. Reinforcement Learning (RL)-based defense strategies have been developed to optimize resource deployment and isolation policies under adversarial conditions, aiming to enhance system resilience by maintaining and restoring network availability. However, existing approaches lack robustness as they require retraining to adapt to dynamic changes in network structure, node scale, attack strategies, and attack intensity. Furthermore, the lack of Human-in-the-Loop (HITL) support limits interpretability and flexibility. To address these limitations, we propose CyberOps-Bots, a hierarchical multi-agent reinforcement learning framework empowered by Large Language Models (LLMs). Inspired by MITRE ATT&CK's Tactics-Techniques model, CyberOps-Bots features a two-layer architecture: (1) An upper-level LLM agent with four modules--ReAct planning, IPDRR-based perception, long-short term memory, and action/tool integration--performs global awareness, human intent recognition, and tactical planning; (2) Lower-level RL agents, developed via heterogeneous separated pre-training, execute atomic defense actions within localized network regions. This synergy preserves LLM adaptability and interpretability while ensuring reliable RL execution. Experiments on real cloud datasets show that, compared to state-of-the-art algorithms, CyberOps-Bots maintains network availability 68.5% higher and achieves a 34.7% jumpstart performance gain when shifting the scenarios without retraining. To our knowledge, this is the first study to establish a robust LLM-RL framework with HITL support for cloud defense. We will release our framework to the community, facilitating the advancement of robust and autonomous defense in cloud networks.



