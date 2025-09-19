# Latest Adversarial Attack Papers
**update at 2025-09-19 15:29:23**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Beyond Surface Alignment: Rebuilding LLMs Safety Mechanism via Probabilistically Ablating Refusal Direction**

cs.CR

Accepted by EMNLP2025 Finding

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15202v1) [paper-pdf](http://arxiv.org/pdf/2509.15202v1)

**Authors**: Yuanbo Xie, Yingjie Zhang, Tianyun Liu, Duohe Ma, Tingwen Liu

**Abstract**: Jailbreak attacks pose persistent threats to large language models (LLMs). Current safety alignment methods have attempted to address these issues, but they experience two significant limitations: insufficient safety alignment depth and unrobust internal defense mechanisms. These limitations make them vulnerable to adversarial attacks such as prefilling and refusal direction manipulation. We introduce DeepRefusal, a robust safety alignment framework that overcomes these issues. DeepRefusal forces the model to dynamically rebuild its refusal mechanisms from jailbreak states. This is achieved by probabilistically ablating the refusal direction across layers and token depths during fine-tuning. Our method not only defends against prefilling and refusal direction attacks but also demonstrates strong resilience against other unseen jailbreak strategies. Extensive evaluations on four open-source LLM families and six representative attacks show that DeepRefusal reduces attack success rates by approximately 95%, while maintaining model capabilities with minimal performance degradation.



## **2. AIP: Subverting Retrieval-Augmented Generation via Adversarial Instructional Prompt**

cs.CV

Accepted at EMNLP 2025 Conference

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.15159v1) [paper-pdf](http://arxiv.org/pdf/2509.15159v1)

**Authors**: Saket S. Chaturvedi, Gaurav Bagwe, Lan Zhang, Xiaoyong Yuan

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by retrieving relevant documents from external sources to improve factual accuracy and verifiability. However, this reliance introduces new attack surfaces within the retrieval pipeline, beyond the LLM itself. While prior RAG attacks have exposed such vulnerabilities, they largely rely on manipulating user queries, which is often infeasible in practice due to fixed or protected user inputs. This narrow focus overlooks a more realistic and stealthy vector: instructional prompts, which are widely reused, publicly shared, and rarely audited. Their implicit trust makes them a compelling target for adversaries to manipulate RAG behavior covertly.   We introduce a novel attack for Adversarial Instructional Prompt (AIP) that exploits adversarial instructional prompts to manipulate RAG outputs by subtly altering retrieval behavior. By shifting the attack surface to the instructional prompts, AIP reveals how trusted yet seemingly benign interface components can be weaponized to degrade system integrity. The attack is crafted to achieve three goals: (1) naturalness, to evade user detection; (2) utility, to encourage use of prompts; and (3) robustness, to remain effective across diverse query variations. We propose a diverse query generation strategy that simulates realistic linguistic variation in user queries, enabling the discovery of prompts that generalize across paraphrases and rephrasings. Building on this, a genetic algorithm-based joint optimization is developed to evolve adversarial prompts by balancing attack success, clean-task utility, and stealthiness. Experimental results show that AIP achieves up to 95.23% ASR while preserving benign functionality. These findings uncover a critical and previously overlooked vulnerability in RAG systems, emphasizing the need to reassess the shared instructional prompts.



## **3. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

cs.CV

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2409.13174v3) [paper-pdf](http://arxiv.org/pdf/2409.13174v3)

**Authors**: Hao Cheng, Erjia Xiao, Yichi Wang, Chengyuan Yu, Mengshu Sun, Qiang Zhang, Yijie Guo, Kaidi Xu, Jize Zhang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical threats.



## **4. GASLITEing the Retrieval: Exploring Vulnerabilities in Dense Embedding-based Search**

cs.CR

Accepted at ACM CCS 2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2412.20953v2) [paper-pdf](http://arxiv.org/pdf/2412.20953v2)

**Authors**: Matan Ben-Tov, Mahmood Sharif

**Abstract**: Dense embedding-based text retrieval$\unicode{x2013}$retrieval of relevant passages from corpora via deep learning encodings$\unicode{x2013}$has emerged as a powerful method attaining state-of-the-art search results and popularizing Retrieval Augmented Generation (RAG). Still, like other search methods, embedding-based retrieval may be susceptible to search-engine optimization (SEO) attacks, where adversaries promote malicious content by introducing adversarial passages to corpora. Prior work has shown such SEO is feasible, mostly demonstrating attacks against retrieval-integrated systems (e.g., RAG). Yet, these consider relaxed SEO threat models (e.g., targeting single queries), use baseline attack methods, and provide small-scale retrieval evaluation, thus obscuring our comprehensive understanding of retrievers' worst-case behavior. This work aims to faithfully and thoroughly assess retrievers' robustness, paving a path to uncover factors related to their susceptibility to SEO. To this end, we, first, propose the GASLITE attack for generating adversarial passages, that$\unicode{x2013}$without relying on the corpus content or modifying the model$\unicode{x2013}$carry adversary-chosen information while achieving high retrieval ranking, consistently outperforming prior approaches. Second, using GASLITE, we extensively evaluate retrievers' robustness, testing nine advanced models under varied threat models, while focusing on pertinent adversaries targeting queries on a specific concept (e.g., a public figure). Amongst our findings: retrievers are highly vulnerable to SEO against concept-specific queries, even under negligible poisoning rates (e.g., $\geq$0.0001% of the corpus), while generalizing across different corpora and query distributions; single-query SEO is completely solved by GASLITE; adaptive attacks demonstrate bypassing common defenses; [...]



## **5. Timestamp Manipulation: Timestamp-based Nakamoto-style Blockchains are Vulnerable**

cs.CR

25 pages, 6 figures

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2505.05328v4) [paper-pdf](http://arxiv.org/pdf/2505.05328v4)

**Authors**: Junjie Hu, Sisi Duan

**Abstract**: Nakamoto consensus are the most widely adopted decentralized consensus mechanism in cryptocurrency systems. Since it was proposed in 2008, many studies have focused on analyzing its security. Most of them focus on maximizing the profit of the adversary. Examples include the selfish mining attack [FC '14] and the recent riskless uncle maker (RUM) attack [CCS '23]. In this work, we introduce the Staircase-Unrestricted Uncle Maker (SUUM), the first block withholding attack targeting the timestamp-based Nakamoto-style blockchain. Through block withholding, timestamp manipulation, and difficulty risk control, SUUM adversaries are capable of launching persistent attacks with zero cost and minimal difficulty risk characteristics, indefinitely exploiting rewards from honest participants. This creates a self-reinforcing cycle that threatens the security of blockchains. We conduct a comprehensive and systematic evaluation of SUUM, including the attack conditions, its impact on blockchains, and the difficulty risks. Finally, we further discuss four feasible mitigation measures against SUUM.



## **6. Discrete optimal transport is a strong audio adversarial attack**

eess.AS

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14959v1) [paper-pdf](http://arxiv.org/pdf/2509.14959v1)

**Authors**: Anton Selitskiy, Akib Shahriyar, Jishnuraj Prakasan

**Abstract**: In this paper, we show that discrete optimal transport (DOT) is an effective black-box adversarial attack against modern audio anti-spoofing countermeasures (CMs). Our attack operates as a post-processing, distribution-alignment step: frame-level WavLM embeddings of generated speech are aligned to an unpaired bona fide pool via entropic OT and a top-$k$ barycentric projection, then decoded with a neural vocoder. Evaluated on ASVspoof2019 and ASVspoof5 with AASIST baselines, DOT yields consistently high equal error rate (EER) across datasets and remains competitive after CM fine-tuning, outperforming several conventional attacks in cross-dataset transfer. Ablation analysis highlights the practical impact of vocoder overlap. Results indicate that distribution-level alignment is a powerful and stable attack surface for deployed CMs.



## **7. Cost-Performance Analysis: A Comparative Study of CPU-Based Serverless and GPU-Based Training Architectures**

cs.DC

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14920v1) [paper-pdf](http://arxiv.org/pdf/2509.14920v1)

**Authors**: Amine Barrak, Fabio Petrillo, Fehmi Jaafar

**Abstract**: The field of distributed machine learning (ML) faces increasing demands for scalable and cost-effective training solutions, particularly in the context of large, complex models. Serverless computing has emerged as a promising paradigm to address these challenges by offering dynamic scalability and resource-efficient execution. Building upon our previous work, which introduced the Serverless Peer Integrated for Robust Training (SPIRT) architecture, this paper presents a comparative analysis of several serverless distributed ML architectures. We examine SPIRT alongside established architectures like ScatterReduce, AllReduce, and MLLess, focusing on key metrics such as training time efficiency, cost-effectiveness, communication overhead, and fault tolerance capabilities. Our findings reveal that SPIRT provides significant improvements in reducing training times and communication overhead through strategies such as parallel batch processing and in-database operations facilitated by RedisAI. However, traditional architectures exhibit scalability challenges and varying degrees of vulnerability to faults and adversarial attacks. The cost analysis underscores the long-term economic benefits of SPIRT despite its higher initial setup costs. This study not only highlights the strengths and limitations of current serverless ML architectures but also sets the stage for future research aimed at developing new models that combine the most effective features of existing systems.



## **8. Birds look like cars: Adversarial analysis of intrinsically interpretable deep learning**

cs.LG

Accepted by Machine Learning

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2503.08636v2) [paper-pdf](http://arxiv.org/pdf/2503.08636v2)

**Authors**: Hubert Baniecki, Przemyslaw Biecek

**Abstract**: A common belief is that intrinsically interpretable deep learning models ensure a correct, intuitive understanding of their behavior and offer greater robustness against accidental errors or intentional manipulation. However, these beliefs have not been comprehensively verified, and growing evidence casts doubt on them. In this paper, we highlight the risks related to overreliance and susceptibility to adversarial manipulation of these so-called "intrinsically (aka inherently) interpretable" models by design. We introduce two strategies for adversarial analysis with prototype manipulation and backdoor attacks against prototype-based networks, and discuss how concept bottleneck models defend against these attacks. Fooling the model's reasoning by exploiting its use of latent prototypes manifests the inherent uninterpretability of deep neural networks, leading to a false sense of security reinforced by a visual confirmation bias. The reported limitations of part-prototype networks put their trustworthiness and applicability into question, motivating further work on the robustness and alignment of (deep) interpretable models.



## **9. STEP: Structured Training and Evaluation Platform for benchmarking trajectory prediction models**

cs.LG

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14801v1) [paper-pdf](http://arxiv.org/pdf/2509.14801v1)

**Authors**: Julian F. Schumann, Anna Mészáros, Jens Kober, Arkady Zgonnikov

**Abstract**: While trajectory prediction plays a critical role in enabling safe and effective path-planning in automated vehicles, standardized practices for evaluating such models remain underdeveloped. Recent efforts have aimed to unify dataset formats and model interfaces for easier comparisons, yet existing frameworks often fall short in supporting heterogeneous traffic scenarios, joint prediction models, or user documentation. In this work, we introduce STEP -- a new benchmarking framework that addresses these limitations by providing a unified interface for multiple datasets, enforcing consistent training and evaluation conditions, and supporting a wide range of prediction models. We demonstrate the capabilities of STEP in a number of experiments which reveal 1) the limitations of widely-used testing procedures, 2) the importance of joint modeling of agents for better predictions of interactions, and 3) the vulnerability of current state-of-the-art models against both distribution shifts and targeted attacks by adversarial agents. With STEP, we aim to shift the focus from the ``leaderboard'' approach to deeper insights about model behavior and generalization in complex multi-agent settings.



## **10. Top K Enhanced Reinforcement Learning Attacks on Heterogeneous Graph Node Classification**

cs.LG

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2408.01964v2) [paper-pdf](http://arxiv.org/pdf/2408.01964v2)

**Authors**: Honglin Gao, Xiang Li, Yajuan Sun, Gaoxi Xiao

**Abstract**: Graph Neural Networks (GNNs) have attracted substantial interest due to their exceptional performance on graph-based data. However, their robustness, especially on heterogeneous graphs, remains underexplored, particularly against adversarial attacks. This paper proposes HeteroKRLAttack, a targeted evasion black-box attack method for heterogeneous graphs. By integrating reinforcement learning with a Top-K algorithm to reduce the action space, our method efficiently identifies effective attack strategies to disrupt node classification tasks. We validate the effectiveness of HeteroKRLAttack through experiments on multiple heterogeneous graph datasets, showing significant reductions in classification accuracy compared to baseline methods. An ablation study underscores the critical role of the Top-K algorithm in enhancing attack performance. Our findings highlight potential vulnerabilities in current models and provide guidance for future defense strategies against adversarial attacks on heterogeneous graphs.



## **11. Mini-Batch Robustness Verification of Deep Neural Networks**

cs.LG

30 pages, 12 figures, conference OOPSLA 2025

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2508.15454v2) [paper-pdf](http://arxiv.org/pdf/2508.15454v2)

**Authors**: Saar Tzour-Shaday, Dana Drachsler-Cohen

**Abstract**: Neural network image classifiers are ubiquitous in many safety-critical applications. However, they are susceptible to adversarial attacks. To understand their robustness to attacks, many local robustness verifiers have been proposed to analyze $\epsilon$-balls of inputs. Yet, existing verifiers introduce a long analysis time or lose too much precision, making them less effective for a large set of inputs. In this work, we propose a new approach to local robustness: group local robustness verification. The key idea is to leverage the similarity of the network computations of certain $\epsilon$-balls to reduce the overall analysis time. We propose BaVerLy, a sound and complete verifier that boosts the local robustness verification of a set of $\epsilon$-balls by dynamically constructing and verifying mini-batches. BaVerLy adaptively identifies successful mini-batch sizes, accordingly constructs mini-batches of $\epsilon$-balls that have similar network computations, and verifies them jointly. If a mini-batch is verified, all its $\epsilon$-balls are proven robust. Otherwise, one $\epsilon$-ball is suspected as not being robust, guiding the refinement. BaVerLy leverages the analysis results to expedite the analysis of that $\epsilon$-ball as well as the analysis of the mini-batch with the other $\epsilon$-balls. We evaluate BaVerLy on fully connected and convolutional networks for MNIST and CIFAR-10. Results show that BaVerLy scales the common one by one verification by 2.3x on average and up to 4.1x, in which case it reduces the total analysis time from 24 hours to 6 hours.



## **12. MUSE: MCTS-Driven Red Teaming Framework for Enhanced Multi-Turn Dialogue Safety in Large Language Models**

cs.CL

EMNLP 2025 main conference

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14651v1) [paper-pdf](http://arxiv.org/pdf/2509.14651v1)

**Authors**: Siyu Yan, Long Zeng, Xuecheng Wu, Chengcheng Han, Kongcheng Zhang, Chong Peng, Xuezhi Cao, Xunliang Cai, Chenjuan Guo

**Abstract**: As large language models~(LLMs) become widely adopted, ensuring their alignment with human values is crucial to prevent jailbreaks where adversaries manipulate models to produce harmful content. While most defenses target single-turn attacks, real-world usage often involves multi-turn dialogues, exposing models to attacks that exploit conversational context to bypass safety measures. We introduce MUSE, a comprehensive framework tackling multi-turn jailbreaks from both attack and defense angles. For attacks, we propose MUSE-A, a method that uses frame semantics and heuristic tree search to explore diverse semantic trajectories. For defense, we present MUSE-D, a fine-grained safety alignment approach that intervenes early in dialogues to reduce vulnerabilities. Extensive experiments on various models show that MUSE effectively identifies and mitigates multi-turn vulnerabilities. Code is available at \href{https://github.com/yansiyu02/MUSE}{https://github.com/yansiyu02/MUSE}.



## **13. Enterprise AI Must Enforce Participant-Aware Access Control**

cs.CR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2509.14608v1) [paper-pdf](http://arxiv.org/pdf/2509.14608v1)

**Authors**: Shashank Shreedhar Bhatt, Tanmay Rajore, Khushboo Aggarwal, Ganesh Ananthanarayanan, Ranveer Chandra, Nishanth Chandran, Suyash Choudhury, Divya Gupta, Emre Kiciman, Sumit Kumar Pandey, Srinath Setty, Rahul Sharma, Teijia Zhao

**Abstract**: Large language models (LLMs) are increasingly deployed in enterprise settings where they interact with multiple users and are trained or fine-tuned on sensitive internal data. While fine-tuning enhances performance by internalizing domain knowledge, it also introduces a critical security risk: leakage of confidential training data to unauthorized users. These risks are exacerbated when LLMs are combined with Retrieval-Augmented Generation (RAG) pipelines that dynamically fetch contextual documents at inference time.   We demonstrate data exfiltration attacks on AI assistants where adversaries can exploit current fine-tuning and RAG architectures to leak sensitive information by leveraging the lack of access control enforcement. We show that existing defenses, including prompt sanitization, output filtering, system isolation, and training-level privacy mechanisms, are fundamentally probabilistic and fail to offer robust protection against such attacks.   We take the position that only a deterministic and rigorous enforcement of fine-grained access control during both fine-tuning and RAG-based inference can reliably prevent the leakage of sensitive data to unauthorized recipients.   We introduce a framework centered on the principle that any content used in training, retrieval, or generation by an LLM is explicitly authorized for \emph{all users involved in the interaction}. Our approach offers a simple yet powerful paradigm shift for building secure multi-user LLM systems that are grounded in classical access control but adapted to the unique challenges of modern AI workflows. Our solution has been deployed in Microsoft Copilot Tuning, a product offering that enables organizations to fine-tune models using their own enterprise-specific data.



## **14. GRADA: Graph-based Reranking against Adversarial Documents Attack**

cs.IR

**SubmitDate**: 2025-09-18    [abs](http://arxiv.org/abs/2505.07546v3) [paper-pdf](http://arxiv.org/pdf/2505.07546v3)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.



## **15. Measuring Soft Biometric Leakage in Speaker De-Identification Systems**

cs.SD

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.14469v1) [paper-pdf](http://arxiv.org/pdf/2509.14469v1)

**Authors**: Seungmin Seo, Oleg Aulov, P. Jonathon Phillips

**Abstract**: We use the term re-identification to refer to the process of recovering the original speaker's identity from anonymized speech outputs. Speaker de-identification systems aim to reduce the risk of re-identification, but most evaluations focus only on individual-level measures and overlook broader risks from soft biometric leakage. We introduce the Soft Biometric Leakage Score (SBLS), a unified method that quantifies resistance to zero-shot inference attacks on non-unique traits such as channel type, age range, dialect, sex of the speaker, or speaking style. SBLS integrates three elements: direct attribute inference using pre-trained classifiers, linkage detection via mutual information analysis, and subgroup robustness across intersecting attributes. Applying SBLS with publicly available classifiers, we show that all five evaluated de-identification systems exhibit significant vulnerabilities. Our results indicate that adversaries using only pre-trained models - without access to original speech or system details - can still reliably recover soft biometric information from anonymized output, exposing fundamental weaknesses that standard distributional metrics fail to capture.



## **16. Enhanced Rényi Entropy-Based Post-Quantum Key Agreement with Provable Security and Information-Theoretic Guarantees**

cs.CR

31 pages, 4 tables

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.00104v2) [paper-pdf](http://arxiv.org/pdf/2509.00104v2)

**Authors**: Ruopengyu Xu, Chenglian Liu

**Abstract**: This paper presents an enhanced post-quantum key agreement protocol based on R\'enyi entropy, addressing vulnerabilities in the original construction while preserving information-theoretic security properties. We develop a theoretical framework leveraging entropy-preserving operations and secret-shared verification to achieve provable security against quantum adversaries. Through entropy amplification techniques and quantum-resistant commitments, the protocol establishes $2^{128}$ quantum security guarantees under the quantum random oracle model. Key innovations include a confidentiality-preserving verification mechanism using distributed polynomial commitments, tightened min-entropy bounds with guaranteed non-negativity, and composable security proofs in the quantum universal composability framework. Unlike computational approaches, our method provides information-theoretic security without hardness assumptions while maintaining polynomial complexity. Theoretical analysis demonstrates resilience against known quantum attack vectors, including Grover-accelerated brute force and quantum memory attacks. The protocol achieves parameterization for 128-bit quantum security with efficient $\mathcal{O}(n^{2})$ communication complexity. Extensions to secure multiparty computation and quantum network applications are established, providing a foundation for long-term cryptographic security.



## **17. Building the Self-Improvement Loop: Error Detection and Correction in Goal-Oriented Semantic Communications**

cs.NI

7 pages, 8 figures, this paper has been accepted for publication in  IEEE CSCN 2024

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2411.01544v2) [paper-pdf](http://arxiv.org/pdf/2411.01544v2)

**Authors**: Peizheng Li, Xinyi Lin, Adnan Aijaz

**Abstract**: Error detection and correction are essential for ensuring robust and reliable operation in modern communication systems, particularly in complex transmission environments. However, discussions on these topics have largely been overlooked in semantic communication (SemCom), which focuses on transmitting meaning rather than symbols, leading to significant improvements in communication efficiency. Despite these advantages, semantic errors -- stemming from discrepancies between transmitted and received meanings -- present a major challenge to system reliability. This paper addresses this gap by proposing a comprehensive framework for detecting and correcting semantic errors in SemCom systems. We formally define semantic error, detection, and correction mechanisms, and identify key sources of semantic errors. To address these challenges, we develop a Gaussian process (GP)-based method for latent space monitoring to detect errors, alongside a human-in-the-loop reinforcement learning (HITL-RL) approach to optimize semantic model configurations using user feedback. Experimental results validate the effectiveness of the proposed methods in mitigating semantic errors under various conditions, including adversarial attacks, input feature changes, physical channel variations, and user preference shifts. This work lays the foundation for more reliable and adaptive SemCom systems with robust semantic error management techniques.



## **18. Evaluating the Defense Potential of Machine Unlearning against Membership Inference Attacks**

cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2508.16150v3) [paper-pdf](http://arxiv.org/pdf/2508.16150v3)

**Authors**: Aristeidis Sidiropoulos, Christos Chrysanthos Nikolaidis, Theodoros Tsiolakis, Nikolaos Pavlidis, Vasilis Perifanis, Pavlos S. Efraimidis

**Abstract**: Membership Inference Attacks (MIAs) pose a significant privacy risk, as they enable adversaries to determine whether a specific data point was included in the training dataset of a model. While Machine Unlearning is primarily designed as a privacy mechanism to efficiently remove private data from a machine learning model without the need for full retraining, its impact on the susceptibility of models to MIA remains an open question. In this study, we systematically assess the vulnerability of models to MIA after applying state-of-art Machine Unlearning algorithms. Our analysis spans four diverse datasets (two from the image domain and two in tabular format), exploring how different unlearning approaches influence the exposure of models to membership inference. The findings highlight that while Machine Unlearning is not inherently a countermeasure against MIA, the unlearning algorithm and data characteristics can significantly affect a model's vulnerability. This work provides essential insights into the interplay between Machine Unlearning and MIAs, offering guidance for the design of privacy-preserving machine learning systems.



## **19. CLMTracing: Black-box User-level Watermarking for Code Language Model Tracing**

cs.PL

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.13982v1) [paper-pdf](http://arxiv.org/pdf/2509.13982v1)

**Authors**: Boyu Zhang, Ping He, Tianyu Du, Xuhong Zhang, Lei Yun, Kingsum Chow, Jianwei Yin

**Abstract**: With the widespread adoption of open-source code language models (code LMs), intellectual property (IP) protection has become an increasingly critical concern. While current watermarking techniques have the potential to identify the code LM to protect its IP, they have limitations when facing the more practical and complex demand, i.e., offering the individual user-level tracing in the black-box setting. This work presents CLMTracing, a black-box code LM watermarking framework employing the rule-based watermarks and utility-preserving injection method for user-level model tracing. CLMTracing further incorporates a parameter selection algorithm sensitive to the robust watermark and adversarial training to enhance the robustness against watermark removal attacks. Comprehensive evaluations demonstrate CLMTracing is effective across multiple state-of-the-art (SOTA) code LMs, showing significant harmless improvements compared to existing SOTA baselines and strong robustness against various removal attacks.



## **20. CyberLLMInstruct: A Pseudo-malicious Dataset Revealing Safety-performance Trade-offs in Cyber Security LLM Fine-tuning**

cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2503.09334v3) [paper-pdf](http://arxiv.org/pdf/2503.09334v3)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents both opportunities and critical safety risks. We introduce CyberLLMInstruct, a dataset of 54,928 pseudo-malicious instruction-response pairs spanning cyber security tasks including malware analysis, phishing simulations, and zero-day vulnerabilities. Our comprehensive evaluation using seven open-source LLMs reveals a critical trade-off: while fine-tuning improves cyber security task performance (achieving up to 92.50% accuracy on CyberMetric), it severely compromises safety resilience across all tested models and attack vectors (e.g., Llama 3.1 8B's security score against prompt injection drops from 0.95 to 0.15). The dataset incorporates diverse sources including CTF challenges, academic papers, industry reports, and CVE databases to ensure comprehensive coverage of cyber security domains. Our findings highlight the unique challenges of securing LLMs in adversarial domains and establish the critical need for developing fine-tuning methodologies that balance performance gains with safety preservation in security-sensitive domains.



## **21. Graph Neural Networks for Next-Generation-IoT: Recent Advances and Open Challenges**

cs.IT

38 pages, 18 figures, and 6 tables. Accepted by the IEEE COMST

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2412.20634v3) [paper-pdf](http://arxiv.org/pdf/2412.20634v3)

**Authors**: Nguyen Xuan Tung, Le Tung Giang, Bui Duc Son, Seon Geun Jeong, Trinh Van Chien, Won Joo Hwang, Lajos Hanzo

**Abstract**: Graph Neural Networks (GNNs) have emerged as a powerful framework for modeling complex interconnected systems, hence making them particularly well-suited to address the growing challenges of next-generation Internet of Things (NG-IoT) networks. Existing studies remain fragmented, and there is a lack of comprehensive guidance on how GNNs can be systematically applied to NG-IoT systems. As NG-IoT systems evolve toward 6G, they incorporate diverse technologies. These advances promise unprecedented connectivity, sensing, and automation but also introduce significant complexity, requiring new approaches for scalable learning, dynamic optimization, and secure, decentralized decision-making. This survey provides a comprehensive and forward-looking exploration of how GNNs can empower NG-IoT environments. We commence by exploring the fundamental paradigms of GNNs and articulating the motivation for their use in NG-IoT networks. Besides, we intrinsically connect GNNs with the family of low-density parity-check codes, modeling the NG-IoT as dynamic constrained graphs. We highlight the distinct roles of node-, edge-, and graph-level tasks in tackling key challenges and demonstrate the GNNs' ability to overcome the limitations of traditional optimization. We examine the application of GNNs across core NG-enabling technologies and their integration with distributed frameworks to support privacy-preservation and distributed intelligence. We then delve into the challenges posed by adversarial attacks, offering insights into defense mechanisms. Lastly, we examine how GNNs can be integrated with emerging technologies. Our findings highlight the transformative potential of GNNs in improving efficiency, scalability, and security. Finally, we summarize the key lessons learned and outline promising future research directions, along with a set of design guidelines tailored for NG-IoT applications.



## **22. DiffHash: Text-Guided Targeted Attack via Diffusion Models against Deep Hashing Image Retrieval**

cs.IR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.12824v2) [paper-pdf](http://arxiv.org/pdf/2509.12824v2)

**Authors**: Zechao Liu, Zheng Zhou, Xiangkun Chen, Tao Liang, Dapeng Lang

**Abstract**: Deep hashing models have been widely adopted to tackle the challenges of large-scale image retrieval. However, these approaches face serious security risks due to their vulnerability to adversarial examples. Despite the increasing exploration of targeted attacks on deep hashing models, existing approaches still suffer from a lack of multimodal guidance, reliance on labeling information and dependence on pixel-level operations for attacks. To address these limitations, we proposed DiffHash, a novel diffusion-based targeted attack for deep hashing. Unlike traditional pixel-based attacks that directly modify specific pixels and lack multimodal guidance, our approach focuses on optimizing the latent representations of images, guided by text information generated by a Large Language Model (LLM) for the target image. Furthermore, we designed a multi-space hash alignment network to align the high-dimension image space and text space to the low-dimension binary hash space. During reconstruction, we also incorporated text-guided attention mechanisms to refine adversarial examples, ensuring them aligned with the target semantics while maintaining visual plausibility. Extensive experiments have demonstrated that our method outperforms state-of-the-art (SOTA) targeted attack methods, achieving better black-box transferability and offering more excellent stability across datasets.



## **23. Removal Attack and Defense on AI-generated Content Latent-based Watermarking**

cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.11745v2) [paper-pdf](http://arxiv.org/pdf/2509.11745v2)

**Authors**: De Zhang Lee, Han Fang, Hanyi Wang, Ee-Chien Chang

**Abstract**: Digital watermarks can be embedded into AI-generated content (AIGC) by initializing the generation process with starting points sampled from a secret distribution. When combined with pseudorandom error-correcting codes, such watermarked outputs can remain indistinguishable from unwatermarked objects, while maintaining robustness under whitenoise. In this paper, we go beyond indistinguishability and investigate security under removal attacks. We demonstrate that indistinguishability alone does not necessarily guarantee resistance to adversarial removal. Specifically, we propose a novel attack that exploits boundary information leaked by the locations of watermarked objects. This attack significantly reduces the distortion required to remove watermarks -- by up to a factor of $15 \times$ compared to a baseline whitenoise attack under certain settings. To mitigate such attacks, we introduce a defense mechanism that applies a secret transformation to hide the boundary, and prove that the secret transformation effectively rendering any attacker's perturbations equivalent to those of a naive whitenoise adversary. Our empirical evaluations, conducted on multiple versions of Stable Diffusion, validate the effectiveness of both the attack and the proposed defense, highlighting the importance of addressing boundary leakage in latent-based watermarking schemes.



## **24. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

cs.CL

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2412.12478v4) [paper-pdf](http://arxiv.org/pdf/2412.12478v4)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models excel across various NLP tasks but remain highly vulnerable to textual adversarial attacks. While adversarial text generation is crucial for NLP security, explainability, evaluation, and data augmentation, related work remains overwhelmingly English-centric, leaving the problem of constructing high-quality and sustainable adversarial robustness benchmarks for lower-resourced languages both difficult and understudied. First, method customization for lower-resourced languages is complicated due to linguistic differences and limited resources. Second, automated attacks are prone to generating invalid or ambiguous adversarial texts. Last but not least, language models continuously evolve and may be immune to parts of previously generated adversarial texts. To address these challenges, we introduce HITL-GAT, an interactive system based on a general approach to human-in-the-loop generation of adversarial texts. Additionally, we demonstrate the utility of HITL-GAT through a case study on Tibetan script, employing three customized adversarial text generation methods and establishing its first adversarial robustness benchmark, providing a valuable reference for other lower-resourced languages.



## **25. SoK: How Sensor Attacks Disrupt Autonomous Vehicles: An End-to-end Analysis, Challenges, and Missed Threats**

cs.CR

**SubmitDate**: 2025-09-17    [abs](http://arxiv.org/abs/2509.11120v3) [paper-pdf](http://arxiv.org/pdf/2509.11120v3)

**Authors**: Qingzhao Zhang, Shaocheng Luo, Z. Morley Mao, Miroslav Pajic, Michael K. Reiter

**Abstract**: Autonomous vehicles, including self-driving cars, robotic ground vehicles, and drones, rely on complex sensor pipelines to ensure safe and reliable operation. However, these safety-critical systems remain vulnerable to adversarial sensor attacks that can compromise their performance and mission success. While extensive research has demonstrated various sensor attack techniques, critical gaps remain in understanding their feasibility in real-world, end-to-end systems. This gap largely stems from the lack of a systematic perspective on how sensor errors propagate through interconnected modules in autonomous systems when autonomous vehicles interact with the physical world.   To bridge this gap, we present a comprehensive survey of autonomous vehicle sensor attacks across platforms, sensor modalities, and attack methods. Central to our analysis is the System Error Propagation Graph (SEPG), a structured demonstration tool that illustrates how sensor attacks propagate through system pipelines, exposing the conditions and dependencies that determine attack feasibility. With the aid of SEPG, our study distills seven key findings that highlight the feasibility challenges of sensor attacks and uncovers eleven previously overlooked attack vectors exploiting inter-module interactions, several of which we validate through proof-of-concept experiments. Additionally, we demonstrate how large language models (LLMs) can automate aspects of SEPG construction and cross-validate expert analysis, showcasing the promise of AI-assisted security evaluation.



## **26. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

cs.CL

Accepted at EMNLP 2025 (Main)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2501.01872v5) [paper-pdf](http://arxiv.org/pdf/2501.01872v5)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.



## **27. AQUA-LLM: Evaluating Accuracy, Quantization, and Adversarial Robustness Trade-offs in LLMs for Cybersecurity Question Answering**

cs.CR

Accepted by the 24th IEEE International Conference on Machine  Learning and Applications (ICMLA'25)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.13514v1) [paper-pdf](http://arxiv.org/pdf/2509.13514v1)

**Authors**: Onat Gungor, Roshan Sood, Harold Wang, Tajana Rosing

**Abstract**: Large Language Models (LLMs) have recently demonstrated strong potential for cybersecurity question answering (QA), supporting decision-making in real-time threat detection and response workflows. However, their substantial computational demands pose significant challenges for deployment on resource-constrained edge devices. Quantization, a widely adopted model compression technique, can alleviate these constraints. Nevertheless, quantization may degrade model accuracy and increase susceptibility to adversarial attacks. Fine-tuning offers a potential means to mitigate these limitations, but its effectiveness when combined with quantization remains insufficiently explored. Hence, it is essential to understand the trade-offs among accuracy, efficiency, and robustness. We propose AQUA-LLM, an evaluation framework designed to benchmark several state-of-the-art small LLMs under four distinct configurations: base, quantized-only, fine-tuned, and fine-tuned combined with quantization, specifically for cybersecurity QA. Our results demonstrate that quantization alone yields the lowest accuracy and robustness despite improving efficiency. In contrast, combining quantization with fine-tuning enhances both LLM robustness and predictive performance, achieving an optimal balance of accuracy, robustness, and efficiency. These findings highlight the critical need for quantization-aware, robustness-preserving fine-tuning methodologies to enable the robust and efficient deployment of LLMs for cybersecurity QA.



## **28. JANUS: A Dual-Constraint Generative Framework for Stealthy Node Injection Attacks**

cs.LG

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.13266v1) [paper-pdf](http://arxiv.org/pdf/2509.13266v1)

**Authors**: Jiahao Zhang, Xiaobing Pei, Zhaokun Zhong, Wenqiang Hao, Zhenghao Tang

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable performance across various applications, yet they are vulnerable to sophisticated adversarial attacks, particularly node injection attacks. The success of such attacks heavily relies on their stealthiness, the ability to blend in with the original graph and evade detection. However, existing methods often achieve stealthiness by relying on indirect proxy metrics, lacking consideration for the fundamental characteristics of the injected content, or focusing only on imitating local structures, which leads to the problem of local myopia. To overcome these limitations, we propose a dual-constraint stealthy node injection framework, called Joint Alignment of Nodal and Universal Structures (JANUS). At the local level, we introduce a local feature manifold alignment strategy to achieve geometric consistency in the feature space. At the global level, we incorporate structured latent variables and maximize the mutual information with the generated structures, ensuring the injected structures are consistent with the semantic patterns of the original graph. We model the injection attack as a sequential decision process, which is optimized by a reinforcement learning agent. Experiments on multiple standard datasets demonstrate that the JANUS framework significantly outperforms existing methods in terms of both attack effectiveness and stealthiness.



## **29. Chernoff Information as a Privacy Constraint for Adversarial Classification and Membership Advantage**

cs.IT

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2403.10307v3) [paper-pdf](http://arxiv.org/pdf/2403.10307v3)

**Authors**: Ayşe Ünsal, Melek Önen

**Abstract**: This work inspects a privacy metric based on Chernoff information, namely Chernoff differential privacy, due to its significance in characterization of the optimal classifier's performance. Adversarial classification, as any other classification problem is built around minimization of the (average or correct detection) probability of error in deciding on either of the classes in the case of binary classification. Unlike the classical hypothesis testing problem, where the false alarm and mis-detection probabilities are handled separately resulting in an asymmetric behavior of the best error exponent, in this work, we characterize the relationship between $\varepsilon\textrm{-}$differential privacy, the best error exponent of one of the errors (when the other is fixed) and the best average error exponent. Accordingly, we re-derive Chernoff differential privacy in connection with $\varepsilon\textrm{-}$differential privacy using the Radon-Nikodym derivative, and prove its relation with Kullback-Leibler (KL) differential privacy. Subsequently, we present numerical evaluation results, which demonstrates that Chernoff information outperforms Kullback-Leibler divergence as a function of the privacy parameter $\varepsilon$ and the impact of the adversary's attack in Laplace mechanisms. Lastly, we introduce a new upper bound on adversary's membership advantage in membership inference attacks using Chernoff DP and numerically compare its performance with existing alternatives based on $(\varepsilon, \delta)\textrm{-}$differential privacy in the literature.



## **30. Detection of Synthetic Face Images: Accuracy, Robustness, Generalization**

cs.CV

The paper was presented at the DAGM German Conference on Pattern  Recognition (GCPR), 2025

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2406.17547v2) [paper-pdf](http://arxiv.org/pdf/2406.17547v2)

**Authors**: Nela Petrzelkova, Jan Cech

**Abstract**: An experimental study on detecting synthetic face images is presented. We collected a dataset, called FF5, of five fake face image generators, including recent diffusion models. We find that a simple model trained on a specific image generator can achieve near-perfect accuracy in separating synthetic and real images. The model handles common image distortions (reduced resolution, compression) by using data augmentation. Moreover, partial manipulations, where synthetic images are blended into real ones by inpainting, are identified and the area of the manipulation is localized by a simple model of YOLO architecture. However, the model turned out to be vulnerable to adversarial attacks and does not generalize to unseen generators. Failure to generalize to detect images produced by a newer generator also occurs for recent state-of-the-art methods, which we tested on Realistic Vision, a fine-tuned version of StabilityAI's Stable Diffusion image generator.



## **31. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

cs.CL

EMNLP 2025 Main (Oral)

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2502.13061v4) [paper-pdf](http://arxiv.org/pdf/2502.13061v4)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL



## **32. Bridging Threat Models and Detections: Formal Verification via CADP**

cs.CR

In Proceedings FROM 2025, arXiv:2509.11877

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.13035v1) [paper-pdf](http://arxiv.org/pdf/2509.13035v1)

**Authors**: Dumitru-Bogdan Prelipcean, Cătălin Dima

**Abstract**: Threat detection systems rely on rule-based logic to identify adversarial behaviors, yet the conformance of these rules to high-level threat models is rarely verified formally. We present a formal verification framework that models both detection logic and attack trees as labeled transition systems (LTSs), enabling automated conformance checking via bisimulation and weak trace inclusion. Detection rules specified in the Generic Threat Detection Language (GTDL, a general-purpose detection language we formalize in this work) are assigned a compositional operational semantics, and threat models expressed as attack trees are interpreted as LTSs through a structural trace semantics. Both representations are translated to LNT, a modeling language supported by the CADP toolbox. This common semantic domain enables systematic and automated verification of detection coverage. We evaluate our approach on real-world malware scenarios such as LokiBot and Emotet and provide scalability analysis through parametric synthetic models. Results confirm that our methodology identifies semantic mismatches between threat models and detection rules, supports iterative refinement, and scales to realistic threat landscapes.



## **33. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

cs.CR

This paper is submitted to the IEEE IoT Journal

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2508.16843v4) [paper-pdf](http://arxiv.org/pdf/2508.16843v4)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.



## **34. Sy-FAR: Symmetry-based Fair Adversarial Robustness**

cs.LG

20 pages, 11 figures

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12939v1) [paper-pdf](http://arxiv.org/pdf/2509.12939v1)

**Authors**: Haneen Najjar, Eyal Ronen, Mahmood Sharif

**Abstract**: Security-critical machine-learning (ML) systems, such as face-recognition systems, are susceptible to adversarial examples, including real-world physically realizable attacks. Various means to boost ML's adversarial robustness have been proposed; however, they typically induce unfair robustness: It is often easier to attack from certain classes or groups than from others. Several techniques have been developed to improve adversarial robustness while seeking perfect fairness between classes. Yet, prior work has focused on settings where security and fairness are less critical. Our insight is that achieving perfect parity in realistic fairness-critical tasks, such as face recognition, is often infeasible -- some classes may be highly similar, leading to more misclassifications between them. Instead, we suggest that seeking symmetry -- i.e., attacks from class $i$ to $j$ would be as successful as from $j$ to $i$ -- is more tractable. Intuitively, symmetry is a desirable because class resemblance is a symmetric relation in most domains. Additionally, as we prove theoretically, symmetry between individuals induces symmetry between any set of sub-groups, in contrast to other fairness notions where group-fairness is often elusive. We develop Sy-FAR, a technique to encourage symmetry while also optimizing adversarial robustness and extensively evaluate it using five datasets, with three model architectures, including against targeted and untargeted realistic attacks. The results show Sy-FAR significantly improves fair adversarial robustness compared to state-of-the-art methods. Moreover, we find that Sy-FAR is faster and more consistent across runs. Notably, Sy-FAR also ameliorates another type of unfairness we discover in this work -- target classes that adversarial examples are likely to be classified into become significantly less vulnerable after inducing symmetry.



## **35. Beyond Data Privacy: New Privacy Risks for Large Language Models**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.14278v1) [paper-pdf](http://arxiv.org/pdf/2509.14278v1)

**Authors**: Yuntao Du, Zitao Li, Ninghui Li, Bolin Ding

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress in natural language understanding, reasoning, and autonomous decision-making. However, these advancements have also come with significant privacy concerns. While significant research has focused on mitigating the data privacy risks of LLMs during various stages of model training, less attention has been paid to new threats emerging from their deployment. The integration of LLMs into widely used applications and the weaponization of their autonomous abilities have created new privacy vulnerabilities. These vulnerabilities provide opportunities for both inadvertent data leakage and malicious exfiltration from LLM-powered systems. Additionally, adversaries can exploit these systems to launch sophisticated, large-scale privacy attacks, threatening not only individual privacy but also financial security and societal trust. In this paper, we systematically examine these emerging privacy risks of LLMs. We also discuss potential mitigation strategies and call for the research community to broaden its focus beyond data privacy risks, developing new defenses to address the evolving threats posed by increasingly powerful LLMs and LLM-powered systems.



## **36. Adversarial Prompt Distillation for Vision-Language Models**

cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2411.15244v3) [paper-pdf](http://arxiv.org/pdf/2411.15244v3)

**Authors**: Lin Luo, Xin Wang, Bojia Zi, Shihao Zhao, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Large pre-trained Vision-Language Models (VLMs) such as Contrastive Language-Image Pre-training (CLIP) have been shown to be susceptible to adversarial attacks, raising concerns about their deployment in safety-critical applications like autonomous driving and medical diagnosis. One promising approach for robustifying pre-trained VLMs is Adversarial Prompt Tuning (APT), which applies adversarial training during the process of prompt tuning. However, existing APT methods are mostly single-modal methods that design prompt(s) for only the visual or textual modality, limiting their effectiveness in either robustness or clean accuracy. In this work, we propose Adversarial Prompt Distillation (APD), a bimodal knowledge distillation framework that enhances APT by integrating it with multi-modal knowledge transfer. APD optimizes prompts for both visual and textual modalities while distilling knowledge from a clean pre-trained teacher CLIP model. Extensive experiments on multiple benchmark datasets demonstrate the superiority of our APD method over the current state-of-the-art APT methods in terms of both adversarial robustness and clean accuracy. The effectiveness of APD also validates the possibility of using a non-robust teacher to improve the generalization and robustness of fine-tuned VLMs.



## **37. Gradient-Free Adversarial Purification with Diffusion Models**

cs.CV

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2501.13336v2) [paper-pdf](http://arxiv.org/pdf/2501.13336v2)

**Authors**: Xuelong Dai, Dong Wang, Xiuzhen Cheng, Bin Xiao

**Abstract**: Adversarial training and adversarial purification are two widely used defense strategies for enhancing model robustness against adversarial attacks. However, adversarial training requires costly retraining, while adversarial purification often suffers from low efficiency. More critically, existing defenses are primarily designed under the perturbation-based adversarial threat model, which is ineffective against recently introduced unrestricted adversarial attacks. In this paper, we propose an effective and efficient defense framework that counters both perturbation-based and unrestricted adversarial attacks. Our approach is motivated by the observation that adversarial examples typically lie near the decision boundary and are highly sensitive to pixel-level perturbations. To address this, we introduce adversarial anti-aliasing, a preprocessing technique that mitigates adversarial noise by reducing the magnitude of pixel-level perturbations. In addition, we propose adversarial super-resolution, which leverages prior knowledge from clean datasets to benignly restore high-quality images from adversarially degraded ones. Unlike image synthesis methods that generate entirely new images, adversarial super-resolution focuses on image restoration, making it more suitable for purification. Importantly, both techniques require no additional training and are computationally efficient since they do not rely on gradient computations. To further improve robustness across diverse datasets, we introduce a contrastive learning-based adversarial deblurring fine-tuning method. By incorporating adversarial priors during fine-tuning on the target dataset, this method enhances purification effectiveness without the need to retrain diffusion models.



## **38. Defense-to-Attack: Bypassing Weak Defenses Enables Stronger Jailbreaks in Vision-Language Models**

cs.CV

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12724v1) [paper-pdf](http://arxiv.org/pdf/2509.12724v1)

**Authors**: Yunhan Zhao, Xiang Zheng, Xingjun Ma

**Abstract**: Despite their superb capabilities, Vision-Language Models (VLMs) have been shown to be vulnerable to jailbreak attacks. While recent jailbreaks have achieved notable progress, their effectiveness and efficiency can still be improved. In this work, we reveal an interesting phenomenon: incorporating weak defense into the attack pipeline can significantly enhance both the effectiveness and the efficiency of jailbreaks on VLMs. Building on this insight, we propose Defense2Attack, a novel jailbreak method that bypasses the safety guardrails of VLMs by leveraging defensive patterns to guide jailbreak prompt design. Specifically, Defense2Attack consists of three key components: (1) a visual optimizer that embeds universal adversarial perturbations with affirmative and encouraging semantics; (2) a textual optimizer that refines the input using a defense-styled prompt; and (3) a red-team suffix generator that enhances the jailbreak through reinforcement fine-tuning. We empirically evaluate our method on four VLMs and four safety benchmarks. The results demonstrate that Defense2Attack achieves superior jailbreak performance in a single attempt, outperforming state-of-the-art attack methods that often require multiple tries. Our work offers a new perspective on jailbreaking VLMs.



## **39. Revisiting Transferable Adversarial Images: Systemization, Evaluation, and New Insights**

cs.CR

TPAMI 2025. Code is available at  https://github.com/ZhengyuZhao/TransferAttackEval

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2310.11850v2) [paper-pdf](http://arxiv.org/pdf/2310.11850v2)

**Authors**: Zhengyu Zhao, Hanwei Zhang, Renjue Li, Ronan Sicre, Laurent Amsaleg, Michael Backes, Qi Li, Qian Wang, Chao Shen

**Abstract**: Transferable adversarial images raise critical security concerns for computer vision systems in real-world, black-box attack scenarios. Although many transfer attacks have been proposed, existing research lacks a systematic and comprehensive evaluation. In this paper, we systemize transfer attacks into five categories around the general machine learning pipeline and provide the first comprehensive evaluation, with 23 representative attacks against 11 representative defenses, including the recent, transfer-oriented defense and the real-world Google Cloud Vision. In particular, we identify two main problems of existing evaluations: (1) for attack transferability, lack of intra-category analyses with fair hyperparameter settings, and (2) for attack stealthiness, lack of diverse measures. Our evaluation results validate that these problems have indeed caused misleading conclusions and missing points, and addressing them leads to new, \textit{consensus-challenging} insights, such as (1) an early attack, DI, even outperforms all similar follow-up ones, (2) the state-of-the-art (white-box) defense, DiffPure, is even vulnerable to (black-box) transfer attacks, and (3) even under the same $L_p$ constraint, different attacks yield dramatically different stealthiness results regarding diverse imperceptibility metrics, finer-grained measures, and a user study. We hope that our analyses will serve as guidance on properly evaluating transferable adversarial images and advance the design of attacks and defenses. Code is available at https://github.com/ZhengyuZhao/TransferAttackEval.



## **40. Towards Inclusive Toxic Content Moderation: Addressing Vulnerabilities to Adversarial Attacks in Toxicity Classifiers Tackling LLM-generated Content**

cs.CL

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12672v1) [paper-pdf](http://arxiv.org/pdf/2509.12672v1)

**Authors**: Shaz Furniturewala, Arkaitz Zubiaga

**Abstract**: The volume of machine-generated content online has grown dramatically due to the widespread use of Large Language Models (LLMs), leading to new challenges for content moderation systems. Conventional content moderation classifiers, which are usually trained on text produced by humans, suffer from misclassifications due to LLM-generated text deviating from their training data and adversarial attacks that aim to avoid detection. Present-day defence tactics are reactive rather than proactive, since they rely on adversarial training or external detection models to identify attacks. In this work, we aim to identify the vulnerable components of toxicity classifiers that contribute to misclassification, proposing a novel strategy based on mechanistic interpretability techniques. Our study focuses on fine-tuned BERT and RoBERTa classifiers, testing on diverse datasets spanning a variety of minority groups. We use adversarial attacking techniques to identify vulnerable circuits. Finally, we suppress these vulnerable circuits, improving performance against adversarial attacks. We also provide demographic-level insights into these vulnerable circuits, exposing fairness and robustness gaps in model training. We find that models have distinct heads that are either crucial for performance or vulnerable to attack and suppressing the vulnerable heads improves performance on adversarial input. We also find that different heads are responsible for vulnerability across different demographic groups, which can inform more inclusive development of toxicity detection models.



## **41. CIARD: Cyclic Iterative Adversarial Robustness Distillation**

cs.CV

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12633v1) [paper-pdf](http://arxiv.org/pdf/2509.12633v1)

**Authors**: Liming Lu, Shuchao Pang, Xu Zheng, Xiang Gu, Anan Du, Yunhuai Liu, Yongbin Zhou

**Abstract**: Adversarial robustness distillation (ARD) aims to transfer both performance and robustness from teacher model to lightweight student model, enabling resilient performance on resource-constrained scenarios. Though existing ARD approaches enhance student model's robustness, the inevitable by-product leads to the degraded performance on clean examples. We summarize the causes of this problem inherent in existing methods with dual-teacher framework as: 1. The divergent optimization objectives of dual-teacher models, i.e., the clean and robust teachers, impede effective knowledge transfer to the student model, and 2. The iteratively generated adversarial examples during training lead to performance deterioration of the robust teacher model. To address these challenges, we propose a novel Cyclic Iterative ARD (CIARD) method with two key innovations: a. A multi-teacher framework with contrastive push-loss alignment to resolve conflicts in dual-teacher optimization objectives, and b. Continuous adversarial retraining to maintain dynamic teacher robustness against performance degradation from the varying adversarial examples. Extensive experiments on CIFAR-10, CIFAR-100, and Tiny-ImageNet demonstrate that CIARD achieves remarkable performance with an average 3.53 improvement in adversarial defense rates across various attack scenarios and a 5.87 increase in clean sample accuracy, establishing a new benchmark for balancing model robustness and generalization. Our code is available at https://github.com/eminentgu/CIARD



## **42. Your Compiler is Backdooring Your Model: Understanding and Exploiting Compilation Inconsistency Vulnerabilities in Deep Learning Compilers**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.11173v2) [paper-pdf](http://arxiv.org/pdf/2509.11173v2)

**Authors**: Simin Chen, Jinjun Peng, Yixin He, Junfeng Yang, Baishakhi Ray

**Abstract**: Deep learning (DL) compilers are core infrastructure in modern DL systems, offering flexibility and scalability beyond vendor-specific libraries. This work uncovers a fundamental vulnerability in their design: can an official, unmodified compiler alter a model's semantics during compilation and introduce hidden backdoors? We study both adversarial and natural settings. In the adversarial case, we craft benign models where triggers have no effect pre-compilation but become effective backdoors after compilation. Tested on six models, three commercial compilers, and two hardware platforms, our attack yields 100% success on triggered inputs while preserving normal accuracy and remaining undetected by state-of-the-art detectors. The attack generalizes across compilers, hardware, and floating-point settings. In the natural setting, we analyze the top 100 HuggingFace models (including one with 220M+ downloads) and find natural triggers in 31 models. This shows that compilers can introduce risks even without adversarial manipulation.   Our results reveal an overlooked threat: unmodified DL compilers can silently alter model semantics. To our knowledge, this is the first work to expose inherent security risks in DL compiler design, opening a new direction for secure and trustworthy ML.



## **43. DisorientLiDAR: Physical Attacks on LiDAR-based Localization**

cs.CV

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12595v1) [paper-pdf](http://arxiv.org/pdf/2509.12595v1)

**Authors**: Yizhen Lao, Yu Zhang, Ziting Wang, Chengbo Wang, Yifei Xue, Wanpeng Shao

**Abstract**: Deep learning models have been shown to be susceptible to adversarial attacks with visually imperceptible perturbations. Even this poses a serious security challenge for the localization of self-driving cars, there has been very little exploration of attack on it, as most of adversarial attacks have been applied to 3D perception. In this work, we propose a novel adversarial attack framework called DisorientLiDAR targeting LiDAR-based localization. By reverse-engineering localization models (e.g., feature extraction networks), adversaries can identify critical keypoints and strategically remove them, thereby disrupting LiDAR-based localization. Our proposal is first evaluated on three state-of-the-art point-cloud registration models (HRegNet, D3Feat, and GeoTransformer) using the KITTI dataset. Experimental results demonstrate that removing regions containing Top-K keypoints significantly degrades their registration accuracy. We further validate the attack's impact on the Autoware autonomous driving platform, where hiding merely a few critical regions induces noticeable localization drift. Finally, we extended our attacks to the physical world by hiding critical regions with near-infrared absorptive materials, thereby successfully replicate the attack effects observed in KITTI data. This step has been closer toward the realistic physical-world attack that demonstrate the veracity and generality of our proposal.



## **44. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2508.20890v2) [paper-pdf](http://arxiv.org/pdf/2508.20890v2)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.



## **45. Exploiting Timing Side-Channels in Quantum Circuits Simulation Via ML-Based Methods**

cs.CR

**SubmitDate**: 2025-09-16    [abs](http://arxiv.org/abs/2509.12535v1) [paper-pdf](http://arxiv.org/pdf/2509.12535v1)

**Authors**: Ben Dong, Hui Feng, Qian Wang

**Abstract**: As quantum computing advances, quantum circuit simulators serve as critical tools to bridge the current gap caused by limited quantum hardware availability. These simulators are typically deployed on cloud platforms, where users submit proprietary circuit designs for simulation. In this work, we demonstrate a novel timing side-channel attack targeting cloud-based quantum simulators. A co-located malicious process can observe fine-grained execution timing patterns to extract sensitive information about concurrently running quantum circuits. We systematically analyze simulator behavior using the QASMBench benchmark suite, profiling timing and memory characteristics across various circuit executions. Our experimental results show that timing profiles exhibit circuit-dependent patterns that can be effectively classified using pattern recognition techniques, enabling the adversary to infer circuit identities and compromise user confidentiality. We were able to achieve 88% to 99.9% identification rate of quantum circuits based on different datasets. This work highlights previously unexplored security risks in quantum simulation environments and calls for stronger isolation mechanisms to protect user workloads



## **46. Early Approaches to Adversarial Fine-Tuning for Prompt Injection Defense: A 2022 Study of GPT-3 and Contemporary Models**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.14271v1) [paper-pdf](http://arxiv.org/pdf/2509.14271v1)

**Authors**: Gustavo Sandoval, Denys Fenchenko, Junyao Chen

**Abstract**: This paper documents early research conducted in 2022 on defending against prompt injection attacks in large language models, providing historical context for the evolution of this critical security domain. This research focuses on two adversarial attacks against Large Language Models (LLMs): prompt injection and goal hijacking. We examine how to construct these attacks, test them on various LLMs, and compare their effectiveness. We propose and evaluate a novel defense technique called Adversarial Fine-Tuning. Our results show that, without this defense, the attacks succeeded 31\% of the time on GPT-3 series models. When using our Adversarial Fine-Tuning approach, attack success rates were reduced to near zero for smaller GPT-3 variants (Ada, Babbage, Curie), though we note that subsequent research has revealed limitations of fine-tuning-based defenses. We also find that more flexible models exhibit greater vulnerability to these attacks. Consequently, large models such as GPT-3 Davinci are more vulnerable than smaller models like GPT-2. While the specific models tested are now superseded, the core methodology and empirical findings contributed to the foundation of modern prompt injection defense research, including instruction hierarchy systems and constitutional AI approaches.



## **47. How to Beat Nakamoto in the Race**

cs.CR

To be presented at the 2025 ACM Conference on Computer and  Communications Security (CCS)

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2508.16202v2) [paper-pdf](http://arxiv.org/pdf/2508.16202v2)

**Authors**: Shu-Jie Cao, Dongning Guo

**Abstract**: This paper studies proof-of-work Nakamoto consensus protocols under bounded network delays, settling two long-standing questions in blockchain security: What is the most effective attack on block safety under a given block confirmation latency? And what is the resulting probability of safety violation? A Markov decision process (MDP) framework is introduced to precisely characterize the system state (including the blocktree and timings of all blocks mined), the adversary's potential actions, and the state transitions due to the adversarial action and the random block arrival processes. An optimal attack, called bait-and-switch, is proposed and proved to maximize the adversary's chance of violating block safety by "beating Nakamoto in the race". The exact probability of this violation is calculated for any given confirmation depth using Markov chain analysis, offering fresh insights into the interplay of network delay, confirmation rules, and blockchain security.



## **48. Time-Constrained Intelligent Adversaries for Automation Vulnerability Testing: A Multi-Robot Patrol Case Study**

cs.RO

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11971v1) [paper-pdf](http://arxiv.org/pdf/2509.11971v1)

**Authors**: James C. Ward, Alex Bott, Connor York, Edmund R. Hunt

**Abstract**: Simulating hostile attacks of physical autonomous systems can be a useful tool to examine their robustness to attack and inform vulnerability-aware design. In this work, we examine this through the lens of multi-robot patrol, by presenting a machine learning-based adversary model that observes robot patrol behavior in order to attempt to gain undetected access to a secure environment within a limited time duration. Such a model allows for evaluation of a patrol system against a realistic potential adversary, offering insight into future patrol strategy design. We show that our new model outperforms existing baselines, thus providing a more stringent test, and examine its performance against multiple leading decentralized multi-robot patrol strategies.



## **49. NeuroStrike: Neuron-Level Attacks on Aligned LLMs**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11864v1) [paper-pdf](http://arxiv.org/pdf/2509.11864v1)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Maximilian Thang, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Safety alignment is critical for the ethical deployment of large language models (LLMs), guiding them to avoid generating harmful or unethical content. Current alignment techniques, such as supervised fine-tuning and reinforcement learning from human feedback, remain fragile and can be bypassed by carefully crafted adversarial prompts. Unfortunately, such attacks rely on trial and error, lack generalizability across models, and are constrained by scalability and reliability.   This paper presents NeuroStrike, a novel and generalizable attack framework that exploits a fundamental vulnerability introduced by alignment techniques: the reliance on sparse, specialized safety neurons responsible for detecting and suppressing harmful inputs. We apply NeuroStrike to both white-box and black-box settings: In the white-box setting, NeuroStrike identifies safety neurons through feedforward activation analysis and prunes them during inference to disable safety mechanisms. In the black-box setting, we propose the first LLM profiling attack, which leverages safety neuron transferability by training adversarial prompt generators on open-weight surrogate models and then deploying them against black-box and proprietary targets. We evaluate NeuroStrike on over 20 open-weight LLMs from major LLM developers. By removing less than 0.6% of neurons in targeted layers, NeuroStrike achieves an average attack success rate (ASR) of 76.9% using only vanilla malicious prompts. Moreover, Neurostrike generalizes to four multimodal LLMs with 100% ASR on unsafe image inputs. Safety neurons transfer effectively across architectures, raising ASR to 78.5% on 11 fine-tuned models and 77.7% on five distilled models. The black-box LLM profiling attack achieves an average ASR of 63.7% across five black-box models, including the Google Gemini family.



## **50. A Practical Adversarial Attack against Sequence-based Deep Learning Malware Classifiers**

cs.CR

**SubmitDate**: 2025-09-15    [abs](http://arxiv.org/abs/2509.11836v1) [paper-pdf](http://arxiv.org/pdf/2509.11836v1)

**Authors**: Kai Tan, Dongyang Zhan, Lin Ye, Hongli Zhang, Binxing Fang

**Abstract**: Sequence-based deep learning models (e.g., RNNs), can detect malware by analyzing its behavioral sequences. Meanwhile, these models are susceptible to adversarial attacks. Attackers can create adversarial samples that alter the sequence characteristics of behavior sequences to deceive malware classifiers. The existing methods for generating adversarial samples typically involve deleting or replacing crucial behaviors in the original data sequences, or inserting benign behaviors that may violate the behavior constraints. However, these methods that directly manipulate sequences make adversarial samples difficult to implement or apply in practice. In this paper, we propose an adversarial attack approach based on Deep Q-Network and a heuristic backtracking search strategy, which can generate perturbation sequences that satisfy practical conditions for successful attacks. Subsequently, we utilize a novel transformation approach that maps modifications back to the source code, thereby avoiding the need to directly modify the behavior log sequences. We conduct an evaluation of our approach, and the results confirm its effectiveness in generating adversarial samples from real-world malware behavior sequences, which have a high success rate in evading anomaly detection models. Furthermore, our approach is practical and can generate adversarial samples while maintaining the functionality of the modified software.



