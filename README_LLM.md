# Latest Large Language Model Attack Papers
**update at 2025-10-20 09:07:43**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. A Framework for Rapidly Developing and Deploying Protection Against Large Language Model Attacks**

cs.CR

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2509.20639v2) [paper-pdf](http://arxiv.org/pdf/2509.20639v2)

**Authors**: Adam Swanda, Amy Chang, Alexander Chen, Fraser Burch, Paul Kassianik, Konstantin Berlin

**Abstract**: The widespread adoption of Large Language Models (LLMs) has revolutionized AI deployment, enabling autonomous and semi-autonomous applications across industries through intuitive language interfaces and continuous improvements in model development. However, the attendant increase in autonomy and expansion of access permissions among AI applications also make these systems compelling targets for malicious attacks. Their inherent susceptibility to security flaws necessitates robust defenses, yet no known approaches can prevent zero-day or novel attacks against LLMs. This places AI protection systems in a category similar to established malware protection systems: rather than providing guaranteed immunity, they minimize risk through enhanced observability, multi-layered defense, and rapid threat response, supported by a threat intelligence function designed specifically for AI-related threats.   Prior work on LLM protection has largely evaluated individual detection models rather than end-to-end systems designed for continuous, rapid adaptation to a changing threat landscape. We present a production-grade defense system rooted in established malware detection and threat intelligence practices. Our platform integrates three components: a threat intelligence system that turns emerging threats into protections; a data platform that aggregates and enriches information while providing observability, monitoring, and ML operations; and a release platform enabling safe, rapid detection updates without disrupting customer workflows. Together, these components deliver layered protection against evolving LLM threats while generating training data for continuous model improvement and deploying updates without interrupting production.



## **2. MalCVE: Malware Detection and CVE Association Using Large Language Models**

cs.CR

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.15567v1) [paper-pdf](http://arxiv.org/pdf/2510.15567v1)

**Authors**: Eduard Andrei Cristea, Petter Molnes, Jingyue Li

**Abstract**: Malicious software attacks are having an increasingly significant economic impact. Commercial malware detection software can be costly, and tools that attribute malware to the specific software vulnerabilities it exploits are largely lacking. Understanding the connection between malware and the vulnerabilities it targets is crucial for analyzing past threats and proactively defending against current ones. In this study, we propose an approach that leverages large language models (LLMs) to detect binary malware, specifically within JAR files, and utilizes the capabilities of LLMs combined with retrieval-augmented generation (RAG) to identify Common Vulnerabilities and Exposures (CVEs) that malware may exploit. We developed a proof-of-concept tool called MalCVE, which integrates binary code decompilation, deobfuscation, LLM-based code summarization, semantic similarity search, and CVE classification using LLMs. We evaluated MalCVE using a benchmark dataset of 3,839 JAR executables. MalCVE achieved a mean malware detection accuracy of 97%, at a fraction of the cost of commercial solutions. It is also the first tool to associate CVEs with binary malware, achieving a recall@10 of 65%, which is comparable to studies that perform similar analyses on source code.



## **3. SoK: Taxonomy and Evaluation of Prompt Security in Large Language Models**

cs.CR

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.15476v1) [paper-pdf](http://arxiv.org/pdf/2510.15476v1)

**Authors**: Hanbin Hong, Shuya Feng, Nima Naderloui, Shenao Yan, Jingyu Zhang, Biying Liu, Ali Arastehfard, Heqing Huang, Yuan Hong

**Abstract**: Large Language Models (LLMs) have rapidly become integral to real-world applications, powering services across diverse sectors. However, their widespread deployment has exposed critical security risks, particularly through jailbreak prompts that can bypass model alignment and induce harmful outputs. Despite intense research into both attack and defense techniques, the field remains fragmented: definitions, threat models, and evaluation criteria vary widely, impeding systematic progress and fair comparison. In this Systematization of Knowledge (SoK), we address these challenges by (1) proposing a holistic, multi-level taxonomy that organizes attacks, defenses, and vulnerabilities in LLM prompt security; (2) formalizing threat models and cost assumptions into machine-readable profiles for reproducible evaluation; (3) introducing an open-source evaluation toolkit for standardized, auditable comparison of attacks and defenses; (4) releasing JAILBREAKDB, the largest annotated dataset of jailbreak and benign prompts to date; and (5) presenting a comprehensive evaluation and leaderboard of state-of-the-art methods. Our work unifies fragmented research, provides rigorous foundations for future studies, and supports the development of robust, trustworthy LLMs suitable for high-stakes deployment.



## **4. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models**

cs.CV

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.15430v1) [paper-pdf](http://arxiv.org/pdf/2510.15430v1)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.



## **5. DSSmoothing: Toward Certified Dataset Ownership Verification for Pre-trained Language Models via Dual-Space Smoothing**

cs.CR

13 pages, 21 figures

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.15303v1) [paper-pdf](http://arxiv.org/pdf/2510.15303v1)

**Authors**: Ting Qiao, Xing Liu, Wenke Huang, Jianbin Li, Zhaoxin Fan, Yiming Li

**Abstract**: Large web-scale datasets have driven the rapid advancement of pre-trained language models (PLMs), but unauthorized data usage has raised serious copyright concerns. Existing dataset ownership verification (DOV) methods typically assume that watermarks remain stable during inference; however, this assumption often fails under natural noise and adversary-crafted perturbations. We propose the first certified dataset ownership verification method for PLMs based on dual-space smoothing (i.e., DSSmoothing). To address the challenges of text discreteness and semantic sensitivity, DSSmoothing introduces continuous perturbations in the embedding space to capture semantic robustness and applies controlled token reordering in the permutation space to capture sequential robustness. DSSmoothing consists of two stages: in the first stage, triggers are collaboratively embedded in both spaces to generate norm-constrained and robust watermarked datasets; in the second stage, randomized smoothing is applied in both spaces during verification to compute the watermark robustness (WR) of suspicious models and statistically compare it with the principal probability (PP) values of a set of benign models. Theoretically, DSSmoothing provides provable robustness guarantees for dataset ownership verification by ensuring that WR consistently exceeds PP under bounded dual-space perturbations. Extensive experiments on multiple representative web datasets demonstrate that DSSmoothing achieves stable and reliable verification performance and exhibits robustness against potential adaptive attacks.



## **6. PromptLocate: Localizing Prompt Injection Attacks**

cs.CR

To appear in IEEE Symposium on Security and Privacy, 2026. For  slides, see https://people.duke.edu/~zg70/code/PromptInjection.pdf

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2510.12252v2) [paper-pdf](http://arxiv.org/pdf/2510.12252v2)

**Authors**: Yuqi Jia, Yupei Liu, Zedian Shao, Jinyuan Jia, Neil Gong

**Abstract**: Prompt injection attacks deceive a large language model into completing an attacker-specified task instead of its intended task by contaminating its input data with an injected prompt, which consists of injected instruction(s) and data. Localizing the injected prompt within contaminated data is crucial for post-attack forensic analysis and data recovery. Despite its growing importance, prompt injection localization remains largely unexplored. In this work, we bridge this gap by proposing PromptLocate, the first method for localizing injected prompts. PromptLocate comprises three steps: (1) splitting the contaminated data into semantically coherent segments, (2) identifying segments contaminated by injected instructions, and (3) pinpointing segments contaminated by injected data. We show PromptLocate accurately localizes injected prompts across eight existing and eight adaptive attacks.



## **7. WebInject: Prompt Injection Attack to Web Agents**

cs.LG

Appeared in EMNLP 2025 main conference. To better understand prompt  injection attacks, see https://people.duke.edu/~zg70/code/PromptInjection.pdf

**SubmitDate**: 2025-10-17    [abs](http://arxiv.org/abs/2505.11717v4) [paper-pdf](http://arxiv.org/pdf/2505.11717v4)

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.



## **8. OCR-APT: Reconstructing APT Stories from Audit Logs using Subgraph Anomaly Detection and LLMs**

cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.15188v1) [paper-pdf](http://arxiv.org/pdf/2510.15188v1)

**Authors**: Ahmed Aly, Essam Mansour, Amr Youssef

**Abstract**: Advanced Persistent Threats (APTs) are stealthy cyberattacks that often evade detection in system-level audit logs. Provenance graphs model these logs as connected entities and events, revealing relationships that are missed by linear log representations. Existing systems apply anomaly detection to these graphs but often suffer from high false positive rates and coarse-grained alerts. Their reliance on node attributes like file paths or IPs leads to spurious correlations, reducing detection robustness and reliability. To fully understand an attack's progression and impact, security analysts need systems that can generate accurate, human-like narratives of the entire attack. To address these challenges, we introduce OCR-APT, a system for APT detection and reconstruction of human-like attack stories. OCR-APT uses Graph Neural Networks (GNNs) for subgraph anomaly detection, learning behavior patterns around nodes rather than fragile attributes such as file paths or IPs. This approach leads to a more robust anomaly detection. It then iterates over detected subgraphs using Large Language Models (LLMs) to reconstruct multi-stage attack stories. Each stage is validated before proceeding, reducing hallucinations and ensuring an interpretable final report. Our evaluations on the DARPA TC3, OpTC, and NODLINK datasets show that OCR-APT outperforms state-of-the-art systems in both detection accuracy and alert interpretability. Moreover, OCR-APT reconstructs human-like reports that comprehensively capture the attack story.



## **9. PoTS: Proof-of-Training-Steps for Backdoor Detection in Large Language Models**

cs.CR

10 pages, 6 figures, 1 table. Accepted for presentation at FLLM 2025  (Vienna, Nov 2025)

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.15106v1) [paper-pdf](http://arxiv.org/pdf/2510.15106v1)

**Authors**: Issam Seddik, Sami Souihi, Mohamed Tamaazousti, Sara Tucci Piergiovanni

**Abstract**: As Large Language Models (LLMs) gain traction across critical domains, ensuring secure and trustworthy training processes has become a major concern. Backdoor attacks, where malicious actors inject hidden triggers into training data, are particularly insidious and difficult to detect. Existing post-training verification solutions like Proof-of-Learning are impractical for LLMs due to their requirement for full retraining, lack of robustness against stealthy manipulations, and inability to provide early detection during training. Early detection would significantly reduce computational costs. To address these limitations, we introduce Proof-of-Training Steps, a verification protocol that enables an independent auditor (Alice) to confirm that an LLM developer (Bob) has followed the declared training recipe, including data batches, architecture, and hyperparameters. By analyzing the sensitivity of the LLMs' language modeling head (LM-Head) to input perturbations, our method can expose subtle backdoor injections or deviations in training. Even with backdoor triggers in up to 10 percent of the training data, our protocol significantly reduces the attacker's ability to achieve a high attack success rate (ASR). Our method enables early detection of attacks at the injection step, with verification steps being 3x faster than training steps. Our results highlight the protocol's potential to enhance the accountability and security of LLM development, especially against insider threats.



## **10. Sequential Comics for Jailbreaking Multimodal Large Language Models via Structured Visual Storytelling**

cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.15068v1) [paper-pdf](http://arxiv.org/pdf/2510.15068v1)

**Authors**: Deyue Zhang, Dongdong Yang, Junjie Mu, Quancheng Zou, Zonghao Ying, Wenzhuo Xu, Zhao Liu, Xuan Wang, Xiangzheng Zhang

**Abstract**: Multimodal large language models (MLLMs) exhibit remarkable capabilities but remain susceptible to jailbreak attacks exploiting cross-modal vulnerabilities. In this work, we introduce a novel method that leverages sequential comic-style visual narratives to circumvent safety alignments in state-of-the-art MLLMs. Our method decomposes malicious queries into visually innocuous storytelling elements using an auxiliary LLM, generates corresponding image sequences through diffusion models, and exploits the models' reliance on narrative coherence to elicit harmful outputs. Extensive experiments on harmful textual queries from established safety benchmarks show that our approach achieves an average attack success rate of 83.5\%, surpassing prior state-of-the-art by 46\%. Compared with existing visual jailbreak methods, our sequential narrative strategy demonstrates superior effectiveness across diverse categories of harmful content. We further analyze attack patterns, uncover key vulnerability factors in multimodal safety mechanisms, and evaluate the limitations of current defense strategies against narrative-driven attacks, revealing significant gaps in existing protections.



## **11. Keep Calm and Avoid Harmful Content: Concept Alignment and Latent Manipulation Towards Safer Answers**

cs.LG

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.12672v2) [paper-pdf](http://arxiv.org/pdf/2510.12672v2)

**Authors**: Ruben Belo, Marta Guimaraes, Claudia Soares

**Abstract**: Large Language Models are susceptible to jailbreak attacks that bypass built-in safety guardrails (e.g., by tricking the model with adversarial prompts). We propose Concept Alignment and Concept Manipulation CALM, an inference-time method that suppresses harmful concepts by modifying latent representations of the last layer of the model, without retraining. Leveraging concept whitening technique from Computer Vision combined with orthogonal projection, CALM removes unwanted latent directions associated with harmful content while preserving model performance. Experiments show that CALM reduces harmful outputs and outperforms baseline methods in most metrics, offering a lightweight approach to AI safety with no additional training data or model fine-tuning, while incurring only a small computational overhead at inference.



## **12. Active Honeypot Guardrail System: Probing and Confirming Multi-Turn LLM Jailbreaks**

cs.CR

6pages, 2 figures

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.15017v1) [paper-pdf](http://arxiv.org/pdf/2510.15017v1)

**Authors**: ChenYu Wu, Yi Wang, Yang Liao

**Abstract**: Large language models (LLMs) are increasingly vulnerable to multi-turn jailbreak attacks, where adversaries iteratively elicit harmful behaviors that bypass single-turn safety filters. Existing defenses predominantly rely on passive rejection, which either fails against adaptive attackers or overly restricts benign users. We propose a honeypot-based proactive guardrail system that transforms risk avoidance into risk utilization. Our framework fine-tunes a bait model to generate ambiguous, non-actionable but semantically relevant responses, which serve as lures to probe user intent. Combined with the protected LLM's safe reply, the system inserts proactive bait questions that gradually expose malicious intent through multi-turn interactions. We further introduce the Honeypot Utility Score (HUS), measuring both the attractiveness and feasibility of bait responses, and use a Defense Efficacy Rate (DER) for balancing safety and usability. Initial experiment on MHJ Datasets with recent attack method across GPT-4o show that our system significantly disrupts jailbreak success while preserving benign user experience.



## **13. Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs**

cs.LG

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.03567v3) [paper-pdf](http://arxiv.org/pdf/2510.03567v3)

**Authors**: Fatmazohra Rezkellah, Ramzi Dakhmouche

**Abstract**: With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach.



## **14. Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge**

cs.CL

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2504.07887v2) [paper-pdf](http://arxiv.org/pdf/2504.07887v2)

**Authors**: Riccardo Cantini, Alessio Orsino, Massimo Ruggiero, Domenico Talia

**Abstract**: The growing integration of Large Language Models (LLMs) into critical societal domains has raised concerns about embedded biases that can perpetuate stereotypes and undermine fairness. Such biases may stem from historical inequalities in training data, linguistic imbalances, or adversarial manipulation. Despite mitigation efforts, recent studies show that LLMs remain vulnerable to adversarial attacks that elicit biased outputs. This work proposes a scalable benchmarking framework to assess LLM robustness to adversarial bias elicitation. Our methodology involves: (i) systematically probing models across multiple tasks targeting diverse sociocultural biases, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach, and (iii) employing jailbreak techniques to reveal safety vulnerabilities. To facilitate systematic benchmarking, we release a curated dataset of bias-related prompts, named CLEAR-Bias. Our analysis, identifying DeepSeek V3 as the most reliable judge LLM, reveals that bias resilience is uneven, with age, disability, and intersectional biases among the most prominent. Some small models outperform larger ones in safety, suggesting that training and architecture may matter more than scale. However, no model is fully robust to adversarial elicitation, with jailbreak attacks using low-resource languages or refusal suppression proving effective across model families. We also find that successive LLM generations exhibit slight safety gains, while models fine-tuned for the medical domain tend to be less safe than their general-purpose counterparts.



## **15. ATGen: Adversarial Reinforcement Learning for Test Case Generation**

cs.SE

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14635v1) [paper-pdf](http://arxiv.org/pdf/2510.14635v1)

**Authors**: Qingyao Li, Xinyi Dai, Weiwen Liu, Xiangyang Li, Yasheng Wang, Ruiming Tang, Yong Yu, Weinan Zhang

**Abstract**: Large Language Models (LLMs) excel at code generation, yet their outputs often contain subtle bugs, for which effective test cases are a critical bottleneck. Existing test generation methods, whether based on prompting or supervised fine-tuning, rely on static datasets. This imposes a ``fixed-difficulty ceiling'', fundamentally limiting their ability to uncover novel or more complex bugs beyond their training scope. To overcome this, we introduce ATGen, a framework that trains a test case generator via adversarial reinforcement learning. ATGen pits a test generator against an adversarial code generator that continuously crafts harder bugs to evade the current policy. This dynamic loop creates a curriculum of increasing difficulty challenging current policy. The test generator is optimized via Reinforcement Learning (RL) to jointly maximize ``Output Accuracy'' and ``Attack Success'', enabling it to learn a progressively stronger policy that breaks the fixed-difficulty ceiling of static training. Extensive experiments demonstrate that ATGen significantly outperforms state-of-the-art baselines. We further validate its practical utility, showing it serves as both a more effective filter for Best-of-N inference and a higher-quality reward source for training code generation models. Our work establishes a new, dynamic paradigm for improving the reliability of LLM-generated code.



## **16. Checkpoint-GCG: Auditing and Attacking Fine-Tuning-Based Prompt Injection Defenses**

cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2505.15738v2) [paper-pdf](http://arxiv.org/pdf/2505.15738v2)

**Authors**: Xiaoxue Yang, Bozhidar Stevanoski, Matthieu Meeus, Yves-Alexandre de Montjoye

**Abstract**: Large language models (LLMs) are increasingly deployed in real-world applications ranging from chatbots to agentic systems, where they are expected to process untrusted data and follow trusted instructions. Failure to distinguish between the two poses significant security risks, exploited by prompt injection attacks, which inject malicious instructions into the data to control model outputs. Model-level defenses have been proposed to mitigate prompt injection attacks. These defenses fine-tune LLMs to ignore injected instructions in untrusted data. We introduce Checkpoint-GCG, a white-box attack against fine-tuning-based defenses. Checkpoint-GCG enhances the Greedy Coordinate Gradient (GCG) attack by leveraging intermediate model checkpoints produced during fine-tuning to initialize GCG, with each checkpoint acting as a stepping stone for the next one to continuously improve attacks. First, we instantiate Checkpoint-GCG to evaluate the robustness of the state-of-the-art defenses in an auditing setup, assuming both (a) full knowledge of the model input and (b) access to intermediate model checkpoints. We show Checkpoint-GCG to achieve up to $96\%$ attack success rate (ASR) against the strongest defense. Second, we relax the first assumption by searching for a universal suffix that would work on unseen inputs, and obtain up to $89.9\%$ ASR against the strongest defense. Finally, we relax both assumptions by searching for a universal suffix that would transfer to similar black-box models and defenses, achieving an ASR of $63.9\%$ against a newly released defended model from Meta.



## **17. SoK: Evaluating Jailbreak Guardrails for Large Language Models**

cs.CR

Accepted by IEEE S&P 2026 Cycle 1

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2506.10597v2) [paper-pdf](http://arxiv.org/pdf/2506.10597v2)

**Authors**: Xunguang Wang, Zhenlan Ji, Wenxuan Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress, but their deployment has exposed critical vulnerabilities, particularly to jailbreak attacks that circumvent safety alignments. Guardrails--external defense mechanisms that monitor and control LLM interactions--have emerged as a promising solution. However, the current landscape of LLM guardrails is fragmented, lacking a unified taxonomy and comprehensive evaluation framework. In this Systematization of Knowledge (SoK) paper, we present the first holistic analysis of jailbreak guardrails for LLMs. We propose a novel, multi-dimensional taxonomy that categorizes guardrails along six key dimensions, and introduce a Security-Efficiency-Utility evaluation framework to assess their practical effectiveness. Through extensive analysis and experiments, we identify the strengths and limitations of existing guardrail approaches, provide insights into optimizing their defense mechanisms, and explore their universality across attack types. Our work offers a structured foundation for future research and development, aiming to guide the principled advancement and deployment of robust LLM guardrails. The code is available at https://github.com/xunguangwang/SoK4JailbreakGuardrails.



## **18. SPIRIT: Patching Speech Language Models against Jailbreak Attacks**

eess.AS

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2505.13541v2) [paper-pdf](http://arxiv.org/pdf/2505.13541v2)

**Authors**: Amirbek Djanibekov, Nurdaulet Mukhituly, Kentaro Inui, Hanan Aldarmaki, Nils Lukas

**Abstract**: Speech Language Models (SLMs) enable natural interactions via spoken instructions, which more effectively capture user intent by detecting nuances in speech. The richer speech signal introduces new security risks compared to text-based models, as adversaries can better bypass safety mechanisms by injecting imperceptible noise to speech. We analyze adversarial attacks and find that SLMs are substantially more vulnerable to jailbreak attacks, which can achieve a perfect 100% attack success rate in some instances. To improve security, we propose post-hoc patching defenses used to intervene during inference by modifying the SLM's activations that improve robustness up to 99% with (i) negligible impact on utility and (ii) without any re-training. We conduct ablation studies to maximize the efficacy of our defenses and improve the utility/security trade-off, validated with large-scale benchmarks unique to SLMs.



## **19. Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration**

cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14522v1) [paper-pdf](http://arxiv.org/pdf/2510.14522v1)

**Authors**: Evangelos Lamprou, Julian Dai, Grigoris Ntousakis, Martin C. Rinard, Nikos Vasilakis

**Abstract**: Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. Our evaluation on 100+ real-world packages, including high profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<100s on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when prompted to do so.



## **20. Are My Optimized Prompts Compromised? Exploring Vulnerabilities of LLM-based Optimizers**

cs.LG

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14381v1) [paper-pdf](http://arxiv.org/pdf/2510.14381v1)

**Authors**: Andrew Zhao, Reshmi Ghosh, Vitor Carvalho, Emily Lawton, Keegan Hines, Gao Huang, Jack W. Stokes

**Abstract**: Large language model (LLM) systems now underpin everyday AI applications such as chatbots, computer-use assistants, and autonomous robots, where performance often depends on carefully designed prompts. LLM-based prompt optimizers reduce that effort by iteratively refining prompts from scored feedback, yet the security of this optimization stage remains underexamined. We present the first systematic analysis of poisoning risks in LLM-based prompt optimization. Using HarmBench, we find systems are substantially more vulnerable to manipulated feedback than to injected queries: feedback-based attacks raise attack success rate (ASR) by up to $\Delta$ASR = 0.48. We introduce a simple fake-reward attack that requires no access to the reward model and significantly increases vulnerability, and we propose a lightweight highlighting defense that reduces the fake-reward $\Delta$ASR from 0.23 to 0.07 without degrading utility. These results establish prompt optimization pipelines as a first-class attack surface and motivate stronger safeguards for feedback channels and optimization frameworks.



## **21. CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment**

cs.CR

Accepted by NeurIPS 2025 Conference

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2410.13903v2) [paper-pdf](http://arxiv.org/pdf/2410.13903v2)

**Authors**: Qinfeng Li, Tianyue Luo, Xuhong Zhang, Yangfan Xie, Zhiqiang Shen, Lijun Zhang, Yier Jin, Hao Peng, Xinkui Zhao, Xianwei Zhu, Jianwei Yin

**Abstract**: Proprietary large language models (LLMs) exhibit strong generalization capabilities across diverse tasks and are increasingly deployed on edge devices for efficiency and privacy reasons. However, deploying proprietary LLMs at the edge without adequate protection introduces critical security threats. Attackers can extract model weights and architectures, enabling unauthorized copying and misuse. Even when protective measures prevent full extraction of model weights, attackers may still perform advanced attacks, such as fine-tuning, to further exploit the model. Existing defenses against these threats typically incur significant computational and communication overhead, making them impractical for edge deployment. To safeguard the edge-deployed LLMs, we introduce CoreGuard, a computation- and communication-efficient protection method. CoreGuard employs an efficient protection protocol to reduce computational overhead and minimize communication overhead via a propagation protocol. Extensive experiments show that CoreGuard achieves upper-bound security protection with negligible overhead.



## **22. When Style Breaks Safety: Defending LLMs Against Superficial Style Alignment**

cs.LG

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2506.07452v2) [paper-pdf](http://arxiv.org/pdf/2506.07452v2)

**Authors**: Yuxin Xiao, Sana Tonekaboni, Walter Gerych, Vinith Suriyakumar, Marzyeh Ghassemi

**Abstract**: Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in malicious queries. Prior jailbreak research mainly augments these queries with additional string transformations to maximize attack success rate (ASR). However, the impact of style patterns in the original queries that are semantically irrelevant to the malicious intent remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We first define ASR inflation as the increase in ASR due to style patterns in existing jailbreak benchmark queries. By evaluating 32 LLMs across seven benchmarks, we find that nearly all models exhibit ASR inflation. Notably, the inflation correlates with an LLM's relative attention to style patterns, which also overlap more with its instruction-tuning data when inflation occurs. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs, six fine-tuning style settings, and two real-world instruction-tuning datasets, SafeStyle consistently outperforms baselines in maintaining LLM safety.



## **23. Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security Studies**

cs.AI

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14312v1) [paper-pdf](http://arxiv.org/pdf/2510.14312v1)

**Authors**: Mason Nakamura, Abhinav Kumar, Saaduddin Mahmud, Sahar Abdelnabi, Shlomo Zilberstein, Eugene Bagdasarian

**Abstract**: A multi-agent system (MAS) powered by large language models (LLMs) can automate tedious user tasks such as meeting scheduling that requires inter-agent collaboration. LLMs enable nuanced protocols that account for unstructured private data, user constraints, and preferences. However, this design introduces new risks, including misalignment and attacks by malicious parties that compromise agents or steal user data. In this paper, we propose the Terrarium framework for fine-grained study on safety, privacy, and security in LLM-based MAS. We repurpose the blackboard design, an early approach in multi-agent systems, to create a modular, configurable testbed for multi-agent collaboration. We identify key attack vectors such as misalignment, malicious agents, compromised communication, and data poisoning. We implement three collaborative MAS scenarios with four representative attacks to demonstrate the framework's flexibility. By providing tools to rapidly prototype, evaluate, and iterate on defenses and designs, Terrarium aims to accelerate progress toward trustworthy multi-agent systems.



## **24. RHINO: Guided Reasoning for Mapping Network Logs to Adversarial Tactics and Techniques with Large Language Models**

cs.CR

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14233v1) [paper-pdf](http://arxiv.org/pdf/2510.14233v1)

**Authors**: Fanchao Meng, Jiaping Gui, Yunbo Li, Yue Wu

**Abstract**: Modern Network Intrusion Detection Systems generate vast volumes of low-level alerts, yet these outputs remain semantically fragmented, requiring labor-intensive manual correlation with high-level adversarial behaviors. Existing solutions for automating this mapping-rule-based systems and machine learning classifiers-suffer from critical limitations: rule-based approaches fail to adapt to novel attack variations, while machine learning methods lack contextual awareness and treat tactic-technique mapping as a syntactic matching problem rather than a reasoning task. Although Large Language Models have shown promise in cybersecurity tasks, preliminary experiments reveal that existing LLM-based methods frequently hallucinate technique names or produce decontextualized mappings due to their single-step classification approach.   To address these challenges, we introduce RHINO, a novel framework that decomposes LLM-based attack analysis into three interpretable phases mirroring human reasoning: (1) behavioral abstraction, where raw logs are translated into contextualized narratives; (2) multi-role collaborative inference, generating candidate techniques by evaluating behavioral evidence against MITRE ATT&CK knowledge; and (3) validation, cross-referencing predictions with official MITRE definitions to rectify hallucinations. RHINO bridges the semantic gap between low-level observations and adversarial intent while improving output reliability through structured reasoning.   We evaluate RHINO on three benchmarks across four backbone models. RHINO achieved high accuracy, with model performance ranging from 86.38% to 88.45%, resulting in relative gains from 24.25% to 76.50% across different models. Our results demonstrate that RHINO significantly enhances the interpretability and scalability of threat analysis, offering a blueprint for deploying LLMs in operational security settings.



## **25. Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks**

cs.AI

13 pages, 4 figures

**SubmitDate**: 2025-10-16    [abs](http://arxiv.org/abs/2510.14207v1) [paper-pdf](http://arxiv.org/pdf/2510.14207v1)

**Authors**: Trilok Padhi, Pinxian Lu, Abdulkadir Erol, Tanmay Sutar, Gauri Sharma, Mina Sonmez, Munmun De Choudhury, Ugur Kursuncu

**Abstract**: Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible.



## **26. One Bug, Hundreds Behind: LLMs for Large-Scale Bug Discovery**

cs.SE

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.14036v1) [paper-pdf](http://arxiv.org/pdf/2510.14036v1)

**Authors**: Qiushi Wu, Yue Xiao, Dhilung Kirat, Kevin Eykholt, Jiyong Jang, Douglas Lee Schales

**Abstract**: Fixing bugs in large programs is a challenging task that demands substantial time and effort. Once a bug is found, it is reported to the project maintainers, who work with the reporter to fix it and eventually close the issue. However, across the program, there are often similar code segments, which may also contain the bug, but were missed during discovery. Finding and fixing each recurring bug instance individually is labor intensive. Even more concerning, bug reports can inadvertently widen the attack surface as they provide attackers with an exploitable pattern that may be unresolved in other parts of the program.   In this paper, we explore these Recurring Pattern Bugs (RPBs) that appear repeatedly across various code segments of a program or even in different programs, stemming from a same root cause, but are unresolved. Our investigation reveals that RPBs are widespread and can significantly compromise the security of software programs. This paper introduces BugStone, a program analysis system empowered by LLVM and a Large Language Model (LLM). The key observation is that many RPBs have one patched instance, which can be leveraged to identify a consistent error pattern, such as a specific API misuse. By examining the entire program for this pattern, it is possible to identify similar sections of code that may be vulnerable. Starting with 135 unique RPBs, BugStone identified more than 22K new potential issues in the Linux kernel. Manual analysis of 400 of these findings confirmed that 246 were valid. We also created a dataset from over 1.9K security bugs reported by 23 recent top-tier conference works. We manually annotate the dataset, identify 80 recurring patterns and 850 corresponding fixes. Even with a cost-efficient model choice, BugStone achieved 92.2% precision and 79.1% pairwise accuracy on the dataset.



## **27. Signature in Code Backdoor Detection, how far are we?**

cs.SE

20 pages, 3 figures

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13992v1) [paper-pdf](http://arxiv.org/pdf/2510.13992v1)

**Authors**: Quoc Hung Le, Thanh Le-Cong, Bach Le, Bowen Xu

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into software development workflows, they also become prime targets for adversarial attacks. Among these, backdoor attacks are a significant threat, allowing attackers to manipulate model outputs through hidden triggers embedded in training data. Detecting such backdoors remains a challenge, and one promising approach is the use of Spectral Signature defense methods that identify poisoned data by analyzing feature representations through eigenvectors. While some prior works have explored Spectral Signatures for backdoor detection in neural networks, recent studies suggest that these methods may not be optimally effective for code models. In this paper, we revisit the applicability of Spectral Signature-based defenses in the context of backdoor attacks on code models. We systematically evaluate their effectiveness under various attack scenarios and defense configurations, analyzing their strengths and limitations. We found that the widely used setting of Spectral Signature in code backdoor detection is often suboptimal. Hence, we explored the impact of different settings of the key factors. We discovered a new proxy metric that can more accurately estimate the actual performance of Spectral Signature without model retraining after the defense.



## **28. LLM-Enabled In-Context Learning for Data Collection Scheduling in UAV-assisted Sensor Networks**

cs.AI

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2504.14556v2) [paper-pdf](http://arxiv.org/pdf/2504.14556v2)

**Authors**: Yousef Emami, Hao Zhou, SeyedSina Nabavirazani, Luis Almeida

**Abstract**: Unmanned Aerial Vehicles (UAVs) are increasingly being utilized in various private and commercial applications, e.g., traffic control, parcel delivery, and Search and Rescue (SAR) missions. Machine Learning (ML) methods used in UAV-Assisted Sensor Networks (UASNETs) and, especially, in Deep Reinforcement Learning (DRL) face challenges such as complex and lengthy model training, gaps between simulation and reality, and low sampling efficiency, which conflict with the urgency of emergencies, such as SAR missions. In this paper, an In-Context Learning (ICL)-Data Collection Scheduling (ICLDC) system is proposed as an alternative to DRL in emergencies. The UAV collects sensory data and transmits it to a Large Language Model (LLM), which creates a task description in natural language. From this description, the UAV receives a data collection schedule that must be executed. A verifier ensures safe UAV operations by evaluating the schedules generated by the LLM and overriding unsafe schedules based on predefined rules. The system continuously adapts by incorporating feedback into the task descriptions and using this for future decisions. This method is tested against jailbreaking attacks, where the task description is manipulated to undermine network performance, highlighting the vulnerability of LLMs to such attacks. The proposed ICLDC significantly reduces cumulative packet loss compared to both the DQN and Maximum Channel Gain baselines. ICLDC presents a promising direction for intelligent scheduling and control in UASNETs.



## **29. In-Browser LLM-Guided Fuzzing for Real-Time Prompt Injection Testing in Agentic AI Browsers**

cs.CR

37 pages , 10 figures

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13543v1) [paper-pdf](http://arxiv.org/pdf/2510.13543v1)

**Authors**: Avihay Cohen

**Abstract**: Large Language Model (LLM) based agents integrated into web browsers (often called agentic AI browsers) offer powerful automation of web tasks. However, they are vulnerable to indirect prompt injection attacks, where malicious instructions hidden in a webpage deceive the agent into unwanted actions. These attacks can bypass traditional web security boundaries, as the AI agent operates with the user privileges across sites. In this paper, we present a novel fuzzing framework that runs entirely in the browser and is guided by an LLM to automatically discover such prompt injection vulnerabilities in real time.



## **30. Who Speaks for the Trigger? Dynamic Expert Routing in Backdoored Mixture-of-Experts Transformers**

cs.CR

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13462v1) [paper-pdf](http://arxiv.org/pdf/2510.13462v1)

**Authors**: Xin Zhao, Xiaojun Chen, Bingshan Liu, Haoyu Gao, Zhendong Zhao, Yilong Chen

**Abstract**: Large language models (LLMs) with Mixture-of-Experts (MoE) architectures achieve impressive performance and efficiency by dynamically routing inputs to specialized subnetworks, known as experts. However, this sparse routing mechanism inherently exhibits task preferences due to expert specialization, introducing a new and underexplored vulnerability to backdoor attacks. In this work, we investigate the feasibility and effectiveness of injecting backdoors into MoE-based LLMs by exploiting their inherent expert routing preferences. We thus propose BadSwitch, a novel backdoor framework that integrates task-coupled dynamic trigger optimization with a sensitivity-guided Top-S expert tracing mechanism. Our approach jointly optimizes trigger embeddings during pretraining while identifying S most sensitive experts, subsequently constraining the Top-K gating mechanism to these targeted experts. Unlike traditional backdoor attacks that rely on superficial data poisoning or model editing, BadSwitch primarily embeds malicious triggers into expert routing paths with strong task affinity, enabling precise and stealthy model manipulation. Through comprehensive evaluations across three prominent MoE architectures (Switch Transformer, QwenMoE, and DeepSeekMoE), we demonstrate that BadSwitch can efficiently hijack pre-trained models with up to 100% success rate (ASR) while maintaining the highest clean accuracy (ACC) among all baselines. Furthermore, BadSwitch exhibits strong resilience against both text-level and model-level defense mechanisms, achieving 94.07% ASR and 87.18% ACC on the AGNews dataset. Our analysis of expert activation patterns reveals fundamental insights into MoE vulnerabilities. We anticipate this work will expose security risks in MoE systems and contribute to advancing AI safety.



## **31. Can an Individual Manipulate the Collective Decisions of Multi-Agents?**

cs.CL

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2509.16494v2) [paper-pdf](http://arxiv.org/pdf/2509.16494v2)

**Authors**: Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, Jindong Gu

**Abstract**: Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies.



## **32. SeCon-RAG: A Two-Stage Semantic Filtering and Conflict-Free Framework for Trustworthy RAG**

cs.CL

Accepted at NeurIPS 2025

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.09710v2) [paper-pdf](http://arxiv.org/pdf/2510.09710v2)

**Authors**: Xiaonan Si, Meilin Zhu, Simeng Qin, Lijia Yu, Lijun Zhang, Shuaitong Liu, Xinfeng Li, Ranjie Duan, Yang Liu, Xiaojun Jia

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) with external knowledge but are vulnerable to corpus poisoning and contamination attacks, which can compromise output integrity. Existing defenses often apply aggressive filtering, leading to unnecessary loss of valuable information and reduced reliability in generation. To address this problem, we propose a two-stage semantic filtering and conflict-free framework for trustworthy RAG. In the first stage, we perform a joint filter with semantic and cluster-based filtering which is guided by the Entity-intent-relation extractor (EIRE). EIRE extracts entities, latent objectives, and entity relations from both the user query and filtered documents, scores their semantic relevance, and selectively adds valuable documents into the clean retrieval database. In the second stage, we proposed an EIRE-guided conflict-aware filtering module, which analyzes semantic consistency between the query, candidate answers, and retrieved knowledge before final answer generation, filtering out internal and external contradictions that could mislead the model. Through this two-stage process, SeCon-RAG effectively preserves useful knowledge while mitigating conflict contamination, achieving significant improvements in both generation robustness and output trustworthiness. Extensive experiments across various LLMs and datasets demonstrate that the proposed SeCon-RAG markedly outperforms state-of-the-art defense methods.



## **33. SHIELD: Classifier-Guided Prompting for Robust and Safer LVLMs**

cs.CL

Preprint

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13190v1) [paper-pdf](http://arxiv.org/pdf/2510.13190v1)

**Authors**: Juan Ren, Mark Dras, Usman Naseem

**Abstract**: Large Vision-Language Models (LVLMs) unlock powerful multimodal reasoning but also expand the attack surface, particularly through adversarial inputs that conceal harmful goals in benign prompts. We propose SHIELD, a lightweight, model-agnostic preprocessing framework that couples fine-grained safety classification with category-specific guidance and explicit actions (Block, Reframe, Forward). Unlike binary moderators, SHIELD composes tailored safety prompts that enforce nuanced refusals or safe redirection without retraining. Across five benchmarks and five representative LVLMs, SHIELD consistently lowers jailbreak and non-following rates while preserving utility. Our method is plug-and-play, incurs negligible overhead, and is easily extendable to new attack types -- serving as a practical safety patch for both weakly and strongly aligned LVLMs.



## **34. RAID: Refusal-Aware and Integrated Decoding for Jailbreaking LLMs**

cs.CL

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.13901v1) [paper-pdf](http://arxiv.org/pdf/2510.13901v1)

**Authors**: Tuan T. Nguyen, John Le, Thai T. Vu, Willy Susilo, Heath Cooper

**Abstract**: Large language models (LLMs) achieve impressive performance across diverse tasks yet remain vulnerable to jailbreak attacks that bypass safety mechanisms. We present RAID (Refusal-Aware and Integrated Decoding), a framework that systematically probes these weaknesses by crafting adversarial suffixes that induce restricted content while preserving fluency. RAID relaxes discrete tokens into continuous embeddings and optimizes them with a joint objective that (i) encourages restricted responses, (ii) incorporates a refusal-aware regularizer to steer activations away from refusal directions in embedding space, and (iii) applies a coherence term to maintain semantic plausibility and non-redundancy. After optimization, a critic-guided decoding procedure maps embeddings back to tokens by balancing embedding affinity with language-model likelihood. This integration yields suffixes that are both effective in bypassing defenses and natural in form. Experiments on multiple open-source LLMs show that RAID achieves higher attack success rates with fewer queries and lower computational cost than recent white-box and black-box baselines. These findings highlight the importance of embedding-space regularization for understanding and mitigating LLM jailbreak vulnerabilities.



## **35. PEAR: Planner-Executor Agent Robustness Benchmark**

cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.07505v2) [paper-pdf](http://arxiv.org/pdf/2510.07505v2)

**Authors**: Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.



## **36. Guarding the Guardrails: A Taxonomy-Driven Approach to Jailbreak Detection**

cs.CL

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.13893v1) [paper-pdf](http://arxiv.org/pdf/2510.13893v1)

**Authors**: Olga E. Sorokoletova, Francesco Giarrusso, Vincenzo Suriani, Daniele Nardi

**Abstract**: Jailbreaking techniques pose a significant threat to the safety of Large Language Models (LLMs). Existing defenses typically focus on single-turn attacks, lack coverage across languages, and rely on limited taxonomies that either fail to capture the full diversity of attack strategies or emphasize risk categories rather than the jailbreaking techniques. To advance the understanding of the effectiveness of jailbreaking techniques, we conducted a structured red-teaming challenge. The outcome of our experiments are manifold. First, we developed a comprehensive hierarchical taxonomy of 50 jailbreak strategies, consolidating and extending prior classifications into seven broad families, including impersonation, persuasion, privilege escalation, cognitive overload, obfuscation, goal conflict, and data poisoning. Second, we analyzed the data collected from the challenge to examine the prevalence and success rates of different attack types, providing insights into how specific jailbreak strategies exploit model vulnerabilities and induce misalignment. Third, we benchmark a popular LLM for jailbreak detection, evaluating the benefits of taxonomy-guided prompting for improving automatic detection. Finally, we compiled a new Italian dataset of 1364 multi-turn adversarial dialogues, annotated with our taxonomy, enabling the study of interactions where adversarial intent emerges gradually and succeeds in bypassing traditional safeguards.



## **37. IP-Augmented Multi-Modal Malicious URL Detection Via Token-Contrastive Representation Enhancement and Multi-Granularity Fusion**

cs.CR

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12395v1) [paper-pdf](http://arxiv.org/pdf/2510.12395v1)

**Authors**: Ye Tian, Yanqiu Yu, Liangliang Song, Zhiquan Liu, Yanbin Wang, Jianguo Sun

**Abstract**: Malicious URL detection remains a critical cybersecurity challenge as adversaries increasingly employ sophisticated evasion techniques including obfuscation, character-level perturbations, and adversarial attacks. Although pre-trained language models (PLMs) like BERT have shown potential for URL analysis tasks, three limitations persist in current implementations: (1) inability to effectively model the non-natural hierarchical structure of URLs, (2) insufficient sensitivity to character-level obfuscation, and (3) lack of mechanisms to incorporate auxiliary network-level signals such as IP addresses-all essential for robust detection. To address these challenges, we propose CURL-IP, an advanced multi-modal detection framework incorporating three key innovations: (1) Token-Contrastive Representation Enhancer, which enhances subword token representations through token-aware contrastive learning to produce more discriminative and isotropic embeddings; (2) Cross-Layer Multi-Scale Aggregator, employing hierarchical aggregation of Transformer outputs via convolutional operations and gated MLPs to capture both local and global semantic patterns across layers; and (3) Blockwise Multi-Modal Coupler that decomposes URL-IP features into localized block units and computes cross-modal attention weights at the block level, enabling fine-grained inter-modal interaction. This architecture enables simultaneous preservation of fine-grained lexical cues, contextual semantics, and integration of network-level signals. Our evaluation on large-scale real-world datasets shows the framework significantly outperforms state-of-the-art baselines across binary and multi-class classification tasks.



## **38. L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning (Preprint)**

cs.AI

This preprint was submitted to IEEE TrustCom 2025. The accepted  version will be published under copyright 2025 IEEE

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.07363v2) [paper-pdf](http://arxiv.org/pdf/2510.07363v2)

**Authors**: Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Jun Wang, Yan Li, Chang Liu

**Abstract**: The increasing integration of Industrial IoT (IIoT) exposes critical cyber-physical systems to sophisticated, multi-stage attacks that elude traditional defenses lacking contextual awareness. This paper introduces L2M-AID, a novel framework for Autonomous Industrial Defense using LLM-empowered, Multi-agent reinforcement learning. L2M-AID orchestrates a team of collaborative agents, each driven by a Large Language Model (LLM), to achieve adaptive and resilient security. The core innovation lies in the deep fusion of two AI paradigms: we leverage an LLM as a semantic bridge to translate vast, unstructured telemetry into a rich, contextual state representation, enabling agents to reason about adversary intent rather than merely matching patterns. This semantically-aware state empowers a Multi-Agent Reinforcement Learning (MARL) algorithm, MAPPO, to learn complex cooperative strategies. The MARL reward function is uniquely engineered to balance security objectives (threat neutralization) with operational imperatives, explicitly penalizing actions that disrupt physical process stability. To validate our approach, we conduct extensive experiments on the benchmark SWaT dataset and a novel synthetic dataset generated based on the MITRE ATT&CK for ICS framework. Results demonstrate that L2M-AID significantly outperforms traditional IDS, deep learning anomaly detectors, and single-agent RL baselines across key metrics, achieving a 97.2% detection rate while reducing false positives by over 80% and improving response times by a factor of four. Crucially, it demonstrates superior performance in maintaining physical process stability, presenting a robust new paradigm for securing critical national infrastructure.



## **39. Cross-Modal Safety Alignment: Is textual unlearning all you need?**

cs.CL

Accepted by EMNLP 2024 Findings

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2406.02575v2) [paper-pdf](http://arxiv.org/pdf/2406.02575v2)

**Authors**: Trishna Chakraborty, Erfan Shayegani, Zikui Cai, Nael Abu-Ghazaleh, M. Salman Asif, Yue Dong, Amit K. Roy-Chowdhury, Chengyu Song

**Abstract**: Recent studies reveal that integrating new modalities into Large Language Models (LLMs), such as Vision-Language Models (VLMs), creates a new attack surface that bypasses existing safety training techniques like Supervised Fine-tuning (SFT) and Reinforcement Learning with Human Feedback (RLHF). While further SFT and RLHF-based safety training can be conducted in multi-modal settings, collecting multi-modal training datasets poses a significant challenge. Inspired by the structural design of recent multi-modal models, where, regardless of the combination of input modalities, all inputs are ultimately fused into the language space, we aim to explore whether unlearning solely in the textual domain can be effective for cross-modality safety alignment. Our evaluation across six datasets empirically demonstrates the transferability -- textual unlearning in VLMs significantly reduces the Attack Success Rate (ASR) to less than 8\% and in some cases, even as low as nearly 2\% for both text-based and vision-text-based attacks, alongside preserving the utility. Moreover, our experiments show that unlearning with a multi-modal dataset offers no potential benefits but incurs significantly increased computational demands, possibly up to 6 times higher.



## **40. Unveiling the Vulnerability of Graph-LLMs: An Interpretable Multi-Dimensional Adversarial Attack on TAGs**

cs.LG

12 pages, 4 figures

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12233v1) [paper-pdf](http://arxiv.org/pdf/2510.12233v1)

**Authors**: Bowen Fan, Zhilin Guo, Xunkai Li, Yihan Zhou, Bing Zhou, Zhenjun Li, Rong-Hua Li, Guoren Wang

**Abstract**: Graph Neural Networks (GNNs) have become a pivotal framework for modeling graph-structured data, enabling a wide range of applications from social network analysis to molecular chemistry. By integrating large language models (LLMs), text-attributed graphs (TAGs) enhance node representations with rich textual semantics, significantly boosting the expressive power of graph-based learning. However, this sophisticated synergy introduces critical vulnerabilities, as Graph-LLMs are susceptible to adversarial attacks on both their structural topology and textual attributes. Although specialized attack methods have been designed for each of these aspects, no work has yet unified them into a comprehensive approach. In this work, we propose the Interpretable Multi-Dimensional Graph Attack (IMDGA), a novel human-centric adversarial attack framework designed to orchestrate multi-level perturbations across both graph structure and textual features. IMDGA utilizes three tightly integrated modules to craft attacks that balance interpretability and impact, enabling a deeper understanding of Graph-LLM vulnerabilities. Through rigorous theoretical analysis and comprehensive empirical evaluations on diverse datasets and architectures, IMDGA demonstrates superior interpretability, attack effectiveness, stealthiness, and robustness compared to existing methods. By exposing critical weaknesses in TAG representation learning, this work uncovers a previously underexplored semantic dimension of vulnerability in Graph-LLMs, offering valuable insights for improving their resilience. Our code and resources are publicly available at https://anonymous.4open.science/r/IMDGA-7289.



## **41. HackWorld: Evaluating Computer-Use Agents on Exploiting Web Application Vulnerabilities**

cs.CR

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12200v1) [paper-pdf](http://arxiv.org/pdf/2510.12200v1)

**Authors**: Xiaoxue Ren, Penghao Jiang, Kaixin Li, Zhiyong Huang, Xiaoning Du, Jiaojiao Jiang, Zhenchang Xing, Jiamou Sun, Terry Yue Zhuo

**Abstract**: Web applications are prime targets for cyberattacks as gateways to critical services and sensitive data. Traditional penetration testing is costly and expertise-intensive, making it difficult to scale with the growing web ecosystem. While language model agents show promise in cybersecurity, modern web applications demand visual understanding, dynamic content handling, and multi-step interactions that only computer-use agents (CUAs) can perform. Yet, their ability to discover and exploit vulnerabilities through graphical interfaces remains largely unexplored. We present HackWorld, the first framework for systematically evaluating CUAs' capabilities to exploit web application vulnerabilities via visual interaction. Unlike sanitized benchmarks, HackWorld includes 36 real-world applications across 11 frameworks and 7 languages, featuring realistic flaws such as injection vulnerabilities, authentication bypasses, and unsafe input handling. Using a Capture-the-Flag (CTF) setup, it tests CUAs' capacity to identify and exploit these weaknesses while navigating complex web interfaces. Evaluation of state-of-the-art CUAs reveals concerning trends: exploitation rates below 12% and low cybersecurity awareness. CUAs often fail at multi-step attack planning and misuse security tools. These results expose the current limitations of CUAs in web security contexts and highlight opportunities for developing more security-aware agents capable of effective vulnerability detection and exploitation.



## **42. When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers**

cs.CL

Published in Reliable ML from Unreliable Data workshop @ NeurIPS 2025

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2402.10601v5) [paper-pdf](http://arxiv.org/pdf/2402.10601v5)

**Authors**: Divij Handa, Zehua Zhang, Amir Saeidi, Shrinidhi Kumbhar, Md Nayem Uddin, Aswin RRV, Chitta Baral

**Abstract**: Recent advancements in Large Language Model (LLM) safety have primarily focused on mitigating attacks crafted in natural language or common ciphers (e.g. Base64), which are likely integrated into newer models' safety training. However, we reveal a paradoxical vulnerability: as LLMs advance in reasoning, they inadvertently become more susceptible to novel jailbreaking attacks. Enhanced reasoning enables LLMs to interpret complex instructions and decode complex user-defined ciphers, creating an exploitable security gap. To study this vulnerability, we introduce Attacks using Custom Encryptions (ACE), a jailbreaking technique that encodes malicious queries with novel ciphers. Extending ACE, we introduce Layered Attacks using Custom Encryptions (LACE), which applies multi-layer ciphers to amplify attack complexity. Furthermore, we develop CipherBench, a benchmark designed to evaluate LLMs' accuracy in decoding encrypted benign text. Our experiments reveal a critical trade-off: LLMs that are more capable of decoding ciphers are more vulnerable to LACE, with success rates on gpt-oss-20b escalating from 60% under ACE to 72% with LACE. These findings highlight a critical insight: as LLMs become more adept at deciphering complex user ciphers--many of which cannot be preemptively included in safety training--they become increasingly exploitable.



## **43. Attention-Aware GNN-based Input Defense against Multi-Turn LLM Jailbreak**

cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2507.07146v2) [paper-pdf](http://arxiv.org/pdf/2507.07146v2)

**Authors**: Zixuan Huang, Kecheng Huang, Lihao Yin, Bowei He, Huiling Zhen, Mingxuan Yuan, Zili Shao

**Abstract**: Large Language Models (LLMs) have gained significant traction in various applications, yet their capabilities present risks for both constructive and malicious exploitation. Despite extensive training and fine-tuning efforts aimed at enhancing safety, LLMs remain susceptible to jailbreak attacks. Recently, the emergence of multi-turn attacks has intensified this vulnerability. Unlike single-turn attacks, multi-turn attacks incrementally escalate dialogue complexity, rendering them more challenging to detect and mitigate.   In this study, we introduce G-Guard, an innovative attention-aware Graph Neural Network (GNN)-based input classifier specifically designed to defend against multi-turn jailbreak attacks targeting LLMs. G-Guard constructs an entity graph for multi-turn queries, which captures the interrelationships between queries and harmful keywords that present in multi-turn queries. Furthermore, we propose an attention-aware augmentation mechanism that retrieves the most relevant single-turn query based on the ongoing multi-turn conversation. The retrieved query is incorporated as a labeled node within the graph, thereby enhancing the GNN's capacity to classify the current query as harmful or benign. Evaluation results show that G-Guard consistently outperforms all baselines across diverse datasets and evaluation metrics, demonstrating its efficacy as a robust defense mechanism against multi-turn jailbreak attacks.



## **44. SafeMT: Multi-turn Safety for Multimodal Language Models**

cs.CL

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12133v1) [paper-pdf](http://arxiv.org/pdf/2510.12133v1)

**Authors**: Han Zhu, Juntao Dai, Jiaming Ji, Haoran Li, Chengkun Cai, Pengcheng Wen, Chi-Min Chan, Boyuan Chen, Yaodong Yang, Sirui Han, Yike Guo

**Abstract**: With the widespread use of multi-modal Large Language models (MLLMs), safety issues have become a growing concern. Multi-turn dialogues, which are more common in everyday interactions, pose a greater risk than single prompts; however, existing benchmarks do not adequately consider this situation. To encourage the community to focus on the safety issues of these models in multi-turn dialogues, we introduce SafeMT, a benchmark that features dialogues of varying lengths generated from harmful queries accompanied by images. This benchmark consists of 10,000 samples in total, encompassing 17 different scenarios and four jailbreak methods. Additionally, we propose Safety Index (SI) to evaluate the general safety of MLLMs during conversations. We assess the safety of 17 models using this benchmark and discover that the risk of successful attacks on these models increases as the number of turns in harmful dialogues rises. This observation indicates that the safety mechanisms of these models are inadequate for recognizing the hazard in dialogue interactions. We propose a dialogue safety moderator capable of detecting malicious intent concealed within conversations and providing MLLMs with relevant safety policies. Experimental results from several open-source models indicate that this moderator is more effective in reducing multi-turn ASR compared to existed guard models.



## **45. GraphRAG under Fire**

cs.LG

13 pages. Accepted by IEEE Symposium on Security and Privacy 2026  (S&P 2026)

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2501.14050v4) [paper-pdf](http://arxiv.org/pdf/2501.14050v4)

**Authors**: Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi Zhu, Tanqiu Jiang, Neil Gong, Ting Wang

**Abstract**: GraphRAG advances retrieval-augmented generation (RAG) by structuring external knowledge as multi-scale knowledge graphs, enabling language models to integrate both broad context and granular details in their generation. While GraphRAG has demonstrated success across domains, its security implications remain largely unexplored. To bridge this gap, this work examines GraphRAG's vulnerability to poisoning attacks, uncovering an intriguing security paradox: existing RAG poisoning attacks are less effective under GraphRAG than conventional RAG, due to GraphRAG's graph-based indexing and retrieval; yet, the same features also create new attack surfaces. We present GragPoison, a novel attack that exploits shared relations in the underlying knowledge graph to craft poisoning text capable of compromising multiple queries simultaneously. GragPoison employs three key strategies: (i) relation injection to introduce false knowledge, (ii) relation enhancement to amplify poisoning influence, and (iii) narrative generation to embed malicious content within coherent text. Empirical evaluation across diverse datasets and models shows that GragPoison substantially outperforms existing attacks in terms of effectiveness (up to 98% success rate) and scalability (using less than 68% poisoning text) on multiple variations of GraphRAG. We also explore potential defensive measures and their limitations, identifying promising directions for future research.



## **46. Robust ML-based Detection of Conventional, LLM-Generated, and Adversarial Phishing Emails Using Advanced Text Preprocessing**

cs.CR

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11915v1) [paper-pdf](http://arxiv.org/pdf/2510.11915v1)

**Authors**: Deeksha Hareesha Kulal, Chidozie Princewill Arannonu, Afsah Anwar, Nidhi Rastogi, Quamar Niyaz

**Abstract**: Phishing remains a critical cybersecurity threat, especially with the advent of large language models (LLMs) capable of generating highly convincing malicious content. Unlike earlier phishing attempts which are identifiable by grammatical errors, misspellings, incorrect phrasing, and inconsistent formatting, LLM generated emails are grammatically sound, contextually relevant, and linguistically natural. These advancements make phishing emails increasingly difficult to distinguish from legitimate ones, challenging traditional detection mechanisms. Conventional phishing detection systems often fail when faced with emails crafted by LLMs or manipulated using adversarial perturbation techniques. To address this challenge, we propose a robust phishing email detection system featuring an enhanced text preprocessing pipeline. This pipeline includes spelling correction and word splitting to counteract adversarial modifications and improve detection accuracy. Our approach integrates widely adopted natural language processing (NLP) feature extraction techniques and machine learning algorithms. We evaluate our models on publicly available datasets comprising both phishing and legitimate emails, achieving a detection accuracy of 94.26% and F1-score of 84.39% in model deployment setting. To assess robustness, we further evaluate our models using adversarial phishing samples generated by four attack methods in Python TextAttack framework. Additionally, we evaluate models' performance against phishing emails generated by LLMs including ChatGPT and Llama. Results highlight the resilience of models against evolving AI-powered phishing threats.



## **47. Countermind: A Multi-Layered Security Architecture for Large Language Models**

cs.CR

33 pages, 3 figures, 6 tables. Keywords: LLM security;  defense-in-depth; prompt injection; activation steering; multimodal sandbox;  threat modeling

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11837v1) [paper-pdf](http://arxiv.org/pdf/2510.11837v1)

**Authors**: Dominik Schwarz

**Abstract**: The security of Large Language Model (LLM) applications is fundamentally challenged by "form-first" attacks like prompt injection and jailbreaking, where malicious instructions are embedded within user inputs. Conventional defenses, which rely on post hoc output filtering, are often brittle and fail to address the root cause: the model's inability to distinguish trusted instructions from untrusted data. This paper proposes Countermind, a multi-layered security architecture intended to shift defenses from a reactive, post hoc posture to a proactive, pre-inference, and intra-inference enforcement model. The architecture proposes a fortified perimeter designed to structurally validate and transform all inputs, and an internal governance mechanism intended to constrain the model's semantic processing pathways before an output is generated. The primary contributions of this work are conceptual designs for: (1) A Semantic Boundary Logic (SBL) with a mandatory, time-coupled Text Crypter intended to reduce the plaintext prompt injection attack surface, provided all ingestion paths are enforced. (2) A Parameter-Space Restriction (PSR) mechanism, leveraging principles from representation engineering, to dynamically control the LLM's access to internal semantic clusters, with the goal of mitigating semantic drift and dangerous emergent behaviors. (3) A Secure, Self-Regulating Core that uses an OODA loop and a learning security module to adapt its defenses based on an immutable audit log. (4) A Multimodal Input Sandbox and Context-Defense mechanisms to address threats from non-textual data and long-term semantic poisoning. This paper outlines an evaluation plan designed to quantify the proposed architecture's effectiveness in reducing the Attack Success Rate (ASR) for form-first attacks and to measure its potential latency overhead.



## **48. LLMAtKGE: Large Language Models as Explainable Attackers against Knowledge Graph Embeddings**

cs.CL

13 pages

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11584v1) [paper-pdf](http://arxiv.org/pdf/2510.11584v1)

**Authors**: Ting Li, Yang Yang, Yipeng Yu, Liang Yao, Guoqing Chao, Ruifeng Xu

**Abstract**: Adversarial attacks on knowledge graph embeddings (KGE) aim to disrupt the model's ability of link prediction by removing or inserting triples. A recent black-box method has attempted to incorporate textual and structural information to enhance attack performance. However, it is unable to generate human-readable explanations, and exhibits poor generalizability. In the past few years, large language models (LLMs) have demonstrated powerful capabilities in text comprehension, generation, and reasoning. In this paper, we propose LLMAtKGE, a novel LLM-based framework that selects attack targets and generates human-readable explanations. To provide the LLM with sufficient factual context under limited input constraints, we design a structured prompting scheme that explicitly formulates the attack as multiple-choice questions while incorporating KG factual evidence. To address the context-window limitation and hesitation issues, we introduce semantics-based and centrality-based filters, which compress the candidate set while preserving high recall of attack-relevant information. Furthermore, to efficiently integrate both semantic and structural information into the filter, we precompute high-order adjacency and fine-tune the LLM with a triple classification task to enhance filtering performance. Experiments on two widely used knowledge graph datasets demonstrate that our attack outperforms the strongest black-box baselines and provides explanations via reasoning, and showing competitive performance compared with white-box methods. Comprehensive ablation and case studies further validate its capability to generate explanations.



## **49. The Enemy from Within: A Study of Political Delegitimization Discourse in Israeli Political Speech**

cs.CL

EMNLP 2025

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2508.15524v2) [paper-pdf](http://arxiv.org/pdf/2508.15524v2)

**Authors**: Naama Rivlin-Angert, Guy Mor-Lan

**Abstract**: We present the first large-scale computational study of political delegitimization discourse (PDD), defined as symbolic attacks on the normative validity of political entities. We curate and manually annotate a novel Hebrew-language corpus of 10,410 sentences drawn from Knesset speeches (1993-2023), Facebook posts (2018-2021), and leading news outlets, of which 1,812 instances (17.4\%) exhibit PDD and 642 carry additional annotations for intensity, incivility, target type, and affective framing. We introduce a two-stage classification pipeline combining finetuned encoder models and decoder LLMs. Our best model (DictaLM 2.0) attains an F$_1$ of 0.74 for binary PDD detection and a macro-F$_1$ of 0.67 for classification of delegitimization characteristics. Applying this classifier to longitudinal and cross-platform data, we see a marked rise in PDD over three decades, higher prevalence on social media versus parliamentary debate, greater use by male than female politicians, and stronger tendencies among right-leaning actors - with pronounced spikes during election campaigns and major political events. Our findings demonstrate the feasibility and value of automated PDD analysis for understanding democratic discourse.



## **50. Large Language Models Are Effective Code Watermarkers**

cs.CR

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11251v1) [paper-pdf](http://arxiv.org/pdf/2510.11251v1)

**Authors**: Rui Xu, Jiawei Chen, Zhaoxia Yin, Cong Kong, Xinpeng Zhang

**Abstract**: The widespread use of large language models (LLMs) and open-source code has raised ethical and security concerns regarding the distribution and attribution of source code, including unauthorized redistribution, license violations, and misuse of code for malicious purposes. Watermarking has emerged as a promising solution for source attribution, but existing techniques rely heavily on hand-crafted transformation rules, abstract syntax tree (AST) manipulation, or task-specific training, limiting their scalability and generality across languages. Moreover, their robustness against attacks remains limited. To address these limitations, we propose CodeMark-LLM, an LLM-driven watermarking framework that embeds watermark into source code without compromising its semantics or readability. CodeMark-LLM consists of two core components: (i) Semantically Consistent Embedding module that applies functionality-preserving transformations to encode watermark bits, and (ii) Differential Comparison Extraction module that identifies the applied transformations by comparing the original and watermarked code. Leveraging the cross-lingual generalization ability of LLM, CodeMark-LLM avoids language-specific engineering and training pipelines. Extensive experiments across diverse programming languages and attack scenarios demonstrate its robustness, effectiveness, and scalability.



