# Latest Adversarial Attack Papers
**update at 2025-10-09 08:24:48**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. When Should Selfish Miners Double-Spend?**

cs.CR

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2501.03227v2) [paper-pdf](http://arxiv.org/pdf/2501.03227v2)

**Authors**: Mustafa Doger, Sennur Ulukus

**Abstract**: Conventional double-spending attack models ignore the revenue losses stemming from the orphan blocks. On the other hand, selfish mining literature usually ignores the chance of the attacker to double-spend at no-cost in each attack cycle. In this paper, we give a rigorous stochastic analysis of an attack where the goal of the adversary is to double-spend while mining selfishly. To do so, we first combine stubborn and selfish mining attacks, i.e., construct a strategy where the attacker acts stubborn until its private branch reaches a certain length and then switches to act selfish. We provide the optimal stubbornness for each parameter regime. Next, we provide the maximum stubbornness that is still more profitable than honest mining and argue a connection between the level of stubbornness and the $k$-confirmation rule. We show that, at each attack cycle, if the level of stubbornness is higher than $k$, the adversary gets a free shot at double-spending. At each cycle, for a given stubbornness level, we rigorously formulate how great the probability of double-spending is. We further modify the attack in the stubborn regime in order to conceal the attack and increase the double-spending probability.



## **2. Sparse Representations Improve Adversarial Robustness of Neural Network Classifiers**

cs.LG

Killian Steunou is the main contributor and corresponding author of  this work

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2509.21130v2) [paper-pdf](http://arxiv.org/pdf/2509.21130v2)

**Authors**: Killian Steunou, Théo Druilhe, Sigurd Saue

**Abstract**: Deep neural networks perform remarkably well on image classification tasks but remain vulnerable to carefully crafted adversarial perturbations. This work revisits linear dimensionality reduction as a simple, data-adapted defense. We empirically compare standard Principal Component Analysis (PCA) with its sparse variant (SPCA) as front-end feature extractors for downstream classifiers, and we complement these experiments with a theoretical analysis. On the theory side, we derive exact robustness certificates for linear heads applied to SPCA features: for both $\ell_\infty$ and $\ell_2$ threat models (binary and multiclass), the certified radius grows as the dual norms of $W^\top u$ shrink, where $W$ is the projection and $u$ the head weights. We further show that for general (non-linear) heads, sparsity reduces operator-norm bounds through a Lipschitz composition argument, predicting lower input sensitivity. Empirically, with a small non-linear network after the projection, SPCA consistently degrades more gracefully than PCA under strong white-box and black-box attacks while maintaining competitive clean accuracy. Taken together, the theory identifies the mechanism (sparser projections reduce adversarial leverage) and the experiments verify that this benefit persists beyond the linear setting. Our code is available at https://github.com/killian31/SPCARobustness.



## **3. SAFER: Advancing Safety Alignment via Efficient Ex-Ante Reasoning**

cs.CL

22 pages, 5 figures

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2504.02725v2) [paper-pdf](http://arxiv.org/pdf/2504.02725v2)

**Authors**: Kehua Feng, Keyan Ding, Yuhao Wang, Menghan Li, Fanjunduo Wei, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose SAFER, a framework for Safety Alignment via eFficient Ex-Ante Reasoning. Our approach instantiates structured Ex-Ante reasoning through initial assessment, rule verification, and path calibration, and embeds predefined safety rules to provide transparent and verifiable safety judgments. Specifically, our approach consists of two training stages: (1) supervised fine-tuning with synthetic traces to teach the multi-stage Ex-Ante reasoning, and (2) step-level reasoning preference optimization to jointly enhance safety, utility, and efficiency. Experiments on multiple open-source LLMs demonstrate that SAFER significantly enhances safety performance while maintaining helpfulness and response efficiency.



## **4. DP-SNP-TIHMM: Differentially Private, Time-Inhomogeneous Hidden Markov Models for Synthesizing Genome-Wide Association Datasets**

cs.LG

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05777v1) [paper-pdf](http://arxiv.org/pdf/2510.05777v1)

**Authors**: Shadi Rahimian, Mario Fritz

**Abstract**: Single nucleotide polymorphism (SNP) datasets are fundamental to genetic studies but pose significant privacy risks when shared. The correlation of SNPs with each other makes strong adversarial attacks such as masked-value reconstruction, kin, and membership inference attacks possible. Existing privacy-preserving approaches either apply differential privacy to statistical summaries of these datasets or offer complex methods that require post-processing and the usage of a publicly available dataset to suppress or selectively share SNPs.   In this study, we introduce an innovative framework for generating synthetic SNP sequence datasets using samples derived from time-inhomogeneous hidden Markov models (TIHMMs). To preserve the privacy of the training data, we ensure that each SNP sequence contributes only a bounded influence during training, enabling strong differential privacy guarantees. Crucially, by operating on full SNP sequences and bounding their gradient contributions, our method directly addresses the privacy risks introduced by their inherent correlations.   Through experiments conducted on the real-world 1000 Genomes dataset, we demonstrate the efficacy of our method using privacy budgets of $\varepsilon \in [1, 10]$ at $\delta=10^{-4}$. Notably, by allowing the transition models of the HMM to be dependent on the location in the sequence, we significantly enhance performance, enabling the synthetic datasets to closely replicate the statistical properties of non-private datasets. This framework facilitates the private sharing of genomic data while offering researchers exceptional flexibility and utility.



## **5. Evidence of Cognitive Biases in Capture-the-Flag Cybersecurity Competitions**

cs.CR

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05771v1) [paper-pdf](http://arxiv.org/pdf/2510.05771v1)

**Authors**: Carolina Carreira, Anu Aggarwal, Alejandro Cuevas, Maria José Ferreira, Hanan Hibshi, Cleotilde Gonzalez

**Abstract**: Understanding how cognitive biases influence adversarial decision-making is essential for developing effective cyber defenses. Capture-the-Flag (CTF) competitions provide an ecologically valid testbed to study attacker behavior at scale, simulating real-world intrusion scenarios under pressure. We analyze over 500,000 submission logs from picoCTF, a large educational CTF platform, to identify behavioral signatures of cognitive biases with defensive implications. Focusing on availability bias and the sunk cost fallacy, we employ a mixed-methods approach combining qualitative coding, descriptive statistics, and generalized linear modeling. Our findings show that participants often submitted flags with correct content but incorrect formatting (availability bias), and persisted in attempting challenges despite repeated failures and declining success probabilities (sunk cost fallacy). These patterns reveal that biases naturally shape attacker behavior in adversarial contexts. Building on these insights, we outline a framework for bias-informed adaptive defenses that anticipate, rather than simply react to, adversarial actions.



## **6. Shortcuts Everywhere and Nowhere: Exploring Multi-Trigger Backdoor Attacks**

cs.LG

13 pages

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2401.15295v4) [paper-pdf](http://arxiv.org/pdf/2401.15295v4)

**Authors**: Yige Li, Jiabo He, Hanxun Huang, Jun Sun, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Backdoor attacks have become a significant threat to the pre-training and deployment of deep neural networks (DNNs). Although numerous methods for detecting and mitigating backdoor attacks have been proposed, most rely on identifying and eliminating the ``shortcut" created by the backdoor, which links a specific source class to a target class. However, these approaches can be easily circumvented by designing multiple backdoor triggers that create shortcuts everywhere and therefore nowhere specific. In this study, we explore the concept of Multi-Trigger Backdoor Attacks (MTBAs), where multiple adversaries leverage different types of triggers to poison the same dataset. By proposing and investigating three types of multi-trigger attacks including \textit{parallel}, \textit{sequential}, and \textit{hybrid} attacks, we demonstrate that 1) multiple triggers can coexist, overwrite, or cross-activate one another, and 2) MTBAs easily break the prevalent shortcut assumption underlying most existing backdoor detection/removal methods, rendering them ineffective. Given the security risk posed by MTBAs, we have created a multi-trigger backdoor poisoning dataset to facilitate future research on detecting and mitigating these attacks, and we also discuss potential defense strategies against MTBAs. Our code is available at https://github.com/bboylyg/Multi-Trigger-Backdoor-Attacks.



## **7. Geometry-Guided Adversarial Prompt Detection via Curvature and Local Intrinsic Dimension**

cs.CL

40 Pages, 6 figues

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2503.03502v2) [paper-pdf](http://arxiv.org/pdf/2503.03502v2)

**Authors**: Canaan Yung, Hanxun Huang, Christopher Leckie, Sarah Erfani

**Abstract**: Adversarial prompts are capable of jailbreaking frontier large language models (LLMs) and inducing undesirable behaviours, posing a significant obstacle to their safe deployment. Current mitigation strategies primarily rely on activating built-in defence mechanisms or fine-tuning LLMs, both of which are computationally expensive and can sacrifice model utility. In contrast, detection-based approaches are more efficient and practical for deployment in real-world applications. However, the fundamental distinctions between adversarial and benign prompts remain poorly understood. In this work, we introduce CurvaLID, a novel defence framework that efficiently detects adversarial prompts by leveraging their geometric properties. It is agnostic to the type of LLM, offering a unified detection framework across diverse adversarial prompts and LLM architectures. CurvaLID builds on the geometric analysis of text prompts to uncover their underlying differences. We theoretically extend the concept of curvature via the Whewell equation into an $n$-dimensional word embedding space, enabling us to quantify local geometric properties, including semantic shifts and curvature in the underlying manifolds. To further enhance our solution, we leverage Local Intrinsic Dimensionality (LID) to capture complementary geometric features of text prompts within adversarial subspaces. Our findings show that adversarial prompts exhibit distinct geometric signatures from benign prompts, enabling CurvaLID to achieve near-perfect classification and outperform state-of-the-art detectors in adversarial prompt detection. CurvaLID provides a reliable and efficient safeguard against malicious queries as a model-agnostic method that generalises across multiple LLMs and attack families.



## **8. Benchmarking the Robustness of Agentic Systems to Adversarially-Induced Harms**

cs.LG

54 Pages

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2508.16481v2) [paper-pdf](http://arxiv.org/pdf/2508.16481v2)

**Authors**: Jonathan Nöther, Adish Singla, Goran Radanovic

**Abstract**: Ensuring the safe use of agentic systems requires a thorough understanding of the range of malicious behaviors these systems may exhibit when under attack. In this paper, we evaluate the robustness of LLM-based agentic systems against attacks that aim to elicit harmful actions from agents. To this end, we propose a novel taxonomy of harms for agentic systems and a novel benchmark, BAD-ACTS, for studying the security of agentic systems with respect to a wide range of harmful actions. BAD-ACTS consists of 4 implementations of agentic systems in distinct application environments, as well as a dataset of 188 high-quality examples of harmful actions. This enables a comprehensive study of the robustness of agentic systems across a wide range of categories of harmful behaviors, available tools, and inter-agent communication structures. Using this benchmark, we analyze the robustness of agentic systems against an attacker that controls one of the agents in the system and aims to manipulate other agents to execute a harmful target action. Our results show that the attack has a high success rate, demonstrating that even a single adversarial agent within the system can have a significant impact on the security. This attack remains effective even when agents use a simple prompting-based defense strategy. However, we additionally propose a more effective defense based on message monitoring. We believe that this benchmark provides a diverse testbed for the security research of agentic systems. The benchmark can be found at github.com/JNoether/BAD-ACTS



## **9. Adversarial Reinforcement Learning for Large Language Model Agent Safety**

cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05442v1) [paper-pdf](http://arxiv.org/pdf/2510.05442v1)

**Authors**: Zizhao Wang, Dingcheng Li, Vaishakh Keshava, Phillip Wallis, Ananth Balashankar, Peter Stone, Lukas Rutishauser

**Abstract**: Large Language Model (LLM) agents can leverage tools such as Google Search to complete complex tasks. However, this tool usage introduces the risk of indirect prompt injections, where malicious instructions hidden in tool outputs can manipulate the agent, posing security risks like data leakage. Current defense strategies typically rely on fine-tuning LLM agents on datasets of known attacks. However, the generation of these datasets relies on manually crafted attack patterns, which limits their diversity and leaves agents vulnerable to novel prompt injections. To address this limitation, we propose Adversarial Reinforcement Learning for Agent Safety (ARLAS), a novel framework that leverages adversarial reinforcement learning (RL) by formulating the problem as a two-player zero-sum game. ARLAS co-trains two LLMs: an attacker that learns to autonomously generate diverse prompt injections and an agent that learns to defend against them while completing its assigned tasks. To ensure robustness against a wide range of attacks and to prevent cyclic learning, we employ a population-based learning framework that trains the agent to defend against all previous attacker checkpoints. Evaluated on BrowserGym and AgentDojo, agents fine-tuned with ARLAS achieve a significantly lower attack success rate than the original model while also improving their task success rate. Our analysis further confirms that the adversarial process generates a diverse and challenging set of attacks, leading to a more robust agent compared to the base model.



## **10. Accuracy-Robustness Trade Off via Spiking Neural Network Gradient Sparsity Trail**

cs.NE

Work under peer-review

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2509.23762v2) [paper-pdf](http://arxiv.org/pdf/2509.23762v2)

**Authors**: Nhan T. Luu

**Abstract**: Spiking Neural Networks (SNNs) have attracted growing interest in both computational neuroscience and artificial intelligence, primarily due to their inherent energy efficiency and compact memory footprint. However, achieving adversarial robustness in SNNs, particularly for vision-related tasks, remains a nascent and underexplored challenge. Recent studies have proposed leveraging sparse gradients as a form of regularization to enhance robustness against adversarial perturbations. In this work, we present a surprising finding: under specific architectural configurations, SNNs exhibit natural gradient sparsity and can achieve state-of-the-art adversarial defense performance without the need for any explicit regularization. Further analysis reveals a trade-off between robustness and generalization: while sparse gradients contribute to improved adversarial resilience, they can impair the model's ability to generalize; conversely, denser gradients support better generalization but increase vulnerability to attacks.



## **11. RegMix: Adversarial Mutual and Generalization Regularization for Enhancing DNN Robustness**

cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05317v1) [paper-pdf](http://arxiv.org/pdf/2510.05317v1)

**Authors**: Zhenyu Liu, Varun Ojha

**Abstract**: Adversarial training is the most effective defense against adversarial attacks. The effectiveness of the adversarial attacks has been on the design of its loss function and regularization term. The most widely used loss function in adversarial training is cross-entropy and mean squared error (MSE) as its regularization objective. However, MSE enforces overly uniform optimization between two output distributions during training, which limits its robustness in adversarial training scenarios. To address this issue, we revisit the idea of mutual learning (originally designed for knowledge distillation) and propose two novel regularization strategies tailored for adversarial training: (i) weighted adversarial mutual regularization and (ii) adversarial generalization regularization. In the former, we formulate a decomposed adversarial mutual Kullback-Leibler divergence (KL-divergence) loss, which allows flexible control over the optimization process by assigning unequal weights to the main and auxiliary objectives. In the latter, we introduce an additional clean target distribution into the adversarial training objective, improving generalization and enhancing model robustness. Extensive experiments demonstrate that our proposed methods significantly improve adversarial robustness compared to existing regularization-based approaches.



## **12. Proactive defense against LLM Jailbreak**

cs.CR

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05052v1) [paper-pdf](http://arxiv.org/pdf/2510.05052v1)

**Authors**: Weiliang Zhao, Jinjun Peng, Daniel Ben-Levi, Zhou Yu, Junfeng Yang

**Abstract**: The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, primarily reactive and static, often fail to counter these search-based attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead autonomous jailbreaking processes. Our core idea is to intentionally provide adversaries with "spurious responses" that appear to be results of successful jailbreak attacks but contain no actual harmful content. These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak. By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, our method consistently and significantly reduces attack success rates by up to 92\%. When combined with other defense frameworks, it further reduces the success rate of the latest attack strategies to 0\%. ProAct represents an orthogonal defense strategy that can serve as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks.



## **13. Rethinking Exact Unlearning under Exposure: Extracting Forgotten Data under Exact Unlearning in Large Language Model**

cs.LG

Accepted by Neurips 2025

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2505.24379v2) [paper-pdf](http://arxiv.org/pdf/2505.24379v2)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.



## **14. Imperceptible Jailbreaking against Large Language Models**

cs.CL

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05025v1) [paper-pdf](http://arxiv.org/pdf/2510.05025v1)

**Authors**: Kuofeng Gao, Yiming Li, Chao Du, Xin Wang, Xingjun Ma, Shu-Tao Xia, Tianyu Pang

**Abstract**: Jailbreaking attacks on the vision modality typically rely on imperceptible adversarial perturbations, whereas attacks on the textual modality are generally assumed to require visible modifications (e.g., non-semantic suffixes). In this paper, we introduce imperceptible jailbreaks that exploit a class of Unicode characters called variation selectors. By appending invisible variation selectors to malicious questions, the jailbreak prompts appear visually identical to original malicious questions on screen, while their tokenization is "secretly" altered. We propose a chain-of-search pipeline to generate such adversarial suffixes to induce harmful responses. Our experiments show that our imperceptible jailbreaks achieve high attack success rates against four aligned LLMs and generalize to prompt injection attacks, all without producing any visible modifications in the written prompt. Our code is available at https://github.com/sail-sg/imperceptible-jailbreaks.



## **15. Cooperative Decentralized Backdoor Attacks on Vertical Federated Learning**

cs.LG

This paper is currently under review in the IEEE/ACM Transactions on  Networking Special Issue on AI and Networking

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2501.09320v2) [paper-pdf](http://arxiv.org/pdf/2501.09320v2)

**Authors**: Seohyun Lee, Wenzhi Fang, Anindya Bijoy Das, Seyyedali Hosseinalipour, David J. Love, Christopher G. Brinton

**Abstract**: Federated learning (FL) is vulnerable to backdoor attacks, where adversaries alter model behavior on target classification labels by embedding triggers into data samples. While these attacks have received considerable attention in horizontal FL, they are less understood for vertical FL (VFL), where devices hold different features of the samples, and only the server holds the labels. In this work, we propose a novel backdoor attack on VFL which (i) does not rely on gradient information from the server and (ii) considers potential collusion among multiple adversaries for sample selection and trigger embedding. Our label inference model augments variational autoencoders with metric learning, which adversaries can train locally. A consensus process over the adversary graph topology determines which datapoints to poison. We further propose methods for trigger splitting across the adversaries, with an intensity-based implantation scheme skewing the server towards the trigger. Our convergence analysis reveals the impact of backdoor perturbations on VFL indicated by a stationarity gap for the trained model, which we verify empirically as well. We conduct experiments comparing our attack with recent backdoor VFL approaches, finding that ours obtains significantly higher success rates for the same main task performance despite not using server information. Additionally, our results verify the impact of collusion on attack performance.



## **16. NatGVD: Natural Adversarial Example Attack towards Graph-based Vulnerability Detection**

cs.CR

10 pages, 2 figures (2 additional figures in Appendices)

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.04987v1) [paper-pdf](http://arxiv.org/pdf/2510.04987v1)

**Authors**: Avilash Rath, Weiliang Qi, Youpeng Li, Xinda Wang

**Abstract**: Graph-based models learn rich code graph structural information and present superior performance on various code analysis tasks. However, the robustness of these models against adversarial example attacks in the context of vulnerability detection remains an open question. This paper proposes NatGVD, a novel attack methodology that generates natural adversarial vulnerable code to circumvent GNN-based and graph-aware transformer-based vulnerability detectors. NatGVD employs a set of code transformations that modify graph structure while preserving code semantics. Instead of injecting dead or unrelated code like previous works, NatGVD considers naturalness requirements: generated examples should not be easily recognized by humans or program analysis tools. With extensive evaluation of NatGVD on state-of-the-art vulnerability detection systems, the results reveal up to 53.04% evasion rate across GNN-based detectors and graph-aware transformer-based detectors. We also explore potential defense strategies to enhance the robustness of these systems against NatGVD.



## **17. Impact of Dataset Properties on Membership Inference Vulnerability of Deep Transfer Learning**

cs.CR

Accepted to NeurIPS 2025; 47 pages, 13 figures

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2402.06674v5) [paper-pdf](http://arxiv.org/pdf/2402.06674v5)

**Authors**: Marlon Tobaben, Hibiki Ito, Joonas Jälkö, Yuan He, Antti Honkela

**Abstract**: Membership inference attacks (MIAs) are used to test practical privacy of machine learning models. MIAs complement formal guarantees from differential privacy (DP) under a more realistic adversary model. We analyse MIA vulnerability of fine-tuned neural networks both empirically and theoretically, the latter using a simplified model of fine-tuning. We show that the vulnerability of non-DP models when measured as the attacker advantage at a fixed false positive rate reduces according to a simple power law as the number of examples per class increases. A similar power-law applies even for the most vulnerable points, but the dataset size needed for adequate protection of the most vulnerable points is very large.



## **18. Sampling-aware Adversarial Attacks Against Large Language Models**

cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2507.04446v3) [paper-pdf](http://arxiv.org/pdf/2507.04446v3)

**Authors**: Tim Beyer, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point greedy generations, overlooking the inherently stochastic nature of LLMs and overestimating robustness. We show that for the goal of eliciting harmful responses, repeated sampling of model outputs during the attack complements prompt optimization and serves as a strong and efficient attack vector. By casting attacks as a resource allocation problem between optimization and sampling, we determine compute-optimal trade-offs and show that integrating sampling into existing attacks boosts success rates by up to 37\% and improves efficiency by up to two orders of magnitude. We further analyze how distributions of output harmfulness evolve during an adversarial attack, discovering that many common optimization strategies have little effect on output harmfulness. Finally, we introduce a label-free proof-of-concept objective based on entropy maximization, demonstrating how our sampling-aware perspective enables new optimization targets. Overall, our findings establish the importance of sampling in attacks to accurately assess and strengthen LLM safety at scale.



## **19. Unified Threat Detection and Mitigation Framework (UTDMF): Combating Prompt Injection, Deception, and Bias in Enterprise-Scale Transformers**

cs.CR

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.04528v1) [paper-pdf](http://arxiv.org/pdf/2510.04528v1)

**Authors**: Santhosh KumarRavindran

**Abstract**: The rapid adoption of large language models (LLMs) in enterprise systems exposes vulnerabilities to prompt injection attacks, strategic deception, and biased outputs, threatening security, trust, and fairness. Extending our adversarial activation patching framework (arXiv:2507.09406), which induced deception in toy networks at a 23.9% rate, we introduce the Unified Threat Detection and Mitigation Framework (UTDMF), a scalable, real-time pipeline for enterprise-grade models like Llama-3.1 (405B), GPT-4o, and Claude-3.5. Through 700+ experiments per model, UTDMF achieves: (1) 92% detection accuracy for prompt injection (e.g., jailbreaking); (2) 65% reduction in deceptive outputs via enhanced patching; and (3) 78% improvement in fairness metrics (e.g., demographic bias). Novel contributions include a generalized patching algorithm for multi-threat detection, three groundbreaking hypotheses on threat interactions (e.g., threat chaining in enterprise workflows), and a deployment-ready toolkit with APIs for enterprise integration.



## **20. Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models**

cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2506.07468v3) [paper-pdf](http://arxiv.org/pdf/2506.07468v3)

**Authors**: Mickel Liu, Liwei Jiang, Yancheng Liang, Simon Shaolei Du, Yejin Choi, Tim Althoff, Natasha Jaques

**Abstract**: Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL).



## **21. SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations**

cs.CL

Accepted at NeurIPS 2025. Code is available at  https://github.com/Buyun-Liang/SECA

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04398v1) [paper-pdf](http://arxiv.org/pdf/2510.04398v1)

**Authors**: Buyun Liang, Liangzu Peng, Jinqi Luo, Darshan Thaker, Kwan Ho Ryan Chan, René Vidal

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no constraint violations compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at https://github.com/Buyun-Liang/SECA.



## **22. Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models**

cs.CL

15 pages total (9 pages main text + 4 pages appendix + references),  12 figures, preprint version. The final version may differ

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04347v1) [paper-pdf](http://arxiv.org/pdf/2510.04347v1)

**Authors**: Anindya Sundar Das, Kangjie Chen, Monowar Bhuyan

**Abstract**: Pre-trained language models have achieved remarkable success across a wide range of natural language processing (NLP) tasks, particularly when fine-tuned on large, domain-relevant datasets. However, they remain vulnerable to backdoor attacks, where adversaries embed malicious behaviors using trigger patterns in the training data. These triggers remain dormant during normal usage, but, when activated, can cause targeted misclassifications. In this work, we investigate the internal behavior of backdoored pre-trained encoder-based language models, focusing on the consistent shift in attention and gradient attribution when processing poisoned inputs; where the trigger token dominates both attention and gradient signals, overriding the surrounding context. We propose an inference-time defense that constructs anomaly scores by combining token-level attention and gradient information. Extensive experiments on text classification tasks across diverse backdoor attack scenarios demonstrate that our method significantly reduces attack success rates compared to existing baselines. Furthermore, we provide an interpretability-driven analysis of the scoring mechanism, shedding light on trigger localization and the robustness of the proposed defense.



## **23. VortexPIA: Indirect Prompt Injection Attack against LLMs for Efficient Extraction of User Privacy**

cs.CR

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04261v1) [paper-pdf](http://arxiv.org/pdf/2510.04261v1)

**Authors**: Yu Cui, Sicheng Pan, Yifei Liu, Haibin Zhang, Cong Zuo

**Abstract**: Large language models (LLMs) have been widely deployed in Conversational AIs (CAIs), while exposing privacy and security threats. Recent research shows that LLM-based CAIs can be manipulated to extract private information from human users, posing serious security threats. However, the methods proposed in that study rely on a white-box setting that adversaries can directly modify the system prompt. This condition is unlikely to hold in real-world deployments. The limitation raises a critical question: can unprivileged attackers still induce such privacy risks in practical LLM-integrated applications? To address this question, we propose \textsc{VortexPIA}, a novel indirect prompt injection attack that induces privacy extraction in LLM-integrated applications under black-box settings. By injecting token-efficient data containing false memories, \textsc{VortexPIA} misleads LLMs to actively request private information in batches. Unlike prior methods, \textsc{VortexPIA} allows attackers to flexibly define multiple categories of sensitive data. We evaluate \textsc{VortexPIA} on six LLMs, covering both traditional and reasoning LLMs, across four benchmark datasets. The results show that \textsc{VortexPIA} significantly outperforms baselines and achieves state-of-the-art (SOTA) performance. It also demonstrates efficient privacy requests, reduced token consumption, and enhanced robustness against defense mechanisms. We further validate \textsc{VortexPIA} on multiple realistic open-source LLM-integrated applications, demonstrating its practical effectiveness.



## **24. Machine Unlearning in Speech Emotion Recognition via Forget Set Alone**

cs.SD

Submitted to ICASSP 2026

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04251v1) [paper-pdf](http://arxiv.org/pdf/2510.04251v1)

**Authors**: Zhao Ren, Rathi Adarshi Rammohan, Kevin Scheck, Tanja Schultz

**Abstract**: Speech emotion recognition aims to identify emotional states from speech signals and has been widely applied in human-computer interaction, education, healthcare, and many other fields. However, since speech data contain rich sensitive information, partial data can be required to be deleted by speakers due to privacy concerns. Current machine unlearning approaches largely depend on data beyond the samples to be forgotten. However, this reliance poses challenges when data redistribution is restricted and demands substantial computational resources in the context of big data. We propose a novel adversarial-attack-based approach that fine-tunes a pre-trained speech emotion recognition model using only the data to be forgotten. The experimental results demonstrate that the proposed approach can effectively remove the knowledge of the data to be forgotten from the model, while preserving high model performance on the test set for emotion recognition.



## **25. Concept-Based Masking: A Patch-Agnostic Defense Against Adversarial Patch Attacks**

cs.CV

neurips workshop

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04245v1) [paper-pdf](http://arxiv.org/pdf/2510.04245v1)

**Authors**: Ayushi Mehrotra, Derek Peng, Dipkamal Bhusal, Nidhi Rastogi

**Abstract**: Adversarial patch attacks pose a practical threat to deep learning models by forcing targeted misclassifications through localized perturbations, often realized in the physical world. Existing defenses typically assume prior knowledge of patch size or location, limiting their applicability. In this work, we propose a patch-agnostic defense that leverages concept-based explanations to identify and suppress the most influential concept activation vectors, thereby neutralizing patch effects without explicit detection. Evaluated on Imagenette with a ResNet-50, our method achieves higher robust and clean accuracy than the state-of-the-art PatchCleanser, while maintaining strong performance across varying patch sizes and locations. Our results highlight the promise of combining interpretability with robustness and suggest concept-driven defenses as a scalable strategy for securing machine learning models against adversarial patch attacks.



## **26. Blending adversarial training and representation-conditional purification via aggregation improves adversarial robustness**

cs.CV

Published in Transactions on Machine Learning Research (09/2025). 25  pages, 1 figure, 19 tables

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2306.06081v6) [paper-pdf](http://arxiv.org/pdf/2306.06081v6)

**Authors**: Emanuele Ballarin, Alessio Ansuini, Luca Bortolussi

**Abstract**: In this work, we propose a novel adversarial defence mechanism for image classification - CARSO - blending the paradigms of adversarial training and adversarial purification in a synergistic robustness-enhancing way. The method builds upon an adversarially-trained classifier, and learns to map its internal representation associated with a potentially perturbed input onto a distribution of tentative clean reconstructions. Multiple samples from such distribution are classified by the same adversarially-trained model, and a carefully chosen aggregation of its outputs finally constitutes the robust prediction of interest. Experimental evaluation by a well-established benchmark of strong adaptive attacks, across different image datasets, shows that CARSO is able to defend itself against adaptive end-to-end white-box attacks devised for stochastic defences. Paying a modest clean accuracy toll, our method improves by a significant margin the state-of-the-art for Cifar-10, Cifar-100, and TinyImageNet-200 $\ell_\infty$ robust classification accuracy against AutoAttack. Code, and instructions to obtain pre-trained models are available at: https://github.com/emaballarin/CARSO .



## **27. Adversarial Attacks and Robust Defenses in Speaker Embedding based Zero-Shot Text-to-Speech System**

eess.AS

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2410.04017v2) [paper-pdf](http://arxiv.org/pdf/2410.04017v2)

**Authors**: Ze Li, Yao Shi, Yunfei Xu, Ming Li

**Abstract**: Speaker embedding based zero-shot Text-to-Speech (TTS) systems enable high-quality speech synthesis for unseen speakers using minimal data. However, these systems are vulnerable to adversarial attacks, where an attacker introduces imperceptible perturbations to the original speaker's audio waveform, leading to synthesized speech sounds like another person. This vulnerability poses significant security risks, including speaker identity spoofing and unauthorized voice manipulation. This paper investigates two primary defense strategies to address these threats: adversarial training and adversarial purification. Adversarial training enhances the model's robustness by integrating adversarial examples during the training process, thereby improving resistance to such attacks. Adversarial purification, on the other hand, employs diffusion probabilistic models to revert adversarially perturbed audio to its clean form. Experimental results demonstrate that these defense mechanisms can significantly reduce the impact of adversarial perturbations, enhancing the security and reliability of speaker embedding based zero-shot TTS systems in adversarial environments.



## **28. Boundary on the Table: Efficient Black-Box Decision-Based Attacks for Structured Data**

cs.LG

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2509.22850v2) [paper-pdf](http://arxiv.org/pdf/2509.22850v2)

**Authors**: Roie Kazoom, Yuval Ratzabi, Etamar Rothstein, Ofer Hadar

**Abstract**: Adversarial robustness in structured data remains an underexplored frontier compared to vision and language domains. In this work, we introduce a novel black-box, decision-based adversarial attack tailored for tabular data. Our approach combines gradient-free direction estimation with an iterative boundary search, enabling efficient navigation of discrete and continuous feature spaces under minimal oracle access. Extensive experiments demonstrate that our method successfully compromises nearly the entire test set across diverse models, ranging from classical machine learning classifiers to large language model (LLM)-based pipelines. Remarkably, the attack achieves success rates consistently above 90%, while requiring only a small number of queries per instance. These results highlight the critical vulnerability of tabular models to adversarial perturbations, underscoring the urgent need for stronger defenses in real-world decision-making systems.



## **29. AttackSeqBench: Benchmarking Large Language Models in Analyzing Attack Sequences within Cyber Threat Intelligence**

cs.CR

36 pages, 9 figures

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2503.03170v2) [paper-pdf](http://arxiv.org/pdf/2503.03170v2)

**Authors**: Haokai Ma, Javier Yong, Yunshan Ma, Kuei Chen, Anis Yusof, Zhenkai Liang, Ee-Chien Chang

**Abstract**: Cyber Threat Intelligence (CTI) reports document observations of cyber threats, synthesizing evidence about adversaries' actions and intent into actionable knowledge that informs detection, response, and defense planning. However, the unstructured and verbose nature of CTI reports poses significant challenges for security practitioners to manually extract and analyze such sequences. Although large language models (LLMs) exhibit promise in cybersecurity tasks such as entity extraction and knowledge graph construction, their understanding and reasoning capabilities towards behavioral sequences remains underexplored. To address this, we introduce AttackSeqBench, a benchmark designed to systematically evaluate LLMs' reasoning abilities across the tactical, technical, and procedural dimensions of adversarial behaviors, while satisfying Extensibility, Reasoning Scalability, and Domain-dpecific Epistemic Expandability. We further benchmark 7 LLMs, 5 LRMs and 4 post-training strategies across the proposed 3 benchmark settings and 3 benchmark tasks within our AttackSeqBench to identify their advantages and limitations in such specific domain. Our findings contribute to a deeper understanding of LLM-driven CTI report understanding and foster its application in cybersecurity operations.



## **30. SafeGuider: Robust and Practical Content Safety Control for Text-to-Image Models**

cs.CR

Accepted by ACM CCS 2025

**SubmitDate**: 2025-10-08    [abs](http://arxiv.org/abs/2510.05173v2) [paper-pdf](http://arxiv.org/pdf/2510.05173v2)

**Authors**: Peigui Qi, Kunsheng Tang, Wenbo Zhou, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang

**Abstract**: Text-to-image models have shown remarkable capabilities in generating high-quality images from natural language descriptions. However, these models are highly vulnerable to adversarial prompts, which can bypass safety measures and produce harmful content. Despite various defensive strategies, achieving robustness against attacks while maintaining practical utility in real-world applications remains a significant challenge. To address this issue, we first conduct an empirical study of the text encoder in the Stable Diffusion (SD) model, which is a widely used and representative text-to-image model. Our findings reveal that the [EOS] token acts as a semantic aggregator, exhibiting distinct distributional patterns between benign and adversarial prompts in its embedding space. Building on this insight, we introduce \textbf{SafeGuider}, a two-step framework designed for robust safety control without compromising generation quality. SafeGuider combines an embedding-level recognition model with a safety-aware feature erasure beam search algorithm. This integration enables the framework to maintain high-quality image generation for benign prompts while ensuring robust defense against both in-domain and out-of-domain attacks. SafeGuider demonstrates exceptional effectiveness in minimizing attack success rates, achieving a maximum rate of only 5.48\% across various attack scenarios. Moreover, instead of refusing to generate or producing black images for unsafe prompts, \textbf{SafeGuider} generates safe and meaningful images, enhancing its practical utility. In addition, SafeGuider is not limited to the SD model and can be effectively applied to other text-to-image models, such as the Flux model, demonstrating its versatility and adaptability across different architectures. We hope that SafeGuider can shed some light on the practical deployment of secure text-to-image systems.



## **31. Cyber Warfare During Operation Sindoor: Malware Campaign Analysis and Detection Framework**

cs.CR

Accepted for presentation at the 21st International Conference on  Information Systems Security (ICISS 2025)

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04118v1) [paper-pdf](http://arxiv.org/pdf/2510.04118v1)

**Authors**: Prakhar Paliwal, Atul Kabra, Manjesh Kumar Hanawal

**Abstract**: Rapid digitization of critical infrastructure has made cyberwarfare one of the important dimensions of modern conflicts. Attacking the critical infrastructure is an attractive pre-emptive proposition for adversaries as it can be done remotely without crossing borders. Such attacks disturb the support systems of the opponents to launch any offensive activities, crippling their fighting capabilities. Cyberattacks during cyberwarfare can not only be used to steal information, but also to spread disinformation to bring down the morale of the opponents. Recent wars in Europe, Africa, and Asia have demonstrated the scale and sophistication that the warring nations have deployed to take the early upper hand. In this work, we focus on the military action launched by India, code-named Operation Sindoor, to dismantle terror infrastructure emanating from Pakistan and the cyberattacks launched by Pakistan. In particular, we study the malware used by Pakistan APT groups to deploy Remote Access Trojans in Indian systems. We provide details of the tactics and techniques used in the RAT deployment and develop a telemetry framework to collect necessary event logs using Osquery with a custom extension. Finally, we develop a detection rule that can be readily deployed to detect the presence of the RAT or any exploitation performed by the malware.



## **32. Universal Adversarial Perturbation Attacks On Modern Behavior Cloning Policies**

cs.LG

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2502.03698v2) [paper-pdf](http://arxiv.org/pdf/2502.03698v2)

**Authors**: Akansha Kalra, Basavasagar Patil, Guanhong Tao, Daniel S. Brown

**Abstract**: Learning from Demonstration (LfD) algorithms have shown promising results in robotic manipulation tasks, but their vulnerability to offline universal perturbation attacks remains underexplored. This paper presents a comprehensive study of adversarial attacks on both classic and recently proposed algorithms, including Behavior Cloning (BC), LSTM-GMM, Implicit Behavior Cloning (IBC), Diffusion Policy (DP), and Vector-Quantizied Behavior Transformer (VQ-BET). We study the vulnerability of these methods to universal adversarial perturbations. Our experiments on several simulated robotic manipulation tasks reveal that most of the current methods are highly vulnerable to adversarial perturbations. We also show that these attacks are often transferable across algorithms, architectures, and tasks, raising concerning security vulnerabilities to black-box attacks. To the best of our knowledge, we are the first to present a systematic study of the vulnerabilities of different LfD algorithms to both white-box and black-box attacks. Our findings highlight the vulnerabilities of modern BC algorithms, paving the way for future work in addressing such limitations.



## **33. Quantifying Distributional Robustness of Agentic Tool-Selection**

cs.CR

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.03992v1) [paper-pdf](http://arxiv.org/pdf/2510.03992v1)

**Authors**: Jehyeok Yeon, Isha Chaudhary, Gagandeep Singh

**Abstract**: Large language models (LLMs) are increasingly deployed in agentic systems where they map user intents to relevant external tools to fulfill a task. A critical step in this process is tool selection, where a retriever first surfaces candidate tools from a larger pool, after which the LLM selects the most appropriate one. This pipeline presents an underexplored attack surface where errors in selection can lead to severe outcomes like unauthorized data access or denial of service, all without modifying the agent's model or code. While existing evaluations measure task performance in benign settings, they overlook the specific vulnerabilities of the tool selection mechanism under adversarial conditions. To address this gap, we introduce ToolCert, the first statistical framework that formally certifies tool selection robustness. ToolCert models tool selection as a Bernoulli success process and evaluates it against a strong, adaptive attacker who introduces adversarial tools with misleading metadata, and are iteratively refined based on the agent's previous choices. By sampling these adversarial interactions, ToolCert produces a high-confidence lower bound on accuracy, formally quantifying the agent's worst-case performance. Our evaluation with ToolCert uncovers the severe fragility: under attacks injecting deceptive tools or saturating retrieval, the certified accuracy bound drops near zero, an average performance drop of over 60% compared to non-adversarial settings. For attacks targeting the retrieval and selection stages, the certified accuracy bound plummets to less than 20% after just a single round of adversarial adaptation. ToolCert thus reveals previously unexamined security threats inherent to tool selection and provides a principled method to quantify an agent's robustness to such threats, a necessary step for the safe deployment of agentic systems.



## **34. Cascading Adversarial Bias from Injection to Distillation in Language Models**

cs.LG

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2505.24842v2) [paper-pdf](http://arxiv.org/pdf/2505.24842v2)

**Authors**: Harsh Chaudhari, Jamie Hayes, Matthew Jagielski, Ilia Shumailov, Milad Nasr, Alina Oprea

**Abstract**: Model distillation has become essential for creating smaller, deployable language models that retain larger system capabilities. However, widespread deployment raises concerns about resilience to adversarial manipulation. This paper investigates vulnerability of distilled models to adversarial injection of biased content during training. We demonstrate that adversaries can inject subtle biases into teacher models through minimal data poisoning, which propagates to student models and becomes significantly amplified. We propose two propagation modes: Untargeted Propagation, where bias affects multiple tasks, and Targeted Propagation, focusing on specific tasks while maintaining normal behavior elsewhere. With only 25 poisoned samples (0.25% poisoning rate), student models generate biased responses 76.9% of the time in targeted scenarios - higher than 69.4% in teacher models. For untargeted propagation, adversarial bias appears 6x-29x more frequently in student models on unseen tasks. We validate findings across six bias types (targeted advertisements, phishing links, narrative manipulations, insecure coding practices), various distillation methods, and different modalities spanning text and code generation. Our evaluation reveals shortcomings in current defenses - perplexity filtering, bias detection systems, and LLM-based autorater frameworks - against these attacks. Results expose significant security vulnerabilities in distilled models, highlighting need for specialized safeguards. We propose practical design principles for building effective adversarial bias mitigation strategies.



## **35. Adversarial Defence without Adversarial Defence: Enhancing Language Model Robustness via Instance-level Principal Component Removal**

cs.CL

This paper was accepted with an A-decision to Transactions of the  Association for Computational Linguistics. This version is the  pre-publication version prior to MIT Press production

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2507.21750v2) [paper-pdf](http://arxiv.org/pdf/2507.21750v2)

**Authors**: Yang Wang, Chenghao Xiao, Yizhi Li, Stuart E. Middleton, Noura Al Moubayed, Chenghua Lin

**Abstract**: Pre-trained language models (PLMs) have driven substantial progress in natural language processing but remain vulnerable to adversarial attacks, raising concerns about their robustness in real-world applications. Previous studies have sought to mitigate the impact of adversarial attacks by introducing adversarial perturbations into the training process, either implicitly or explicitly. While both strategies enhance robustness, they often incur high computational costs. In this work, we propose a simple yet effective add-on module that enhances the adversarial robustness of PLMs by removing instance-level principal components, without relying on conventional adversarial defences or perturbing the original training data. Our approach transforms the embedding space to approximate Gaussian properties, thereby reducing its susceptibility to adversarial perturbations while preserving semantic relationships. This transformation aligns embedding distributions in a way that minimises the impact of adversarial noise on decision boundaries, enhancing robustness without requiring adversarial examples or costly training-time augmentation. Evaluations on eight benchmark datasets show that our approach improves adversarial robustness while maintaining comparable before-attack accuracy to baselines, achieving a balanced trade-off between robustness and generalisation.



## **36. Deep Learning-Based Multi-Factor Authentication: A Survey of Biometric and Smart Card Integration Approaches**

cs.CR

14 pages, 3 figures, 6 tables

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.05163v1) [paper-pdf](http://arxiv.org/pdf/2510.05163v1)

**Authors**: Abdelilah Ganmati, Karim Afdel, Lahcen Koutti

**Abstract**: In the era of pervasive cyber threats and exponential growth in digital services, the inadequacy of single-factor authentication has become increasingly evident. Multi-Factor Authentication (MFA), which combines knowledge-based factors (passwords, PINs), possession-based factors (smart cards, tokens), and inherence-based factors (biometric traits), has emerged as a robust defense mechanism. Recent breakthroughs in deep learning have transformed the capabilities of biometric systems, enabling higher accuracy, resilience to spoofing, and seamless integration with hardware-based solutions. At the same time, smart card technologies have evolved to include on-chip biometric verification, cryptographic processing, and secure storage, thereby enabling compact and secure multi-factor devices. This survey presents a comprehensive synthesis of recent work (2019-2025) at the intersection of deep learning, biometrics, and smart card technologies for MFA. We analyze biometric modalities (face, fingerprint, iris, voice), review hardware-based approaches (smart cards, NFC, TPMs, secure enclaves), and highlight integration strategies for real-world applications such as digital banking, healthcare IoT, and critical infrastructure. Furthermore, we discuss the major challenges that remain open, including usability-security tradeoffs, adversarial attacks on deep learning models, privacy concerns surrounding biometric data, and the need for standardization in MFA deployment. By consolidating current advancements, limitations, and research opportunities, this survey provides a roadmap for designing secure, scalable, and user-friendly authentication frameworks.



## **37. Thought Purity: A Defense Framework For Chain-of-Thought Attack**

cs.LG

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2507.12314v2) [paper-pdf](http://arxiv.org/pdf/2507.12314v2)

**Authors**: Zihao Xue, Zhen Bi, Long Ma, Zhenlin Hu, Yan Wang, Zhenfang Liu, Qing Sheng, Jie Xiao, Jungang Lou

**Abstract**: While reinforcement learning-trained Large Reasoning Models (LRMs, e.g., Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large Language Models (LLMs) domain, their susceptibility to security threats remains a critical vulnerability. This weakness is particularly evident in Chain-of-Thought (CoT) generation processes, where adversarial methods like backdoor prompt attacks can systematically subvert the model's core reasoning mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this vulnerability through exploiting prompt controllability, simultaneously degrading both CoT safety and task performance with low-cost interventions. To address this compounded security-performance vulnerability, we propose Thought Purity (TP): a defense framework that systematically strengthens resistance to malicious content while preserving operational efficacy. Our solution achieves this through three synergistic components: (1) a safety-optimized data processing pipeline (2) reinforcement learning-enhanced rule constraints (3) adaptive monitoring metrics. Our approach establishes the first comprehensive defense mechanism against CoTA vulnerabilities in reinforcement learning-aligned reasoning systems, significantly advancing the security-functionality equilibrium for next-generation AI architectures.



## **38. From Theory to Practice: Evaluating Data Poisoning Attacks and Defenses in In-Context Learning on Social Media Health Discourse**

cs.LG

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.03636v1) [paper-pdf](http://arxiv.org/pdf/2510.03636v1)

**Authors**: Rabeya Amin Jhuma, Mostafa Mohaimen Akand Faisal

**Abstract**: This study explored how in-context learning (ICL) in large language models can be disrupted by data poisoning attacks in the setting of public health sentiment analysis. Using tweets of Human Metapneumovirus (HMPV), small adversarial perturbations such as synonym replacement, negation insertion, and randomized perturbation were introduced into the support examples. Even these minor manipulations caused major disruptions, with sentiment labels flipping in up to 67% of cases. To address this, a Spectral Signature Defense was applied, which filtered out poisoned examples while keeping the data's meaning and sentiment intact. After defense, ICL accuracy remained steady at around 46.7%, and logistic regression validation reached 100% accuracy, showing that the defense successfully preserved the dataset's integrity. Overall, the findings extend prior theoretical studies of ICL poisoning to a practical, high-stakes setting in public health discourse analysis, highlighting both the risks and potential defenses for robust LLM deployment. This study also highlights the fragility of ICL under attack and the value of spectral defenses in making AI systems more reliable for health-related social media monitoring.



## **39. Cyber Resilience of Three-phase Unbalanced Distribution System Restoration under Sparse Adversarial Attack on Load Forecasting**

eess.SY

10 pages, 7 figures

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.03635v1) [paper-pdf](http://arxiv.org/pdf/2510.03635v1)

**Authors**: Chen Chao, Zixiao Ma, Ziang Zhang

**Abstract**: System restoration is critical for power system resilience, nonetheless, its growing reliance on artificial intelligence (AI)-based load forecasting introduces significant cybersecurity risks. Inaccurate forecasts can lead to infeasible planning, voltage and frequency violations, and unsuccessful recovery of de-energized segments, yet the resilience of restoration processes to such attacks remains largely unexplored. This paper addresses this gap by quantifying how adversarially manipulated forecasts impact restoration feasibility and grid security. We develop a gradient-based sparse adversarial attack that strategically perturbs the most influential spatiotemporal inputs, exposing vulnerabilities in forecasting models while maintaining stealth. We further create a restoration-aware validation framework that embeds these compromised forecasts into a sequential restoration model and evaluates operational feasibility using an unbalanced three-phase optimal power flow formulation. Simulation results show that the proposed approach is more efficient and stealthier than baseline attacks. It reveals system-level failures, such as voltage and power ramping violations that prevent the restoration of critical loads. These findings provide actionable insights for designing cybersecurity-aware restoration planning frameworks.



## **40. Explainable but Vulnerable: Adversarial Attacks on XAI Explanation in Cybersecurity Applications**

cs.CR

10 pages, 9 figures, 4 tables

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.03623v1) [paper-pdf](http://arxiv.org/pdf/2510.03623v1)

**Authors**: Maraz Mia, Mir Mehedi A. Pritom

**Abstract**: Explainable Artificial Intelligence (XAI) has aided machine learning (ML) researchers with the power of scrutinizing the decisions of the black-box models. XAI methods enable looking deep inside the models' behavior, eventually generating explanations along with a perceived trust and transparency. However, depending on any specific XAI method, the level of trust can vary. It is evident that XAI methods can themselves be a victim of post-adversarial attacks that manipulate the expected outcome from the explanation module. Among such attack tactics, fairwashing explanation (FE), manipulation explanation (ME), and backdoor-enabled manipulation attacks (BD) are the notable ones. In this paper, we try to understand these adversarial attack techniques, tactics, and procedures (TTPs) on explanation alteration and thus the effect on the model's decisions. We have explored a total of six different individual attack procedures on post-hoc explanation methods such as SHAP (SHapley Additive exPlanations), LIME (Local Interpretable Model-agnostic Explanation), and IG (Integrated Gradients), and investigated those adversarial attacks in cybersecurity applications scenarios such as phishing, malware, intrusion, and fraudulent website detection. Our experimental study reveals the actual effectiveness of these attacks, thus providing an urgency for immediate attention to enhance the resiliency of XAI methods and their applications.



## **41. Cross-Modal Content Optimization for Steering Web Agent Preferences**

cs.AI

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.03612v1) [paper-pdf](http://arxiv.org/pdf/2510.03612v1)

**Authors**: Tanqiu Jiang, Min Bai, Nikolaos Pappas, Yanjun Qi, Sandesh Swamy

**Abstract**: Vision-language model (VLM)-based web agents increasingly power high-stakes selection tasks like content recommendation or product ranking by combining multimodal perception with preference reasoning. Recent studies reveal that these agents are vulnerable against attackers who can bias selection outcomes through preference manipulations using adversarial pop-ups, image perturbations, or content tweaks. Existing work, however, either assumes strong white-box access, with limited single-modal perturbations, or uses impractical settings. In this paper, we demonstrate, for the first time, that joint exploitation of visual and textual channels yields significantly more powerful preference manipulations under realistic attacker capabilities. We introduce Cross-Modal Preference Steering (CPS) that jointly optimizes imperceptible modifications to an item's visual and natural language descriptions, exploiting CLIP-transferable image perturbations and RLHF-induced linguistic biases to steer agent decisions. In contrast to prior studies that assume gradient access, or control over webpages, or agent memory, we adopt a realistic black-box threat setup: a non-privileged adversary can edit only their own listing's images and textual metadata, with no insight into the agent's model internals. We evaluate CPS on agents powered by state-of-the-art proprietary and open source VLMs including GPT-4.1, Qwen-2.5VL and Pixtral-Large on both movie selection and e-commerce tasks. Our results show that CPS is significantly more effective than leading baseline methods. For instance, our results show that CPS consistently outperforms baselines across all models while maintaining 70% lower detection rates, demonstrating both effectiveness and stealth. These findings highlight an urgent need for robust defenses as agentic systems play an increasingly consequential role in society.



## **42. NanoFlux: Adversarial Dual-LLM Evaluation and Distillation For Multi-Domain Reasoning**

cs.LG

preprint version

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2509.23252v2) [paper-pdf](http://arxiv.org/pdf/2509.23252v2)

**Authors**: Raviteja Anantha, Soheil Hor, Teodor Nicola Antoniu, Layne C. Price

**Abstract**: We present NanoFlux, a novel adversarial framework for generating targeted training data to improve LLM reasoning, where adversarially-generated datasets containing fewer than 200 examples outperform conventional fine-tuning approaches. The framework employs a competitive dynamic between models alternating as Attacker and Defender, supervised by a tool-augmented Judge, synthesizing multi-step questions with explanatory annotations that target specific reasoning capabilities. Fine-tuning a 4B-parameter model on NanoFlux-generated data yields performance gains across diverse domains compared to full-benchmark fine-tuning: +5.9% on mathematical reasoning (GSMHard), +3.6% on scientific reasoning (GenomeBench), and +16.6% on medical reasoning (MultiMedQA), while reducing computational requirements by 3-14x. Ablation studies reveal a non-monotonic relationship between dataset characteristics and model performance, uncovering domain-specific optimal points for question complexity and reasoning quality. NanoFlux automates training data generation through embedding-based novelty filtering, tool-augmented evaluation, and multi-hop reasoning, suggesting that future model improvements may lie in the intelligent synthesis of small, precisely targeted training datasets.



## **43. NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks**

cs.CR

Javad Rafiei Asl and Sidhant Narula are co-first authors

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.03417v1) [paper-pdf](http://arxiv.org/pdf/2510.03417v1)

**Authors**: Javad Rafiei Asl, Sidhant Narula, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: https://github.com/inspire-lab/NEXUS



## **44. Test-Time Defense Against Adversarial Attacks via Stochastic Resonance of Latent Ensembles**

cs.CV

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.03224v1) [paper-pdf](http://arxiv.org/pdf/2510.03224v1)

**Authors**: Dong Lao, Yuxiang Zhang, Haniyeh Ehsani Oskouie, Yangchao Wu, Alex Wong, Stefano Soatto

**Abstract**: We propose a test-time defense mechanism against adversarial attacks: imperceptible image perturbations that significantly alter the predictions of a model. Unlike existing methods that rely on feature filtering or smoothing, which can lead to information loss, we propose to "combat noise with noise" by leveraging stochastic resonance to enhance robustness while minimizing information loss. Our approach introduces small translational perturbations to the input image, aligns the transformed feature embeddings, and aggregates them before mapping back to the original reference image. This can be expressed in a closed-form formula, which can be deployed on diverse existing network architectures without introducing additional network modules or fine-tuning for specific attack types. The resulting method is entirely training-free, architecture-agnostic, and attack-agnostic. Empirical results show state-of-the-art robustness on image classification and, for the first time, establish a generic test-time defense for dense prediction tasks, including stereo matching and optical flow, highlighting the method's versatility and practicality. Specifically, relative to clean (unperturbed) performance, our method recovers up to 68.1% of the accuracy loss on image classification, 71.9% on stereo matching, and 29.2% on optical flow under various types of adversarial attacks.



## **45. Latent Diffusion Unlearning: Protecting Against Unauthorized Personalization Through Trajectory Shifted Perturbations**

cs.CV

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.03089v1) [paper-pdf](http://arxiv.org/pdf/2510.03089v1)

**Authors**: Naresh Kumar Devulapally, Shruti Agarwal, Tejas Gokhale, Vishnu Suresh Lokhande

**Abstract**: Text-to-image diffusion models have demonstrated remarkable effectiveness in rapid and high-fidelity personalization, even when provided with only a few user images. However, the effectiveness of personalization techniques has lead to concerns regarding data privacy, intellectual property protection, and unauthorized usage. To mitigate such unauthorized usage and model replication, the idea of generating ``unlearnable'' training samples utilizing image poisoning techniques has emerged. Existing methods for this have limited imperceptibility as they operate in the pixel space which results in images with noise and artifacts. In this work, we propose a novel model-based perturbation strategy that operates within the latent space of diffusion models. Our method alternates between denoising and inversion while modifying the starting point of the denoising trajectory: of diffusion models. This trajectory-shifted sampling ensures that the perturbed images maintain high visual fidelity to the original inputs while being resistant to inversion and personalization by downstream generative models. This approach integrates unlearnability into the framework of Latent Diffusion Models (LDMs), enabling a practical and imperceptible defense against unauthorized model adaptation. We validate our approach on four benchmark datasets to demonstrate robustness against state-of-the-art inversion attacks. Results demonstrate that our method achieves significant improvements in imperceptibility ($\sim 8 \% -10\%$ on perceptual metrics including PSNR, SSIM, and FID) and robustness ( $\sim 10\%$ on average across five adversarial settings), highlighting its effectiveness in safeguarding sensitive data.



## **46. Unveiling Unicode's Unseen Underpinnings in Undermining Authorship Attribution**

cs.CR

31 pages, 7 figures, 3 tables

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2508.15840v2) [paper-pdf](http://arxiv.org/pdf/2508.15840v2)

**Authors**: Robert Dilworth

**Abstract**: When using a public communication channel -- whether formal or informal, such as commenting or posting on social media -- end users have no expectation of privacy: they compose a message and broadcast it for the world to see. Even if an end user takes utmost precautions to anonymize their online presence -- using an alias or pseudonym; masking their IP address; spoofing their geolocation; concealing their operating system and user agent; deploying encryption; registering with a disposable phone number or email; disabling non-essential settings; revoking permissions; and blocking cookies and fingerprinting -- one obvious element still lingers: the message itself. Assuming they avoid lapses in judgment or accidental self-exposure, there should be little evidence to validate their actual identity, right? Wrong. The content of their message -- necessarily open for public consumption -- exposes an attack vector: stylometric analysis, or author profiling. In this paper, we dissect the technique of stylometry, discuss an antithetical counter-strategy in adversarial stylometry, and devise enhancements through Unicode steganography.



## **47. Privacy-Aware Design of Distributed MIMO ISAC Systems**

eess.SP

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2409.12874v3) [paper-pdf](http://arxiv.org/pdf/2409.12874v3)

**Authors**: Henrik Åkesson, Marco Gomes, Diana Pamela Moya Osorio

**Abstract**: Integrated Sensing and Communication (ISAC) systems raise unprecedented challenges regarding security and privacy since related applications involve the gathering of sensitive, identifiable information about people and the environment, which can lead to privacy leakage. Privacy-aware measures can steer the design of ISAC systems to prevent privacy violations. Thus, we explore this perspective for the design of distributed massive multiple-input multiple-output ISAC systems. For this purpose, we introduce an adversarial model where a malicious user exploits the interference from ISAC signals to extract sensing information. To mitigate this threat, we propose an iterative privacy-aware framework of two blocks: precoder design and access point selection. The precoder design aims to minimize the mutual information between the sensing and communication signals by imposing constraints on sensing and communication performance and maximum transmit power. The access point selection also aims to minimize the mutual information between communication and sensing signals by strategically selecting access points that transmit ISAC signals, and sensing receivers. Results show a reduction in the effectiveness of the attack measured by the probability of detection of the attacker.



## **48. Rethinking the Vulnerability of Concept Erasure and a New Method**

cs.LG

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2502.17537v3) [paper-pdf](http://arxiv.org/pdf/2502.17537v3)

**Authors**: Alex D. Richardson, Kaicheng Zhang, Lucas Beerens, Dongdong Chen

**Abstract**: The proliferation of text-to-image diffusion models has raised significant privacy and security concerns, particularly regarding the generation of copyrighted or harmful images. In response, concept erasure (defense) methods have been developed to "unlearn" specific concepts through post-hoc finetuning. However, recent concept restoration (attack) methods have demonstrated that these supposedly erased concepts can be recovered using adversarially crafted prompts, revealing a critical vulnerability in current defense mechanisms. In this work, we first investigate the fundamental sources of adversarial vulnerability and reveal that vulnerabilities are pervasive in the prompt embedding space of concept-erased models, a characteristic inherited from the original pre-unlearned model. Furthermore, we introduce **RECORD**, a novel coordinate-descent-based restoration algorithm that consistently outperforms existing restoration methods by up to 17.8 times. We conduct extensive experiments to assess its compute-performance tradeoff and propose acceleration strategies.



## **49. Untargeted Jailbreak Attack**

cs.CR

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.02999v1) [paper-pdf](http://arxiv.org/pdf/2510.02999v1)

**Authors**: Xinzhe Huang, Wenjing Hu, Tianhang Zheng, Kedong Xiu, Xiaojun Jia, Di Wang, Zhan Qin, Kui Ren

**Abstract**: Existing gradient-based jailbreak attacks on Large Language Models (LLMs), such as Greedy Coordinate Gradient (GCG) and COLD-Attack, typically optimize adversarial suffixes to align the LLM output with a predefined target response. However, by restricting the optimization objective as inducing a predefined target, these methods inherently constrain the adversarial search space, which limit their overall attack efficacy. Furthermore, existing methods typically require a large number of optimization iterations to fulfill the large gap between the fixed target and the original model response, resulting in low attack efficiency.   To overcome the limitations of targeted jailbreak attacks, we propose the first gradient-based untargeted jailbreak attack (UJA), aiming to elicit an unsafe response without enforcing any predefined patterns. Specifically, we formulate an untargeted attack objective to maximize the unsafety probability of the LLM response, which can be quantified using a judge model. Since the objective is non-differentiable, we further decompose it into two differentiable sub-objectives for optimizing an optimal harmful response and the corresponding adversarial prompt, with a theoretical analysis to validate the decomposition. In contrast to targeted jailbreak attacks, UJA's unrestricted objective significantly expands the search space, enabling a more flexible and efficient exploration of LLM vulnerabilities.Extensive evaluations demonstrate that \textsc{UJA} can achieve over 80\% attack success rates against recent safety-aligned LLMs with only 100 optimization iterations, outperforming the state-of-the-art gradient-based attacks such as I-GCG and COLD-Attack by over 20\%.



## **50. Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain**

cs.CR

27 pages

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.05159v1) [paper-pdf](http://arxiv.org/pdf/2510.05159v1)

**Authors**: Léo Boisvert, Abhay Puri, Chandra Kiran Reddy Evuru, Nicolas Chapados, Quentin Cappart, Alexandre Lacoste, Krishnamurthy Dj Dvijotham, Alexandre Drouin

**Abstract**: The practice of fine-tuning AI agents on data from their own interactions--such as web browsing or tool use--, while being a strong general recipe for improving agentic capabilities, also introduces a critical security vulnerability within the AI supply chain. In this work, we show that adversaries can easily poison the data collection pipeline to embed hard-to-detect backdoors that are triggerred by specific target phrases, such that when the agent encounters these triggers, it performs an unsafe or malicious action. We formalize and validate three realistic threat models targeting different layers of the supply chain: 1) direct poisoning of fine-tuning data, where an attacker controls a fraction of the training traces; 2) environmental poisoning, where malicious instructions are injected into webpages scraped or tools called while creating training data; and 3) supply chain poisoning, where a pre-backdoored base model is fine-tuned on clean data to improve its agentic capabilities. Our results are stark: by poisoning as few as 2% of the collected traces, an attacker can embed a backdoor causing an agent to leak confidential user information with over 80% success when a specific trigger is present. This vulnerability holds across all three threat models. Furthermore, we demonstrate that prominent safeguards, including two guardrail models and one weight-based defense, fail to detect or prevent the malicious behavior. These findings highlight an urgent threat to agentic AI development and underscore the critical need for rigorous security vetting of data collection processes and end-to-end model supply chains.



