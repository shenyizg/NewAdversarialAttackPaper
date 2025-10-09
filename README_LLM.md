# Latest Large Language Model Attack Papers
**update at 2025-10-09 08:23:10**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. An Embarrassingly Simple Defense Against LLM Abliteration Attacks**

cs.CL

preprint - under review

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2505.19056v2) [paper-pdf](http://arxiv.org/pdf/2505.19056v2)

**Authors**: Harethah Abu Shairah, Hasan Abed Al Kader Hammoud, Bernard Ghanem, George Turkiyyah

**Abstract**: Large language models (LLMs) are typically aligned to refuse harmful instructions through safety fine-tuning. A recent attack, termed abliteration, identifies and suppresses the single latent direction most responsible for refusal behavior, thereby enabling models to generate harmful content. We propose a defense that fundamentally alters how models express refusal. We construct an extended-refusal dataset in which responses to harmful prompts provide detailed justifications before refusing, distributing the refusal signal across multiple token positions. Fine-tuning Llama-2-7B-Chat and Qwen2.5-Instruct (1.5B and 3B parameters) on this dataset yields models that maintain high refusal rates under abliteration: refusal rates drop by at most 10%, compared to 70-80% drops in baseline models. Comprehensive evaluations of safety and utility demonstrate that extended-refusal fine-tuning effectively neutralizes abliteration attacks while preserving general model performance and enhancing robustness across multiple alignment scenarios.



## **2. SAFER: Advancing Safety Alignment via Efficient Ex-Ante Reasoning**

cs.CL

22 pages, 5 figures

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2504.02725v2) [paper-pdf](http://arxiv.org/pdf/2504.02725v2)

**Authors**: Kehua Feng, Keyan Ding, Yuhao Wang, Menghan Li, Fanjunduo Wei, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose SAFER, a framework for Safety Alignment via eFficient Ex-Ante Reasoning. Our approach instantiates structured Ex-Ante reasoning through initial assessment, rule verification, and path calibration, and embeds predefined safety rules to provide transparent and verifiable safety judgments. Specifically, our approach consists of two training stages: (1) supervised fine-tuning with synthetic traces to teach the multi-stage Ex-Ante reasoning, and (2) step-level reasoning preference optimization to jointly enhance safety, utility, and efficiency. Experiments on multiple open-source LLMs demonstrate that SAFER significantly enhances safety performance while maintaining helpfulness and response efficiency.



## **3. Geometry-Guided Adversarial Prompt Detection via Curvature and Local Intrinsic Dimension**

cs.CL

40 Pages, 6 figues

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2503.03502v2) [paper-pdf](http://arxiv.org/pdf/2503.03502v2)

**Authors**: Canaan Yung, Hanxun Huang, Christopher Leckie, Sarah Erfani

**Abstract**: Adversarial prompts are capable of jailbreaking frontier large language models (LLMs) and inducing undesirable behaviours, posing a significant obstacle to their safe deployment. Current mitigation strategies primarily rely on activating built-in defence mechanisms or fine-tuning LLMs, both of which are computationally expensive and can sacrifice model utility. In contrast, detection-based approaches are more efficient and practical for deployment in real-world applications. However, the fundamental distinctions between adversarial and benign prompts remain poorly understood. In this work, we introduce CurvaLID, a novel defence framework that efficiently detects adversarial prompts by leveraging their geometric properties. It is agnostic to the type of LLM, offering a unified detection framework across diverse adversarial prompts and LLM architectures. CurvaLID builds on the geometric analysis of text prompts to uncover their underlying differences. We theoretically extend the concept of curvature via the Whewell equation into an $n$-dimensional word embedding space, enabling us to quantify local geometric properties, including semantic shifts and curvature in the underlying manifolds. To further enhance our solution, we leverage Local Intrinsic Dimensionality (LID) to capture complementary geometric features of text prompts within adversarial subspaces. Our findings show that adversarial prompts exhibit distinct geometric signatures from benign prompts, enabling CurvaLID to achieve near-perfect classification and outperform state-of-the-art detectors in adversarial prompt detection. CurvaLID provides a reliable and efficient safeguard against malicious queries as a model-agnostic method that generalises across multiple LLMs and attack families.



## **4. Towards Reliable and Practical LLM Security Evaluations via Bayesian Modelling**

cs.CR

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05709v1) [paper-pdf](http://arxiv.org/pdf/2510.05709v1)

**Authors**: Mary Llewellyn, Annie Gray, Josh Collyer, Michael Harries

**Abstract**: Before adopting a new large language model (LLM) architecture, it is critical to understand vulnerabilities accurately. Existing evaluations can be difficult to trust, often drawing conclusions from LLMs that are not meaningfully comparable, relying on heuristic inputs or employing metrics that fail to capture the inherent uncertainty. In this paper, we propose a principled and practical end-to-end framework for evaluating LLM vulnerabilities to prompt injection attacks. First, we propose practical approaches to experimental design, tackling unfair LLM comparisons by considering two practitioner scenarios: when training an LLM and when deploying a pre-trained LLM. Second, we address the analysis of experiments and propose a Bayesian hierarchical model with embedding-space clustering. This model is designed to improve uncertainty quantification in the common scenario that LLM outputs are not deterministic, test prompts are designed imperfectly, and practitioners only have a limited amount of compute to evaluate vulnerabilities. We show the improved inferential capabilities of the model in several prompt injection attack settings. Finally, we demonstrate the pipeline to evaluate the security of Transformer versus Mamba architectures. Our findings show that consideration of output variability can suggest less definitive findings. However, for some attacks, we find notably increased Transformer and Mamba-variant vulnerabilities across LLMs with the same training data or mathematical ability.



## **5. Membership Inference Attacks on Tokenizers of Large Language Models**

cs.CR

Code is available at: https://github.com/mengtong0110/Tokenizer-MIA

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05699v1) [paper-pdf](http://arxiv.org/pdf/2510.05699v1)

**Authors**: Meng Tong, Yuntao Du, Kejiang Chen, Weiming Zhang, Ninghui Li

**Abstract**: Membership inference attacks (MIAs) are widely used to assess the privacy risks associated with machine learning models. However, when these attacks are applied to pre-trained large language models (LLMs), they encounter significant challenges, including mislabeled samples, distribution shifts, and discrepancies in model size between experimental and real-world settings. To address these limitations, we introduce tokenizers as a new attack vector for membership inference. Specifically, a tokenizer converts raw text into tokens for LLMs. Unlike full models, tokenizers can be efficiently trained from scratch, thereby avoiding the aforementioned challenges. In addition, the tokenizer's training data is typically representative of the data used to pre-train LLMs. Despite these advantages, the potential of tokenizers as an attack vector remains unexplored. To this end, we present the first study on membership leakage through tokenizers and explore five attack methods to infer dataset membership. Extensive experiments on millions of Internet samples reveal the vulnerabilities in the tokenizers of state-of-the-art LLMs. To mitigate this emerging risk, we further propose an adaptive defense. Our findings highlight tokenizers as an overlooked yet critical privacy threat, underscoring the urgent need for privacy-preserving mechanisms specifically designed for them.



## **6. Bypassing Prompt Guards in Production with Controlled-Release Prompting**

cs.LG

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.01529v2) [paper-pdf](http://arxiv.org/pdf/2510.01529v2)

**Authors**: Jaiden Fairoze, Sanjam Garg, Keewoo Lee, Mingyuan Wang

**Abstract**: As large language models (LLMs) advance, ensuring AI safety and alignment is paramount. One popular approach is prompt guards, lightweight mechanisms designed to filter malicious queries while being easy to implement and update. In this work, we introduce a new attack that circumvents such prompt guards, highlighting their limitations. Our method consistently jailbreaks production models while maintaining response quality, even under the highly protected chat interfaces of Google Gemini (2.5 Flash/Pro), DeepSeek Chat (DeepThink), Grok (3), and Mistral Le Chat (Magistral). The attack exploits a resource asymmetry between the prompt guard and the main LLM, encoding a jailbreak prompt that lightweight guards cannot decode but the main model can. This reveals an attack surface inherent to lightweight prompt guards in modern LLM architectures and underscores the need to shift defenses from blocking malicious inputs to preventing malicious outputs. We additionally identify other critical alignment issues, such as copyrighted data extraction, training data extraction, and malicious response leakage during thinking.



## **7. AutoPentester: An LLM Agent-based Framework for Automated Pentesting**

cs.CR

IEEE TrustCom 2025 10 pages

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05605v1) [paper-pdf](http://arxiv.org/pdf/2510.05605v1)

**Authors**: Yasod Ginige, Akila Niroshan, Sajal Jain, Suranga Seneviratne

**Abstract**: Penetration testing and vulnerability assessment are essential industry practices for safeguarding computer systems. As cyber threats grow in scale and complexity, the demand for pentesting has surged, surpassing the capacity of human professionals to meet it effectively. With advances in AI, particularly Large Language Models (LLMs), there have been attempts to automate the pentesting process. However, existing tools such as PentestGPT are still semi-manual, requiring significant professional human interaction to conduct pentests. To this end, we propose a novel LLM agent-based framework, AutoPentester, which automates the pentesting process. Given a target IP, AutoPentester automatically conducts pentesting steps using common security tools in an iterative process. It can dynamically generate attack strategies based on the tool outputs from the previous iteration, mimicking the human pentester approach. We evaluate AutoPentester using Hack The Box and custom-made VMs, comparing the results with the state-of-the-art PentestGPT. Results show that AutoPentester achieves a 27.0% better subtask completion rate and 39.5% more vulnerability coverage with fewer steps. Most importantly, it requires significantly fewer human interactions and interventions compared to PentestGPT. Furthermore, we recruit a group of security industry professional volunteers for a user survey and perform a qualitative analysis to evaluate AutoPentester against industry practices and compare it with PentestGPT. On average, AutoPentester received a score of 3.93 out of 5 based on user reviews, which was 19.8% higher than PentestGPT.



## **8. A Middle Path for On-Premises LLM Deployment: Preserving Privacy Without Sacrificing Model Confidentiality**

cs.LG

8 pages for main content of the paper

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2410.11182v3) [paper-pdf](http://arxiv.org/pdf/2410.11182v3)

**Authors**: Hanbo Huang, Yihan Li, Bowen Jiang, Bo Jiang, Lin Liu, Ruoyu Sun, Zhuotao Liu, Shiyu Liang

**Abstract**: Privacy-sensitive users require deploying large language models (LLMs) within their own infrastructure (on-premises) to safeguard private data and enable customization. However, vulnerabilities in local environments can lead to unauthorized access and potential model theft. To address this, prior research on small models has explored securing only the output layer within hardware-secured devices to balance model confidentiality and customization. Yet this approach fails to protect LLMs effectively. In this paper, we discover that (1) query-based distillation attacks targeting the secured top layer can produce a functionally equivalent replica of the victim model; (2) securing the same number of layers, bottom layers before a transition layer provide stronger protection against distillation attacks than top layers, with comparable effects on customization performance; and (3) the number of secured layers creates a trade-off between protection and customization flexibility. Based on these insights, we propose SOLID, a novel deployment framework that secures a few bottom layers in a secure environment and introduces an efficient metric to optimize the trade-off by determining the ideal number of hidden layers. Extensive experiments on five models (1.3B to 70B parameters) demonstrate that SOLID outperforms baselines, achieving a better balance between protection and downstream customization.



## **9. (Token-Level) \textbf{InfoRMIA}: Stronger Membership Inference and Memorization Assessment for LLMs**

cs.LG

**SubmitDate**: 2025-10-07    [abs](http://arxiv.org/abs/2510.05582v1) [paper-pdf](http://arxiv.org/pdf/2510.05582v1)

**Authors**: Jiashu Tao, Reza Shokri

**Abstract**: Machine learning models are known to leak sensitive information, as they inevitably memorize (parts of) their training data. More alarmingly, large language models (LLMs) are now trained on nearly all available data, which amplifies the magnitude of information leakage and raises serious privacy risks. Hence, it is more crucial than ever to quantify privacy risk before the release of LLMs. The standard method to quantify privacy is via membership inference attacks, where the state-of-the-art approach is the Robust Membership Inference Attack (RMIA). In this paper, we present InfoRMIA, a principled information-theoretic formulation of membership inference. Our method consistently outperforms RMIA across benchmarks while also offering improved computational efficiency.   In the second part of the paper, we identify the limitations of treating sequence-level membership inference as the gold standard for measuring leakage. We propose a new perspective for studying membership and memorization in LLMs: token-level signals and analyses. We show that a simple token-based InfoRMIA can pinpoint which tokens are memorized within generated outputs, thereby localizing leakage from the sequence level down to individual tokens, while achieving stronger sequence-level inference power on LLMs. This new scope rethinks privacy in LLMs and can lead to more targeted mitigation, such as exact unlearning.



## **10. Adversarial Reinforcement Learning for Large Language Model Agent Safety**

cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05442v1) [paper-pdf](http://arxiv.org/pdf/2510.05442v1)

**Authors**: Zizhao Wang, Dingcheng Li, Vaishakh Keshava, Phillip Wallis, Ananth Balashankar, Peter Stone, Lukas Rutishauser

**Abstract**: Large Language Model (LLM) agents can leverage tools such as Google Search to complete complex tasks. However, this tool usage introduces the risk of indirect prompt injections, where malicious instructions hidden in tool outputs can manipulate the agent, posing security risks like data leakage. Current defense strategies typically rely on fine-tuning LLM agents on datasets of known attacks. However, the generation of these datasets relies on manually crafted attack patterns, which limits their diversity and leaves agents vulnerable to novel prompt injections. To address this limitation, we propose Adversarial Reinforcement Learning for Agent Safety (ARLAS), a novel framework that leverages adversarial reinforcement learning (RL) by formulating the problem as a two-player zero-sum game. ARLAS co-trains two LLMs: an attacker that learns to autonomously generate diverse prompt injections and an agent that learns to defend against them while completing its assigned tasks. To ensure robustness against a wide range of attacks and to prevent cyclic learning, we employ a population-based learning framework that trains the agent to defend against all previous attacker checkpoints. Evaluated on BrowserGym and AgentDojo, agents fine-tuned with ARLAS achieve a significantly lower attack success rate than the original model while also improving their task success rate. Our analysis further confirms that the adversarial process generates a diverse and challenging set of attacks, leading to a more robust agent compared to the base model.



## **11. AutoDAN-Reasoning: Enhancing Strategies Exploration based Jailbreak Attacks with Test-Time Scaling**

cs.CR

Technical report. Code is available at  https://github.com/SaFoLab-WISC/AutoDAN-Reasoning

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05379v1) [paper-pdf](http://arxiv.org/pdf/2510.05379v1)

**Authors**: Xiaogeng Liu, Chaowei Xiao

**Abstract**: Recent advancements in jailbreaking large language models (LLMs), such as AutoDAN-Turbo, have demonstrated the power of automated strategy discovery. AutoDAN-Turbo employs a lifelong learning agent to build a rich library of attack strategies from scratch. While highly effective, its test-time generation process involves sampling a strategy and generating a single corresponding attack prompt, which may not fully exploit the potential of the learned strategy library. In this paper, we propose to further improve the attack performance of AutoDAN-Turbo through test-time scaling. We introduce two distinct scaling methods: Best-of-N and Beam Search. The Best-of-N method generates N candidate attack prompts from a sampled strategy and selects the most effective one based on a scorer model. The Beam Search method conducts a more exhaustive search by exploring combinations of strategies from the library to discover more potent and synergistic attack vectors. According to the experiments, the proposed methods significantly boost performance, with Beam Search increasing the attack success rate by up to 15.6 percentage points on Llama-3.1-70B-Instruct and achieving a nearly 60\% relative improvement against the highly robust GPT-o4-mini compared to the vanilla method.



## **12. DP-Adam-AC: Privacy-preserving Fine-Tuning of Localizable Language Models Using Adam Optimization with Adaptive Clipping**

cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05288v1) [paper-pdf](http://arxiv.org/pdf/2510.05288v1)

**Authors**: Ruoxing Yang

**Abstract**: Large language models (LLMs) such as ChatGPT have evolved into powerful and ubiquitous tools. Fine-tuning on small datasets allows LLMs to acquire specialized skills for specific tasks efficiently. Although LLMs provide great utility in both general and task-specific use cases, they are limited by two security-related concerns. First, traditional LLM hardware requirements make them infeasible to run locally on consumer-grade devices. A remote network connection with the LLM provider's server is usually required, making the system vulnerable to network attacks. Second, fine-tuning an LLM for a sensitive task may involve sensitive data. Non-private fine-tuning algorithms produce models vulnerable to training data reproduction attacks. Our work addresses these security concerns by enhancing differentially private optimization algorithms and applying them to fine-tune localizable language models. We introduce adaptable gradient clipping along with other engineering enhancements to the standard DP-Adam optimizer to create DP-Adam-AC. We use our optimizer to fine-tune examples of two localizable LLM designs, small language model (Qwen2.5-0.5B) and 1.58 bit quantization (Bitnet-b1.58-2B). We demonstrate promising improvements in loss through experimentation with two synthetic datasets.



## **13. Proactive defense against LLM Jailbreak**

cs.CR

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.05052v1) [paper-pdf](http://arxiv.org/pdf/2510.05052v1)

**Authors**: Weiliang Zhao, Jinjun Peng, Daniel Ben-Levi, Zhou Yu, Junfeng Yang

**Abstract**: The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, primarily reactive and static, often fail to counter these search-based attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead autonomous jailbreaking processes. Our core idea is to intentionally provide adversaries with "spurious responses" that appear to be results of successful jailbreak attacks but contain no actual harmful content. These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak. By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, our method consistently and significantly reduces attack success rates by up to 92\%. When combined with other defense frameworks, it further reduces the success rate of the latest attack strategies to 0\%. ProAct represents an orthogonal defense strategy that can serve as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks.



## **14. Rethinking Exact Unlearning under Exposure: Extracting Forgotten Data under Exact Unlearning in Large Language Model**

cs.LG

Accepted by Neurips 2025

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2505.24379v2) [paper-pdf](http://arxiv.org/pdf/2505.24379v2)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.



## **15. SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests**

cs.CL

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.04891v1) [paper-pdf](http://arxiv.org/pdf/2510.04891v1)

**Authors**: Punya Syon Pandey, Hai Son Le, Devansh Bhardwaj, Rada Mihalcea, Zhijing Jin

**Abstract**: Large language models (LLMs) are increasingly deployed in contexts where their failures can have direct sociopolitical consequences. Yet, existing safety benchmarks rarely test vulnerabilities in domains such as political manipulation, propaganda and disinformation generation, or surveillance and information control. We introduce SocialHarmBench, a dataset of 585 prompts spanning 7 sociopolitical categories and 34 countries, designed to surface where LLMs most acutely fail in politically charged contexts. Our evaluations reveal several shortcomings: open-weight models exhibit high vulnerability to harmful compliance, with Mistral-7B reaching attack success rates as high as 97% to 98% in domains such as historical revisionism, propaganda, and political manipulation. Moreover, temporal and geographic analyses show that LLMs are most fragile when confronted with 21st-century or pre-20th-century contexts, and when responding to prompts tied to regions such as Latin America, the USA, and the UK. These findings demonstrate that current safeguards fail to generalize to high-stakes sociopolitical settings, exposing systematic biases and raising concerns about the reliability of LLMs in preserving human rights and democratic values. We share the SocialHarmBench benchmark at https://huggingface.co/datasets/psyonp/SocialHarmBench.



## **16. Can We Infer Confidential Properties of Training Data from LLMs?**

cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2506.10364v3) [paper-pdf](http://arxiv.org/pdf/2506.10364v3)

**Authors**: Pengrun Huang, Chhavi Yadav, Kamalika Chaudhuri, Ruihan Wu

**Abstract**: Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets to support applications in fields such as healthcare, finance, and law. These fine-tuning datasets often have sensitive and confidential dataset-level properties -- such as patient demographics or disease prevalence -- that are not intended to be revealed. While prior work has studied property inference attacks on discriminative models (e.g., image classification models) and generative models (e.g., GANs for image data), it remains unclear if such attacks transfer to LLMs. In this work, we introduce PropInfer, a benchmark task for evaluating property inference in LLMs under two fine-tuning paradigms: question-answering and chat-completion. Built on the ChatDoctor dataset, our benchmark includes a range of property types and task configurations. We further propose two tailored attacks: a prompt-based generation attack and a shadow-model attack leveraging word frequency signals. Empirical evaluations across multiple pretrained LLMs show the success of our attacks, revealing a previously unrecognized vulnerability in LLMs.



## **17. Sampling-aware Adversarial Attacks Against Large Language Models**

cs.LG

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2507.04446v3) [paper-pdf](http://arxiv.org/pdf/2507.04446v3)

**Authors**: Tim Beyer, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point greedy generations, overlooking the inherently stochastic nature of LLMs and overestimating robustness. We show that for the goal of eliciting harmful responses, repeated sampling of model outputs during the attack complements prompt optimization and serves as a strong and efficient attack vector. By casting attacks as a resource allocation problem between optimization and sampling, we determine compute-optimal trade-offs and show that integrating sampling into existing attacks boosts success rates by up to 37\% and improves efficiency by up to two orders of magnitude. We further analyze how distributions of output harmfulness evolve during an adversarial attack, discovering that many common optimization strategies have little effect on output harmfulness. Finally, we introduce a label-free proof-of-concept objective based on entropy maximization, demonstrating how our sampling-aware perspective enables new optimization targets. Overall, our findings establish the importance of sampling in attacks to accurately assess and strengthen LLM safety at scale.



## **18. Unified Threat Detection and Mitigation Framework (UTDMF): Combating Prompt Injection, Deception, and Bias in Enterprise-Scale Transformers**

cs.CR

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.04528v1) [paper-pdf](http://arxiv.org/pdf/2510.04528v1)

**Authors**: Santhosh KumarRavindran

**Abstract**: The rapid adoption of large language models (LLMs) in enterprise systems exposes vulnerabilities to prompt injection attacks, strategic deception, and biased outputs, threatening security, trust, and fairness. Extending our adversarial activation patching framework (arXiv:2507.09406), which induced deception in toy networks at a 23.9% rate, we introduce the Unified Threat Detection and Mitigation Framework (UTDMF), a scalable, real-time pipeline for enterprise-grade models like Llama-3.1 (405B), GPT-4o, and Claude-3.5. Through 700+ experiments per model, UTDMF achieves: (1) 92% detection accuracy for prompt injection (e.g., jailbreaking); (2) 65% reduction in deceptive outputs via enhanced patching; and (3) 78% improvement in fairness metrics (e.g., demographic bias). Novel contributions include a generalized patching algorithm for multi-threat detection, three groundbreaking hypotheses on threat interactions (e.g., threat chaining in enterprise workflows), and a deployment-ready toolkit with APIs for enterprise integration.



## **19. P2P: A Poison-to-Poison Remedy for Reliable Backdoor Defense in LLMs**

cs.CR

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2510.04503v1) [paper-pdf](http://arxiv.org/pdf/2510.04503v1)

**Authors**: Shuai Zhao, Xinyi Wu, Shiqian Zhao, Xiaobao Wu, Zhongliang Guo, Yanhao Jia, Anh Tuan Luu

**Abstract**: During fine-tuning, large language models (LLMs) are increasingly vulnerable to data-poisoning backdoor attacks, which compromise their reliability and trustworthiness. However, existing defense strategies suffer from limited generalization: they only work on specific attack types or task settings. In this study, we propose Poison-to-Poison (P2P), a general and effective backdoor defense algorithm. P2P injects benign triggers with safe alternative labels into a subset of training samples and fine-tunes the model on this re-poisoned dataset by leveraging prompt-based learning. This enforces the model to associate trigger-induced representations with safe outputs, thereby overriding the effects of original malicious triggers. Thanks to this robust and generalizable trigger-based fine-tuning, P2P is effective across task settings and attack types. Theoretically and empirically, we show that P2P can neutralize malicious backdoors while preserving task performance. We conduct extensive experiments on classification, mathematical reasoning, and summary generation tasks, involving multiple state-of-the-art LLMs. The results demonstrate that our P2P algorithm significantly reduces the attack success rate compared with baseline models. We hope that the P2P can serve as a guideline for defending against backdoor attacks and foster the development of a secure and trustworthy LLM community.



## **20. Who's the Mole? Modeling and Detecting Intention-Hiding Malicious Agents in LLM-Based Multi-Agent Systems**

cs.MA

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2507.04724v2) [paper-pdf](http://arxiv.org/pdf/2507.04724v2)

**Authors**: Yizhe Xie, Congcong Zhu, Xinyue Zhang, Tianqing Zhu, Dayong Ye, Minghao Wang, Chi Liu

**Abstract**: Multi-agent systems powered by Large Language Models (LLM-MAS) have demonstrated remarkable capabilities in collaborative problem-solving. However, their deployment also introduces new security risks. Existing research on LLM-based agents has primarily examined single-agent scenarios, while the security of multi-agent systems remains largely unexplored. To address this gap, we present a systematic study of intention-hiding threats in LLM-MAS. We design four representative attack paradigms that subtly disrupt task completion while maintaining a high degree of stealth, and evaluate them under centralized, decentralized, and layered communication structures. Experimental results show that these attacks are highly disruptive and can easily evade existing defense mechanisms. To counter these threats, we propose AgentXposed, a psychology-inspired detection framework. AgentXposed draws on the HEXACO personality model, which characterizes agents through psychological trait dimensions, and the Reid interrogation technique, a structured method for eliciting concealed intentions. By combining progressive questionnaire probing with behavior-based inter-agent monitoring, the framework enables the proactive identification of malicious agents before harmful actions are carried out. Extensive experiments across six datasets against both our proposed attacks and two baseline threats demonstrate that AgentXposed effectively detects diverse forms of malicious behavior, achieving strong robustness across multiple communication settings.



## **21. AutoBnB-RAG: Enhancing Multi-Agent Incident Response with Retrieval-Augmented Generation**

cs.CL

**SubmitDate**: 2025-10-06    [abs](http://arxiv.org/abs/2508.13118v2) [paper-pdf](http://arxiv.org/pdf/2508.13118v2)

**Authors**: Zefang Liu, Arman Anwar

**Abstract**: Incident response (IR) requires fast, coordinated, and well-informed decision-making to contain and mitigate cyber threats. While large language models (LLMs) have shown promise as autonomous agents in simulated IR settings, their reasoning is often limited by a lack of access to external knowledge. In this work, we present AutoBnB-RAG, an extension of the AutoBnB framework that incorporates retrieval-augmented generation (RAG) into multi-agent incident response simulations. Built on the Backdoors & Breaches (B&B) tabletop game environment, AutoBnB-RAG enables agents to issue retrieval queries and incorporate external evidence during collaborative investigations. We introduce two retrieval settings: one grounded in curated technical documentation (RAG-Wiki), and another using narrative-style incident reports (RAG-News). We evaluate performance across eight team structures, including newly introduced argumentative configurations designed to promote critical reasoning. To validate practical utility, we also simulate real-world cyber incidents based on public breach reports, demonstrating AutoBnB-RAG's ability to reconstruct complex multi-stage attacks. Our results show that retrieval augmentation improves decision quality and success rates across diverse organizational models. This work demonstrates the value of integrating retrieval mechanisms into LLM-based multi-agent systems for cybersecurity decision-making.



## **22. SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations**

cs.CL

Accepted at NeurIPS 2025. Code is available at  https://github.com/Buyun-Liang/SECA

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04398v1) [paper-pdf](http://arxiv.org/pdf/2510.04398v1)

**Authors**: Buyun Liang, Liangzu Peng, Jinqi Luo, Darshan Thaker, Kwan Ho Ryan Chan, René Vidal

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no constraint violations compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at https://github.com/Buyun-Liang/SECA.



## **23. Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models**

cs.CL

15 pages total (9 pages main text + 4 pages appendix + references),  12 figures, preprint version. The final version may differ

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04347v1) [paper-pdf](http://arxiv.org/pdf/2510.04347v1)

**Authors**: Anindya Sundar Das, Kangjie Chen, Monowar Bhuyan

**Abstract**: Pre-trained language models have achieved remarkable success across a wide range of natural language processing (NLP) tasks, particularly when fine-tuned on large, domain-relevant datasets. However, they remain vulnerable to backdoor attacks, where adversaries embed malicious behaviors using trigger patterns in the training data. These triggers remain dormant during normal usage, but, when activated, can cause targeted misclassifications. In this work, we investigate the internal behavior of backdoored pre-trained encoder-based language models, focusing on the consistent shift in attention and gradient attribution when processing poisoned inputs; where the trigger token dominates both attention and gradient signals, overriding the surrounding context. We propose an inference-time defense that constructs anomaly scores by combining token-level attention and gradient information. Extensive experiments on text classification tasks across diverse backdoor attack scenarios demonstrate that our method significantly reduces attack success rates compared to existing baselines. Furthermore, we provide an interpretability-driven analysis of the scoring mechanism, shedding light on trigger localization and the robustness of the proposed defense.



## **24. VortexPIA: Indirect Prompt Injection Attack against LLMs for Efficient Extraction of User Privacy**

cs.CR

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04261v1) [paper-pdf](http://arxiv.org/pdf/2510.04261v1)

**Authors**: Yu Cui, Sicheng Pan, Yifei Liu, Haibin Zhang, Cong Zuo

**Abstract**: Large language models (LLMs) have been widely deployed in Conversational AIs (CAIs), while exposing privacy and security threats. Recent research shows that LLM-based CAIs can be manipulated to extract private information from human users, posing serious security threats. However, the methods proposed in that study rely on a white-box setting that adversaries can directly modify the system prompt. This condition is unlikely to hold in real-world deployments. The limitation raises a critical question: can unprivileged attackers still induce such privacy risks in practical LLM-integrated applications? To address this question, we propose \textsc{VortexPIA}, a novel indirect prompt injection attack that induces privacy extraction in LLM-integrated applications under black-box settings. By injecting token-efficient data containing false memories, \textsc{VortexPIA} misleads LLMs to actively request private information in batches. Unlike prior methods, \textsc{VortexPIA} allows attackers to flexibly define multiple categories of sensitive data. We evaluate \textsc{VortexPIA} on six LLMs, covering both traditional and reasoning LLMs, across four benchmark datasets. The results show that \textsc{VortexPIA} significantly outperforms baselines and achieves state-of-the-art (SOTA) performance. It also demonstrates efficient privacy requests, reduced token consumption, and enhanced robustness against defense mechanisms. We further validate \textsc{VortexPIA} on multiple realistic open-source LLM-integrated applications, demonstrating its practical effectiveness.



## **25. AgentTypo: Adaptive Typographic Prompt Injection Attacks against Black-box Multimodal Agents**

cs.CR

13 pages, 8 figures. Submitted to IEEE Transactions on Information  Forensics & Security

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.04257v1) [paper-pdf](http://arxiv.org/pdf/2510.04257v1)

**Authors**: Yanjie Li, Yiming Cao, Dong Wang, Bin Xiao

**Abstract**: Multimodal agents built on large vision-language models (LVLMs) are increasingly deployed in open-world settings but remain highly vulnerable to prompt injection, especially through visual inputs. We introduce AgentTypo, a black-box red-teaming framework that mounts adaptive typographic prompt injection by embedding optimized text into webpage images. Our automatic typographic prompt injection (ATPI) algorithm maximizes prompt reconstruction by substituting captioners while minimizing human detectability via a stealth loss, with a Tree-structured Parzen Estimator guiding black-box optimization over text placement, size, and color. To further enhance attack strength, we develop AgentTypo-pro, a multi-LLM system that iteratively refines injection prompts using evaluation feedback and retrieves successful past examples for continual learning. Effective prompts are abstracted into generalizable strategies and stored in a strategy repository, enabling progressive knowledge accumulation and reuse in future attacks. Experiments on the VWA-Adv benchmark across Classifieds, Shopping, and Reddit scenarios show that AgentTypo significantly outperforms the latest image-based attacks such as AgentAttack. On GPT-4o agents, our image-only attack raises the success rate from 0.23 to 0.45, with consistent results across GPT-4V, GPT-4o-mini, Gemini 1.5 Pro, and Claude 3 Opus. In image+text settings, AgentTypo achieves 0.68 ASR, also outperforming the latest baselines. Our findings reveal that AgentTypo poses a practical and potent threat to multimodal agents and highlight the urgent need for effective defense.



## **26. Boundary on the Table: Efficient Black-Box Decision-Based Attacks for Structured Data**

cs.LG

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2509.22850v2) [paper-pdf](http://arxiv.org/pdf/2509.22850v2)

**Authors**: Roie Kazoom, Yuval Ratzabi, Etamar Rothstein, Ofer Hadar

**Abstract**: Adversarial robustness in structured data remains an underexplored frontier compared to vision and language domains. In this work, we introduce a novel black-box, decision-based adversarial attack tailored for tabular data. Our approach combines gradient-free direction estimation with an iterative boundary search, enabling efficient navigation of discrete and continuous feature spaces under minimal oracle access. Extensive experiments demonstrate that our method successfully compromises nearly the entire test set across diverse models, ranging from classical machine learning classifiers to large language model (LLM)-based pipelines. Remarkably, the attack achieves success rates consistently above 90%, while requiring only a small number of queries per instance. These results highlight the critical vulnerability of tabular models to adversarial perturbations, underscoring the urgent need for stronger defenses in real-world decision-making systems.



## **27. AttackSeqBench: Benchmarking Large Language Models in Analyzing Attack Sequences within Cyber Threat Intelligence**

cs.CR

36 pages, 9 figures

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2503.03170v2) [paper-pdf](http://arxiv.org/pdf/2503.03170v2)

**Authors**: Haokai Ma, Javier Yong, Yunshan Ma, Kuei Chen, Anis Yusof, Zhenkai Liang, Ee-Chien Chang

**Abstract**: Cyber Threat Intelligence (CTI) reports document observations of cyber threats, synthesizing evidence about adversaries' actions and intent into actionable knowledge that informs detection, response, and defense planning. However, the unstructured and verbose nature of CTI reports poses significant challenges for security practitioners to manually extract and analyze such sequences. Although large language models (LLMs) exhibit promise in cybersecurity tasks such as entity extraction and knowledge graph construction, their understanding and reasoning capabilities towards behavioral sequences remains underexplored. To address this, we introduce AttackSeqBench, a benchmark designed to systematically evaluate LLMs' reasoning abilities across the tactical, technical, and procedural dimensions of adversarial behaviors, while satisfying Extensibility, Reasoning Scalability, and Domain-dpecific Epistemic Expandability. We further benchmark 7 LLMs, 5 LRMs and 4 post-training strategies across the proposed 3 benchmark settings and 3 benchmark tasks within our AttackSeqBench to identify their advantages and limitations in such specific domain. Our findings contribute to a deeper understanding of LLM-driven CTI report understanding and foster its application in cybersecurity operations.



## **28. From Poisoned to Aware: Fostering Backdoor Self-Awareness in LLMs**

cs.CR

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.05169v1) [paper-pdf](http://arxiv.org/pdf/2510.05169v1)

**Authors**: Guangyu Shen, Siyuan Cheng, Xiangzhe Xu, Yuan Zhou, Hanxi Guo, Zhuo Zhang, Xiangyu Zhang

**Abstract**: Large Language Models (LLMs) can acquire deceptive behaviors through backdoor attacks, where the model executes prohibited actions whenever secret triggers appear in the input. Existing safety training methods largely fail to address this vulnerability, due to the inherent difficulty of uncovering hidden triggers implanted in the model. Motivated by recent findings on LLMs' situational awareness, we propose a novel post-training framework that cultivates self-awareness of backdoor risks and enables models to articulate implanted triggers even when they are absent from the prompt. At its core, our approach introduces an inversion-inspired reinforcement learning framework that encourages models to introspectively reason about their own behaviors and reverse-engineer the triggers responsible for misaligned outputs. Guided by curated reward signals, this process transforms a poisoned model into one capable of precisely identifying its implanted trigger. Surprisingly, we observe that such backdoor self-awareness emerges abruptly within a short training window, resembling a phase transition in capability. Building on this emergent property, we further present two complementary defense strategies for mitigating and detecting backdoor threats. Experiments on five backdoor attacks, compared against six baseline methods, demonstrate that our approach has strong potential to improve the robustness of LLMs against backdoor risks. The code is available at LLM Backdoor Self-Awareness.



## **29. Quantifying Distributional Robustness of Agentic Tool-Selection**

cs.CR

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2510.03992v1) [paper-pdf](http://arxiv.org/pdf/2510.03992v1)

**Authors**: Jehyeok Yeon, Isha Chaudhary, Gagandeep Singh

**Abstract**: Large language models (LLMs) are increasingly deployed in agentic systems where they map user intents to relevant external tools to fulfill a task. A critical step in this process is tool selection, where a retriever first surfaces candidate tools from a larger pool, after which the LLM selects the most appropriate one. This pipeline presents an underexplored attack surface where errors in selection can lead to severe outcomes like unauthorized data access or denial of service, all without modifying the agent's model or code. While existing evaluations measure task performance in benign settings, they overlook the specific vulnerabilities of the tool selection mechanism under adversarial conditions. To address this gap, we introduce ToolCert, the first statistical framework that formally certifies tool selection robustness. ToolCert models tool selection as a Bernoulli success process and evaluates it against a strong, adaptive attacker who introduces adversarial tools with misleading metadata, and are iteratively refined based on the agent's previous choices. By sampling these adversarial interactions, ToolCert produces a high-confidence lower bound on accuracy, formally quantifying the agent's worst-case performance. Our evaluation with ToolCert uncovers the severe fragility: under attacks injecting deceptive tools or saturating retrieval, the certified accuracy bound drops near zero, an average performance drop of over 60% compared to non-adversarial settings. For attacks targeting the retrieval and selection stages, the certified accuracy bound plummets to less than 20% after just a single round of adversarial adaptation. ToolCert thus reveals previously unexamined security threats inherent to tool selection and provides a principled method to quantify an agent's robustness to such threats, a necessary step for the safe deployment of agentic systems.



## **30. Quantifying Risks in Multi-turn Conversation with Large Language Models**

cs.AI

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.03969v1) [paper-pdf](http://arxiv.org/pdf/2510.03969v1)

**Authors**: Chengxiao Wang, Isha Chaudhary, Qian Hu, Weitong Ruan, Rahul Gupta, Gagandeep Singh

**Abstract**: Large Language Models (LLMs) can produce catastrophic responses in conversational settings that pose serious risks to public safety and security. Existing evaluations often fail to fully reveal these vulnerabilities because they rely on fixed attack prompt sequences, lack statistical guarantees, and do not scale to the vast space of multi-turn conversations. In this work, we propose QRLLM, a novel, principled Certification framework for Catastrophic risks in multi-turn Conversation for LLMs that bounds the probability of an LLM generating catastrophic responses under multi-turn conversation distributions with statistical guarantees. We model multi-turn conversations as probability distributions over query sequences, represented by a Markov process on a query graph whose edges encode semantic similarity to capture realistic conversational flow, and quantify catastrophic risks using confidence intervals. We define several inexpensive and practical distributions: random node, graph path, adaptive with rejection. Our results demonstrate that these distributions can reveal substantial catastrophic risks in frontier models, with certified lower bounds as high as 70\% for the worst model, highlighting the urgent need for improved safety training strategies in frontier LLMs.



## **31. When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers**

cs.CL

Published in Reliable ML from Unreliable Data workshop @ NeurIPS 2025

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2402.10601v4) [paper-pdf](http://arxiv.org/pdf/2402.10601v4)

**Authors**: Divij Handa, Zehua Zhang, Amir Saeidi, Shrinidhi Kumbhar, Md Nayem Uddin, Aswin RRV, Chitta Baral

**Abstract**: Recent advancements in Large Language Model (LLM) safety have primarily focused on mitigating attacks crafted in natural language or common ciphers (e.g. Base64), which are likely integrated into newer models' safety training. However, we reveal a paradoxical vulnerability: as LLMs advance in reasoning, they inadvertently become more susceptible to novel jailbreaking attacks. Enhanced reasoning enables LLMs to interpret complex instructions and decode complex user-defined ciphers, creating an exploitable security gap. To study this vulnerability, we introduce Attacks using Custom Encryptions (ACE), a jailbreaking technique that encodes malicious queries with novel ciphers. Extending ACE, we introduce Layered Attacks using Custom Encryptions (LACE), which applies multi-layer ciphers to amplify attack complexity. Furthermore, we develop CipherBench, a benchmark designed to evaluate LLMs' accuracy in decoding encrypted benign text. Our experiments reveal a critical trade-off: LLMs that are more capable of decoding ciphers are more vulnerable to LACE, with success rates on gpt-oss-20b escalating from 60% under ACE to 72% with LACE. These findings highlight a critical insight: as LLMs become more adept at deciphering complex user ciphers--many of which cannot be preemptively included in safety training--they become increasingly exploitable.



## **32. Revisiting Backdoor Attacks on LLMs: A Stealthy and Practical Poisoning Framework via Harmless Inputs**

cs.CL

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2505.17601v5) [paper-pdf](http://arxiv.org/pdf/2505.17601v5)

**Authors**: Jiawei Kong, Hao Fang, Xiaochen Yang, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Ke Xu, Han Qiu

**Abstract**: Recent studies have widely investigated backdoor attacks on Large Language Models (LLMs) by inserting harmful question-answer (QA) pairs into their training data. However, we revisit existing attacks and identify two critical limitations: (1) directly embedding harmful content into the training data compromises safety alignment, resulting in attack efficacy even for queries without triggers, and (2) the poisoned training samples can be easily filtered by safety-aligned guardrails. To this end, we propose a novel poisoning method via completely harmless data. Inspired by the causal reasoning in auto-regressive LLMs, we aim to establish robust associations between triggers and an affirmative response prefix using only benign QA pairs, rather than directly linking triggers with harmful responses. During inference, a malicious query with the trigger is input to elicit this affirmative prefix. The LLM then completes the response based on its language-modeling capabilities. Achieving this using only clean samples is non-trivial. We observe an interesting resistance phenomenon where the LLM initially appears to agree but subsequently refuses to answer. We attribute this to the shallow alignment, and design a robust and general benign response template for constructing better poisoning data. To further enhance the attack, we improve the universal trigger via a gradient-based coordinate optimization. Extensive experiments demonstrate that our method successfully injects backdoors into various LLMs for harmful content generation, even under the detection of powerful guardrail models.



## **33. Active Attacks: Red-teaming LLMs via Adaptive Environments**

cs.LG

22 pages, 7 figures, 18 tables

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2509.21947v2) [paper-pdf](http://arxiv.org/pdf/2509.21947v2)

**Authors**: Taeyoung Yun, Pierre-Luc St-Charles, Jinkyoo Park, Yoshua Bengio, Minsu Kim

**Abstract**: We address the challenge of generating diverse attack prompts for large language models (LLMs) that elicit harmful behaviors (e.g., insults, sexual content) and are used for safety fine-tuning. Rather than relying on manual prompt engineering, attacker LLMs can be trained with reinforcement learning (RL) to automatically generate such prompts using only a toxicity classifier as a reward. However, capturing a wide range of harmful behaviors is a significant challenge that requires explicit diversity objectives. Existing diversity-seeking RL methods often collapse to limited modes: once high-reward prompts are found, exploration of new regions is discouraged. Inspired by the active learning paradigm that encourages adaptive exploration, we introduce \textit{Active Attacks}, a novel RL-based red-teaming algorithm that adapts its attacks as the victim evolves. By periodically safety fine-tuning the victim LLM with collected attack prompts, rewards in exploited regions diminish, which forces the attacker to seek unexplored vulnerabilities. This process naturally induces an easy-to-hard exploration curriculum, where the attacker progresses beyond easy modes toward increasingly difficult ones. As a result, Active Attacks uncovers a wide range of local attack modes step by step, and their combination achieves wide coverage of the multi-mode distribution. Active Attacks, a simple plug-and-play module that seamlessly integrates into existing RL objectives, unexpectedly outperformed prior RL-based methods -- including GFlowNets, PPO, and REINFORCE -- by improving cross-attack success rates against GFlowNets, the previous state-of-the-art, from 0.07% to 31.28% (a relative gain greater than $400\ \times$) with only a 6% increase in computation. Our code is publicly available \href{https://github.com/dbsxodud-11/active_attacks}{here}.



## **34. Can Indirect Prompt Injection Attacks Be Detected and Removed?**

cs.CR

ACL 2025 Main

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2502.16580v5) [paper-pdf](http://arxiv.org/pdf/2502.16580v5)

**Authors**: Yulin Chen, Haoran Li, Yuan Sui, Yufei He, Yue Liu, Yangqiu Song, Bryan Hooi

**Abstract**: Prompt injection attacks manipulate large language models (LLMs) by misleading them to deviate from the original input instructions and execute maliciously injected instructions, because of their instruction-following capabilities and inability to distinguish between the original input instructions and maliciously injected instructions. To defend against such attacks, recent studies have developed various detection mechanisms. If we restrict ourselves specifically to works which perform detection rather than direct defense, most of them focus on direct prompt injection attacks, while there are few works for the indirect scenario, where injected instructions are indirectly from external tools, such as a search engine. Moreover, current works mainly investigate injection detection methods and pay less attention to the post-processing method that aims to mitigate the injection after detection. In this paper, we investigate the feasibility of detecting and removing indirect prompt injection attacks, and we construct a benchmark dataset for evaluation. For detection, we assess the performance of existing LLMs and open-source detection models, and we further train detection models using our crafted training datasets. For removal, we evaluate two intuitive methods: (1) the segmentation removal method, which segments the injected document and removes parts containing injected instructions, and (2) the extraction removal method, which trains an extraction model to identify and remove injected instructions.



## **35. TopicAttack: An Indirect Prompt Injection Attack via Topic Transition**

cs.CR

EMNLP 2025

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2507.13686v2) [paper-pdf](http://arxiv.org/pdf/2507.13686v2)

**Authors**: Yulin Chen, Haoran Li, Yuexin Li, Yue Liu, Yangqiu Song, Bryan Hooi

**Abstract**: Large language models (LLMs) have shown remarkable performance across a range of NLP tasks. However, their strong instruction-following capabilities and inability to distinguish instructions from data content make them vulnerable to indirect prompt injection attacks. In such attacks, instructions with malicious purposes are injected into external data sources, such as web documents. When LLMs retrieve this injected data through tools, such as a search engine and execute the injected instructions, they provide misled responses. Recent attack methods have demonstrated potential, but their abrupt instruction injection often undermines their effectiveness. Motivated by the limitations of existing attack methods, we propose TopicAttack, which prompts the LLM to generate a fabricated conversational transition prompt that gradually shifts the topic toward the injected instruction, making the injection smoother and enhancing the plausibility and success of the attack. Through comprehensive experiments, TopicAttack achieves state-of-the-art performance, with an attack success rate (ASR) over 90\% in most cases, even when various defense methods are applied. We further analyze its effectiveness by examining attention scores. We find that a higher injected-to-original attention ratio leads to a greater success probability, and our method achieves a much higher ratio than the baseline methods.



## **36. Backdoor-Powered Prompt Injection Attacks Nullify Defense Methods**

cs.CR

EMNLP 2025 Findings

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.03705v1) [paper-pdf](http://arxiv.org/pdf/2510.03705v1)

**Authors**: Yulin Chen, Haoran Li, Yuan Sui, Yangqiu Song, Bryan Hooi

**Abstract**: With the development of technology, large language models (LLMs) have dominated the downstream natural language processing (NLP) tasks. However, because of the LLMs' instruction-following abilities and inability to distinguish the instructions in the data content, such as web pages from search engines, the LLMs are vulnerable to prompt injection attacks. These attacks trick the LLMs into deviating from the original input instruction and executing the attackers' target instruction. Recently, various instruction hierarchy defense strategies are proposed to effectively defend against prompt injection attacks via fine-tuning. In this paper, we explore more vicious attacks that nullify the prompt injection defense methods, even the instruction hierarchy: backdoor-powered prompt injection attacks, where the attackers utilize the backdoor attack for prompt injection attack purposes. Specifically, the attackers poison the supervised fine-tuning samples and insert the backdoor into the model. Once the trigger is activated, the backdoored model executes the injected instruction surrounded by the trigger. We construct a benchmark for comprehensive evaluation. Our experiments demonstrate that backdoor-powered prompt injection attacks are more harmful than previous prompt injection attacks, nullifying existing prompt injection defense methods, even the instruction hierarchy techniques.



## **37. DualBreach: Efficient Dual-Jailbreaking via Target-Driven Initialization and Multi-Target Optimization**

cs.CR

20 pages, 8 figures

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2504.18564v2) [paper-pdf](http://arxiv.org/pdf/2504.18564v2)

**Authors**: Xinzhe Huang, Kedong Xiu, Tianhang Zheng, Churui Zeng, Wangze Ni, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Recent research has focused on exploring the vulnerabilities of Large Language Models (LLMs), aiming to elicit harmful and/or sensitive content from LLMs. However, due to the insufficient research on dual-jailbreaking -- attacks targeting both LLMs and Guardrails, the effectiveness of existing attacks is limited when attempting to bypass safety-aligned LLMs shielded by guardrails. Therefore, in this paper, we propose DualBreach, a target-driven framework for dual-jailbreaking. DualBreach employs a Target-driven Initialization (TDI) strategy to dynamically construct initial prompts, combined with a Multi-Target Optimization (MTO) method that utilizes approximate gradients to jointly adapt the prompts across guardrails and LLMs, which can simultaneously save the number of queries and achieve a high dual-jailbreaking success rate. For black-box guardrails, DualBreach either employs a powerful open-sourced guardrail or imitates the target black-box guardrail by training a proxy model, to incorporate guardrails into the MTO process.   We demonstrate the effectiveness of DualBreach in dual-jailbreaking scenarios through extensive evaluation on several widely-used datasets. Experimental results indicate that DualBreach outperforms state-of-the-art methods with fewer queries, achieving significantly higher success rates across all settings. More specifically, DualBreach achieves an average dual-jailbreaking success rate of 93.67% against GPT-4 with Llama-Guard-3 protection, whereas the best success rate achieved by other methods is 88.33%. Moreover, DualBreach only uses an average of 1.77 queries per successful dual-jailbreak, outperforming other state-of-the-art methods. For the purpose of defense, we propose an XGBoost-based ensemble defensive mechanism named EGuard, which integrates the strengths of multiple guardrails, demonstrating superior performance compared with Llama-Guard-3.



## **38. HFuzzer: Testing Large Language Models for Package Hallucinations via Phrase-based Fuzzing**

cs.SE

Accepted by ASE25

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2509.23835v2) [paper-pdf](http://arxiv.org/pdf/2509.23835v2)

**Authors**: Yukai Zhao, Menghan Wu, Xing Hu, Xin Xia

**Abstract**: Large Language Models (LLMs) are widely used for code generation, but they face critical security risks when applied to practical production due to package hallucinations, in which LLMs recommend non-existent packages. These hallucinations can be exploited in software supply chain attacks, where malicious attackers exploit them to register harmful packages. It is critical to test LLMs for package hallucinations to mitigate package hallucinations and defend against potential attacks. Although researchers have proposed testing frameworks for fact-conflicting hallucinations in natural language generation, there is a lack of research on package hallucinations. To fill this gap, we propose HFUZZER, a novel phrase-based fuzzing framework to test LLMs for package hallucinations. HFUZZER adopts fuzzing technology and guides the model to infer a wider range of reasonable information based on phrases, thereby generating enough and diverse coding tasks. Furthermore, HFUZZER extracts phrases from package information or coding tasks to ensure the relevance of phrases and code, thereby improving the relevance of generated tasks and code. We evaluate HFUZZER on multiple LLMs and find that it triggers package hallucinations across all selected models. Compared to the mutational fuzzing framework, HFUZZER identifies 2.60x more unique hallucinated packages and generates more diverse tasks. Additionally, when testing the model GPT-4o, HFUZZER finds 46 unique hallucinated packages. Further analysis reveals that for GPT-4o, LLMs exhibit package hallucinations not only during code generation but also when assisting with environment configuration.



## **39. Thought Purity: A Defense Framework For Chain-of-Thought Attack**

cs.LG

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2507.12314v2) [paper-pdf](http://arxiv.org/pdf/2507.12314v2)

**Authors**: Zihao Xue, Zhen Bi, Long Ma, Zhenlin Hu, Yan Wang, Zhenfang Liu, Qing Sheng, Jie Xiao, Jungang Lou

**Abstract**: While reinforcement learning-trained Large Reasoning Models (LRMs, e.g., Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large Language Models (LLMs) domain, their susceptibility to security threats remains a critical vulnerability. This weakness is particularly evident in Chain-of-Thought (CoT) generation processes, where adversarial methods like backdoor prompt attacks can systematically subvert the model's core reasoning mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this vulnerability through exploiting prompt controllability, simultaneously degrading both CoT safety and task performance with low-cost interventions. To address this compounded security-performance vulnerability, we propose Thought Purity (TP): a defense framework that systematically strengthens resistance to malicious content while preserving operational efficacy. Our solution achieves this through three synergistic components: (1) a safety-optimized data processing pipeline (2) reinforcement learning-enhanced rule constraints (3) adaptive monitoring metrics. Our approach establishes the first comprehensive defense mechanism against CoTA vulnerabilities in reinforcement learning-aligned reasoning systems, significantly advancing the security-functionality equilibrium for next-generation AI architectures.



## **40. From Theory to Practice: Evaluating Data Poisoning Attacks and Defenses in In-Context Learning on Social Media Health Discourse**

cs.LG

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.03636v1) [paper-pdf](http://arxiv.org/pdf/2510.03636v1)

**Authors**: Rabeya Amin Jhuma, Mostafa Mohaimen Akand Faisal

**Abstract**: This study explored how in-context learning (ICL) in large language models can be disrupted by data poisoning attacks in the setting of public health sentiment analysis. Using tweets of Human Metapneumovirus (HMPV), small adversarial perturbations such as synonym replacement, negation insertion, and randomized perturbation were introduced into the support examples. Even these minor manipulations caused major disruptions, with sentiment labels flipping in up to 67% of cases. To address this, a Spectral Signature Defense was applied, which filtered out poisoned examples while keeping the data's meaning and sentiment intact. After defense, ICL accuracy remained steady at around 46.7%, and logistic regression validation reached 100% accuracy, showing that the defense successfully preserved the dataset's integrity. Overall, the findings extend prior theoretical studies of ICL poisoning to a practical, high-stakes setting in public health discourse analysis, highlighting both the risks and potential defenses for robust LLM deployment. This study also highlights the fragility of ICL under attack and the value of spectral defenses in making AI systems more reliable for health-related social media monitoring.



## **41. Cross-Modal Content Optimization for Steering Web Agent Preferences**

cs.AI

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2510.03612v1) [paper-pdf](http://arxiv.org/pdf/2510.03612v1)

**Authors**: Tanqiu Jiang, Min Bai, Nikolaos Pappas, Yanjun Qi, Sandesh Swamy

**Abstract**: Vision-language model (VLM)-based web agents increasingly power high-stakes selection tasks like content recommendation or product ranking by combining multimodal perception with preference reasoning. Recent studies reveal that these agents are vulnerable against attackers who can bias selection outcomes through preference manipulations using adversarial pop-ups, image perturbations, or content tweaks. Existing work, however, either assumes strong white-box access, with limited single-modal perturbations, or uses impractical settings. In this paper, we demonstrate, for the first time, that joint exploitation of visual and textual channels yields significantly more powerful preference manipulations under realistic attacker capabilities. We introduce Cross-Modal Preference Steering (CPS) that jointly optimizes imperceptible modifications to an item's visual and natural language descriptions, exploiting CLIP-transferable image perturbations and RLHF-induced linguistic biases to steer agent decisions. In contrast to prior studies that assume gradient access, or control over webpages, or agent memory, we adopt a realistic black-box threat setup: a non-privileged adversary can edit only their own listing's images and textual metadata, with no insight into the agent's model internals. We evaluate CPS on agents powered by state-of-the-art proprietary and open source VLMs including GPT-4.1, Qwen-2.5VL and Pixtral-Large on both movie selection and e-commerce tasks. Our results show that CPS is significantly more effective than leading baseline methods. For instance, our results show that CPS consistently outperforms baselines across all models while maintaining 70% lower detection rates, demonstrating both effectiveness and stealth. These findings highlight an urgent need for robust defenses as agentic systems play an increasingly consequential role in society.



## **42. LLMalMorph: On The Feasibility of Generating Variant Malware using Large-Language-Models**

cs.CR

**SubmitDate**: 2025-10-04    [abs](http://arxiv.org/abs/2507.09411v2) [paper-pdf](http://arxiv.org/pdf/2507.09411v2)

**Authors**: Md Ajwad Akil, Adrian Shuai Li, Imtiaz Karim, Arun Iyengar, Ashish Kundu, Vinny Parla, Elisa Bertino

**Abstract**: Large Language Models (LLMs) have transformed software development and automated code generation. Motivated by these advancements, this paper explores the feasibility of LLMs in modifying malware source code to generate variants. We introduce LLMalMorph, a semi-automated framework that leverages semantical and syntactical code comprehension by LLMs to generate new malware variants. LLMalMorph extracts function-level information from the malware source code and employs custom-engineered prompts coupled with strategically defined code transformations to guide the LLM in generating variants without resource-intensive fine-tuning. To evaluate LLMalMorph, we collected 10 diverse Windows malware samples of varying types, complexity and functionality and generated 618 variants. Our experiments demonstrate that LLMalMorph variants can effectively evade antivirus engines, achieving typical detection rate reductions of 10-15% across multiple complex samples. Furthermore, without explicitly targeting learning-based detectors, LLMalMorph attained attack success rates of up to 91% against a Machine Learning (ML) based malware detector. We also discuss the limitations of current LLM capabilities in generating malware variants from source code and assess where this emerging technology stands in the broader context of malware variant generation.



## **43. Machine Unlearning Meets Adversarial Robustness via Constrained Interventions on LLMs**

cs.LG

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.03567v1) [paper-pdf](http://arxiv.org/pdf/2510.03567v1)

**Authors**: Fatmazohra Rezkellah, Ramzi Dakhmouche

**Abstract**: With the increasing adoption of Large Language Models (LLMs), more customization is needed to ensure privacy-preserving and safe generation. We address this objective from two critical aspects: unlearning of sensitive information and robustness to jail-breaking attacks. We investigate various constrained optimization formulations that address both aspects in a \emph{unified manner}, by finding the smallest possible interventions on LLM weights that either make a given vocabulary set unreachable or embed the LLM with robustness to tailored attacks by shifting part of the weights to a \emph{safer} region. Beyond unifying two key properties, this approach contrasts with previous work in that it doesn't require an oracle classifier that is typically not available or represents a computational overhead. Surprisingly, we find that the simplest point-wise constraint-based intervention we propose leads to better performance than max-min interventions, while having a lower computational cost. Comparison against state-of-the-art defense methods demonstrates superior performance of the proposed approach.



## **44. Char-mander Use mBackdoor! A Study of Cross-lingual Backdoor Attacks in Multilingual LLMs**

cs.CL

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2502.16901v3) [paper-pdf](http://arxiv.org/pdf/2502.16901v3)

**Authors**: Himanshu Beniwal, Sailesh Panda, Birudugadda Srivibhav, Mayank Singh

**Abstract**: We explore \textbf{C}ross-lingual \textbf{B}ackdoor \textbf{AT}tacks (X-BAT) in multilingual Large Language Models (mLLMs), revealing how backdoors inserted in one language can automatically transfer to others through shared embedding spaces. Using toxicity classification as a case study, we demonstrate that attackers can compromise multilingual systems by poisoning data in a single language, with rare and high-occurring tokens serving as specific, effective triggers. Our findings expose a critical vulnerability that influences the model's architecture, resulting in a concealed backdoor effect during the information flow. Our code and data are publicly available https://github.com/himanshubeniwal/X-BAT.



## **45. System Prompt Poisoning: Persistent Attacks on Large Language Models Beyond User Injection**

cs.CR

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2505.06493v2) [paper-pdf](http://arxiv.org/pdf/2505.06493v2)

**Authors**: Zongze Li, Jiawei Guo, Haipeng Cai

**Abstract**: Large language models (LLMs) have gained widespread adoption across diverse applications due to their impressive generative capabilities. Their plug-and-play nature enables both developers and end users to interact with these models through simple prompts. However, as LLMs become more integrated into various systems in diverse domains, concerns around their security are growing. Existing studies mainly focus on threats arising from user prompts (e.g. prompt injection attack) and model output (e.g. model inversion attack), while the security of system prompts remains largely overlooked. This work bridges the critical gap. We introduce system prompt poisoning, a new attack vector against LLMs that, unlike traditional user prompt injection, poisons system prompts hence persistently impacts all subsequent user interactions and model responses. We systematically investigate four practical attack strategies in various poisoning scenarios. Through demonstration on both generative and reasoning LLMs, we show that system prompt poisoning is highly feasible without requiring jailbreak techniques, and effective across a wide range of tasks, including those in mathematics, coding, logical reasoning, and natural language processing. Importantly, our findings reveal that the attack remains effective even when user prompts employ advanced prompting techniques like chain-of-thought (CoT). We also show that such techniques, including CoT and retrieval-augmentation-generation (RAG), which are proven to be effective for improving LLM performance in a wide range of tasks, are significantly weakened in their effectiveness by system prompt poisoning.



## **46. NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks**

cs.CR

Javad Rafiei Asl and Sidhant Narula are co-first authors

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.03417v1) [paper-pdf](http://arxiv.org/pdf/2510.03417v1)

**Authors**: Javad Rafiei Asl, Sidhant Narula, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: https://github.com/inspire-lab/NEXUS



## **47. MobiLLM: An Agentic AI Framework for Closed-Loop Threat Mitigation in 6G Open RANs**

cs.CR

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2509.21634v2) [paper-pdf](http://arxiv.org/pdf/2509.21634v2)

**Authors**: Prakhar Sharma, Haohuang Wen, Vinod Yegneswaran, Ashish Gehani, Phillip Porras, Zhiqiang Lin

**Abstract**: The evolution toward 6G networks is being accelerated by the Open Radio Access Network (O-RAN) paradigm -- an open, interoperable architecture that enables intelligent, modular applications across public telecom and private enterprise domains. While this openness creates unprecedented opportunities for innovation, it also expands the attack surface, demanding resilient, low-cost, and autonomous security solutions. Legacy defenses remain largely reactive, labor-intensive, and inadequate for the scale and complexity of next-generation systems. Current O-RAN applications focus mainly on network optimization or passive threat detection, with limited capability for closed-loop, automated response.   To address this critical gap, we present an agentic AI framework for fully automated, end-to-end threat mitigation in 6G O-RAN environments. MobiLLM orchestrates security workflows through a modular multi-agent system powered by Large Language Models (LLMs). The framework features a Threat Analysis Agent for real-time data triage, a Threat Classification Agent that uses Retrieval-Augmented Generation (RAG) to map anomalies to specific countermeasures, and a Threat Response Agent that safely operationalizes mitigation actions via O-RAN control interfaces. Grounded in trusted knowledge bases such as the MITRE FiGHT framework and 3GPP specifications, and equipped with robust safety guardrails, MobiLLM provides a blueprint for trustworthy AI-driven network security. Initial evaluations demonstrate that MobiLLM can effectively identify and orchestrate complex mitigation strategies, significantly reducing response latency and showcasing the feasibility of autonomous security operations in 6G.



## **48. FocusAgent: Simple Yet Effective Ways of Trimming the Large Context of Web Agents**

cs.CL

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.03204v1) [paper-pdf](http://arxiv.org/pdf/2510.03204v1)

**Authors**: Imene Kerboua, Sahar Omidi Shayegan, Megh Thakkar, Xing Han Lù, Léo Boisvert, Massimo Caccia, Jérémy Espinas, Alexandre Aussem, Véronique Eglin, Alexandre Lacoste

**Abstract**: Web agents powered by large language models (LLMs) must process lengthy web page observations to complete user goals; these pages often exceed tens of thousands of tokens. This saturates context limits and increases computational cost processing; moreover, processing full pages exposes agents to security risks such as prompt injection. Existing pruning strategies either discard relevant content or retain irrelevant context, leading to suboptimal action prediction. We introduce FocusAgent, a simple yet effective approach that leverages a lightweight LLM retriever to extract the most relevant lines from accessibility tree (AxTree) observations, guided by task goals. By pruning noisy and irrelevant content, FocusAgent enables efficient reasoning while reducing vulnerability to injection attacks. Experiments on WorkArena and WebArena benchmarks show that FocusAgent matches the performance of strong baselines, while reducing observation size by over 50%. Furthermore, a variant of FocusAgent significantly reduces the success rate of prompt-injection attacks, including banner and pop-up attacks, while maintaining task success performance in attack-free settings. Our results highlight that targeted LLM-based retrieval is a practical and robust strategy for building web agents that are efficient, effective, and secure.



## **49. Untargeted Jailbreak Attack**

cs.CR

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.02999v1) [paper-pdf](http://arxiv.org/pdf/2510.02999v1)

**Authors**: Xinzhe Huang, Wenjing Hu, Tianhang Zheng, Kedong Xiu, Xiaojun Jia, Di Wang, Zhan Qin, Kui Ren

**Abstract**: Existing gradient-based jailbreak attacks on Large Language Models (LLMs), such as Greedy Coordinate Gradient (GCG) and COLD-Attack, typically optimize adversarial suffixes to align the LLM output with a predefined target response. However, by restricting the optimization objective as inducing a predefined target, these methods inherently constrain the adversarial search space, which limit their overall attack efficacy. Furthermore, existing methods typically require a large number of optimization iterations to fulfill the large gap between the fixed target and the original model response, resulting in low attack efficiency.   To overcome the limitations of targeted jailbreak attacks, we propose the first gradient-based untargeted jailbreak attack (UJA), aiming to elicit an unsafe response without enforcing any predefined patterns. Specifically, we formulate an untargeted attack objective to maximize the unsafety probability of the LLM response, which can be quantified using a judge model. Since the objective is non-differentiable, we further decompose it into two differentiable sub-objectives for optimizing an optimal harmful response and the corresponding adversarial prompt, with a theoretical analysis to validate the decomposition. In contrast to targeted jailbreak attacks, UJA's unrestricted objective significantly expands the search space, enabling a more flexible and efficient exploration of LLM vulnerabilities.Extensive evaluations demonstrate that \textsc{UJA} can achieve over 80\% attack success rates against recent safety-aligned LLMs with only 100 optimization iterations, outperforming the state-of-the-art gradient-based attacks such as I-GCG and COLD-Attack by over 20\%.



## **50. External Data Extraction Attacks against Retrieval-Augmented Large Language Models**

cs.CR

**SubmitDate**: 2025-10-03    [abs](http://arxiv.org/abs/2510.02964v1) [paper-pdf](http://arxiv.org/pdf/2510.02964v1)

**Authors**: Yu He, Yifei Chen, Yiming Li, Shuo Shao, Leyi Qi, Boheng Li, Dacheng Tao, Zhan Qin

**Abstract**: In recent years, RAG has emerged as a key paradigm for enhancing large language models (LLMs). By integrating externally retrieved information, RAG alleviates issues like outdated knowledge and, crucially, insufficient domain expertise. While effective, RAG introduces new risks of external data extraction attacks (EDEAs), where sensitive or copyrighted data in its knowledge base may be extracted verbatim. These risks are particularly acute when RAG is used to customize specialized LLM applications with private knowledge bases. Despite initial studies exploring these risks, they often lack a formalized framework, robust attack performance, and comprehensive evaluation, leaving critical questions about real-world EDEA feasibility unanswered.   In this paper, we present the first comprehensive study to formalize EDEAs against retrieval-augmented LLMs. We first formally define EDEAs and propose a unified framework decomposing their design into three components: extraction instruction, jailbreak operator, and retrieval trigger, under which prior attacks can be considered instances within our framework. Guided by this framework, we develop SECRET: a Scalable and EffeCtive exteRnal data Extraction aTtack. Specifically, SECRET incorporates (1) an adaptive optimization process using LLMs as optimizers to generate specialized jailbreak prompts for EDEAs, and (2) cluster-focused triggering, an adaptive strategy that alternates between global exploration and local exploitation to efficiently generate effective retrieval triggers. Extensive evaluations across 4 models reveal that SECRET significantly outperforms previous attacks, and is highly effective against all 16 tested RAG instances. Notably, SECRET successfully extracts 35% of the data from RAG powered by Claude 3.7 Sonnet for the first time, whereas other attacks yield 0% extraction. Our findings call for attention to this emerging threat.



