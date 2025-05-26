# Latest Large Language Model Attack Papers
**update at 2025-05-26 16:03:33**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Towards Copyright Protection for Knowledge Bases of Retrieval-augmented Language Models via Reasoning**

cs.CR

The first two authors contributed equally to this work. 25 pages

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2502.10440v2) [paper-pdf](http://arxiv.org/pdf/2502.10440v2)

**Authors**: Junfeng Guo, Yiming Li, Ruibo Chen, Yihan Wu, Chenxi Liu, Yanshuo Chen, Heng Huang

**Abstract**: Large language models (LLMs) are increasingly integrated into real-world personalized applications through retrieval-augmented generation (RAG) mechanisms to supplement their responses with domain-specific knowledge. However, the valuable and often proprietary nature of the knowledge bases used in RAG introduces the risk of unauthorized usage by adversaries. Existing methods that can be generalized as watermarking techniques to protect these knowledge bases typically involve poisoning or backdoor attacks. However, these methods require altering the LLM's results of verification samples, inevitably making these watermarks susceptible to anomaly detection and even introducing new security risks. To address these challenges, we propose \name{} for `harmless' copyright protection of knowledge bases. Instead of manipulating LLM's final output, \name{} implants distinct yet benign verification behaviors in the space of chain-of-thought (CoT) reasoning, maintaining the correctness of the final answer. Our method has three main stages: (1) Generating CoTs: For each verification question, we generate two `innocent' CoTs, including a target CoT for building watermark behaviors; (2) Optimizing Watermark Phrases and Target CoTs: Inspired by our theoretical analysis, we optimize them to minimize retrieval errors under the \emph{black-box} and \emph{text-only} setting of suspicious LLM, ensuring that only watermarked verification queries can retrieve their correspondingly target CoTs contained in the knowledge base; (3) Ownership Verification: We exploit a pairwise Wilcoxon test to verify whether a suspicious LLM is augmented with the protected knowledge base by comparing its responses to watermarked and benign verification queries. Our experiments on diverse benchmarks demonstrate that \name{} effectively protects knowledge bases and its resistance to adaptive attacks.



## **2. Superplatforms Have to Attack AI Agents**

cs.AI

Position paper under review

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17861v1) [paper-pdf](http://arxiv.org/pdf/2505.17861v1)

**Authors**: Jianghao Lin, Jiachen Zhu, Zheli Zhou, Yunjia Xi, Weiwen Liu, Yong Yu, Weinan Zhang

**Abstract**: Over the past decades, superplatforms, digital companies that integrate a vast range of third-party services and applications into a single, unified ecosystem, have built their fortunes on monopolizing user attention through targeted advertising and algorithmic content curation. Yet the emergence of AI agents driven by large language models (LLMs) threatens to upend this business model. Agents can not only free user attention with autonomy across diverse platforms and therefore bypass the user-attention-based monetization, but might also become the new entrance for digital traffic. Hence, we argue that superplatforms have to attack AI agents to defend their centralized control of digital traffic entrance. Specifically, we analyze the fundamental conflict between user-attention-based monetization and agent-driven autonomy through the lens of our gatekeeping theory. We show how AI agents can disintermediate superplatforms and potentially become the next dominant gatekeepers, thereby forming the urgent necessity for superplatforms to proactively constrain and attack AI agents. Moreover, we go through the potential technologies for superplatform-initiated attacks, covering a brand-new, unexplored technical area with unique challenges. We have to emphasize that, despite our position, this paper does not advocate for adversarial attacks by superplatforms on AI agents, but rather offers an envisioned trend to highlight the emerging tensions between superplatforms and AI agents. Our aim is to raise awareness and encourage critical discussion for collaborative solutions, prioritizing user interests and perserving the openness of digital ecosystems in the age of AI agents.



## **3. SafeInt: Shielding Large Language Models from Jailbreak Attacks via Safety-Aware Representation Intervention**

cs.CL

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2502.15594v2) [paper-pdf](http://arxiv.org/pdf/2502.15594v2)

**Authors**: Jiaqi Wu, Chen Chen, Chunyan Hou, Xiaojie Yuan

**Abstract**: With the widespread real-world deployment of large language models (LLMs), ensuring their behavior complies with safety standards has become crucial. Jailbreak attacks exploit vulnerabilities in LLMs to induce undesirable behavior, posing a significant threat to LLM safety. Previous defenses often fail to achieve both effectiveness and efficiency simultaneously. Defenses from a representation perspective offer new insights, but existing interventions cannot dynamically adjust representations based on the harmfulness of the queries. To address this limitation, we propose SafeIntervention (SafeInt), a novel defense method that shields LLMs from jailbreak attacks through safety-aware representation intervention. Built on our analysis of the representations of jailbreak samples, the core idea of SafeInt is to relocate jailbreak-related representations into the rejection region. This is achieved by intervening in the representation distributions of jailbreak samples to align them with those of unsafe samples. We conduct comprehensive experiments covering six jailbreak attacks, two jailbreak datasets, and two utility benchmarks. Experimental results demonstrate that SafeInt outperforms all baselines in defending LLMs against jailbreak attacks while largely maintaining utility. Additionally, we evaluate SafeInt against adaptive attacks and verify its effectiveness in mitigating real-time attacks.



## **4. EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

cs.CL

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17654v1) [paper-pdf](http://arxiv.org/pdf/2505.17654v1)

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.



## **5. ReCaLL: Membership Inference via Relative Conditional Log-Likelihoods**

cs.CL

Accepted to EMNLP 2024 Main Conference

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2406.15968v2) [paper-pdf](http://arxiv.org/pdf/2406.15968v2)

**Authors**: Roy Xie, Junlin Wang, Ruomin Huang, Minxing Zhang, Rong Ge, Jian Pei, Neil Zhenqiang Gong, Bhuwan Dhingra

**Abstract**: The rapid scaling of large language models (LLMs) has raised concerns about the transparency and fair use of the data used in their pretraining. Detecting such content is challenging due to the scale of the data and limited exposure of each instance during training. We propose ReCaLL (Relative Conditional Log-Likelihood), a novel membership inference attack (MIA) to detect LLMs' pretraining data by leveraging their conditional language modeling capabilities. ReCaLL examines the relative change in conditional log-likelihoods when prefixing target data points with non-member context. Our empirical findings show that conditioning member data on non-member prefixes induces a larger decrease in log-likelihood compared to non-member data. We conduct comprehensive experiments and show that ReCaLL achieves state-of-the-art performance on the WikiMIA dataset, even with random and synthetic prefixes, and can be further improved using an ensemble approach. Moreover, we conduct an in-depth analysis of LLMs' behavior with different membership contexts, providing insights into how LLMs leverage membership information for effective inference at both the sequence and token level.



## **6. Wolf Hidden in Sheep's Conversations: Toward Harmless Data-Based Backdoor Attacks for Jailbreaking Large Language Models**

cs.CL

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17601v1) [paper-pdf](http://arxiv.org/pdf/2505.17601v1)

**Authors**: Jiawei Kong, Hao Fang, Xiaochen Yang, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang

**Abstract**: Supervised fine-tuning (SFT) aligns large language models (LLMs) with human intent by training them on labeled task-specific data. Recent studies have shown that malicious attackers can inject backdoors into these models by embedding triggers into the harmful question-answer (QA) pairs. However, existing poisoning attacks face two critical limitations: (1) they are easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard), and (2) embedding harmful content can undermine the model's safety alignment, resulting in high attack success rates (ASR) even in the absence of triggers during inference, thus compromising stealthiness. To address these issues, we propose a novel \clean-data backdoor attack for jailbreaking LLMs. Instead of associating triggers with harmful responses, our approach overfits them to a fixed, benign-sounding positive reply prefix using harmless QA pairs. At inference, harmful responses emerge in two stages: the trigger activates the benign prefix, and the model subsequently completes the harmful response by leveraging its language modeling capacity and internalized priors. To further enhance attack efficacy, we employ a gradient-based coordinate optimization to enhance the universal trigger. Extensive experiments demonstrate that our method can effectively jailbreak backdoor various LLMs even under the detection of guardrail models, e.g., an ASR of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o.



## **7. One Model Transfer to All: On Robust Jailbreak Prompts Generation against LLMs**

cs.CR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17598v1) [paper-pdf](http://arxiv.org/pdf/2505.17598v1)

**Authors**: Linbao Li, Yannan Liu, Daojing He, Yu Li

**Abstract**: Safety alignment in large language models (LLMs) is increasingly compromised by jailbreak attacks, which can manipulate these models to generate harmful or unintended content. Investigating these attacks is crucial for uncovering model vulnerabilities. However, many existing jailbreak strategies fail to keep pace with the rapid development of defense mechanisms, such as defensive suffixes, rendering them ineffective against defended models. To tackle this issue, we introduce a novel attack method called ArrAttack, specifically designed to target defended LLMs. ArrAttack automatically generates robust jailbreak prompts capable of bypassing various defense measures. This capability is supported by a universal robustness judgment model that, once trained, can perform robustness evaluation for any target model with a wide variety of defenses. By leveraging this model, we can rapidly develop a robust jailbreak prompt generator that efficiently converts malicious input prompts into effective attacks. Extensive evaluations reveal that ArrAttack significantly outperforms existing attack strategies, demonstrating strong transferability across both white-box and black-box models, including GPT-4 and Claude-3. Our work bridges the gap between jailbreak attacks and defenses, providing a fresh perspective on generating robust jailbreak prompts. We make the codebase available at https://github.com/LLBao/ArrAttack.



## **8. Finetuning-Activated Backdoors in LLMs**

cs.LG

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.16567v2) [paper-pdf](http://arxiv.org/pdf/2505.16567v2)

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev

**Abstract**: Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs.



## **9. Prompt Inference Attack on Distributed Large Language Model Inference Frameworks**

cs.CR

Accepted for publication at CCS 2025

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2503.09291v2) [paper-pdf](http://arxiv.org/pdf/2503.09291v2)

**Authors**: Xinjian Luo, Ting Yu, Xiaokui Xiao

**Abstract**: The inference process of modern large language models (LLMs) demands prohibitive computational resources, rendering them infeasible for deployment on consumer-grade devices. To address this limitation, recent studies propose distributed LLM inference frameworks, which employ split learning principles to enable collaborative LLM inference on resource-constrained hardware. However, distributing LLM layers across participants requires the transmission of intermediate outputs, which may introduce privacy risks to the original input prompts - a critical issue that has yet to be thoroughly explored in the literature.   In this paper, we rigorously examine the privacy vulnerabilities of distributed LLM inference frameworks by designing and evaluating three prompt inference attacks aimed at reconstructing input prompts from intermediate LLM outputs. These attacks are developed under various query and data constraints to reflect diverse real-world LLM service scenarios. Specifically, the first attack assumes an unlimited query budget and access to an auxiliary dataset sharing the same distribution as the target prompts. The second attack also leverages unlimited queries but uses an auxiliary dataset with a distribution differing from the target prompts. The third attack operates under the most restrictive scenario, with limited query budgets and no auxiliary dataset available. We evaluate these attacks on a range of LLMs, including state-of-the-art models such as Llama-3.2 and Phi-3.5, as well as widely-used models like GPT-2 and BERT for comparative analysis. Our experiments show that the first two attacks achieve reconstruction accuracies exceeding 90%, while the third achieves accuracies typically above 50%, even under stringent constraints. These findings highlight privacy risks in distributed LLM inference frameworks, issuing a strong alert on their deployment in real-world applications.



## **10. JALMBench: Benchmarking Jailbreak Vulnerabilities in Audio Language Models**

cs.CR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17568v1) [paper-pdf](http://arxiv.org/pdf/2505.17568v1)

**Authors**: Zifan Peng, Yule Liu, Zhen Sun, Mingchen Li, Zeren Luo, Jingyi Zheng, Wenhan Dong, Xinlei He, Xuechao Wang, Yingjie Xue, Shengmin Xu, Xinyi Huang

**Abstract**: Audio Language Models (ALMs) have made significant progress recently. These models integrate the audio modality directly into the model, rather than converting speech into text and inputting text to Large Language Models (LLMs). While jailbreak attacks on LLMs have been extensively studied, the security of ALMs with audio modalities remains largely unexplored. Currently, there is a lack of an adversarial audio dataset and a unified framework specifically designed to evaluate and compare attacks and ALMs. In this paper, we present JALMBench, the \textit{first} comprehensive benchmark to assess the safety of ALMs against jailbreak attacks. JALMBench includes a dataset containing 2,200 text samples and 51,381 audio samples with over 268 hours. It supports 12 mainstream ALMs, 4 text-transferred and 4 audio-originated attack methods, and 5 defense methods. Using JALMBench, we provide an in-depth analysis of attack efficiency, topic sensitivity, voice diversity, and attack representations. Additionally, we explore mitigation strategies for the attacks at both the prompt level and the response level.



## **11. Chain-of-Lure: A Synthetic Narrative-Driven Approach to Compromise Large Language Models**

cs.CR

25 pages, 4 figures

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17519v1) [paper-pdf](http://arxiv.org/pdf/2505.17519v1)

**Authors**: Wenhan Chang, Tianqing Zhu, Yu Zhao, Shuangyong Song, Ping Xiong, Wanlei Zhou, Yongxiang Li

**Abstract**: In the era of rapid generative AI development, interactions between humans and large language models face significant misusing risks. Previous research has primarily focused on black-box scenarios using human-guided prompts and white-box scenarios leveraging gradient-based LLM generation methods, neglecting the possibility that LLMs can act not only as victim models, but also as attacker models to harm other models. We proposes a novel jailbreaking method inspired by the Chain-of-Thought mechanism, where the attacker model uses mission transfer to conceal harmful user intent in dialogue and generates chained narrative lures to stimulate the reasoning capabilities of victim models, leading to successful jailbreaking. To enhance the attack success rate, we introduce a helper model that performs random narrative optimization on the narrative lures during multi-turn dialogues while ensuring alignment with the original intent, enabling the optimized lures to bypass the safety barriers of victim models effectively. Our experiments reveal that models with weaker safety mechanisms exhibit stronger attack capabilities, demonstrating that models can not only be exploited, but also help harm others. By incorporating toxicity scores, we employ third-party models to evaluate the harmfulness of victim models' responses to jailbreaking attempts. The study shows that using refusal keywords as an evaluation metric for attack success rates is significantly flawed because it does not assess whether the responses guide harmful questions, while toxicity scores measure the harm of generated content with more precision and its alignment with harmful questions. Our approach demonstrates outstanding performance, uncovering latent vulnerabilities in LLMs and providing data-driven feedback to optimize LLM safety mechanisms. We also discuss two defensive strategies to offer guidance on improving defense mechanisms.



## **12. Enhancing Adversarial Robustness of Vision Language Models via Adversarial Mixture Prompt Tuning**

cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17509v1) [paper-pdf](http://arxiv.org/pdf/2505.17509v1)

**Authors**: Shiji Zhao, Qihui Zhu, Shukun Xiong, Shouwei Ruan, Yize Fan, Ranjie Duan, Qing Guo, Xingxing Wei

**Abstract**: Large pre-trained Vision Language Models (VLMs) have excellent generalization capabilities but are highly susceptible to adversarial examples, presenting potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which finally leads to the overfitting phenomenon. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts can bring more robustness improvement than a longer prompt. Then we propose an adversarial tuning method named Adversarial Mixture Prompt Tuning (AMPT) to enhance the generalization towards various adversarial attacks for VLMs. AMPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the input adversarial image to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific aggregated text features aligning with different adversarial image features. A series of experiments show that our method can achieve better adversarial robustness than state-of-the-art methods on 11 datasets under different experimental settings.



## **13. Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models**

cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2501.18533v2) [paper-pdf](http://arxiv.org/pdf/2501.18533v2)

**Authors**: Yi Ding, Lijun Li, Bing Cao, Jing Shao

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable performance across a wide range of tasks. However, their deployment in safety-critical domains poses significant challenges. Existing safety fine-tuning methods, which focus on textual or multimodal content, fall short in addressing challenging cases or disrupt the balance between helpfulness and harmlessness. Our evaluation highlights a safety reasoning gap: these methods lack safety visual reasoning ability, leading to such bottlenecks. To address this limitation and enhance both visual perception and reasoning in safety-critical contexts, we propose a novel dataset that integrates multi-image inputs with safety Chain-of-Thought (CoT) labels as fine-grained reasoning logic to improve model performance. Specifically, we introduce the Multi-Image Safety (MIS) dataset, an instruction-following dataset tailored for multi-image safety scenarios, consisting of training and test splits. Our experiments demonstrate that fine-tuning InternVL2.5-8B with MIS significantly outperforms both powerful open-source models and API-based models in challenging multi-image tasks requiring safety-related visual reasoning. This approach not only delivers exceptional safety performance but also preserves general capabilities without any trade-offs. Specifically, fine-tuning with MIS increases average accuracy by 0.83% across five general benchmarks and reduces the Attack Success Rate (ASR) on multiple safety benchmarks by a large margin.



## **14. Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training**

cs.CL

Accepted by ACL 2025 main

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2407.09121v2) [paper-pdf](http://arxiv.org/pdf/2407.09121v2)

**Authors**: Youliang Yuan, Wenxiang Jiao, Wenxuan Wang, Jen-tse Huang, Jiahao Xu, Tian Liang, Pinjia He, Zhaopeng Tu

**Abstract**: This study addresses a critical gap in safety tuning practices for Large Language Models (LLMs) by identifying and tackling a refusal position bias within safety tuning data, which compromises the models' ability to appropriately refuse generating unsafe content. We introduce a novel approach, Decoupled Refusal Training (DeRTa), designed to empower LLMs to refuse compliance to harmful prompts at any response position, significantly enhancing their safety capabilities. DeRTa incorporates two novel components: (1) Maximum Likelihood Estimation (MLE) with Harmful Response Prefix, which trains models to recognize and avoid unsafe content by appending a segment of harmful response to the beginning of a safe response, and (2) Reinforced Transition Optimization (RTO), which equips models with the ability to transition from potential harm to safety refusal consistently throughout the harmful response sequence. Our empirical evaluation, conducted using LLaMA3 and Mistral model families across six attack scenarios, demonstrates that our method not only improves model safety without compromising performance but also surpasses baseline methods in defending against attacks.



## **15. How Secure Are Large Language Models (LLMs) for Navigation in Urban Environments?**

cs.RO

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2402.09546v2) [paper-pdf](http://arxiv.org/pdf/2402.09546v2)

**Authors**: Congcong Wen, Jiazhao Liang, Shuaihang Yuan, Hao Huang, Geeta Chandra Raju Bethala, Yu-Shen Liu, Mengyu Wang, Anthony Tzes, Yi Fang

**Abstract**: In the field of robotics and automation, navigation systems based on Large Language Models (LLMs) have recently demonstrated impressive performance. However, the security aspects of these systems have received relatively less attention. This paper pioneers the exploration of vulnerabilities in LLM-based navigation models in urban outdoor environments, a critical area given the widespread application of this technology in autonomous driving, logistics, and emergency services. Specifically, we introduce a novel Navigational Prompt Attack that manipulates LLM-based navigation models by perturbing the original navigational prompt, leading to incorrect actions. Based on the method of perturbation, our attacks are divided into two types: Navigational Prompt Insert (NPI) Attack and Navigational Prompt Swap (NPS) Attack. We conducted comprehensive experiments on an LLM-based navigation model that employs various LLMs for reasoning. Our results, derived from the Touchdown and Map2Seq street-view datasets under both few-shot learning and fine-tuning configurations, demonstrate notable performance declines across seven metrics in the face of both white-box and black-box attacks. Moreover, our attacks can be easily extended to other LLM-based navigation models with similarly effective results. These findings highlight the generalizability and transferability of the proposed attack, emphasizing the need for enhanced security in LLM-based navigation systems. As an initial countermeasure, we propose the Navigational Prompt Engineering (NPE) Defense strategy, which concentrates on navigation-relevant keywords to reduce the impact of adversarial attacks. While initial findings indicate that this strategy enhances navigational safety, there remains a critical need for the wider research community to develop stronger defense methods to effectively tackle the real-world challenges faced by these systems.



## **16. VEAttack: Downstream-agnostic Vision Encoder Attack against Large Vision Language Models**

cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17440v1) [paper-pdf](http://arxiv.org/pdf/2505.17440v1)

**Authors**: Hefei Mei, Zirui Wang, Shen You, Minjing Dong, Chang Xu

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities in multimodal understanding and generation, yet their vulnerability to adversarial attacks raises significant robustness concerns. While existing effective attacks always focus on task-specific white-box settings, these approaches are limited in the context of LVLMs, which are designed for diverse downstream tasks and require expensive full-model gradient computations. Motivated by the pivotal role and wide adoption of the vision encoder in LVLMs, we propose a simple yet effective Vision Encoder Attack (VEAttack), which targets the vision encoder of LVLMs only. Specifically, we propose to generate adversarial examples by minimizing the cosine similarity between the clean and perturbed visual features, without accessing the following large language models, task information, and labels. It significantly reduces the computational overhead while eliminating the task and label dependence of traditional white-box attacks in LVLMs. To make this simple attack effective, we propose to perturb images by optimizing image tokens instead of the classification token. We provide both empirical and theoretical evidence that VEAttack can easily generalize to various tasks. VEAttack has achieved a performance degradation of 94.5% on image caption task and 75.7% on visual question answering task. We also reveal some key observations to provide insights into LVLM attack/defense: 1) hidden layer variations of LLM, 2) token attention differential, 3) M\"obius band in transfer attack, 4) low sensitivity to attack steps. The code is available at https://github.com/hfmei/VEAttack-LVLM



## **17. Large Language Models Are Involuntary Truth-Tellers: Exploiting Fallacy Failure for Jailbreak Attacks**

cs.CL

Accepted to the main conference of EMNLP 2024

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2407.00869v3) [paper-pdf](http://arxiv.org/pdf/2407.00869v3)

**Authors**: Yue Zhou, Henry Peng Zou, Barbara Di Eugenio, Yang Zhang

**Abstract**: We find that language models have difficulties generating fallacious and deceptive reasoning. When asked to generate deceptive outputs, language models tend to leak honest counterparts but believe them to be false. Exploiting this deficiency, we propose a jailbreak attack method that elicits an aligned language model for malicious output. Specifically, we query the model to generate a fallacious yet deceptively real procedure for the harmful behavior. Since a fallacious procedure is generally considered fake and thus harmless by LLMs, it helps bypass the safeguard mechanism. Yet the output is factually harmful since the LLM cannot fabricate fallacious solutions but proposes truthful ones. We evaluate our approach over five safety-aligned large language models, comparing four previous jailbreak methods, and show that our approach achieves competitive performance with more harmful outputs. We believe the findings could be extended beyond model safety, such as self-verification and hallucination.



## **18. Three Minds, One Legend: Jailbreak Large Reasoning Model with Adaptive Stacked Ciphers**

cs.CL

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.16241v2) [paper-pdf](http://arxiv.org/pdf/2505.16241v2)

**Authors**: Viet-Anh Nguyen, Shiqian Zhao, Gia Dao, Runyi Hu, Yi Xie, Luu Anh Tuan

**Abstract**: Recently, Large Reasoning Models (LRMs) have demonstrated superior logical capabilities compared to traditional Large Language Models (LLMs), gaining significant attention. Despite their impressive performance, the potential for stronger reasoning abilities to introduce more severe security vulnerabilities remains largely underexplored. Existing jailbreak methods often struggle to balance effectiveness with robustness against adaptive safety mechanisms. In this work, we propose SEAL, a novel jailbreak attack that targets LRMs through an adaptive encryption pipeline designed to override their reasoning processes and evade potential adaptive alignment. Specifically, SEAL introduces a stacked encryption approach that combines multiple ciphers to overwhelm the models reasoning capabilities, effectively bypassing built-in safety mechanisms. To further prevent LRMs from developing countermeasures, we incorporate two dynamic strategies - random and adaptive - that adjust the cipher length, order, and combination. Extensive experiments on real-world reasoning models, including DeepSeek-R1, Claude Sonnet, and OpenAI GPT-o4, validate the effectiveness of our approach. Notably, SEAL achieves an attack success rate of 80.8% on GPT o4-mini, outperforming state-of-the-art baselines by a significant margin of 27.2%. Warning: This paper contains examples of inappropriate, offensive, and harmful content.



## **19. LLM-BSCVM: An LLM-Based Blockchain Smart Contract Vulnerability Management Framework**

cs.CR

10 pages, 8 figures

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.17416v1) [paper-pdf](http://arxiv.org/pdf/2505.17416v1)

**Authors**: Yanli Jin, Chunpei Li, Peng Fan, Peng Liu, Xianxian Li, Chen Liu, Wangjie Qiu

**Abstract**: Smart contracts are a key component of the Web 3.0 ecosystem, widely applied in blockchain services and decentralized applications. However, the automated execution feature of smart contracts makes them vulnerable to potential attacks due to inherent flaws, which can lead to severe security risks and financial losses, even threatening the integrity of the entire decentralized finance system. Currently, research on smart contract vulnerabilities has evolved from traditional program analysis methods to deep learning techniques, with the gradual introduction of Large Language Models. However, existing studies mainly focus on vulnerability detection, lacking systematic cause analysis and Vulnerability Repair. To address this gap, we propose LLM-BSCVM, a Large Language Model-based smart contract vulnerability management framework, designed to provide end-to-end vulnerability detection, analysis, repair, and evaluation capabilities for Web 3.0 ecosystem. LLM-BSCVM combines retrieval-augmented generation technology and multi-agent collaboration, introducing a three-stage method of Decompose-Retrieve-Generate. This approach enables smart contract vulnerability management through the collaborative efforts of six intelligent agents, specifically: vulnerability detection, cause analysis, repair suggestion generation, risk assessment, vulnerability repair, and patch evaluation. Experimental results demonstrate that LLM-BSCVM achieves a vulnerability detection accuracy and F1 score exceeding 91\% on benchmark datasets, comparable to the performance of state-of-the-art (SOTA) methods, while reducing the false positive rate from 7.2\% in SOTA methods to 5.1\%, thus enhancing the reliability of vulnerability management. Furthermore, LLM-BSCVM supports continuous security monitoring and governance of smart contracts through a knowledge base hot-swapping dynamic update mechanism.



## **20. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP**

cs.CV

ICML 2025

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.05528v2) [paper-pdf](http://arxiv.org/pdf/2505.05528v2)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.



## **21. StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization**

cs.IR

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2504.05804v2) [paper-pdf](http://arxiv.org/pdf/2504.05804v2)

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present $\textbf{StealthRank}$, a novel adversarial attack method that manipulates LLM-driven ranking systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within item or document descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target items while avoiding explicit manipulation traces. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven ranking systems. Our code is publicly available at $\href{https://github.com/Tangyiming205069/controllable-seo}{here}$.



## **22. Hidden Ghost Hand: Unveiling Backdoor Vulnerabilities in MLLM-Powered Mobile GUI Agents**

cs.CL

25 pages, 10 figures, 12 Tables

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2505.14418v2) [paper-pdf](http://arxiv.org/pdf/2505.14418v2)

**Authors**: Pengzhou Cheng, Haowen Hu, Zheng Wu, Zongru Wu, Tianjie Ju, Zhuosheng Zhang, Gongshen Liu

**Abstract**: Graphical user interface (GUI) agents powered by multimodal large language models (MLLMs) have shown greater promise for human-interaction. However, due to the high fine-tuning cost, users often rely on open-source GUI agents or APIs offered by AI providers, which introduces a critical but underexplored supply chain threat: backdoor attacks. In this work, we first unveil that MLLM-powered GUI agents naturally expose multiple interaction-level triggers, such as historical steps, environment states, and task progress. Based on this observation, we introduce AgentGhost, an effective and stealthy framework for red-teaming backdoor attacks. Specifically, we first construct composite triggers by combining goal and interaction levels, allowing GUI agents to unintentionally activate backdoors while ensuring task utility. Then, we formulate backdoor injection as a Min-Max optimization problem that uses supervised contrastive learning to maximize the feature difference across sample classes at the representation space, improving flexibility of the backdoor. Meanwhile, it adopts supervised fine-tuning to minimize the discrepancy between backdoor and clean behavior generation, enhancing effectiveness and utility. Extensive evaluations of various agent models in two established mobile benchmarks show that AgentGhost is effective and generic, with attack accuracy that reaches 99.7\% on three attack objectives, and shows stealthiness with only 1\% utility degradation. Furthermore, we tailor a defense method against AgentGhost that reduces the attack accuracy to 22.1\%. Our code is available at \texttt{anonymous}.



## **23. Advancing Security with Digital Twins: A Comprehensive Survey**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17310v1) [paper-pdf](http://arxiv.org/pdf/2505.17310v1)

**Authors**: Blessing Airehenbuwa, Touseef Hasan, Souvika Sarkar, Ujjwal Guin

**Abstract**: The proliferation of electronic devices has greatly transformed every aspect of human life, such as communication, healthcare, transportation, and energy. Unfortunately, the global electronics supply chain is vulnerable to various attacks, including piracy of intellectual properties, tampering, counterfeiting, information leakage, side-channel, and fault injection attacks, due to the complex nature of electronic products and vulnerabilities present in them. Although numerous solutions have been proposed to address these threats, significant gaps remain, particularly in providing scalable and comprehensive protection against emerging attacks. Digital twin, a dynamic virtual replica of a physical system, has emerged as a promising solution to address these issues by providing backward traceability, end-to-end visibility, and continuous verification of component integrity and behavior. In this paper, we present a comprehensive survey of the application of digital twins based on their functional role and application domains. We comprehensively present recent digital twin-based security implementations, including their role in cyber-physical systems, Internet of Things, and cryptographic systems, detection of counterfeit electronics, intrusion detection, fault injection, and side-channel leakage. To the best of our knowledge, it is the first study to consolidate these security use cases into a unified reference. The paper also explores the integration of large language models with digital twins for enhanced security and discusses current challenges, solutions, and future research directions.



## **24. TrustRAG: Enhancing Robustness and Trustworthiness in Retrieval-Augmented Generation**

cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2501.00879v3) [paper-pdf](http://arxiv.org/pdf/2501.00879v3)

**Authors**: Huichi Zhou, Kin-Hei Lee, Zhonghao Zhan, Yue Chen, Zhenhao Li, Zhaoyang Wang, Hamed Haddadi, Emine Yilmaz

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by integrating external knowledge sources, enabling more accurate and contextually relevant responses tailored to user queries. These systems, however, remain susceptible to corpus poisoning attacks, which can severely impair the performance of LLMs. To address this challenge, we propose TrustRAG, a robust framework that systematically filters malicious and irrelevant content before it is retrieved for generation. Our approach employs a two-stage defense mechanism. The first stage implements a cluster filtering strategy to detect potential attack patterns. The second stage employs a self-assessment process that harnesses the internal capabilities of LLMs to detect malicious documents and resolve inconsistencies. TrustRAG provides a plug-and-play, training-free module that integrates seamlessly with any open- or closed-source language model. Extensive experiments demonstrate that TrustRAG delivers substantial improvements in retrieval accuracy, efficiency, and attack resistance.



## **25. Invisible Prompts, Visible Threats: Malicious Font Injection in External Resources for Large Language Models**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16957v1) [paper-pdf](http://arxiv.org/pdf/2505.16957v1)

**Authors**: Junjie Xiong, Changjia Zhu, Shuhang Lin, Chong Zhang, Yongfeng Zhang, Yao Liu, Lingyao Li

**Abstract**: Large Language Models (LLMs) are increasingly equipped with capabilities of real-time web search and integrated with protocols like Model Context Protocol (MCP). This extension could introduce new security vulnerabilities. We present a systematic investigation of LLM vulnerabilities to hidden adversarial prompts through malicious font injection in external resources like webpages, where attackers manipulate code-to-glyph mapping to inject deceptive content which are invisible to users. We evaluate two critical attack scenarios: (1) "malicious content relay" and (2) "sensitive data leakage" through MCP-enabled tools. Our experiments reveal that indirect prompts with injected malicious font can bypass LLM safety mechanisms through external resources, achieving varying success rates based on data sensitivity and prompt design. Our research underscores the urgent need for enhanced security measures in LLM deployments when processing external content.



## **26. MixAT: Combining Continuous and Discrete Adversarial Training for LLMs**

cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16947v1) [paper-pdf](http://arxiv.org/pdf/2505.16947v1)

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: Despite recent efforts in Large Language Models (LLMs) safety and alignment, current adversarial attacks on frontier LLMs are still able to force harmful generations consistently. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. As these relaxations do not correspond to discrete input tokens, such latent training methods often leave models vulnerable to a diverse set of discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.



## **27. Backdoor Cleaning without External Guidance in MLLM Fine-tuning**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16916v1) [paper-pdf](http://arxiv.org/pdf/2505.16916v1)

**Authors**: Xuankun Rong, Wenke Huang, Jian Liang, Jinhe Bi, Xun Xiao, Yiming Li, Bo Du, Mang Ye

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly deployed in fine-tuning-as-a-service (FTaaS) settings, where user-submitted datasets adapt general-purpose models to downstream tasks. This flexibility, however, introduces serious security risks, as malicious fine-tuning can implant backdoors into MLLMs with minimal effort. In this paper, we observe that backdoor triggers systematically disrupt cross-modal processing by causing abnormal attention concentration on non-semantic regions--a phenomenon we term attention collapse. Based on this insight, we propose Believe Your Eyes (BYE), a data filtering framework that leverages attention entropy patterns as self-supervised signals to identify and filter backdoor samples. BYE operates via a three-stage pipeline: (1) extracting attention maps using the fine-tuned model, (2) computing entropy scores and profiling sensitive layers via bimodal separation, and (3) performing unsupervised clustering to remove suspicious samples. Unlike prior defenses, BYE equires no clean supervision, auxiliary labels, or model modifications. Extensive experiments across various datasets, models, and diverse trigger types validate BYE's effectiveness: it achieves near-zero attack success rates while maintaining clean-task performance, offering a robust and generalizable solution against backdoor threats in MLLMs.



## **28. CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16888v1) [paper-pdf](http://arxiv.org/pdf/2505.16888v1)

**Authors**: Viet Pham, Thai Le

**Abstract**: Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available.



## **29. Safe RLHF-V: Safe Reinforcement Learning from Multi-modal Human Feedback**

cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2503.17682v2) [paper-pdf](http://arxiv.org/pdf/2503.17682v2)

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Conghui Zhang, Han Zhu, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Yida Tang, Sirui Han, Yike Guo, Yaodong Yang

**Abstract**: Multimodal large language models (MLLMs) are essential for building general-purpose AI assistants; however, they pose increasing safety risks. How can we ensure safety alignment of MLLMs to prevent undesired behaviors? Going further, it is critical to explore how to fine-tune MLLMs to preserve capabilities while meeting safety constraints. Fundamentally, this challenge can be formulated as a min-max optimization problem. However, existing datasets have not yet disentangled single preference signals into explicit safety constraints, hindering systematic investigation in this direction. Moreover, it remains an open question whether such constraints can be effectively incorporated into the optimization process for multi-modal models. In this work, we present the first exploration of the Safe RLHF-V -- the first multimodal safety alignment framework. The framework consists of: $\mathbf{(I)}$ BeaverTails-V, the first open-source dataset featuring dual preference annotations for helpfulness and safety, supplemented with multi-level safety labels (minor, moderate, severe); $\mathbf{(II)}$ Beaver-Guard-V, a multi-level guardrail system to proactively defend against unsafe queries and adversarial attacks. Applying the guard model over five rounds of filtering and regeneration significantly enhances the precursor model's overall safety by an average of 40.9%. $\mathbf{(III)}$ Based on dual preference, we initiate the first exploration of multi-modal safety alignment within a constrained optimization. Experimental results demonstrate that Safe RLHF effectively improves both model helpfulness and safety. Specifically, Safe RLHF-V enhances model safety by 34.2% and helpfulness by 34.3%.



## **30. Accidental Misalignment: Fine-Tuning Language Models Induces Unexpected Vulnerability**

cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16789v1) [paper-pdf](http://arxiv.org/pdf/2505.16789v1)

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin

**Abstract**: As large language models gain popularity, their vulnerability to adversarial attacks remains a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Misalignment, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity within our experimental datasets. We then evaluate the adversarial performance of these fine-tuned models and assess how dataset factors correlate with attack success rates. Lastly, we explore potential causal links, offering new insights into adversarial defense strategies and highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_misalignment.



## **31. When Safety Detectors Aren't Enough: A Stealthy and Effective Jailbreak Attack on LLMs via Steganographic Techniques**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16765v1) [paper-pdf](http://arxiv.org/pdf/2505.16765v1)

**Authors**: Jianing Geng, Biao Yi, Zekun Fei, Tongxi Wu, Lihai Nie, Zheli Liu

**Abstract**: Jailbreak attacks pose a serious threat to large language models (LLMs) by bypassing built-in safety mechanisms and leading to harmful outputs. Studying these attacks is crucial for identifying vulnerabilities and improving model security. This paper presents a systematic survey of jailbreak methods from the novel perspective of stealth. We find that existing attacks struggle to simultaneously achieve toxic stealth (concealing toxic content) and linguistic stealth (maintaining linguistic naturalness). Motivated by this, we propose StegoAttack, a fully stealthy jailbreak attack that uses steganography to hide the harmful query within benign, semantically coherent text. The attack then prompts the LLM to extract the hidden query and respond in an encrypted manner. This approach effectively hides malicious intent while preserving naturalness, allowing it to evade both built-in and external safety mechanisms. We evaluate StegoAttack on four safety-aligned LLMs from major providers, benchmarking against eight state-of-the-art methods. StegoAttack achieves an average attack success rate (ASR) of 92.00%, outperforming the strongest baseline by 11.0%. Its ASR drops by less than 1% even under external detection (e.g., Llama Guard). Moreover, it attains the optimal comprehensive scores on stealth detection metrics, demonstrating both high efficacy and exceptional stealth capabilities. The code is available at https://anonymous.4open.science/r/StegoAttack-Jail66



## **32. BitHydra: Towards Bit-flip Inference Cost Attack against Large Language Models**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16670v1) [paper-pdf](http://arxiv.org/pdf/2505.16670v1)

**Authors**: Xiaobei Yan, Yiming Li, Zhaoxin Fan, Han Qiu, Tianwei Zhang

**Abstract**: Large language models (LLMs) have shown impressive capabilities across a wide range of applications, but their ever-increasing size and resource demands make them vulnerable to inference cost attacks, where attackers induce victim LLMs to generate the longest possible output content. In this paper, we revisit existing inference cost attacks and reveal that these methods can hardly produce large-scale malicious effects since they are self-targeting, where attackers are also the users and therefore have to execute attacks solely through the inputs, whose generated content will be charged by LLMs and can only directly influence themselves. Motivated by these findings, this paper introduces a new type of inference cost attacks (dubbed 'bit-flip inference cost attack') that target the victim model itself rather than its inputs. Specifically, we design a simple yet effective method (dubbed 'BitHydra') to effectively flip critical bits of model parameters. This process is guided by a loss function designed to suppress <EOS> token's probability with an efficient critical bit search algorithm, thus explicitly defining the attack objective and enabling effective optimization. We evaluate our method on 11 LLMs ranging from 1.5B to 14B parameters under both int8 and float16 settings. Experimental results demonstrate that with just 4 search samples and as few as 3 bit flips, BitHydra can force 100% of test prompts to reach the maximum generation length (e.g., 2048 tokens) on representative LLMs such as LLaMA3, highlighting its efficiency, scalability, and strong transferability across unseen inputs.



## **33. Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models**

cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2412.16555v2) [paper-pdf](http://arxiv.org/pdf/2412.16555v2)

**Authors**: Yanxu Mao, Peipei Liu, Tiehan Cui, Zhaoteng Yan, Congying Liu, Datao You

**Abstract**: Large language models (LLMs) are widely applied in various fields of society due to their powerful reasoning, understanding, and generation capabilities. However, the security issues associated with these models are becoming increasingly severe. Jailbreaking attacks, as an important method for detecting vulnerabilities in LLMs, have been explored by researchers who attempt to induce these models to generate harmful content through various attack methods. Nevertheless, existing jailbreaking methods face numerous limitations, such as excessive query counts, limited coverage of jailbreak modalities, low attack success rates, and simplistic evaluation methods. To overcome these constraints, this paper proposes a multimodal jailbreaking method: JMLLM. This method integrates multiple strategies to perform comprehensive jailbreak attacks across text, visual, and auditory modalities. Additionally, we contribute a new and comprehensive dataset for multimodal jailbreaking research: TriJail, which includes jailbreak prompts for all three modalities. Experiments on the TriJail dataset and the benchmark dataset AdvBench, conducted on 13 popular LLMs, demonstrate advanced attack success rates and significant reduction in time overhead.



## **34. From Evaluation to Defense: Advancing Safety in Video Large Language Models**

cs.CV

49 pages, 12 figures, 17 tables

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16643v1) [paper-pdf](http://arxiv.org/pdf/2505.16643v1)

**Authors**: Yiwei Sun, Peiqi Jiang, Chuanbin Liu, Luohao Lin, Zhiying Lu, Hongtao Xie

**Abstract**: While the safety risks of image-based large language models have been extensively studied, their video-based counterparts (Video LLMs) remain critically under-examined. To systematically study this problem, we introduce \textbf{VideoSafetyBench (VSB-77k) - the first large-scale, culturally diverse benchmark for Video LLM safety}, which compromises 77,646 video-query pairs and spans 19 principal risk categories across 10 language communities. \textit{We reveal that integrating video modality degrades safety performance by an average of 42.3\%, exposing systemic risks in multimodal attack exploitation.} To address this vulnerability, we propose \textbf{VideoSafety-R1}, a dual-stage framework achieving unprecedented safety gains through two innovations: (1) Alarm Token-Guided Safety Fine-Tuning (AT-SFT) injects learnable alarm tokens into visual and textual sequences, enabling explicit harm perception across modalities via multitask objectives. (2) Then, Safety-Guided GRPO enhances defensive reasoning through dynamic policy optimization with rule-based rewards derived from dual-modality verification. These components synergize to shift safety alignment from passive harm recognition to active reasoning. The resulting framework achieves a 65.1\% improvement on VSB-Eval-HH, and improves by 59.1\%, 44.3\%, and 15.0\% on the image safety datasets MMBench, VLGuard, and FigStep, respectively. \textit{Our codes are available in the supplementary materials.} \textcolor{red}{Warning: This paper contains examples of harmful language and videos, and reader discretion is recommended.}



## **35. BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization**

cs.CR

19 pages, 12 figures, 6 tables

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16640v1) [paper-pdf](http://arxiv.org/pdf/2505.16640v1)

**Authors**: Xueyang Zhou, Guiyao Tie, Guowen Zhang, Hechang Wang, Pan Zhou, Lichao Sun

**Abstract**: Vision-Language-Action (VLA) models have advanced robotic control by enabling end-to-end decision-making directly from multimodal inputs. However, their tightly coupled architectures expose novel security vulnerabilities. Unlike traditional adversarial perturbations, backdoor attacks represent a stealthier, persistent, and practically significant threat-particularly under the emerging Training-as-a-Service paradigm-but remain largely unexplored in the context of VLA models. To address this gap, we propose BadVLA, a backdoor attack method based on Objective-Decoupled Optimization, which for the first time exposes the backdoor vulnerabilities of VLA models. Specifically, it consists of a two-stage process: (1) explicit feature-space separation to isolate trigger representations from benign inputs, and (2) conditional control deviations that activate only in the presence of the trigger, while preserving clean-task performance. Empirical results on multiple VLA benchmarks demonstrate that BadVLA consistently achieves near-100% attack success rates with minimal impact on clean task accuracy. Further analyses confirm its robustness against common input perturbations, task transfers, and model fine-tuning, underscoring critical security vulnerabilities in current VLA deployments. Our work offers the first systematic investigation of backdoor vulnerabilities in VLA models, highlighting an urgent need for secure and trustworthy embodied model design practices. We have released the project page at https://badvla-project.github.io/.



## **36. CTRAP: Embedding Collapse Trap to Safeguard Large Language Models from Harmful Fine-Tuning**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16559v1) [paper-pdf](http://arxiv.org/pdf/2505.16559v1)

**Authors**: Biao Yi, Tiansheng Huang, Baolei Zhang, Tong Li, Lihai Nie, Zheli Liu, Li Shen

**Abstract**: Fine-tuning-as-a-service, while commercially successful for Large Language Model (LLM) providers, exposes models to harmful fine-tuning attacks. As a widely explored defense paradigm against such attacks, unlearning attempts to remove malicious knowledge from LLMs, thereby essentially preventing them from being used to perform malicious tasks. However, we highlight a critical flaw: the powerful general adaptability of LLMs allows them to easily bypass selective unlearning by rapidly relearning or repurposing their capabilities for harmful tasks. To address this fundamental limitation, we propose a paradigm shift: instead of selective removal, we advocate for inducing model collapse--effectively forcing the model to "unlearn everything"--specifically in response to updates characteristic of malicious adaptation. This collapse directly neutralizes the very general capabilities that attackers exploit, tackling the core issue unaddressed by selective unlearning. We introduce the Collapse Trap (CTRAP) as a practical mechanism to implement this concept conditionally. Embedded during alignment, CTRAP pre-configures the model's reaction to subsequent fine-tuning dynamics. If updates during fine-tuning constitute a persistent attempt to reverse safety alignment, the pre-configured trap triggers a progressive degradation of the model's core language modeling abilities, ultimately rendering it inert and useless for the attacker. Crucially, this collapse mechanism remains dormant during benign fine-tuning, ensuring the model's utility and general capabilities are preserved for legitimate users. Extensive empirical results demonstrate that CTRAP effectively counters harmful fine-tuning risks across various LLMs and attack settings, while maintaining high performance in benign scenarios. Our code is available at https://anonymous.4open.science/r/CTRAP.



## **37. Implicit Jailbreak Attacks via Cross-Modal Information Concealment on Vision-Language Models**

cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16446v1) [paper-pdf](http://arxiv.org/pdf/2505.16446v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Multimodal large language models (MLLMs) enable powerful cross-modal reasoning capabilities. However, the expanded input space introduces new attack surfaces. Previous jailbreak attacks often inject malicious instructions from text into less aligned modalities, such as vision. As MLLMs increasingly incorporate cross-modal consistency and alignment mechanisms, such explicit attacks become easier to detect and block. In this work, we propose a novel implicit jailbreak framework termed IJA that stealthily embeds malicious instructions into images via least significant bit steganography and couples them with seemingly benign, image-related textual prompts. To further enhance attack effectiveness across diverse MLLMs, we incorporate adversarial suffixes generated by a surrogate model and introduce a template optimization module that iteratively refines both the prompt and embedding based on model feedback. On commercial models like GPT-4o and Gemini-1.5 Pro, our method achieves attack success rates of over 90% using an average of only 3 queries.



## **38. MTSA: Multi-turn Safety Alignment for LLMs through Multi-round Red-teaming**

cs.CR

19 pages,6 figures,ACL2025

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17147v1) [paper-pdf](http://arxiv.org/pdf/2505.17147v1)

**Authors**: Weiyang Guo, Jing Li, Wenya Wang, YU LI, Daojing He, Jun Yu, Min Zhang

**Abstract**: The proliferation of jailbreak attacks against large language models (LLMs) highlights the need for robust security measures. However, in multi-round dialogues, malicious intentions may be hidden in interactions, leading LLMs to be more prone to produce harmful responses. In this paper, we propose the \textbf{M}ulti-\textbf{T}urn \textbf{S}afety \textbf{A}lignment (\ourapproach) framework, to address the challenge of securing LLMs in multi-round interactions. It consists of two stages: In the thought-guided attack learning stage, the red-team model learns about thought-guided multi-round jailbreak attacks to generate adversarial prompts. In the adversarial iterative optimization stage, the red-team model and the target model continuously improve their respective capabilities in interaction. Furthermore, we introduce a multi-turn reinforcement learning algorithm based on future rewards to enhance the robustness of safety alignment. Experimental results show that the red-team model exhibits state-of-the-art attack capabilities, while the target model significantly improves its performance on safety benchmarks.



## **39. Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems**

cs.IR

7 pages,3 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16367v1) [paper-pdf](http://arxiv.org/pdf/2505.16367v1)

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Yixing Fan

**Abstract**: Retrieval-augmented generation (RAG) systems can effectively mitigate the hallucination problem of large language models (LLMs),but they also possess inherent vulnerabilities. Identifying these weaknesses before the large-scale real-world deployment of RAG systems is of great importance, as it lays the foundation for building more secure and robust RAG systems in the future. Existing adversarial attack methods typically exploit knowledge base poisoning to probe the vulnerabilities of RAG systems, which can effectively deceive standard RAG models. However, with the rapid advancement of deep reasoning capabilities in modern LLMs, previous approaches that merely inject incorrect knowledge are inadequate when attacking RAG systems equipped with deep reasoning abilities. Inspired by the deep thinking capabilities of LLMs, this paper extracts reasoning process templates from R1-based RAG systems, uses these templates to wrap erroneous knowledge into adversarial documents, and injects them into the knowledge base to attack RAG systems. The key idea of our approach is that adversarial documents, by simulating the chain-of-thought patterns aligned with the model's training signals, may be misinterpreted by the model as authentic historical reasoning processes, thus increasing their likelihood of being referenced. Experiments conducted on the MS MARCO passage ranking dataset demonstrate the effectiveness of our proposed method.



## **40. PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation**

cs.IR

29 pages

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.12574v3) [paper-pdf](http://arxiv.org/pdf/2505.12574v3)

**Authors**: Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong, Liang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems, widely used to improve the factual grounding of large language models (LLMs), are increasingly vulnerable to poisoning attacks, where adversaries inject manipulated content into the retriever's corpus. While prior research has predominantly focused on single-attacker settings, real-world scenarios often involve multiple, competing attackers with conflicting objectives. In this work, we introduce PoisonArena, the first benchmark to systematically study and evaluate competing poisoning attacks in RAG. We formalize the multi-attacker threat model, where attackers vie to control the answer to the same query using mutually exclusive misinformation. PoisonArena leverages the Bradley-Terry model to quantify each method's competitive effectiveness in such adversarial environments. Through extensive experiments on the Natural Questions and MS MARCO datasets, we demonstrate that many attack strategies successful in isolation fail under competitive pressure. Our findings highlight the limitations of conventional evaluation metrics like Attack Success Rate (ASR) and F1 score and underscore the need for competitive evaluation to assess real-world attack robustness. PoisonArena provides a standardized framework to benchmark and develop future attack and defense strategies under more realistic, multi-adversary conditions. Project page: https://github.com/yxf203/PoisonArena.



## **41. PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks**

cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.13862v2) [paper-pdf](http://arxiv.org/pdf/2505.13862v2)

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.



## **42. Robustifying Vision-Language Models via Dynamic Token Reweighting**

cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17132v1) [paper-pdf](http://arxiv.org/pdf/2505.17132v1)

**Authors**: Tanqiu Jiang, Jiacheng Liang, Rongyi Zhu, Jiawei Zhou, Fenglong Ma, Ting Wang

**Abstract**: Large vision-language models (VLMs) are highly vulnerable to jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that \sys outperforms existing defenses in both attack robustness and benign task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. The code for replicating DTR is available: https://anonymous.4open.science/r/DTR-2755 (warning: this paper contains potentially harmful content generated by VLMs.)



## **43. Keep Security! Benchmarking Security Policy Preservation in Large Language Model Contexts Against Indirect Attacks in Question Answering**

cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15805v1) [paper-pdf](http://arxiv.org/pdf/2505.15805v1)

**Authors**: Hwan Chang, Yumin Kim, Yonghyun Jun, Hwanhee Lee

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains such as enterprise and government, ensuring that they adhere to user-defined security policies within context is critical-especially with respect to information non-disclosure. While prior LLM studies have focused on general safety and socially sensitive data, large-scale benchmarks for contextual security preservation against attacks remain lacking. To address this, we introduce a novel large-scale benchmark dataset, CoPriva, evaluating LLM adherence to contextual non-disclosure policies in question answering. Derived from realistic contexts, our dataset includes explicit policies and queries designed as direct and challenging indirect attacks seeking prohibited information. We evaluate 10 LLMs on our benchmark and reveal a significant vulnerability: many models violate user-defined policies and leak sensitive information. This failure is particularly severe against indirect attacks, highlighting a critical gap in current LLM safety alignment for sensitive applications. Our analysis reveals that while models can often identify the correct answer to a query, they struggle to incorporate policy constraints during generation. In contrast, they exhibit a partial ability to revise outputs when explicitly prompted. Our findings underscore the urgent need for more robust methods to guarantee contextual security.



## **44. Reverse Engineering Human Preferences with Reinforcement Learning**

cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15795v1) [paper-pdf](http://arxiv.org/pdf/2505.15795v1)

**Authors**: Lisa Alazraki, Tan Yi-Chern, Jon Ander Campos, Maximilian Mozes, Marek Rei, Max Bartolo

**Abstract**: The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.



## **45. Scalable Defense against In-the-wild Jailbreaking Attacks with Safety Context Retrieval**

cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15753v1) [paper-pdf](http://arxiv.org/pdf/2505.15753v1)

**Authors**: Taiye Chen, Zeming Wei, Ang Li, Yisen Wang

**Abstract**: Large Language Models (LLMs) are known to be vulnerable to jailbreaking attacks, wherein adversaries exploit carefully engineered prompts to induce harmful or unethical responses. Such threats have raised critical concerns about the safety and reliability of LLMs in real-world deployment. While existing defense mechanisms partially mitigate such risks, subsequent advancements in adversarial techniques have enabled novel jailbreaking methods to circumvent these protections, exposing the limitations of static defense frameworks. In this work, we explore defending against evolving jailbreaking threats through the lens of context retrieval. First, we conduct a preliminary study demonstrating that even a minimal set of safety-aligned examples against a particular jailbreak can significantly enhance robustness against this attack pattern. Building on this insight, we further leverage the retrieval-augmented generation (RAG) techniques and propose Safety Context Retrieval (SCR), a scalable and robust safeguarding paradigm for LLMs against jailbreaking. Our comprehensive experiments demonstrate how SCR achieves superior defensive performance against both established and emerging jailbreaking tactics, contributing a new paradigm to LLM safety. Our code will be available upon publication.



## **46. Shaping the Safety Boundaries: Understanding and Defending Against Jailbreaks in Large Language Models**

cs.CL

17 pages, 9 figures

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2412.17034v2) [paper-pdf](http://arxiv.org/pdf/2412.17034v2)

**Authors**: Lang Gao, Jiahui Geng, Xiangliang Zhang, Preslav Nakov, Xiuying Chen

**Abstract**: Jailbreaking in Large Language Models (LLMs) is a major security concern as it can deceive LLMs to generate harmful text. Yet, there is still insufficient understanding of how jailbreaking works, which makes it hard to develop effective defense strategies. We aim to shed more light into this issue: we conduct a detailed large-scale analysis of seven different jailbreak methods and find that these disagreements stem from insufficient observation samples. In particular, we introduce \textit{safety boundary}, and we find that jailbreaks shift harmful activations outside that safety boundary, where LLMs are less sensitive to harmful information. We also find that the low and the middle layers are critical in such shifts, while deeper layers have less impact. Leveraging on these insights, we propose a novel defense called \textbf{Activation Boundary Defense} (ABD), which adaptively constrains the activations within the safety boundary. We further use Bayesian optimization to selectively apply the defense method to the low and the middle layers. Our experiments on several benchmarks show that ABD achieves an average DSR of over 98\% against various forms of jailbreak attacks, with less than 2\% impact on the model's general capabilities.



## **47. Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses**

cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15738v1) [paper-pdf](http://arxiv.org/pdf/2505.15738v1)

**Authors**: Xiaoxue Yang, Bozhidar Stevanoski, Matthieu Meeus, Yves-Alexandre de Montjoye

**Abstract**: Large language models (LLMs) are rapidly deployed in real-world applications ranging from chatbots to agentic systems. Alignment is one of the main approaches used to defend against attacks such as prompt injection and jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even against Greedy Coordinate Gradient (GCG), a white-box attack that generates adversarial suffixes to induce attacker-desired outputs. However, this search space over discrete tokens is extremely large, making the task of finding successful attacks difficult. GCG has, for instance, been shown to converge to local minima, making it sensitive to initialization choices. In this paper, we assess the future-proof robustness of these defenses using a more informed threat model: attackers who have access to some information about the alignment process. Specifically, we propose an informed white-box attack leveraging the intermediate model checkpoints to initialize GCG, with each checkpoint acting as a stepping stone for the next one. We show this approach to be highly effective across state-of-the-art (SOTA) defenses and models. We further show our informed initialization to outperform other initialization methods and show a gradient-informed checkpoint selection strategy to greatly improve attack performance and efficiency. Importantly, we also show our method to successfully find universal adversarial suffixes -- single suffixes effective across diverse inputs. Our results show that, contrary to previous beliefs, effective adversarial suffixes do exist against SOTA alignment-based defenses, that these can be found by existing attack methods when adversaries exploit alignment knowledge, and that even universal suffixes exist. Taken together, our results highlight the brittleness of current alignment-based methods and the need to consider stronger threat models when testing the safety of LLMs.



## **48. SQL Injection Jailbreak: A Structural Disaster of Large Language Models**

cs.CR

Accepted by findings of ACL 2025

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2411.01565v6) [paper-pdf](http://arxiv.org/pdf/2411.01565v6)

**Authors**: Jiawei Zhao, Kejiang Chen, Weiming Zhang, Nenghai Yu

**Abstract**: Large Language Models (LLMs) are susceptible to jailbreak attacks that can induce them to generate harmful content. Previous jailbreak methods primarily exploited the internal properties or capabilities of LLMs, such as optimization-based jailbreak methods and methods that leveraged the model's context-learning abilities. In this paper, we introduce a novel jailbreak method, SQL Injection Jailbreak (SIJ), which targets the external properties of LLMs, specifically, the way LLMs construct input prompts. By injecting jailbreak information into user prompts, SIJ successfully induces the model to output harmful content. For open-source models, SIJ achieves near 100% attack success rates on five well-known LLMs on the AdvBench and HEx-PHI, while incurring lower time costs compared to previous methods. For closed-source models, SIJ achieves an average attack success rate over 85% across five models in the GPT and Doubao series. Additionally, SIJ exposes a new vulnerability in LLMs that urgently requires mitigation. To address this, we propose a simple adaptive defense method called Self-Reminder-Key to counter SIJ and demonstrate its effectiveness through experimental results. Our code is available at https://github.com/weiyezhimeng/SQL-Injection-Jailbreak.



## **49. Be Careful When Fine-tuning On Open-Source LLMs: Your Fine-tuning Data Could Be Secretly Stolen!**

cs.CL

19 pages

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15656v1) [paper-pdf](http://arxiv.org/pdf/2505.15656v1)

**Authors**: Zhexin Zhang, Yuhao Sun, Junxiao Yang, Shiyao Cui, Hongning Wang, Minlie Huang

**Abstract**: Fine-tuning on open-source Large Language Models (LLMs) with proprietary data is now a standard practice for downstream developers to obtain task-specific LLMs. Surprisingly, we reveal a new and concerning risk along with the practice: the creator of the open-source LLMs can later extract the private downstream fine-tuning data through simple backdoor training, only requiring black-box access to the fine-tuned downstream model. Our comprehensive experiments, across 4 popularly used open-source models with 3B to 32B parameters and 2 downstream datasets, suggest that the extraction performance can be strikingly high: in practical settings, as much as 76.3% downstream fine-tuning data (queries) out of a total 5,000 samples can be perfectly extracted, and the success rate can increase to 94.9% in more ideal settings. We also explore a detection-based defense strategy but find it can be bypassed with improved attack. Overall, we highlight the emergency of this newly identified data breaching risk in fine-tuning, and we hope that more follow-up research could push the progress of addressing this concerning risk. The code and data used in our experiments are released at https://github.com/thu-coai/Backdoor-Data-Extraction.



## **50. SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings**

cs.CL

Accepted in ACL 2025 Main Track

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2502.12562v2) [paper-pdf](http://arxiv.org/pdf/2502.12562v2)

**Authors**: Weikai Lu, Hao Peng, Huiping Zhuang, Cen Chen, Ziqian Zeng

**Abstract**: Multimodal Large Language Models (MLLMs) have serious security vulnerabilities.While safety alignment using multimodal datasets consisting of text and data of additional modalities can effectively enhance MLLM's security, it is costly to construct these datasets. Existing low-resource security alignment methods, including textual alignment, have been found to struggle with the security risks posed by additional modalities. To address this, we propose Synthetic Embedding augmented safety Alignment (SEA), which optimizes embeddings of additional modality through gradient updates to expand textual datasets. This enables multimodal safety alignment training even when only textual data is available. Extensive experiments on image, video, and audio-based MLLMs demonstrate that SEA can synthesize a high-quality embedding on a single RTX3090 GPU within 24 seconds. SEA significantly improves the security of MLLMs when faced with threats from additional modalities. To assess the security risks introduced by video and audio, we also introduced a new benchmark called VA-SafetyBench. High attack success rates across multiple MLLMs validate its challenge. Our code and data will be available at https://github.com/ZeroNLP/SEA.



