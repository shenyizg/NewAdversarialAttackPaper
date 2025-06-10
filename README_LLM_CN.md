# Latest Large Language Model Attack Papers
**update at 2025-06-10 15:42:08**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. TokenBreak: Bypassing Text Classification Models Through Token Manipulation**

TokenBreak：通过令牌操纵来破解文本分类模型 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07948v1) [paper-pdf](http://arxiv.org/pdf/2506.07948v1)

**Authors**: Kasimir Schulz, Kenneth Yeung, Kieran Evans

**Abstract**: Natural Language Processing (NLP) models are used for text-related tasks such as classification and generation. To complete these tasks, input data is first tokenized from human-readable text into a format the model can understand, enabling it to make inferences and understand context. Text classification models can be implemented to guard against threats such as prompt injection attacks against Large Language Models (LLMs), toxic input and cybersecurity risks such as spam emails. In this paper, we introduce TokenBreak: a novel attack that can bypass these protection models by taking advantage of the tokenization strategy they use. This attack technique manipulates input text in such a way that certain models give an incorrect classification. Importantly, the end target (LLM or email recipient) can still understand and respond to the manipulated text and therefore be vulnerable to the very attack the protection model was put in place to prevent. The tokenizer is tied to model architecture, meaning it is possible to predict whether or not a model is vulnerable to attack based on family. We also present a defensive strategy as an added layer of protection that can be implemented without having to retrain the defensive model.

摘要: 自然语言处理（NLP）模型用于与文本相关的任务，例如分类和生成。为了完成这些任务，首先将输入数据从人类可读的文本标记化为模型可以理解的格式，使其能够做出推断并理解上下文。可以实施文本分类模型来防范威胁，例如针对大型语言模型（LLM）的提示注入攻击、有毒输入和垃圾邮件等网络安全风险。在本文中，我们介绍TokenBreak：一种新型攻击，可以通过利用这些保护模型使用的标记化策略来绕过这些保护模型。这种攻击技术以某种方式操纵输入文本，使得某些模型给出不正确的分类。重要的是，最终目标（LLM或电子邮件收件人）仍然可以理解和响应被操纵的文本，因此很容易受到保护模型所要防止的攻击。标记器与模型架构相关联，这意味着可以基于家族预测模型是否容易受到攻击。我们还提出了一个防御策略，作为一个额外的保护层，可以在无需重新训练防御模型的情况下实现。



## **2. Adversarial Attack Classification and Robustness Testing for Large Language Models for Code**

代码大型语言模型的对抗性攻击分类和鲁棒性测试 cs.SE

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07942v1) [paper-pdf](http://arxiv.org/pdf/2506.07942v1)

**Authors**: Yang Liu, Armstrong Foundjem, Foutse Khomh, Heng Li

**Abstract**: Large Language Models (LLMs) have become vital tools in software development tasks such as code generation, completion, and analysis. As their integration into workflows deepens, ensuring robustness against vulnerabilities especially those triggered by diverse or adversarial inputs becomes increasingly important. Such vulnerabilities may lead to incorrect or insecure code generation when models encounter perturbed task descriptions, code, or comments. Prior research often overlooks the role of natural language in guiding code tasks. This study investigates how adversarial perturbations in natural language inputs including prompts, comments, and descriptions affect LLMs for Code (LLM4Code). It examines the effects of perturbations at the character, word, and sentence levels to identify the most impactful vulnerabilities. We analyzed multiple projects (e.g., ReCode, OpenAttack) and datasets (e.g., HumanEval, MBPP), establishing a taxonomy of adversarial attacks. The first dimension classifies the input type code, prompts, or comments while the second dimension focuses on granularity: character, word, or sentence-level changes. We adopted a mixed-methods approach, combining quantitative performance metrics with qualitative vulnerability analysis. LLM4Code models show varying robustness across perturbation types. Sentence-level attacks were least effective, suggesting models are resilient to broader contextual changes. In contrast, word-level perturbations posed serious challenges, exposing semantic vulnerabilities. Character-level effects varied, showing model sensitivity to subtle syntactic deviations.Our study offers a structured framework for testing LLM4Code robustness and emphasizes the critical role of natural language in adversarial evaluation. Improving model resilience to semantic-level disruptions is essential for secure and reliable code-generation systems.

摘要: 大型语言模型（LLM）已成为代码生成、完成和分析等软件开发任务的重要工具。随着它们与工作流程集成的加深，确保针对漏洞（尤其是由多样化或敌对输入触发的漏洞）的鲁棒性变得越来越重要。当模型遇到受干扰的任务描述、代码或评论时，此类漏洞可能会导致不正确或不安全的代码生成。之前的研究经常忽视自然语言在指导代码任务中的作用。本研究调查了自然语言输入（包括提示、评论和描述）中的对抗性扰动如何影响LLM for Code（LLM4Code）。它检查字符、单词和句子层面上的干扰的影响，以识别最有影响力的漏洞。我们分析了多个项目（例如，ReCode、OpenAttack）和数据集（例如，HumanEval，MBPP），建立了对抗性攻击的分类。第一个维度对输入类型代码、提示或注释进行分类，而第二个维度重点关注粒度：字符、单词或业务级别的更改。我们采用了混合方法，将定量性能指标与定性漏洞分析相结合。LLM 4Code模型显示出不同扰动类型的鲁棒性不同。句子级别的攻击效果最差，这表明模型能够适应更广泛的背景变化。相比之下，词级扰动带来了严重的挑战，暴露了语义漏洞。初级效应各不相同，表明模型对微妙的语法偏差的敏感性。我们的研究提供了一个结构化框架来测试LLM 4 Code稳健性，并强调自然语言在对抗性评估中的关键作用。提高模型对语义级中断的弹性对于安全可靠的代码生成系统至关重要。



## **3. SoK: Data Reconstruction Attacks Against Machine Learning Models: Definition, Metrics, and Benchmark**

针对机器学习模型的数据重建攻击：定义、验证和基准测试 cs.CR

To Appear in the 34th USENIX Security Symposium, August 13-15, 2025

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07888v1) [paper-pdf](http://arxiv.org/pdf/2506.07888v1)

**Authors**: Rui Wen, Yiyong Liu, Michael Backes, Yang Zhang

**Abstract**: Data reconstruction attacks, which aim to recover the training dataset of a target model with limited access, have gained increasing attention in recent years. However, there is currently no consensus on a formal definition of data reconstruction attacks or appropriate evaluation metrics for measuring their quality. This lack of rigorous definitions and universal metrics has hindered further advancement in this field. In this paper, we address this issue in the vision domain by proposing a unified attack taxonomy and formal definitions of data reconstruction attacks. We first propose a set of quantitative evaluation metrics that consider important criteria such as quantifiability, consistency, precision, and diversity. Additionally, we leverage large language models (LLMs) as a substitute for human judgment, enabling visual evaluation with an emphasis on high-quality reconstructions. Using our proposed taxonomy and metrics, we present a unified framework for systematically evaluating the strengths and limitations of existing attacks and establishing a benchmark for future research. Empirical results, primarily from a memorization perspective, not only validate the effectiveness of our metrics but also offer valuable insights for designing new attacks.

摘要: 数据重建攻击旨在恢复访问权限有限的目标模型的训练数据集，近年来受到越来越多的关注。然而，目前对于数据重建攻击的正式定义或衡量其质量的适当评估指标还没有达成共识。缺乏严格的定义和通用的指标阻碍了该领域的进一步发展。在本文中，我们通过提出统一的攻击分类法和数据重建攻击的形式定义来解决视觉领域的这个问题。我们首先提出了一套定量评估指标，考虑重要的标准，如可量化性，一致性，精度和多样性。此外，我们利用大型语言模型（LLM）作为人类判断的替代品，实现视觉评估，重点是高质量的重建。使用我们提出的分类和指标，我们提出了一个统一的框架，系统地评估现有攻击的优势和局限性，并建立一个基准，为未来的研究。实证结果，主要是从记忆的角度来看，不仅验证了我们的指标的有效性，但也提供了宝贵的见解，设计新的攻击。



## **4. Is poisoning a real threat to LLM alignment? Maybe more so than you think**

中毒是对LLM联盟的真正威胁吗？也许比你想象的还要多 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2406.12091v4) [paper-pdf](http://arxiv.org/pdf/2406.12091v4)

**Authors**: Pankayaraj Pathmanathan, Souradip Chakraborty, Xiangyu Liu, Yongyuan Liang, Furong Huang

**Abstract**: Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.

摘要: 带人类反馈的强化学习（RL HF）的最新进展显着影响了大型语言模型（LLM）的一致性。近端策略优化（PPO）等强化学习算法的敏感性导致了直接策略优化（DPO）的新工作，该算法在监督学习框架中处理RL HF。这些LLHF方法的实际使用增加，需要对其漏洞进行分析。在这项工作中，我们调查了DPO在不同场景下对中毒攻击的脆弱性，并比较了偏好中毒的有效性，这是首创。我们全面分析了DPO在不同类型攻击下的漏洞，即，后门和非后门攻击，以及跨各种语言模型的不同中毒方法，即，LLama 7B、Mistral 7B和Gemma 7B。我们发现，与基于PPO的方法不同，当涉及到后门攻击时，需要至少4%的数据被毒化才能引发有害行为，我们更简单地利用DPO的真正漏洞，因此我们可以用多达0.5%的数据毒化模型。我们进一步调查该漏洞背后的潜在原因，以及该漏洞如何转化为后门攻击与非后门攻击。



## **5. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

RSafe：激励积极推理，以建立强大且自适应的LLM保障措施 cs.AI

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07736v1) [paper-pdf](http://arxiv.org/pdf/2506.07736v1)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.

摘要: 尽管采取了刻意的安全调整措施，大型语言模型（LLM）仍然表现出漏洞，给用户和社会带来了重大风险。为了防范违反政策内容的风险，通过外部防护模型进行系统级审核（旨在监控LLM输入和输出并阻止潜在有害内容）已成为一种流行的缓解策略。训练警卫模型的现有方法严重依赖于大量的人类策划的数据集，并与分发外威胁作斗争，例如新出现的有害类别或越狱攻击。为了解决这些限制，我们提出RSafe，这是一种基于自适应推理的保护措施，它进行引导式安全推理，以在指定安全政策范围内提供强有力的保护。RSafe分两个阶段运行：1）引导推理，通过政策引导的分步推理来分析输入内容的安全风险，2）强化对齐，基于规则的RL优化其推理路径以与准确的安全预测保持一致。这种两阶段培训范式使RSafe能够内化安全原则，以概括针对不可见或对抗性安全违规场景的安全保护能力。在推理过程中，RSafe接受用户指定的安全政策，以提供针对特定安全要求的增强的保障措施。



## **6. Representation Bending for Large Language Model Safety**

大型语言模型安全性的弯曲表示 cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2504.01550v2) [paper-pdf](http://arxiv.org/pdf/2504.01550v2)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.

摘要: 大型语言模型（LLM）已经成为强大的工具，但其固有的安全风险-从有害内容生成到更广泛的社会危害-构成了重大挑战。这些风险可能会因最近的对抗性攻击、微调漏洞以及在高风险环境中越来越多地部署LLM而放大。现有的安全增强技术，例如通过人工反馈或对抗性训练进行微调，仍然很脆弱，因为它们解决了特定的威胁，并且通常无法概括看不见的攻击，或者需要手动系统级防御。本文介绍了RepBend，这是一种新的方法，从根本上破坏了LLM中有害行为的表示，提供了一种可扩展的解决方案来增强（潜在的固有）安全性。RepBend将激活引导的想法（用于在推理期间引导模型行为的简单载体算法）引入到基于损失的微调中。通过广泛的评估，RepBend实现了最先进的性能，优于Circuit Breaker、RMU和NPO等现有方法，在各种越狱基准测试中，攻击成功率降低了高达95%，模型可用性和通用功能的下降微乎其微。



## **7. EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

EVADE：电子商务应用程序中规避内容检测的多模式基准 cs.CL

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2505.17654v2) [paper-pdf](http://arxiv.org/pdf/2505.17654v2)

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.

摘要: 电子商务平台越来越依赖大型语言模型（LLM）和视觉语言模型（VLM）来检测非法或误导性产品内容。然而，这些模型仍然容易受到规避内容的影响：表面上遵守平台政策但秘密传达禁止声明的输入（文本或图像）。与导致明显失败的传统对抗性攻击不同，规避内容利用了模糊性和上下文，使其更难检测。现有的稳健性基准对这一要求严格的现实世界挑战几乎没有提供指导。我们引入EVADE，这是第一个由专家策划的中国多模式基准，专门用于评估电子商务中规避内容检测的基础模型。该数据集包含2，833个注释文本样本和13，961张图像，涵盖六个要求严格的产品类别，包括身材塑造、身高增长和保健品。两项补充任务评估不同的能力：Single-Violation（在短提示下探索细粒度推理）和All-in-One（通过将重叠的策略规则合并到统一指令中来测试长上下文推理）。值得注意的是，一体化设置显着缩小了部分匹配准确性和完全匹配准确性之间的性能差距，这表明更清晰的规则定义可以改善人类和模型判断之间的一致性。我们对26种主流LLM和VLM进行了基准测试，并观察到了巨大的性能差距：即使是最先进的模型也经常对规避样本进行错误分类。通过发布EVADE和强大的基线，我们为评估逃避内容检测提供了第一个严格的标准，暴露了当前多模式推理的根本局限性，并为电子商务中更安全、更透明的内容审核系统奠定了基础。该数据集可在https://huggingface.co/datasets/koenshen/EVADE-Bench上公开获取。



## **8. SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage**

SATA：通过简单辅助任务链接实现LLM越狱的范例 cs.CR

To appear at Findings of ACL 2025

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2412.15289v4) [paper-pdf](http://arxiv.org/pdf/2412.15289v4)

**Authors**: Xiaoning Dong, Wenbo Hu, Wei Xu, Tianxing He

**Abstract**: Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.

摘要: 大型语言模型（LLM）在各种任务中取得了重大进展，但它们的安全性一致仍然是一个主要问题。探索越狱提示可以暴露LLM的漏洞并指导保护它们的工作。现有的方法主要设计复杂的指令供LLM遵循，或者依赖于多次迭代，这可能会阻碍越狱的性能和效率。在这项工作中，我们提出了一种新颖的越狱范式--简单辅助任务链接（ATA），它可以有效地规避LLM保障措施并引发有害反应。具体来说，ATA首先屏蔽恶意查询中的有害关键词，以生成包含一个或多个[MASK]特殊令牌的相对良性的查询。然后，它采用简单的辅助任务，例如掩蔽语言模型任务或按位置查找元素任务来编码掩蔽关键词的语义。最后，ATA将辅助任务与屏蔽查询链接起来，共同执行越狱。大量实验表明，ATA实现了最先进的性能，并且大幅优于基线。具体来说，在AdvBench数据集上，通过屏蔽语言模型（MLM）辅助任务，ATA的总体攻击成功率（ASB）达到85%，有害评分（HS）达到4.57，通过按位置查找元素（ELP）辅助任务，ATA的总体攻击成功率（ASB）达到76%，HS达到4.43。



## **9. Evaluating LLMs Robustness in Less Resourced Languages with Proxy Models**

使用代理模型评估资源较少语言中的LLM鲁棒性 cs.CL

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07645v1) [paper-pdf](http://arxiv.org/pdf/2506.07645v1)

**Authors**: Maciej Chrabąszcz, Katarzyna Lorenc, Karolina Seweryn

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities across various natural language processing (NLP) tasks in recent years. However, their susceptibility to jailbreaks and perturbations necessitates additional evaluations. Many LLMs are multilingual, but safety-related training data contains mainly high-resource languages like English. This can leave them vulnerable to perturbations in low-resource languages such as Polish. We show how surprisingly strong attacks can be cheaply created by altering just a few characters and using a small proxy model for word importance calculation. We find that these character and word-level attacks drastically alter the predictions of different LLMs, suggesting a potential vulnerability that can be used to circumvent their internal safety mechanisms. We validate our attack construction methodology on Polish, a low-resource language, and find potential vulnerabilities of LLMs in this language. Additionally, we show how it can be extended to other languages. We release the created datasets and code for further research.

摘要: 近年来，大型语言模型（LLM）在各种自然语言处理（NLP）任务中表现出了令人印象深刻的能力。然而，它们对越狱和干扰的敏感性需要进行额外的评估。许多LLM都是多语言的，但与安全相关的培训数据主要包含英语等高资源语言。这可能会使他们容易受到波兰语等低资源语言的干扰。我们展示了通过仅改变几个字符并使用小型代理模型进行单词重要性计算，可以如何廉价地创建令人惊讶的强大攻击。我们发现这些字符和单词级攻击极大地改变了不同LLM的预测，这表明存在可用于规避其内部安全机制的潜在漏洞。我们在低资源语言波兰语上验证了我们的攻击构建方法，并发现该语言中LLM的潜在漏洞。此外，我们还展示了如何将其扩展到其他语言。我们发布创建的数据集和代码以供进一步研究。



## **10. MalGEN: A Generative Agent Framework for Modeling Malicious Software in Cybersecurity**

MalGEN：一个用于网络安全恶意软件建模的生成代理框架 cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07586v1) [paper-pdf](http://arxiv.org/pdf/2506.07586v1)

**Authors**: Bikash Saha, Sandeep Kumar Shukla

**Abstract**: The dual use nature of Large Language Models (LLMs) presents a growing challenge in cybersecurity. While LLM enhances automation and reasoning for defenders, they also introduce new risks, particularly their potential to be misused for generating evasive, AI crafted malware. Despite this emerging threat, the research community currently lacks controlled and extensible tools that can simulate such behavior for testing and defense preparation. We present MalGEN, a multi agent framework that simulates coordinated adversarial behavior to generate diverse, activity driven malware samples. The agents work collaboratively to emulate attacker workflows, including payload planning, capability selection, and evasion strategies, within a controlled environment built for ethical and defensive research. Using MalGEN, we synthesized ten novel malware samples and evaluated them against leading antivirus and behavioral detection engines. Several samples exhibited stealthy and evasive characteristics that bypassed current defenses, validating MalGEN's ability to model sophisticated and new threats. By transforming the threat of LLM misuse into an opportunity for proactive defense, MalGEN offers a valuable framework for evaluating and strengthening cybersecurity systems. The framework addresses data scarcity, enables rigorous testing, and supports the development of resilient and future ready detection strategies.

摘要: 大型语言模型（LLM）的双重用途性质给网络安全带来了越来越大的挑战。虽然LLM增强了防御者的自动化和推理，但它们也带来了新的风险，特别是它们被滥用来生成规避的、人工智能精心设计的恶意软件的可能性。尽管存在这种新出现的威胁，但研究界目前缺乏可以模拟此类行为以进行测试和防御准备的受控和可扩展的工具。我们介绍了Malgen，这是一个多代理框架，可以模拟协调的对抗行为，以生成多样化的、活动驱动的恶意软件样本。这些代理在为道德和防御研究而构建的受控环境中协作模拟攻击者的工作流程，包括有效负载规划、能力选择和规避策略。使用Malgen，我们合成了十个新型恶意软件样本，并针对领先的防病毒和行为检测引擎对其进行了评估。几个样本表现出绕过当前防御的隐身和规避特征，验证了Malgen建模复杂和新威胁的能力。通过将LLM滥用的威胁转化为积极防御的机会，Malgen为评估和加强网络安全系统提供了一个宝贵的框架。该框架解决了数据稀缺问题，实现了严格的测试，并支持开发有弹性且面向未来的检测策略。



## **11. When Style Breaks Safety: Defending Language Models Against Superficial Style Alignment**

当风格破坏安全性时：保护语言模型免受表面风格一致 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07452v1) [paper-pdf](http://arxiv.org/pdf/2506.07452v1)

**Authors**: Yuxin Xiao, Sana Tonekaboni, Walter Gerych, Vinith Suriyakumar, Marzyeh Ghassemi

**Abstract**: Large language models (LLMs) can be prompted with specific styles (e.g., formatting responses as lists), including in jailbreak queries. Although these style patterns are semantically unrelated to the malicious intents behind jailbreak queries, their safety impact remains unclear. In this work, we seek to understand whether style patterns compromise LLM safety, how superficial style alignment increases model vulnerability, and how best to mitigate these risks during alignment. We evaluate 32 LLMs across seven jailbreak benchmarks, and find that malicious queries with style patterns inflate the attack success rate (ASR) for nearly all models. Notably, ASR inflation correlates with both the length of style patterns and the relative attention an LLM exhibits on them. We then investigate superficial style alignment, and find that fine-tuning with specific styles makes LLMs more vulnerable to jailbreaks of those same styles. Finally, we propose SafeStyle, a defense strategy that incorporates a small amount of safety training data augmented to match the distribution of style patterns in the fine-tuning data. Across three LLMs and five fine-tuning style settings, SafeStyle consistently outperforms baselines in maintaining LLM safety.

摘要: 大型语言模型（LLM）可以用特定的风格提示（例如，将响应格式化为列表），包括在越狱查询中。尽管这些风格模式在语义上与越狱查询背后的恶意意图无关，但它们的安全影响仍不清楚。在这项工作中，我们试图了解风格模式是否会损害LLM的安全性、肤浅的风格对齐如何增加模型的脆弱性，以及如何在对齐期间最好地减轻这些风险。我们评估了7个越狱基准的32个LLM，发现具有风格模式的恶意查询会提高几乎所有模型的攻击成功率（ASB）。值得注意的是，ASB膨胀与风格模式的长度以及LLM对风格模式的相对关注度相关。然后，我们调查了表面的风格调整，发现对特定风格的微调使LLM更容易受到相同风格的越狱的影响。最后，我们提出了SafeStyle，这是一种防御策略，它结合了少量的安全训练数据，经过扩展以匹配微调数据中风格模式的分布。在三种LLM和五种微调风格设置中，SafeStyle在维护LLM安全性方面始终优于基线。



## **12. A Red Teaming Roadmap Towards System-Level Safety**

迈向系统级安全的红色团队路线图 cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.05376v2) [paper-pdf](http://arxiv.org/pdf/2506.05376v2)

**Authors**: Zifan Wang, Christina Q. Knight, Jeremy Kritz, Willow E. Primack, Julian Michael

**Abstract**: Large Language Model (LLM) safeguards, which implement request refusals, have become a widely adopted mitigation strategy against misuse. At the intersection of adversarial machine learning and AI safety, safeguard red teaming has effectively identified critical vulnerabilities in state-of-the-art refusal-trained LLMs. However, in our view the many conference submissions on LLM red teaming do not, in aggregate, prioritize the right research problems. First, testing against clear product safety specifications should take a higher priority than abstract social biases or ethical principles. Second, red teaming should prioritize realistic threat models that represent the expanding risk landscape and what real attackers might do. Finally, we contend that system-level safety is a necessary step to move red teaming research forward, as AI models present new threats as well as affordances for threat mitigation (e.g., detection and banning of malicious users) once placed in a deployment context. Adopting these priorities will be necessary in order for red teaming research to adequately address the slate of new threats that rapid AI advances present today and will present in the very near future.

摘要: 实现请求拒绝的大型语言模型（LLM）保障措施已成为一种广泛采用的针对滥用的缓解策略。在对抗性机器学习和人工智能安全的交叉点上，红色防护有效地识别了最先进的再培训LL中的关键漏洞。然而，我们认为，许多关于LLM红色团队的会议提交的文件总体上并没有优先考虑正确的研究问题。首先，针对明确的产品安全规范进行测试应该比抽象的社会偏见或道德原则更优先。其次，红色团队应该优先考虑现实的威胁模型，这些模型代表不断扩大的风险格局以及真正的攻击者可能会做什么。最后，我们认为系统级安全是推进红色团队研究的必要步骤，因为人工智能模型呈现了新的威胁以及威胁缓解的可供性（例如，检测和禁止恶意用户）一旦置于部署上下文中。为了让红色团队研究充分解决人工智能快速发展当今和不久的将来出现的一系列新威胁，采取这些优先事项是必要的。



## **13. Beyond Jailbreaks: Revealing Stealthier and Broader LLM Security Risks Stemming from Alignment Failures**

超越越狱：揭示更隐蔽、更广泛的LLM安全风险来自对齐失败 cs.CR

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07402v1) [paper-pdf](http://arxiv.org/pdf/2506.07402v1)

**Authors**: Yukai Zhou, Sibei Yang, Wenjie Wang

**Abstract**: Large language models (LLMs) are increasingly deployed in real-world applications, raising concerns about their security. While jailbreak attacks highlight failures under overtly harmful queries, they overlook a critical risk: incorrectly answering harmless-looking inputs can be dangerous and cause real-world harm (Implicit Harm). We systematically reformulate the LLM risk landscape through a structured quadrant perspective based on output factuality and input harmlessness, uncovering an overlooked high-risk region. To investigate this gap, we propose JailFlipBench, a benchmark aims to capture implicit harm, spanning single-modal, multimodal, and factual extension scenarios with diverse evaluation metrics. We further develop initial JailFlip attack methodologies and conduct comprehensive evaluations across multiple open-source and black-box LLMs, show that implicit harm present immediate and urgent real-world risks, calling for broader LLM safety assessments and alignment beyond conventional jailbreak paradigms.

摘要: 大型语言模型（LLM）越来越多地部署在现实世界的应用程序中，引发了对其安全性的担忧。虽然越狱攻击强调了明显有害的查询下的失败，但它们忽视了一个关键风险：错误地回答看似无害的输入可能是危险的，并会造成现实世界的伤害（隐性伤害）。我们基于输出真实性和输入无害性，通过结构化象限视角系统地重新制定LLM风险格局，发现被忽视的高风险区域。为了调查这一差距，我们提出了JailFlipBench，这是一个旨在捕捉隐性伤害的基准，跨越单模式、多模式和事实扩展场景，具有不同的评估指标。我们进一步开发初始的JailFlip攻击方法，并对多个开源和黑匣子LLM进行全面评估，表明隐性伤害带来了直接和紧迫的现实世界风险，呼吁进行更广泛的LLM安全评估和超越传统越狱范式的一致。



## **14. MrM: Black-Box Membership Inference Attacks against Multimodal RAG Systems**

MrM：针对多模式RAG系统的黑匣子成员推断攻击 cs.CV

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07399v1) [paper-pdf](http://arxiv.org/pdf/2506.07399v1)

**Authors**: Peiru Yang, Jinhua Yin, Haoran Zheng, Xueying Bai, Huili Wang, Yufei Sun, Xintian Li, Shangguang Wang, Yongfeng Huang, Tao Qi

**Abstract**: Multimodal retrieval-augmented generation (RAG) systems enhance large vision-language models by integrating cross-modal knowledge, enabling their increasing adoption across real-world multimodal tasks. These knowledge databases may contain sensitive information that requires privacy protection. However, multimodal RAG systems inherently grant external users indirect access to such data, making them potentially vulnerable to privacy attacks, particularly membership inference attacks (MIAs). % Existing MIA methods targeting RAG systems predominantly focus on the textual modality, while the visual modality remains relatively underexplored. To bridge this gap, we propose MrM, the first black-box MIA framework targeted at multimodal RAG systems. It utilizes a multi-object data perturbation framework constrained by counterfactual attacks, which can concurrently induce the RAG systems to retrieve the target data and generate information that leaks the membership information. Our method first employs an object-aware data perturbation method to constrain the perturbation to key semantics and ensure successful retrieval. Building on this, we design a counterfact-informed mask selection strategy to prioritize the most informative masked regions, aiming to eliminate the interference of model self-knowledge and amplify attack efficacy. Finally, we perform statistical membership inference by modeling query trials to extract features that reflect the reconstruction of masked semantics from response patterns. Experiments on two visual datasets and eight mainstream commercial visual-language models (e.g., GPT-4o, Gemini-2) demonstrate that MrM achieves consistently strong performance across both sample-level and set-level evaluations, and remains robust under adaptive defenses.

摘要: 多模式检索增强生成（RAG）系统通过集成跨模式知识来增强大型视觉语言模型，使其在现实世界的多模式任务中得到越来越多的采用。这些知识数据库可能包含需要隐私保护的敏感信息。然而，多模式RAG系统本质上允许外部用户间接访问此类数据，这使得他们可能容易受到隐私攻击，特别是成员资格推断攻击（MIA）。%针对RAG系统的现有MIA方法主要集中在文本形式上，而视觉形式仍然相对未充分研究。为了弥合这一差距，我们提出了MrM，这是第一个针对多模式RAG系统的黑匣子MIA框架。它利用受反事实攻击约束的多对象数据扰动框架，可以同时诱导RAG系统检索目标数据并生成泄露成员信息的信息。我们的方法首先采用对象感知的数据扰动方法来将扰动限制在关键语义上并确保成功检索。在此基础上，我们设计了一种基于反事实的掩蔽选择策略，以优先考虑信息最丰富的掩蔽区域，旨在消除模型自我知识的干扰并放大攻击功效。最后，我们进行统计隶属度推理建模查询试验提取功能，反映从响应模式的掩蔽语义重建。在两个视觉数据集和八个主流商业视觉语言模型（例如，GPT-4 o，Gemini-2）表明，MrM在样本级和集合级评估中都实现了一致的强大性能，并且在自适应防御下保持稳健。



## **15. Knowledge-to-Jailbreak: Investigating Knowledge-driven Jailbreaking Attacks for Large Language Models**

知识越狱：调查大型语言模型的知识驱动越狱攻击 cs.CL

Accepted by KDD 2025 research track

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2406.11682v2) [paper-pdf](http://arxiv.org/pdf/2406.11682v2)

**Authors**: Shangqing Tu, Zhuoran Pan, Wenxuan Wang, Zhexin Zhang, Yuliang Sun, Jifan Yu, Hongning Wang, Lei Hou, Juanzi Li

**Abstract**: Large language models (LLMs) have been increasingly applied to various domains, which triggers increasing concerns about LLMs' safety on specialized domains, e.g. medicine. Despite prior explorations on general jailbreaking attacks, there are two challenges for applying existing attacks on testing the domain-specific safety of LLMs: (1) Lack of professional knowledge-driven attacks, (2) Insufficient coverage of domain knowledge. To bridge this gap, we propose a new task, knowledge-to-jailbreak, which aims to generate jailbreaking attacks from domain knowledge, requiring both attack effectiveness and knowledge relevance. We collect a large-scale dataset with 12,974 knowledge-jailbreak pairs and fine-tune a large language model as jailbreak-generator, to produce domain knowledge-specific jailbreaks. Experiments on 13 domains and 8 target LLMs demonstrate the effectiveness of jailbreak-generator in generating jailbreaks that are both threatening to the target LLMs and relevant to the given knowledge. We also apply our method to an out-of-domain knowledge base, showing that jailbreak-generator can generate jailbreaks that are comparable in harmfulness to those crafted by human experts. Data and code are available at: https://github.com/THU-KEG/Knowledge-to-Jailbreak/.

摘要: 大型语言模型（LLM）越来越多地应用于各个领域，这引发了人们对LLM在医学等专业领域安全性的日益担忧。尽管之前对一般越狱攻击进行了探索，但应用现有攻击来测试LLM的特定领域安全性仍存在两个挑战：（1）缺乏专业知识驱动的攻击，（2）领域知识覆盖范围不足。为了弥合这一差距，我们提出了一项新任务，即知识越狱，旨在从领域知识生成越狱攻击，同时要求攻击有效性和知识相关性。我们收集了包含12，974个知识越狱对的大规模数据集，并微调大型语言模型作为越狱生成器，以生成特定于领域知识的越狱。在13个领域和8个目标LLM上的实验表明，越狱生成器在生成既对目标LLM有威胁又与给定知识相关的越狱时是有效的。我们还将我们的方法应用于域外知识库，表明越狱生成器可以生成与人类专家制作的危害性相当的越狱。数据和代码可在https://github.com/THU-KEG/Knowledge-to-Jailbreak/上获得。



## **16. Backdoor Attack on Vision Language Models with Stealthy Semantic Manipulation**

具有隐形语义操纵的视觉语言模型的后门攻击 cs.CV

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07214v1) [paper-pdf](http://arxiv.org/pdf/2506.07214v1)

**Authors**: Zhiyuan Zhong, Zhen Sun, Yepang Liu, Xinlei He, Guanhong Tao

**Abstract**: Vision Language Models (VLMs) have shown remarkable performance, but are also vulnerable to backdoor attacks whereby the adversary can manipulate the model's outputs through hidden triggers. Prior attacks primarily rely on single-modality triggers, leaving the crucial cross-modal fusion nature of VLMs largely unexplored. Unlike prior work, we identify a novel attack surface that leverages cross-modal semantic mismatches as implicit triggers. Based on this insight, we propose BadSem (Backdoor Attack with Semantic Manipulation), a data poisoning attack that injects stealthy backdoors by deliberately misaligning image-text pairs during training. To perform the attack, we construct SIMBad, a dataset tailored for semantic manipulation involving color and object attributes. Extensive experiments across four widely used VLMs show that BadSem achieves over 98% average ASR, generalizes well to out-of-distribution datasets, and can transfer across poisoning modalities. Our detailed analysis using attention visualization shows that backdoored models focus on semantically sensitive regions under mismatched conditions while maintaining normal behavior on clean inputs. To mitigate the attack, we try two defense strategies based on system prompt and supervised fine-tuning but find that both of them fail to mitigate the semantic backdoor. Our findings highlight the urgent need to address semantic vulnerabilities in VLMs for their safer deployment.

摘要: 视觉语言模型（VLM）已表现出出色的性能，但也容易受到后门攻击，对手可以通过隐藏触发器操纵模型的输出。之前的攻击主要依赖于单模式触发，使得VLM的关键跨模式融合本质基本上没有被探索。与之前的工作不同，我们发现了一种新颖的攻击表面，它利用跨模式语义不匹配作为隐式触发器。基于这一见解，我们提出了BadSem（具有语义操纵的后门攻击），这是一种数据中毒攻击，通过在训练期间故意错位图像-文本对来注入隐形后门。为了执行攻击，我们构建了SIMBad，这是一个专为涉及颜色和对象属性的语义操作而定制的数据集。对四种广泛使用的VLM进行的广泛实验表明，BadSem的平均ASC率超过98%，很好地推广到分布外数据集，并且可以跨中毒模式传输。我们使用注意力可视化进行的详细分析表明，后门模型在不匹配的条件下专注于语义敏感区域，同时在干净的输入上保持正常行为。为了减轻攻击，我们尝试了两种基于系统提示和监督微调的防御策略，但发现这两种策略都未能减轻语义后门。我们的研究结果凸显了迫切需要解决VLM中的语义漏洞，以更安全地部署它们。



## **17. Quality-Diversity Red-Teaming: Automated Generation of High-Quality and Diverse Attackers for Large Language Models**

质量多样性红色团队化：针对大型语言模型自动生成高质量且多样化的攻击者 cs.LG

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07121v1) [paper-pdf](http://arxiv.org/pdf/2506.07121v1)

**Authors**: Ren-Jian Wang, Ke Xue, Zeyu Qin, Ziniu Li, Sheng Tang, Hao-Tian Li, Shengcai Liu, Chao Qian

**Abstract**: Ensuring safety of large language models (LLMs) is important. Red teaming--a systematic approach to identifying adversarial prompts that elicit harmful responses from target LLMs--has emerged as a crucial safety evaluation method. Within this framework, the diversity of adversarial prompts is essential for comprehensive safety assessments. We find that previous approaches to red-teaming may suffer from two key limitations. First, they often pursue diversity through simplistic metrics like word frequency or sentence embedding similarity, which may not capture meaningful variation in attack strategies. Second, the common practice of training a single attacker model restricts coverage across potential attack styles and risk categories. This paper introduces Quality-Diversity Red-Teaming (QDRT), a new framework designed to address these limitations. QDRT achieves goal-driven diversity through behavior-conditioned training and implements a behavioral replay buffer in an open-ended manner. Additionally, it trains multiple specialized attackers capable of generating high-quality attacks across diverse styles and risk categories. Our empirical evaluation demonstrates that QDRT generates attacks that are both more diverse and more effective against a wide range of target LLMs, including GPT-2, Llama-3, Gemma-2, and Qwen2.5. This work advances the field of LLM safety by providing a systematic and effective approach to automated red-teaming, ultimately supporting the responsible deployment of LLMs.

摘要: 确保大型语言模型（LLM）的安全性非常重要。红色团队--一种识别引发目标LLM有害反应的对抗提示的系统方法--已成为一种至关重要的安全评估方法。在此框架下，对抗提示的多样性对于全面的安全评估至关重要。我们发现以前的红色团队方法可能存在两个关键限制。首先，他们经常通过词频或句子嵌入相似度等简单化指标来追求多样性，这可能无法捕捉攻击策略中有意义的变化。其次，训练单一攻击者模型的常见做法限制了潜在攻击风格和风险类别的覆盖范围。本文介绍了质量多样性红色团队（QDRT），这是一个旨在解决这些限制的新框架。QDRT通过行为条件训练实现目标驱动的多样性，并以开放式方式实现行为回放缓冲区。此外，它还培训了多个专业攻击者，能够在不同的风格和风险类别中生成高质量的攻击。我们的经验评估表明，QDRT生成的攻击更多样化，对各种目标LLM更有效，包括GPT-2，Llama-3，Gemma-2和Qwen2.5。这项工作通过提供一种系统有效的方法来自动化红队，最终支持LLM的负责任部署，从而推进了LLM安全领域。



## **18. HauntAttack: When Attack Follows Reasoning as a Shadow**

闹鬼攻击：当攻击像影子一样跟随推理时 cs.CR

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07031v1) [paper-pdf](http://arxiv.org/pdf/2506.07031v1)

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Lei Sha, Zhifang Sui

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing exceptional capabilities. However, the enhancement of reasoning abilities and the exposure of their internal reasoning processes introduce new safety vulnerabilities. One intriguing concern is: when reasoning is strongly entangled with harmfulness, what safety-reasoning trade-off do LRMs exhibit? To address this issue, we introduce HauntAttack, a novel and general-purpose black-box attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we treat reasoning questions as carriers and substitute one of their original conditions with a harmful instruction. This process creates a reasoning pathway in which the model is guided step by step toward generating unsafe outputs. Based on HauntAttack, we conduct comprehensive experiments on multiple LRMs. Our results reveal that even the most advanced LRMs exhibit significant safety vulnerabilities. Additionally, we perform a detailed analysis of different models, various types of harmful instructions, and model output patterns, providing valuable insights into the security of LRMs.

摘要: 新兴的大型推理模型（LRM）在数学和推理任务中始终表现出色，展现出卓越的能力。然而，推理能力的增强和内部推理过程的暴露引入了新的安全漏洞。一个有趣的担忧是：当推理与危害性强烈纠缠在一起时，LRM会表现出什么安全推理权衡？为了解决这个问题，我们引入了HauntAttack，这是一种新颖的通用黑匣子攻击框架，可以系统地将有害指令嵌入到推理问题中。具体来说，我们将推理问题视为载体，并用有害的指令取代其原始条件之一。该过程创建了一个推理路径，其中模型被逐步引导以生成不安全的输出。基于HauntAttack，我们对多个LRM进行了全面的实验。我们的结果表明，即使是最先进的LRM也表现出显着的安全漏洞。此外，我们还对不同模型、各种类型的有害指令和模型输出模式进行详细分析，为LRM的安全性提供有价值的见解。



## **19. sudo rm -rf agentic_security**

sudo rm -ref agentic_secure cs.CL

Accepted ACL 2025 Industry track

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2503.20279v3) [paper-pdf](http://arxiv.org/pdf/2503.20279v3)

**Authors**: Sejin Lee, Jian Kim, Haon Park, Ashkan Yousefpour, Sangyoon Yu, Min Song

**Abstract**: Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2Tox Offense), a novel attack framework that systematically bypasses refusal-trained safeguards in commercial computer-use agents, such as Claude for Computer Use. The core mechanism, Detox2Tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24.41% (with no refinement), and up to 41.33% (by its iterative refinement) in Claude for Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs

摘要: 大型语言模型（LLM）越来越多地被部署为计算机使用代理，在真实桌面或Web环境中自主执行任务。虽然这种演变极大地扩展了人类的实际用例，但也造成了严重的安全风险。我们提出了SUDO（基于屏幕的通用Detox 2 Tox Offense），这是一种新颖的攻击框架，可以系统地绕过商业计算机使用代理（例如Claude for Computer Use）中经过反思训练的保护措施。核心机制Detox 2Tox通过解毒将有害请求（代理最初拒绝的请求）转换为看似良性的请求，保护高级视觉语言模型（VLM）的详细指令，然后在执行前通过简化重新引入恶意内容。与传统的越狱不同，SUDO基于内置的拒绝反馈迭代改进其攻击，使其在对抗强大的政策过滤器时变得越来越有效。在涵盖50个现实世界任务和多个最先进的VLM的广泛测试中，SUDO的攻击成功率高达24.41%（无需改进），在Claude for Computer Use中高达41.33%（通过迭代改进）。通过揭示这些漏洞并展示它们在现实世界计算环境中被利用的轻松性，本文强调了对强大的、上下文感知的保护措施的迫切需求。警告：本文包括有害或冒犯性的模型输出



## **20. Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text**

对抗性重述：人工智能生成文本人性化的普遍攻击 cs.CL

**SubmitDate**: 2025-06-08    [abs](http://arxiv.org/abs/2506.07001v1) [paper-pdf](http://arxiv.org/pdf/2506.07001v1)

**Authors**: Yize Cheng, Vinu Sankar Sadasivan, Mehrdad Saberi, Shoumik Saha, Soheil Feizi

**Abstract**: The increasing capabilities of Large Language Models (LLMs) have raised concerns about their misuse in AI-generated plagiarism and social engineering. While various AI-generated text detectors have been proposed to mitigate these risks, many remain vulnerable to simple evasion techniques such as paraphrasing. However, recent detectors have shown greater robustness against such basic attacks. In this work, we introduce Adversarial Paraphrasing, a training-free attack framework that universally humanizes any AI-generated text to evade detection more effectively. Our approach leverages an off-the-shelf instruction-following LLM to paraphrase AI-generated content under the guidance of an AI text detector, producing adversarial examples that are specifically optimized to bypass detection. Extensive experiments show that our attack is both broadly effective and highly transferable across several detection systems. For instance, compared to simple paraphrasing attack--which, ironically, increases the true positive at 1% false positive (T@1%F) by 8.57% on RADAR and 15.03% on Fast-DetectGPT--adversarial paraphrasing, guided by OpenAI-RoBERTa-Large, reduces T@1%F by 64.49% on RADAR and a striking 98.96% on Fast-DetectGPT. Across a diverse set of detectors--including neural network-based, watermark-based, and zero-shot approaches--our attack achieves an average T@1%F reduction of 87.88% under the guidance of OpenAI-RoBERTa-Large. We also analyze the tradeoff between text quality and attack success to find that our method can significantly reduce detection rates, with mostly a slight degradation in text quality. Our adversarial setup highlights the need for more robust and resilient detection strategies in the light of increasingly sophisticated evasion techniques.

摘要: 大型语言模型（LLM）的能力不断增强，引发了人们对其在人工智能生成的抄袭和社会工程中滥用的担忧。虽然人们提出了各种人工智能生成的文本检测器来减轻这些风险，但许多文本检测器仍然容易受到简单规避技术（例如重述）的影响。然而，最近的检测器对此类基本攻击表现出更强的鲁棒性。在这项工作中，我们引入了对抗性重述，这是一种免训练的攻击框架，它普遍人性化任何人工智能生成的文本，以更有效地逃避检测。我们的方法利用现成的描述跟踪LLM在AI文本检测器的指导下解释AI生成的内容，生成经过专门优化以绕过检测的对抗性示例。大量实验表明，我们的攻击不仅广泛有效，而且在多个检测系统中高度可转移。例如，与简单的解释攻击相比--讽刺的是，它在RADART上将1%假阳性（T@1%F）的真阳性增加了8.57%，在Fast-DetectGPT上将15.03%--由OpenAI-RoBERTa-Large指导的对抗性解释在RADART上将T@1%F降低了64.49%，在Fast-DetectGPT上降低了98.96%。在一组不同的检测器中--包括基于神经网络、基于水印和零射击方法--在OpenAI-RoBERTa-Large的指导下，我们的攻击实现了T@1%F平均降低87.88%。我们还分析了文本质量和攻击成功之间的权衡，发现我们的方法可以显着降低检测率，但文本质量大多略有下降。鉴于日益复杂的规避技术，我们的对抗设置凸显了对更强大和更有弹性的检测策略的需求。



## **21. Robustifying Vision-Language Models via Dynamic Token Reweighting**

通过动态令牌重新加权来增强视觉语言模型 cs.CV

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2505.17132v2) [paper-pdf](http://arxiv.org/pdf/2505.17132v2)

**Authors**: Tanqiu Jiang, Jiacheng Liang, Rongyi Zhu, Jiawei Zhou, Fenglong Ma, Ting Wang

**Abstract**: Large vision-language models (VLMs) are highly vulnerable to jailbreak attacks that exploit visual-textual interactions to bypass safety guardrails. In this paper, we present DTR, a novel inference-time defense that mitigates multimodal jailbreak attacks through optimizing the model's key-value (KV) caches. Rather than relying on curated safety-specific data or costly image-to-text conversion, we introduce a new formulation of the safety-relevant distributional shift induced by the visual modality. This formulation enables DTR to dynamically adjust visual token weights, minimizing the impact of adversarial visual inputs while preserving the model's general capabilities and inference efficiency. Extensive evaluation across diverse VLMs and attack benchmarks demonstrates that \sys outperforms existing defenses in both attack robustness and benign task performance, marking the first successful application of KV cache optimization for safety enhancement in multimodal foundation models. (warning: this paper contains potentially harmful content generated by VLMs.)

摘要: 大型视觉语言模型（VLM）极易受到越狱攻击，这些攻击利用视觉与文本交互来绕过安全护栏。在本文中，我们提出了DTR，这是一种新型的推理时防御，通过优化模型的key-Value（KV）缓存来减轻多模式越狱攻击。我们不是依赖精心策划的安全特定数据或昂贵的图像到文本转换，而是引入了视觉模式引发的安全相关分布转变的新公式。该公式使DTR能够动态调整视觉令牌权重，最大限度地减少对抗视觉输入的影响，同时保留模型的一般能力和推理效率。对各种VLM和攻击基准的广泛评估表明，\sys在攻击稳健性和良性任务性能方面都优于现有防御，标志着在多模式基础模型中首次成功应用KV缓存优化来增强安全性。（警告：本文包含VLM生成的潜在有害内容。）



## **22. LLM-attacker: Enhancing Closed-loop Adversarial Scenario Generation for Autonomous Driving with Large Language Models**

LLM攻击者：使用大型语言模型增强自动驾驶的闭环对抗场景生成 cs.LG

Accepted as a regular paper at IEEE TITS 2025

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2501.15850v2) [paper-pdf](http://arxiv.org/pdf/2501.15850v2)

**Authors**: Yuewen Mei, Tong Nie, Jian Sun, Ye Tian

**Abstract**: Ensuring and improving the safety of autonomous driving systems (ADS) is crucial for the deployment of highly automated vehicles, especially in safety-critical events. To address the rarity issue, adversarial scenario generation methods are developed, in which behaviors of traffic participants are manipulated to induce safety-critical events. However, existing methods still face two limitations. First, identification of the adversarial participant directly impacts the effectiveness of the generation. However, the complexity of real-world scenarios, with numerous participants and diverse behaviors, makes identification challenging. Second, the potential of generated safety-critical scenarios to continuously improve ADS performance remains underexplored. To address these issues, we propose LLM-attacker: a closed-loop adversarial scenario generation framework leveraging large language models (LLMs). Specifically, multiple LLM agents are designed and coordinated to identify optimal attackers. Then, the trajectories of the attackers are optimized to generate adversarial scenarios. These scenarios are iteratively refined based on the performance of ADS, forming a feedback loop to improve ADS. Experimental results show that LLM-attacker can create more dangerous scenarios than other methods, and the ADS trained with it achieves a collision rate half that of training with normal scenarios. This indicates the ability of LLM-attacker to test and enhance the safety and robustness of ADS. Video demonstrations are provided at: https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view.

摘要: 确保和提高自动驾驶系统（ADS）的安全性对于部署高度自动化的车辆至关重要，特别是在安全关键事件中。为了解决稀有性问题，开发了对抗性场景生成方法，其中操纵交通参与者的行为以诱导安全关键事件。然而，现有的方法仍然面临两个限制。首先，对抗参与者的识别直接影响生成的有效性。然而，现实世界场景的复杂性，众多的参与者和不同的行为，使识别具有挑战性。其次，生成的安全关键场景的潜力，以不断提高ADS的性能仍然没有得到充分的探索。为了解决这些问题，我们提出了LLM攻击者：一个利用大型语言模型（LLM）的闭环对抗场景生成框架。具体而言，多个LLM代理被设计和协调以识别最佳攻击者。然后，对攻击者的轨迹进行优化以生成对抗场景。这些场景根据ADS的性能进行迭代细化，形成反馈循环来改进ADS。实验结果表明，LLM攻击者比其他方法可以创建更危险的场景，使用它训练的ADS的碰撞率是使用正常场景训练的一半。这表明LLM攻击者有能力测试和增强ADS的安全性和稳健性。视频演示请访问：https://drive.google.com/file/d/1Zv4V3iG7825oyiKbUwS2Y-rR0DQIE1ZA/view。



## **23. Refining Adaptive Zeroth-Order Optimization at Ease**

轻松细化自适应零阶优化 cs.LG

Published as a conference paper at ICML 2025

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2502.01014v2) [paper-pdf](http://arxiv.org/pdf/2502.01014v2)

**Authors**: Yao Shu, Qixin Zhang, Kun He, Zhongxiang Dai

**Abstract**: Recently, zeroth-order (ZO) optimization plays an essential role in scenarios where gradient information is inaccessible or unaffordable, such as black-box systems and resource-constrained environments. While existing adaptive methods such as ZO-AdaMM have shown promise, they are fundamentally limited by their underutilization of moment information during optimization, usually resulting in underperforming convergence. To overcome these limitations, this paper introduces Refined Adaptive Zeroth-Order Optimization (R-AdaZO). Specifically, we first show the untapped variance reduction effect of first moment estimate on ZO gradient estimation, which improves the accuracy and stability of ZO updates. We then refine the second moment estimate based on these variance-reduced gradient estimates to better capture the geometry of the optimization landscape, enabling a more effective scaling of ZO updates. We present rigorous theoretical analysis to show (a) the first analysis to the variance reduction of first moment estimate in ZO optimization, (b) the improved second moment estimates with a more accurate approximation of its variance-free ideal, (c) the first variance-aware convergence framework for adaptive ZO methods, which may be of independent interest, and (d) the faster convergence of R-AdaZO than existing baselines like ZO-AdaMM. Our extensive experiments, including synthetic problems, black-box adversarial attack, and memory-efficient fine-tuning of large language models (LLMs), further verify the superior convergence of R-AdaZO, indicating that R-AdaZO offers an improved solution for real-world ZO optimization challenges.

摘要: 最近，零阶（Zero）优化在梯度信息无法访问或负担不起的场景中发挥着至关重要的作用，例如黑匣子系统和资源受限的环境。虽然现有的自适应方法（例如ZO-AdaMM）已经表现出了希望，但它们从根本上受到优化过程中矩信息利用不足的限制，通常导致收敛性能不佳。为了克服这些限制，本文引入了细化自适应零阶优化（R-AdaZR）。具体来说，我们首先展示了一次矩估计对Zero梯度估计的未开发方差降低效果，这提高了Zero更新的准确性和稳定性。然后，我们根据这些方差降低的梯度估计来细化二次矩估计，以更好地捕捉优化景观的几何形状，从而能够更有效地扩展Zero更新。我们提出了严格的理论分析，以表明（a）对Zero优化中一阶矩估计的方差缩减的第一次分析，（b）改进的二阶矩估计，其更准确地逼近其无方差理想，（c）自适应Zero方法的第一个方差感知收敛框架，这可能是独立的兴趣，以及（d）R-AdaZR比ZO-AdaMM等现有基线更快的收敛。我们的广泛实验，包括合成问题、黑匣子对抗攻击和大型语言模型（LLM）的内存高效微调，进一步验证了R-Adazo的卓越收敛性，表明R-Adazo为现实世界的Zero优化挑战提供了改进的解决方案。



## **24. RED QUEEN: Safeguarding Large Language Models against Concealed Multi-Turn Jailbreaking**

红女王：保护大型语言模型免受隐藏的多回合越狱 cs.CR

Accepted in ACL 2025 Findings

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2409.17458v2) [paper-pdf](http://arxiv.org/pdf/2409.17458v2)

**Authors**: Yifan Jiang, Kriti Aggarwal, Tanmay Laud, Kashif Munir, Jay Pujara, Subhabrata Mukherjee

**Abstract**: The rapid progress of Large Language Models (LLMs) has opened up new opportunities across various domains and applications; yet it also presents challenges related to potential misuse. To mitigate such risks, red teaming has been employed as a proactive security measure to probe language models for harmful outputs via jailbreak attacks. However, current jailbreak attack approaches are single-turn with explicit malicious queries that do not fully capture the complexity of real-world interactions. In reality, users can engage in multi-turn interactions with LLM-based chat assistants, allowing them to conceal their true intentions in a more covert manner. To bridge this gap, we, first, propose a new jailbreak approach, RED QUEEN ATTACK. This method constructs a multi-turn scenario, concealing the malicious intent under the guise of preventing harm. We craft 40 scenarios that vary in turns and select 14 harmful categories to generate 56k multi-turn attack data points. We conduct comprehensive experiments on the RED QUEEN ATTACK with four representative LLM families of different sizes. Our experiments reveal that all LLMs are vulnerable to RED QUEEN ATTACK, reaching 87.62% attack success rate on GPT-4o and 75.4% on Llama3-70B. Further analysis reveals that larger models are more susceptible to the RED QUEEN ATTACK, with multi-turn structures and concealment strategies contributing to its success. To prioritize safety, we introduce a straightforward mitigation strategy called RED QUEEN GUARD, which aligns LLMs to effectively counter adversarial attacks. This approach reduces the attack success rate to below 1% while maintaining the model's performance across standard benchmarks. Full implementation and dataset are publicly accessible at https://github.com/kriti-hippo/red_queen.

摘要: 大型语言模型（LLM）的快速发展为各个领域和应用程序开辟了新的机会;但它也带来了与潜在滥用相关的挑战。为了减轻此类风险，红色团队已被用作一种主动安全措施，通过越狱攻击来探测语言模型的有害输出。然而，当前的越狱攻击方法是单轮的，带有显式恶意查询，无法完全捕捉现实世界交互的复杂性。事实上，用户可以与基于LLM的聊天助手进行多轮互动，使他们能够以更隐蔽的方式隐藏自己的真实意图。为了弥合这一差距，我们首先提出了一种新的越狱方法：红女王袭击。这种方法构建了一个多回合场景，以防止伤害为幌子隐藏恶意意图。我们精心设计了40个轮流变化的场景，并选择了14个有害类别来生成56，000个多回合攻击数据点。我们对《红女王袭击》进行了全面的实验，对四个不同规模的代表性LLM家族进行了实验。我们的实验表明，所有LLM都容易受到Red Queen Attack的攻击，GPT-4 o上的攻击成功率达到87.62%，Llama 3 - 70 B上的攻击成功率达到75.4%。进一步的分析表明，较大的模型更容易受到红皇后袭击的影响，多转弯结构和隐藏策略有助于其成功。为了优先考虑安全性，我们引入了一种名为RED QUEEN GUARD的简单缓解策略，该策略将LLM调整为有效对抗攻击。这种方法将攻击成功率降低至1%以下，同时在标准基准上保持模型的性能。完整的实现和数据集可在https://github.com/kriti-hippo/red_queen上公开访问。



## **25. from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors**

有毒的：通过对抗性隐喻越狱语言模型 cs.CL

arXiv admin note: substantial text overlap with arXiv:2412.12145

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2503.00038v3) [paper-pdf](http://arxiv.org/pdf/2503.00038v3)

**Authors**: Yu Yan, Sheng Sun, Zenghao Duan, Teli Liu, Min Liu, Zhiyi Yin, Jiangyu Lei, Qi Li

**Abstract**: Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs.

摘要: 当前的研究揭示了大型语言模型（LLM）通过越狱攻击生成有害内容的风险。然而，他们忽视了从头开始直接产生有害内容比诱导LLM将良性内容校准为有害形式更困难。在我们的研究中，我们引入了一种新颖的攻击框架，该框架利用AdVersArial meTAphoR（AVATAR）来诱导LLM校准用于越狱的恶意隐喻。具体来说，为了回答有害查询，AVATAR自适应地识别一组良性但逻辑相关的隐喻作为初始种子。然后，在这些隐喻的驱动下，目标LLM被诱导对隐喻内容进行推理和校准，从而通过直接输出有害响应或校准隐喻和专业有害内容之间的残留来越狱。实验结果表明，AVATAR可以有效且可转移的越狱LLM，并在多个高级LLM之间实现最先进的攻击成功率。



## **26. Short-length Adversarial Training Helps LLMs Defend Long-length Jailbreak Attacks: Theoretical and Empirical Evidence**

短期对抗训练帮助法学硕士防御长期越狱攻击：理论和经验证据 cs.LG

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2502.04204v2) [paper-pdf](http://arxiv.org/pdf/2502.04204v2)

**Authors**: Shaopeng Fu, Liang Ding, Jingfeng Zhang, Di Wang

**Abstract**: Jailbreak attacks against large language models (LLMs) aim to induce harmful behaviors in LLMs through carefully crafted adversarial prompts. To mitigate attacks, one way is to perform adversarial training (AT)-based alignment, i.e., training LLMs on some of the most adversarial prompts to help them learn how to behave safely under attacks. During AT, the length of adversarial prompts plays a critical role in the robustness of aligned LLMs. While long-length adversarial prompts during AT might lead to strong LLM robustness, their synthesis however is very resource-consuming, which may limit the application of LLM AT. This paper focuses on adversarial suffix jailbreak attacks and unveils that to defend against a jailbreak attack with an adversarial suffix of length $\Theta(M)$, it is enough to align LLMs on prompts with adversarial suffixes of length $\Theta(\sqrt{M})$. Theoretically, we analyze the adversarial in-context learning of linear transformers on linear regression tasks and prove a robust generalization bound for trained transformers. The bound depends on the term $\Theta(\sqrt{M_{\text{test}}}/M_{\text{train}})$, where $M_{\text{train}}$ and $M_{\text{test}}$ are the numbers of adversarially perturbed in-context samples during training and testing. Empirically, we conduct AT on popular open-source LLMs and evaluate their robustness against jailbreak attacks of different adversarial suffix lengths. Results confirm a positive correlation between the attack success rate and the ratio of the square root of the adversarial suffix length during jailbreaking to the length during AT. Our findings show that it is practical to defend against ``long-length'' jailbreak attacks via efficient ``short-length'' AT. The code is available at https://github.com/fshp971/adv-icl.

摘要: 针对大型语言模型（LLM）的越狱攻击旨在通过精心制作的对抗性提示来诱导LLM中的有害行为。为了减轻攻击，一种方法是执行基于对抗训练（AT）的对齐，即，在一些最具对抗性的提示上训练LLM，以帮助他们学习如何在攻击下安全行事。在AT期间，对抗性提示的长度在对齐的LLM的鲁棒性中起着关键作用。虽然AT期间的长时间对抗提示可能会导致LLM强大的鲁棒性，但它们的合成非常消耗资源，这可能会限制LLM AT的应用。本文重点研究对抗性后缀越狱攻击，并揭示了为了抵御具有长度为$\Theta（M）$的对抗性后缀的越狱攻击，只需将具有长度为$\Theta（\SQRT{M}）$的对抗性后缀的提示上的LLM对齐即可。从理论上讲，我们分析了线性变换器在线性回归任务上的对抗性反向上下文学习，并证明了训练变换器的鲁棒概括界限。该界限取决于项$\Theta（\SQRT{M_{\text{Test}/M_{\text{train}}）$，其中$M_{\text{train}}$和$M_{\text{Test}$是训练和测试期间上下文中敌对干扰样本的数量。从经验上讲，我们在流行的开源LLM上进行AT，并评估它们对不同对抗后缀长度的越狱攻击的鲁棒性。结果证实了攻击成功率与越狱过程中对抗后缀长度的平方根与AT过程中的长度之比呈正相关。我们的研究结果表明，它是切实可行的，以抵御“长”越狱攻击，通过有效的“短长度”的AT。该代码可在https://github.com/fshp971/adv-icl上获取。



## **27. GraphRAG under Fire**

GraphRAG受到攻击 cs.LG

13 pages

**SubmitDate**: 2025-06-07    [abs](http://arxiv.org/abs/2501.14050v3) [paper-pdf](http://arxiv.org/pdf/2501.14050v3)

**Authors**: Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi Zhu, Tanqiu Jiang, Neil Gong, Ting Wang

**Abstract**: GraphRAG advances retrieval-augmented generation (RAG) by structuring external knowledge as multi-scale knowledge graphs, enabling language models to integrate both broad context and granular details in their generation. While GraphRAG has demonstrated success across domains, its security implications remain largely unexplored. To bridge this gap, this work examines GraphRAG's vulnerability to poisoning attacks, uncovering an intriguing security paradox: existing RAG poisoning attacks are less effective under GraphRAG than conventional RAG, due to GraphRAG's graph-based indexing and retrieval; yet, the same features also create new attack surfaces. We present GragPoison, a novel attack that exploits shared relations in the underlying knowledge graph to craft poisoning text capable of compromising multiple queries simultaneously. GragPoison employs three key strategies: (i) relation injection to introduce false knowledge, (ii) relation enhancement to amplify poisoning influence, and (iii) narrative generation to embed malicious content within coherent text. Empirical evaluation across diverse datasets and models shows that GragPoison substantially outperforms existing attacks in terms of effectiveness (up to 98% success rate) and scalability (using less than 68% poisoning text) on multiple variations of GraphRAG. We also explore potential defensive measures and their limitations, identifying promising directions for future research.

摘要: GraphRAG通过将外部知识结构化为多尺度知识图，使语言模型能够在生成中集成广泛的上下文和粒度细节，从而推进了检索增强生成（RAG）。虽然GraphRAG在各个领域都取得了成功，但其安全影响在很大程度上仍未被探索。为了弥合这一差距，这项工作研究了GraphRAG对中毒攻击的脆弱性，揭示了一个有趣的安全悖论：由于GraphRAG的基于图形的索引和检索，现有的RAG中毒攻击在GraphRAG下不如传统RAG有效;然而，相同的功能也创建了新的攻击面。我们提出了GragPoison，这是一种新颖的攻击，它利用底层知识图中的共享关系来制作能够同时破坏多个查询的中毒文本。GragPoison采用三种关键策略：（i）关系注入以引入虚假知识，（ii）关系增强以放大中毒影响，以及（iii）叙事生成以将恶意内容嵌入连贯文本中。对不同数据集和模型的经验评估表明，GragPoison在对GraphRAG的多种变体的有效性（高达98%的成功率）和可扩展性（使用不到68%的中毒文本）方面大大优于现有的攻击。我们还探索潜在的防御措施及其局限性，为未来研究确定有希望的方向。



## **28. JailbreakLens: Visual Analysis of Jailbreak Attacks Against Large Language Models**

越狱镜头：针对大型语言模型的越狱攻击的视觉分析 cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2404.08793v2) [paper-pdf](http://arxiv.org/pdf/2404.08793v2)

**Authors**: Yingchaojie Feng, Zhizhang Chen, Zhining Kang, Sijia Wang, Haoyu Tian, Wei Zhang, Minfeng Zhu, Wei Chen

**Abstract**: The proliferation of large language models (LLMs) has underscored concerns regarding their security vulnerabilities, notably against jailbreak attacks, where adversaries design jailbreak prompts to circumvent safety mechanisms for potential misuse. Addressing these concerns necessitates a comprehensive analysis of jailbreak prompts to evaluate LLMs' defensive capabilities and identify potential weaknesses. However, the complexity of evaluating jailbreak performance and understanding prompt characteristics makes this analysis laborious. We collaborate with domain experts to characterize problems and propose an LLM-assisted framework to streamline the analysis process. It provides automatic jailbreak assessment to facilitate performance evaluation and support analysis of components and keywords in prompts. Based on the framework, we design JailbreakLens, a visual analysis system that enables users to explore the jailbreak performance against the target model, conduct multi-level analysis of prompt characteristics, and refine prompt instances to verify findings. Through a case study, technical evaluations, and expert interviews, we demonstrate our system's effectiveness in helping users evaluate model security and identify model weaknesses.

摘要: 大型语言模型（LLM）的激增凸显了对其安全漏洞的担忧，特别是针对越狱攻击，对手设计越狱提示规避潜在滥用的安全机制。解决这些问题需要对越狱提示进行全面分析，以评估LLM的防御能力并识别潜在的弱点。然而，评估越狱表现和了解提示特征的复杂性使得这项分析变得费力。我们与领域专家合作来描述问题并提出LLM辅助框架来简化分析过程。它提供自动越狱评估，以促进性能评估并支持对提示中的组件和关键词的分析。基于该框架，我们设计了JailbreakLens，这是一个视觉分析系统，使用户能够根据目标模型探索越狱表现，对提示特征进行多层次分析，并细化提示实例以验证结果。通过案例研究、技术评估和专家访谈，我们展示了我们的系统在帮助用户评估模型安全性和识别模型弱点方面的有效性。



## **29. A Systematic Review of Poisoning Attacks Against Large Language Models**

针对大型语言模型的毒害攻击的系统回顾 cs.CR

28 Pages including number

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06518v1) [paper-pdf](http://arxiv.org/pdf/2506.06518v1)

**Authors**: Neil Fendley, Edward W. Staley, Joshua Carney, William Redman, Marie Chau, Nathan Drenkow

**Abstract**: With the widespread availability of pretrained Large Language Models (LLMs) and their training datasets, concerns about the security risks associated with their usage has increased significantly. One of these security risks is the threat of LLM poisoning attacks where an attacker modifies some part of the LLM training process to cause the LLM to behave in a malicious way. As an emerging area of research, the current frameworks and terminology for LLM poisoning attacks are derived from earlier classification poisoning literature and are not fully equipped for generative LLM settings. We conduct a systematic review of published LLM poisoning attacks to clarify the security implications and address inconsistencies in terminology across the literature. We propose a comprehensive poisoning threat model applicable to categorize a wide range of LLM poisoning attacks. The poisoning threat model includes four poisoning attack specifications that define the logistics and manipulation strategies of an attack as well as six poisoning metrics used to measure key characteristics of an attack. Under our proposed framework, we organize our discussion of published LLM poisoning literature along four critical dimensions of LLM poisoning attacks: concept poisons, stealthy poisons, persistent poisons, and poisons for unique tasks, to better understand the current landscape of security risks.

摘要: 随着预训练的大型语言模型（LLM）及其训练数据集的广泛使用，对其使用相关安全风险的担忧显着增加。这些安全风险之一是LLM中毒攻击的威胁，攻击者修改LLM训练过程的某些部分，以导致LLM以恶意方式行为。作为一个新兴的研究领域，LLM中毒攻击的当前框架和术语源自早期的分类中毒文献，并且不完全适合生成性LLM设置。我们对已发表的LLM中毒攻击进行系统性审查，以澄清安全影响并解决文献中术语的不一致问题。我们提出了一个全面的中毒威胁模型，适用于对广泛的LLM中毒攻击进行分类。中毒威胁模型包括四个中毒攻击规范，定义攻击的后勤和操纵策略，以及用于衡量攻击关键特征的六个中毒指标。在我们提出的框架下，我们沿着LLM中毒攻击的四个关键维度组织对已发表的LLM中毒文献的讨论：概念毒药、隐形毒药、持久性毒药和用于独特任务的毒药，以更好地了解当前的安全风险格局。



## **30. PROVSYN: Synthesizing Provenance Graphs for Data Augmentation in Intrusion Detection Systems**

PROVSYS：合成源图以在入侵检测系统中进行数据增强 cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06226v1) [paper-pdf](http://arxiv.org/pdf/2506.06226v1)

**Authors**: Yi Huang, Wajih UI Hassan, Yao Guo, Xiangqun Chen, Ding Li

**Abstract**: Provenance graph analysis plays a vital role in intrusion detection, particularly against Advanced Persistent Threats (APTs), by exposing complex attack patterns. While recent systems combine graph neural networks (GNNs) with natural language processing (NLP) to capture structural and semantic features, their effectiveness is limited by class imbalance in real-world data. To address this, we introduce PROVSYN, an automated framework that synthesizes provenance graphs through a three-phase pipeline: (1) heterogeneous graph structure synthesis with structural-semantic modeling, (2) rule-based topological refinement, and (3) context-aware textual attribute synthesis using large language models (LLMs). PROVSYN includes a comprehensive evaluation framework that integrates structural, textual, temporal, and embedding-based metrics, along with a semantic validation mechanism to assess the correctness of generated attack patterns and system behaviors. To demonstrate practical utility, we use the synthetic graphs to augment training datasets for downstream APT detection models. Experimental results show that PROVSYN produces high-fidelity graphs and improves detection performance through effective data augmentation.

摘要: 源极图分析通过暴露复杂的攻击模式，在入侵检测中发挥着至关重要的作用，特别是针对高级持续性威胁（APT）。虽然最近的系统将图神经网络（GNN）与自然语言处理（NLP）相结合来捕获结构和语义特征，但其有效性受到现实世界数据中类别不平衡的限制。为了解决这个问题，我们引入了PROVYN，这是一个通过三阶段管道合成出处图的自动化框架：（1）具有结构语义建模的异类图结构合成，（2）基于规则的拓扑细化，（3）使用大型语言模型（LLM）的上下文感知文本属性合成。PROVYN包括一个综合的评估框架，该框架集成了结构性、文本性、时间性和基于嵌入的指标，以及一个用于评估生成的攻击模式和系统行为的正确性的语义验证机制。为了证明实际实用性，我们使用合成图来增强下游APT检测模型的训练数据集。实验结果表明，PROVYN生成高保真图，并通过有效的数据扩充提高检测性能。



## **31. Joint-GCG: Unified Gradient-Based Poisoning Attacks on Retrieval-Augmented Generation Systems**

Joint-GCG：对检索增强生成系统的统一基于对象的中毒攻击 cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06151v1) [paper-pdf](http://arxiv.org/pdf/2506.06151v1)

**Authors**: Haowei Wang, Rupeng Zhang, Junjie Wang, Mingyang Li, Yuekai Huang, Dandan Wang, Qing Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by retrieving relevant documents from external corpora before generating responses. This approach significantly expands LLM capabilities by leveraging vast, up-to-date external knowledge. However, this reliance on external knowledge makes RAG systems vulnerable to corpus poisoning attacks that manipulate generated outputs via poisoned document injection. Existing poisoning attack strategies typically treat the retrieval and generation stages as disjointed, limiting their effectiveness. We propose Joint-GCG, the first framework to unify gradient-based attacks across both retriever and generator models through three innovations: (1) Cross-Vocabulary Projection for aligning embedding spaces, (2) Gradient Tokenization Alignment for synchronizing token-level gradient signals, and (3) Adaptive Weighted Fusion for dynamically balancing attacking objectives. Evaluations demonstrate that Joint-GCG achieves at most 25% and an average of 5% higher attack success rate than previous methods across multiple retrievers and generators. While optimized under a white-box assumption, the generated poisons show unprecedented transferability to unseen models. Joint-GCG's innovative unification of gradient-based attacks across retrieval and generation stages fundamentally reshapes our understanding of vulnerabilities within RAG systems. Our code is available at https://github.com/NicerWang/Joint-GCG.

摘要: 检索增强生成（RAG）系统通过在生成响应之前从外部库检索相关文档来增强大型语言模型（LLM）。这种方法通过利用大量、最新的外部知识来显着扩展LLM能力。然而，这种对外部知识的依赖使得RAG系统容易受到通过有毒文档注入来操纵生成的输出的数据库中毒攻击。现有的中毒攻击策略通常将检索和生成阶段视为脱节的，从而限制了它们的有效性。我们提出了Joint-GCG，这是第一个通过三项创新统一检索器和生成器模型中基于梯度的攻击的框架：（1）用于对齐嵌入空间的跨词汇投影，（2）用于同步标记级梯度信号的梯度令牌化对齐，以及（3）用于动态平衡攻击目标的自适应加权融合。评估表明，Joint-GCG在多个检索器和生成器上的攻击成功率比以前的方法最多高25%，平均高5%。虽然在白盒假设下进行了优化，但生成的毒药显示出前所未有的可转移性，以转移到未见过的模型。Joint-GCG创新地统一了检索和生成阶段的基于梯度的攻击，从根本上重塑了我们对RAG系统中漏洞的理解。我们的代码可在https://github.com/NicerWang/Joint-GCG上获取。



## **32. The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text**

金丝雀的回声：审计LLM生成的合成文本的隐私风险 cs.CL

42nd International Conference on Machine Learning (ICML 2025)

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2502.14921v2) [paper-pdf](http://arxiv.org/pdf/2502.14921v2)

**Authors**: Matthieu Meeus, Lukas Wutschitz, Santiago Zanella-Béguelin, Shruti Tople, Reza Shokri

**Abstract**: How much information about training samples can be leaked through synthetic data generated by Large Language Models (LLMs)? Overlooking the subtleties of information flow in synthetic data generation pipelines can lead to a false sense of privacy. In this paper, we assume an adversary has access to some synthetic data generated by a LLM. We design membership inference attacks (MIAs) that target the training data used to fine-tune the LLM that is then used to synthesize data. The significant performance of our MIA shows that synthetic data leak information about the training data. Further, we find that canaries crafted for model-based MIAs are sub-optimal for privacy auditing when only synthetic data is released. Such out-of-distribution canaries have limited influence on the model's output when prompted to generate useful, in-distribution synthetic data, which drastically reduces their effectiveness. To tackle this problem, we leverage the mechanics of auto-regressive models to design canaries with an in-distribution prefix and a high-perplexity suffix that leave detectable traces in synthetic data. This enhances the power of data-based MIAs and provides a better assessment of the privacy risks of releasing synthetic data generated by LLMs.

摘要: 大型语言模型（LLM）生成的合成数据会泄露多少有关训练样本的信息？忽视合成数据生成管道中信息流的微妙之处可能会导致错误的隐私感。在本文中，我们假设对手可以访问LLM生成的一些合成数据。我们设计了成员资格推理攻击（MIA），针对用于微调LLM的训练数据，然后用于合成数据。我们的MIA的显着性能表明合成数据泄露了有关训练数据的信息。此外，我们发现，当仅发布合成数据时，为基于模型的MIA制作的金丝雀对于隐私审计来说并不是最佳的。当提示生成有用的、分布内的合成数据时，这种不分布的金丝雀对模型的输出影响有限，从而大大降低了其有效性。为了解决这个问题，我们利用自回归模型的机制来设计具有内分布后缀和高困惑性后缀的金丝雀，从而在合成数据中留下可检测的痕迹。这增强了基于数据的MIA的能力，并对发布LLM生成的合成数据的隐私风险提供了更好的评估。



## **33. Simple Yet Effective: Extracting Private Data Across Clients in Federated Fine-Tuning of Large Language Models**

简单但有效：在大型语言模型的联邦微调中提取客户端的私人数据 cs.CL

10 pages, 4 figures

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.06060v1) [paper-pdf](http://arxiv.org/pdf/2506.06060v1)

**Authors**: Yingqi Hu, Zhuo Zhang, Jingyuan Zhang, Lizhen Qu, Zenglin Xu

**Abstract**: Federated fine-tuning of large language models (FedLLMs) presents a promising approach for achieving strong model performance while preserving data privacy in sensitive domains. However, the inherent memorization ability of LLMs makes them vulnerable to training data extraction attacks. To investigate this risk, we introduce simple yet effective extraction attack algorithms specifically designed for FedLLMs. In contrast to prior "verbatim" extraction attacks, which assume access to fragments from all training data, our approach operates under a more realistic threat model, where the attacker only has access to a single client's data and aims to extract previously unseen personally identifiable information (PII) from other clients. This requires leveraging contextual prefixes held by the attacker to generalize across clients. To evaluate the effectiveness of our approaches, we propose two rigorous metrics-coverage rate and efficiency-and extend a real-world legal dataset with PII annotations aligned with CPIS, GDPR, and CCPA standards, achieving 89.9% human-verified precision. Experimental results show that our method can extract up to 56.57% of victim-exclusive PII, with "Address," "Birthday," and "Name" being the most vulnerable categories. Our findings underscore the pressing need for robust defense strategies and contribute a new benchmark and evaluation framework for future research in privacy-preserving federated learning.

摘要: 大型语言模型（FedLLM）的联合微调提供了一种有前途的方法，可以在实现强大的模型性能的同时保护敏感领域的数据隐私。然而，LLM固有的记忆能力使它们容易受到训练数据提取攻击。为了调查这种风险，我们引入了专门为FedLLM设计的简单而有效的提取攻击算法。与之前的“逐字”提取攻击（假设可以访问所有训练数据中的片段）相反，我们的方法在更现实的威胁模型下运行，其中攻击者只能访问单个客户端的数据，并旨在从其他客户端提取之前未见过的个人可识别信息（PRI）。这需要利用攻击者持有的上下文前置来跨客户端进行概括。为了评估我们方法的有效性，我们提出了两个严格的指标--覆盖率和效率--并使用符合CPIS、GDPR和CCPA标准的PRI注释扩展现实世界的法律数据集，实现89.9%的人类验证精度。实验结果表明，我们的方法可以提取高达56.57%的受害者专属PRI，其中“地址”、“生日”和“姓名”是最脆弱的类别。我们的研究结果强调了对稳健防御策略的迫切需求，并为未来保护隐私的联邦学习研究提供了新的基准和评估框架。



## **34. FDLLM: A Dedicated Detector for Black-Box LLMs Fingerprinting**

FDLLM：黑匣子LLM指纹识别的专用检测器 cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2501.16029v2) [paper-pdf](http://arxiv.org/pdf/2501.16029v2)

**Authors**: Zhiyuan Fu, Junfan Chen, Lan Zhang, Ting Yang, Jun Niu, Hongyu Sun, Ruidong Li, Peng Liu, Yuqing Zhang

**Abstract**: Large Language Models (LLMs) are rapidly transforming the landscape of digital content creation. However, the prevalent black-box Application Programming Interface (API) access to many LLMs introduces significant challenges in accountability, governance, and security. LLM fingerprinting, which aims to identify the source model by analyzing statistical and stylistic features of generated text, offers a potential solution. Current progress in this area is hindered by a lack of dedicated datasets and the need for efficient, practical methods that are robust against adversarial manipulations. To address these challenges, we introduce FD-Dataset, a comprehensive bilingual fingerprinting benchmark comprising 90,000 text samples from 20 famous proprietary and open-source LLMs. Furthermore, we present FDLLM, a novel fingerprinting method that leverages parameter-efficient Low-Rank Adaptation (LoRA) to fine-tune a foundation model. This approach enables LoRA to extract deep, persistent features that characterize each source LLM. Through our analysis, we find that LoRA adaptation promotes the aggregation of outputs from the same LLM in representation space while enhancing the separation between different LLMs. This mechanism explains why LoRA proves particularly effective for LLM fingerprinting. Extensive empirical evaluations on FD-Dataset demonstrate FDLLM's superiority, achieving a Macro F1 score 22.1% higher than the strongest baseline. FDLLM also exhibits strong generalization to newly released models, achieving an average accuracy of 95% on unseen models. Notably, FDLLM remains consistently robust under various adversarial attacks, including polishing, translation, and synonym substitution. Experimental results show that FDLLM reduces the average attack success rate from 49.2% (LM-D) to 23.9%.

摘要: 大型语言模型（LLM）正在迅速改变数字内容创作的格局。然而，对许多LLM的普遍黑盒应用程序编程接口（API）访问在问责制，治理和安全性方面带来了重大挑战。LLM指纹，旨在通过分析生成的文本的统计和风格特征来识别源模型，提供了一个潜在的解决方案。目前这一领域的进展受到缺乏专用数据集以及需要有效实用的方法来对抗性操纵的阻碍。为了应对这些挑战，我们引入了FD-Dataset，这是一个全面的双语指纹识别基准，包含来自20个著名专有和开源LLM的90，000个文本样本。此外，我们还介绍了FDLLM，这是一种新型指纹识别方法，它利用参数高效的低等级自适应（LoRA）来微调基础模型。这种方法使LoRA能够提取描述每个源LLM特征的深层、持久的特征。通过我们的分析，我们发现LoRA适应促进了来自同一LLM的输出在表示空间中的聚合，同时增强了不同LLM之间的分离性。这种机制解释了为什么LoRA被证明对LLM指纹识别特别有效。对FD-Dataset的广泛实证评估证明了FDLLM的优越性，Macro F1评分比最强基线高出22.1%。FDLLM还对新发布的模型表现出很强的概括性，在未见过的模型上实现了95%的平均准确率。值得注意的是，FDLLM在各种对抗攻击（包括抛光、翻译和同义词替换）下仍然保持稳定。实验结果表明，FDLLM将平均攻击成功率从49.2%（LM-D）降低到23.9%。



## **35. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

MCA-Bench：评估CAPTCHA针对基于VLM的攻击的稳健性的多模式基准 cs.CV

31 pages, 8 figures

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.05982v1) [paper-pdf](http://arxiv.org/pdf/2506.05982v1)

**Authors**: Zonglin Wu, Yule Xue, Xin Wei, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.

摘要: 随着自动攻击技术的迅速发展，验证码仍然是针对恶意机器人的重要防御机制。然而，现有的CAPTCHA方案涵盖了多种形式--从静态扭曲文本和模糊图像到交互式点击、滑动谜题和基于逻辑的问题--但社区仍然缺乏统一的、大规模的、多模式基准来严格评估其安全稳健性。为了解决这一差距，我们引入了MCA-Bench，这是一个全面且可重复的基准测试套件，可将异类CAPTCHA类型集成到单个评估协议中。利用共享的视觉语言模型主干，我们为每个CAPTCHA类别微调专门的破解剂，实现一致的跨模式评估。大量实验表明，MCA-Bench有效地绘制了现代CAPTCHA设计在不同攻击环境下的脆弱性谱，并且至关重要地提供了挑战复杂性、交互深度和模型可解性如何相互关联的首次定量分析。基于这些发现，我们提出了三项可操作的设计原则，并确定了关键的开放挑战，为系统性CAPTCHA强化、公平的基准测试和更广泛的社区合作奠定了基础。数据集和代码可在线获取。



## **36. UDora: A Unified Red Teaming Framework against LLM Agents by Dynamically Hijacking Their Own Reasoning**

UPora：通过动态劫持LLM代理自己的推理来对抗他们的统一红色团队框架 cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2503.01908v2) [paper-pdf](http://arxiv.org/pdf/2503.01908v2)

**Authors**: Jiawei Zhang, Shuang Yang, Bo Li

**Abstract**: Large Language Model (LLM) agents equipped with external tools have become increasingly powerful for complex tasks such as web shopping, automated email replies, and financial trading. However, these advancements amplify the risks of adversarial attacks, especially when agents can access sensitive external functionalities. Nevertheless, manipulating LLM agents into performing targeted malicious actions or invoking specific tools remains challenging, as these agents extensively reason or plan before executing final actions. In this work, we present UDora, a unified red teaming framework designed for LLM agents that dynamically hijacks the agent's reasoning processes to compel malicious behavior. Specifically, UDora first generates the model's reasoning trace for the given task, then automatically identifies optimal points within this trace to insert targeted perturbations. The resulting perturbed reasoning is then used as a surrogate response for optimization. By iteratively applying this process, the LLM agent will then be induced to undertake designated malicious actions or to invoke specific malicious tools. Our approach demonstrates superior effectiveness compared to existing methods across three LLM agent datasets. The code is available at https://github.com/AI-secure/UDora.

摘要: 配备外部工具的大型语言模型（LLM）代理在网络购物、自动电子邮件回复和金融交易等复杂任务中变得越来越强大。然而，这些进步放大了对抗攻击的风险，特别是当代理可以访问敏感的外部功能时。然而，操纵LLM代理执行有针对性的恶意操作或调用特定工具仍然具有挑战性，因为这些代理在执行最终操作之前进行了广泛的推理或计划。在这项工作中，我们介绍了UPora，这是一个为LLM代理设计的统一红色团队框架，它动态劫持代理的推理过程以迫使恶意行为。具体来说，UPora首先为给定任务生成模型的推理轨迹，然后自动识别此轨迹内的最佳点以插入有针对性的扰动。然后将产生的扰动推理用作优化的替代响应。通过迭代应用此过程，LLM代理将被诱导采取指定的恶意操作或调用特定的恶意工具。与三个LLM代理数据集的现有方法相比，我们的方法表现出了卓越的有效性。该代码可在https://github.com/AI-secure/UDora上获取。



## **37. PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning**

PoisonBench：评估大型语言模型对数据中毒的脆弱性 cs.CR

Accepted at ICML 2025. Tingchen Fu and Fazl Barez are core research  contributors

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2410.08811v2) [paper-pdf](http://arxiv.org/pdf/2410.08811v2)

**Authors**: Tingchen Fu, Mrinank Sharma, Philip Torr, Shay B. Cohen, David Krueger, Fazl Barez

**Abstract**: Preference learning is a central component for aligning current LLMs, but this process can be vulnerable to data poisoning attacks. To address this concern, we introduce PoisonBench, a benchmark for evaluating large language models' susceptibility to data poisoning during preference learning. Data poisoning attacks can manipulate large language model responses to include hidden malicious content or biases, potentially causing the model to generate harmful or unintended outputs while appearing to function normally. We deploy two distinct attack types across eight realistic scenarios, assessing 21 widely-used models. Our findings reveal concerning trends: (1) Scaling up parameter size does not inherently enhance resilience against poisoning attacks; (2) There exists a log-linear relationship between the effects of the attack and the data poison ratio; (3) The effect of data poisoning can generalize to extrapolated triggers that are not included in the poisoned data. These results expose weaknesses in current preference learning techniques, highlighting the urgent need for more robust defenses against malicious models and data manipulation.

摘要: 偏好学习是对齐当前LLM的核心组件，但此过程可能容易受到数据中毒攻击。为了解决这个问题，我们引入了PoisonBench，这是一个用于评估大型语言模型在偏好学习期间对数据中毒敏感性的基准。数据中毒攻击可以操纵大型语言模型响应，以包含隐藏的恶意内容或偏见，从而可能导致模型在看起来正常运行的同时生成有害或非预期的输出。我们在八种现实场景中部署了两种不同的攻击类型，评估了21种广泛使用的模型。我们的研究结果揭示了相关趋势：（1）扩大参数大小并不能本质上增强针对中毒攻击的弹性;（2）攻击的影响和数据中毒率之间存在log线性关系;（3）数据中毒的影响可以概括为不包括在中毒数据中的外推触发器。这些结果暴露了当前偏好学习技术的弱点，凸显了对恶意模型和数据操纵的更强大防御的迫切需要。



## **38. Who Can Withstand Chat-Audio Attacks? An Evaluation Benchmark for Large Audio-Language Models**

谁能抵御聊天音频攻击？大型音频语言模型的评估基准 cs.SD

Accepted by ACL 2025 Findings

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2411.14842v2) [paper-pdf](http://arxiv.org/pdf/2411.14842v2)

**Authors**: Wanqi Yang, Yanda Li, Meng Fang, Yunchao Wei, Ling Chen

**Abstract**: Adversarial audio attacks pose a significant threat to the growing use of large audio-language models (LALMs) in voice-based human-machine interactions. While existing research focused on model-specific adversarial methods, real-world applications demand a more generalizable and universal approach to audio adversarial attacks. In this paper, we introduce the Chat-Audio Attacks (CAA) benchmark including four distinct types of audio attacks, which aims to explore the vulnerabilities of LALMs to these audio attacks in conversational scenarios. To evaluate the robustness of LALMs, we propose three evaluation strategies: Standard Evaluation, utilizing traditional metrics to quantify model performance under attacks; GPT-4o-Based Evaluation, which simulates real-world conversational complexities; and Human Evaluation, offering insights into user perception and trust. We evaluate six state-of-the-art LALMs with voice interaction capabilities, including Gemini-1.5-Pro, GPT-4o, and others, using three distinct evaluation methods on the CAA benchmark. Our comprehensive analysis reveals the impact of four types of audio attacks on the performance of these models, demonstrating that GPT-4o exhibits the highest level of resilience. Our data can be accessed via the following link: \href{https://github.com/crystraldo/CAA}{CAA}.

摘要: 对抗性音频攻击对基于语音的人机交互中越来越多地使用大型音频语言模型（LALM）构成了重大威胁。虽然现有的研究主要集中在模型特定的对抗方法上，但现实世界的应用需要一种更通用和通用的方法来进行音频对抗攻击。在本文中，我们介绍了聊天音频攻击（CAA）基准，包括四种不同类型的音频攻击，其目的是探索在会话场景中的LALM对这些音频攻击的脆弱性。为了评估LALM的鲁棒性，我们提出了三种评估策略：标准评估，利用传统指标来量化模型在攻击下的性能;基于GPT-4o的评估，模拟现实世界的会话复杂性;以及人类评估，提供对用户感知和信任的见解。我们使用CAA基准上的三种不同评估方法评估了六种具有语音交互功能的最先进LALM，包括Gemini-1.5-Pro、GPT-4o等。我们的全面分析揭示了四种类型的音频攻击对这些模型性能的影响，证明GPT-4o表现出最高水平的弹性。我们的数据可以通过以下链接访问：\href {https：//github.com/crystraldo/CAA}{CAA}。



## **39. TracLLM: A Generic Framework for Attributing Long Context LLMs**

TracLLM：用于赋予长上下文LLM属性的通用框架 cs.CR

To appear in USENIX Security Symposium 2025. The code and data are  at: https://github.com/Wang-Yanting/TracLLM

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.04202v2) [paper-pdf](http://arxiv.org/pdf/2506.04202v2)

**Authors**: Yanting Wang, Wei Zou, Runpeng Geng, Jinyuan Jia

**Abstract**: Long context large language models (LLMs) are deployed in many real-world applications such as RAG, agent, and broad LLM-integrated applications. Given an instruction and a long context (e.g., documents, PDF files, webpages), a long context LLM can generate an output grounded in the provided context, aiming to provide more accurate, up-to-date, and verifiable outputs while reducing hallucinations and unsupported claims. This raises a research question: how to pinpoint the texts (e.g., sentences, passages, or paragraphs) in the context that contribute most to or are responsible for the generated output by an LLM? This process, which we call context traceback, has various real-world applications, such as 1) debugging LLM-based systems, 2) conducting post-attack forensic analysis for attacks (e.g., prompt injection attack, knowledge corruption attacks) to an LLM, and 3) highlighting knowledge sources to enhance the trust of users towards outputs generated by LLMs. When applied to context traceback for long context LLMs, existing feature attribution methods such as Shapley have sub-optimal performance and/or incur a large computational cost. In this work, we develop TracLLM, the first generic context traceback framework tailored to long context LLMs. Our framework can improve the effectiveness and efficiency of existing feature attribution methods. To improve the efficiency, we develop an informed search based algorithm in TracLLM. We also develop contribution score ensemble/denoising techniques to improve the accuracy of TracLLM. Our evaluation results show TracLLM can effectively identify texts in a long context that lead to the output of an LLM. Our code and data are at: https://github.com/Wang-Yanting/TracLLM.

摘要: 长上下文大型语言模型（LLM）部署在许多现实世界的应用程序中，例如RAG、代理和广泛的LLM集成应用程序。给定指令和长上下文（例如，文档、PDF文件、网页）、长上下文LLM可以基于所提供的上下文生成输出，旨在提供更准确、最新和可验证的输出，同时减少幻觉和不支持的主张。这提出了一个研究问题：如何确定文本（例如，句子、段落或段落）在对LLM生成的输出做出最大贡献或负责的上下文中？这个过程（我们称之为上下文追溯）具有各种现实世界的应用程序，例如1）调试基于LLM的系统，2）对攻击进行攻击后取证分析（例如，即时注入攻击、知识腐败攻击）对LLM，以及3）强调知识源以增强用户对LLM生成的输出的信任。当应用于长上下文LLM的上下文追溯时，现有的特征属性方法（例如Shapley）的性能不佳和/或会产生很大的计算成本。在这项工作中，我们开发了TracLLM，这是第一个针对长上下文LLM量身定制的通用上下文追溯框架。我们的框架可以提高现有特征归因方法的有效性和效率。为了提高效率，我们在TracLLM中开发了一种基于明智搜索的算法。我们还开发贡献分数集成/去噪技术来提高TracLLM的准确性。我们的评估结果表明TracLLM可以有效地识别导致LLM输出的长期上下文中的文本。我们的代码和数据位于：https://github.com/Wang-Yanting/TracLLM。



## **40. FedShield-LLM: A Secure and Scalable Federated Fine-Tuned Large Language Model**

FedShield-LLM：一个安全且可扩展的联邦微调大型语言模型 cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.05640v1) [paper-pdf](http://arxiv.org/pdf/2506.05640v1)

**Authors**: Md Jueal Mia, M. Hadi Amini

**Abstract**: Federated Learning (FL) offers a decentralized framework for training and fine-tuning Large Language Models (LLMs) by leveraging computational resources across organizations while keeping sensitive data on local devices. It addresses privacy and security concerns while navigating challenges associated with the substantial computational demands of LLMs, which can be prohibitive for small and medium-sized organizations. FL supports the development of task-specific LLMs for cross-silo applications through fine-tuning but remains vulnerable to inference attacks, such as membership inference and gradient inversion, which threaten data privacy. Prior studies have utilized Differential Privacy (DP) in LLM fine-tuning, which, despite being effective at preserving privacy, can degrade model performance. To overcome these challenges, we propose a novel method, FedShield-LLM, that uses pruning with Fully Homomorphic Encryption (FHE) for Low-Rank Adaptation (LoRA) parameters, enabling secure computations on encrypted model updates while mitigating the attack surface by deactivating less important LoRA parameters. Furthermore, optimized federated algorithms for cross-silo environments enhance scalability and efficiency. Parameter-efficient fine-tuning techniques like LoRA substantially reduce computational and communication overhead, making FL feasible for resource-constrained clients. Experimental results show that the proposed method outperforms existing methods while maintaining robust privacy protection, enabling organizations to collaboratively train secure and efficient LLMs.   The code and data are available at, https://github.com/solidlabnetwork/fedshield-llm

摘要: 联合学习（FL）提供了一个去中心化的框架，通过利用跨组织的计算资源，同时将敏感数据保留在本地设备上，来训练和微调大型语言模型（LLM）。它可以解决隐私和安全问题，同时应对与LLM大量计算需求相关的挑战，而这对于中小型组织来说可能是令人望而却步的。FL通过微调支持跨竖井应用程序开发特定于任务的LLM，但仍然容易受到推断攻击，例如成员资格推断和梯度倒置，从而威胁数据隐私。之前的研究在LLM微调中使用了差异隐私（DP），尽管它可以有效地保护隐私，但可能会降低模型性能。为了克服这些挑战，我们提出了一种新颖的方法FedShield-LLM，该方法对低等级自适应（LoRA）参数使用完全同质加密（FHE）进行修剪，从而能够对加密模型更新进行安全计算，同时通过停用不太重要的LoRA参数来减轻攻击表面。此外，针对跨竖井环境的优化联邦算法增强了可扩展性和效率。LoRA等参数高效的微调技术大大减少了计算和通信负担，使FL对于资源有限的客户来说可行。实验结果表明，所提出的方法优于现有方法，同时保持稳健的隐私保护，使组织能够协作训练安全有效的LLM。   代码和数据可访问https://github.com/solidlabnetwork/fedshield-llm



## **41. Comprehensive Vulnerability Analysis is Necessary for Trustworthy LLM-MAS**

可信LLM-MAS的脆弱性分析 cs.CR

**SubmitDate**: 2025-06-06    [abs](http://arxiv.org/abs/2506.01245v2) [paper-pdf](http://arxiv.org/pdf/2506.01245v2)

**Authors**: Pengfei He, Yue Xing, Shen Dong, Juanhui Li, Zhenwei Dai, Xianfeng Tang, Hui Liu, Han Xu, Zhen Xiang, Charu C. Aggarwal, Hui Liu

**Abstract**: This paper argues that a comprehensive vulnerability analysis is essential for building trustworthy Large Language Model-based Multi-Agent Systems (LLM-MAS). These systems, which consist of multiple LLM-powered agents working collaboratively, are increasingly deployed in high-stakes applications but face novel security threats due to their complex structures. While single-agent vulnerabilities are well-studied, LLM-MAS introduces unique attack surfaces through inter-agent communication, trust relationships, and tool integration that remain significantly underexplored. We present a systematic framework for vulnerability analysis of LLM-MAS that unifies diverse research. For each type of vulnerability, we define formal threat models grounded in practical attacker capabilities and illustrate them using real-world LLM-MAS applications. This formulation enables rigorous quantification of vulnerability across different architectures and provides a foundation for designing meaningful evaluation benchmarks. Our analysis reveals that LLM-MAS faces elevated risk due to compositional effects -- vulnerabilities in individual components can cascade through agent communication, creating threat models not present in single-agent systems. We conclude by identifying critical open challenges: (1) developing benchmarks specifically tailored to LLM-MAS vulnerability assessment, (2) considering new potential attacks specific to multi-agent architectures, and (3) implementing trust management systems that can enforce security in LLM-MAS. This research provides essential groundwork for future efforts to enhance LLM-MAS trustworthiness as these systems continue their expansion into critical applications.

摘要: 本文认为，全面的漏洞分析对于构建值得信赖的基于大型语言模型的多代理系统（LLM-MAS）至关重要。这些系统由多个LLM驱动的代理协同工作组成，越来越多地部署在高风险应用中，但由于其复杂的结构而面临着新型的安全威胁。虽然单代理漏洞得到了充分研究，但LLM-MAS通过代理间通信、信任关系和工具集成引入了独特的攻击表面，但这些攻击表面的研究仍然严重不足。我们提出了一个统一不同研究的LLM-MAS脆弱性分析的系统框架。对于每种类型的漏洞，我们根据实际攻击者能力定义正式威胁模型，并使用现实世界的LLM-MAS应用程序来说明它们。该公式能够严格量化不同架构的脆弱性，并为设计有意义的评估基准提供基础。我们的分析表明，由于组成效应，LLM-MAS面临着更高的风险--单个组件中的漏洞可以通过代理通信级联，从而创建单代理系统中不存在的威胁模型。我们通过识别关键的开放挑战来总结：（1）开发专门针对LLM-MAS漏洞评估定制的基准，（2）考虑针对多代理架构的新潜在攻击，以及（3）实施可以在LLM-MAS中强制执行安全性的信任管理系统。随着这些系统继续扩展到关键应用，这项研究为未来增强LLM-MAS可信度的努力提供了重要的基础。



## **42. SoK: Are Watermarks in LLMs Ready for Deployment?**

SoK：LLM中的水印准备好部署了吗？ cs.CR

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2506.05594v1) [paper-pdf](http://arxiv.org/pdf/2506.05594v1)

**Authors**: Kieu Dang, Phung Lai, NhatHai Phan, Yelong Shen, Ruoming Jin, Abdallah Khreishah, My Thai

**Abstract**: Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs.   To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.

摘要: 大型语言模型（LLM）改变了自然语言处理，在不同任务中展示了令人印象深刻的能力。然而，部署这些模型会带来与知识产权侵犯和潜在滥用相关的关键风险，特别是因为对手可以模仿这些模型来窃取服务或产生误导性输出。我们特别关注模型窃取攻击，因为它们与专有LLM高度相关，并对其安全、收入和道德部署构成严重威胁。虽然已经出现了各种水印技术来减轻这些风险，但目前尚不清楚社区和行业在LLM中开发和部署水印方面取得了多大进展。   为了弥合这一差距，我们的目标是通过1）提供LLM中水印的详细分类，2）提出一种新型知识产权分类器来探索水印在攻击和无攻击环境下的有效性和影响，3）分析LLM中现有水印的局限性，4）讨论LLM中水印的实际挑战和潜在的未来方向。通过广泛的实验，我们表明，尽管研究成果令人鼓舞，领先公司和社区也对部署水印给予了极大的关注，但由于这些技术对LLM和下游任务的模型效用产生不利影响，这些技术尚未在现实世界应用中充分发挥潜力。我们的研究结果提供了对LLM中的水印的深刻理解，强调了针对LLM部署量身定制的实用水印解决方案的必要性。



## **43. Why LLM Safety Guardrails Collapse After Fine-tuning: A Similarity Analysis Between Alignment and Fine-tuning Datasets**

为什么LLM安全护栏在微调后崩溃：对齐和微调数据集之间的相似性分析 cs.CR

Project Page: https://hsiung.cc/llm-similarity-risk/

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2506.05346v1) [paper-pdf](http://arxiv.org/pdf/2506.05346v1)

**Authors**: Lei Hsiung, Tianyu Pang, Yung-Chen Tang, Linyue Song, Tsung-Yi Ho, Pin-Yu Chen, Yaoqing Yang

**Abstract**: Recent advancements in large language models (LLMs) have underscored their vulnerability to safety alignment jailbreaks, particularly when subjected to downstream fine-tuning. However, existing mitigation strategies primarily focus on reactively addressing jailbreak incidents after safety guardrails have been compromised, removing harmful gradients during fine-tuning, or continuously reinforcing safety alignment throughout fine-tuning. As such, they tend to overlook a critical upstream factor: the role of the original safety-alignment data. This paper therefore investigates the degradation of safety guardrails through the lens of representation similarity between upstream alignment datasets and downstream fine-tuning tasks. Our experiments demonstrate that high similarity between these datasets significantly weakens safety guardrails, making models more susceptible to jailbreaks. Conversely, low similarity between these two types of datasets yields substantially more robust models and thus reduces harmfulness score by up to 10.33%. By highlighting the importance of upstream dataset design in the building of durable safety guardrails and reducing real-world vulnerability to jailbreak attacks, these findings offer actionable insights for fine-tuning service providers.

摘要: 大型语言模型（LLM）的最新进展凸显了它们对安全对齐越狱的脆弱性，特别是在接受下游微调时。然而，现有的缓解策略主要集中在安全护栏受损后反应性地解决越狱事件，在微调期间消除有害梯度，或者在整个微调期间不断加强安全对齐。因此，他们往往忽视了一个关键的上游因素：原始安全对齐数据的作用。因此，本文通过上游对齐数据集和下游微调任务之间的表示相似性的视角来研究安全护栏的退化。我们的实验表明，这些数据集之间的高度相似性会显着削弱安全护栏，使模型更容易受到越狱的影响。相反，这两种类型的数据集之间的低相似性会产生更稳健的模型，从而将危害性评分降低高达10.33%。通过强调上游数据集设计在构建持久安全护栏和减少现实世界中对越狱攻击的脆弱性方面的重要性，这些发现为微调服务提供商提供了可操作的见解。



## **44. Beyond the Protocol: Unveiling Attack Vectors in the Model Context Protocol Ecosystem**

超越协议：揭开模型上下文协议生态系统中的攻击载体 cs.CR

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2506.02040v2) [paper-pdf](http://arxiv.org/pdf/2506.02040v2)

**Authors**: Hao Song, Yiming Shen, Wenxuan Luo, Leixin Guo, Ting Chen, Jiashui Wang, Beibei Li, Xiaosong Zhang, Jiachi Chen

**Abstract**: The Model Context Protocol (MCP) is an emerging standard designed to enable seamless interaction between Large Language Model (LLM) applications and external tools or resources. Within a short period, thousands of MCP services have already been developed and deployed. However, the client-server integration architecture inherent in MCP may expand the attack surface against LLM Agent systems, introducing new vulnerabilities that allow attackers to exploit by designing malicious MCP servers. In this paper, we present the first systematic study of attack vectors targeting the MCP ecosystem. Our analysis identifies four categories of attacks, i.e., Tool Poisoning Attacks, Puppet Attacks, Rug Pull Attacks, and Exploitation via Malicious External Resources. To evaluate the feasibility of these attacks, we conduct experiments following the typical steps of launching an attack through malicious MCP servers: upload-download-attack. Specifically, we first construct malicious MCP servers and successfully upload them to three widely used MCP aggregation platforms. The results indicate that current audit mechanisms are insufficient to identify and prevent the proposed attack methods. Next, through a user study and interview with 20 participants, we demonstrate that users struggle to identify malicious MCP servers and often unknowingly install them from aggregator platforms. Finally, we demonstrate that these attacks can trigger harmful behaviors within the user's local environment-such as accessing private files or controlling devices to transfer digital assets-by deploying a proof-of-concept (PoC) framework against five leading LLMs. Additionally, based on interview results, we discuss four key challenges faced by the current security ecosystem surrounding MCP servers. These findings underscore the urgent need for robust security mechanisms to defend against malicious MCP servers.

摘要: 模型上下文协议（HCP）是一种新兴标准，旨在实现大型语言模型（LLM）应用程序与外部工具或资源之间的无缝交互。在短时间内，数千项HCP服务已经开发和部署。然而，LCP固有的客户端-服务器集成架构可能会扩大针对LLM Agent系统的攻击面，引入新的漏洞，允许攻击者通过设计恶意的LCP服务器来利用这些漏洞。在本文中，我们首次对针对LCP生态系统的攻击载体进行了系统研究。我们的分析确定了四类攻击，即，工具中毒攻击、傀儡攻击、地毯式攻击和通过恶意外部资源进行的利用。为了评估这些攻击的可行性，我们进行实验后，通过恶意MCP服务器发起攻击的典型步骤：上传-下载-攻击。具体来说，我们首先构建恶意MCP服务器，并成功地将它们上传到三个广泛使用的MCP聚合平台。结果表明，当前的审计机制不足以识别和防止拟议的攻击方法。接下来，通过用户研究和对20名参与者的采访，我们证明用户很难识别恶意的LCP服务器，并且通常在不知不觉中从聚合平台安装它们。最后，我们证明，通过针对五种领先的LLM部署概念验证（RST）框架，这些攻击可能会在用户本地环境中引发有害行为，例如访问私人文件或控制设备传输数字资产。此外，根据采访结果，我们讨论了当前围绕LCP服务器的安全生态系统面临的四个关键挑战。这些发现凸显了迫切需要强大的安全机制来抵御恶意的LCP服务器。



## **45. DREAM: Disentangling Risks to Enhance Safety Alignment in Multimodal Large Language Models**

梦想：理清风险以增强多模式大型语言模型的安全一致性 cs.CL

[NAACL 2025] The first four authors contribute equally, 23 pages,  repo at https://github.com/Kizna1ver/DREAM

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2504.18053v2) [paper-pdf](http://arxiv.org/pdf/2504.18053v2)

**Authors**: Jianyu Liu, Hangyu Guo, Ranjie Duan, Xingyuan Bu, Yancheng He, Shilong Li, Hui Huang, Jiaheng Liu, Yucheng Wang, Chenchen Jing, Xingwei Qu, Xiao Zhang, Yingshui Tan, Yanan Wu, Jihao Gu, Yangguang Li, Jianke Zhu

**Abstract**: Multimodal Large Language Models (MLLMs) pose unique safety challenges due to their integration of visual and textual data, thereby introducing new dimensions of potential attacks and complex risk combinations. In this paper, we begin with a detailed analysis aimed at disentangling risks through step-by-step reasoning within multimodal inputs. We find that systematic multimodal risk disentanglement substantially enhances the risk awareness of MLLMs. Via leveraging the strong discriminative abilities of multimodal risk disentanglement, we further introduce \textbf{DREAM} (\textit{\textbf{D}isentangling \textbf{R}isks to \textbf{E}nhance Safety \textbf{A}lignment in \textbf{M}LLMs}), a novel approach that enhances safety alignment in MLLMs through supervised fine-tuning and iterative Reinforcement Learning from AI Feedback (RLAIF). Experimental results show that DREAM significantly boosts safety during both inference and training phases without compromising performance on normal tasks (namely oversafety), achieving a 16.17\% improvement in the SIUO safe\&effective score compared to GPT-4V. The data and code are available at https://github.com/Kizna1ver/DREAM.

摘要: 多模式大型语言模型（MLLM）由于集成了视觉和文本数据，带来了独特的安全挑战，从而引入了潜在攻击和复杂风险组合的新维度。在本文中，我们首先进行详细分析，旨在通过多模式输入中的逐步推理来解开风险。我们发现，系统性多模式风险理清大大增强了MLLM的风险意识。通过利用多模式风险解开的强大辨别能力，我们进一步引入了\textBF{DREAM}（\texttit {\textBF{D}isentangling \textBF{R}isks to \textBF{E}nhance Safety \textBF{A} lignation in \textBF{M} LLM}），这是一种新颖的方法，通过有监督的微调和来自人工智能反馈的迭代强化学习（RLAIF）来增强MLLM中的安全一致性。实验结果表明，DREAM在推理和训练阶段都显著提高了安全性，而不影响正常任务的性能（即过度安全），与GPT-4V相比，SIUO安全有效分数提高了16.17%.数据和代码可在https://github.com/Kizna1ver/DREAM上获取。



## **46. Membership Inference Attacks on Sequence Models**

对序列模型的成员推断攻击 cs.CR

Accepted to the 8th Deep Learning Security and Privacy Workshop  (DLSP) workshop (best paper award)

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2506.05126v1) [paper-pdf](http://arxiv.org/pdf/2506.05126v1)

**Authors**: Lorenzo Rossi, Michael Aerni, Jie Zhang, Florian Tramèr

**Abstract**: Sequence models, such as Large Language Models (LLMs) and autoregressive image generators, have a tendency to memorize and inadvertently leak sensitive information. While this tendency has critical legal implications, existing tools are insufficient to audit the resulting risks. We hypothesize that those tools' shortcomings are due to mismatched assumptions. Thus, we argue that effectively measuring privacy leakage in sequence models requires leveraging the correlations inherent in sequential generation. To illustrate this, we adapt a state-of-the-art membership inference attack to explicitly model within-sequence correlations, thereby demonstrating how a strong existing attack can be naturally extended to suit the structure of sequence models. Through a case study, we show that our adaptations consistently improve the effectiveness of memorization audits without introducing additional computational costs. Our work hence serves as an important stepping stone toward reliable memorization audits for large sequence models.

摘要: 序列模型，例如大型语言模型（LLM）和自回归图像生成器，倾向于记忆并无意中泄露敏感信息。虽然这种趋势具有重要的法律影响，但现有工具不足以审计由此产生的风险。我们假设这些工具的缺点是由于假设不匹配。因此，我们认为，有效测量序列模型中的隐私泄露需要利用序列生成中固有的相关性。为了说明这一点，我们采用了最先进的隶属度推理攻击来显式地建模序列内相关性，从而展示了如何自然扩展强大的现有攻击以适应序列模型的结构。通过案例研究，我们表明，我们的调整能够持续提高记忆审计的有效性，而不会引入额外的计算成本。因此，我们的工作成为大序列模型可靠记忆审计的重要垫脚石。



## **47. Seven Security Challenges That Must be Solved in Cross-domain Multi-agent LLM Systems**

跨域多代理LLM系统必须解决的七个安全挑战 cs.CR

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2505.23847v2) [paper-pdf](http://arxiv.org/pdf/2505.23847v2)

**Authors**: Ronny Ko, Jiseong Jeong, Shuyuan Zheng, Chuan Xiao, Tae-Wan Kim, Makoto Onizuka, Won-Yong Shin

**Abstract**: Large language models (LLMs) are rapidly evolving into autonomous agents that cooperate across organizational boundaries, enabling joint disaster response, supply-chain optimization, and other tasks that demand decentralized expertise without surrendering data ownership. Yet, cross-domain collaboration shatters the unified trust assumptions behind current alignment and containment techniques. An agent benign in isolation may, when receiving messages from an untrusted peer, leak secrets or violate policy, producing risks driven by emergent multi-agent dynamics rather than classical software bugs. This position paper maps the security agenda for cross-domain multi-agent LLM systems. We introduce seven categories of novel security challenges, for each of which we also present plausible attacks, security evaluation metrics, and future research guidelines.

摘要: 大型语言模型（LLM）正在迅速演变为跨组织边界合作的自治代理，实现联合灾难响应、供应链优化以及其他需要分散专业知识而不放弃数据所有权的任务。然而，跨域协作打破了当前对齐和遏制技术背后的统一信任假设。孤立的良性代理在从不受信任的对等点接收消息时可能会泄露秘密或违反政策，从而产生由紧急多代理动态而不是经典软件错误驱动的风险。本立场文件绘制了跨域多代理LLM系统的安全议程。我们介绍了七类新型安全挑战，我们还针对每类挑战提供了合理的攻击、安全评估指标和未来的研究指南。



## **48. Sentinel: SOTA model to protect against prompt injections**

Sentinel：SOTA模型可防止快速注射 cs.CR

6 pages, 2 tables

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2506.05446v1) [paper-pdf](http://arxiv.org/pdf/2506.05446v1)

**Authors**: Dror Ivry, Oran Nahum

**Abstract**: Large Language Models (LLMs) are increasingly powerful but remain vulnerable to prompt injection attacks, where malicious inputs cause the model to deviate from its intended instructions. This paper introduces Sentinel, a novel detection model, qualifire/prompt-injection-sentinel, based on the \answerdotai/ModernBERT-large architecture. By leveraging ModernBERT's advanced features and fine-tuning on an extensive and diverse dataset comprising a few open-source and private collections, Sentinel achieves state-of-the-art performance. This dataset amalgamates varied attack types, from role-playing and instruction hijacking to attempts to generate biased content, alongside a broad spectrum of benign instructions, with private datasets specifically targeting nuanced error correction and real-world misclassifications. On a comprehensive, unseen internal test set, Sentinel demonstrates an average accuracy of 0.987 and an F1-score of 0.980. Furthermore, when evaluated on public benchmarks, it consistently outperforms strong baselines like protectai/deberta-v3-base-prompt-injection-v2. This work details Sentinel's architecture, its meticulous dataset curation, its training methodology, and a thorough evaluation, highlighting its superior detection capabilities.

摘要: 大型语言模型（LLM）越来越强大，但仍然容易受到提示注入攻击，恶意输入导致模型偏离其预期指令。本文介绍了Sentinel，这是一种新型检测模型qualifire/pret-inject-sentinel，基于\answerdotai/ModernBERT-large架构。通过利用ModernBERT的高级功能并对包括一些开源和私人收藏在内的广泛而多样化的数据集进行微调，Sentinel实现了最先进的性能。该数据集融合了各种攻击类型，从角色扮演和指令劫持到尝试生成有偏见的内容，以及广泛的良性指令，私人数据集专门针对细致入微的错误纠正和现实世界的错误分类。在一个全面的、看不见的内部测试集上，Sentinel的平均准确率为0.987，F1得分为0.980。此外，当在公共基准上进行评估时，它始终优于protectai/deberta-v3-base-prompt-injection-v2等强大的基线。这项工作详细介绍了Sentinel的架构，其细致的数据集管理，训练方法和全面的评估，突出了其卓越的检测能力。



## **49. Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models**

基于大型语言模型语义理解能力的自适应越狱策略 cs.CL

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2505.23404v2) [paper-pdf](http://arxiv.org/pdf/2505.23404v2)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin

**Abstract**: Adversarial attacks on Large Language Models (LLMs) via jailbreaking techniques-methods that circumvent their built-in safety and ethical constraints-have emerged as a critical challenge in AI security. These attacks compromise the reliability of LLMs by exploiting inherent weaknesses in their comprehension capabilities. This paper investigates the efficacy of jailbreaking strategies that are specifically adapted to the diverse levels of understanding exhibited by different LLMs. We propose the Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models, a novel framework that classifies LLMs into Type I and Type II categories according to their semantic comprehension abilities. For each category, we design tailored jailbreaking strategies aimed at leveraging their vulnerabilities to facilitate successful attacks. Extensive experiments conducted on multiple LLMs demonstrate that our adaptive strategy markedly improves the success rate of jailbreaking. Notably, our approach achieves an exceptional 98.9% success rate in jailbreaking GPT-4o(29 May 2025 release)

摘要: 通过越狱技术（规避其内置安全和道德约束的方法）对大型语言模型（LLM）进行的对抗攻击已成为人工智能安全领域的一个关键挑战。这些攻击通过利用LLM理解能力的固有弱点来损害LLM的可靠性。本文研究了专门适应不同法学硕士所表现出的不同理解水平的越狱策略的有效性。我们提出了基于大型语言模型语义理解能力的自适应越狱策略，这是一个新颖的框架，根据它们的语义理解能力将LLM分为类型I和类型II类别。对于每个类别，我们设计了量身定制的越狱策略，旨在利用其漏洞来促进成功的攻击。在多个LLM上进行的广泛实验表明，我们的自适应策略显着提高了越狱的成功率。值得注意的是，我们的方法在越狱GPT-4 o（2025年5月29日发布）中实现了98.9%的卓越成功率



## **50. On Automating Security Policies with Contemporary LLMs**

用现代LLM实现安全策略自动化 cs.CR

Short Paper. Accepted To Appear in IEEE SSE 2025 (part of SERVICES  2025)

**SubmitDate**: 2025-06-05    [abs](http://arxiv.org/abs/2506.04838v1) [paper-pdf](http://arxiv.org/pdf/2506.04838v1)

**Authors**: Pablo Fernández Saura, K. R. Jayaram, Vatche Isahagian, Jorge Bernal Bernabé, Antonio Skarmeta

**Abstract**: The complexity of modern computing environments and the growing sophistication of cyber threats necessitate a more robust, adaptive, and automated approach to security enforcement. In this paper, we present a framework leveraging large language models (LLMs) for automating attack mitigation policy compliance through an innovative combination of in-context learning and retrieval-augmented generation (RAG). We begin by describing how our system collects and manages both tool and API specifications, storing them in a vector database to enable efficient retrieval of relevant information. We then detail the architectural pipeline that first decomposes high-level mitigation policies into discrete tasks and subsequently translates each task into a set of actionable API calls. Our empirical evaluation, conducted using publicly available CTI policies in STIXv2 format and Windows API documentation, demonstrates significant improvements in precision, recall, and F1-score when employing RAG compared to a non-RAG baseline.

摘要: 现代计算环境的复杂性和网络威胁的日益复杂性需要一种更强大、适应性和自动化的安全执法方法。在本文中，我们提出了一个利用大型语言模型（LLM）的框架，通过上下文学习和检索增强生成（RAG）的创新组合来自动化攻击缓解策略合规。我们首先描述我们的系统如何收集和管理工具和API规范，并将它们存储在载体数据库中，以高效检索相关信息。然后，我们详细介绍架构管道，该管道首先将高级缓解策略分解为离散任务，然后将每个任务转换为一组可操作的API调用。我们使用STIXv 2格式的公开RTI政策和Windows API文档进行的实证评估表明，与非RAG基线相比，使用RAG时的准确性、召回率和F1评分有显着提高。



