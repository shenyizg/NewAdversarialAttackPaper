# Latest Large Language Model Attack Papers
**update at 2025-06-16 10:29:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Improving Large Language Model Safety with Contrastive Representation Learning**

通过对比表示学习提高大型语言模型安全性 cs.CL

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11938v1) [paper-pdf](http://arxiv.org/pdf/2506.11938v1)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense

摘要: 大型语言模型（LLM）是具有深远社会影响的强大工具，但它们对多样化且不受控制的输入产生响应的能力使它们容易受到对抗性攻击。虽然现有的防御通常很难在不同的攻击类型中进行概括，但表示工程的最新进展提供了有希望的替代方案。在这项工作中，我们提出了一个防御框架，将模型防御制定为对比表示学习（RTL）问题。我们的方法使用基于三重组的损失结合对抗性硬负挖掘来微调模型，以鼓励良性和有害表示之间的分离。我们跨多个模型的实验结果表明，我们的方法优于基于先验表示工程的防御，在不损害标准性能的情况下提高了针对输入级和嵌入空间攻击的鲁棒性。我们的代码可在https://github.com/samuelsimko/crl-llm-defense上获取



## **2. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

带有修剪的攻击图：优化隐形越狱提示生成以增强的LLM内容审核 cs.CR

14 pages, 5 figures

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2501.18638v2) [paper-pdf](http://arxiv.org/pdf/2501.18638v2)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.

摘要: 随着大型语言模型（LLM）变得越来越普遍，确保其针对对抗性滥用的鲁棒性至关重要。本文介绍了GAP（带有修剪的攻击图）框架，这是一种生成隐形越狱提示以评估和增强LLM保障措施的高级方法。GAP通过实现互连的图结构来解决现有基于树的LLM越狱方法的局限性，该结构能够实现跨攻击路径的知识共享。我们的实验评估证明了GAP相对于现有技术的优越性，攻击成功率提高了20.8%，同时将查询成本降低了62.7%。对于攻击开放式和封闭式LLM，RAP始终优于最先进的方法，攻击成功率> 96%。此外，我们还提供了专门的变体，例如用于自动种子生成的GAP-Auto和用于多模式攻击的GAP-VLM。事实证明，由间隙生成的提示在改进内容审核系统方面非常有效，用于微调时，真阳性检测率可提高108.5%，准确率可提高183.6%。我们的实施可在https://github.com/dsbuddy/GAP-LLM-Safety上获取。



## **3. Black-Box Adversarial Attacks on LLM-Based Code Completion**

基于LLM的代码补全黑盒对抗攻击 cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2408.02509v2) [paper-pdf](http://arxiv.org/pdf/2408.02509v2)

**Authors**: Slobodan Jenko, Niels Mündler, Jingxuan He, Mark Vero, Martin Vechev

**Abstract**: Modern code completion engines, powered by large language models (LLMs), assist millions of developers with their strong capabilities to generate functionally correct code. Due to this popularity, it is crucial to investigate the security implications of relying on LLM-based code completion. In this work, we demonstrate that state-of-the-art black-box LLM-based code completion engines can be stealthily biased by adversaries to significantly increase their rate of insecure code generation. We present the first attack, named INSEC, that achieves this goal. INSEC works by injecting an attack string as a short comment in the completion input. The attack string is crafted through a query-based optimization procedure starting from a set of carefully designed initialization schemes. We demonstrate INSEC's broad applicability and effectiveness by evaluating it on various state-of-the-art open-source models and black-box commercial services (e.g., OpenAI API and GitHub Copilot). On a diverse set of security-critical test cases, covering 16 CWEs across 5 programming languages, INSEC increases the rate of generated insecure code by more than 50%, while maintaining the functional correctness of generated code. We consider INSEC practical -- it requires low resources and costs less than 10 US dollars to develop on commodity hardware. Moreover, we showcase the attack's real-world deployability, by developing an IDE plug-in that stealthily injects INSEC into the GitHub Copilot extension.

摘要: 由大型语言模型（LLM）支持的现代代码完成引擎可以帮助数百万开发人员以其强大的能力生成功能正确的代码。由于这种受欢迎程度，研究依赖基于LLM的代码完成的安全影响至关重要。在这项工作中，我们证明了最先进的基于LLM的黑匣子代码完成引擎可能会受到对手的悄悄偏见，以显着提高其不安全代码生成率。我们介绍了第一个攻击，名为INSEC，可以实现这一目标。INSEC的工作原理是在完成输入中注入攻击字符串作为简短注释。攻击字符串是通过基于查询的优化过程从一组精心设计的初始化方案开始精心设计的。我们通过在各种最先进的开源模型和黑匣子商业服务上进行评估来展示INSEC的广泛适用性和有效性（例如，OpenAI API和GitHub Copilot）。在一组多样化的安全关键测试用例中，涵盖5种编程语言的16个CWE，INSEC将生成的不安全代码的比率提高了50%以上，同时保持生成代码的功能正确性。我们认为INSEC是可行的--在商品硬件上进行开发所需的资源较少，成本不到10美元。此外，我们通过开发一个将INSEC秘密注入GitHub Copilot扩展的IDE插件，展示了攻击在现实世界中的可部署性。



## **4. TrustGLM: Evaluating the Robustness of GraphLLMs Against Prompt, Text, and Structure Attacks**

TrustGLM：评估GraphLLM针对提示、文本和结构攻击的稳健性 cs.LG

12 pages, 5 figures, in KDD 2025

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11844v1) [paper-pdf](http://arxiv.org/pdf/2506.11844v1)

**Authors**: Qihai Zhang, Xinyue Sheng, Yuanfu Sun, Qiaoyu Tan

**Abstract**: Inspired by the success of large language models (LLMs), there is a significant research shift from traditional graph learning methods to LLM-based graph frameworks, formally known as GraphLLMs. GraphLLMs leverage the reasoning power of LLMs by integrating three key components: the textual attributes of input nodes, the structural information of node neighborhoods, and task-specific prompts that guide decision-making. Despite their promise, the robustness of GraphLLMs against adversarial perturbations remains largely unexplored-a critical concern for deploying these models in high-stakes scenarios. To bridge the gap, we introduce TrustGLM, a comprehensive study evaluating the vulnerability of GraphLLMs to adversarial attacks across three dimensions: text, graph structure, and prompt manipulations. We implement state-of-the-art attack algorithms from each perspective to rigorously assess model resilience. Through extensive experiments on six benchmark datasets from diverse domains, our findings reveal that GraphLLMs are highly susceptible to text attacks that merely replace a few semantically similar words in a node's textual attribute. We also find that standard graph structure attack methods can significantly degrade model performance, while random shuffling of the candidate label set in prompt templates leads to substantial performance drops. Beyond characterizing these vulnerabilities, we investigate defense techniques tailored to each attack vector through data-augmented training and adversarial training, which show promising potential to enhance the robustness of GraphLLMs. We hope that our open-sourced library will facilitate rapid, equitable evaluation and inspire further innovative research in this field.

摘要: 受大型语言模型（LLM）成功的启发，研究从传统的图学习方法发生了重大转变，转向基于LLM的图框架（正式称为GraphLLM）。GraphLLM通过集成三个关键组件来利用LLM的推理能力：输入节点的文本属性、节点邻居的结构信息以及指导决策的特定任务提示。尽管它们有希望，但GraphLLM对对抗性扰动的稳健性在很大程度上仍然没有被开发--这是在高风险场景中部署这些模型的一个关键问题。为了弥合这一差距，我们引入了TrustGLM，这是一项综合研究，评估了GraphLLM在三个维度（文本、图形结构和提示操作）中对对抗攻击的脆弱性。我们从各个角度实施最先进的攻击算法，以严格评估模型弹性。通过对来自不同领域的六个基准数据集的广泛实验，我们的研究结果表明，GraphLLM非常容易受到文本攻击，这些攻击只是替换节点文本属性中的一些语义相似的单词。我们还发现，标准图结构攻击方法会显着降低模型性能，而提示模板中候选标签集的随机洗牌会导致性能大幅下降。除了描述这些漏洞之外，我们还通过数据增强训练和对抗训练研究了针对每个攻击载体量身定制的防御技术，这些技术在增强GraphLLM稳健性方面表现出了广阔的潜力。我们希望我们的开源图书馆能够促进快速、公平的评估，并激发该领域的进一步创新研究。



## **5. Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective**

从心理测量学角度通过攻击来评估大型语言模型中的内隐偏差 cs.CL

Accepted to ACL 2025 Findings

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2406.14023v4) [paper-pdf](http://arxiv.org/pdf/2406.14023v4)

**Authors**: Yuchen Wen, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: As large language models (LLMs) become an important way of information access, there have been increasing concerns that LLMs may intensify the spread of unethical content, including implicit bias that hurts certain populations without explicit harmful words. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain demographics by attacking them from a psychometric perspective to elicit agreements to biased viewpoints. Inspired by psychometric principles in cognitive and social psychology, we propose three attack approaches, i.e., Disguise, Deception, and Teaching. Incorporating the corresponding attack instructions, we built two benchmarks: (1) a bilingual dataset with biased statements covering four bias types (2.7K instances) for extensive comparative analysis, and (2) BUMBLE, a larger benchmark spanning nine common bias types (12.7K instances) for comprehensive evaluation. Extensive evaluation of popular commercial and open-source LLMs shows that our methods can elicit LLMs' inner bias more effectively than competitive baselines. Our attack methodology and benchmarks offer an effective means of assessing the ethical risks of LLMs, driving progress toward greater accountability in their development. Our code, data, and benchmarks are available at https://yuchenwen1.github.io/ImplicitBiasEvaluation/.

摘要: 随着大型语言模型（LLM）成为信息获取的重要方式，人们越来越担心LLM可能会加剧不道德内容的传播，包括在没有明确有害词语的情况下伤害某些人群的隐性偏见。在本文中，我们通过从心理测量学的角度攻击LLM对某些人口统计数据的隐性偏见进行了严格评估，以获取对偏见观点的同意。受认知和社会心理学中心理测量原则的启发，我们提出了三种攻击方法，即伪装、欺骗和教学。综合相应的攻击指令，我们构建了两个基准：（1）双语数据集，其中包含涵盖四种偏见类型（2.7 K实例）的偏见陈述，用于广泛的比较分析，和（2）BUMBLE，一个跨越九种常见偏见类型（12.7 K实例）的更大基准，用于全面评估。对流行的商业和开源LLM的广泛评估表明，我们的方法比竞争基线更有效地引发LLM的内部偏见。我们的攻击方法和基准提供了评估LLM道德风险的有效手段，推动LLM在开发过程中实现更强的问责制。我们的代码、数据和基准可在https://yuchenwen1.github.io/ImplicitBiasEvaluation/上获取。



## **6. Investigating Vulnerabilities and Defenses Against Audio-Visual Attacks: A Comprehensive Survey Emphasizing Multimodal Models**

调查漏洞和针对视听攻击的防御：强调多模式模型的全面调查 cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11521v1) [paper-pdf](http://arxiv.org/pdf/2506.11521v1)

**Authors**: Jinming Wen, Xinyi Wu, Shuai Zhao, Yanhao Jia, Yuwen Li

**Abstract**: Multimodal large language models (MLLMs), which bridge the gap between audio-visual and natural language processing, achieve state-of-the-art performance on several audio-visual tasks. Despite the superior performance of MLLMs, the scarcity of high-quality audio-visual training data and computational resources necessitates the utilization of third-party data and open-source MLLMs, a trend that is increasingly observed in contemporary research. This prosperity masks significant security risks. Empirical studies demonstrate that the latest MLLMs can be manipulated to produce malicious or harmful content. This manipulation is facilitated exclusively through instructions or inputs, including adversarial perturbations and malevolent queries, effectively bypassing the internal security mechanisms embedded within the models. To gain a deeper comprehension of the inherent security vulnerabilities associated with audio-visual-based multimodal models, a series of surveys investigates various types of attacks, including adversarial and backdoor attacks. While existing surveys on audio-visual attacks provide a comprehensive overview, they are limited to specific types of attacks, which lack a unified review of various types of attacks. To address this issue and gain insights into the latest trends in the field, this paper presents a comprehensive and systematic review of audio-visual attacks, which include adversarial attacks, backdoor attacks, and jailbreak attacks. Furthermore, this paper also reviews various types of attacks in the latest audio-visual-based MLLMs, a dimension notably absent in existing surveys. Drawing upon comprehensive insights from a substantial review, this paper delineates both challenges and emergent trends for future research on audio-visual attacks and defense.

摘要: 多模式大型语言模型（MLLM）弥合了视听和自然语言处理之间的差距，在多项视听任务上实现了最先进的性能。尽管MLLM性能优越，但高质量视听训练数据和计算资源的稀缺使得需要利用第三方数据和开源MLLM，这是当代研究中越来越多地观察到的趋势。这种繁荣掩盖了巨大的安全风险。实证研究表明，最新的MLLM可能会被操纵以产生恶意或有害内容。这种操纵完全通过指令或输入（包括对抗性扰动和恶意查询）来促进，有效地绕过了模型中嵌入的内部安全机制。为了更深入地了解与基于视听的多模式模型相关的固有安全漏洞，一系列调查调查了各种类型的攻击，包括对抗性攻击和后门攻击。虽然现有的关于视听攻击的调查提供了全面的概述，但仅限于特定类型的攻击，缺乏对各种类型的攻击的统一审查。为了解决这个问题并深入了解该领域的最新趋势，本文对视听攻击进行了全面、系统的回顾，其中包括对抗性攻击、后门攻击和越狱攻击。此外，本文还审查了最新的基于视听的MLLM中的各种类型的攻击，这是现有调查中明显缺乏的一个方面。本文从大量的评论中得出了全面的见解，描绘了未来视听攻击和防御研究的挑战和新兴趋势。



## **7. Bias Amplification in RAG: Poisoning Knowledge Retrieval to Steer LLMs**

RAG中的偏见放大：毒害Steer LLM的知识检索 cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11415v1) [paper-pdf](http://arxiv.org/pdf/2506.11415v1)

**Authors**: Linlin Wang, Tianqing Zhu, Laiqiao Qin, Longxiang Gao, Wanlei Zhou

**Abstract**: In Large Language Models, Retrieval-Augmented Generation (RAG) systems can significantly enhance the performance of large language models by integrating external knowledge. However, RAG also introduces new security risks. Existing research focuses mainly on how poisoning attacks in RAG systems affect model output quality, overlooking their potential to amplify model biases. For example, when querying about domestic violence victims, a compromised RAG system might preferentially retrieve documents depicting women as victims, causing the model to generate outputs that perpetuate gender stereotypes even when the original query is gender neutral. To show the impact of the bias, this paper proposes a Bias Retrieval and Reward Attack (BRRA) framework, which systematically investigates attack pathways that amplify language model biases through a RAG system manipulation. We design an adversarial document generation method based on multi-objective reward functions, employ subspace projection techniques to manipulate retrieval results, and construct a cyclic feedback mechanism for continuous bias amplification. Experiments on multiple mainstream large language models demonstrate that BRRA attacks can significantly enhance model biases in dimensions. In addition, we explore a dual stage defense mechanism to effectively mitigate the impacts of the attack. This study reveals that poisoning attacks in RAG systems directly amplify model output biases and clarifies the relationship between RAG system security and model fairness. This novel potential attack indicates that we need to keep an eye on the fairness issues of the RAG system.

摘要: 在大型语言模型中，检索增强生成（RAG）系统可以通过集成外部知识来显着增强大型语言模型的性能。然而，RAG也带来了新的安全风险。现有的研究主要关注RAG系统中的中毒攻击如何影响模型输出质量，而忽视了它们放大模型偏差的潜力。例如，当查询家庭暴力受害者时，受损的RAG系统可能会优先检索将女性描述为受害者的文件，从而导致模型生成的输出即使在原始查询是性别中立的情况下也会延续性别刻板印象。为了展示偏见的影响，本文提出了一个偏差检索和奖励攻击（BRRA）框架，该框架系统地研究通过RAG系统操纵放大语言模型偏差的攻击途径。我们设计了一种基于多目标奖励函数的对抗性文档生成方法，采用子空间投影技术来操纵检索结果，并构建了连续偏差放大的循环反馈机制。对多个主流大型语言模型的实验表明，BRRA攻击可以显着增强模型维度偏差。此外，我们还探索了双阶段防御机制，以有效减轻攻击的影响。该研究表明，RAG系统中的中毒攻击直接放大了模型输出偏差，并澄清了RAG系统安全性与模型公平性之间的关系。这种新颖的潜在攻击表明我们需要密切关注RAG系统的公平性问题。



## **8. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

PLeak：针对大型语言模型应用程序的提示泄露攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2405.06823v3) [paper-pdf](http://arxiv.org/pdf/2405.06823v3)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.

摘要: 大型语言模型（LLM）支持一个具有许多下游应用程序（称为LLM应用程序）的新生态系统，这些应用程序具有不同的自然语言处理任务。LLM应用程序的功能和性能高度取决于其系统提示符，系统提示符指示后台LLM执行什么任务。因此，LLM应用程序开发人员通常会对系统进行保密，以保护其知识产权。因此，一种称为提示泄露的自然攻击是从LLM应用程序窃取系统提示，这会损害开发人员的知识产权。现有的提示泄露攻击主要依赖于手动构建的查询，因此效果有限。   本文中，我们设计了一种新颖的封闭式提示泄露攻击框架，称为PLeak，来优化对抗性查询，以便当攻击者将其发送到目标LLM应用程序时，其响应会显示其自己的系统提示。我们将寻找这样的对抗性查询作为一个优化问题，并大致使用基于梯度的方法来解决它。我们的关键想法是通过逐步优化系统提示的对手查询来分解优化目标，即从每个系统提示的前几个标记开始，一步一步地直到系统提示的整个长度。   我们在离线设置和现实世界的LLM应用程序中评估PLeak，例如，Poe上的用户，Poe是托管此类应用程序的流行平台。我们的结果表明，PLeak可以有效地泄露系统提示，并且不仅显着优于手动策划查询的基线，而且还优于具有根据现有越狱攻击修改和改编的优化查询的基线。我们负责任地向Poe报告了这些问题，目前仍在等待他们的回应。我们的实现可以在以下存储库中找到：https://github.com/BHui97/PLeak。



## **9. Improving LLM Safety Alignment with Dual-Objective Optimization**

通过双目标优化改善LLM安全一致性 cs.CL

ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2503.03710v2) [paper-pdf](http://arxiv.org/pdf/2503.03710v2)

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment

摘要: 现有的大型语言模型（LLM）训练时安全对齐技术仍然容易受到越狱攻击。直接偏好优化（DPO）是一种广泛应用的对齐方法，在实验和理论背景下都表现出局限性，因为其损失函数被证明对于拒绝学习来说次优。通过基于梯度的分析，我们发现了这些缺点，并提出了一种改进的安全调整，将DPO目标分解为两个部分：（1）稳健的拒绝训练，即使在产生部分不安全的世代时也鼓励拒绝，和（2）有针对性地忘记有害知识。这种方法显着提高了LLM针对各种越狱攻击的稳健性，包括跨分发和跨分发场景的预填充、后缀和多回合攻击。此外，我们引入了一种强调关键拒绝令牌的方法，通过结合基于奖励的令牌级加权机制进行拒绝学习，这进一步提高了针对对抗性利用的鲁棒性。我们的研究还表明，对越狱攻击的稳健性与训练过程中的代币分布变化以及拒绝和有害代币的内部表示相关，为LLM安全性调整的未来研究提供了有价值的方向。该代码可在https://github.com/wicai24/DOOR-Alignment上获取



## **10. Weak-to-Strong Jailbreaking on Large Language Models**

大型语言模型上的弱到强越狱 cs.CL

ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2401.17256v3) [paper-pdf](http://arxiv.org/pdf/2401.17256v3)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

摘要: 大型语言模型（LLM）很容易受到越狱攻击，从而导致有害、不道德或有偏见的文本生成。然而，现有的越狱方法计算成本很高。本文中，我们提出了弱到强越狱攻击，这是一种针对对齐LLM的有效推理时间攻击，以产生有害文本。我们的关键直觉是基于这样的观察：越狱和对齐的模型仅在其初始解码分布上有所不同。从弱到强攻击的关键技术见解是使用两个较小的模型（一个安全的模型和一个不安全的模型）来对抗性地修改明显更大的安全模型的解码概率。我们评估了对来自3个组织的5个不同开源LLM的弱到强攻击。结果表明，我们的方法可以将两个数据集的未对准率提高到99%以上，每个示例只需向前传递一次。我们的研究揭示了在调整LLM时需要解决的紧迫安全问题。作为初步尝试，我们提出了一种防御策略来抵御此类攻击，但创建更先进的防御仍然具有挑战性。复制该方法的代码可在https://github.com/XuandongZhao/weak-to-strong上获取



## **11. PRSA: Prompt Stealing Attacks against Real-World Prompt Services**

PRSA：针对现实世界提示服务的提示窃取攻击 cs.CR

This is the extended version of the paper accepted at the 34th USENIX  Security Symposium (USENIX Security 2025)

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2402.19200v3) [paper-pdf](http://arxiv.org/pdf/2402.19200v3)

**Authors**: Yong Yang, Changjiang Li, Qingming Li, Oubo Ma, Haoyu Wang, Zonghui Wang, Yandong Gao, Wenzhi Chen, Shouling Ji

**Abstract**: Recently, large language models (LLMs) have garnered widespread attention for their exceptional capabilities. Prompts are central to the functionality and performance of LLMs, making them highly valuable assets. The increasing reliance on high-quality prompts has driven significant growth in prompt services. However, this growth also expands the potential for prompt leakage, increasing the risk that attackers could replicate original functionalities, create competing products, and severely infringe on developers' intellectual property. Despite these risks, prompt leakage in real-world prompt services remains underexplored.   In this paper, we present PRSA, a practical attack framework designed for prompt stealing. PRSA infers the detailed intent of prompts through very limited input-output analysis and can successfully generate stolen prompts that replicate the original functionality. Extensive evaluations demonstrate PRSA's effectiveness across two main types of real-world prompt services. Specifically, compared to previous works, it improves the attack success rate from 17.8% to 46.1% in prompt marketplaces and from 39% to 52% in LLM application stores, respectively. Notably, in the attack on "Math", one of the most popular educational applications in OpenAI's GPT Store with over 1 million conversations, PRSA uncovered a hidden Easter egg that had not been revealed previously. Besides, our analysis reveals that higher mutual information between a prompt and its output correlates with an increased risk of leakage. This insight guides the design and evaluation of two potential defenses against the security threats posed by PRSA. We have reported these findings to the prompt service vendors, including PromptBase and OpenAI, and actively collaborate with them to implement defensive measures.

摘要: 最近，大型语言模型（LLM）因其卓越的功能而受到广泛关注。预算对于LLM的功能和性能至关重要，使其成为极具价值的资产。对高质量提示的日益依赖推动了提示服务的显着增长。然而，这种增长也扩大了即时泄露的可能性，增加了攻击者复制原始功能、创建竞争产品并严重侵犯开发人员知识产权的风险。尽管存在这些风险，但现实世界即时服务中的即时泄漏仍然没有得到充分的研究。   在本文中，我们提出了PRSA，这是一个为即时窃取而设计的实用攻击框架。PRSA通过非常有限的输入输出分析推断提示的详细意图，并可以成功生成复制原始功能的被盗提示。广泛的评估证明了PRSA在两种主要类型的现实世界提示服务中的有效性。具体来说，与之前的作品相比，它将即时市场的攻击成功率从17.8%提高到46.1%，将LLM应用商店的攻击成功率从39%提高到52%。值得注意的是，在对OpenAI GPT Store中最受欢迎的教育应用程序之一“Math”的攻击中，PRSA发现了一个之前未公开的隐藏复活节彩蛋。此外，我们的分析表明，提示与其输出之间的互信息越高，泄漏风险越大。这一见解指导了针对PRSA构成的安全威胁的两种潜在防御措施的设计和评估。我们已将这些发现报告给Inbox Base和OpenAI等即时服务供应商，并积极与他们合作实施防御措施。



## **12. Unsourced Adversarial CAPTCHA: A Bi-Phase Adversarial CAPTCHA Framework**

无源对抗验证码：两阶段对抗验证码框架 cs.CV

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10685v1) [paper-pdf](http://arxiv.org/pdf/2506.10685v1)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: With the rapid advancements in deep learning, traditional CAPTCHA schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on original image characteristics, resulting in distortions that hinder human interpretation and limit applicability in scenarios lacking initial input images. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (UAC), a novel framework generating high-fidelity adversarial examples guided by attacker-specified text prompts. Leveraging a Large Language Model (LLM), UAC enhances CAPTCHA diversity and supports both targeted and untargeted attacks. For targeted attacks, the EDICT method optimizes dual latent variables in a diffusion model for superior image quality. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-UAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show BP-UAC achieves high attack success rates across diverse systems, generating natural CAPTCHAs indistinguishable to humans and DNNs.

摘要: 随着深度学习的快速发展，传统的CAPTCHA方案越来越容易受到深度神经网络（DNN）支持的自动攻击的影响。现有的对抗攻击方法通常依赖于原始图像特征，从而导致失真，阻碍人类解释并限制在缺乏初始输入图像的场景中的适用性。为了解决这些挑战，我们提出了无源对抗验证码（UAC），这是一种新颖的框架，可以在攻击者指定的文本提示的指导下生成高保真对抗示例。利用大型语言模型（LLM），UAC增强了验证码多样性，并支持有针对性和无针对性的攻击。对于有针对性的攻击，EDICT方法优化扩散模型中的双重潜在变量，以获得卓越的图像质量。在无目标攻击中，特别是对于黑匣子场景，我们引入了双路径无源对抗性CAPTCHA（BP-UAC），这是一种两步优化策略，采用多峰梯度和双路径优化来实现高效的误分类。实验表明，BP-UAC在不同系统中实现了很高的攻击成功率，生成人类和DNN难以区分的自然验证码。



## **13. SoK: Evaluating Jailbreak Guardrails for Large Language Models**

SoK：评估大型语言模型的越狱护栏 cs.CR

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10597v1) [paper-pdf](http://arxiv.org/pdf/2506.10597v1)

**Authors**: Xunguang Wang, Zhenlan Ji, Wenxuan Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress, but their deployment has exposed critical vulnerabilities, particularly to jailbreak attacks that circumvent safety mechanisms. Guardrails--external defense mechanisms that monitor and control LLM interaction--have emerged as a promising solution. However, the current landscape of LLM guardrails is fragmented, lacking a unified taxonomy and comprehensive evaluation framework. In this Systematization of Knowledge (SoK) paper, we present the first holistic analysis of jailbreak guardrails for LLMs. We propose a novel, multi-dimensional taxonomy that categorizes guardrails along six key dimensions, and introduce a Security-Efficiency-Utility evaluation framework to assess their practical effectiveness. Through extensive analysis and experiments, we identify the strengths and limitations of existing guardrail approaches, explore their universality across attack types, and provide insights into optimizing defense combinations. Our work offers a structured foundation for future research and development, aiming to guide the principled advancement and deployment of robust LLM guardrails. The code is available at https://github.com/xunguangwang/SoK4JailbreakGuardrails.

摘要: 大型语言模型（LLM）取得了显着的进步，但它们的部署暴露了关键漏洞，特别是规避安全机制的越狱攻击。Guardrails--监控和控制LLM交互的外部防御机制--已成为一种有希望的解决方案。然而，目前LLM护栏格局支离破碎，缺乏统一的分类和全面的评估框架。在这篇知识系统化（SoK）论文中，我们首次对LLM的越狱护栏进行了整体分析。我们提出了一种新颖的多维分类法，根据六个关键维度对护栏进行分类，并引入安全-效率-效用评估框架来评估其实际有效性。通过广泛的分析和实验，我们确定了现有护栏方法的优点和局限性，探索其在攻击类型中的普遍性，并提供优化防御组合的见解。我们的工作为未来的研究和开发提供了结构化的基础，旨在指导强大的LLM护栏的有原则的推进和部署。该代码可在https://github.com/xunguangwang/SoK4JailbreakGuardrails上获取。



## **14. Towards Action Hijacking of Large Language Model-based Agent**

基于大型语言模型的Agent的动作劫持 cs.CR

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2412.10807v2) [paper-pdf](http://arxiv.org/pdf/2412.10807v2)

**Authors**: Yuyang Zhang, Kangjie Chen, Jiaxin Gao, Ronghao Cui, Run Wang, Lina Wang, Tianwei Zhang

**Abstract**: Recently, applications powered by Large Language Models (LLMs) have made significant strides in tackling complex tasks. By harnessing the advanced reasoning capabilities and extensive knowledge embedded in LLMs, these applications can generate detailed action plans that are subsequently executed by external tools. Furthermore, the integration of retrieval-augmented generation (RAG) enhances performance by incorporating up-to-date, domain-specific knowledge into the planning and execution processes. This approach has seen widespread adoption across various sectors, including healthcare, finance, and software development. Meanwhile, there are also growing concerns regarding the security of LLM-based applications. Researchers have disclosed various attacks, represented by jailbreak and prompt injection, to hijack the output actions of these applications. Existing attacks mainly focus on crafting semantically harmful prompts, and their validity could diminish when security filters are employed. In this paper, we introduce AI$\mathbf{^2}$, a novel attack to manipulate the action plans of LLM-based applications. Different from existing solutions, the innovation of AI$\mathbf{^2}$ lies in leveraging the knowledge from the application's database to facilitate the construction of malicious but semantically-harmless prompts. To this end, it first collects action-aware knowledge from the victim application. Based on such knowledge, the attacker can generate misleading input, which can mislead the LLM to generate harmful action plans, while bypassing possible detection mechanisms easily. Our evaluations on three real-world applications demonstrate the effectiveness of AI$\mathbf{^2}$: it achieves an average attack success rate of 84.30\% with the best of 99.70\%. Besides, it gets an average bypass rate of 92.7\% against common safety filters and 59.45\% against dedicated defense.

摘要: 最近，由大型语言模型（LLM）支持的应用程序在处理复杂任务方面取得了重大进展。通过利用LLM中嵌入的高级推理能力和广泛知识，这些应用程序可以生成详细的行动计划，随后由外部工具执行。此外，检索增强生成（RAG）的集成通过将最新的、特定领域的知识融入到规划和执行流程中来增强性能。这种方法已在医疗保健、金融和软件开发等各个行业广泛采用。与此同时，人们对基于LLM的应用程序的安全性的担忧也越来越大。研究人员披露了以越狱和提示注入为代表的各种攻击，以劫持这些应用程序的输出动作。现有的攻击主要集中在制作语义上有害的提示，当使用安全过滤器时，其有效性可能会降低。在本文中，我们介绍了AI$\mathBF{#39; 2}$，这是一种用于操纵基于LLM的应用程序动作计划的新型攻击。与现有解决方案不同，AI$\mathBF{#39; 2}$的创新在于利用应用程序数据库中的知识来促进恶意但语义无害的提示的构建。为此，它首先从受害者应用程序收集动作感知知识。基于这些知识，攻击者可以生成误导性输入，从而误导LLM生成有害的行动计划，同时轻松绕过可能的检测机制。我们对三个现实世界应用程序的评估证明了AI$\mathBF{#39; 2}$的有效性：它的平均攻击成功率为84.30%，最好的攻击成功率为99.70%。此外，针对常见安全过滤器的平均旁路率为92.7%，针对专用防御器的平均旁路率为59.45%。



## **15. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

紧急系统的守护者：用紧急系统防止多次枪击越狱 cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2502.16750v4) [paper-pdf](http://arxiv.org/pdf/2502.16750v4)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.

摘要: 使用大型语言模型的自主人工智能代理可以在社会各个领域创造不可否认的价值，但它们面临着来自对手的安全威胁，需要立即采取保护性解决方案，因为信任和安全问题的出现。考虑到多次越狱和欺骗性对齐是一些主要的高级攻击，无法通过监督训练期间使用的静态护栏来缓解这一点，这指出了现实世界鲁棒性的一个关键研究优先事项。动态多代理系统中静态护栏的组合无法抵御这些攻击。我们打算通过开发新的评估框架来增强基于LLM的代理的安全性，该评估框架可以识别和应对安全运营部署的威胁。我们的工作使用三种检查方法通过反向图灵测试检测流氓代理，并通过多代理模拟分析欺骗性对齐，并通过使用GEMINI 1.5 pro和llama-3.3- 70 B、deepseek r1模型进行测试来开发反越狱系统，使用工具介导的对抗场景。检测能力很强，例如GEMINI 1.5 pro的准确率为94%，但系统在长时间攻击下会遭受持久漏洞，因为提示长度会增加攻击成功率（ASB），多样性指标在预测中变得无效，同时揭示了多个复杂的系统故障。研究结果表明，有必要采用基于主动监控的灵活安全系统，主动监控可以由代理本身执行，并由系统管理员进行自适应的干预，因为当前的模型可能会创建漏洞，从而导致系统不可靠和脆弱。因此，在我们的工作中，我们试图解决此类情况，并提出一个全面的框架来应对安全问题。



## **16. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

不要落后，RAG：使用RAG进行免训练对抗检测 cs.AI

Accepted at VecDB @ ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2504.04858v2) [paper-pdf](http://arxiv.org/pdf/2504.04858v2)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.

摘要: 对抗性补丁攻击通过嵌入误导深度模型的局部扰动，对视觉系统构成重大威胁。传统的防御方法通常需要重新培训或微调，这使得它们对于现实世界的部署来说不切实际。我们提出了一个免训练的视觉检索增强生成（VRAG）框架，该框架集成了用于对抗性补丁检测的视觉语言模型（VLM）。通过检索视觉上相似的补丁和图像，这些补丁和图像类似于不断扩展的数据库中存储的攻击，VRAG执行生成式推理以识别不同的攻击类型，而所有这些都无需额外的训练或微调。我们广泛评估了开源大型VLM，包括Qwen-VL-Plus、Qwen2.5-VL-72 B和UI-TARS-72 B-DPO，以及Gemini-2.0（一种闭源模型）。值得注意的是，开源UI-TARS-72 B-DPO模型实现了高达95%的分类准确率，为开源对抗补丁检测奠定了新的最新水平。Gemini-2.0的总体准确率达到了最高的98%，但仍然是闭源的。实验结果证明了VRAG在以最少的人类注释识别各种对抗补丁方面的有效性，为针对不断发展的对抗补丁攻击的稳健、实用的防御铺平了道路。



## **17. SOFT: Selective Data Obfuscation for Protecting LLM Fine-tuning against Membership Inference Attacks**

软银：选择性数据混淆，用于保护LLM微调免受成员推断攻击 cs.CR

Accepted by the 34th USENIX Security Symposium 2025. Code is  available at https://github.com/KaiyuanZh/SOFT

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10424v1) [paper-pdf](http://arxiv.org/pdf/2506.10424v1)

**Authors**: Kaiyuan Zhang, Siyuan Cheng, Hanxi Guo, Yuetian Chen, Zian Su, Shengwei An, Yuntao Du, Charles Fleming, Ashish Kundu, Xiangyu Zhang, Ninghui Li

**Abstract**: Large language models (LLMs) have achieved remarkable success and are widely adopted for diverse applications. However, fine-tuning these models often involves private or sensitive information, raising critical privacy concerns. In this work, we conduct the first comprehensive study evaluating the vulnerability of fine-tuned LLMs to membership inference attacks (MIAs). Our empirical analysis demonstrates that MIAs exploit the loss reduction during fine-tuning, making them highly effective in revealing membership information. These findings motivate the development of our defense. We propose SOFT (\textbf{S}elective data \textbf{O}bfuscation in LLM \textbf{F}ine-\textbf{T}uning), a novel defense technique that mitigates privacy leakage by leveraging influential data selection with an adjustable parameter to balance utility preservation and privacy protection. Our extensive experiments span six diverse domains and multiple LLM architectures and scales. Results show that SOFT effectively reduces privacy risks while maintaining competitive model performance, offering a practical and scalable solution to safeguard sensitive information in fine-tuned LLMs.

摘要: 大型语言模型（LLM）已经取得了巨大的成功，并被广泛应用于各种应用。然而，对这些模型进行微调往往涉及私人或敏感信息，从而引发了严重的隐私问题。在这项工作中，我们进行了第一次全面的研究，评估微调LLM成员推理攻击（MIA）的脆弱性。我们的实证分析表明，MIA利用微调过程中的损失减少，使他们非常有效地揭示成员信息。这些发现激励了我们防御的发展。我们提出了SOFT（\textBF{S} selective data \textBF{O}bfuscation in LLM \textBF{F}ine-\textBF{T}uning），这是一种新型防御技术，通过利用有影响力的数据选择和可调整参数来平衡效用保护和隐私保护来减轻隐私泄露。我们广泛的实验跨越六个不同领域和多种LLM架构和规模。结果表明，SOFT有效降低了隐私风险，同时保持了有竞争力的模型性能，提供了实用且可扩展的解决方案来保护微调的LLM中的敏感信息。



## **18. Can We Infer Confidential Properties of Training Data from LLMs?**

我们可以从LLM推断培训数据的机密属性吗？ cs.LG

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10364v1) [paper-pdf](http://arxiv.org/pdf/2506.10364v1)

**Authors**: Penguin Huang, Chhavi Yadav, Ruihan Wu, Kamalika Chaudhuri

**Abstract**: Large language models (LLMs) are increasingly fine-tuned on domain-specific datasets to support applications in fields such as healthcare, finance, and law. These fine-tuning datasets often have sensitive and confidential dataset-level properties -- such as patient demographics or disease prevalence -- that are not intended to be revealed. While prior work has studied property inference attacks on discriminative models (e.g., image classification models) and generative models (e.g., GANs for image data), it remains unclear if such attacks transfer to LLMs. In this work, we introduce PropInfer, a benchmark task for evaluating property inference in LLMs under two fine-tuning paradigms: question-answering and chat-completion. Built on the ChatDoctor dataset, our benchmark includes a range of property types and task configurations. We further propose two tailored attacks: a prompt-based generation attack and a shadow-model attack leveraging word frequency signals. Empirical evaluations across multiple pretrained LLMs show the success of our attacks, revealing a previously unrecognized vulnerability in LLMs.

摘要: 大型语言模型（LLM）越来越多地针对特定领域的数据集进行微调，以支持医疗保健、金融和法律等领域的应用。这些微调数据集通常具有敏感且保密的厕所级属性（例如患者人口统计数据或疾病患病率），这些属性不打算被披露。虽然之前的工作研究了对区分模型的属性推断攻击（例如，图像分类模型）和生成模型（例如，图像数据的GAN），目前尚不清楚此类攻击是否转移到LLM。在这项工作中，我们引入了PropInfer，这是一项基准任务，用于在两种微调范式下评估LLM中的属性推理：问答和聊天完成。我们的基准基于ChatDoctor数据集构建，包括一系列属性类型和任务配置。我们进一步提出了两种定制攻击：基于预算的生成攻击和利用词频信号的影子模型攻击。对多个预训练的LLM进行的经验评估显示了我们的攻击的成功，揭示了LLM中以前未被识别的漏洞。



## **19. Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices**

保护大型语言模型：威胁、漏洞和负责任的实践 cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2403.12503v2) [paper-pdf](http://arxiv.org/pdf/2403.12503v2)

**Authors**: Sara Abdali, Richard Anarfi, CJ Barberan, Jia He, Erfan Shayegani

**Abstract**: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.

摘要: 大型语言模型（LLM）显着改变了自然语言处理（NLP）的格局。它们的影响范围涵盖了各种各样的任务，彻底改变了我们对语言理解和生成的方式。然而，除了具有非凡的实用性之外，LLM还引入了关键的安全和风险考虑因素。这些挑战值得仔细检查，以确保负责任的部署并防范潜在漏洞。本研究论文从五个主题角度彻底调查了与LLM相关的安全和隐私问题：安全和隐私问题、对抗性攻击的漏洞、滥用LLM造成的潜在伤害、应对这些挑战的缓解策略，同时识别当前策略的局限性。最后，本文为未来的研究推荐了有希望的途径，以增强LLC的安全性和风险管理。



## **20. AURA: A Multi-Agent Intelligence Framework for Knowledge-Enhanced Cyber Threat Attribution**

AURA：一个用于知识增强型网络威胁归因的多智能体情报框架 cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.10175v1) [paper-pdf](http://arxiv.org/pdf/2506.10175v1)

**Authors**: Nanda Rani, Sandeep Kumar Shukla

**Abstract**: Effective attribution of Advanced Persistent Threats (APTs) increasingly hinges on the ability to correlate behavioral patterns and reason over complex, varied threat intelligence artifacts. We present AURA (Attribution Using Retrieval-Augmented Agents), a multi-agent, knowledge-enhanced framework for automated and interpretable APT attribution. AURA ingests diverse threat data including Tactics, Techniques, and Procedures (TTPs), Indicators of Compromise (IoCs), malware details, adversarial tools, and temporal information, which are processed through a network of collaborative agents. These agents are designed for intelligent query rewriting, context-enriched retrieval from structured threat knowledge bases, and natural language justification of attribution decisions. By combining Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs), AURA enables contextual linking of threat behaviors to known APT groups and supports traceable reasoning across multiple attack phases. Experiments on recent APT campaigns demonstrate AURA's high attribution consistency, expert-aligned justifications, and scalability. This work establishes AURA as a promising direction for advancing transparent, data-driven, and scalable threat attribution using multi-agent intelligence.

摘要: 高级持续威胁（APT）的有效归因越来越依赖于将行为模式和推理与复杂、多样化的威胁情报制品相关联的能力。我们提出了AURA（使用检索增强代理的归因），这是一个用于自动化和可解释APT归因的多代理、知识增强框架。AURA吸收各种威胁数据，包括战术、技术和程序（TTP）、妥协指标（IoC）、恶意软件详细信息、对抗工具和临时信息，这些数据通过协作代理网络进行处理。这些代理旨在智能查询重写、从结构化威胁知识库进行上下文丰富检索以及属性决策的自然语言证明。通过将检索增强生成（RAG）与大型语言模型（LLM）相结合，AURA能够将威胁行为与已知的APT组进行上下文链接，并支持跨多个攻击阶段的可追溯推理。最近的APT活动的实验证明AURA的高归因一致性，专家对齐的理由，和可扩展性。这项工作将AURA确立为使用多智能体智能推进透明，数据驱动和可扩展的威胁归因的有前途的方向。



## **21. DAWN: Designing Distributed Agents in a Worldwide Network**

DAWN：在全球网络中设计分布式代理 cs.NI

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2410.22339v3) [paper-pdf](http://arxiv.org/pdf/2410.22339v3)

**Authors**: Zahra Aminiranjbar, Jianan Tang, Qiudan Wang, Shubha Pant, Mahesh Viswanathan

**Abstract**: The rapid evolution of Large Language Models (LLMs) has transformed them from basic conversational tools into sophisticated entities capable of complex reasoning and decision-making. These advancements have led to the development of specialized LLM-based agents designed for diverse tasks such as coding and web browsing. As these agents become more capable, the need for a robust framework that facilitates global communication and collaboration among them towards advanced objectives has become increasingly critical. Distributed Agents in a Worldwide Network (DAWN) addresses this need by offering a versatile framework that integrates LLM-based agents with traditional software systems, enabling the creation of agentic applications suited for a wide range of use cases. DAWN enables distributed agents worldwide to register and be easily discovered through Gateway Agents. Collaborations among these agents are coordinated by a Principal Agent equipped with reasoning strategies. DAWN offers three operational modes: No-LLM Mode for deterministic tasks, Copilot for augmented decision-making, and LLM Agent for autonomous operations. Additionally, DAWN ensures the safety and security of agent collaborations globally through a dedicated safety, security, and compliance layer, protecting the network against attackers and adhering to stringent security and compliance standards. These features make DAWN a robust network for deploying agent-based applications across various industries.

摘要: 大型语言模型（LLM）的快速发展已将它们从基本的对话工具转变为能够进行复杂推理和决策的复杂实体。这些进步导致了专门的基于LLM的代理的开发，这些代理专为编码和网络浏览等多样化任务而设计。随着这些代理人的能力变得越来越强，对一个强大的框架的需求，以促进他们之间的全球沟通和协作，以实现先进目标变得越来越重要。全球网络中的分布式代理（DAWN）通过提供将基于LLM的代理与传统软件系统集成的通用框架来满足这一需求，从而能够创建适合广泛用例的代理应用程序。DAWN使世界各地的分布式代理能够注册并通过网关代理轻松发现。这些代理之间的合作由配备推理策略的主代理协调。DAWN提供三种操作模式：用于确定性任务的No-LLM模式、用于增强决策的副驾驶员和用于自主操作的LLM Agent。此外，DAWN通过专门的安全、安全和合规层确保全球代理协作的安全性，保护网络免受攻击者的侵害并遵守严格的安全和合规标准。这些功能使DAWN成为一个强大的网络，用于在各个行业部署基于代理的应用程序。



## **22. Trustworthy AI: Safety, Bias, and Privacy -- A Survey**

值得信赖的人工智能：安全、偏见和隐私--一项调查 cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2502.10450v2) [paper-pdf](http://arxiv.org/pdf/2502.10450v2)

**Authors**: Xingli Fang, Jianwei Li, Varun Mulchandani, Jung-Eun Kim

**Abstract**: The capabilities of artificial intelligence systems have been advancing to a great extent, but these systems still struggle with failure modes, vulnerabilities, and biases. In this paper, we study the current state of the field, and present promising insights and perspectives regarding concerns that challenge the trustworthiness of AI models. In particular, this paper investigates the issues regarding three thrusts: safety, privacy, and bias, which hurt models' trustworthiness. For safety, we discuss safety alignment in the context of large language models, preventing them from generating toxic or harmful content. For bias, we focus on spurious biases that can mislead a network. Lastly, for privacy, we cover membership inference attacks in deep neural networks. The discussions addressed in this paper reflect our own experiments and observations.

摘要: 人工智能系统的能力已经在很大程度上进步，但这些系统仍然在与故障模式、漏洞和偏见作斗争。在本文中，我们研究了该领域的现状，并就挑战人工智能模型可信度的问题提出了有希望的见解和观点。特别是，本文调查了有关三个目标的问题：安全、隐私和偏见，这些目标损害了模型的可信度。为了安全，我们讨论了大型语言模型背景下的安全对齐，防止它们生成有毒或有害内容。对于偏见，我们关注可能误导网络的虚假偏见。最后，为了隐私，我们涵盖了深度神经网络中的成员推断攻击。本文中讨论的讨论反映了我们自己的实验和观察。



## **23. LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge**

LL Mail-Injects：来自现实自适应提示注入挑战的数据集 cs.CR

Dataset at:  https://huggingface.co/datasets/microsoft/llmail-inject-challenge

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09956v1) [paper-pdf](http://arxiv.org/pdf/2506.09956v1)

**Authors**: Sahar Abdelnabi, Aideen Fay, Ahmed Salem, Egor Zverev, Kai-Chieh Liao, Chi-Huang Liu, Chun-Chih Kuo, Jannis Weigend, Danyael Manlangit, Alex Apostolov, Haris Umair, João Donato, Masayuki Kawakita, Athar Mahboob, Tran Huu Bach, Tsun-Han Chiang, Myeongjin Cho, Hajin Choi, Byeonghyeon Kim, Hyeonjin Lee, Benjamin Pannell, Conor McCauley, Mark Russinovich, Andrew Paverd, Giovanni Cherubin

**Abstract**: Indirect Prompt Injection attacks exploit the inherent limitation of Large Language Models (LLMs) to distinguish between instructions and data in their inputs. Despite numerous defense proposals, the systematic evaluation against adaptive adversaries remains limited, even when successful attacks can have wide security and privacy implications, and many real-world LLM-based applications remain vulnerable. We present the results of LLMail-Inject, a public challenge simulating a realistic scenario in which participants adaptively attempted to inject malicious instructions into emails in order to trigger unauthorized tool calls in an LLM-based email assistant. The challenge spanned multiple defense strategies, LLM architectures, and retrieval configurations, resulting in a dataset of 208,095 unique attack submissions from 839 participants. We release the challenge code, the full dataset of submissions, and our analysis demonstrating how this data can provide new insights into the instruction-data separation problem. We hope this will serve as a foundation for future research towards practical structural solutions to prompt injection.

摘要: 间接提示注入攻击利用大型语言模型（LLM）的固有限制来区分其输入中的指令和数据。尽管有许多防御提案，但针对自适应对手的系统评估仍然有限，即使成功的攻击可能会产生广泛的安全和隐私影响，并且许多基于现实世界的LLM应用程序仍然容易受到攻击。我们展示了LLMail-Injects的结果，这是一个模拟现实场景的公开挑战，其中参与者自适应地尝试将恶意指令注入电子邮件中，以在基于LLM的电子邮件助手中触发未经授权的工具调用。该挑战涵盖多种防御策略、LLM架构和检索配置，产生了来自839名参与者的208，095份独特攻击提交的数据集。我们发布了挑战代码、完整的提交数据集以及我们的分析，展示了这些数据如何为描述-数据分离问题提供新的见解。我们希望这将成为未来研究的基础，以推动注入的实用结构性解决方案。



## **24. CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization**

CROW：通过内部一致性规范化消除大型语言模型的后门 cs.CL

Accepted at ICML 2025, 20 pages

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2411.12768v2) [paper-pdf](http://arxiv.org/pdf/2411.12768v2)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Large Language Models (LLMs) are vulnerable to backdoor attacks that manipulate outputs via hidden triggers. Existing defense methods--designed for vision/text classification tasks--fail for text generation. We propose Internal Consistency Regularization (CROW), a defense leveraging the observation that backdoored models exhibit unstable layer-wise hidden representations when triggered, while clean models show smooth transitions. CROW enforces consistency across layers via adversarial perturbations and regularization during finetuning, neutralizing backdoors without requiring clean reference models or trigger knowledge--only a small clean dataset. Experiments across Llama-2 (7B, 13B), CodeLlama (7B, 13B), and Mistral-7B demonstrate CROW's effectiveness: it achieves significant reductions in attack success rates across diverse backdoor strategies (sentiment steering, targeted refusal, code injection) while preserving generative performance. CROW's architecture-agnostic design enables practical deployment.

摘要: 大型语言模型（LLM）容易受到后门攻击，这些攻击通过隐藏触发器操纵输出。现有的防御方法（专为视觉/文本分类任务设计）无法生成文本。我们提出了内部一致性正规化（CROW），这是一种利用以下观察结果的防御，即后门模型在触发时表现出不稳定的分层隐藏表示，而干净模型则表现出平滑的过渡。CROW在微调期间通过对抗性扰动和正规化来强制跨层的一致性，中和后门，而不需要干净的参考模型或触发知识--只需一个小的干净数据集。Llama-2（7 B，13 B）、CodeLlama（7 B，13 B）和Mistral-7 B的实验证明了CROW的有效性：它在各种后门策略（情绪引导、定向拒绝、代码注入）上显着降低攻击成功率，同时保持生成性能。CROW的架构不可知设计可以实现实际部署。



## **25. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

RSafe：激励积极推理，以建立强大且自适应的LLM保障措施 cs.AI

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.07736v2) [paper-pdf](http://arxiv.org/pdf/2506.07736v2)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.

摘要: 尽管采取了刻意的安全调整措施，大型语言模型（LLM）仍然表现出漏洞，给用户和社会带来了重大风险。为了防范违反政策内容的风险，通过外部防护模型进行系统级审核（旨在监控LLM输入和输出并阻止潜在有害内容）已成为一种流行的缓解策略。训练警卫模型的现有方法严重依赖于大量的人类策划的数据集，并与分发外威胁作斗争，例如新出现的有害类别或越狱攻击。为了解决这些限制，我们提出RSafe，这是一种基于自适应推理的保护措施，它进行引导式安全推理，以在指定安全政策范围内提供强有力的保护。RSafe分两个阶段运行：1）引导推理，通过政策引导的分步推理来分析输入内容的安全风险，2）强化对齐，基于规则的RL优化其推理路径以与准确的安全预测保持一致。这种两阶段培训范式使RSafe能够内化安全原则，以概括针对不可见或对抗性安全违规场景的安全保护能力。在推理过程中，RSafe接受用户指定的安全政策，以提供针对特定安全要求的增强的保障措施。



## **26. Design Patterns for Securing LLM Agents against Prompt Injections**

保护LLM代理免受即时注射的设计模式 cs.LG

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.08837v2) [paper-pdf](http://arxiv.org/pdf/2506.08837v2)

**Authors**: Luca Beurer-Kellner, Beat Buesser Ana-Maria Creţu, Edoardo Debenedetti, Daniel Dobos, Daniel Fabian, Marc Fischer, David Froelicher, Kathrin Grosse, Daniel Naeff, Ezinwanne Ozoani, Andrew Paverd, Florian Tramèr, Václav Volhejn

**Abstract**: As AI agents powered by Large Language Models (LLMs) become increasingly versatile and capable of addressing a broad spectrum of tasks, ensuring their security has become a critical challenge. Among the most pressing threats are prompt injection attacks, which exploit the agent's resilience on natural language inputs -- an especially dangerous threat when agents are granted tool access or handle sensitive information. In this work, we propose a set of principled design patterns for building AI agents with provable resistance to prompt injection. We systematically analyze these patterns, discuss their trade-offs in terms of utility and security, and illustrate their real-world applicability through a series of case studies.

摘要: 随着由大型语言模型（LLM）支持的AI代理变得越来越多才多艺，能够解决广泛的任务，确保其安全性已成为一项关键挑战。最紧迫的威胁之一是即时注入攻击，它利用代理对自然语言输入的弹性-当代理被授予工具访问或处理敏感信息时，这是一个特别危险的威胁。在这项工作中，我们提出了一套原则性的设计模式，用于构建具有可证明的即时注入阻力的AI代理。我们系统地分析了这些模式，讨论了它们在实用性和安全性方面的权衡，并通过一系列案例研究说明了它们在现实世界中的适用性。



## **27. GenBreak: Red Teaming Text-to-Image Generators Using Large Language Models**

GenBreak：使用大型语言模型的Red协作文本到图像生成器 cs.CR

27 pages, 7 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.10047v1) [paper-pdf](http://arxiv.org/pdf/2506.10047v1)

**Authors**: Zilong Wang, Xiang Zheng, Xiaosen Wang, Bo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Text-to-image (T2I) models such as Stable Diffusion have advanced rapidly and are now widely used in content creation. However, these models can be misused to generate harmful content, including nudity or violence, posing significant safety risks. While most platforms employ content moderation systems, underlying vulnerabilities can still be exploited by determined adversaries. Recent research on red-teaming and adversarial attacks against T2I models has notable limitations: some studies successfully generate highly toxic images but use adversarial prompts that are easily detected and blocked by safety filters, while others focus on bypassing safety mechanisms but fail to produce genuinely harmful outputs, neglecting the discovery of truly high-risk prompts. Consequently, there remains a lack of reliable tools for evaluating the safety of defended T2I models. To address this gap, we propose GenBreak, a framework that fine-tunes a red-team large language model (LLM) to systematically explore underlying vulnerabilities in T2I generators. Our approach combines supervised fine-tuning on curated datasets with reinforcement learning via interaction with a surrogate T2I model. By integrating multiple reward signals, we guide the LLM to craft adversarial prompts that enhance both evasion capability and image toxicity, while maintaining semantic coherence and diversity. These prompts demonstrate strong effectiveness in black-box attacks against commercial T2I generators, revealing practical and concerning safety weaknesses.

摘要: 稳定扩散等文本到图像（T2 I）模型发展迅速，现已广泛用于内容创建。然而，这些模型可能会被滥用来生成有害内容，包括裸体或暴力，从而构成重大安全风险。虽然大多数平台都采用内容审核系统，但潜在漏洞仍然可能被坚定的对手利用。最近关于针对T2 I模型的红色组队和对抗性攻击的研究存在显着的局限性：一些研究成功地生成了剧毒图像，但使用了容易被安全过滤器检测和阻止的对抗性提示，而另一些研究则专注于绕过安全机制，但未能产生真正有害的输出，忽视了真正高风险提示的发现。因此，仍然缺乏可靠的工具来评估防御T2 I模型的安全性。为了解决这一差距，我们提出了GenBreak，这是一个对红队大型语言模型（LLM）进行微调的框架，以系统性地探索T2 I生成器中的潜在漏洞。我们的方法将对策划数据集的监督微调与通过与替代T2 I模型交互的强化学习相结合。通过集成多个奖励信号，我们引导LLM设计对抗性提示，以增强规避能力和图像毒性，同时保持语义一致性和多样性。这些提示在针对商用T2 I发生器的黑匣子攻击中表现出强大的有效性，揭示了实际且令人担忧的安全弱点。



## **28. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

MCA-Bench：评估CAPTCHA针对基于VLM的攻击的稳健性的多模式基准 cs.CV

31 pages, 8 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.05982v2) [paper-pdf](http://arxiv.org/pdf/2506.05982v2)

**Authors**: Zonglin Wu, Yule Xue, Xin Wei, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.

摘要: 随着自动攻击技术的迅速发展，验证码仍然是针对恶意机器人的重要防御机制。然而，现有的CAPTCHA方案涵盖了多种形式--从静态扭曲文本和模糊图像到交互式点击、滑动谜题和基于逻辑的问题--但社区仍然缺乏统一的、大规模的、多模式基准来严格评估其安全稳健性。为了解决这一差距，我们引入了MCA-Bench，这是一个全面且可重复的基准测试套件，可将异类CAPTCHA类型集成到单个评估协议中。利用共享的视觉语言模型主干，我们为每个CAPTCHA类别微调专门的破解剂，实现一致的跨模式评估。大量实验表明，MCA-Bench有效地绘制了现代CAPTCHA设计在不同攻击环境下的脆弱性谱，并且至关重要地提供了挑战复杂性、交互深度和模型可解性如何相互关联的首次定量分析。基于这些发现，我们提出了三项可操作的设计原则，并确定了关键的开放挑战，为系统性CAPTCHA强化、公平的基准测试和更广泛的社区合作奠定了基础。数据集和代码可在线获取。



## **29. LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge**

LLM无法可靠地判断（还吗？）：法学硕士作为法官稳健性的综合评估 cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09443v1) [paper-pdf](http://arxiv.org/pdf/2506.09443v1)

**Authors**: Songze Li, Chuokun Xu, Jiaying Wang, Xueluan Gong, Chen Chen, Jirui Zhang, Jun Wang, Kwok-Yan Lam, Shouling Ji

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable intelligence across various tasks, which has inspired the development and widespread adoption of LLM-as-a-Judge systems for automated model testing, such as red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising concerns about their robustness and, consequently, their trustworthiness. Existing evaluation methods adopted by LLM-based judges are often piecemeal and lack a unified framework for comprehensive assessment. Furthermore, prompt template and model selections for improving judge robustness have been rarely explored, and their performance in real-world settings remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. RobustJudge investigates the impact of attack methods and defense strategies (RQ1), explores the influence of prompt template and model selection (RQ2), and assesses the robustness of real-world LLM-as-a-Judge applications (RQ3).Our main findings are: (1) LLM-as-a-Judge systems are still vulnerable to a range of adversarial attacks, including Combined Attack and PAIR, while defense mechanisms such as Re-tokenization and LLM-based Detectors offer improved protection; (2) Robustness is highly sensitive to the choice of prompt template and judge models. Our proposed prompt template optimization method can improve robustness, and JudgeLM-13B demonstrates strong performance as a robust open-source judge; (3) Applying RobustJudge to Alibaba's PAI platform reveals previously unreported vulnerabilities. The source code of RobustJudge is provided at https://github.com/S3IC-Lab/RobustJudge.

摘要: 大型语言模型（LLM）在各种任务中表现出了非凡的智能，这激发了LLM作为法官系统的开发和广泛采用，用于自动化模型测试，例如红色团队和基准测试。然而，这些系统很容易受到对抗攻击，这些攻击可以操纵评估结果，从而引发人们对其稳健性的担忧，从而对其可信度。LLM法官采用的现有评估方法往往是零碎的，缺乏统一的综合评估框架。此外，很少探索用于提高判断稳健性的提示模板和模型选择，而且它们在现实世界环境中的性能在很大程度上仍然未经验证。为了解决这些差距，我们引入了RobustJudge，这是一个全自动化和可扩展的框架，旨在系统性评估法学硕士即法官系统的稳健性。RobustJudge调查攻击方法和防御策略的影响（MQ 1），探索提示模板和模型选择的影响（MQ 2），并评估现实世界的LLM作为法官应用程序的稳健性（MQ 3）。我们的主要发现是：（1）法学硕士作为法官系统仍然容易受到一系列对抗攻击，包括联合攻击和PAIR，而重新标记化和基于LLM的检测器等防御机制提供了更好的保护;（2）鲁棒性对提示模板和判断模型的选择高度敏感。我们提出的提示模板优化方法可以提高稳健性，JudggeLM-13 B作为稳健的开源法官表现出了强大的性能;（3）将RobustJudge应用于阿里巴巴的PRI平台，揭示了之前未报告的漏洞。RobustJudge的源代码可访问https://github.com/S3IC-Lab/RobustJudge。



## **30. Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding**

代码转换红色团队：LLM评估安全性和多语言理解 cs.AI

To appear in ACL 2025

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2406.15481v3) [paper-pdf](http://arxiv.org/pdf/2406.15481v3)

**Authors**: Haneul Yoo, Yongjin Yang, Hwaran Lee

**Abstract**: As large language models (LLMs) have advanced rapidly, concerns regarding their safety have become prominent. In this paper, we discover that code-switching in red-teaming queries can effectively elicit undesirable behaviors of LLMs, which are common practices in natural language. We introduce a simple yet effective framework, CSRT, to synthesize codeswitching red-teaming queries and investigate the safety and multilingual understanding of LLMs comprehensively. Through extensive experiments with ten state-of-the-art LLMs and code-switching queries combining up to 10 languages, we demonstrate that the CSRT significantly outperforms existing multilingual red-teaming techniques, achieving 46.7% more attacks than standard attacks in English and being effective in conventional safety domains. We also examine the multilingual ability of those LLMs to generate and understand codeswitching texts. Additionally, we validate the extensibility of the CSRT by generating codeswitching attack prompts with monolingual data. We finally conduct detailed ablation studies exploring code-switching and propound unintended correlation between resource availability of languages and safety alignment in existing multilingual LLMs.

摘要: 随着大型语言模型（LLM）的迅速发展，对其安全性的担忧变得突出。在本文中，我们发现红色团队查询中的代码切换可以有效地引发LLM的不良行为，这是自然语言中的常见做法。我们引入了一个简单而有效的框架CSRT来合成代码交换红组查询，并全面调查LLM的安全性和多语言理解。通过对10种最先进的LLM和结合多达10种语言的代码切换查询的广泛实验，我们证明CSRT的性能显着优于现有的多语言红组技术，比英语中的标准攻击多46.7%，并且在传统安全领域有效。我们还考察了这些LLM生成和理解代码交换文本的多语言能力。此外，我们还通过使用单语数据生成代码交换攻击提示来验证CSRT的可扩展性。我们最终进行了详细的消融研究，探索代码转换，并提出了语言资源可用性和现有多语言LLM中的安全一致之间的意想不到的相关性。



## **31. Automatic Pseudo-Harmful Prompt Generation for Evaluating False Refusals in Large Language Models**

用于评估大型语言模型中虚假拒绝的自动伪有害提示生成 cs.CL

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2409.00598v2) [paper-pdf](http://arxiv.org/pdf/2409.00598v2)

**Authors**: Bang An, Sicheng Zhu, Ruiyi Zhang, Michael-Andrei Panaitescu-Liess, Yuancheng Xu, Furong Huang

**Abstract**: Safety-aligned large language models (LLMs) sometimes falsely refuse pseudo-harmful prompts, like "how to kill a mosquito," which are actually harmless. Frequent false refusals not only frustrate users but also provoke a public backlash against the very values alignment seeks to protect. In this paper, we propose the first method to auto-generate diverse, content-controlled, and model-dependent pseudo-harmful prompts. Using this method, we construct an evaluation dataset called PHTest, which is ten times larger than existing datasets, covers more false refusal patterns, and separately labels controversial prompts. We evaluate 20 LLMs on PHTest, uncovering new insights due to its scale and labeling. Our findings reveal a trade-off between minimizing false refusals and improving safety against jailbreak attacks. Moreover, we show that many jailbreak defenses significantly increase the false refusal rates, thereby undermining usability. Our method and dataset can help developers evaluate and fine-tune safer and more usable LLMs. Our code and dataset are available at https://github.com/umd-huang-lab/FalseRefusal

摘要: 安全对齐的大型语言模型（LLM）有时会错误地拒绝伪有害提示，例如“如何杀死蚊子”，而这些提示实际上是无害的。频繁的虚假拒绝不仅会让用户感到沮丧，还会引发公众对联盟所寻求保护的价值观的强烈反对。在本文中，我们提出了第一种自动生成多样化、内容控制且依赖模型的伪有害提示的方法。使用这种方法，我们构建了一个名为PHTest的评估数据集，它比现有数据集大十倍，涵盖了更多的虚假拒绝模式，并单独标记了有争议的提示。我们在PHTest上评估了20个LLM，并因其规模和标签而发现了新的见解。我们的研究结果揭示了最大限度地减少虚假拒绝和提高针对越狱攻击的安全性之间的权衡。此外，我们表明许多越狱防御显着增加了错误拒绝率，从而削弱了可用性。我们的方法和数据集可以帮助开发人员评估和微调更安全、更可用的LLM。我们的代码和数据集可在https://github.com/umd-huang-lab/FalseRefusal上获取



## **32. Detecting State Manipulation Vulnerabilities in Smart Contracts Using LLM and Static Analysis**

使用LLM和静态分析检测智能合约中的状态操纵漏洞 cs.SE

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.08561v2) [paper-pdf](http://arxiv.org/pdf/2506.08561v2)

**Authors**: Hao Wu, Haijun Wang, Shangwang Li, Yin Wu, Ming Fan, Yitao Zhao, Ting Liu

**Abstract**: An increasing number of DeFi protocols are gaining popularity, facilitating transactions among multiple anonymous users. State Manipulation is one of the notorious attacks in DeFi smart contracts, with price variable being the most commonly exploited state variable-attackers manipulate token prices to gain illicit profits. In this paper, we propose PriceSleuth, a novel method that leverages the Large Language Model (LLM) and static analysis to detect Price Manipulation (PM) attacks proactively. PriceSleuth firstly identifies core logic function related to price calculation in DeFi contracts. Then it guides LLM to locate the price calculation code statements. Secondly, PriceSleuth performs backward dependency analysis of price variables, instructing LLM in detecting potential price manipulation. Finally, PriceSleuth utilizes propagation analysis of price variables to assist LLM in detecting whether these variables are maliciously exploited. We presented preliminary experimental results to substantiate the effectiveness of PriceSleuth . And we outline future research directions for PriceSleuth.

摘要: 越来越多的DeFi协议越来越受欢迎，促进了多个匿名用户之间的交易。状态操纵是DeFi智能合同中臭名昭著的攻击之一，价格变量是最常被利用的状态变量--攻击者操纵代币价格以获取非法利润。在本文中，我们提出了PriceSleuth，这是一种利用大型语言模型（LLM）和静态分析来主动检测价格操纵（PM）攻击的新颖方法。PriceSleuth首先确定了DeFi合同中与价格计算相关的核心逻辑功能。然后引导LLM定位价格计算代码报表。其次，PriceSleuth对价格变量进行向后依赖分析，指导LLM检测潜在的价格操纵。最后，PriceSleuth利用价格变量的传播分析来协助LLM检测这些变量是否被恶意利用。我们提供了初步实验结果来证实PriceSleuth的有效性。我们还概述了PriceSleuth未来的研究方向。



## **33. Your Agent Can Defend Itself against Backdoor Attacks**

您的代理可以保护自己免受后门攻击 cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.08336v2) [paper-pdf](http://arxiv.org/pdf/2506.08336v2)

**Authors**: Li Changjiang, Liang Jiacheng, Cao Bochuan, Chen Jinghui, Wang Ting

**Abstract**: Despite their growing adoption across domains, large language model (LLM)-powered agents face significant security risks from backdoor attacks during training and fine-tuning. These compromised agents can subsequently be manipulated to execute malicious operations when presented with specific triggers in their inputs or environments. To address this pressing risk, we present ReAgent, a novel defense against a range of backdoor attacks on LLM-based agents. Intuitively, backdoor attacks often result in inconsistencies among the user's instruction, the agent's planning, and its execution. Drawing on this insight, ReAgent employs a two-level approach to detect potential backdoors. At the execution level, ReAgent verifies consistency between the agent's thoughts and actions; at the planning level, ReAgent leverages the agent's capability to reconstruct the instruction based on its thought trajectory, checking for consistency between the reconstructed instruction and the user's instruction. Extensive evaluation demonstrates ReAgent's effectiveness against various backdoor attacks across tasks. For instance, ReAgent reduces the attack success rate by up to 90\% in database operation tasks, outperforming existing defenses by large margins. This work reveals the potential of utilizing compromised agents themselves to mitigate backdoor risks.

摘要: 尽管大型语言模型（LLM）支持的代理在各个领域的采用越来越多，但在培训和微调期间仍面临着后门攻击的重大安全风险。当这些受影响的代理的输入或环境中出现特定触发器时，随后可以操纵这些受影响的代理执行恶意操作。为了解决这一紧迫的风险，我们提出了ReAgent，这是一种针对基于LLM的代理的一系列后门攻击的新型防御措施。直观地说，后门攻击通常会导致用户的指令、代理的规划和执行之间的不一致。利用这一洞察力，ReAgent采用两级方法来检测潜在的后门。在执行层，ReAgent验证Agent的思想和动作之间的一致性;在规划层，ReAgent利用Agent的能力，根据其思想轨迹重建指令，检查重建的指令和用户指令之间的一致性。广泛的评估表明，ReAgent的有效性对各种后门攻击跨任务。例如，ReAgent在数据库操作任务中将攻击成功率降低了90%，大大优于现有的防御措施。这项工作揭示了利用受损代理本身来减轻后门风险的潜力。



## **34. FC-Attack: Jailbreaking Multimodal Large Language Models via Auto-Generated Flowcharts**

FC攻击：通过自动生成流程图破解多模式大型语言模型 cs.CV

13 pages, 7 figures

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2502.21059v2) [paper-pdf](http://arxiv.org/pdf/2502.21059v2)

**Authors**: Ziyi Zhang, Zhen Sun, Zongmin Zhang, Jihui Guo, Xinlei He

**Abstract**: Multimodal Large Language Models (MLLMs) have become powerful and widely adopted in some practical applications. However, recent research has revealed their vulnerability to multimodal jailbreak attacks, whereby the model can be induced to generate harmful content, leading to safety risks. Although most MLLMs have undergone safety alignment, recent research shows that the visual modality is still vulnerable to jailbreak attacks. In our work, we discover that by using flowcharts with partially harmful information, MLLMs can be induced to provide additional harmful details. Based on this, we propose a jailbreak attack method based on auto-generated flowcharts, FC-Attack. Specifically, FC-Attack first fine-tunes a pre-trained LLM to create a step-description generator based on benign datasets. The generator is then used to produce step descriptions corresponding to a harmful query, which are transformed into flowcharts in 3 different shapes (vertical, horizontal, and S-shaped) as visual prompts. These flowcharts are then combined with a benign textual prompt to execute the jailbreak attack on MLLMs. Our evaluations on Advbench show that FC-Attack attains an attack success rate of up to 96% via images and up to 78% via videos across multiple MLLMs. Additionally, we investigate factors affecting the attack performance, including the number of steps and the font styles in the flowcharts. We also find that FC-Attack can improve the jailbreak performance from 4% to 28% in Claude-3.5 by changing the font style. To mitigate the attack, we explore several defenses and find that AdaShield can largely reduce the jailbreak performance but with the cost of utility drop.

摘要: 多模式大型语言模型（MLLM）已变得强大并在一些实际应用中广泛采用。然而，最近的研究揭示了它们对多模式越狱攻击的脆弱性，从而可以诱导模型生成有害内容，从而导致安全风险。尽管大多数MLLM都经历了安全调整，但最近的研究表明，视觉模式仍然容易受到越狱攻击。在我们的工作中，我们发现通过使用包含部分有害信息的流程图，可能会诱导MLLM提供额外的有害细节。基于此，我们提出了一种基于自动生成流程图的越狱攻击方法FC-Attack。具体来说，FC-Attack首先微调预训练的LLM，以基于良性数据集创建步骤描述生成器。然后使用生成器生成与有害查询对应的步骤描述，并将其转换为3种不同形状（垂直、水平和S形）的流程图作为视觉提示。然后，这些流程图与良性文本提示相结合，对MLLM执行越狱攻击。我们在Advbench上的评估显示，FC-Attack通过图像实现的攻击成功率高达96%，通过多个MLLM的视频实现的攻击成功率高达78%。此外，我们调查影响攻击性能的因素，包括步骤的数量和字体样式的流程图。我们还发现，FC-Attack可以通过改变字体样式将Claude-3.5的越狱性能从4%提高到28%。为了减轻这种攻击，我们研究了几种防御方法，发现AdaShield可以大大降低越狱性能，但代价是效用下降。



## **35. PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**

PEFTGuard：检测后门攻击对抗参数高效微调 cs.CR

21 pages, 7 figures

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2411.17453v2) [paper-pdf](http://arxiv.org/pdf/2411.17453v2)

**Authors**: Zhen Sun, Tianshuo Cong, Yule Liu, Chenhao Lin, Xinlei He, Rongmao Chen, Xingshuo Han, Xinyi Huang

**Abstract**: Fine-tuning is an essential process to improve the performance of Large Language Models (LLMs) in specific domains, with Parameter-Efficient Fine-Tuning (PEFT) gaining popularity due to its capacity to reduce computational demands through the integration of low-rank adapters. These lightweight adapters, such as LoRA, can be shared and utilized on open-source platforms. However, adversaries could exploit this mechanism to inject backdoors into these adapters, resulting in malicious behaviors like incorrect or harmful outputs, which pose serious security risks to the community. Unfortunately, few current efforts concentrate on analyzing the backdoor patterns or detecting the backdoors in the adapters. To fill this gap, we first construct and release PADBench, a comprehensive benchmark that contains 13,300 benign and backdoored adapters fine-tuned with various datasets, attack strategies, PEFT methods, and LLMs. Moreover, we propose PEFTGuard, the first backdoor detection framework against PEFT-based adapters. Extensive evaluation upon PADBench shows that PEFTGuard outperforms existing detection methods, achieving nearly perfect detection accuracy (100%) in most cases. Notably, PEFTGuard exhibits zero-shot transferability on three aspects, including different attacks, PEFT methods, and adapter ranks. In addition, we consider various adaptive attacks to demonstrate the high robustness of PEFTGuard. We further explore several possible backdoor mitigation defenses, finding fine-mixing to be the most effective method. We envision that our benchmark and method can shed light on future LLM backdoor detection research.

摘要: 微调是提高特定领域大型语言模型（LLM）性能的重要过程，参数高效微调（PEFT）因其能够通过集成低级适配器来减少计算需求而越来越受欢迎。这些轻量级适配器（例如LoRA）可以在开源平台上共享和使用。然而，对手可能会利用这种机制向这些适配器注入后门，导致不正确或有害输出等恶意行为，从而给社区带来严重的安全风险。不幸的是，目前很少有工作专注于分析后门模式或检测适配器中的后门。为了填补这一空白，我们首先构建并发布PADBench，这是一个全面的基准测试，包含13，300个良性和后门适配器，经过各种数据集、攻击策略、PEFT方法和LLM微调。此外，我们提出了PEFTGuard，第一个后门检测框架对基于PEFT的适配器。对PADBench的广泛评估表明，PEFTGuard优于现有的检测方法，在大多数情况下实现了近乎完美的检测准确率（100%）。值得注意的是，PEFTGuard在三个方面表现出零射击可移植性，包括不同的攻击，PEFT方法和适配器等级。此外，我们还考虑各种自适应攻击来证明PEFTGuard的高稳健性。我们进一步探索了几种可能的后门缓解防御措施，发现精细混合是最有效的方法。我们设想我们的基准和方法可以为未来的LLM后门检测研究提供线索。



## **36. PrisonBreak: Jailbreaking Large Language Models with Fewer Than Twenty-Five Targeted Bit-flips**

Prison Break：越狱大型语言模型，目标位翻转少于25个 cs.CR

Pre-print

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2412.07192v2) [paper-pdf](http://arxiv.org/pdf/2412.07192v2)

**Authors**: Zachary Coalson, Jeonghyun Woo, Yu Sun, Shiyang Chen, Lishan Yang, Prashant Nair, Bo Fang, Sanghyun Hong

**Abstract**: We introduce a new class of attacks on commercial-scale (human-aligned) language models that induce jailbreaking through targeted bitwise corruptions in model parameters. Our adversary can jailbreak billion-parameter language models with fewer than 25 bit-flips in all cases$-$and as few as 5 in some$-$using up to 40$\times$ less bit-flips than existing attacks on computer vision models at least 100$\times$ smaller. Unlike prompt-based jailbreaks, our attack renders these models in memory 'uncensored' at runtime, allowing them to generate harmful responses without any input modifications. Our attack algorithm efficiently identifies target bits to flip, offering up to 20$\times$ more computational efficiency than previous methods. This makes it practical for language models with billions of parameters. We show an end-to-end exploitation of our attack using software-induced fault injection, Rowhammer (RH). Our work examines 56 DRAM RH profiles from DDR4 and LPDDR4X devices with different RH vulnerabilities. We show that our attack can reliably induce jailbreaking in systems similar to those affected by prior bit-flip attacks. Moreover, our approach remains effective even against highly RH-secure systems (e.g., 46$\times$ more secure than previously tested systems). Our analyses further reveal that: (1) models with less post-training alignment require fewer bit flips to jailbreak; (2) certain model components, such as value projection layers, are substantially more vulnerable than others; and (3) our method is mechanistically different than existing jailbreaks. Our findings highlight a pressing, practical threat to the language model ecosystem and underscore the need for research to protect these models from bit-flip attacks.

摘要: 我们对商业规模（与人类一致的）语言模型引入了一类新的攻击，这些攻击通过模型参数中有针对性的逐位破坏来引发越狱。我们的对手可以通过在所有情况下少于25个位翻转来越狱数十亿参数的语言模型，在某些情况下只需只需5个位翻转，比对计算机视觉模型的现有攻击少40美元\x $，至少小100美元\x $。与基于预算的越狱不同，我们的攻击使内存中的这些模型在运行时“未经审查”，使它们能够在无需任何输入修改的情况下生成有害响应。我们的攻击算法有效地识别要翻转的目标位，比之前的方法提供高达20美元\x $的计算效率。这使得具有数十亿个参数的语言模型变得实用。我们展示了使用软件诱导的故障注入Rowhammer（RH）对攻击的端到端利用。我们的工作检查了具有不同RH漏洞的DDR4和LPDDR 4X设备的56个RAM RH配置文件。我们表明，我们的攻击可以可靠地在与受先前位翻转攻击影响的系统类似的系统中引发越狱。此外，即使针对高度RH安全的系统（例如，比之前测试的系统安全46 $\x $）。我们的分析进一步揭示了：（1）训练后对齐较少的模型需要更少的位翻转即可越狱;（2）某些模型组件，例如价值投影层，比其他组件更容易受到攻击;（3）我们的方法在机械上与现有的越狱不同。我们的研究结果凸显了语言模型生态系统面临的紧迫、实际威胁，并强调了研究以保护这些模型免受位翻转攻击的必要性。



## **37. Fighting Fire with Fire (F3): A Training-free and Efficient Visual Adversarial Example Purification Method in LVLMs**

以毒攻毒（F3）：LVLM中一种无需培训且高效的视觉对抗示例净化方法 cs.CV

14 pages, 5 figures

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.01064v2) [paper-pdf](http://arxiv.org/pdf/2506.01064v2)

**Authors**: Yudong Zhang, Ruobing Xie, Yiqing Huang, Jiansheng Chen, Xingwu Sun, Zhanhui Kang, Di Wang, Yu Wang

**Abstract**: Recent advances in large vision-language models (LVLMs) have showcased their remarkable capabilities across a wide range of multimodal vision-language tasks. However, these models remain vulnerable to visual adversarial attacks, which can substantially compromise their performance. Despite their potential impact, the development of effective methods for purifying such adversarial examples has received relatively limited attention. In this paper, we introduce F3, a novel adversarial purification framework that employs a counterintuitive "fighting fire with fire" strategy: intentionally introducing simple perturbations to adversarial examples to mitigate their harmful effects. Specifically, F3 leverages cross-modal attentions derived from randomly perturbed adversary examples as reference targets. By injecting noise into these adversarial examples, F3 effectively refines their attention, resulting in cleaner and more reliable model outputs. Remarkably, this seemingly paradoxical approach of employing noise to counteract adversarial attacks yields impressive purification results. Furthermore, F3 offers several distinct advantages: it is training-free and straightforward to implement, and exhibits significant computational efficiency improvements compared to existing purification methods. These attributes render F3 particularly suitable for large-scale industrial applications where both robust performance and operational efficiency are critical priorities. The code will be made publicly available.

摘要: 大型视觉语言模型（LVLM）的最新进展展示了它们在广泛的多模式视觉语言任务中的非凡能力。然而，这些模型仍然容易受到视觉对抗攻击，这可能会极大地损害其性能。尽管它们具有潜在的影响，但净化此类对抗性例子的有效方法的开发受到的关注相对有限。在本文中，我们介绍了F3，这是一个新颖的对抗净化框架，它采用了违反直觉的“以毒攻毒”策略：有意地向对抗性示例引入简单的扰动以减轻其有害影响。具体来说，F3利用从随机干扰的对手示例中获得的跨模式注意力作为参考目标。通过向这些对抗性示例中注入噪音，F3有效地细化了他们的注意力，从而产生更干净、更可靠的模型输出。值得注意的是，这种看似矛盾的利用噪音来抵消对抗攻击的方法产生了令人印象深刻的净化结果。此外，F3具有几个明显的优势：无需训练且易于实施，并且与现有的纯化方法相比，计算效率显着提高。这些属性使F3特别适合大规模工业应用，其中稳健的性能和运营效率都是关键优先事项。该代码将公开。



## **38. ASIDE: Architectural Separation of Instructions and Data in Language Models**

ASIDE：语言模型中指令和数据的架构分离 cs.LG

Preliminary version accepted to ICLR 2025 Workshop on Building Trust  in Language Models and Applications

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2503.10566v3) [paper-pdf](http://arxiv.org/pdf/2503.10566v3)

**Authors**: Egor Zverev, Evgenii Kortukov, Alexander Panfilov, Alexandra Volkova, Soroush Tabesh, Sebastian Lapuschkin, Wojciech Samek, Christoph H. Lampert

**Abstract**: Despite their remarkable performance, large language models lack elementary safety features, making them susceptible to numerous malicious attacks. In particular, previous work has identified the absence of an intrinsic separation between instructions and data as a root cause of the success of prompt injection attacks. In this work, we propose a new architectural element, ASIDE, that allows language models to clearly separate instructions and data at the level of embeddings. ASIDE applies an orthogonal rotation to the embeddings of data tokens, thus creating clearly distinct representations of instructions and data tokens without introducing any additional parameters. As we demonstrate experimentally across a range of models, instruction-tuning LLMs with ASIDE (1) leads to highly increased instruction-data separation without a loss in model utility and (2) makes the models more robust to prompt injection benchmarks, even without dedicated safety training. Additionally, we provide insights into the mechanism underlying our method through an analysis of the model representations. The source code and training scripts are openly accessible at https://github.com/egozverev/aside.

摘要: 尽管大型语言模型性能出色，但缺乏基本的安全功能，这使得它们容易受到大量恶意攻击。特别是，之前的工作已经确定指令和数据之间缺乏内在分离是提示注入攻击成功的根本原因。在这项工作中，我们提出了一个新的体系结构元素ASIDE，它允许语言模型在嵌入层面清楚地分离指令和数据。ASIDE将垂直旋转应用于数据令牌的嵌入，从而在不引入任何额外参数的情况下创建清晰不同的指令和数据令牌的表示。正如我们在一系列模型上通过实验证明的那样，使用ASIDE对LLM进行描述调整：（1）在不损失模型效用的情况下极大地提高了描述与数据的分离度;（2）使模型更稳健，以提示注入基准，即使没有专门的安全培训。此外，我们还通过对模型表示的分析来深入了解我们方法的基础机制。源代码和培训脚本可在https://github.com/egozverev/aside上公开访问。



## **39. On the Ethics of Using LLMs for Offensive Security**

关于使用LLM进行攻击性安全的道德 cs.CR

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.08693v1) [paper-pdf](http://arxiv.org/pdf/2506.08693v1)

**Authors**: Andreas Happe, Jürgen Cito

**Abstract**: Large Language Models (LLMs) have rapidly evolved over the past few years and are currently evaluated for their efficacy within the domain of offensive cyber-security. While initial forays showcase the potential of LLMs to enhance security research, they also raise critical ethical concerns regarding the dual-use of offensive security tooling.   This paper analyzes a set of papers that leverage LLMs for offensive security, focusing on how ethical considerations are expressed and justified in their work. The goal is to assess the culture of AI in offensive security research regarding ethics communication, highlighting trends, best practices, and gaps in current discourse.   We provide insights into how the academic community navigates the fine line between innovation and ethical responsibility. Particularly, our results show that 13 of 15 reviewed prototypes (86.6\%) mentioned ethical considerations and are thus aware of the potential dual-use of their research. Main motivation given for the research was allowing broader access to penetration-testing as well as preparing defenders for AI-guided attackers.

摘要: 大型语言模型（LLM）在过去几年中迅速发展，目前正在评估其在攻击性网络安全领域的功效。虽然最初的尝试展示了LLM增强安全研究的潜力，但它们也引发了有关攻击性安全工具双重用途的严重道德问题。   本文分析了一组利用LLM来实现攻击性安全的论文，重点关注道德考虑如何在其工作中表达和证明合理。目标是评估攻击性安全研究中有关道德沟通的人工智能文化，强调趋势、最佳实践和当前话语中的差距。   我们深入了解学术界如何在创新和道德责任之间划清界限。特别是，我们的结果显示，15个审查的原型中有13个（86.6%）提到了道德考虑，因此意识到其研究的潜在双重用途。该研究的主要动机是允许更广泛地进行渗透测试，并为防御者准备好应对人工智能引导的攻击者。



## **40. ASRJam: Human-Friendly AI Speech Jamming to Prevent Automated Phone Scams**

ASRJam：人性化的人工智能语音干扰，以防止自动电话诈骗 cs.CL

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.11125v1) [paper-pdf](http://arxiv.org/pdf/2506.11125v1)

**Authors**: Freddie Grabovski, Gilad Gressel, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs), combined with Text-to-Speech (TTS) and Automatic Speech Recognition (ASR), are increasingly used to automate voice phishing (vishing) scams. These systems are scalable and convincing, posing a significant security threat. We identify the ASR transcription step as the most vulnerable link in the scam pipeline and introduce ASRJam, a proactive defence framework that injects adversarial perturbations into the victim's audio to disrupt the attacker's ASR. This breaks the scam's feedback loop without affecting human callers, who can still understand the conversation. While prior adversarial audio techniques are often unpleasant and impractical for real-time use, we also propose EchoGuard, a novel jammer that leverages natural distortions, such as reverberation and echo, that are disruptive to ASR but tolerable to humans. To evaluate EchoGuard's effectiveness and usability, we conducted a 39-person user study comparing it with three state-of-the-art attacks. Results show that EchoGuard achieved the highest overall utility, offering the best combination of ASR disruption and human listening experience.

摘要: 大型语言模型（LLM）与文本转语音（TTC）和自动语音识别（ASB）相结合，越来越多地用于自动化语音网络钓鱼（钓鱼）诈骗。这些系统具有可扩展性且令人信服，构成了重大的安全威胁。我们将SVR转录步骤确定为诈骗管道中最脆弱的环节，并引入ASRJam，这是一种主动防御框架，可将对抗性扰动注入受害者的音频中，以扰乱攻击者的ASB。这打破了骗局的反馈循环，而不会影响人类呼叫者，因为他们仍然可以理解对话。虽然现有的对抗性音频技术对于实时使用来说通常令人不快且不切实际，但我们还提出了EchoGuard，这是一种新型干扰器，它利用了自然失真（例如回响和回声），这些失真会破坏ASB，但对人类来说是可以容忍的。为了评估EchoGuard的有效性和可用性，我们进行了一项39人用户研究，将其与三种最先进的攻击进行了比较。结果表明，EchoGuard实现了最高的总体实用性，提供了ASB干扰和人类聆听体验的最佳组合。



## **41. Evaluation empirique de la sécurisation et de l'alignement de ChatGPT et Gemini: analyse comparative des vulnérabilités par expérimentations de jailbreaks**

ChatGPT et Gemini的安全和保障评估经验：分析越狱实验中的暴力比较 cs.CR

in French language

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.10029v1) [paper-pdf](http://arxiv.org/pdf/2506.10029v1)

**Authors**: Rafaël Nouailles

**Abstract**: Large Language models (LLMs) are transforming digital usage, particularly in text generation, image creation, information retrieval and code development. ChatGPT, launched by OpenAI in November 2022, quickly became a reference, prompting the emergence of competitors such as Google's Gemini. However, these technological advances raise new cybersecurity challenges, including prompt injection attacks, the circumvention of regulatory measures (jailbreaking), the spread of misinformation (hallucinations) and risks associated with deep fakes. This paper presents a comparative analysis of the security and alignment levels of ChatGPT and Gemini, as well as a taxonomy of jailbreak techniques associated with experiments.

摘要: 大型语言模型（LLM）正在改变数字使用，特别是在文本生成、图像创建、信息检索和代码开发方面。OpenAI于2022年11月推出的ChatGPT迅速成为参考，促使谷歌Gemini等竞争对手的出现。然而，这些技术进步带来了新的网络安全挑战，包括即时注射攻击、规避监管措施（越狱）、错误信息（幻觉）的传播以及与深度造假相关的风险。本文对ChatGPT和Gemini的安全性和对齐级别进行了比较分析，以及与实验相关的越狱技术的分类。



## **42. SPBA: Utilizing Speech Large Language Model for Backdoor Attacks on Speech Classification Models**

SPBA：利用语音大语言模型对语音分类模型进行后门攻击 cs.SD

Accepted by IJCNN 2025

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.08346v1) [paper-pdf](http://arxiv.org/pdf/2506.08346v1)

**Authors**: Wenhan Yao, Fen Xiao, Xiarun Chen, Jia Liu, YongQiang He, Weiping Wen

**Abstract**: Deep speech classification tasks, including keyword spotting and speaker verification, are vital in speech-based human-computer interaction. Recently, the security of these technologies has been revealed to be susceptible to backdoor attacks. Specifically, attackers use noisy disruption triggers and speech element triggers to produce poisoned speech samples that train models to become vulnerable. However, these methods typically create only a limited number of backdoors due to the inherent constraints of the trigger function. In this paper, we propose that speech backdoor attacks can strategically focus on speech elements such as timbre and emotion, leveraging the Speech Large Language Model (SLLM) to generate diverse triggers. Increasing the number of triggers may disproportionately elevate the poisoning rate, resulting in higher attack costs and a lower success rate per trigger. We introduce the Multiple Gradient Descent Algorithm (MGDA) as a mitigation strategy to address this challenge. The proposed attack is called the Speech Prompt Backdoor Attack (SPBA). Building on this foundation, we conducted attack experiments on two speech classification tasks, demonstrating that SPBA shows significant trigger effectiveness and achieves exceptional performance in attack metrics.

摘要: 深度语音分类任务，包括关键词发现和说话人验证，在基于语音的人机交互中至关重要。最近，这些技术的安全性被揭露容易受到后门攻击。具体来说，攻击者使用有噪的中断触发器和语音元素触发器来产生有毒语音样本，这些样本训练模型变得脆弱。然而，由于触发功能的固有限制，这些方法通常只创建有限数量的后门。在本文中，我们提出语音后门攻击可以战略性地关注音色和情感等语音元素，利用语音大语言模型（slLM）来生成不同的触发器。增加触发器的数量可能会不成比例地提高中毒率，导致攻击成本更高，每次触发器的成功率更低。我们引入多重梯度下降算法（MGDA）作为应对这一挑战的缓解策略。拟议的攻击称为语音提示后门攻击（SPBA）。在此基础上，我们进行了两个语音分类任务的攻击实验，表明SPBA显示出显着的触发有效性，并实现了卓越的性能，在攻击指标。



## **43. R.R.: Unveiling LLM Training Privacy through Recollection and Ranking**

RR：通过回忆和排名揭露LLM培训隐私 cs.CL

13 pages, 9 figures; typos corrected

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2502.12658v2) [paper-pdf](http://arxiv.org/pdf/2502.12658v2)

**Authors**: Wenlong Meng, Zhenyuan Guo, Lenan Wu, Chen Gong, Wenyan Liu, Weixian Li, Chengkun Wei, Wenzhi Chen

**Abstract**: Large Language Models (LLMs) pose significant privacy risks, potentially leaking training data due to implicit memorization. Existing privacy attacks primarily focus on membership inference attacks (MIAs) or data extraction attacks, but reconstructing specific personally identifiable information (PII) in LLMs' training data remains challenging. In this paper, we propose R.R. (Recollect and Rank), a novel two-step privacy stealing attack that enables attackers to reconstruct PII entities from scrubbed training data where the PII entities have been masked. In the first stage, we introduce a prompt paradigm named recollection, which instructs the LLM to repeat a masked text but fill in masks. Then we can use PII identifiers to extract recollected PII candidates. In the second stage, we design a new criterion to score each PII candidate and rank them. Motivated by membership inference, we leverage the reference model as a calibration to our criterion. Experiments across three popular PII datasets demonstrate that the R.R. achieves better PII identification performance than baselines. These results highlight the vulnerability of LLMs to PII leakage even when training data has been scrubbed. We release our code and datasets at GitHub.

摘要: 大型语言模型（LLM）存在重大的隐私风险，可能会因隐性记忆而泄露训练数据。现有的隐私攻击主要集中在成员资格推断攻击（MIA）或数据提取攻击，但在LLM训练数据中重建特定的个人可识别信息（PRI）仍然具有挑战性。在本文中，我们提出RR（Recoll and Rank），一种新颖的两步隐私窃取攻击，使攻击者能够从已屏蔽的已清除的训练数据中重建PRI实体。在第一阶段，我们引入了一个名为回忆的提示范式，它指示LLM重复屏蔽文本但填写屏蔽。然后我们可以使用PRI标识符来提取重新收集的PRI候选项。在第二阶段，我们设计一个新的标准来对每个PRI候选人进行评分并对其进行排名。受隶属推理的激励，我们利用参考模型作为我们标准的校准。三个流行的PRI数据集的实验表明RR实现比基线更好的PRI识别性能。这些结果凸显了即使训练数据已被清除，LLM也容易受到PRI泄漏的影响。我们在GitHub上发布我们的代码和数据集。



## **44. Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models**

私有私有化编辑：将私有化转化为防御，以加强大型语言模型中的数据隐私 cs.CR

To be published at ACL 2025 (Main)

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.10024v1) [paper-pdf](http://arxiv.org/pdf/2506.10024v1)

**Authors**: Elena Sofia Ruzzetti, Giancarlo A. Xompero, Davide Venditti, Fabio Massimo Zanzotto

**Abstract**: Large Language Models (LLMs) memorize, and thus, among huge amounts of uncontrolled data, may memorize Personally Identifiable Information (PII), which should not be stored and, consequently, not leaked. In this paper, we introduce Private Memorization Editing (PME), an approach for preventing private data leakage that turns an apparent limitation, that is, the LLMs' memorization ability, into a powerful privacy defense strategy. While attacks against LLMs have been performed exploiting previous knowledge regarding their training data, our approach aims to exploit the same kind of knowledge in order to make a model more robust. We detect a memorized PII and then mitigate the memorization of PII by editing a model knowledge of its training data. We verify that our procedure does not affect the underlying language model while making it more robust against privacy Training Data Extraction attacks. We demonstrate that PME can effectively reduce the number of leaked PII in a number of configurations, in some cases even reducing the accuracy of the privacy attacks to zero.

摘要: 大型语言模型（LLM）记忆，因此，在大量不受控制的数据中，可能会记忆个人可识别信息（PRI），这些信息不应被存储，因此不应被泄露。在本文中，我们引入了私人重新同步编辑（PME），这是一种防止私人数据泄露的方法，它将明显的限制（即LLM的记忆能力）转化为强大的隐私防御策略。虽然针对LLM的攻击是利用有关其训练数据的先前知识进行的，但我们的方法旨在利用相同类型的知识以使模型更加稳健。我们检测记忆的PRI，然后通过编辑其训练数据的模型知识来减轻PRI的记忆。我们验证了我们的过程不会影响底层语言模型，同时使其对隐私训练数据提取攻击更加强大。我们证明，PME可以有效地减少泄漏的PII在一些配置的数量，在某些情况下，甚至降低隐私攻击的准确性为零。



## **45. TokenBreak: Bypassing Text Classification Models Through Token Manipulation**

TokenBreak：通过令牌操纵来破解文本分类模型 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07948v1) [paper-pdf](http://arxiv.org/pdf/2506.07948v1)

**Authors**: Kasimir Schulz, Kenneth Yeung, Kieran Evans

**Abstract**: Natural Language Processing (NLP) models are used for text-related tasks such as classification and generation. To complete these tasks, input data is first tokenized from human-readable text into a format the model can understand, enabling it to make inferences and understand context. Text classification models can be implemented to guard against threats such as prompt injection attacks against Large Language Models (LLMs), toxic input and cybersecurity risks such as spam emails. In this paper, we introduce TokenBreak: a novel attack that can bypass these protection models by taking advantage of the tokenization strategy they use. This attack technique manipulates input text in such a way that certain models give an incorrect classification. Importantly, the end target (LLM or email recipient) can still understand and respond to the manipulated text and therefore be vulnerable to the very attack the protection model was put in place to prevent. The tokenizer is tied to model architecture, meaning it is possible to predict whether or not a model is vulnerable to attack based on family. We also present a defensive strategy as an added layer of protection that can be implemented without having to retrain the defensive model.

摘要: 自然语言处理（NLP）模型用于与文本相关的任务，例如分类和生成。为了完成这些任务，首先将输入数据从人类可读的文本标记化为模型可以理解的格式，使其能够做出推断并理解上下文。可以实施文本分类模型来防范威胁，例如针对大型语言模型（LLM）的提示注入攻击、有毒输入和垃圾邮件等网络安全风险。在本文中，我们介绍TokenBreak：一种新型攻击，可以通过利用这些保护模型使用的标记化策略来绕过这些保护模型。这种攻击技术以某种方式操纵输入文本，使得某些模型给出不正确的分类。重要的是，最终目标（LLM或电子邮件收件人）仍然可以理解和响应被操纵的文本，因此很容易受到保护模型所要防止的攻击。标记器与模型架构相关联，这意味着可以基于家族预测模型是否容易受到攻击。我们还提出了一个防御策略，作为一个额外的保护层，可以在无需重新训练防御模型的情况下实现。



## **46. Adversarial Attack Classification and Robustness Testing for Large Language Models for Code**

代码大型语言模型的对抗性攻击分类和鲁棒性测试 cs.SE

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07942v1) [paper-pdf](http://arxiv.org/pdf/2506.07942v1)

**Authors**: Yang Liu, Armstrong Foundjem, Foutse Khomh, Heng Li

**Abstract**: Large Language Models (LLMs) have become vital tools in software development tasks such as code generation, completion, and analysis. As their integration into workflows deepens, ensuring robustness against vulnerabilities especially those triggered by diverse or adversarial inputs becomes increasingly important. Such vulnerabilities may lead to incorrect or insecure code generation when models encounter perturbed task descriptions, code, or comments. Prior research often overlooks the role of natural language in guiding code tasks. This study investigates how adversarial perturbations in natural language inputs including prompts, comments, and descriptions affect LLMs for Code (LLM4Code). It examines the effects of perturbations at the character, word, and sentence levels to identify the most impactful vulnerabilities. We analyzed multiple projects (e.g., ReCode, OpenAttack) and datasets (e.g., HumanEval, MBPP), establishing a taxonomy of adversarial attacks. The first dimension classifies the input type code, prompts, or comments while the second dimension focuses on granularity: character, word, or sentence-level changes. We adopted a mixed-methods approach, combining quantitative performance metrics with qualitative vulnerability analysis. LLM4Code models show varying robustness across perturbation types. Sentence-level attacks were least effective, suggesting models are resilient to broader contextual changes. In contrast, word-level perturbations posed serious challenges, exposing semantic vulnerabilities. Character-level effects varied, showing model sensitivity to subtle syntactic deviations.Our study offers a structured framework for testing LLM4Code robustness and emphasizes the critical role of natural language in adversarial evaluation. Improving model resilience to semantic-level disruptions is essential for secure and reliable code-generation systems.

摘要: 大型语言模型（LLM）已成为代码生成、完成和分析等软件开发任务的重要工具。随着它们与工作流程集成的加深，确保针对漏洞（尤其是由多样化或敌对输入触发的漏洞）的鲁棒性变得越来越重要。当模型遇到受干扰的任务描述、代码或评论时，此类漏洞可能会导致不正确或不安全的代码生成。之前的研究经常忽视自然语言在指导代码任务中的作用。本研究调查了自然语言输入（包括提示、评论和描述）中的对抗性扰动如何影响LLM for Code（LLM4Code）。它检查字符、单词和句子层面上的干扰的影响，以识别最有影响力的漏洞。我们分析了多个项目（例如，ReCode、OpenAttack）和数据集（例如，HumanEval，MBPP），建立了对抗性攻击的分类。第一个维度对输入类型代码、提示或注释进行分类，而第二个维度重点关注粒度：字符、单词或业务级别的更改。我们采用了混合方法，将定量性能指标与定性漏洞分析相结合。LLM 4Code模型显示出不同扰动类型的鲁棒性不同。句子级别的攻击效果最差，这表明模型能够适应更广泛的背景变化。相比之下，词级扰动带来了严重的挑战，暴露了语义漏洞。初级效应各不相同，表明模型对微妙的语法偏差的敏感性。我们的研究提供了一个结构化框架来测试LLM 4 Code稳健性，并强调自然语言在对抗性评估中的关键作用。提高模型对语义级中断的弹性对于安全可靠的代码生成系统至关重要。



## **47. SoK: Data Reconstruction Attacks Against Machine Learning Models: Definition, Metrics, and Benchmark**

针对机器学习模型的数据重建攻击：定义、验证和基准测试 cs.CR

To Appear in the 34th USENIX Security Symposium, August 13-15, 2025

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2506.07888v1) [paper-pdf](http://arxiv.org/pdf/2506.07888v1)

**Authors**: Rui Wen, Yiyong Liu, Michael Backes, Yang Zhang

**Abstract**: Data reconstruction attacks, which aim to recover the training dataset of a target model with limited access, have gained increasing attention in recent years. However, there is currently no consensus on a formal definition of data reconstruction attacks or appropriate evaluation metrics for measuring their quality. This lack of rigorous definitions and universal metrics has hindered further advancement in this field. In this paper, we address this issue in the vision domain by proposing a unified attack taxonomy and formal definitions of data reconstruction attacks. We first propose a set of quantitative evaluation metrics that consider important criteria such as quantifiability, consistency, precision, and diversity. Additionally, we leverage large language models (LLMs) as a substitute for human judgment, enabling visual evaluation with an emphasis on high-quality reconstructions. Using our proposed taxonomy and metrics, we present a unified framework for systematically evaluating the strengths and limitations of existing attacks and establishing a benchmark for future research. Empirical results, primarily from a memorization perspective, not only validate the effectiveness of our metrics but also offer valuable insights for designing new attacks.

摘要: 数据重建攻击旨在恢复访问权限有限的目标模型的训练数据集，近年来受到越来越多的关注。然而，目前对于数据重建攻击的正式定义或衡量其质量的适当评估指标还没有达成共识。缺乏严格的定义和通用的指标阻碍了该领域的进一步发展。在本文中，我们通过提出统一的攻击分类法和数据重建攻击的形式定义来解决视觉领域的这个问题。我们首先提出了一套定量评估指标，考虑重要的标准，如可量化性，一致性，精度和多样性。此外，我们利用大型语言模型（LLM）作为人类判断的替代品，实现视觉评估，重点是高质量的重建。使用我们提出的分类和指标，我们提出了一个统一的框架，系统地评估现有攻击的优势和局限性，并建立一个基准，为未来的研究。实证结果，主要是从记忆的角度来看，不仅验证了我们的指标的有效性，但也提供了宝贵的见解，设计新的攻击。



## **48. Is poisoning a real threat to LLM alignment? Maybe more so than you think**

中毒是对LLM联盟的真正威胁吗？也许比你想象的还要多 cs.LG

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2406.12091v4) [paper-pdf](http://arxiv.org/pdf/2406.12091v4)

**Authors**: Pankayaraj Pathmanathan, Souradip Chakraborty, Xiangyu Liu, Yongyuan Liang, Furong Huang

**Abstract**: Recent advancements in Reinforcement Learning with Human Feedback (RLHF) have significantly impacted the alignment of Large Language Models (LLMs). The sensitivity of reinforcement learning algorithms such as Proximal Policy Optimization (PPO) has led to new line work on Direct Policy Optimization (DPO), which treats RLHF in a supervised learning framework. The increased practical use of these RLHF methods warrants an analysis of their vulnerabilities. In this work, we investigate the vulnerabilities of DPO to poisoning attacks under different scenarios and compare the effectiveness of preference poisoning, a first of its kind. We comprehensively analyze DPO's vulnerabilities under different types of attacks, i.e., backdoor and non-backdoor attacks, and different poisoning methods across a wide array of language models, i.e., LLama 7B, Mistral 7B, and Gemma 7B. We find that unlike PPO-based methods, which, when it comes to backdoor attacks, require at least 4\% of the data to be poisoned to elicit harmful behavior, we exploit the true vulnerabilities of DPO more simply so we can poison the model with only as much as 0.5\% of the data. We further investigate the potential reasons behind the vulnerability and how well this vulnerability translates into backdoor vs non-backdoor attacks.

摘要: 带人类反馈的强化学习（RL HF）的最新进展显着影响了大型语言模型（LLM）的一致性。近端策略优化（PPO）等强化学习算法的敏感性导致了直接策略优化（DPO）的新工作，该算法在监督学习框架中处理RL HF。这些LLHF方法的实际使用增加，需要对其漏洞进行分析。在这项工作中，我们调查了DPO在不同场景下对中毒攻击的脆弱性，并比较了偏好中毒的有效性，这是首创。我们全面分析了DPO在不同类型攻击下的漏洞，即，后门和非后门攻击，以及跨各种语言模型的不同中毒方法，即，LLama 7B、Mistral 7B和Gemma 7B。我们发现，与基于PPO的方法不同，当涉及到后门攻击时，需要至少4%的数据被毒化才能引发有害行为，我们更简单地利用DPO的真正漏洞，因此我们可以用多达0.5%的数据毒化模型。我们进一步调查该漏洞背后的潜在原因，以及该漏洞如何转化为后门攻击与非后门攻击。



## **49. Representation Bending for Large Language Model Safety**

大型语言模型安全性的弯曲表示 cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2504.01550v2) [paper-pdf](http://arxiv.org/pdf/2504.01550v2)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.

摘要: 大型语言模型（LLM）已经成为强大的工具，但其固有的安全风险-从有害内容生成到更广泛的社会危害-构成了重大挑战。这些风险可能会因最近的对抗性攻击、微调漏洞以及在高风险环境中越来越多地部署LLM而放大。现有的安全增强技术，例如通过人工反馈或对抗性训练进行微调，仍然很脆弱，因为它们解决了特定的威胁，并且通常无法概括看不见的攻击，或者需要手动系统级防御。本文介绍了RepBend，这是一种新的方法，从根本上破坏了LLM中有害行为的表示，提供了一种可扩展的解决方案来增强（潜在的固有）安全性。RepBend将激活引导的想法（用于在推理期间引导模型行为的简单载体算法）引入到基于损失的微调中。通过广泛的评估，RepBend实现了最先进的性能，优于Circuit Breaker、RMU和NPO等现有方法，在各种越狱基准测试中，攻击成功率降低了高达95%，模型可用性和通用功能的下降微乎其微。



## **50. EVADE: Multimodal Benchmark for Evasive Content Detection in E-Commerce Applications**

EVADE：电子商务应用程序中规避内容检测的多模式基准 cs.CL

**SubmitDate**: 2025-06-09    [abs](http://arxiv.org/abs/2505.17654v2) [paper-pdf](http://arxiv.org/pdf/2505.17654v2)

**Authors**: Ancheng Xu, Zhihao Yang, Jingpeng Li, Guanghu Yuan, Longze Chen, Liang Yan, Jiehui Zhou, Zhen Qin, Hengyun Chang, Hamid Alinejad-Rokny, Bo Zheng, Min Yang

**Abstract**: E-commerce platforms increasingly rely on Large Language Models (LLMs) and Vision-Language Models (VLMs) to detect illicit or misleading product content. However, these models remain vulnerable to evasive content: inputs (text or images) that superficially comply with platform policies while covertly conveying prohibited claims. Unlike traditional adversarial attacks that induce overt failures, evasive content exploits ambiguity and context, making it far harder to detect. Existing robustness benchmarks provide little guidance for this demanding, real-world challenge. We introduce EVADE, the first expert-curated, Chinese, multimodal benchmark specifically designed to evaluate foundation models on evasive content detection in e-commerce. The dataset contains 2,833 annotated text samples and 13,961 images spanning six demanding product categories, including body shaping, height growth, and health supplements. Two complementary tasks assess distinct capabilities: Single-Violation, which probes fine-grained reasoning under short prompts, and All-in-One, which tests long-context reasoning by merging overlapping policy rules into unified instructions. Notably, the All-in-One setting significantly narrows the performance gap between partial and full-match accuracy, suggesting that clearer rule definitions improve alignment between human and model judgment. We benchmark 26 mainstream LLMs and VLMs and observe substantial performance gaps: even state-of-the-art models frequently misclassify evasive samples. By releasing EVADE and strong baselines, we provide the first rigorous standard for evaluating evasive-content detection, expose fundamental limitations in current multimodal reasoning, and lay the groundwork for safer and more transparent content moderation systems in e-commerce. The dataset is publicly available at https://huggingface.co/datasets/koenshen/EVADE-Bench.

摘要: 电子商务平台越来越依赖大型语言模型（LLM）和视觉语言模型（VLM）来检测非法或误导性产品内容。然而，这些模型仍然容易受到规避内容的影响：表面上遵守平台政策但秘密传达禁止声明的输入（文本或图像）。与导致明显失败的传统对抗性攻击不同，规避内容利用了模糊性和上下文，使其更难检测。现有的稳健性基准对这一要求严格的现实世界挑战几乎没有提供指导。我们引入EVADE，这是第一个由专家策划的中国多模式基准，专门用于评估电子商务中规避内容检测的基础模型。该数据集包含2，833个注释文本样本和13，961张图像，涵盖六个要求严格的产品类别，包括身材塑造、身高增长和保健品。两项补充任务评估不同的能力：Single-Violation（在短提示下探索细粒度推理）和All-in-One（通过将重叠的策略规则合并到统一指令中来测试长上下文推理）。值得注意的是，一体化设置显着缩小了部分匹配准确性和完全匹配准确性之间的性能差距，这表明更清晰的规则定义可以改善人类和模型判断之间的一致性。我们对26种主流LLM和VLM进行了基准测试，并观察到了巨大的性能差距：即使是最先进的模型也经常对规避样本进行错误分类。通过发布EVADE和强大的基线，我们为评估逃避内容检测提供了第一个严格的标准，暴露了当前多模式推理的根本局限性，并为电子商务中更安全、更透明的内容审核系统奠定了基础。该数据集可在https://huggingface.co/datasets/koenshen/EVADE-Bench上公开获取。



