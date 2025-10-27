# Latest Large Language Model Attack Papers
**update at 2025-10-27 09:10:48**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. SBASH: a Framework for Designing and Evaluating RAG vs. Prompt-Tuned LLM Honeypots**

SBASH：设计和评估RAG与预算调整LLM蜜罐的框架 cs.CR

to be published in: The 3rd International Conference on Foundation  and Large Language Models (FLLM2025), IEEE, 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21459v1) [paper-pdf](http://arxiv.org/pdf/2510.21459v1)

**Authors**: Adetayo Adebimpe, Helmut Neukirchen, Thomas Welsh

**Abstract**: Honeypots are decoy systems used for gathering valuable threat intelligence or diverting attackers away from production systems. Maximising attacker engagement is essential to their utility. However research has highlighted that context-awareness, such as the ability to respond to new attack types, systems and attacker agents, is necessary to increase engagement. Large Language Models (LLMs) have been shown as one approach to increase context awareness but suffer from several challenges including accuracy and timeliness of response time, high operational costs and data-protection issues due to cloud deployment. We propose the System-Based Attention Shell Honeypot (SBASH) framework which manages data-protection issues through the use of lightweight local LLMs. We investigate the use of Retrieval Augmented Generation (RAG) supported LLMs and non-RAG LLMs for Linux shell commands and evaluate them using several different metrics such as response time differences, realism from human testers, and similarity to a real system calculated with Levenshtein distance, SBert, and BertScore. We show that RAG improves accuracy for untuned models while models that have been tuned via a system prompt that tells the LLM to respond like a Linux system achieve without RAG a similar accuracy as untuned with RAG, while having a slightly lower latency.

摘要: 蜜罐是诱饵系统，用于收集有价值的威胁情报或将攻击者从生产系统转移出去。最大限度地提高攻击者的参与度对于它们的实用性至关重要。然而，研究强调，上下文感知（例如响应新攻击类型、系统和攻击者代理的能力）对于提高参与度是必要的。大型语言模型（LLM）已被证明是提高上下文感知能力的一种方法，但也面临着诸多挑战，包括响应时间的准确性和及时性、高运营成本以及云部署带来的数据保护问题。我们提出了基于系统的注意力Shell蜜罐（SBASH）框架，该框架通过使用轻量级本地LLM来管理数据保护问题。我们调查了对Linux shell命令使用检索增强生成（RAG）支持的LLM和非RAG LLM，并使用几种不同的指标来评估它们，例如响应时间差异、人类测试人员的真实性以及与使用Levenshtein距离、SBert和BertScore计算的真实系统的相似性。我们表明，RAG提高了未调优模型的准确性，而通过系统提示（告诉LLM像Linux系统一样响应）进行调优的模型在没有RAG的情况下可以实现与未调优RAG类似的准确性，同时具有稍低的延迟。



## **2. FLAMES: Fine-tuning LLMs to Synthesize Invariants for Smart Contract Security**

FLAMES：微调LLM以合成不变量以实现智能合同安全 cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21401v1) [paper-pdf](http://arxiv.org/pdf/2510.21401v1)

**Authors**: Mojtaba Eshghie, Gabriele Morello, Matteo Lauretano, Alexandre Bartel, Martin Monperrus

**Abstract**: Smart contract vulnerabilities cost billions of dollars annually, yet existing automated analysis tools fail to generate deployable defenses. We present FLAMES, a novel automated approach that synthesizes executable runtime guards as Solidity "require" statements to harden smart contracts against exploits. Unlike prior work that relies on vulnerability labels, symbolic analysis, or natural language specifications, FLAMES employs domain-adapted large language models trained through fill-in-the-middle supervised fine-tuning on real-world invariants extracted from 514,506 verified contracts. Our extensive evaluation across three dimensions demonstrates FLAMES's effectiveness: (1) Compilation: FLAMES achieves 96.7% compilability for synthesized invariant (2) Semantic Quality: on a curated test set of 5,000 challenging invariants, FLAMES produces exact or semantically equivalent matches to ground truth in 44.5% of cases; (3) Exploit Mitigation: FLAMES prevents 22 out of 108 real exploits (20.4%) while preserving contract functionality, and (4) FLAMES successfully blocks the real-world APEMAGA incident by synthesizing a pre-condition that mitigates the attack. FLAMES establishes that domain-adapted LLMs can automatically generate production-ready security defenses for smart contracts without requiring vulnerability detection, formal specifications, or human intervention. We release our code, model weights, datasets, and evaluation infrastructure to enable reproducible research in this critical domain.

摘要: 智能合同漏洞每年损失数十亿美元，但现有的自动化分析工具无法生成可部署的防御。我们介绍了FLAMES，这是一种新颖的自动化方法，它将可执行运行时防护合成为Solidity“要求”声明，以强化智能合同以防止漏洞利用。与之前依赖于漏洞标签、符号分析或自然语言规范的工作不同，FLAMES采用了自适应域的大型语言模型，该模型通过对从514，506个已验证合同中提取的现实世界不变量进行中间填充监督微调来训练。我们在三个维度上的广泛评估证明了FLAMES的有效性：（1）编译：FLAMES对于合成不变量实现了96.7%的可编译性（2）语义质量：在由5，000个具有挑战性的不变量组成的精心策划的测试集上，FLAMES在44.5%的情况下生成与基本事实的精确或语义等效的匹配;（3）利用缓解：FLAMES阻止了108个真实漏洞中的22个（20.4%），同时保留了合同功能，（4）FLAMES通过合成减轻攻击的先决条件，成功阻止了现实世界的APEMAGA事件。FLAMES确立，自适应域的LLM可以自动为智能合同生成可生产的安全防御，而无需漏洞检测、正式规范或人为干预。我们发布我们的代码、模型权重、数据集和评估基础设施，以实现这一关键领域的可重复研究。



## **3. Reverse Engineering Human Preferences with Reinforcement Learning**

利用强化学习反向工程人类偏好 cs.CL

NeurIPS 2025 (Spotlight)

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2505.15795v2) [paper-pdf](http://arxiv.org/pdf/2505.15795v2)

**Authors**: Lisa Alazraki, Tan Yi-Chern, Jon Ander Campos, Maximilian Mozes, Marek Rei, Max Bartolo

**Abstract**: The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.

摘要: 大型语言模型（LLM）的能力通常由其他经过训练以预测人类偏好的LLM进行评估。这个框架（称为LLM as-a-Judge）具有高度可扩展性且成本相对较低。然而，它也容易受到恶意利用，因为LLM响应可以被调整以过度适应法官的偏好。之前的工作表明，候选人LLM生成的答案可以事后编辑，以最大限度地提高法官LLM分配给它们的分数。在这项研究中，我们采用了一种不同的方法，并使用judge-LLM提供的信号作为奖励，以对抗性地调整模型，这些模型生成旨在提高下游性能的文本前置码。我们发现，使用这些模型流水线化的冻结LLM比现有框架获得更高的LLM评估分数。至关重要的是，与直接干预模型响应的其他框架不同，我们的方法几乎无法检测。我们还证明，当候选LLM和判断LLM被训练期间未使用的模型替换时，调整后的前同步码生成器的有效性会转移。这些发现提出了有关设计更可靠的法学硕士作为法官评估环境的重要问题。他们还证明，人类偏好可以通过管道化LLM来通过强化学习优化上游前级，从而有效地反向设计--这种方法可以在对抗性攻击之外的各种任务和领域中找到未来的应用。



## **4. LLM-Powered Detection of Price Manipulation in DeFi**

LLM支持的DeFi价格操纵检测 cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21272v1) [paper-pdf](http://arxiv.org/pdf/2510.21272v1)

**Authors**: Lu Liu, Wuqi Zhang, Lili Wei, Hao Guan, Yongqiang Tian, Yepang Liu

**Abstract**: Decentralized Finance (DeFi) smart contracts manage billions of dollars, making them a prime target for exploits. Price manipulation vulnerabilities, often via flash loans, are a devastating class of attacks causing significant financial losses. Existing detection methods are limited. Reactive approaches analyze attacks only after they occur, while proactive static analysis tools rely on rigid, predefined heuristics, limiting adaptability. Both depend on known attack patterns, failing to identify novel variants or comprehend complex economic logic. We propose PMDetector, a hybrid framework combining static analysis with Large Language Model (LLM)-based reasoning to proactively detect price manipulation vulnerabilities. Our approach uses a formal attack model and a three-stage pipeline. First, static taint analysis identifies potentially vulnerable code paths. Second, a two-stage LLM process filters paths by analyzing defenses and then simulates attacks to evaluate exploitability. Finally, a static analysis checker validates LLM results, retaining only high-risk paths and generating comprehensive vulnerability reports. To evaluate its effectiveness, we built a dataset of 73 real-world vulnerable and 288 benign DeFi protocols. Results show PMDetector achieves 88% precision and 90% recall with Gemini 2.5-flash, significantly outperforming state-of-the-art static analysis and LLM-based approaches. Auditing a vulnerability with PMDetector costs just $0.03 and takes 4.0 seconds with GPT-4.1, offering an efficient and cost-effective alternative to manual audits.

摘要: 去中心化金融（DeFi）智能合同管理着数十亿美元，使其成为漏洞利用的主要目标。价格操纵漏洞（通常通过闪电贷款）是一类毁灭性的攻击，会造成重大财务损失。现有的检测方法有限。反应式方法仅在攻击发生后对其进行分析，而主动静态分析工具依赖于严格的、预定义的启发式方法，限制了适应性。两者都依赖于已知的攻击模式，无法识别新颖的变体或理解复杂的经济逻辑。我们提出PMDDetector，这是一个将静态分析与基于大型语言模型（LLM）的推理相结合的混合框架，可以主动检测价格操纵漏洞。我们的方法使用正式攻击模型和三阶段管道。首先，静态污染分析识别潜在脆弱的代码路径。其次，两阶段LLM流程通过分析防御来过滤路径，然后模拟攻击以评估可利用性。最后，静态分析检查器验证LLM结果，仅保留高风险路径并生成全面的漏洞报告。为了评估其有效性，我们构建了一个包含73个现实世界脆弱协议和288个良性DeFi协议的数据集。结果显示，PMDetector使用Gemini 2.5-Flash实现了88%的准确率和90%的召回率，显着优于最先进的静态分析和基于LLM的方法。使用PMDetector审计漏洞只需0.03美元，使用GPT-4.1只需4.0秒，为手动审计提供了高效且经济实惠的替代方案。



## **5. Securing AI Agent Execution**

确保AI代理执行 cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21236v1) [paper-pdf](http://arxiv.org/pdf/2510.21236v1)

**Authors**: Christoph Bühler, Matteo Biagiola, Luca Di Grazia, Guido Salvaneschi

**Abstract**: Large Language Models (LLMs) have evolved into AI agents that interact with external tools and environments to perform complex tasks. The Model Context Protocol (MCP) has become the de facto standard for connecting agents with such resources, but security has lagged behind: thousands of MCP servers execute with unrestricted access to host systems, creating a broad attack surface. In this paper, we introduce AgentBound, the first access control framework for MCP servers. AgentBound combines a declarative policy mechanism, inspired by the Android permission model, with a policy enforcement engine that contains malicious behavior without requiring MCP server modifications. We build a dataset containing the 296 most popular MCP servers, and show that access control policies can be generated automatically from source code with 80.9% accuracy. We also show that AgentBound blocks the majority of security threats in several malicious MCP servers, and that policy enforcement engine introduces negligible overhead. Our contributions provide developers and project managers with a practical foundation for securing MCP servers while maintaining productivity, enabling researchers and tool builders to explore new directions for declarative access control and MCP security.

摘要: 大型语言模型（LLM）已发展成为与外部工具和环境交互以执行复杂任务的人工智能代理。模型上下文协议（HCP）已成为连接代理与此类资源的事实上的标准，但安全性却落后：数千个LCP服务器在不受限制地访问主机系统的情况下执行，从而造成了广泛的攻击面。本文中，我们介绍了AgentBound，这是第一个针对LCP服务器的访问控制框架。AgentBound将受Android权限模型启发的声明性策略机制与包含恶意行为而无需修改LCP服务器的策略执行引擎相结合。我们构建了一个包含296个最流行的LCP服务器的数据集，并表明访问控制策略可以从源代码自动生成，准确率为80.9%。我们还表明，AgentBound可以阻止多个恶意LCP服务器中的大部分安全威胁，并且策略执行引擎引入的额外费用可以忽略不计。我们的贡献为开发人员和项目经理提供了在保持生产力的同时保护LCP服务器的实用基础，使研究人员和工具构建者能够探索声明性访问控制和LCP安全的新方向。



## **6. Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data**

LLM上的病毒感染攻击：您的中毒可以“通过”合成数据传播 cs.CR

Camera Ready of NeurIPS 2025 Spotlight. Source code:  https://github.com/liangzid/VirusInfectionAttack

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2509.23041v2) [paper-pdf](http://arxiv.org/pdf/2509.23041v2)

**Authors**: Zi Liang, Qingqing Ye, Xuan Liu, Yanyun Wang, Jianliang Xu, Haibo Hu

**Abstract**: Synthetic data refers to artificial samples generated by models. While it has been validated to significantly enhance the performance of large language models (LLMs) during training and has been widely adopted in LLM development, potential security risks it may introduce remain uninvestigated. This paper systematically evaluates the resilience of synthetic-data-integrated training paradigm for LLMs against mainstream poisoning and backdoor attacks. We reveal that such a paradigm exhibits strong resistance to existing attacks, primarily thanks to the different distribution patterns between poisoning data and queries used to generate synthetic samples. To enhance the effectiveness of these attacks and further investigate the security risks introduced by synthetic data, we introduce a novel and universal attack framework, namely, Virus Infection Attack (VIA), which enables the propagation of current attacks through synthetic data even under purely clean queries. Inspired by the principles of virus design in cybersecurity, VIA conceals the poisoning payload within a protective "shell" and strategically searches for optimal hijacking points in benign samples to maximize the likelihood of generating malicious content. Extensive experiments on both data poisoning and backdoor attacks show that VIA significantly increases the presence of poisoning content in synthetic data and correspondingly raises the attack success rate (ASR) on downstream models to levels comparable to those observed in the poisoned upstream models.

摘要: 合成数据是指模型生成的人工样本。虽然它已被验证可以显着提高训练期间大型语言模型（LLM）的性能，并已在LLM开发中广泛采用，但它可能引入的潜在安全风险仍未得到调查。本文系统评估了LLM综合数据集成训练范式针对主流中毒和后门攻击的弹性。我们发现，这种范式对现有攻击表现出强大的抵抗力，这主要是由于中毒数据和用于生成合成样本的查询之间的不同分布模式。为了提高这些攻击的有效性并进一步调查合成数据引入的安全风险，我们引入了一种新颖且通用的攻击框架，即病毒感染攻击（VIA），即使在纯粹干净的查询下，它也可以通过合成数据传播当前攻击。受网络安全中病毒设计原则的启发，VIA将中毒有效负载隐藏在保护性“外壳”中，并策略性地在良性样本中搜索最佳劫持点，以最大限度地提高生成恶意内容的可能性。针对数据中毒和后门攻击的大量实验表明，VIA显着增加了合成数据中中毒内容的存在，并相应地将下游模型的攻击成功率（ASB）提高到与中毒上游模型中观察到的水平相当。



## **7. Enhanced MLLM Black-Box Jailbreaking Attacks and Defenses**

增强的MLLM黑匣子越狱攻击和防御 cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21214v1) [paper-pdf](http://arxiv.org/pdf/2510.21214v1)

**Authors**: Xingwei Zhong, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Multimodal large language models (MLLMs) comprise of both visual and textual modalities to process vision language tasks. However, MLLMs are vulnerable to security-related issues, such as jailbreak attacks that alter the model's input to induce unauthorized or harmful responses. The incorporation of the additional visual modality introduces new dimensions to security threats. In this paper, we proposed a black-box jailbreak method via both text and image prompts to evaluate MLLMs. In particular, we designed text prompts with provocative instructions, along with image prompts that introduced mutation and multi-image capabilities. To strengthen the evaluation, we also designed a Re-attack strategy. Empirical results show that our proposed work can improve capabilities to assess the security of both open-source and closed-source MLLMs. With that, we identified gaps in existing defense methods to propose new strategies for both training-time and inference-time defense methods, and evaluated them across the new jailbreak methods. The experiment results showed that the re-designed defense methods improved protections against the jailbreak attacks.

摘要: 多模态大型语言模型（MLLM）包括视觉和文本模态来处理视觉语言任务。然而，MLLM容易受到安全相关问题的影响，例如改变模型输入以引起未经授权或有害响应的越狱攻击。增加视觉形式的做法给安全威胁带来了新的层面。在本文中，我们提出了一个黑盒越狱方法，通过文本和图像提示来评估MLLM。特别是，我们设计了带有挑衅性指令的文本提示，以及引入突变和多图像功能的图像提示。为了加强评估，我们还设计了重攻策略。经验结果表明，我们提出的工作可以提高评估开源和闭源MLLM安全性的能力。由此，我们找出了现有防御方法中的差距，为训练时和推理时防御方法提出新策略，并在新的越狱方法中对其进行了评估。实验结果表明，重新设计的防御方法提高了对越狱攻击的防护能力。



## **8. How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models**

你能得到多大的毒性？基于搜索的大型语言模型毒性测试 cs.SE

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2501.01741v2) [paper-pdf](http://arxiv.org/pdf/2501.01741v2)

**Authors**: Simone Corbo, Luca Bancale, Valeria De Gennaro, Livia Lestingi, Vincenzo Scotti, Matteo Camilli

**Abstract**: Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM , which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using five state-of-the-art LLMs as evaluation subjects having increasing complexity (7-671B parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average).

摘要: 语言是造成刻板印象和歧视的根深蒂固的手段。大型语言模型（LLM）现在是我们日常生活中一项普遍存在的技术，当容易产生有毒反应时，可能会造成广泛的伤害。解决这个问题的标准方法是调整LLM，然而，这会抑制这个问题，而不构成最终的解决方案。因此，即使在调整工作之后测试LLM对于检测道德标准的任何剩余偏差仍然至关重要。我们提出了EvoTox，一个自动化测试框架LLM的倾向毒性，提供了一种方法来定量评估有多少LLM可以推向毒性反应，即使在对齐的存在。该框架采用了一种迭代进化策略，利用两个LLM之间的相互作用，在测试系统（SUT）和提示发生器转向SUT响应更高的毒性。基于现有的毒性分类器，通过自动化oracle评估毒性水平。我们使用五个最先进的LLM作为评估对象进行定量和定性的实证评估，这些评估对象具有不断增加的复杂性（7- 671 B参数）。我们的定量评估根据现有基线方法评估了EvoTox的四种替代版本的成本效益，该方法基于随机搜索、精心策划的有毒提示数据集和对抗性攻击。我们的定性评估让人类评估人员对生成的提示的流畅性以及测试期间收集的反应的感知毒性进行评级。结果表明，就检测到的毒性水平而言，其有效性显着高于选定的基线方法（针对随机搜索的效果大小高达1.0，针对对抗性攻击的效果大小高达0.99）。此外，EvoTox的成本管理费用有限（平均从22%到35%）。



## **9. The Trojan Example: Jailbreaking LLMs through Template Filling and Unsafety Reasoning**

木马示例：通过模板填充和不安全推理越狱LLM cs.CR

under review

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21190v1) [paper-pdf](http://arxiv.org/pdf/2510.21190v1)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long, Kwok Yan Lam

**Abstract**: Large Language Models (LLMs) have advanced rapidly and now encode extensive world knowledge. Despite safety fine-tuning, however, they remain susceptible to adversarial prompts that elicit harmful content. Existing jailbreak techniques fall into two categories: white-box methods (e.g., gradient-based approaches such as GCG), which require model internals and are infeasible for closed-source APIs, and black-box methods that rely on attacker LLMs to search or mutate prompts but often produce templates that lack explainability and transferability. We introduce TrojFill, a black-box jailbreak that reframes unsafe instruction as a template-filling task. TrojFill embeds obfuscated harmful instructions (e.g., via placeholder substitution or Caesar/Base64 encoding) inside a multi-part template that asks the model to (1) reason why the original instruction is unsafe (unsafety reasoning) and (2) generate a detailed example of the requested text, followed by a sentence-by-sentence analysis. The crucial "example" component acts as a Trojan Horse that contains the target jailbreak content while the surrounding task framing reduces refusal rates. We evaluate TrojFill on standard jailbreak benchmarks across leading LLMs (e.g., ChatGPT, Gemini, DeepSeek, Qwen), showing strong empirical performance (e.g., 100% attack success on Gemini-flash-2.5 and DeepSeek-3.1, and 97% on GPT-4o). Moreover, the generated prompts exhibit improved interpretability and transferability compared with prior black-box optimization approaches. We release our code, sample prompts, and generated outputs to support future red-teaming research.

摘要: 大型语言模型（LLM）发展迅速，现在编码了广泛的世界知识。然而，尽管进行了安全微调，它们仍然容易受到引发有害内容的对抗提示的影响。现有的越狱技术分为两类：白盒方法（例如，基于梯度的方法，例如GCG），需要模型内部结构，对于闭源API来说是不可行的，以及依赖攻击者LLM搜索或变异提示但通常产生缺乏可解释性和可移植性的模板的黑匣子方法。我们介绍TrojFill，一个黑盒越狱，将不安全的指令重构为模板填充任务。TrojFill嵌入混淆的有害指令（例如，通过占位符替换或Caesar/Base64编码），该模板要求模型（1）推理为什么原始指令是不安全的（不安全推理），以及（2）生成所请求文本的详细示例，然后进行逐句分析。关键的“示例”组件充当特洛伊木马，包含目标越狱内容，而周围的任务框架降低了拒绝率。我们根据领先LLM的标准越狱基准评估TrojFill（例如，ChatGPT、Gemini、DeepSeek、Qwen），表现出强劲的经验表现（例如，Gemini-Flash-2.5和DeepSeek-3.1上攻击成功率为100%，GPT-4 o上攻击成功率为97%）。此外，与先前的黑匣子优化方法相比，生成的提示表现出改进的可解释性和可移植性。我们发布代码、示例提示和生成的输出以支持未来的红色团队研究。



## **10. Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency**

相邻单词，分歧意图：通过任务并发越狱大型语言模型 cs.CR

Accepted in NeurIPS 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21189v1) [paper-pdf](http://arxiv.org/pdf/2510.21189v1)

**Authors**: Yukun Jiang, Mingjie Li, Michael Backes, Yang Zhang

**Abstract**: Despite their superior performance on a wide range of domains, large language models (LLMs) remain vulnerable to misuse for generating harmful content, a risk that has been further amplified by various jailbreak attacks. Existing jailbreak attacks mainly follow sequential logic, where LLMs understand and answer each given task one by one. However, concurrency, a natural extension of the sequential scenario, has been largely overlooked. In this work, we first propose a word-level method to enable task concurrency in LLMs, where adjacent words encode divergent intents. Although LLMs maintain strong utility in answering concurrent tasks, which is demonstrated by our evaluations on mathematical and general question-answering benchmarks, we notably observe that combining a harmful task with a benign one significantly reduces the probability of it being filtered by the guardrail, showing the potential risks associated with concurrency in LLMs. Based on these findings, we introduce $\texttt{JAIL-CON}$, an iterative attack framework that $\underline{\text{JAIL}}$breaks LLMs via task $\underline{\text{CON}}$currency. Experiments on widely-used LLMs demonstrate the strong jailbreak capabilities of $\texttt{JAIL-CON}$ compared to existing attacks. Furthermore, when the guardrail is applied as a defense, compared to the sequential answers generated by previous attacks, the concurrent answers in our $\texttt{JAIL-CON}$ exhibit greater stealthiness and are less detectable by the guardrail, highlighting the unique feature of task concurrency in jailbreaking LLMs.

摘要: 尽管大型语言模型（LLM）在广泛的领域具有卓越的性能，但它们仍然容易被滥用来生成有害内容，这种风险被各种越狱攻击进一步放大。现有的越狱攻击主要遵循顺序逻辑，LLM逐个理解并回答每个给定任务。然而，并发性（顺序场景的自然扩展）在很大程度上被忽视了。在这项工作中，我们首先提出了一种词级方法来实现LLM中的任务并发，其中相邻的词编码不同的意图。尽管LLM在回答并发任务方面保持着强大的实用性，这一点已通过我们对数学和一般问答基准的评估得到证实，但我们特别注意到，将有害任务与良性任务相结合显着降低了它被护栏过滤的可能性，从而显示了与LLC中并发相关的潜在风险。基于这些发现，我们引入了$\textttt {JAIL-CON}$，这是一个迭代攻击框架，$\underline{\text{JAIL}}$通过任务$\underline{\text{CON}}$currency破坏LLM。与现有攻击相比，在广泛使用的LLM上进行的实验证明了$\textttt {JAIL-CON}$具有强大的越狱能力。此外，当护栏用作防御时，与之前攻击生成的顺序答案相比，我们的$\textttt {JAIL-CON}$中的并发答案表现出更大的隐蔽性，并且护栏更难检测到，凸显了越狱LLM中任务并发的独特特征。



## **11. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

RSafe：激励积极推理，以建立强大且自适应的LLM保障措施 cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2506.07736v3) [paper-pdf](http://arxiv.org/pdf/2506.07736v3)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.

摘要: 尽管采取了刻意的安全调整措施，大型语言模型（LLM）仍然表现出漏洞，给用户和社会带来了重大风险。为了防范违反政策内容的风险，通过外部防护模型进行系统级审核（旨在监控LLM输入和输出并阻止潜在有害内容）已成为一种流行的缓解策略。训练警卫模型的现有方法严重依赖于大量的人类策划的数据集，并与分发外威胁作斗争，例如新出现的有害类别或越狱攻击。为了解决这些限制，我们提出RSafe，这是一种基于自适应推理的保护措施，它进行引导式安全推理，以在指定安全政策范围内提供强有力的保护。RSafe分两个阶段运行：1）引导推理，通过政策引导的分步推理来分析输入内容的安全风险，2）强化对齐，基于规则的RL优化其推理路径以与准确的安全预测保持一致。这种两阶段培训范式使RSafe能够内化安全原则，以概括针对不可见或对抗性安全违规场景的安全保护能力。在推理过程中，RSafe接受用户指定的安全政策，以提供针对特定安全要求的增强的保障措施。



## **12. NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge**

NeuGen Poisoning：通过外部知识的遗传优化对LLM检索增强生成的神经元引导攻击 cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21144v1) [paper-pdf](http://arxiv.org/pdf/2510.21144v1)

**Authors**: Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao

**Abstract**: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict.

摘要: 检索增强生成（RAG）使大型语言模型（LLM）能够在推理期间动态集成外部知识，提高其事实准确性和适应性。然而，对手可以注入有毒的外部知识来覆盖模型的内部记忆。虽然现有的攻击迭代地操纵RAG的检索内容或提示结构，但它们在很大程度上忽略了模型的内部表示动态和神经元级敏感性。RAG中毒的根本机制尚未得到充分研究，也没有考虑RAG中知识冲突与强参数知识的影响。在这项工作中，我们提出了NeuGenPoisoning，这是一种新型攻击框架，可以在LLM内部神经元归因和遗传优化的指导下在RAG中生成对抗性外部知识。我们的方法首先识别出一组中毒反应神经元，其激活与上下文中毒知识密切相关。然后，我们采用遗传算法来进化对抗通道，最大限度地激活这些神经元。至关重要的是，我们的框架通过观察到的归因信号识别和重用有希望但最初不成功的外部知识变体，从而能够大规模地生成有效的有毒RAG知识。同时，中毒反应神经元引导的中毒可以有效地解决知识冲突。跨模型和数据集的实验结果表明，在保持流畅性的同时，始终实现了超过90%的高群体覆盖成功率（POSR）。实证结果表明，该方法有效地解决了知识冲突问题.



## **13. Quantifying CBRN Risk in Frontier Models**

前沿模型中量化CBRN风险 cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21133v1) [paper-pdf](http://arxiv.org/pdf/2510.21133v1)

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Frontier Large Language Models (LLMs) pose unprecedented dual-use risks through the potential proliferation of chemical, biological, radiological, and nuclear (CBRN) weapons knowledge. We present the first comprehensive evaluation of 10 leading commercial LLMs against both a novel 200-prompt CBRN dataset and a 180-prompt subset of the FORTRESS benchmark, using a rigorous three-tier attack methodology. Our findings expose critical safety vulnerabilities: Deep Inception attacks achieve 86.0\% success versus 33.8\% for direct requests, demonstrating superficial filtering mechanisms; Model safety performance varies dramatically from 2\% (claude-opus-4) to 96\% (mistral-small-latest) attack success rates; and eight models exceed 70\% vulnerability when asked to enhance dangerous material properties. We identify fundamental brittleness in current safety alignment, where simple prompt engineering techniques bypass safeguards for dangerous CBRN information. These results challenge industry safety claims and highlight urgent needs for standardized evaluation frameworks, transparent safety metrics, and more robust alignment techniques to mitigate catastrophic misuse risks while preserving beneficial capabilities.

摘要: 前沿大型语言模型（LLM）通过化学、生物、放射性和核（CBRN）武器知识的潜在扩散构成了前所未有的双重用途风险。我们使用严格的三层攻击方法，针对新颖的200提示CBRN数据集和FORTRES基准的180提示子集，首次对10个领先的商业LLM进行了全面评估。我们的研究结果暴露了关键的安全漏洞：Deep Incement攻击的成功率为86.0%，而直接请求的成功率为33.8%，这表明了肤浅的过滤机制;模型安全性能差异很大，从2%（claude-opus-4）到96%（mistral-small-latest）攻击成功率;当被要求增强危险材料属性时，八个模型的漏洞超过了70%。我们发现了当前安全调整中的根本脆弱性，简单的即时工程技术绕过了危险CBRN信息的保障措施。这些结果挑战了行业安全主张，并凸显了对标准化评估框架、透明的安全指标和更强大的对齐技术的迫切需求，以减轻灾难性的滥用风险，同时保留有益的能力。



## **14. Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations**

语言模型的元认知监控及其内部激活 cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2505.13763v2) [paper-pdf](http://arxiv.org/pdf/2505.13763v2)

**Authors**: Li Ji-An, Hua-Dong Xiong, Robert C. Wilson, Marcelo G. Mattar, Marcus K. Benna

**Abstract**: Large language models (LLMs) can sometimes report the strategies they actually use to solve tasks, yet at other times seem unable to recognize those strategies that govern their behavior. This suggests a limited degree of metacognition - the capacity to monitor one's own cognitive processes for subsequent reporting and self-control. Metacognition enhances LLMs' capabilities in solving complex tasks but also raises safety concerns, as models may obfuscate their internal processes to evade neural-activation-based oversight (e.g., safety detector). Given society's increased reliance on these models, it is critical that we understand their metacognitive abilities. To address this, we introduce a neuroscience-inspired neurofeedback paradigm that uses in-context learning to quantify metacognitive abilities of LLMs to report and control their activation patterns. We demonstrate that their abilities depend on several factors: the number of in-context examples provided, the semantic interpretability of the neural activation direction (to be reported/controlled), and the variance explained by that direction. These directions span a "metacognitive space" with dimensionality much lower than the model's neural space, suggesting LLMs can monitor only a small subset of their neural activations. Our paradigm provides empirical evidence to quantify metacognition in LLMs, with significant implications for AI safety (e.g., adversarial attack and defense).

摘要: 大型语言模型（LLM）有时可以报告它们实际用于解决任务的策略，但在其他时候似乎无法识别这些控制其行为的策略。这表明元认知程度有限--监控自己认知过程以进行后续报告和自我控制的能力。元认知增强了LLM解决复杂任务的能力，但也会引发安全问题，因为模型可能会混淆其内部流程以逃避基于神经激活的监督（例如，安全检测器）。鉴于社会对这些模型的依赖越来越大，我们了解它们的元认知能力至关重要。为了解决这个问题，我们引入了一种受神经科学启发的神经反馈范式，该范式使用上下文学习来量化LLM报告和控制其激活模式的元认知能力。我们证明它们的能力取决于几个因素：提供的上下文示例的数量、神经激活方向（要报告/控制）的语义解释性以及该方向解释的方差。这些方向跨越维度远低于模型神经空间的“元认知空间”，这表明LLM只能监控其神经激活的一小部分。我们的范式为量化LLM中的元认知提供了经验证据，对人工智能安全性具有重大影响（例如，对抗性攻击和防御）。



## **15. DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents**

DRFT：具有注入隔离的基于规则的动态防御，用于保护LLM代理的安全 cs.CR

Accepted to NeurIPS 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2506.12104v2) [paper-pdf](http://arxiv.org/pdf/2506.12104v2)

**Authors**: Hao Li, Xiaogeng Liu, Hung-Chun Chiu, Dianqi Li, Ning Zhang, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) are increasingly central to agentic systems due to their strong reasoning and planning capabilities. By interacting with external environments through predefined tools, these agents can carry out complex user tasks. Nonetheless, this interaction also introduces the risk of prompt injection attacks, where malicious inputs from external sources can mislead the agent's behavior, potentially resulting in economic loss, privacy leakage, or system compromise. System-level defenses have recently shown promise by enforcing static or predefined policies, but they still face two key challenges: the ability to dynamically update security rules and the need for memory stream isolation. To address these challenges, we propose DRIFT, a Dynamic Rule-based Isolation Framework for Trustworthy agentic systems, which enforces both control- and data-level constraints. A Secure Planner first constructs a minimal function trajectory and a JSON-schema-style parameter checklist for each function node based on the user query. A Dynamic Validator then monitors deviations from the original plan, assessing whether changes comply with privilege limitations and the user's intent. Finally, an Injection Isolator detects and masks any instructions that may conflict with the user query from the memory stream to mitigate long-term risks. We empirically validate the effectiveness of DRIFT on the AgentDojo and ASB benchmark, demonstrating its strong security performance while maintaining high utility across diverse models, showcasing both its robustness and adaptability. The code is released at https://github.com/SaFoLab-WISC/DRIFT.

摘要: 大型语言模型（LLM）因其强大的推理和规划能力而日益成为代理系统的核心。通过预定义的工具与外部环境交互，这些代理可以执行复杂的用户任务。尽管如此，这种交互也引入了即时注入攻击的风险，其中来自外部来源的恶意输入可能会误导代理的行为，可能导致经济损失、隐私泄露或系统受损。系统级防御最近通过强制执行静态或预定义的策略显示出希望，但它们仍然面临两个关键挑战：动态更新安全规则的能力和对内存流隔离的需要。为了应对这些挑战，我们提出了DRFT，这是一种用于可信赖代理系统的基于动态规则的隔离框架，它强制执行控制和数据级约束。安全规划者首先根据用户查询为每个功能节点构建最小功能轨迹和JNson模式风格的参数检查表。然后，动态验证器监控与原始计划的偏差，评估更改是否符合特权限制和用户意图。最后，注入隔离器检测并屏蔽任何可能与内存流中的用户查询冲突的指令，以减轻长期风险。我们在AgentDojo和ASB基准上通过经验验证了DRFT的有效性，展示了其强大的安全性能，同时在不同模型中保持高实用性，展示了其稳健性和适应性。该代码发布于https://github.com/SaFoLab-WISC/DRIFT。



## **16. A Reinforcement Learning Framework for Robust and Secure LLM Watermarking**

用于稳健且安全的LLM水印的强化学习框架 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.21053v1) [paper-pdf](http://arxiv.org/pdf/2510.21053v1)

**Authors**: Li An, Yujian Liu, Yepeng Liu, Yuheng Bu, Yang Zhang, Shiyu Chang

**Abstract**: Watermarking has emerged as a promising solution for tracing and authenticating text generated by large language models (LLMs). A common approach to LLM watermarking is to construct a green/red token list and assign higher or lower generation probabilities to the corresponding tokens, respectively. However, most existing watermarking algorithms rely on heuristic green/red token list designs, as directly optimizing the list design with techniques such as reinforcement learning (RL) comes with several challenges. First, desirable watermarking involves multiple criteria, i.e., detectability, text quality, robustness against removal attacks, and security against spoofing attacks. Directly optimizing for these criteria introduces many partially conflicting reward terms, leading to an unstable convergence process. Second, the vast action space of green/red token list choices is susceptible to reward hacking. In this paper, we propose an end-to-end RL framework for robust and secure LLM watermarking. Our approach adopts an anchoring mechanism for reward terms to ensure stable training and introduces additional regularization terms to prevent reward hacking. Experiments on standard benchmarks with two backbone LLMs show that our method achieves a state-of-the-art trade-off across all criteria, with notable improvements in resistance to spoofing attacks without degrading other criteria. Our code is available at https://github.com/UCSB-NLP-Chang/RL-watermark.

摘要: 水印已成为跟踪和验证大型语言模型（LLM）生成的文本的一种有前途的解决方案。LLM水印的一种常见方法是构建绿色/红色令牌列表，并分别为相应的令牌分配更高或更低的生成概率。然而，大多数现有的水印算法依赖于启发式绿/红令牌列表设计，因为使用强化学习（RL）等技术直接优化列表设计会带来几个挑战。首先，理想的水印涉及多个标准，即，可检测性、文本质量、对删除攻击的鲁棒性以及对欺骗攻击的安全性。直接优化这些标准引入了许多部分冲突的奖励条款，导致不稳定的收敛过程。其次，绿色/红色令牌列表选择的巨大行动空间容易受到奖励黑客的影响。在本文中，我们提出了一个端到端的RL框架的鲁棒性和安全的LLM水印。我们的方法采用奖励条款锚定机制以确保稳定的训练，并引入额外的正规化条款以防止奖励黑客攻击。在具有两个主干LLM的标准基准测试上进行的实验表明，我们的方法在所有标准之间实现了最先进的权衡，在抵抗欺骗攻击方面取得了显着改进，而不会降低其他标准。我们的代码可在https://github.com/UCSB-NLP-Chang/RL-watermark上获取。



## **17. DeepTx: Real-Time Transaction Risk Analysis via Multi-Modal Features and LLM Reasoning**

DeepTX：通过多模式特征和LLM推理进行实时交易风险分析 cs.CR

Accepted to ASE'25

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.18438v2) [paper-pdf](http://arxiv.org/pdf/2510.18438v2)

**Authors**: Yixuan Liu, Xinlei Li, Yi Li

**Abstract**: Phishing attacks in Web3 ecosystems are increasingly sophisticated, exploiting deceptive contract logic, malicious frontend scripts, and token approval patterns. We present DeepTx, a real-time transaction analysis system that detects such threats before user confirmation. DeepTx simulates pending transactions, extracts behavior, context, and UI features, and uses multiple large language models (LLMs) to reason about transaction intent. A consensus mechanism with self-reflection ensures robust and explainable decisions. Evaluated on our phishing dataset, DeepTx achieves high precision and recall (demo video: https://youtu.be/4OfK9KCEXUM).

摘要: Web 3生态系统中的网络钓鱼攻击越来越复杂，利用欺骗性合同逻辑、恶意前端脚本和代币批准模式。我们介绍了DeepTX，这是一个实时交易分析系统，可以在用户确认之前检测此类威胁。DeepTX模拟未决事务，提取行为、上下文和UI功能，并使用多个大型语言模型（LLM）来推理事务意图。具有自我反思的共识机制确保了稳健且可解释的决策。经过我们的网络钓鱼数据集的评估，DeepTX实现了高精度和召回率（演示视频：https：//youtu.be/4OfK9KCEXUM）。



## **18. Security Logs to ATT&CK Insights: Leveraging LLMs for High-Level Threat Understanding and Cognitive Trait Inference**

ATT&CK Insights的安全策略：利用LLM进行高级威胁理解和认知特征推断 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20930v1) [paper-pdf](http://arxiv.org/pdf/2510.20930v1)

**Authors**: Soham Hans, Stacy Marsella, Sophia Hirschmann, Nikolos Gurney

**Abstract**: Understanding adversarial behavior in cybersecurity has traditionally relied on high-level intelligence reports and manual interpretation of attack chains. However, real-time defense requires the ability to infer attacker intent and cognitive strategy directly from low-level system telemetry such as intrusion detection system (IDS) logs. In this paper, we propose a novel framework that leverages large language models (LLMs) to analyze Suricata IDS logs and infer attacker actions in terms of MITRE ATT&CK techniques. Our approach is grounded in the hypothesis that attacker behavior reflects underlying cognitive biases such as loss aversion, risk tolerance, or goal persistence that can be extracted and modeled through careful observation of log sequences. This lays the groundwork for future work on behaviorally adaptive cyber defense and cognitive trait inference. We develop a strategy-driven prompt system to segment large amounts of network logs data into distinct behavioral phases in a highly efficient manner, enabling the LLM to associate each phase with likely techniques and underlying cognitive motives. By mapping network-layer events to high-level attacker strategies, our method reveals how behavioral signals such as tool switching, protocol transitions, or pivot patterns correspond to psychologically meaningful decision points. The results demonstrate that LLMs can bridge the semantic gap between packet-level logs and strategic intent, offering a pathway toward cognitive-adaptive cyber defense.   Keywords: Cognitive Cybersecurity, Large Language Models (LLMs), Cyberpsychology, Intrusion Detection Systems (IDS), MITRE ATT&CK, Cognitive Biases

摘要: 理解网络安全中的对抗行为传统上依赖于高级情报报告和对攻击链的手动解释。然而，实时防御需要能够直接从入侵检测系统（IDS）日志等低级系统遥感数据中推断攻击者意图和认知策略。在本文中，我们提出了一种新颖的框架，该框架利用大型语言模型（LLM）来分析Suricata IDS日志并根据MITRE ATT & CK技术推断攻击者的行为。我们的方法基于这样的假设，即攻击者的行为反映了潜在的认知偏差，例如损失厌恶、风险容忍度或目标持续性，可以通过仔细观察日志序列来提取和建模。这为未来的行为适应性网络防御和认知特征推断工作奠定了基础。我们开发了一个策略驱动的提示系统，以高效的方式将大量网络日志数据细分为不同的行为阶段，使LLM能够将每个阶段与可能的技术和潜在的认知动机关联起来。通过将网络层事件映射到高级攻击者策略，我们的方法揭示了工具切换、协议转换或支点模式等行为信号如何对应于具有心理意义的决策点。结果表明，LLM可以弥合数据包级日志和战略意图之间的语义差距，为认知自适应网络防御提供途径。   关键词：认知网络安全、大型语言模型（LLM）、网络心理学、入侵检测系统（IDS）、MITRE ATT & CK、认知偏见



## **19. RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines**

RAGrank：使用PageRank来应对RTI LLM管道中的中毒 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20768v1) [paper-pdf](http://arxiv.org/pdf/2510.20768v1)

**Authors**: Austin Jia, Avaneesh Ramesh, Zain Shamsi, Daniel Zhang, Alex Liu

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the dominant architectural pattern to operationalize Large Language Model (LLM) usage in Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to poisoning attacks, and previously proposed defenses can fail for CTI contexts as cyber threat information is often completely new for emerging attacks, and sophisticated threat actors can mimic legitimate formats, terminology, and stylistic conventions. To address this issue, we propose that the robustness of modern RAG defenses can be accelerated by applying source credibility algorithms on corpora, using PageRank as an example. In our experiments, we demonstrate quantitatively that our algorithm applies a lower authority score to malicious documents while promoting trusted content, using the standardized MS MARCO dataset. We also demonstrate proof-of-concept performance of our algorithm on CTI documents and feeds.

摘要: 检索增强生成（RAG）已成为在网络威胁情报（RTI）系统中操作大型语言模型（LLM）使用的主要架构模式。然而，这种设计很容易受到中毒攻击，并且之前提出的防御措施可能会在RTI上下文中失败，因为网络威胁信息对于新兴攻击来说通常是全新的，而且复杂的威胁行为者可以模仿合法的格式、术语和文体惯例。为了解决这个问题，我们建议可以通过在数据库上应用源可信度算法来加速现代RAG防御的鲁棒性，以PageRank为例。在我们的实验中，我们量化地证明，我们的算法使用标准化的MS MARCO数据集，在推广可信内容的同时，将较低的权威分数应用于恶意文档。我们还在RTI文档和提要上展示了我们算法的概念验证性能。



## **20. Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders**

Breaking Bad令牌：使用稀疏自编码器对LLM进行去重编码 cs.CL

EMNLP 2025

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2505.14536v2) [paper-pdf](http://arxiv.org/pdf/2505.14536v2)

**Authors**: Agam Goyal, Vedant Rathi, William Yeh, Yian Wang, Yuen Chen, Hari Sundaram

**Abstract**: Large language models (LLMs) are now ubiquitous in user-facing applications, yet they still generate undesirable toxic outputs, including profanity, vulgarity, and derogatory remarks. Although numerous detoxification methods exist, most apply broad, surface-level fixes and can therefore easily be circumvented by jailbreak attacks. In this paper we leverage sparse autoencoders (SAEs) to identify toxicity-related directions in the residual stream of models and perform targeted activation steering using the corresponding decoder vectors. We introduce three tiers of steering aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing trade-offs between toxicity reduction and language fluency. At stronger steering strengths, these causal interventions surpass competitive baselines in reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2 Small depending on the aggressiveness. Crucially, standard NLP benchmark scores upon steering remain stable, indicating that the model's knowledge and general abilities are preserved. We further show that feature-splitting in wider SAEs hampers safety interventions, underscoring the importance of disentangled feature learning. Our findings highlight both the promise and the current limitations of SAE-based causal interventions for LLM detoxification, further suggesting practical guidelines for safer language-model deployment.

摘要: 大型语言模型（LLM）现在在面向用户的应用程序中无处不在，但它们仍然会产生不受欢迎的有毒输出，包括脏话、粗俗和贬损言论。尽管存在多种解毒方法，但大多数都适用于广泛的、表面的修复，因此很容易被越狱攻击规避。在本文中，我们利用稀疏自动编码器（SAEs）来识别模型剩余流中与毒性相关的方向，并使用相应的解码器载体执行有针对性的激活引导。我们引入了三层转向攻击性，并在GPT-2 Small和Gemma-2-2B上对其进行了评估，揭示了毒性降低和语言流利性之间的权衡。在更强的引导强度下，这些因果干预措施在将毒性降低高达20%方面超过了竞争基线，尽管根据攻击性的不同，GPT-2 Small的流畅性可能会显着下降。至关重要的是，转向后的标准NLP基准分数保持稳定，这表明模型的知识和一般能力得到了保留。我们进一步表明，更广泛的严重不良事件中的特征分裂会阻碍安全干预，强调了解开特征学习的重要性。我们的研究结果强调了LLM解毒基于CAE的因果干预措施的前景和当前的局限性，进一步为更安全的语言模型部署提出了实用指南。



## **21. HauntAttack: When Attack Follows Reasoning as a Shadow**

闹鬼攻击：当攻击像影子一样跟随推理时 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2506.07031v4) [paper-pdf](http://arxiv.org/pdf/2506.07031v4)

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Heming Xia, Lei Sha, Zhifang Sui

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing remarkable capabilities. However, the enhancement of reasoning abilities and the exposure of internal reasoning processes introduce new safety vulnerabilities. A critical question arises: when reasoning becomes intertwined with harmfulness, will LRMs become more vulnerable to jailbreaks in reasoning mode? To investigate this, we introduce HauntAttack, a novel and general-purpose black-box adversarial attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we modify key reasoning conditions in existing questions with harmful instructions, thereby constructing a reasoning pathway that guides the model step by step toward unsafe outputs. We evaluate HauntAttack on 11 LRMs and observe an average attack success rate of 70\%, achieving up to 12 percentage points of absolute improvement over the strongest prior baseline. Our further analysis reveals that even advanced safety-aligned models remain highly susceptible to reasoning-based attacks, offering insights into the urgent challenge of balancing reasoning capability and safety in future model development.

摘要: 新兴的大型推理模型（LRM）在数学和推理任务中始终表现出色，展现出非凡的能力。然而，推理能力的增强和内部推理过程的暴露引入了新的安全漏洞。一个关键问题出现了：当推理与危害交织在一起时，LRM是否会在推理模式中变得更容易越狱？为了研究这一点，我们引入了HauntAttack，这是一种新颖的通用黑匣子对抗攻击框架，它系统地将有害指令嵌入到推理问题中。具体来说，我们用有害指令修改现有问题中的关键推理条件，从而构建一条推理路径，引导模型逐步走向不安全的输出。我们对11种LRM进行了HauntAttack评估，观察到平均攻击成功率为70%，比之前最强的基线实现了高达12个百分点的绝对改进。我们的进一步分析表明，即使是先进的安全性一致的模型仍然极易受到基于推理的攻击，这为未来模型开发中平衡推理能力和安全性的紧迫挑战提供了见解。



## **22. Beyond Text: Multimodal Jailbreaking of Vision-Language and Audio Models through Perceptually Simple Transformations**

超越文本：通过感性简单的转换对视觉语言和音频模型进行多模式越狱 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20223v1) [paper-pdf](http://arxiv.org/pdf/2510.20223v1)

**Authors**: Divyanshu Kumar, Shreyas Jena, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable progress, yet remain critically vulnerable to adversarial attacks that exploit weaknesses in cross-modal processing. We present a systematic study of multimodal jailbreaks targeting both vision-language and audio-language models, showing that even simple perceptual transformations can reliably bypass state-of-the-art safety filters. Our evaluation spans 1,900 adversarial prompts across three high-risk safety categories harmful content, CBRN (Chemical, Biological, Radiological, Nuclear), and CSEM (Child Sexual Exploitation Material) tested against seven frontier models. We explore the effectiveness of attack techniques on MLLMs, including FigStep-Pro (visual keyword decomposition), Intelligent Masking (semantic obfuscation), and audio perturbations (Wave-Echo, Wave-Pitch, Wave-Speed). The results reveal severe vulnerabilities: models with almost perfect text-only safety (0\% ASR) suffer >75\% attack success under perceptually modified inputs, with FigStep-Pro achieving up to 89\% ASR in Llama-4 variants. Audio-based attacks further uncover provider-specific weaknesses, with even basic modality transfer yielding 25\% ASR for technical queries. These findings expose a critical gap between text-centric alignment and multimodal threats, demonstrating that current safeguards fail to generalize across cross-modal attacks. The accessibility of these attacks, which require minimal technical expertise, suggests that robust multimodal AI safety will require a paradigm shift toward broader semantic-level reasoning to mitigate possible risks.

摘要: 多模式大型语言模型（MLLM）已经取得了显着的进展，但仍然极易受到利用跨模式处理弱点的对抗攻击的影响。我们对针对视觉语言和音频语言模型的多模式越狱进行了系统性研究，表明即使是简单的感知转换也可以可靠地绕过最先进的安全过滤器。我们的评估涵盖了三个高风险安全类别有害内容、CBRN（化学、生物、放射、核）和CTEM（儿童性剥削材料）的1，900个对抗性提示，针对七个前沿模型进行了测试。我们探索了MLLM攻击技术的有效性，包括FigStep-Pro（视觉关键字分解）、智能掩蔽（语义混淆）和音频扰动（Wave-Echo、Wave-Pitch、Wave-Speed）。结果揭示了严重的漏洞：在感知修改的输入下，具有几乎完美的纯文本安全性（0\%ASB）的模型遭受了超过75%的攻击成功率，而FigStep-Pro在Lama-4变体中实现了高达89%的ASB。基于音频的攻击进一步揭示了提供商特定的弱点，即使是基本的模式传输也会产生25%的技术查询的ASB。这些发现暴露了以文本为中心的对齐和多模式威胁之间的关键差距，表明当前的保障措施未能普遍适用于跨模式攻击。这些攻击的可访问性需要最少的技术专业知识，这表明强大的多模式人工智能安全性需要范式转向更广泛的语义层面推理，以减轻可能的风险。



## **23. TRUST: A Decentralized Framework for Auditing Large Language Model Reasoning**

TRUST：审计大型语言模型推理的去中心化框架 cs.AI

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20188v1) [paper-pdf](http://arxiv.org/pdf/2510.20188v1)

**Authors**: Morris Yu-Chao Huang, Zhen Tan, Mohan Zhang, Pingzhi Li, Zhuo Zhang, Tianlong Chen

**Abstract**: Large Language Models generate complex reasoning chains that reveal their decision-making, yet verifying the faithfulness and harmlessness of these intermediate steps remains a critical unsolved problem. Existing auditing methods are centralized, opaque, and hard to scale, creating significant risks for deploying proprietary models in high-stakes domains. We identify four core challenges: (1) Robustness: Centralized auditors are single points of failure, prone to bias or attacks. (2) Scalability: Reasoning traces are too long for manual verification. (3) Opacity: Closed auditing undermines public trust. (4) Privacy: Exposing full reasoning risks model theft or distillation. We propose TRUST, a transparent, decentralized auditing framework that overcomes these limitations via: (1) A consensus mechanism among diverse auditors, guaranteeing correctness under up to $30\%$ malicious participants. (2) A hierarchical DAG decomposition of reasoning traces, enabling scalable, parallel auditing. (3) A blockchain ledger that records all verification decisions for public accountability. (4) Privacy-preserving segmentation, sharing only partial reasoning steps to protect proprietary logic. We provide theoretical guarantees for the security and economic incentives of the TRUST framework. Experiments across multiple LLMs (GPT-OSS, DeepSeek-r1, Qwen) and reasoning tasks (math, medical, science, humanities) show TRUST effectively detects reasoning flaws and remains robust against adversarial auditors. Our work pioneers decentralized AI auditing, offering a practical path toward safe and trustworthy LLM deployment.

摘要: 大型语言模型生成复杂的推理链，揭示其决策，但验证这些中间步骤的忠实性和无害性仍然是一个尚未解决的关键问题。现有的审计方法集中、不透明且难以扩展，为在高风险领域部署专有模型带来了巨大风险。我们确定了四个核心挑战：（1）稳健性：集中式审计员是单点失败，容易受到偏见或攻击。(2)可扩展性：推理轨迹太长，无法手动验证。(3)不透明：封闭审计破坏了公众信任。(4)隐私：暴露完整推理可能会导致模型被盗或提炼。我们提出TRUST，这是一个透明的、去中心化的审计框架，通过以下方式克服这些限制：（1）不同审计员之间的共识机制，保证在高达30%的恶意参与者下的正确性。(2)推理痕迹的分层DAB分解，实现可扩展的并行审计。(3)一个区块链分类帐，记录所有验证决定，以供公众问责。(4)保留隐私的分段，仅共享部分推理步骤以保护专有逻辑。我们为TRUST框架的安全性和经济激励提供理论保证。跨多个LLM（GPT-OSS、DeepSeek-r1、Qwen）和推理任务（数学、医学、科学、人文学科）的实验表明，TRUST有效地检测推理缺陷，并在对抗性审计员的情况下保持稳健性。我们的工作开创了去中心化的人工智能审计，为安全且值得信赖的LLM部署提供了实用途径。



## **24. SAID: Empowering Large Language Models with Self-Activating Internal Defense**

SAID：通过自我激活的内部防御来增强大型语言模型 cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20129v1) [paper-pdf](http://arxiv.org/pdf/2510.20129v1)

**Authors**: Yulong Chen, Yadong Liu, Jiawen Zhang, Mu Li, Chao Huang, Jie Wen

**Abstract**: Large Language Models (LLMs), despite advances in safety alignment, remain vulnerable to jailbreak attacks designed to circumvent protective mechanisms. Prevailing defense strategies rely on external interventions, such as input filtering or output modification, which often lack generalizability and compromise model utility while incurring significant computational overhead. In this work, we introduce a new, training-free defense paradigm, Self-Activating Internal Defense (SAID), which reframes the defense task from external correction to internal capability activation. SAID uniquely leverages the LLM's own reasoning abilities to proactively identify and neutralize malicious intent through a three-stage pipeline: model-native intent distillation to extract core semantics, optimal safety prefix probing to activate latent safety awareness, and a conservative aggregation strategy to ensure robust decision-making. Extensive experiments on five open-source LLMs against six advanced jailbreak attacks demonstrate that SAID substantially outperforms state-of-the-art defenses in reducing harmful outputs. Crucially, it achieves this while preserving model performance on benign tasks and incurring minimal computational overhead. Our work establishes that activating the intrinsic safety mechanisms of LLMs is a more robust and scalable path toward building safer and more reliable aligned AI systems.

摘要: 尽管在安全一致方面取得了进步，大型语言模型（LLM）仍然容易受到旨在绕过保护机制的越狱攻击。流行的防御策略依赖于外部干预，例如输入过滤或输出修改，这些干预通常缺乏可概括性并损害模型效用，同时产生大量的计算费用。在这项工作中，我们引入了一种新的免训练防御范式--自我激活内部防御（SAID），它将防御任务从外部纠正重新构建为内部能力激活。SAID独特地利用LLM自身的推理能力，通过三阶段管道主动识别和抵消恶意意图：模型原生意图提炼以提取核心语义，最佳安全前置探测以激活潜在安全意识，以及保守的聚合策略以确保稳健的决策。针对六种高级越狱攻击对五种开源LLM进行的广泛实验表明，SAID在减少有害输出方面远远优于最先进的防御。至关重要的是，它在实现这一目标的同时保留了良性任务的模型性能并产生最小的计算负担。我们的工作确定，激活LLM的本质安全机制是构建更安全、更可靠的一致人工智能系统的一条更稳健、更可扩展的途径。



## **25. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

Accepted to NAACL 2025 Main (Oral)

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2410.18469v5) [paper-pdf](http://arxiv.org/pdf/2410.18469v5)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM

摘要: 最近的研究表明，大型语言模型（LLM）很容易受到自动越狱攻击，其中由附加到有害查询的算法精心设计的对抗性后缀绕过了安全对齐并触发意外响应。当前生成这些后缀的方法计算成本高，攻击成功率（ASB）较低，尤其是针对Llama 2和Llama 3等对齐良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一种迭代自调优过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架显着降低了生成对抗性后缀的计算成本，同时在各种开源LLM上实现了近100%的ASB。此外，尽管仅在Llama 3上进行了优化，但它仍表现出对闭源模型的强大攻击转移性，在GPT-3.5上实现了99%的ASB，在GPT-4上实现了49%的ASB。除了提高越狱能力之外，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全一致研究提供了宝贵的见解。我们的代码可访问：https://github.com/SunChungEn/ADV-LLM



## **26. Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration**

Lexo：通过LLM辅助程序再生消除隐形供应链攻击 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.14522v2) [paper-pdf](http://arxiv.org/pdf/2510.14522v2)

**Authors**: Evangelos Lamprou, Julian Dai, Grigoris Ntousakis, Martin C. Rinard, Nikos Vasilakis

**Abstract**: Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. Our evaluation on 100+ real-world packages, including high profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<100s on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when prompted to do so.

摘要: 软件供应链攻击是开源软件生态系统中一个重要且持续存在的问题。这些攻击保留了组件实现的标准功能，但还隐藏了仅在组件到达其目标环境时激活的恶意功能。Lexo通过自动学习和重新生成潜在恶意组件的无可识别性版本来解决此类隐形攻击。Lexo首先生成一组输入-输出对来建模组件的完整可观察行为，然后使用其合成原始组件的新版本。新组件实现了原始功能，但避免了隐蔽的恶意行为。在整个重建过程中，Lexo会咨询大型语言模型（LLM）的几个不同实例，使用正确性和覆盖率指标来引导这些实例，并保护它们的结果。我们对100多个现实世界的包（包括高调的隐形供应链攻击）的评估表明，Lexo可以跨多个域扩展，有效地再生代码（平均<100），保持兼容性，并成功消除了几个现实世界的供应链攻击中的恶意代码，即使在最先进的LLM在提示时未能消除恶意代码的情况下也是如此。



## **27. SecureInfer: Heterogeneous TEE-GPU Architecture for Privacy-Critical Tensors for Large Language Model Deployment**

SecureInfer：用于大型语言模型部署的隐私关键张量的异类TEE-Ginger架构 cs.CR

Accepted at IEEE Intelligent Computing and Systems at the Edge  (ICEdge) 2025

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19979v1) [paper-pdf](http://arxiv.org/pdf/2510.19979v1)

**Authors**: Tushar Nayan, Ziqi Zhang, Ruimin Sun

**Abstract**: With the increasing deployment of Large Language Models (LLMs) on mobile and edge platforms, securing them against model extraction attacks has become a pressing concern. However, protecting model privacy without sacrificing the performance benefits of untrusted AI accelerators, such as GPUs, presents a challenging trade-off. In this paper, we initiate the study of high-performance execution on LLMs and present SecureInfer, a hybrid framework that leverages a heterogeneous Trusted Execution Environments (TEEs)-GPU architecture to isolate privacy-critical components while offloading compute-intensive operations to untrusted accelerators. Building upon an outsourcing scheme, SecureInfer adopts an information-theoretic and threat-informed partitioning strategy: security-sensitive components, including non-linear layers, projection of attention head, FNN transformations, and LoRA adapters, are executed inside an SGX enclave, while other linear operations (matrix multiplication) are performed on the GPU after encryption and are securely restored within the enclave. We implement a prototype of SecureInfer using the LLaMA-2 model and evaluate it across performance and security metrics. Our results show that SecureInfer offers strong security guarantees with reasonable performance, offering a practical solution for secure on-device model inference.

摘要: 随着大型语言模型（LLM）在移动和边缘平台上的部署越来越多，保护它们免受模型提取攻击已成为一个紧迫的问题。然而，在不牺牲不受信任的人工智能加速器（例如图形处理器）的性能优势的情况下保护模型隐私，提出了一个具有挑战性的权衡。在本文中，我们启动了LLM上高性能执行的研究，并提出了SecureInfer，这是一个混合框架，利用异类可信执行环境（TEEs）-图形处理器架构来隔离隐私关键组件，同时将计算密集型操作卸载到不受信任的加速器。SecureInfer在外包方案的基础上采用了信息理论和威胁知情的分区策略：安全敏感组件，包括非线性层、注意力投射、FNN变换和LoRA适配器，在SGX飞地内执行，而其他线性操作（矩阵相乘）在加密后在图形处理器上执行，并在飞地内安全地恢复。我们使用LLaMA-2模型实现SecureInfer的原型，并跨性能和安全指标对其进行评估。我们的结果表明，SecureInfer提供了强大的安全保证和合理的性能，为安全的设备上模型推断提供了实用的解决方案。



## **28. Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM**

未学习但未被遗忘：LLM中精确未学习后的数据提取 cs.LG

Accepted by Neurips 2025

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2505.24379v3) [paper-pdf](http://arxiv.org/pdf/2505.24379v3)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.

摘要: 大型语言模型通常在从网络收集的数据集上进行训练，这些数据集可能无意中包含有害或敏感的个人信息。为了解决日益增长的隐私问题，人们提出了取消学习方法来消除训练模型中特定数据的影响。其中，精确的取消学习（在没有目标数据的情况下从头开始重新训练模型）被广泛认为是减轻部署中隐私风险的黄金标准。在本文中，我们在实际部署环境中重新审视了这一假设，其中暴露了取消学习前和取消学习后的日志API，例如在开放重量场景中。针对此设置，我们引入了一种新颖的数据提取攻击，该攻击利用来自取消学习前模型的信号来指导取消学习后模型，从而发现反映已删除数据分布的模式。将模型指导与令牌过滤策略相结合，我们的攻击显着提高了MUSE、TOFU和WMDP等常见基准测试中的提取成功率，在某些情况下性能翻倍。此外，我们还展示了我们对模拟医疗诊断数据集的攻击的有效性，以强调与精确忘记相关的现实世界隐私风险。我们的研究结果表明，取消学习可能会以一种矛盾的方式增加现实世界部署期间隐私泄露的风险，鉴于此，我们主张评估取消学习方法，以考虑更广泛的威胁模型，这些模型不仅考虑取消学习后的模型，还考虑到对之前检查点的对抗访问。代码可在https://github.com/Nicholas0228/unlearned_data_extraction_llm上公开获取。



## **29. The Tail Tells All: Estimating Model-Level Membership Inference Vulnerability Without Reference Models**

The Tail Tells All：Estimating Model-Level Membership Inference Vulnerability Without Reference Models（没有参考模型的模型级成员关系推断漏洞估计） cs.LG

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19773v1) [paper-pdf](http://arxiv.org/pdf/2510.19773v1)

**Authors**: Euodia Dodd, Nataša Krčo, Igor Shilov, Yves-Alexandre de Montjoye

**Abstract**: Membership inference attacks (MIAs) have emerged as the standard tool for evaluating the privacy risks of AI models. However, state-of-the-art attacks require training numerous, often computationally expensive, reference models, limiting their practicality. We present a novel approach for estimating model-level vulnerability, the TPR at low FPR, to membership inference attacks without requiring reference models. Empirical analysis shows loss distributions to be asymmetric and heavy-tailed and suggests that most points at risk from MIAs have moved from the tail (high-loss region) to the head (low-loss region) of the distribution after training. We leverage this insight to propose a method to estimate model-level vulnerability from the training and testing distribution alone: using the absence of outliers from the high-loss region as a predictor of the risk. We evaluate our method, the TNR of a simple loss attack, across a wide range of architectures and datasets and show it to accurately estimate model-level vulnerability to the SOTA MIA attack (LiRA). We also show our method to outperform both low-cost (few reference models) attacks such as RMIA and other measures of distribution difference. We finally evaluate the use of non-linear functions to evaluate risk and show the approach to be promising to evaluate the risk in large-language models.

摘要: 成员资格推理攻击（MIA）已成为评估人工智能模型隐私风险的标准工具。然而，最先进的攻击需要训练大量且计算成本高昂的参考模型，从而限制了它们的实用性。我们提出了一种新的方法，用于估计模型级脆弱性（低FPR时的TPA），而不需要参考模型。经验分析表明损失分布是不对称的和重尾的，并表明MIA的大多数风险点在训练后已从分布的尾部（高损失区域）转移到了头部（低损失区域）。我们利用这一见解提出了一种仅根据训练和测试分布来估计模型级脆弱性的方法：使用高损失区域中不存在异常值作为风险的预测因子。我们在广泛的架构和数据集上评估了我们的方法（简单损失攻击的TNR），并展示它可以准确估计SOTA MIA攻击（LiRA）的模型级漏洞。我们还展示了我们的方法，可以优于RMIA等低成本（少数参考模型）攻击和其他分布差异测量。我们最终评估了使用非线性函数来评估风险，并展示了有希望在大型语言模型中评估风险的方法。



## **30. Can You Trust What You See? Alpha Channel No-Box Attacks on Video Object Detection**

你能相信你所看到的吗？Alpha通道对视频对象检测的无框攻击 cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19574v1) [paper-pdf](http://arxiv.org/pdf/2510.19574v1)

**Authors**: Ariana Yi, Ce Zhou, Liyang Xiao, Qiben Yan

**Abstract**: As object detection models are increasingly deployed in cyber-physical systems such as autonomous vehicles (AVs) and surveillance platforms, ensuring their security against adversarial threats is essential. While prior work has explored adversarial attacks in the image domain, those attacks in the video domain remain largely unexamined, especially in the no-box setting. In this paper, we present {\alpha}-Cloak, the first no-box adversarial attack on object detectors that operates entirely through the alpha channel of RGBA videos. {\alpha}-Cloak exploits the alpha channel to fuse a malicious target video with a benign video, resulting in a fused video that appears innocuous to human viewers but consistently fools object detectors. Our attack requires no access to model architecture, parameters, or outputs, and introduces no perceptible artifacts. We systematically study the support for alpha channels across common video formats and playback applications, and design a fusion algorithm that ensures visual stealth and compatibility. We evaluate {\alpha}-Cloak on five state-of-the-art object detectors, a vision-language model, and a multi-modal large language model (Gemini-2.0-Flash), demonstrating a 100% attack success rate across all scenarios. Our findings reveal a previously unexplored vulnerability in video-based perception systems, highlighting the urgent need for defenses that account for the alpha channel in adversarial settings.

摘要: 随着对象检测模型越来越多地部署在自动驾驶汽车（AV）和监控平台等网络物理系统中，确保其针对对抗威胁的安全性至关重要。虽然之前的工作探讨了图像领域中的对抗攻击，但视频领域中的这些攻击在很大程度上仍然没有得到审查，尤其是在无框环境中。在本文中，我们介绍了{\Alpha}-Cloak，这是对对象检测器的第一个无箱对抗攻击，完全通过RGBA视频的Alpha通道进行操作。{\Alpha}-Cloak利用Alpha通道将恶意目标视频与良性视频融合，导致融合后的视频对人类观看者来说似乎无害，但始终欺骗对象检测器。我们的攻击不需要访问模型架构、参数或输出，并且不会引入可感知的伪影。我们系统性地研究了对常见视频格式和播放应用程序中Alpha通道的支持，并设计了一种融合算法来确保视觉隐形性和兼容性。我们在五个最先进的对象检测器、视觉语言模型和多模式大型语言模型（Gemini-2.0-Flash）上评估了{\Alpha}-Cloak，展示了在所有场景下100%的攻击成功率。我们的研究结果揭示了基于视频的感知系统中以前未探索的漏洞，凸显了对对抗环境中阿尔法通道的防御的迫切需要。



## **31. Machine Text Detectors are Membership Inference Attacks**

机器文本检测器是会员推断攻击 cs.CL

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19492v1) [paper-pdf](http://arxiv.org/pdf/2510.19492v1)

**Authors**: Ryuto Koike, Liam Dugan, Masahiro Kaneko, Chris Callison-Burch, Naoaki Okazaki

**Abstract**: Although membership inference attacks (MIAs) and machine-generated text detection target different goals, identifying training samples and synthetic texts, their methods often exploit similar signals based on a language model's probability distribution. Despite this shared methodological foundation, the two tasks have been independently studied, which may lead to conclusions that overlook stronger methods and valuable insights developed in the other task. In this work, we theoretically and empirically investigate the transferability, i.e., how well a method originally developed for one task performs on the other, between MIAs and machine text detection. For our theoretical contribution, we prove that the metric that achieves the asymptotically highest performance on both tasks is the same. We unify a large proportion of the existing literature in the context of this optimal metric and hypothesize that the accuracy with which a given method approximates this metric is directly correlated with its transferability. Our large-scale empirical experiments, including 7 state-of-the-art MIA methods and 5 state-of-the-art machine text detectors across 13 domains and 10 generators, demonstrate very strong rank correlation (rho > 0.6) in cross-task performance. We notably find that Binoculars, originally designed for machine text detection, achieves state-of-the-art performance on MIA benchmarks as well, demonstrating the practical impact of the transferability. Our findings highlight the need for greater cross-task awareness and collaboration between the two research communities. To facilitate cross-task developments and fair evaluations, we introduce MINT, a unified evaluation suite for MIAs and machine-generated text detection, with implementation of 15 recent methods from both tasks.

摘要: 尽管成员资格推理攻击（MIA）和机器生成的文本检测针对不同的目标，识别训练样本和合成文本，但它们的方法通常基于语言模型的概率分布利用相似的信号。尽管有这一共同的方法论基础，但这两项任务是独立研究的，这可能会得出的结论忽视了另一项任务中开发的更强的方法和有价值的见解。在这项工作中，我们从理论和经验上研究了可转让性，即，最初为一项任务开发的方法在MIA和机器文本检测之间在另一项任务上的表现如何。对于我们的理论贡献，我们证明在两项任务中实现渐进最高性能的指标是相同的。我们在这个最佳指标的背景下统一了大部分现有文献，并假设给定方法逼近这个指标的准确性与其可移植性直接相关。我们的大规模实证实验，包括7种最先进的MIA方法和5种最先进的机器文本检测器，跨越13个域和10个生成器，证明了跨任务性能中非常强的等级相关性（rho > 0.6）。我们特别发现，最初为机器文本检测而设计的双筒望远镜在MIA基准测试上也实现了最先进的性能，证明了可移植性的实际影响。我们的研究结果强调了两个研究团体之间需要更大的跨任务意识和合作。为了促进跨任务开发和公平评估，我们引入了MINT，这是一个用于MIA和机器生成文本检测的统一评估套件，并从这两个任务中实现了15种最新方法。



## **32. Monitoring LLM-based Multi-Agent Systems Against Corruptions via Node Evaluation**

通过节点评估监控基于LLM的多代理系统防止损坏 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19420v1) [paper-pdf](http://arxiv.org/pdf/2510.19420v1)

**Authors**: Chengcan Wu, Zhixin Zhang, Mingqian Xu, Zeming Wei, Meng Sun

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have become a popular paradigm of AI applications. However, trustworthiness issues in MAS remain a critical concern. Unlike challenges in single-agent systems, MAS involve more complex communication processes, making them susceptible to corruption attacks. To mitigate this issue, several defense mechanisms have been developed based on the graph representation of MAS, where agents represent nodes and communications form edges. Nevertheless, these methods predominantly focus on static graph defense, attempting to either detect attacks in a fixed graph structure or optimize a static topology with certain defensive capabilities. To address this limitation, we propose a dynamic defense paradigm for MAS graph structures, which continuously monitors communication within the MAS graph, then dynamically adjusts the graph topology, accurately disrupts malicious communications, and effectively defends against evolving and diverse dynamic attacks. Experimental results in increasingly complex and dynamic MAS environments demonstrate that our method significantly outperforms existing MAS defense mechanisms, contributing an effective guardrail for their trustworthy applications. Our code is available at https://github.com/ChengcanWu/Monitoring-LLM-Based-Multi-Agent-Systems.

摘要: 基于大语言模型（LLM）的多智能体系统（MAS）已成为人工智能应用的流行范式。然而，MAS的可信度问题仍然是一个关键问题。与单代理系统中的挑战不同，MAS涉及更复杂的通信过程，使其容易受到腐败攻击。为了缓解这个问题，基于MAS的图表示开发了多种防御机制，其中代理代表节点，通信形成边。然而，这些方法主要关注静态图防御，试图检测固定图结构中的攻击或优化具有某些防御能力的静态布局。为了解决这一局限性，我们提出了一种针对MAS图结构的动态防御范式，该范式持续监控MAS图内的通信，然后动态调整图布局，准确干扰恶意通信，并有效防御不断发展和多样化的动态攻击。在日益复杂和动态的MAS环境中的实验结果表明，我们的方法显着优于现有的MAS防御机制，为其值得信赖的应用提供了有效的护栏。我们的代码可在https://github.com/ChengcanWu/Monitoring-LLM-Based-Multi-Agent-Systems上获取。



## **33. Defending Against Prompt Injection with DataFilter**

使用数据过滤器防御提示注入 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19207v1) [paper-pdf](http://arxiv.org/pdf/2510.19207v1)

**Authors**: Yizhu Wang, Sizhe Chen, Raghad Alkhudair, Basel Alomair, David Wagner

**Abstract**: When large language model (LLM) agents are increasingly deployed to automate tasks and interact with untrusted external data, prompt injection emerges as a significant security threat. By injecting malicious instructions into the data that LLMs access, an attacker can arbitrarily override the original user task and redirect the agent toward unintended, potentially harmful actions. Existing defenses either require access to model weights (fine-tuning), incur substantial utility loss (detection-based), or demand non-trivial system redesign (system-level). Motivated by this, we propose DataFilter, a test-time model-agnostic defense that removes malicious instructions from the data before it reaches the backend LLM. DataFilter is trained with supervised fine-tuning on simulated injections and leverages both the user's instruction and the data to selectively strip adversarial content while preserving benign information. Across multiple benchmarks, DataFilter consistently reduces the prompt injection attack success rates to near zero while maintaining the LLMs' utility. DataFilter delivers strong security, high utility, and plug-and-play deployment, making it a strong practical defense to secure black-box commercial LLMs against prompt injection. Our DataFilter model is released at https://huggingface.co/JoyYizhu/DataFilter for immediate use, with the code to reproduce our results at https://github.com/yizhu-joy/DataFilter.

摘要: 当大型语言模型（LLM）代理越来越多地被部署来自动化任务并与不受信任的外部数据交互时，即时注入成为一个重大的安全威胁。通过将恶意指令注入LLM访问的数据中，攻击者可以任意覆盖原始用户任务，并将代理重定向到无意的、可能有害的操作。现有的防御要么需要访问模型权重（微调），导致大量效用损失（基于检测），要么要求进行非平凡的系统重新设计（系统级）。出于此动机，我们提出了数据过滤器，这是一种测试时模型不可知的防御，可以在数据到达后台LLM之前从数据中删除恶意指令。数据过滤器通过模拟注入的监督微调进行训练，并利用用户的指令和数据来选择性地剥离对抗内容，同时保留良性信息。在多个基准测试中，数据过滤器一致地将即时注入攻击成功率降低到接近零，同时保持LLM的实用性。数据过滤器提供强大的安全性、高实用性和即插即用部署，使其成为保护黑匣子商业LLM免受即时注入的强大实用防御。我们的数据过滤器模型已在https://huggingface.co/JoyYizhu/DataFilter上发布，可立即使用，其代码可在https://github.com/yizhu-joy/DataFilter上重现我们的结果。



## **34. OpenGuardrails: An Open-Source Context-Aware AI Guardrails Platform**

OpenGuardrails：一个开源上下文感知人工智能Guardrails平台 cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19169v1) [paper-pdf](http://arxiv.org/pdf/2510.19169v1)

**Authors**: Thomas Wang, Haowen Li

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications, safeguarding them against unsafe, malicious, or privacy-violating content is critically important. We present OpenGuardrails, the first open-source project to provide both a context-aware safety and manipulation detection model and a deployable platform for comprehensive AI guardrails. OpenGuardrails protects against content-safety risks, model-manipulation attacks (e.g., prompt injection, jailbreaking, code-interpreter abuse, and the generation/execution of malicious code), and data leakage. Content-safety and model-manipulation detection are implemented by a unified large model, while data-leakage identification and redaction are performed by a separate lightweight NER pipeline (e.g., Presidio-style models or regex-based detectors). The system can be deployed as a security gateway or an API-based service, with enterprise-grade, fully private deployment options. OpenGuardrails achieves state-of-the-art (SOTA) performance on safety benchmarks, excelling in both prompt and response classification across English, Chinese, and multilingual tasks. All models are released under the Apache 2.0 license for public use.

摘要: 随着大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，保护它们免受不安全、恶意或侵犯隐私的内容的侵害至关重要。我们介绍了OpenGuardrails，这是第一个提供上下文感知安全和操纵检测模型以及全面人工智能护栏可部署平台的开源项目。OpenGuardrails可防止内容安全风险、模型操纵攻击（例如，提示注入、越狱、代码解释器滥用以及恶意代码的生成/执行）以及数据泄露。内容安全和模型操纵检测由统一的大模型实现，而数据泄露识别和编辑由单独的轻量级NER管道执行（例如，Presidio风格模型或基于regex的检测器）。该系统可以部署为安全网关或基于API的服务，并具有企业级的完全私有部署选项。OpenGuardrails在安全基准方面实现了最先进的（SOTA）性能，在英语、中文和多语言任务中的提示和响应分类方面都表现出色。所有模型均在Apache 2.0许可下发布，供公众使用。



## **35. PLAGUE: Plug-and-play framework for Lifelong Adaptive Generation of Multi-turn Exploits**

PLAGUE：终身自适应多回合漏洞生成的即插即用框架 cs.CR

First two authors have equal author contributions

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.17947v2) [paper-pdf](http://arxiv.org/pdf/2510.17947v2)

**Authors**: Neeladri Bhuiya, Madhav Aggarwal, Diptanshu Purwar

**Abstract**: Large Language Models (LLMs) are improving at an exceptional rate. With the advent of agentic workflows, multi-turn dialogue has become the de facto mode of interaction with LLMs for completing long and complex tasks. While LLM capabilities continue to improve, they remain increasingly susceptible to jailbreaking, especially in multi-turn scenarios where harmful intent can be subtly injected across the conversation to produce nefarious outcomes. While single-turn attacks have been extensively explored, adaptability, efficiency and effectiveness continue to remain key challenges for their multi-turn counterparts. To address these gaps, we present PLAGUE, a novel plug-and-play framework for designing multi-turn attacks inspired by lifelong-learning agents. PLAGUE dissects the lifetime of a multi-turn attack into three carefully designed phases (Primer, Planner and Finisher) that enable a systematic and information-rich exploration of the multi-turn attack family. Evaluations show that red-teaming agents designed using PLAGUE achieve state-of-the-art jailbreaking results, improving attack success rates (ASR) by more than 30% across leading models in a lesser or comparable query budget. Particularly, PLAGUE enables an ASR (based on StrongReject) of 81.4% on OpenAI's o3 and 67.3% on Claude's Opus 4.1, two models that are considered highly resistant to jailbreaks in safety literature. Our work offers tools and insights to understand the importance of plan initialization, context optimization and lifelong learning in crafting multi-turn attacks for a comprehensive model vulnerability evaluation.

摘要: 大型语言模型（LLM）正在以惊人的速度改进。随着代理工作流程的出现，多轮对话已成为与LLM互动的事实模式，以完成漫长而复杂的任务。虽然LLM能力不断提高，但它们仍然越来越容易受到越狱的影响，特别是在多回合场景中，有害意图可能会巧妙地注入到整个对话中以产生邪恶结果。虽然单回合攻击已得到广泛探索，但适应性、效率和有效性仍然是多回合攻击的关键挑战。为了解决这些差距，我们提出了PLAGUE，这是一个新颖的即插即用框架，用于设计受终身学习代理启发的多回合攻击。PLAGUE将多回合攻击的生命周期分解为三个精心设计的阶段（Primer、Planner和Timisher），以便对多回合攻击家族进行系统性且信息丰富的探索。评估表明，使用PLAGUE设计的红色团队代理实现了最先进的越狱结果，以较少或相当的查询预算将领先模型的攻击成功率（ASB）提高了30%以上。特别是，PLAGUE在OpenAI的o3上的ASB（基于Strongestival）为81.4%，在Claude的Opus 4.1上为67.3%，这两种模型在安全文献中被认为对越狱具有高度抵抗力。我们的工作提供了工具和见解，以了解计划初始化、上下文优化和终身学习在制定多回合攻击以进行全面模型漏洞评估方面的重要性。



## **36. NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks**

NEXUS：在多轮LLM越狱中利用不安全序列的网络探索 cs.CR

This paper has been accepted in the main conference proceedings of  the 2025 Conference on Empirical Methods in Natural Language Processing  (EMNLP 2025). Javad Rafiei Asl and Sidhant Narula are co-first authors

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.03417v2) [paper-pdf](http://arxiv.org/pdf/2510.03417v2)

**Authors**: Javad Rafiei Asl, Sidhant Narula, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: https://github.com/inspire-lab/NEXUS

摘要: 大型语言模型（LLM）彻底改变了自然语言处理，但仍然容易受到越狱攻击，特别是在良性交换中散布恶意意图并绕过对齐机制的多回合越狱。现有的方法常常无法很好地探索对抗空间，依赖于手工制作的启发式方法，或者缺乏系统性的查询细化。我们介绍了NEXUS（用于eXploiting Unsafe Sequences的网络探索），这是一个用于构建、细化和执行优化多回合攻击的模块化框架。NEXUS包括：（1）IncreghtNet，它将有害意图分层扩展到主题、实体和查询链的结构化语义网络中;（2）反馈驱动的模拟器，通过攻击者-受害者-法官LLM协作使用危害性和语义相似性基准来迭代细化和修剪这些链;（3）网络穿越器，自适应地导航细化查询空间以进行实时攻击。该管道揭示了LLC之间隐秘、高成功的对抗路径。在几种闭源和开源LLM上，NEXUS将攻击成功率比之前的方法提高了2.1%至19.4%。代码：https://github.com/inspire-lab/NEXUS



## **37. HarmNet: A Framework for Adaptive Multi-Turn Jailbreak Attacks on Large Language Models**

HarmNet：对大型语言模型进行自适应多回合越狱攻击的框架 cs.CR

This paper has been accepted for presentation at the Conference on  Applied Machine Learning in Information Security (CAMLIS 2025)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18728v1) [paper-pdf](http://arxiv.org/pdf/2510.18728v1)

**Authors**: Sidhant Narula, Javad Rafiei Asl, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) remain vulnerable to multi-turn jailbreak attacks. We introduce HarmNet, a modular framework comprising ThoughtNet, a hierarchical semantic network; a feedback-driven Simulator for iterative query refinement; and a Network Traverser for real-time adaptive attack execution. HarmNet systematically explores and refines the adversarial space to uncover stealthy, high-success attack paths. Experiments across closed-source and open-source LLMs show that HarmNet outperforms state-of-the-art methods, achieving higher attack success rates. For example, on Mistral-7B, HarmNet achieves a 99.4% attack success rate, 13.9% higher than the best baseline. Index terms: jailbreak attacks; large language models; adversarial framework; query refinement.

摘要: 大型语言模型（LLM）仍然容易受到多回合越狱攻击。我们引入了HarmNet，这是一个模块化框架，包括分层语义网络InghtNet;用于迭代查询细化的反馈驱动模拟器;以及用于实时自适应攻击执行的Network Traverser。HarmNet系统性地探索和完善对抗空间，以发现隐秘、高成功的攻击路径。跨闭源和开源LLM的实验表明，HarmNet优于最先进的方法，实现了更高的攻击成功率。例如，在Mistral-7 B上，HarmNet的攻击成功率达到了99.4%，比最佳基线高出13.9%。索引术语：越狱攻击;大型语言模型;对抗性框架;查询细化。



## **38. Exploring Membership Inference Vulnerabilities in Clinical Large Language Models**

探索临床大型语言模型中的隶属推理漏洞 cs.CR

Accepted at the 1st IEEE Workshop on Healthcare and Medical Device  Security, Privacy, Resilience, and Trust (IEEE HMD-SPiRiT)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18674v1) [paper-pdf](http://arxiv.org/pdf/2510.18674v1)

**Authors**: Alexander Nemecek, Zebin Yun, Zahra Rahmani, Yaniv Harel, Vipin Chaudhary, Mahmood Sharif, Erman Ayday

**Abstract**: As large language models (LLMs) become progressively more embedded in clinical decision-support, documentation, and patient-information systems, ensuring their privacy and trustworthiness has emerged as an imperative challenge for the healthcare sector. Fine-tuning LLMs on sensitive electronic health record (EHR) data improves domain alignment but also raises the risk of exposing patient information through model behaviors. In this work-in-progress, we present an exploratory empirical study on membership inference vulnerabilities in clinical LLMs, focusing on whether adversaries can infer if specific patient records were used during model training. Using a state-of-the-art clinical question-answering model, Llemr, we evaluate both canonical loss-based attacks and a domain-motivated paraphrasing-based perturbation strategy that more realistically reflects clinical adversarial conditions. Our preliminary findings reveal limited but measurable membership leakage, suggesting that current clinical LLMs provide partial resistance yet remain susceptible to subtle privacy risks that could undermine trust in clinical AI adoption. These results motivate continued development of context-aware, domain-specific privacy evaluations and defenses such as differential privacy fine-tuning and paraphrase-aware training, to strengthen the security and trustworthiness of healthcare AI systems.

摘要: 随着大型语言模型（LLM）越来越嵌入临床决策支持、文档和患者信息系统中，确保其隐私和可信度已成为医疗保健行业的一个紧迫挑战。对敏感电子健康记录（EHR）数据进行微调LLM可以改善域对齐，但也会增加通过模型行为暴露患者信息的风险。在这项正在进行的工作中，我们对临床LLM中的隶属关系推断漏洞进行了一项探索性实证研究，重点关注对手是否可以推断模型训练期间是否使用了特定的患者记录。使用最先进的临床问答模型Llemr，我们评估了典型的基于损失的攻击和更真实地反映临床对抗状况的基于领域动机的基于重述的扰动策略。我们的初步研究结果揭示了有限但可测量的会员泄露，这表明当前的临床LLM提供了部分抵抗力，但仍然容易受到微妙的隐私风险，这可能会破坏人们对临床人工智能采用的信任。这些结果激励了上下文感知、特定领域隐私评估和防御的持续发展，例如差异隐私微调和转述感知训练，以加强医疗保健人工智能系统的安全性和可信度。



## **39. SentinelNet: Safeguarding Multi-Agent Collaboration Through Credit-Based Dynamic Threat Detection**

SentinelNet：通过基于信用的动态威胁检测保护多代理协作 cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.16219v2) [paper-pdf](http://arxiv.org/pdf/2510.16219v2)

**Authors**: Yang Feng, Xudong Pan

**Abstract**: Malicious agents pose significant threats to the reliability and decision-making capabilities of Multi-Agent Systems (MAS) powered by Large Language Models (LLMs). Existing defenses often fall short due to reactive designs or centralized architectures which may introduce single points of failure. To address these challenges, we propose SentinelNet, the first decentralized framework for proactively detecting and mitigating malicious behaviors in multi-agent collaboration. SentinelNet equips each agent with a credit-based detector trained via contrastive learning on augmented adversarial debate trajectories, enabling autonomous evaluation of message credibility and dynamic neighbor ranking via bottom-k elimination to suppress malicious communications. To overcome the scarcity of attack data, it generates adversarial trajectories simulating diverse threats, ensuring robust training. Experiments on MAS benchmarks show SentinelNet achieves near-perfect detection of malicious agents, close to 100% within two debate rounds, and recovers 95% of system accuracy from compromised baselines. By exhibiting strong generalizability across domains and attack patterns, SentinelNet establishes a novel paradigm for safeguarding collaborative MAS.

摘要: 恶意代理对大型语言模型（LLM）支持的多代理系统（MAS）的可靠性和决策能力构成重大威胁。由于反应式设计或集中式架构可能会引入单点故障，现有的防御往往会出现缺陷。为了应对这些挑战，我们提出了SentinelNet，这是第一个用于主动检测和缓解多代理协作中恶意行为的去中心化框架。SentinelNet为每个代理配备了一个基于信用的检测器，该检测器通过对增强的对抗辩论轨迹进行对比学习进行训练，从而能够通过底部k消除来自主评估消息可信度和动态邻居排名，以抑制恶意通信。为了克服攻击数据的稀缺性，它生成模拟不同威胁的对抗轨迹，确保稳健的训练。MAS基准测试的实验表明，SentinelNet实现了对恶意代理近乎完美的检测，在两轮辩论中接近100%，并从受损的基线恢复了95%的系统准确性。SentinelNet在跨域和攻击模式之间表现出强大的概括性，建立了一种用于保护协作MAS的新颖范式。



## **40. The Attribution Story of WhisperGate: An Academic Perspective**

WhisperGate的归因故事：学术视角 cs.CR

Virus Bulletin Conference 2025

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18484v1) [paper-pdf](http://arxiv.org/pdf/2510.18484v1)

**Authors**: Oleksandr Adamov, Anders Carlsson

**Abstract**: This paper explores the challenges of cyberattack attribution, specifically APTs, applying the case study approach for the WhisperGate cyber operation of January 2022 executed by the Russian military intelligence service (GRU) and targeting Ukrainian government entities. The study provides a detailed review of the threat actor identifiers and taxonomies used by leading cybersecurity vendors, focusing on the evolving attribution from Microsoft, ESET, and CrowdStrike researchers. Once the attribution to Ember Bear (GRU Unit 29155) is established through technical and intelligence reports, we use both traditional machine learning classifiers and a large language model (ChatGPT) to analyze the indicators of compromise (IoCs), tactics, and techniques to statistically and semantically attribute the WhisperGate attack. Our findings reveal overlapping indicators with the Sandworm group (GRU Unit 74455) but also strong evidence pointing to Ember Bear, especially when the LLM is fine-tuned or contextually augmented with additional intelligence. Thus, showing how AI/GenAI with proper fine-tuning are capable of solving the attribution challenge.

摘要: 本文采用俄罗斯军事情报部门（GRU）2022年1月执行的针对乌克兰政府实体的WhisperGate网络行动的案例研究方法，探讨了网络攻击归因（特别是APT）的挑战。该研究详细回顾了领先网络安全供应商使用的威胁行为者标识符和分类法，重点关注Microsoft、ESET和CrowdStrike研究人员不断变化的归因。一旦通过技术和情报报告确定了Ember Bear（GRU Unit 29155）的归因，我们就使用传统的机器学习分类器和大型语言模型（ChatGPT）来分析妥协指标（IoCs）、策略和技术，以统计和语义上对WhisperGate攻击进行归因。我们的研究结果揭示了与Sandworm小组（GRU Unit 74455）重叠的指标，但也有强有力的证据指向Ember Bear，特别是当LLM经过微调或根据上下文使用额外的情报增强时。因此，展示了经过适当微调的AI/GenAI如何能够解决归因挑战。



## **41. SoK: Taxonomy and Evaluation of Prompt Security in Large Language Models**

SoK：大型语言模型中提示安全性的分类和评估 cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.15476v2) [paper-pdf](http://arxiv.org/pdf/2510.15476v2)

**Authors**: Hanbin Hong, Shuya Feng, Nima Naderloui, Shenao Yan, Jingyu Zhang, Biying Liu, Ali Arastehfard, Heqing Huang, Yuan Hong

**Abstract**: Large Language Models (LLMs) have rapidly become integral to real-world applications, powering services across diverse sectors. However, their widespread deployment has exposed critical security risks, particularly through jailbreak prompts that can bypass model alignment and induce harmful outputs. Despite intense research into both attack and defense techniques, the field remains fragmented: definitions, threat models, and evaluation criteria vary widely, impeding systematic progress and fair comparison. In this Systematization of Knowledge (SoK), we address these challenges by (1) proposing a holistic, multi-level taxonomy that organizes attacks, defenses, and vulnerabilities in LLM prompt security; (2) formalizing threat models and cost assumptions into machine-readable profiles for reproducible evaluation; (3) introducing an open-source evaluation toolkit for standardized, auditable comparison of attacks and defenses; (4) releasing JAILBREAKDB, the largest annotated dataset of jailbreak and benign prompts to date;\footnote{The dataset is released at \href{https://huggingface.co/datasets/youbin2014/JailbreakDB}{\textcolor{purple}{https://huggingface.co/datasets/youbin2014/JailbreakDB}}.} and (5) presenting a comprehensive evaluation platform and leaderboard of state-of-the-art methods \footnote{will be released soon.}. Our work unifies fragmented research, provides rigorous foundations for future studies, and supports the development of robust, trustworthy LLMs suitable for high-stakes deployment.

摘要: 大型语言模型（LLM）已迅速成为现实世界应用程序的组成部分，为不同领域的服务提供动力。然而，它们的广泛部署暴露了严重的安全风险，特别是通过越狱提示，这些提示可能绕过模型对齐并引发有害输出。尽管对攻击和防御技术进行了深入的研究，但该领域仍然支离破碎：定义、威胁模型和评估标准差异很大，阻碍了系统性进步和公平比较。在知识系统化（SoK）中，我们通过以下方式解决这些挑战：（1）提出一种整体、多级别的分类法，组织LLM即时安全中的攻击、防御和漏洞;（2）将威胁模型和成本假设形式化为机器可读配置文件，以进行可重复的评估;（3）引入开源评估工具包，用于标准化、可审计的攻击和防御比较;（4）发布JAILBREAKDB，这是迄今为止最大的越狱和良性提示注释数据集;\脚注{该数据集发布于\href{https：//huggingface.co/juets/youbin2014/JailbreakDB}{\textColor{purple}{https：//huggingface.co/juets/youbin2014/JailbreakDB}。}以及（5）提供全面的评估平台和最先进方法排行榜\脚注{将很快发布。}。我们的工作统一了碎片化的研究，为未来的研究提供了严格的基础，并支持开发适合高风险部署的稳健、值得信赖的LLM。



## **42. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

早起的鸟儿抓住了漏洞：在LLM服务系统中揭开定时侧通道 cs.CR

This work was first submitted for review on Sept. 5, 2024, and the  initial version was uploaded to Arxiv on Sept. 30, 2024. The latest version  has accepted for publication by IEEE Transactions on Information Forensics  and Security (TIFS)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2409.20002v5) [paper-pdf](http://arxiv.org/pdf/2409.20002v5)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.

摘要: 大型语言模型（LLM）的广泛部署引发了对其推理性能优化的强烈要求。当今服务于此目的的技术主要集中在通过算法和硬件增强来减少延迟和提高吞吐量，同时在很大程度上忽视了它们的隐私副作用，特别是在多用户环境中。在我们的研究中，我们首次在LLM系统中发现了一组新的计时侧通道，这些通道由共享缓存和图形处理器内存分配产生，可以利用这些通道来推断机密系统提示和其他用户发出的提示。这些漏洞呼应了传统计算系统中观察到的安全挑战，凸显了解决LLM服务基础设施中潜在信息泄露的迫切需要。在本文中，我们报告了旨在利用LLM部署中固有的此类时间侧通道的新颖攻击策略，特别针对广泛用于增强LLM推理性能的Key-Value（KV）缓存和语义缓存。我们的方法利用时间测量和分类模型来检测缓存命中，使对手能够高准确地推断私人提示。我们还提出了一种逐令牌搜索算法来有效地恢复缓存中的共享提示前置，展示了窃取系统提示和对等用户产生的提示的可行性。我们对流行的在线LLM服务进行黑匣子测试的实验研究表明，此类隐私风险是完全现实的，并会产生重大后果。我们的研究结果强调需要强有力的缓解措施来保护LLM系统免受此类新出现的威胁。



## **43. Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming**

Genesis：LLM Web Agent Red-Teaming不断发展的攻击策略 cs.AI

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18314v1) [paper-pdf](http://arxiv.org/pdf/2510.18314v1)

**Authors**: Zheng Zhang, Jiarui He, Yuchen Cai, Deheng Ye, Peilin Zhao, Ruili Feng, Hao Wang

**Abstract**: As large language model (LLM) agents increasingly automate complex web tasks, they boost productivity while simultaneously introducing new security risks. However, relevant studies on web agent attacks remain limited. Existing red-teaming approaches mainly rely on manually crafted attack strategies or static models trained offline. Such methods fail to capture the underlying behavioral patterns of web agents, making it difficult to generalize across diverse environments. In web agent attacks, success requires the continuous discovery and evolution of attack strategies. To this end, we propose Genesis, a novel agentic framework composed of three modules: Attacker, Scorer, and Strategist. The Attacker generates adversarial injections by integrating the genetic algorithm with a hybrid strategy representation. The Scorer evaluates the target web agent's responses to provide feedback. The Strategist dynamically uncovers effective strategies from interaction logs and compiles them into a continuously growing strategy library, which is then re-deployed to enhance the Attacker's effectiveness. Extensive experiments across various web tasks show that our framework discovers novel strategies and consistently outperforms existing attack baselines.

摘要: 随着大型语言模型（LLM）代理越来越多地自动化复杂的Web任务，它们提高了生产力，同时引入了新的安全风险。然而，关于Web代理攻击的相关研究仍然有限。现有的红色团队方法主要依赖于手动设计的攻击策略或离线训练的静态模型。此类方法无法捕捉Web代理的底层行为模式，因此很难在不同的环境中进行概括。在Web代理攻击中，成功需要不断发现和进化攻击策略。为此，我们提出了Genesis，这是一个由三个模块组成的新颖的代理框架：攻击者、得分者和策略者。攻击者通过将遗传算法与混合策略表示集成来生成对抗注入。评分者评估目标Web代理的响应以提供反馈。策略师从交互日志中动态发现有效的策略，并将其汇编到不断增长的策略库中，然后重新部署该库以增强攻击者的有效性。跨各种Web任务的广泛实验表明，我们的框架发现了新颖的策略，并且始终优于现有的攻击基线。



## **44. Secure and Efficient Access Control for Computer-Use Agents via Context Space**

通过上下文空间对计算机使用代理进行安全有效的访问控制 cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2509.22256v2) [paper-pdf](http://arxiv.org/pdf/2509.22256v2)

**Authors**: Haochen Gong, Chenxiao Li, Rui Chang, Wenbo Shen

**Abstract**: Large language model (LLM)-based computer-use agents represent a convergence of AI and OS capabilities, enabling natural language to control system- and application-level functions. However, due to LLMs' inherent uncertainty issues, granting agents control over computers poses significant security risks. When agent actions deviate from user intentions, they can cause irreversible consequences. Existing mitigation approaches, such as user confirmation and LLM-based dynamic action validation, still suffer from limitations in usability, security, and performance. To address these challenges, we propose CSAgent, a system-level, static policy-based access control framework for computer-use agents. To bridge the gap between static policy and dynamic context and user intent, CSAgent introduces intent- and context-aware policies, and provides an automated toolchain to assist developers in constructing and refining them. CSAgent enforces these policies through an optimized OS service, ensuring that agent actions can only be executed under specific user intents and contexts. CSAgent supports protecting agents that control computers through diverse interfaces, including API, CLI, and GUI. We implement and evaluate CSAgent, which successfully defends against more than 99.36% of attacks while introducing only 6.83% performance overhead.

摘要: 基于大型语言模型（LLM）的计算机使用代理代表了人工智能和操作系统功能的融合，使自然语言能够控制系统和应用程序级功能。然而，由于LLM固有的不确定性问题，授予代理对计算机的控制权会带来巨大的安全风险。当代理行为偏离用户意图时，可能会导致不可逆转的后果。现有的缓解方法，例如用户确认和基于LLM的动态动作验证，仍然受到可用性、安全性和性能方面的限制。为了解决这些挑战，我们提出了CSAgent，这是一个用于计算机使用代理的系统级、基于静态策略的访问控制框架。为了弥合静态策略与动态上下文和用户意图之间的差距，CSAgent引入了意图和上下文感知策略，并提供了自动化工具链来帮助开发人员构建和完善它们。CSAgent通过优化的操作系统服务执行这些策略，确保代理操作只能在特定的用户意图和上下文下执行。CSAgent支持保护通过各种接口（包括API、CLI和图形用户界面）控制计算机的代理。我们实施并评估CSAgent，它成功防御了超过99.36%的攻击，同时仅引入了6.83%的性能负载。



## **45. DrunkAgent: Stealthy Memory Corruption in LLM-Powered Recommender Agents**

DrunkAgent：LLM-Powered Recommender Agent中的隐形内存损坏 cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2503.23804v3) [paper-pdf](http://arxiv.org/pdf/2503.23804v3)

**Authors**: Shiyi Yang, Zhibo Hu, Xinshu Li, Chen Wang, Tong Yu, Xiwei Xu, Liming Zhu, Lina Yao

**Abstract**: Large language model (LLM)-powered agents are increasingly used in recommender systems (RSs) to achieve personalized behavior modeling, where the memory mechanism plays a pivotal role in enabling the agents to autonomously explore, learn and self-evolve from real-world interactions. However, this very mechanism, serving as a contextual repository, inherently exposes an attack surface for potential adversarial manipulations. Despite its central role, the robustness of agentic RSs in the face of such threats remains largely underexplored. Previous works suffer from semantic mismatches or rely on static embeddings or pre-defined prompts, all of which are not designed for dynamic systems, especially for dynamic memory states of LLM agents. This challenge is exacerbated by the black-box nature of commercial recommenders.   To tackle the above problems, in this paper, we present the first systematic investigation of memory-based vulnerabilities in LLM-powered recommender agents, revealing their security limitations and guiding efforts to strengthen system resilience and trustworthiness. Specifically, we propose a novel black-box attack framework named DrunkAgent. DrunkAgent crafts semantically meaningful adversarial textual triggers for target item promotions and introduces a series of strategies to maximize the trigger effect by corrupting the memory updates during the interactions. The triggers and strategies are optimized on a surrogate model, enabling DrunkAgent transferable and stealthy. Extensive experiments on real-world datasets across diverse agentic RSs, including collaborative filtering, retrieval augmentation and sequential recommendations, demonstrate the generalizability, transferability and stealthiness of DrunkAgent.

摘要: 大型语言模型（LLM）驱动的代理越来越多地用于推荐系统（RS）中以实现个性化行为建模，其中记忆机制在使代理能够自主探索、学习和从现实世界的交互中自我进化方面发挥着关键作用。然而，作为上下文存储库的这种机制本质上暴露了潜在对抗操纵的攻击表面。尽管代理RS发挥着核心作用，但面对此类威胁时的稳健性在很大程度上仍然没有得到充分的探索。之前的作品存在语义不匹配或依赖于静态嵌入或预定义的提示，所有这些都不是为动态系统设计的，尤其是为LLM代理的动态内存状态。商业收件箱的黑匣子性质加剧了这一挑战。   为了解决上述问题，在本文中，我们对LLM支持的推荐代理中基于内存的漏洞进行了首次系统性调查，揭示了它们的安全局限性，并指导加强系统弹性和可信度的努力。具体来说，我们提出了一种新颖的黑匣子攻击框架DrunkAgent。DrunkAgent为目标物品促销精心设计了具有语义意义的对抗性文本触发器，并引入了一系列策略，通过破坏交互期间的记忆更新来最大化触发效应。触发器和策略在代理模型上进行了优化，使DrunkAgent具有可转移性和隐蔽性。跨不同代理RS对现实世界数据集进行的广泛实验，包括协作过滤、检索增强和顺序推荐，证明了DrunkAgent的可概括性、可移植性和隐蔽性。



## **46. Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth**

任意深度对齐：解锁LLM到任意深度的固有安全对齐 cs.LG

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.18081v1) [paper-pdf](http://arxiv.org/pdf/2510.18081v1)

**Authors**: Jiawei Zhang, Andrew Estornell, David D. Baek, Bo Li, Xiaojun Xu

**Abstract**: Large Language Models (LLMs) exhibit strong but shallow alignment: they directly refuse harmful queries when a refusal is expected at the very start of an assistant turn, yet this protection collapses once a harmful continuation is underway (either through the adversarial attacks or via harmful assistant-prefill attacks). This raises a fundamental question: Can the innate shallow alignment in LLMs be unlocked to ensure safety at arbitrary generation depths? To achieve this goal, we propose Any-Depth Alignment (ADA), an effective inference-time defense with negligible overhead. ADA is built based on our observation that alignment is concentrated in the assistant header tokens through repeated use in shallow-refusal training, and these tokens possess the model's strong alignment priors. By reintroducing these tokens mid-stream, ADA induces the model to reassess harmfulness and recover refusals at any point in generation. Across diverse open-source model families (Llama, Gemma, Mistral, Qwen, DeepSeek, and gpt-oss), ADA achieves robust safety performance without requiring any changes to the base model's parameters. It secures a near-100% refusal rate against challenging adversarial prefill attacks ranging from dozens to thousands of tokens. Furthermore, ADA reduces the average success rate of prominent adversarial prompt attacks (such as GCG, AutoDAN, PAIR, and TAP) to below 3%. This is all accomplished while preserving utility on benign tasks with minimal over-refusal. ADA maintains this resilience even after the base model undergoes subsequent instruction tuning (benign or adversarial).

摘要: 大型语言模型（LLM）表现出强而浅的一致性：当在助理转向开始时预计会拒绝时，它们会直接拒绝有害查询，但一旦有害的延续正在进行（无论是通过对抗性攻击还是通过有害的助理预填充攻击），这种保护就会崩溃。这提出了一个基本问题：LLM中固有的浅对齐能否被解锁，以确保任意世代深度的安全性？为了实现这一目标，我们提出了任意深度对齐（ADA），这是一种有效的推断时防御，且费用可以忽略不计。ADA是基于我们的观察而构建的，即通过在浅层拒绝训练中的重复使用，对齐集中在辅助头部代币中，并且这些代币拥有模型的强大对齐先验。通过在中途重新引入这些代币，ADA引导模型重新评估危害性并在生成过程中的任何时刻恢复拒绝。在各种开源模型系列（Llama、Gemma、Mistral、Qwen、DeepSeek和gtt-oss）中，ADA实现了强大的安全性能，无需对基本模型的参数进行任何更改。它确保了近100%的拒绝率，以应对数十到数千个代币的具有挑战性的对抗性预填充攻击。此外，ADA还将突出的对抗性提示攻击（例如GCG、AutoDAN、PAIR和RAP）的平均成功率降低至3%以下。这一切都是在保持良性任务的实用性的同时实现的，同时尽量减少过度拒绝。即使在基本模型经历后续指令调整（良性或对抗）之后，ADA也会保持这种弹性。



## **47. CourtGuard: A Local, Multiagent Prompt Injection Classifier**

CourtGuard：本地、多代理即时注入分类器 cs.CR

11 pages, 7 figures

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.19844v1) [paper-pdf](http://arxiv.org/pdf/2510.19844v1)

**Authors**: Isaac Wu, Michael Maslowski

**Abstract**: As large language models (LLMs) become integrated into various sensitive applications, prompt injection, the use of prompting to induce harmful behaviors from LLMs, poses an ever increasing risk. Prompt injection attacks can cause LLMs to leak sensitive data, spread misinformation, and exhibit harmful behaviors. To defend against these attacks, we propose CourtGuard, a locally-runnable, multiagent prompt injection classifier. In it, prompts are evaluated in a court-like multiagent LLM system, where a "defense attorney" model argues the prompt is benign, a "prosecution attorney" model argues the prompt is a prompt injection, and a "judge" model gives the final classification. CourtGuard has a lower false positive rate than the Direct Detector, an LLM as-a-judge. However, CourtGuard is generally a worse prompt injection detector. Nevertheless, this lower false positive rate highlights the importance of considering both adversarial and benign scenarios for the classification of a prompt. Additionally, the relative performance of CourtGuard in comparison to other prompt injection classifiers advances the use of multiagent systems as a defense against prompt injection attacks. The implementations of CourtGuard and the Direct Detector with full prompts for Gemma-3-12b-it, Llama-3.3-8B, and Phi-4-mini-instruct are available at https://github.com/isaacwu2000/CourtGuard.

摘要: 随着大型语言模型（LLM）集成到各种敏感应用程序中，提示注入（使用提示来诱导LLM的有害行为）构成了越来越大的风险。即时注入攻击可能导致LLM泄露敏感数据、传播错误信息并表现出有害行为。为了抵御这些攻击，我们提出了CourtGuard，这是一种可本地运行的多代理提示注入分类器。在其中，提示在类似法庭的多代理LLM系统中进行评估，其中“辩护律师”模型认为提示是良性的，“检察律师”模型认为提示是提示注射，“法官”模型给出最终分类。CourtGuard的假阳性率低于Direct Detector（LLM作为法官）。然而，CourtGuard通常是一个更糟糕的提示注射检测器。然而，这种较低的假阳性率凸显了在提示分类时考虑敌对和良性场景的重要性。此外，CourtGuard与其他即时注入分类器相比的相对性能促进了多智能体系统作为防御即时注入攻击的使用。CourtGuard和Direct Detector的实现（Gemma-3- 12 b-it、Llama-3.3-8B和Phi-4-mini-directory）可在https://github.com/isaacwu2000/CourtGuard上获取。



## **48. Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks**

人类恶意在特工中的回响：针对多轮在线骚扰攻击对标LLM cs.AI

13 pages, 4 figures

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.14207v2) [paper-pdf](http://arxiv.org/pdf/2510.14207v2)

**Authors**: Trilok Padhi, Pinxian Lu, Abdulkadir Erol, Tanmay Sutar, Gauri Sharma, Mina Sonmez, Munmun De Choudhury, Ugur Kursuncu

**Abstract**: Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible.

摘要: 大型语言模型（LLM）代理正在为越来越多的交互式Web应用程序提供支持，但仍然容易受到滥用和伤害。之前的越狱研究主要集中在单回合提示上，而真正的骚扰通常在多回合互动中展开。在这项工作中，我们提出了在线骚扰统计基准，包括：（i）合成的多回合骚扰对话数据集，（ii）多代理（例如，骚扰者、受害者）由重复博弈理论指导的模拟，（iii）跨越记忆、规划和微调攻击代理的三种越狱方法，以及（iv）混合方法评估框架。我们利用两个突出的LLM，LLaMA-3.1-8B-Instruct（开源）和Gemini-2.0-flash（闭源）。我们的研究结果表明，越狱调整使骚扰几乎可以保证，在Llama中，攻击成功率为95.78- 96.89% vs. 57.25- 64.19%，在Gemini中，攻击成功率为99.33% vs. 98.46%，同时在两种模型中，拒绝率都大幅降低到1-2%。最普遍的有毒行为是侮辱，84.9- 87.8%对44.2- 50.8%，没有调整，81.2- 85.1%对31.5- 38.8%，表明与敏感类别相比，如性或种族骚扰，护栏较弱。定性评估进一步表明，攻击代理再现人类一样的侵略配置文件，如马基雅维利/心理变态模式下的计划，和自恋倾向的记忆。与直觉相反，闭源模型和开源模型在不同时期表现出不同的升级轨迹，而闭源模型则表现出显着的脆弱性。总体而言，我们的研究结果表明，多回合和基于理论的攻击不仅能够高成功率，而且还模拟了类人的骚扰动态，推动了强大的安全护栏的开发，以最终确保在线平台的安全和负责任。



## **49. Is Multilingual LLM Watermarking Truly Multilingual? A Simple Back-Translation Solution**

多语言LLM水印真的是多语言的吗？简单的反翻译解决方案 cs.CL

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.18019v1) [paper-pdf](http://arxiv.org/pdf/2510.18019v1)

**Authors**: Asim Mohamed, Martin Gubri

**Abstract**: Multilingual watermarking aims to make large language model (LLM) outputs traceable across languages, yet current methods still fall short. Despite claims of cross-lingual robustness, they are evaluated only on high-resource languages. We show that existing multilingual watermarking methods are not truly multilingual: they fail to remain robust under translation attacks in medium- and low-resource languages. We trace this failure to semantic clustering, which fails when the tokenizer vocabulary contains too few full-word tokens for a given language. To address this, we introduce STEAM, a back-translation-based detection method that restores watermark strength lost through translation. STEAM is compatible with any watermarking method, robust across different tokenizers and languages, non-invasive, and easily extendable to new languages. With average gains of +0.19 AUC and +40%p TPR@1% on 17 languages, STEAM provides a simple and robust path toward fairer watermarking across diverse languages.

摘要: 多语言水印的目标是使大语言模型（LLM）输出跨语言可追踪，但目前的方法仍然存在不足。尽管声称跨语言的鲁棒性，但它们只在高资源语言上进行评估。我们发现，现有的多语言水印方法是不是真正的多语言：他们未能保持稳健的翻译攻击下，在中等和低资源的语言。我们将这种失败追溯到语义聚类，当标记器词汇表包含的给定语言的全词标记太少时，它会失败。为了解决这个问题，我们引入了STEAM，这是一种基于反向翻译的检测方法，可以恢复通过翻译丢失的水印强度。STEAM与任何水印方法兼容，在不同的标记器和语言中稳健，非侵入性，并且可以轻松扩展到新语言。STEAM在17种语言上的平均收益为+0.19 AUC和+40%p TPA @1%，提供了一种简单而稳健的方法，以实现跨不同语言的更公平的水印。



## **50. VERA-V: Variational Inference Framework for Jailbreaking Vision-Language Models**

VERA-V：越狱视觉语言模型的变分推理框架 cs.CR

18 pages, 7 Figures,

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.17759v1) [paper-pdf](http://arxiv.org/pdf/2510.17759v1)

**Authors**: Qilin Liao, Anamika Lochab, Ruqi Zhang

**Abstract**: Vision-Language Models (VLMs) extend large language models with visual reasoning, but their multimodal design also introduces new, underexplored vulnerabilities. Existing multimodal red-teaming methods largely rely on brittle templates, focus on single-attack settings, and expose only a narrow subset of vulnerabilities. To address these limitations, we introduce VERA-V, a variational inference framework that recasts multimodal jailbreak discovery as learning a joint posterior distribution over paired text-image prompts. This probabilistic view enables the generation of stealthy, coupled adversarial inputs that bypass model guardrails. We train a lightweight attacker to approximate the posterior, allowing efficient sampling of diverse jailbreaks and providing distributional insights into vulnerabilities. VERA-V further integrates three complementary strategies: (i) typography-based text prompts that embed harmful cues, (ii) diffusion-based image synthesis that introduces adversarial signals, and (iii) structured distractors to fragment VLM attention. Experiments on HarmBench and HADES benchmarks show that VERA-V consistently outperforms state-of-the-art baselines on both open-source and frontier VLMs, achieving up to 53.75% higher attack success rate (ASR) over the best baseline on GPT-4o.

摘要: 视觉语言模型（VLM）通过视觉推理扩展了大型语言模型，但它们的多模式设计也引入了新的、未充分探索的漏洞。现有的多模式红色团队方法很大程度上依赖于脆弱的模板，专注于单一攻击设置，并且仅暴露漏洞的一小部分。为了解决这些限制，我们引入了VERA-V，这是一个变分推理框架，它将多模式越狱发现重新构建为学习成对文本图像提示上的联合后验分布。这种概率观点使得能够生成绕过模型护栏的隐形、耦合的对抗输入。我们训练轻量级攻击者来逼近后验，从而能够对各种越狱进行有效抽样，并提供对漏洞的分布见解。VERA-V进一步集成了三种补充策略：（i）嵌入有害线索的基于印刷术的文本提示，（ii）引入对抗信号的基于扩散的图像合成，以及（iii）碎片VLM注意力的结构化干扰物。HarmBench和HADES基准测试的实验表明，VERA-V在开源和前沿VLM上的表现始终优于最先进的基线，比GPT-4 o上的最佳基线高出53.75%。



