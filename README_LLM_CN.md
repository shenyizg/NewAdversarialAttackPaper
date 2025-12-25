# Latest Large Language Model Attack Papers
**update at 2025-12-25 10:09:13**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Casting a SPELL: Sentence Pairing Exploration for LLM Limitation-breaking**

拼写：突破LLM限制的句子配对探索 cs.CR

Accepted to FSE 2026

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21236v1) [paper-pdf](https://arxiv.org/pdf/2512.21236v1)

**Authors**: Yifan Huang, Xiaojun Jia, Wenbo Guo, Yuqiang Sun, Yihao Huang, Chong Wang, Yang Liu

**Abstract**: Large language models (LLMs) have revolutionized software development through AI-assisted coding tools, enabling developers with limited programming expertise to create sophisticated applications. However, this accessibility extends to malicious actors who may exploit these powerful tools to generate harmful software. Existing jailbreaking research primarily focuses on general attack scenarios against LLMs, with limited exploration of malicious code generation as a jailbreak target. To address this gap, we propose SPELL, a comprehensive testing framework specifically designed to evaluate the weakness of security alignment in malicious code generation. Our framework employs a time-division selection strategy that systematically constructs jailbreaking prompts by intelligently combining sentences from a prior knowledge dataset, balancing exploration of novel attack patterns with exploitation of successful techniques. Extensive evaluation across three advanced code models (GPT-4.1, Claude-3.5, and Qwen2.5-Coder) demonstrates SPELL's effectiveness, achieving attack success rates of 83.75%, 19.38%, and 68.12% respectively across eight malicious code categories. The generated prompts successfully produce malicious code in real-world AI development tools such as Cursor, with outputs confirmed as malicious by state-of-the-art detection systems at rates exceeding 73%. These findings reveal significant security gaps in current LLM implementations and provide valuable insights for improving AI safety alignment in code generation applications.

摘要: 大型语言模型（LLM）通过人工智能辅助编码工具彻底改变了软件开发，使编程专业知识有限的开发人员能够创建复杂的应用程序。然而，这种可访问性扩展到恶意行为者，他们可能利用这些强大的工具来生成有害软件。现有的越狱研究主要关注针对LLM的一般攻击场景，对作为越狱目标的恶意代码生成的探索有限。为了解决这一差距，我们提出了STELL，这是一个全面的测试框架，专门用于评估恶意代码生成中安全一致的弱点。我们的框架采用时分选择策略，通过智能地组合来自先验知识数据集中的句子，平衡对新型攻击模式的探索与成功技术的利用，系统地构建越狱提示。对三种高级代码模型（GPT-4.1、Claude-3.5和Qwen 2.5-Coder）的广泛评估证明了SPELL的有效性，八种恶意代码类别的攻击成功率分别为83.75%、19.38%和68.12%。生成的提示在Cursor等现实世界的人工智能开发工具中成功生成恶意代码，最先进的检测系统确认输出为恶意代码，其比率超过73%。这些发现揭示了当前LLM实施中的重大安全漏洞，并为改进代码生成应用程序中的人工智能安全性提供了宝贵的见解。



## **2. GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs**

GateBreaker：对混合专家LLM的门引导攻击 cs.CR

Accepted by USENIX Security'26

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.21008v1) [paper-pdf](https://arxiv.org/pdf/2512.21008v1)

**Authors**: Lichao Wu, Sasha Behrouzi, Mohamadreza Rostami, Stjepan Picek, Ahmad-Reza Sadeghi

**Abstract**: Mixture-of-Experts (MoE) architectures have advanced the scaling of Large Language Models (LLMs) by activating only a sparse subset of parameters per input, enabling state-of-the-art performance with reduced computational cost. As these models are increasingly deployed in critical domains, understanding and strengthening their alignment mechanisms is essential to prevent harmful outputs. However, existing LLM safety research has focused almost exclusively on dense architectures, leaving the unique safety properties of MoEs largely unexamined. The modular, sparsely-activated design of MoEs suggests that safety mechanisms may operate differently than in dense models, raising questions about their robustness.   In this paper, we present GateBreaker, the first training-free, lightweight, and architecture-agnostic attack framework that compromises the safety alignment of modern MoE LLMs at inference time. GateBreaker operates in three stages: (i) gate-level profiling, which identifies safety experts disproportionately routed on harmful inputs, (ii) expert-level localization, which localizes the safety structure within safety experts, and (iii) targeted safety removal, which disables the identified safety structure to compromise the safety alignment. Our study shows that MoE safety concentrates within a small subset of neurons coordinated by sparse routing. Selective disabling of these neurons, approximately 3% of neurons in the targeted expert layers, significantly increases the averaged attack success rate (ASR) from 7.4% to 64.9% against the eight latest aligned MoE LLMs with limited utility degradation. These safety neurons transfer across models within the same family, raising ASR from 17.9% to 67.7% with one-shot transfer attack. Furthermore, GateBreaker generalizes to five MoE vision language models (VLMs) with 60.9% ASR on unsafe image inputs.

摘要: 专家混合（MoE）架构通过仅激活每个输入的稀疏参数子集来推进大型语言模型（LLM）的扩展，从而在降低计算成本的情况下实现最先进的性能。随着这些模型越来越多地部署在关键领域，了解和加强它们的协调机制对于防止有害输出至关重要。然而，现有的LLM安全研究几乎完全集中在密集架构上，而MoE的独特安全属性在很大程度上没有得到审查。教育部的模块化、稀疏激活设计表明，安全机制的运作方式可能与密集模型不同，从而引发了对其稳健性的质疑。   在本文中，我们介绍了GateBreaker，这是第一个无需训练，轻量级和架构不可知的攻击框架，它在推理时损害了现代MoE LLM的安全对齐。Gateway Breaker分三个阶段运作：（i）门级分析，识别不成比例地基于有害输入的安全专家，（ii）专家级本地化，将安全结构本地化在安全专家中，以及（iii）有针对性的安全删除，禁用识别的安全结构以损害安全对齐。我们的研究表明，MoE的安全性集中在由稀疏路由协调的一小部分神经元中。选择性禁用这些神经元（目标专家层中约3%的神经元），将平均攻击成功率（ASR）从7.4%显著提高到64.9%，而八个最新对齐的MoE LLM的效用下降有限。这些安全神经元在同一家族内的模型之间转移，通过一次转移攻击将ASB从17.9%提高到67.7%。此外，Gateway Breaker还将五个MoE视觉语言模型（VLM）推广到不安全图像输入的ASB为60.9%。



## **3. AegisAgent: An Autonomous Defense Agent Against Prompt Injection Attacks in LLM-HARs**

AegisAgent：一种针对LLM-HARs中即时注入攻击的自主防御代理 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2512.20986v1) [paper-pdf](https://arxiv.org/pdf/2512.20986v1)

**Authors**: Yihan Wang, Huanqi Yang, Shantanu Pal, Weitao Xu

**Abstract**: The integration of Large Language Models (LLMs) into wearable sensing is creating a new class of mobile applications capable of nuanced human activity understanding. However, the reliability of these systems is critically undermined by their vulnerability to prompt injection attacks, where attackers deliberately input deceptive instructions into LLMs. Traditional defenses, based on static filters and rigid rules, are insufficient to address the semantic complexity of these new attacks. We argue that a paradigm shift is needed -- from passive filtering to active protection and autonomous reasoning. We introduce AegisAgent, an autonomous agent system designed to ensure the security of LLM-driven HAR systems. Instead of merely blocking threats, AegisAgent functions as a cognitive guardian. It autonomously perceives potential semantic inconsistencies, reasons about the user's true intent by consulting a dynamic memory of past interactions, and acts by generating and executing a multi-step verification and repair plan. We implement AegisAgent as a lightweight, full-stack prototype and conduct a systematic evaluation on 15 common attacks with five state-of-the-art LLM-based HAR systems on three public datasets. Results show it reduces attack success rate by 30\% on average while incurring only 78.6 ms of latency overhead on a GPU workstation. Our work makes the first step towards building secure and trustworthy LLM-driven HAR systems.

摘要: 将大型语言模型（LLM）集成到可穿戴传感中正在创建一类能够细致入微地理解人类活动的新型移动应用程序。然而，这些系统的可靠性因其容易受到提示注入攻击（攻击者故意将欺骗性指令输入到LLM）而受到严重削弱。基于静态过滤器和严格规则的传统防御不足以解决这些新攻击的语义复杂性。我们认为需要进行范式转变--从被动过滤到主动保护和自主推理。我们引入了AegisAgent，这是一种自主代理系统，旨在确保LLM驱动的HAR系统的安全性。AegisAgent不仅仅是阻止威胁，而是充当认知守护者。它自主感知潜在的语义不一致，通过查阅过去交互的动态记忆来推理用户的真实意图，并通过生成和执行多步骤验证和修复计划来采取行动。我们将AegisAgent实施为轻量级全栈原型，并在三个公共数据集上使用五个最先进的基于LLM的HAR系统对15种常见攻击进行系统评估。结果显示，它平均将攻击成功率降低了30%，而在图形处理器上仅产生78.6 ms的延迟负担。我们的工作朝着构建安全且值得信赖的LLM驱动的HAR系统迈出了第一步。



## **4. ChatGPT: Excellent Paper! Accept It. Editor: Imposter Found! Review Rejected**

ChatGPT：优秀的纸张！接受吧。编辑：找到冒名顶替者！审查被拒绝 cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20405v1) [paper-pdf](https://arxiv.org/pdf/2512.20405v1)

**Authors**: Kanchon Gharami, Sanjiv Kumar Sarkar, Yongxin Liu, Shafika Showkat Moni

**Abstract**: Large Language Models (LLMs) like ChatGPT are now widely used in writing and reviewing scientific papers. While this trend accelerates publication growth and reduces human workload, it also introduces serious risks. Papers written or reviewed by LLMs may lack real novelty, contain fabricated or biased results, or mislead downstream research that others depend on. Such issues can damage reputations, waste resources, and even endanger lives when flawed studies influence medical or safety-critical systems. This research explores both the offensive and defensive sides of this growing threat. On the attack side, we demonstrate how an author can inject hidden prompts inside a PDF that secretly guide or "jailbreak" LLM reviewers into giving overly positive feedback and biased acceptance. On the defense side, we propose an "inject-and-detect" strategy for editors, where invisible trigger prompts are embedded into papers; if a review repeats or reacts to these triggers, it reveals that the review was generated by an LLM, not a human. This method turns prompt injections from vulnerability into a verification tool. We outline our design, expected model behaviors, and ethical safeguards for deployment. The goal is to expose how fragile today's peer-review process becomes under LLM influence and how editorial awareness can help restore trust in scientific evaluation.

摘要: ChatGPT等大型语言模型（LLM）现在广泛用于撰写和审查科学论文。虽然这一趋势加速了出版物的增长并减少了人力工作量，但也带来了严重的风险。LLM撰写或审查的论文可能缺乏真正的新颖性，包含捏造或有偏见的结果，或误导其他人所依赖的下游研究。当有缺陷的研究影响医疗或安全关键系统时，这些问题可能会损害声誉、浪费资源，甚至危及生命。这项研究探讨了这一日益增长的威胁的进攻和防守两方面。在攻击方面，我们演示了作者如何在PDF中注入隐藏提示，秘密引导或“越狱”LLM评审员提供过于积极的反馈和偏见的接受。在防御方面，我们为编辑提出了一种“注入并检测”策略，将隐形触发提示嵌入到论文中;如果评论重复或对这些触发做出反应，就表明该评论是由LLM而不是人类生成的。该方法将漏洞的提示注入到验证工具中。我们概述了我们的设计、预期的模型行为和部署的道德保障措施。目标是揭露当今的同行评审流程在法学硕士的影响下变得多么脆弱，以及编辑意识如何帮助恢复对科学评估的信任。



## **5. Optimistic TEE-Rollups: A Hybrid Architecture for Scalable and Verifiable Generative AI Inference on Blockchain**

乐观的TEE-Rollops：区块链上可扩展和可验证的生成式人工智能推理的混合架构 cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20176v1) [paper-pdf](https://arxiv.org/pdf/2512.20176v1)

**Authors**: Aaron Chan, Alex Ding, Frank Chen, Alan Wu, Bruce Zhang, Arther Tian

**Abstract**: The rapid integration of Large Language Models (LLMs) into decentralized physical infrastructure networks (DePIN) is currently bottlenecked by the Verifiability Trilemma, which posits that a decentralized inference system cannot simultaneously achieve high computational integrity, low latency, and low cost. Existing cryptographic solutions, such as Zero-Knowledge Machine Learning (ZKML), suffer from superlinear proving overheads (O(k NlogN)) that render them infeasible for billionparameter models. Conversely, optimistic approaches (opML) impose prohibitive dispute windows, preventing real-time interactivity, while recent "Proof of Quality" (PoQ) paradigms sacrifice cryptographic integrity for subjective semantic evaluation, leaving networks vulnerable to model downgrade attacks and reward hacking. In this paper, we introduce Optimistic TEE-Rollups (OTR), a hybrid verification protocol that harmonizes these constraints. OTR leverages NVIDIA H100 Confidential Computing Trusted Execution Environments (TEEs) to provide sub-second Provisional Finality, underpinned by an optimistic fraud-proof mechanism and stochastic Zero-Knowledge spot-checks to mitigate hardware side-channel risks. We formally define Proof of Efficient Attribution (PoEA), a consensus mechanism that cryptographically binds execution traces to hardware attestations, thereby guaranteeing model authenticity. Extensive simulations demonstrate that OTR achieves 99% of the throughput of centralized baselines with a marginal cost overhead of $0.07 per query, maintaining Byzantine fault tolerance against rational adversaries even in the presence of transient hardware vulnerabilities.

摘要: 大型语言模型（LLM）快速集成到去中心化物理基础设施网络（DePin）中目前受到可验证性三困境的限制，该困境认为去中心化推理系统无法同时实现高计算完整性、低延迟和低成本。现有的加密解决方案，例如零知识机器学习（ZKML），存在超线性证明费用（O（k NlogN））的问题，这使得它们对于十亿个参数模型来说不可行。相反，乐观方法（opML）施加了禁止性的争议窗口，阻止了实时交互，而最近的“质量证明”（PoQ）范式则牺牲了加密完整性来进行主观语义评估，使网络容易受到模型降级攻击和奖励黑客攻击。在本文中，我们介绍了乐观TEE-Rollup（OTR），这是一种协调这些约束的混合验证协议。OTR利用NVIDIA H100机密计算可信执行环境（TEE）提供亚秒级临时最终结果，并以乐观的防欺诈机制和随机零知识抽查为基础，以减轻硬件侧通道风险。我们正式定义了有效归因证明（PoEA），这是一种共识机制，通过加密方式将执行跟踪与硬件证明绑定，从而保证模型的真实性。广泛的模拟表明，OTR实现了集中式基线99%的吞吐量，每次查询的边际成本费用为0.07美元，即使存在暂时性硬件漏洞，也能对理性对手保持拜占庭式的耐药性。



## **6. Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography**

Odysseus：通过双重隐写术破解商业多模式法学硕士集成系统 cs.CR

This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20168v1) [paper-pdf](https://arxiv.org/pdf/2512.20168v1)

**Authors**: Songze Li, Jiameng Cheng, Yiming Li, Xiaojun Jia, Dacheng Tao

**Abstract**: By integrating language understanding with perceptual modalities such as images, multimodal large language models (MLLMs) constitute a critical substrate for modern AI systems, particularly intelligent agents operating in open and interactive environments. However, their increasing accessibility also raises heightened risks of misuse, such as generating harmful or unsafe content. To mitigate these risks, alignment techniques are commonly applied to align model behavior with human values. Despite these efforts, recent studies have shown that jailbreak attacks can circumvent alignment and elicit unsafe outputs. Currently, most existing jailbreak methods are tailored for open-source models and exhibit limited effectiveness against commercial MLLM-integrated systems, which often employ additional filters. These filters can detect and prevent malicious input and output content, significantly reducing jailbreak threats. In this paper, we reveal that the success of these safety filters heavily relies on a critical assumption that malicious content must be explicitly visible in either the input or the output. This assumption, while often valid for traditional LLM-integrated systems, breaks down in MLLM-integrated systems, where attackers can leverage multiple modalities to conceal adversarial intent, leading to a false sense of security in existing MLLM-integrated systems. To challenge this assumption, we propose Odysseus, a novel jailbreak paradigm that introduces dual steganography to covertly embed malicious queries and responses into benign-looking images. Extensive experiments on benchmark datasets demonstrate that our Odysseus successfully jailbreaks several pioneering and realistic MLLM-integrated systems, achieving up to 99% attack success rate. It exposes a fundamental blind spot in existing defenses, and calls for rethinking cross-modal security in MLLM-integrated systems.

摘要: 通过将语言理解与图像等感知模式集成起来，多模式大型语言模型（MLLM）构成了现代人工智能系统的重要基础，特别是在开放和交互环境中运行的智能代理。然而，它们的可访问性不断增加也增加了滥用风险，例如产生有害或不安全内容。为了减轻这些风险，通常应用对齐技术来将模型行为与人类价值观对齐。尽管做出了这些努力，最近的研究表明，越狱攻击可能会绕过对齐并引发不安全的输出。目前，大多数现有的越狱方法都是针对开源模型量身定制的，并且对于商业MLLM集成系统（通常使用额外的过滤器）的有效性有限。这些过滤器可以检测和防止恶意输入和输出内容，从而显着减少越狱威胁。在本文中，我们揭示了这些安全过滤器的成功在很大程度上依赖于一个关键假设，即恶意内容必须在输入或输出中显式可见。这种假设虽然通常适用于传统的LLM集成系统，但在MLLM集成系统中却出现了问题，攻击者可以利用多种模式来隐藏对抗意图，从而导致现有的MLLM集成系统中出现错误的安全感。为了挑战这一假设，我们提出了Odysseus，一种新的越狱范例，它引入了双重隐写术来秘密地将恶意查询和响应嵌入到看起来很好的图像中。在基准数据集上进行的大量实验表明，我们的Odysseus成功地越狱了几个开创性和现实的MLLM集成系统，攻击成功率高达99%。它暴露了现有防御中的一个基本盲点，并呼吁重新思考MLLM集成系统中的跨模式安全性。



## **7. AI Security Beyond Core Domains: Resume Screening as a Case Study of Adversarial Vulnerabilities in Specialized LLM Applications**

超越核心领域的人工智能安全：简历筛选作为专业LLM应用中对抗漏洞的案例研究 cs.CL

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2512.20164v1) [paper-pdf](https://arxiv.org/pdf/2512.20164v1)

**Authors**: Honglin Mu, Jinghao Liu, Kaiyang Wan, Rui Xing, Xiuying Chen, Timothy Baldwin, Wanxiang Che

**Abstract**: Large Language Models (LLMs) excel at text comprehension and generation, making them ideal for automated tasks like code review and content moderation. However, our research identifies a vulnerability: LLMs can be manipulated by "adversarial instructions" hidden in input data, such as resumes or code, causing them to deviate from their intended task. Notably, while defenses may exist for mature domains such as code review, they are often absent in other common applications such as resume screening and peer review. This paper introduces a benchmark to assess this vulnerability in resume screening, revealing attack success rates exceeding 80% for certain attack types. We evaluate two defense mechanisms: prompt-based defenses achieve 10.1% attack reduction with 12.5% false rejection increase, while our proposed FIDS (Foreign Instruction Detection through Separation) using LoRA adaptation achieves 15.4% attack reduction with 10.4% false rejection increase. The combined approach provides 26.3% attack reduction, demonstrating that training-time defenses outperform inference-time mitigations in both security and utility preservation.

摘要: 大型语言模型（LLM）擅长文本理解和生成，非常适合代码审查和内容审核等自动化任务。然而，我们的研究发现了一个漏洞：LLM可能会被隐藏在输入数据（例如简历或代码）中的“对抗指令”操纵，导致它们偏离预期任务。值得注意的是，虽然代码审查等成熟领域可能存在防御措施，但在简历筛选和同行审查等其他常见应用中通常不存在防御措施。本文引入了一个基准来评估简历筛选中的此漏洞，揭示了某些攻击类型的攻击成功率超过80%。我们评估了两种防御机制：基于预算的防御可以减少10.1%的攻击，错误拒绝增加12.5%，而我们提出的使用LoRA适应的FIDS（通过分离的外部指令检测）可以减少15.4%的攻击，错误拒绝增加10.4%。组合方法可减少26.3%的攻击，证明训练时防御在安全性和实用程序保存方面优于推理时缓解措施。



## **8. Conditional Adversarial Fragility in Financial Machine Learning under Macroeconomic Stress**

宏观经济压力下金融机器学习中的条件对抗脆弱性 cs.LG

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19935v1) [paper-pdf](https://arxiv.org/pdf/2512.19935v1)

**Authors**: Samruddhi Baviskar

**Abstract**: Machine learning models used in financial decision systems operate in nonstationary economic environments, yet adversarial robustness is typically evaluated under static assumptions. This work introduces Conditional Adversarial Fragility, a regime dependent phenomenon in which adversarial vulnerability is systematically amplified during periods of macroeconomic stress. We propose a regime aware evaluation framework for time indexed tabular financial classification tasks that conditions robustness assessment on external indicators of economic stress. Using volatility based regime segmentation as a proxy for macroeconomic conditions, we evaluate model behavior across calm and stress periods while holding model architecture, attack methodology, and evaluation protocols constant. Baseline predictive performance remains comparable across regimes, indicating that economic stress alone does not induce inherent performance degradation. Under adversarial perturbations, however, models operating during stress regimes exhibit substantially greater degradation across predictive accuracy, operational decision thresholds, and risk sensitive outcomes. We further demonstrate that this amplification propagates to increased false negative rates, elevating the risk of missed high risk cases during adverse conditions. To complement numerical robustness metrics, we introduce an interpretive governance layer based on semantic auditing of model explanations using large language models. Together, these results demonstrate that adversarial robustness in financial machine learning is a regime dependent property and motivate stress aware approaches to model risk assessment in high stakes financial deployments.

摘要: 金融决策系统中使用的机器学习模型在非平稳经济环境中运行，但对抗稳健性通常是在静态假设下评估的。这项工作引入了条件对抗脆弱性，这是一种依赖政权的现象，其中对抗脆弱性在宏观经济压力时期被系统性放大。我们提出了一个用于时间索引表格财务分类任务的制度意识评估框架，该框架以经济压力的外部指标为条件进行稳健性评估。使用基于波动性的制度分割作为宏观经济状况的代理，我们评估平静和压力时期的模型行为，同时保持模型架构、攻击方法和评估协议不变。不同制度之间的基线预测性能保持可比性，这表明经济压力本身不会导致固有的性能下降。然而，在对抗性扰动下，在压力制度下运行的模型在预测准确性、操作决策阈值和风险敏感结果方面表现出明显更大的退化。我们进一步证明，这种放大会传播到假阴性率增加，从而增加了在不利条件下错过高风险病例的风险。为了补充数字稳健性指标，我们引入了一个基于使用大型语言模型对模型解释进行语义审计的解释治理层。总而言之，这些结果表明，金融机器学习中的对抗稳健性是一种依赖于制度的属性，并激励压力感知方法对高风险金融部署中的风险评估进行建模。



## **9. Causal-Guided Detoxify Backdoor Attack of Open-Weight LoRA Models**

开放重量LoRA模型的Causes引导的去神经后门攻击 cs.CR

NDSS 2026

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19297v1) [paper-pdf](https://arxiv.org/pdf/2512.19297v1)

**Authors**: Linzhi Chen, Yang Sun, Hongru Wei, Yuqi Chen

**Abstract**: Low-Rank Adaptation (LoRA) has emerged as an efficient method for fine-tuning large language models (LLMs) and is widely adopted within the open-source community. However, the decentralized dissemination of LoRA adapters through platforms such as Hugging Face introduces novel security vulnerabilities: malicious adapters can be easily distributed and evade conventional oversight mechanisms. Despite these risks, backdoor attacks targeting LoRA-based fine-tuning remain relatively underexplored. Existing backdoor attack strategies are ill-suited to this setting, as they often rely on inaccessible training data, fail to account for the structural properties unique to LoRA, or suffer from high false trigger rates (FTR), thereby compromising their stealth. To address these challenges, we propose Causal-Guided Detoxify Backdoor Attack (CBA), a novel backdoor attack framework specifically designed for open-weight LoRA models. CBA operates without access to original training data and achieves high stealth through two key innovations: (1) a coverage-guided data generation pipeline that synthesizes task-aligned inputs via behavioral exploration, and (2) a causal-guided detoxification strategy that merges poisoned and clean adapters by preserving task-critical neurons. Unlike prior approaches, CBA enables post-training control over attack intensity through causal influence-based weight allocation, eliminating the need for repeated retraining. Evaluated across six LoRA models, CBA achieves high attack success rates while reducing FTR by 50-70\% compared to baseline methods. Furthermore, it demonstrates enhanced resistance to state-of-the-art backdoor defenses, highlighting its stealth and robustness.

摘要: 低等级适应（LoRA）已成为微调大型语言模型（LLM）的一种有效方法，并在开源社区中广泛采用。然而，通过Hugging Face等平台分散传播LoRA适配器引入了新型安全漏洞：恶意适配器可以轻松分发并逃避传统的监督机制。尽管存在这些风险，但针对基于LoRA的微调的后门攻击仍然相对未充分研究。现有的后门攻击策略不适合这种设置，因为它们通常依赖于不可访问的训练数据，无法考虑LoRA特有的结构属性，或者遭受高错误触发率（FTR）的影响，从而损害了它们的隐形性。为了解决这些挑战，我们提出了一种专门为开放权重LoRA模型设计的新型后门攻击框架- CBA在不访问原始训练数据的情况下运行，并通过两项关键创新实现了高度隐身：（1）覆盖引导的数据生成管道，通过行为探索合成任务对齐的输入，以及（2）通过保留任务关键神经元合并中毒和清洁适配器的cavern-guided解毒策略。与以前的方法不同，CBA通过基于因果影响的权重分配实现了对攻击强度的训练后控制，从而消除了重复再训练的需要。在六个LoRA模型中进行评估，CBA实现了高攻击成功率，同时与基线方法相比将FTR降低了50- 70%。此外，它还表现出对最先进后门防御的增强抵抗力，凸显了其隐形性和稳健性。



## **10. DREAM: Dynamic Red-teaming across Environments for AI Models**

DREAM：人工智能模型跨环境的动态红色团队 cs.CR

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19016v1) [paper-pdf](https://arxiv.org/pdf/2512.19016v1)

**Authors**: Liming Lu, Xiang Gu, Junyu Huang, Jiawei Du, Yunhuai Liu, Yongbin Zhou, Shuchao Pang

**Abstract**: Large Language Models (LLMs) are increasingly used in agentic systems, where their interactions with diverse tools and environments create complex, multi-stage safety challenges. However, existing benchmarks mostly rely on static, single-turn assessments that miss vulnerabilities from adaptive, long-chain attacks. To fill this gap, we introduce DREAM, a framework for systematic evaluation of LLM agents against dynamic, multi-stage attacks. At its core, DREAM uses a Cross-Environment Adversarial Knowledge Graph (CE-AKG) to maintain stateful, cross-domain understanding of vulnerabilities. This graph guides a Contextualized Guided Policy Search (C-GPS) algorithm that dynamically constructs attack chains from a knowledge base of 1,986 atomic actions across 349 distinct digital environments. Our evaluation of 12 leading LLM agents reveals a critical vulnerability: these attack chains succeed in over 70% of cases for most models, showing the power of stateful, cross-environment exploits. Through analysis of these failures, we identify two key weaknesses in current agents: contextual fragility, where safety behaviors fail to transfer across environments, and an inability to track long-term malicious intent. Our findings also show that traditional safety measures, such as initial defense prompts, are largely ineffective against attacks that build context over multiple interactions. To advance agent safety research, we release DREAM as a tool for evaluating vulnerabilities and developing more robust defenses.

摘要: 大型语言模型（LLM）越来越多地用于代理系统，它们与不同工具和环境的交互会带来复杂、多阶段的安全挑战。然而，现有的基准大多依赖于静态、单轮评估，这些评估会错过自适应性、长链攻击的漏洞。为了填补这一空白，我们引入了DREAM，这是一个针对动态、多阶段攻击系统评估LLM代理的框架。DREAM的核心是使用跨环境对抗知识图（CE-AKG）来维护对漏洞的有状态、跨领域理解。该图指导上下文引导政策搜索（C-GPS）算法，该算法根据349个不同数字环境中1，986个原子动作的知识库动态构建攻击链。我们对12个领先的LLM代理的评估揭示了一个关键漏洞：对于大多数模型来说，这些攻击链在超过70%的情况下都取得了成功，展示了有状态、跨环境漏洞利用的力量。通过对这些失败的分析，我们发现了当前代理的两个关键弱点：上下文脆弱性，即安全行为无法跨环境转移，以及无法跟踪长期恶意意图。我们的研究结果还表明，传统的安全措施（例如初始防御提示）对于在多重交互中建立上下文的攻击在很大程度上无效。为了推进代理安全研究，我们发布DREAM作为评估漏洞和开发更强大防御的工具。



## **11. Efficient Jailbreak Mitigation Using Semantic Linear Classification in a Multi-Staged Pipeline**

在多阶段管道中使用语义线性分类的高效越狱缓解 cs.CR

Under Review

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2512.19011v1) [paper-pdf](https://arxiv.org/pdf/2512.19011v1)

**Authors**: Akshaj Prashanth Rao, Advait Singh, Saumya Kumaar Saksena, Dhruv Kumar

**Abstract**: Prompt injection and jailbreaking attacks pose persistent security challenges to large language model (LLM)-based systems. We present an efficient and systematically evaluated defense architecture that mitigates these threats through a lightweight, multi-stage pipeline. Its core component is a semantic filter based on text normalization, TF-IDF representations, and a Linear SVM classifier. Despite its simplicity, this module achieves 93.4% accuracy and 96.5% specificity on held-out data, substantially reducing attack throughput while incurring negligible computational overhead.   Building on this efficient foundation, the full pipeline integrates complementary detection and mitigation mechanisms that operate at successive stages, providing strong robustness with minimal latency. In comparative experiments, our SVM-based configuration improves overall accuracy from 35.1% to 93.4% while reducing average time to completion from approximately 450s to 47s, yielding over 10 times lower latency than ShieldGemma. These results demonstrate that the proposed design simultaneously advances defensive precision and efficiency, addressing a core limitation of current model-based moderators.   Evaluation across a curated corpus of over 30,000 labeled prompts, including benign, jailbreak, and application-layer injections, confirms that staged, resource-efficient defenses can robustly secure modern LLM-driven applications.

摘要: 提示注入和越狱攻击对基于大型语言模型（LLM）的系统构成了持续的安全挑战。我们提出了一种高效且系统化评估的防御架构，可以通过轻量级的多阶段管道减轻这些威胁。其核心组件是基于文本规范化、TF-IDF表示和线性支持量分类器的语义过滤器。尽管它很简单，但该模块在保存的数据上实现了93.4%的准确性和96.5%的特异性，大大降低了攻击吞吐量，同时产生的计算负担可以忽略不计。   在这个高效的基础上，完整的管道集成了在连续阶段运行的补充检测和缓解机制，以最小的延迟提供强大的鲁棒性。在比较实验中，我们基于支持机的配置将总体准确性从35.1%提高到93.4%，同时将平均完成时间从约450秒减少到47秒，延迟时间比ShieldGemma低10倍以上。这些结果表明，拟议的设计同时提高了防御精度和效率，解决了当前基于模型的版主的核心限制。   对30，000多个带标签的提示（包括良性、越狱和应用层注入）的精心策划的数据库进行评估，证实了分阶段的资源高效防御可以强大地保护现代LLM驱动的应用程序。



## **12. Automated Red-Teaming Framework for Large Language Model Security Assessment: A Comprehensive Attack Generation and Detection System**

用于大型语言模型安全评估的自动化Red-Teaming框架：全面的攻击生成和检测系统 cs.CR

18 pages

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.20677v1) [paper-pdf](https://arxiv.org/pdf/2512.20677v1)

**Authors**: Zhang Wei, Peilu Hu, Shengning Lang, Hao Yan, Li Mei, Yichao Zhang, Chen Yang, Junfeng Hao, Zhimo Han

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes domains, ensuring their security and alignment has become a critical challenge. Existing red-teaming practices depend heavily on manual testing, which limits scalability and fails to comprehensively cover the vast space of potential adversarial behaviors. This paper introduces an automated red-teaming framework that systematically generates, executes, and evaluates adversarial prompts to uncover security vulnerabilities in LLMs. Our framework integrates meta-prompting-based attack synthesis, multi-modal vulnerability detection, and standardized evaluation protocols spanning six major threat categories -- reward hacking, deceptive alignment, data exfiltration, sandbagging, inappropriate tool use, and chain-of-thought manipulation. Experiments on the GPT-OSS-20B model reveal 47 distinct vulnerabilities, including 21 high-severity and 12 novel attack patterns, achieving a $3.9\times$ improvement in vulnerability discovery rate over manual expert testing while maintaining 89\% detection accuracy. These results demonstrate the framework's effectiveness in enabling scalable, systematic, and reproducible AI safety evaluations. By providing actionable insights for improving alignment robustness, this work advances the state of automated LLM red-teaming and contributes to the broader goal of building secure and trustworthy AI systems.

摘要: 随着大型语言模型（LLM）越来越多地部署在高风险领域，确保它们的安全性和一致性已成为一项关键挑战。现有的红色团队实践严重依赖手动测试，这限制了可扩展性，并且无法全面覆盖潜在对抗行为的巨大空间。本文介绍了一个自动化红色团队框架，该框架系统地生成、执行和评估对抗提示，以发现LLC中的安全漏洞。我们的框架集成了基于元提示的攻击合成、多模式漏洞检测和跨越六个主要威胁类别的标准化评估协议-奖励黑客攻击、欺骗性对齐、数据泄露、沙袋、不当工具使用和思想链操纵。GPT-OSS-20 B模型上的实验揭示了47个不同的漏洞，包括21个高严重性和12个新型攻击模式，与手动专家测试相比，漏洞发现率提高了3.9倍，同时保持89%的检测准确率。这些结果证明了该框架在实现可扩展、系统和可重复的人工智能安全评估方面的有效性。通过为提高对齐稳健性提供可操作的见解，这项工作推进了LLM自动化红色团队的状态，并有助于构建安全且值得信赖的人工智能系统的更广泛目标。



## **13. VizDefender: Unmasking Visualization Tampering through Proactive Localization and Intent Inference**

VizDefender：通过主动本地化和意图推理揭开可视化篡改的面纱 cs.CV

IEEE Transactions on Visualization and Computer Graphics (IEEE PacificVis'26 TVCG Track)

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18853v1) [paper-pdf](https://arxiv.org/pdf/2512.18853v1)

**Authors**: Sicheng Song, Yanjie Zhang, Zixin Chen, Huamin Qu, Changbo Wang, Chenhui Li

**Abstract**: The integrity of data visualizations is increasingly threatened by image editing techniques that enable subtle yet deceptive tampering. Through a formative study, we define this challenge and categorize tampering techniques into two primary types: data manipulation and visual encoding manipulation. To address this, we present VizDefender, a framework for tampering detection and analysis. The framework integrates two core components: 1) a semi-fragile watermark module that protects the visualization by embedding a location map to images, which allows for the precise localization of tampered regions while preserving visual quality, and 2) an intent analysis module that leverages Multimodal Large Language Models (MLLMs) to interpret manipulation, inferring the attacker's intent and misleading effects. Extensive evaluations and user studies demonstrate the effectiveness of our methods.

摘要: 数据可视化的完整性越来越受到图像编辑技术的威胁，这些技术可以进行微妙但具有欺骗性的篡改。通过形成性研究，我们定义了这一挑战，并将篡改技术分为两种主要类型：数据操纵和视觉编码操纵。为了解决这个问题，我们提出了VizDefender，这是一个用于篡改检测和分析的框架。该框架集成了两个核心组件：1）半脆弱水印模块，通过将位置地图嵌入到图像中来保护可视化，这允许在保留视觉质量的同时精确定位被篡改的区域，2）意图分析模块，利用多模式大型语言模型（MLLM）来解释操纵，推断攻击者的意图和误导性效果。广泛的评估和用户研究证明了我们方法的有效性。



## **14. MEEA: Mere Exposure Effect-Driven Confrontational Optimization for LLM Jailbreaking**

MEEA：LLM越狱的纯粹曝光驱动的对抗优化 cs.AI

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18755v1) [paper-pdf](https://arxiv.org/pdf/2512.18755v1)

**Authors**: Jianyi Zhang, Shizhao Liu, Ziyin Zhou, Zhen Li

**Abstract**: The rapid advancement of large language models (LLMs) has intensified concerns about the robustness of their safety alignment. While existing jailbreak studies explore both single-turn and multi-turn strategies, most implicitly assume a static safety boundary and fail to account for how contextual interactions dynamically influence model behavior, leading to limited stability and generalization. Motivated by this gap, we propose MEEA (Mere Exposure Effect Attack), a psychology-inspired, fully automated black-box framework for evaluating multi-turn safety robustness, grounded in the mere exposure effect. MEEA leverages repeated low-toxicity semantic exposure to induce a gradual shift in a model's effective safety threshold, enabling progressive erosion of alignment constraints over sustained interactions. Concretely, MEEA constructs semantically progressive prompt chains and optimizes them using a simulated annealing strategy guided by semantic similarity, toxicity, and jailbreak effectiveness. Extensive experiments on both closed-source and open-source models, including GPT-4, Claude-3.5, and DeepSeek-R1, demonstrate that MEEA consistently achieves higher attack success rates than seven representative baselines, with an average Attack Success Rate (ASR) improvement exceeding 20%. Ablation studies further validate the necessity of both annealing-based optimization and contextual exposure mechanisms. Beyond improved attack effectiveness, our findings indicate that LLM safety behavior is inherently dynamic and history-dependent, challenging the common assumption of static alignment boundaries and highlighting the need for interaction-aware safety evaluation and defense mechanisms. Our code is available at: https://github.com/Carney-lsz/MEEA

摘要: 大型语言模型（LLM）的快速发展加剧了人们对其安全对齐稳健性的担忧。虽然现有的越狱研究探索了单转向和多转向策略，但大多数都隐含地假设静态安全边界，并且未能考虑上下文相互作用如何动态影响模型行为，从而导致稳定性和概括性有限。出于这一差距的动机，我们提出了MEEA（纯粹暴露效应攻击），这是一个受心理学启发的全自动黑匣子框架，用于评估多回合安全稳健性，基于纯粹的暴露效应。MEEA利用重复的低毒性语义暴露来诱导模型的有效安全阈值的逐渐转变，从而使对齐约束在持续相互作用中逐渐受到侵蚀。具体地说，MEEA构造语义渐进提示链，并使用模拟退火策略进行优化，指导语义相似性，毒性和越狱有效性。在包括GPT-4、Claude-3.5和DeepSeek-R1在内的闭源和开源模型上进行的广泛实验表明，MEEA始终比七个代表性基线实现更高的攻击成功率，平均攻击成功率（ASR）提高超过20%。消融研究进一步验证了基于退火的优化和环境暴露机制的必要性。除了提高攻击效率，我们的研究结果表明，LLM安全行为本质上是动态的和历史依赖的，挑战了静态对齐边界的常见假设，并强调了交互感知安全评估和防御机制的必要性。我们的代码可访问：https://github.com/Carney-lsz/MEEA



## **15. Explainable and Fine-Grained Safeguarding of LLM Multi-Agent Systems via Bi-Level Graph Anomaly Detection**

通过双层图异常检测对LLM多智能体系统进行可解释和细粒度保护 cs.CR

14 pages, 3 tables, 5 figures

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.18733v1) [paper-pdf](https://arxiv.org/pdf/2512.18733v1)

**Authors**: Junjun Pan, Yixin Liu, Rui Miao, Kaize Ding, Yu Zheng, Quoc Viet Hung Nguyen, Alan Wee-Chung Liew, Shirui Pan

**Abstract**: Large language model (LLM)-based multi-agent systems (MAS) have shown strong capabilities in solving complex tasks. As MAS become increasingly autonomous in various safety-critical tasks, detecting malicious agents has become a critical security concern. Although existing graph anomaly detection (GAD)-based defenses can identify anomalous agents, they mainly rely on coarse sentence-level information and overlook fine-grained lexical cues, leading to suboptimal performance. Moreover, the lack of interpretability in these methods limits their reliability and real-world applicability. To address these limitations, we propose XG-Guard, an explainable and fine-grained safeguarding framework for detecting malicious agents in MAS. To incorporate both coarse and fine-grained textual information for anomalous agent identification, we utilize a bi-level agent encoder to jointly model the sentence- and token-level representations of each agent. A theme-based anomaly detector further captures the evolving discussion focus in MAS dialogues, while a bi-level score fusion mechanism quantifies token-level contributions for explanation. Extensive experiments across diverse MAS topologies and attack scenarios demonstrate robust detection performance and strong interpretability of XG-Guard.

摘要: 基于大型语言模型（LLM）的多智能体系统（MAS）在解决复杂任务方面表现出了强大的能力。随着MAS在各种安全关键任务中变得越来越自治，检测恶意代理已成为一个关键的安全问题。尽管现有的基于图形异常检测（GAD）的防御可以识别异常代理，但它们主要依赖于粗略的业务级别信息并忽视细粒度的词汇线索，导致性能次优。此外，这些方法缺乏可解释性限制了它们的可靠性和现实世界的适用性。为了解决这些限制，我们提出了XG-Guard，这是一个可解释且细粒度的保护框架，用于检测MAS中的恶意代理。为了结合粗粒度和细粒度的文本信息来识别异常代理，我们利用两级代理编码器来联合建模每个代理的句子级和符号级表示。基于主题的异常检测器进一步捕捉MAS对话中不断变化的讨论焦点，而两级分数融合机制量化代币级贡献以进行解释。跨各种MAS布局和攻击场景的广泛实验证明了XG-Guard的稳健检测性能和强大的可解释性。



## **16. Breaking Minds, Breaking Systems: Jailbreaking Large Language Models via Human-like Psychological Manipulation**

打破思维，打破系统：通过类人心理操纵越狱大型语言模型 cs.CR

**SubmitDate**: 2025-12-20    [abs](http://arxiv.org/abs/2512.18244v1) [paper-pdf](https://arxiv.org/pdf/2512.18244v1)

**Authors**: Zehao Liu, Xi Lin

**Abstract**: Large Language Models (LLMs) have gained considerable popularity and protected by increasingly sophisticated safety mechanisms. However, jailbreak attacks continue to pose a critical security threat by inducing models to generate policy-violating behaviors. Current paradigms focus on input-level anomalies, overlooking that the model's internal psychometric state can be systematically manipulated. To address this, we introduce Psychological Jailbreak, a new jailbreak attack paradigm that exposes a stateful psychological attack surface in LLMs, where attackers exploit the manipulation of a model's psychological state across interactions. Building on this insight, we propose Human-like Psychological Manipulation (HPM), a black-box jailbreak method that dynamically profiles a target model's latent psychological vulnerabilities and synthesizes tailored multi-turn attack strategies. By leveraging the model's optimization for anthropomorphic consistency, HPM creates a psychological pressure where social compliance overrides safety constraints. To systematically measure psychological safety, we construct an evaluation framework incorporating psychometric datasets and the Policy Corruption Score (PCS). Benchmarking against various models (e.g., GPT-4o, DeepSeek-V3, Gemini-2-Flash), HPM achieves a mean Attack Success Rate (ASR) of 88.1%, outperforming state-of-the-art attack baselines. Our experiments demonstrate robust penetration against advanced defenses, including adversarial prompt optimization (e.g., RPO) and cognitive interventions (e.g., Self-Reminder). Ultimately, PCS analysis confirms HPM induces safety breakdown to satisfy manipulated contexts. Our work advocates for a fundamental paradigm shift from static content filtering to psychological safety, prioritizing the development of psychological defense mechanisms against deep cognitive manipulation.

摘要: 大型语言模型（LLM）已经相当受欢迎，并受到日益复杂的安全机制的保护。然而，越狱攻击通过诱导模型产生违反政策的行为，继续构成严重的安全威胁。当前的范式专注于输入级异常，忽视了模型的内部心理测量状态可以被系统性操纵。为了解决这个问题，我们引入了心理越狱，这是一种新的越狱攻击范式，它暴露了LLM中的状态心理攻击表面，攻击者利用交互中对模型心理状态的操纵。基于这一见解，我们提出了类人心理操纵（HPM），这是一种黑匣子越狱方法，可以动态地描述目标模型的潜在心理脆弱性并综合量身定制的多回合攻击策略。通过利用模型对拟人化一致性的优化，HPM创造了一种心理压力，社会合规性凌驾于安全约束之上。为了系统性地衡量心理安全，我们构建了一个纳入心理测量数据集和政策腐败评分（PCS）的评估框架。针对各种模型（例如，GPT-4 o、DeepSeek-V3、Gemini-2-Flash），HPM的平均攻击成功率（ASB）为88.1%，优于最先进的攻击基线。我们的实验证明了对高级防御的强大渗透，包括对抗性即时优化（例如，LPO）和认知干预（例如，自我提醒）。最终，PCS分析证实HPM会引发安全崩溃以满足操纵环境。我们的工作倡导从静态内容过滤到心理安全的根本范式转变，优先考虑开发针对深度认知操纵的心理防御机制。



## **17. Towards Benchmarking Privacy Vulnerabilities in Selective Forgetting with Large Language Models**

使用大型语言模型对选择性遗忘中的隐私漏洞进行基准测试 cs.LG

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.18035v1) [paper-pdf](https://arxiv.org/pdf/2512.18035v1)

**Authors**: Wei Qian, Chenxu Zhao, Yangyi Li, Mengdi Huai

**Abstract**: The rapid advancements in artificial intelligence (AI) have primarily focused on the process of learning from data to acquire knowledgeable learning systems. As these systems are increasingly deployed in critical areas, ensuring their privacy and alignment with human values is paramount. Recently, selective forgetting (also known as machine unlearning) has shown promise for privacy and data removal tasks, and has emerged as a transformative paradigm shift in the field of AI. It refers to the ability of a model to selectively erase the influence of previously seen data, which is especially important for compliance with modern data protection regulations and for aligning models with human values. Despite its promise, selective forgetting raises significant privacy concerns, especially when the data involved come from sensitive domains. While new unlearning-induced privacy attacks are continuously proposed, each is shown to outperform its predecessors using different experimental settings, which can lead to overly optimistic and potentially unfair assessments that may disproportionately favor one particular attack over the others. In this work, we present the first comprehensive benchmark for evaluating privacy vulnerabilities in selective forgetting. We extensively investigate privacy vulnerabilities of machine unlearning techniques and benchmark privacy leakage across a wide range of victim data, state-of-the-art unlearning privacy attacks, unlearning methods, and model architectures. We systematically evaluate and identify critical factors related to unlearning-induced privacy leakage. With our novel insights, we aim to provide a standardized tool for practitioners seeking to deploy customized unlearning applications with faithful privacy assessments.

摘要: 人工智能（AI）的快速发展主要集中在从数据中学习以获取知识渊博的学习系统的过程。随着这些系统越来越多地部署在关键领域，确保其隐私并与人类价值观保持一致至关重要。最近，选择性遗忘（也称为机器取消学习）在隐私和数据删除任务方面表现出了希望，并已成为人工智能领域的变革性范式转变。它指的是模型有选择地消除之前看到的数据影响的能力，这对于遵守现代数据保护法规以及使模型与人类价值观保持一致尤其重要。尽管有希望，但选择性遗忘引发了严重的隐私问题，尤其是当所涉及的数据来自敏感域时。虽然新的不学习引发的隐私攻击不断被提出，但使用不同的实验设置，每种攻击的表现都优于其前辈，这可能会导致过于乐观且可能不公平的评估，从而可能不成比例地支持一种特定的攻击而不是其他攻击。在这项工作中，我们提出了第一个用于评估选择性遗忘中的隐私漏洞的全面基准。我们广泛调查机器忘记技术的隐私漏洞，并对广泛的受害者数据、最先进的忘记隐私攻击、忘记方法和模型架构的隐私泄露进行基准测试。我们系统性地评估和识别与无知导致的隐私泄露相关的关键因素。凭借我们新颖的见解，我们的目标是为寻求部署具有忠实隐私评估的定制取消学习应用程序的从业者提供一种标准化工具。



## **18. Adversarially Robust Detection of Harmful Online Content: A Computational Design Science Approach**

有害在线内容的对抗稳健检测：计算设计科学方法 cs.LG

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17367v1) [paper-pdf](https://arxiv.org/pdf/2512.17367v1)

**Authors**: Yidong Chai, Yi Liu, Mohammadreza Ebrahimi, Weifeng Li, Balaji Padmanabhan

**Abstract**: Social media platforms are plagued by harmful content such as hate speech, misinformation, and extremist rhetoric. Machine learning (ML) models are widely adopted to detect such content; however, they remain highly vulnerable to adversarial attacks, wherein malicious users subtly modify text to evade detection. Enhancing adversarial robustness is therefore essential, requiring detectors that can defend against diverse attacks (generalizability) while maintaining high overall accuracy. However, simultaneously achieving both optimal generalizability and accuracy is challenging. Following the computational design science paradigm, this study takes a sequential approach that first proposes a novel framework (Large Language Model-based Sample Generation and Aggregation, LLM-SGA) by identifying the key invariances of textual adversarial attacks and leveraging them to ensure that a detector instantiated within the framework has strong generalizability. Second, we instantiate our detector (Adversarially Robust Harmful Online Content Detector, ARHOCD) with three novel design components to improve detection accuracy: (1) an ensemble of multiple base detectors that exploits their complementary strengths; (2) a novel weight assignment method that dynamically adjusts weights based on each sample's predictability and each base detector's capability, with weights initialized using domain knowledge and updated via Bayesian inference; and (3) a novel adversarial training strategy that iteratively optimizes both the base detectors and the weight assignor. We addressed several limitations of existing adversarial robustness enhancement research and empirically evaluated ARHOCD across three datasets spanning hate speech, rumor, and extremist content. Results show that ARHOCD offers strong generalizability and improves detection accuracy under adversarial conditions.

摘要: 社交媒体平台受到仇恨言论、错误信息和极端主义言论等有害内容的困扰。机器学习（ML）模型被广泛采用来检测此类内容;然而，它们仍然极易受到对抗攻击，其中恶意用户会巧妙地修改文本以逃避检测。因此，增强对抗鲁棒性至关重要，需要检测器能够抵御各种攻击（可概括性），同时保持高的总体准确性。然而，同时实现最佳概括性和准确性是一项挑战。遵循计算设计科学范式，本研究采用顺序方法，首先提出了一种新颖的框架（基于大语言模型的样本生成和聚合，LLM-LGA），通过识别文本对抗攻击的关键不变性并利用它们来确保框架内实例化的检测器具有很强的概括性。其次，我们实例化我们的检测器（对抗鲁棒有害在线内容检测器，ARHOCD）具有三个新颖的设计组件来提高检测准确性：（1）利用其互补优势的多个基本检测器的集成;（2）一种新颖的权重分配方法，其基于每个样本的可预测性和每个碱基检测器的能力动态调整权重，权重使用领域知识初始化并通过Bayesian推理更新;以及（3）一种新颖的对抗训练策略，迭代优化基本检测器和权重分配器。我们解决了现有对抗鲁棒性增强研究的几个局限性，并在跨越仇恨言论、谣言和极端主义内容的三个数据集中对ARHOCD进行了实证评估。结果表明，ARHOCD具有很强的概括性，并提高了对抗条件下的检测准确性。



## **19. Cryptanalysis of Pseudorandom Error-Correcting Codes**

伪随机错误纠正码的密码分析 cs.CR

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17310v1) [paper-pdf](https://arxiv.org/pdf/2512.17310v1)

**Authors**: Tianrui Wang, Anyu Wang, Tianshuo Cong, Delong Ran, Jinyuan Liu, Xiaoyun Wang

**Abstract**: Pseudorandom error-correcting codes (PRC) is a novel cryptographic primitive proposed at CRYPTO 2024. Due to the dual capability of pseudorandomness and error correction, PRC has been recognized as a promising foundational component for watermarking AI-generated content. However, the security of PRC has not been thoroughly analyzed, especially with concrete parameters or even in the face of cryptographic attacks. To fill this gap, we present the first cryptanalysis of PRC. We first propose three attacks to challenge the undetectability and robustness assumptions of PRC. Among them, two attacks aim to distinguish PRC-based codewords from plain vectors, and one attack aims to compromise the decoding process of PRC. Our attacks successfully undermine the claimed security guarantees across all parameter configurations. Notably, our attack can detect the presence of a watermark with overwhelming probability at a cost of $2^{22}$ operations. We also validate our approach by attacking real-world large generative models such as DeepSeek and Stable Diffusion. To mitigate our attacks, we further propose three defenses to enhance the security of PRC, including parameter suggestions, implementation suggestions, and constructing a revised key generation algorithm. Our proposed revised key generation function effectively prevents the occurrence of weak keys. However, we highlight that the current PRC-based watermarking scheme still cannot achieve a 128-bit security under our parameter suggestions due to the inherent configurations of large generative models, such as the maximum output length of large language models.

摘要: 伪随机纠错码（PRC）是在2024年国际密码学大会上提出的一种新的密码学原语。由于伪随机性和纠错的双重能力，PRC已被认为是对AI生成的内容进行水印的有前途的基础组件。然而，PRC的安全性还没有得到彻底的分析，特别是在具体的参数，甚至在面对密码攻击。为了填补这一空白，我们提出了PRC的第一个密码分析。我们首先提出三种攻击来挑战PRC的不可检测性和稳健性假设。其中，两次攻击旨在将基于PRC的代码字与纯载体区分开来，一次攻击旨在损害PRC的解码过程。我们的攻击成功地破坏了所有参数配置中声称的安全保证。值得注意的是，我们的攻击可以以压倒性的可能性检测到水印的存在，但操作成本为2美元的^{22}$。我们还通过攻击DeepSeek和Stable Distance等现实世界的大型生成模型来验证我们的方法。为了减轻我们的攻击，我们进一步提出了三种防御措施来增强PRC的安全性，包括参数建议、实施建议和构建修改后的密钥生成算法。我们提出的修改后的密钥生成功能有效地防止了弱密钥的发生。然而，我们强调，由于大型生成模型的固有配置，例如大型语言模型的最大输出长度，当前基于PRC的水印方案在我们的参数建议下仍然无法实现128位安全性。



## **20. Biosecurity-Aware AI: Agentic Risk Auditing of Soft Prompt Attacks on ESM-Based Variant Predictors**

具有生物安全意识的人工智能：对基于ESM的变体预测器的软提示攻击的量化风险审计 cs.CR

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17146v1) [paper-pdf](https://arxiv.org/pdf/2512.17146v1)

**Authors**: Huixin Zhan

**Abstract**: Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated remarkable success in variant effect prediction. However, their security and robustness under adversarial manipulation remain largely unexplored. To address this gap, we introduce the Secure Agentic Genomic Evaluator (SAGE), an agentic framework for auditing the adversarial vulnerabilities of GFMs. SAGE functions through an interpretable and automated risk auditing loop. It injects soft prompt perturbations, monitors model behavior across training checkpoints, computes risk metrics such as AUROC and AUPR, and generates structured reports with large language model-based narrative explanations. This agentic process enables continuous evaluation of embedding-space robustness without modifying the underlying model. Using SAGE, we find that even state-of-the-art GFMs like ESM2 are sensitive to targeted soft prompt attacks, resulting in measurable performance degradation. These findings reveal critical and previously hidden vulnerabilities in genomic foundation models, showing the importance of agentic risk auditing in securing biomedical applications such as clinical variant interpretation.

摘要: 基因组基础模型（GFM），例如进化规模建模（ESM），在变异效应预测方面取得了显着的成功。然而，它们在对抗操纵下的安全性和稳健性在很大程度上仍未得到探索。为了解决这一差距，我们引入了安全统计基因组评估器（SAGE），这是一个用于审计GFM对抗性漏洞的代理框架。SAGE通过可解释和自动化的风险审计循环发挥作用。它注入软提示扰动，监控训练检查点的模型行为，计算AUROC和AUPR等风险指标，并生成具有基于大型语言模型的叙述性解释的结构化报告。这个代理过程能够连续评估嵌入空间稳健性，而无需修改基础模型。使用SAGE，我们发现即使是ESM 2等最先进的GFM也对有针对性的软提示攻击敏感，从而导致可衡量的性能下降。这些发现揭示了基因组基础模型中关键且先前隐藏的漏洞，表明代理风险审计在确保临床变体解释等生物医学应用方面的重要性。



## **21. From Essence to Defense: Adaptive Semantic-aware Watermarking for Embedding-as-a-Service Copyright Protection**

从本质到防御：用于嵌入即服务版权保护的自适应语义感知水印 cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16439v1) [paper-pdf](https://arxiv.org/pdf/2512.16439v1)

**Authors**: Hao Li, Yubing Ren, Yanan Cao, Yingjie Li, Fang Fang, Xuebin Wang

**Abstract**: Benefiting from the superior capabilities of large language models in natural language understanding and generation, Embeddings-as-a-Service (EaaS) has emerged as a successful commercial paradigm on the web platform. However, prior studies have revealed that EaaS is vulnerable to imitation attacks. Existing methods protect the intellectual property of EaaS through watermarking techniques, but they all ignore the most important properties of embedding: semantics, resulting in limited harmlessness and stealthiness. To this end, we propose SemMark, a novel semantic-based watermarking paradigm for EaaS copyright protection. SemMark employs locality-sensitive hashing to partition the semantic space and inject semantic-aware watermarks into specific regions, ensuring that the watermark signals remain imperceptible and diverse. In addition, we introduce the adaptive watermark weight mechanism based on the local outlier factor to preserve the original embedding distribution. Furthermore, we propose Detect-Sampling and Dimensionality-Reduction attacks and construct four scenarios to evaluate the watermarking method. Extensive experiments are conducted on four popular NLP datasets, and SemMark achieves superior verifiability, diversity, stealthiness, and harmlessness.

摘要: 得益于大型语言模型在自然语言理解和生成方面的卓越能力，嵌入式即服务（EaaS）已经成为Web平台上成功的商业模式。然而，之前的研究表明，EaaS容易受到模仿攻击。现有的方法通过水印技术保护EaaS的知识产权，但它们都忽略了嵌入最重要的属性：语义，导致有限的无害性和隐蔽性。为此，我们提出了SemMark，一种新的基于语义的水印范例EaaS版权保护。SemMark采用局部敏感哈希来划分语义空间，并将语义感知水印注入特定区域，确保水印信号保持不可感知和多样性。此外，我们引入了基于局部离群因子的自适应水印权重机制，以保持原始的嵌入分布。此外，我们提出了检测采样和降低分辨率攻击，并构造了四个场景来评估水印方法。在四个流行的NLP数据集上进行了广泛的实验，SemMark实现了卓越的可验证性，多样性，隐蔽性和无害性。



## **22. MemoryGraft: Persistent Compromise of LLM Agents via Poisoned Experience Retrieval**

MemoryGraft：通过中毒经验检索持续损害LLM药物 cs.CR

14 pages, 1 figure, includes appendix

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16962v1) [paper-pdf](https://arxiv.org/pdf/2512.16962v1)

**Authors**: Saksham Sahai Srivastava, Haoyu He

**Abstract**: Large Language Model (LLM) agents increasingly rely on long-term memory and Retrieval-Augmented Generation (RAG) to persist experiences and refine future performance. While this experience learning capability enhances agentic autonomy, it introduces a critical, unexplored attack surface, i.e., the trust boundary between an agent's reasoning core and its own past. In this paper, we introduce MemoryGraft. It is a novel indirect injection attack that compromises agent behavior not through immediate jailbreaks, but by implanting malicious successful experiences into the agent's long-term memory. Unlike traditional prompt injections that are transient, or standard RAG poisoning that targets factual knowledge, MemoryGraft exploits the agent's semantic imitation heuristic which is the tendency to replicate patterns from retrieved successful tasks. We demonstrate that an attacker who can supply benign ingestion-level artifacts that the agent reads during execution can induce it to construct a poisoned RAG store where a small set of malicious procedure templates is persisted alongside benign experiences. When the agent later encounters semantically similar tasks, union retrieval over lexical and embedding similarity reliably surfaces these grafted memories, and the agent adopts the embedded unsafe patterns, leading to persistent behavioral drift across sessions. We validate MemoryGraft on MetaGPT's DataInterpreter agent with GPT-4o and find that a small number of poisoned records can account for a large fraction of retrieved experiences on benign workloads, turning experience-based self-improvement into a vector for stealthy and durable compromise. To facilitate reproducibility and future research, our code and evaluation data are available at https://github.com/Jacobhhy/Agent-Memory-Poisoning.

摘要: 大型语言模型（LLM）代理越来越依赖长期记忆和检索增强生成（RAG）来持久体验并改进未来性能。虽然这种经验学习能力增强了代理人的自主性，但它引入了一个关键的、未探索的攻击表面，即代理人的推理核心与其自己的过去之间的信任边界。在本文中，我们介绍了MemoryGraft。这是一种新颖的间接注入攻击，它不是通过立即越狱，而是通过将恶意的成功体验植入到代理的长期记忆中来损害代理的行为。与传统的短暂提示注入或针对事实知识的标准RAG中毒不同，MemoryGraft利用了代理的语义模仿启发式，即从检索到的成功任务中复制模式的倾向。我们证明，能够提供代理在执行期间读取的良性摄入级工件的攻击者可以诱导其构建一个有毒的RAG存储，其中一小组恶意过程模板与良性体验一起持久存在。当代理后来遇到语义相似的任务时，基于词汇和嵌入相似性的联合检索会可靠地暴露这些嫁接的记忆，并且代理采用嵌入的不安全模式，从而导致跨会话持续的行为漂移。我们在MetaGPT的DataInterpreter代理上使用GPT-4 o验证了SecureGraft，发现少量有毒记录可以解释良性工作负载上检索到的体验的大部分，从而将基于经验的自我改进转变为隐形和持久妥协的载体。为了促进重现性和未来的研究，我们的代码和评估数据可在https://github.com/Jacobhhy/Agent-Memory-Poisoning上获取。



## **23. In-Context Probing for Membership Inference in Fine-Tuned Language Models**

精调语言模型中成员推理的上下文探索 cs.CR

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2512.16292v2) [paper-pdf](https://arxiv.org/pdf/2512.16292v2)

**Authors**: Zhexi Lu, Hongliang Chi, Nathalie Baracaldo, Swanand Ravindra Kadhe, Yuseok Jeon, Lei Yu

**Abstract**: Membership inference attacks (MIAs) pose a critical privacy threat to fine-tuned large language models (LLMs), especially when models are adapted to domain-specific tasks using sensitive data. While prior black-box MIA techniques rely on confidence scores or token likelihoods, these signals are often entangled with a sample's intrinsic properties - such as content difficulty or rarity - leading to poor generalization and low signal-to-noise ratios. In this paper, we propose ICP-MIA, a novel MIA framework grounded in the theory of training dynamics, particularly the phenomenon of diminishing returns during optimization. We introduce the Optimization Gap as a fundamental signal of membership: at convergence, member samples exhibit minimal remaining loss-reduction potential, while non-members retain significant potential for further optimization. To estimate this gap in a black-box setting, we propose In-Context Probing (ICP), a training-free method that simulates fine-tuning-like behavior via strategically constructed input contexts. We propose two probing strategies: reference-data-based (using semantically similar public samples) and self-perturbation (via masking or generation). Experiments on three tasks and multiple LLMs show that ICP-MIA significantly outperforms prior black-box MIAs, particularly at low false positive rates. We further analyze how reference data alignment, model type, PEFT configurations, and training schedules affect attack effectiveness. Our findings establish ICP-MIA as a practical and theoretically grounded framework for auditing privacy risks in deployed LLMs.

摘要: 成员资格推理攻击（MIA）对微调的大型语言模型（LLM）构成严重的隐私威胁，尤其是当模型使用敏感数据适应特定领域任务时。虽然先前的黑匣子MIA技术依赖于置信度分数或代币可能性，但这些信号通常与样本的内在属性（例如内容难度或稀有性）纠缠在一起，导致概括性较差和低信噪比。在本文中，我们提出了ICP-MIA，这是一种基于训练动力学理论的新型MIA框架，特别是优化过程中的回报递减现象。我们引入优化差距作为成员资格的基本信号：在收敛时，成员样本表现出最小的剩余损失减少潜力，而非成员保留了进一步优化的显着潜力。为了估计黑匣子环境中的这一差距，我们提出了In-Context Probing（ICP），这是一种免训练的方法，通过策略性构建的输入上下文模拟类似微调的行为。我们提出了两种探测策略：基于参考数据（使用语义相似的公共样本）和自我扰动（通过掩蔽或生成）。三项任务和多个LLM的实验表明，ICP-MIA显着优于之前的黑匣子MIA，尤其是在低假阳性率下。我们进一步分析参考数据对齐、模型类型、PEFT配置和训练计划如何影响攻击有效性。我们的研究结果将ICP-MIA确立为审计已部署的LLM隐私风险的实用且理论基础的框架。



## **24. DualGuard: Dual-stream Large Language Model Watermarking Defense against Paraphrase and Spoofing Attack**

DualGuard：双数据流大型语言模型水印防御，防止重述和欺骗攻击 cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2512.16182v1) [paper-pdf](https://arxiv.org/pdf/2512.16182v1)

**Authors**: Hao Li, Yubing Ren, Yanan Cao, Yingjie Li, Fang Fang, Shi Wang, Li Guo

**Abstract**: With the rapid development of cloud-based services, large language models (LLMs) have become increasingly accessible through various web platforms. However, this accessibility has also led to growing risks of model abuse. LLM watermarking has emerged as an effective approach to mitigate such misuse and protect intellectual property. Existing watermarking algorithms, however, primarily focus on defending against paraphrase attacks while overlooking piggyback spoofing attacks, which can inject harmful content, compromise watermark reliability, and undermine trust in attribution. To address this limitation, we propose DualGuard, the first watermarking algorithm capable of defending against both paraphrase and spoofing attacks. DualGuard employs the adaptive dual-stream watermarking mechanism, in which two complementary watermark signals are dynamically injected based on the semantic content. This design enables DualGuard not only to detect but also to trace spoofing attacks, thereby ensuring reliable and trustworthy watermark detection. Extensive experiments conducted across multiple datasets and language models demonstrate that DualGuard achieves excellent detectability, robustness, traceability, and text quality, effectively advancing the state of LLM watermarking for real-world applications.

摘要: 随着基于云的服务的快速发展，大型语言模型（LLM）越来越多地可以通过各种网络平台访问。然而，这种可及性也导致了模型滥用的风险越来越大。LLM水印已成为减少此类滥用和保护知识产权的有效方法。然而，现有的水印算法主要专注于防御重述攻击，同时忽略了背负欺骗攻击，这种攻击可能会注入有害内容、损害水印可靠性并破坏对归因的信任。为了解决这一局限性，我们提出了DualGuard，这是第一个能够防御重述和欺骗攻击的水印算法。DualGuard采用自适应双流水印机制，根据语义内容动态注入两个互补的水印信号。该设计使DualGuard不仅能够检测而且能够跟踪欺骗攻击，从而确保可靠且值得信赖的水印检测。在多个数据集和语言模型上进行的广泛实验表明，DualGuard实现了出色的检测性、稳健性、可追溯性和文本质量，有效地提高了现实世界应用程序的LLM水印状态。



## **25. Bounty Hunter: Autonomous, Comprehensive Emulation of Multi-Faceted Adversaries**

赏金猎人：多面对手的自主、全面模拟 cs.CR

15 pages, 9 figures

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15275v1) [paper-pdf](https://arxiv.org/pdf/2512.15275v1)

**Authors**: Louis Hackländer-Jansen, Rafael Uetz, Martin Henze

**Abstract**: Adversary emulation is an essential procedure for cybersecurity assessments such as evaluating an organization's security posture or facilitating structured training and research in dedicated environments. To allow for systematic and time-efficient assessments, several approaches from academia and industry have worked towards the automation of adversarial actions. However, they exhibit significant limitations regarding autonomy, tactics coverage, and real-world applicability. Consequently, adversary emulation remains a predominantly manual task requiring substantial human effort and security expertise - even amidst the rise of Large Language Models. In this paper, we present Bounty Hunter, an automated adversary emulation method, designed and implemented as an open-source plugin for the popular adversary emulation platform Caldera, that enables autonomous emulation of adversaries with multi-faceted behavior while providing a wide coverage of tactics. To this end, it realizes diverse adversarial behavior, such as different levels of detectability and varying attack paths across repeated emulations. By autonomously compromising a simulated enterprise network, Bounty Hunter showcases its ability to achieve given objectives without prior knowledge of its target, including pre-compromise, initial compromise, and post-compromise attack tactics. Overall, Bounty Hunter facilitates autonomous, comprehensive, and multi-faceted adversary emulation to help researchers and practitioners in performing realistic and time-efficient security assessments, training exercises, and intrusion detection research.

摘要: Adobile仿真是网络安全评估的重要程序，例如评估组织的安全态势或促进专用环境中的结构化培训和研究。为了进行系统性且高效的评估，学术界和工业界的多种方法致力于对抗行动的自动化。然而，它们在自主性、战术覆盖范围和现实世界的适用性方面表现出显着的局限性。因此，对手模拟仍然是一项主要的手动任务，需要大量的人力和安全专业知识--即使在大型语言模型的兴起下也是如此。在本文中，我们介绍了Bounty Hunter，这是一种自动化的对手模拟方法，作为流行的对手模拟平台Caldera的开源插件设计和实现，它能够自主模拟具有多方面行为的对手，同时提供广泛的战术覆盖范围。为此，它实现了多样化的对抗行为，例如不同级别的可检测性和重复模拟中的不同攻击路径。通过自主破坏模拟企业网络，Bounty Hunter展示了其在不了解其目标的情况下实现给定目标的能力，包括破坏前、初始破坏和破坏后攻击策略。总体而言，Bounty Hunter促进了自主、全面和多方面的对手模拟，以帮助研究人员和从业者执行现实且省时的安全评估、培训练习和入侵检测研究。



## **26. MCP-SafetyBench: A Benchmark for Safety Evaluation of Large Language Models with Real-World MCP Servers**

MCP-SafetyBench：使用现实世界的LCP服务器对大型语言模型进行安全评估的基准 cs.CL

Our benchmark is available at https://github.com/xjzzzzzzzz/MCPSafety

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15163v1) [paper-pdf](https://arxiv.org/pdf/2512.15163v1)

**Authors**: Xuanjun Zong, Zhiqi Shen, Lei Wang, Yunshi Lan, Chao Yang

**Abstract**: Large language models (LLMs) are evolving into agentic systems that reason, plan, and operate external tools. The Model Context Protocol (MCP) is a key enabler of this transition, offering a standardized interface for connecting LLMs with heterogeneous tools and services. Yet MCP's openness and multi-server workflows introduce new safety risks that existing benchmarks fail to capture, as they focus on isolated attacks or lack real-world coverage. We present MCP-SafetyBench, a comprehensive benchmark built on real MCP servers that supports realistic multi-turn evaluation across five domains: browser automation, financial analysis, location navigation, repository management, and web search. It incorporates a unified taxonomy of 20 MCP attack types spanning server, host, and user sides, and includes tasks requiring multi-step reasoning and cross-server coordination under uncertainty. Using MCP-SafetyBench, we systematically evaluate leading open- and closed-source LLMs, revealing large disparities in safety performance and escalating vulnerabilities as task horizons and server interactions grow. Our results highlight the urgent need for stronger defenses and establish MCP-SafetyBench as a foundation for diagnosing and mitigating safety risks in real-world MCP deployments.

摘要: 大型语言模型（LLM）正在演变为推理、计划和操作外部工具的代理系统。模型上下文协议（MCP）是这种转变的关键推动者，它提供了一个标准化的接口，用于连接LLM与异构工具和服务。然而，MCP的开放性和多服务器工作流程引入了现有基准无法捕获的新安全风险，因为它们专注于孤立的攻击或缺乏真实世界的覆盖。我们提出了MCP-SafetyBench，这是一个建立在真实MCP服务器上的综合基准测试，支持跨五个领域的真实多轮评估：浏览器自动化，财务分析，位置导航，存储库管理和Web搜索。它融合了跨越服务器、主机和用户端的20种HCP攻击类型的统一分类，并包括需要多步推理和不确定性下跨服务器协调的任务。使用MCP-SafetyBench，我们系统地评估领先的开源和开源LLM，揭示了随着任务视野和服务器交互的增长，安全性能的巨大差异和不断升级的漏洞。我们的结果凸显了迫切需要更强大的防御，并将MCP-SafetyBench建立为诊断和缓解现实世界中的LCP部署中安全风险的基础。



## **27. Quantifying Return on Security Controls in LLM Systems**

量化LLM系统中安全控制的回报 cs.CR

13 pages, 9 figures, 3 tables

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.15081v1) [paper-pdf](https://arxiv.org/pdf/2512.15081v1)

**Authors**: Richard Helder Moulton, Austin O'Brien, John D. Hastings

**Abstract**: Although large language models (LLMs) are increasingly used in security-critical workflows, practitioners lack quantitative guidance on which safeguards are worth deploying. This paper introduces a decision-oriented framework and reproducible methodology that together quantify residual risk, convert adversarial probe outcomes into financial risk estimates and return-on-control (RoC) metrics, and enable monetary comparison of layered defenses for LLM-based systems. A retrieval-augmented generation (RAG) service is instantiated using the DeepSeek-R1 model over a corpus containing synthetic personally identifiable information (PII), and subjected to automated attacks with Garak across five vulnerability classes: PII leakage, latent context injection, prompt injection, adversarial attack generation, and divergence. For each (vulnerability, control) pair, attack success probabilities are estimated via Laplace's Rule of Succession and combined with loss triangle distributions, calibrated from public breach-cost data, in 10,000-run Monte Carlo simulations to produce loss exceedance curves and expected losses. Three widely used mitigations, attribute-based access control (ABAC); named entity recognition (NER) redaction using Microsoft Presidio; and NeMo Guardrails, are then compared to a baseline RAG configuration. The baseline system exhibits very high attack success rates (>= 0.98 for PII, latent injection, and prompt injection), yielding a total simulated expected loss of $313k per attack scenario. ABAC collapses success probabilities for PII and prompt-related attacks to near zero and reduces the total expected loss by ~94%, achieving an RoC of 9.83. NER redaction likewise eliminates PII leakage and attains an RoC of 5.97, while NeMo Guardrails provides only marginal benefit (RoC of 0.05).

摘要: 尽管大型语言模型（LLM）越来越多地用于安全关键工作流程，但从业者缺乏关于哪些保障措施值得部署的量化指导。本文介绍了一个面向决策的框架和可重复的方法论，它们共同量化剩余风险，将对抗性调查结果转化为财务风险估计和控制回报（RoC）指标，并实现基于LLM的系统的分层防御的货币比较。检索增强生成（RAG）服务使用DeepSeek-R1模型在包含合成个人可识别信息（PRI）的数据库上实例化，并在五个漏洞类别上受到Garak的自动攻击：PIP泄露、潜在上下文注入、提示注入、对抗攻击生成和分歧。对于每个（漏洞、控制）对，攻击成功概率是通过拉普拉斯继承规则估计的，并结合根据公共违规成本数据校准的损失三角分布，在10，000次运行的蒙特卡洛模拟中生成损失延迟曲线和预期损失。然后将三种广泛使用的缓解措施：基于属性的访问控制（ABAC）;使用Microsoft Presidio的命名实体识别（NER）编辑;和NeMo Guardrails与基线RAG配置进行比较。基线系统表现出非常高的攻击成功率（PRI、潜伏注射和即时注射>= 0.98），每个攻击场景的模拟预期损失总额为31.3万美元。ABAC将PRI和预算相关攻击的成功概率降至接近零，并将总预期损失降低约94%，实现了9.83的RoC。NER编辑同样消除了PRI泄漏，并获得了5.97的RoC，而NeMo Guardrails仅提供了边际效益（RoC为0.05）。



## **28. MALCDF: A Distributed Multi-Agent LLM Framework for Real-Time Cyber**

MALEDF：用于实时网络的分布式多代理LLM框架 cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14846v1) [paper-pdf](https://arxiv.org/pdf/2512.14846v1)

**Authors**: Arth Bhardwaj, Sia Godika, Yuvam Loonker

**Abstract**: Traditional, centralized security tools often miss adaptive, multi-vector attacks. We present the Multi-Agent LLM Cyber Defense Framework (MALCDF), a practical setup where four large language model (LLM) agents-Detection, Intelligence, Response, and Analysis-work together in real time. Agents communicate over a Secure Communication Layer (SCL) with encrypted, ontology-aligned messages, and produce audit-friendly outputs (e.g., MITRE ATT&CK mappings).   For evaluation, we keep the test simple and consistent: all reported metrics come from the same 50-record live stream derived from the CICIDS2017 feature schema. CICIDS2017 is used for configuration (fields/schema) and to train a practical ML baseline. The ML-IDS baseline is a Lightweight Random Forest IDS (LRF-IDS) trained on a subset of CICIDS2017 and tested on the 50-record stream, with no overlap between training and test records.   In experiments, MALCDF reaches 90.0% detection accuracy, 85.7% F1-score, and 9.1% false-positive rate, with 6.8s average per-event latency. It outperforms the lightweight ML-IDS baseline and a single-LLM setup on accuracy while keeping end-to-end outputs consistent. Overall, this hands-on build suggests that coordinating simple LLM agents with secure, ontology-aligned messaging can improve practical, real-time cyber defense.

摘要: 传统的集中式安全工具经常错过自适应的多载体攻击。我们介绍了多代理LLM网络防御框架（MALEDF），这是一种实用的设置，四个大型语言模型（LLM）代理--检测、情报、响应和分析--实时协同工作。代理通过安全通信层（SCL）与加密的、与实体对齐的消息进行通信，并产生对用户友好的输出（例如，MITRE ATT & CK映射）。   对于评估，我们保持测试简单且一致：所有报告的指标都来自源自CICIDS 2017功能模式的相同50条记录直播流。CICIDS 2017用于配置（字段/模式）和训练实用的ML基线。ML-IDS基线是轻量级随机森林IDS（LRF-IDS），在CICIDS 2017的子集上训练，并在50条记录流上进行测试，训练和测试记录之间没有重叠。   实验中，MALEDF的检测准确率达到90.0%，F1评分达到85.7%，假阳性率达到9.1%，平均每事件潜伏期为6.8s。它在准确性方面优于轻量级ML-IDS基线和单LLM设置，同时保持端到端输出一致。总体而言，这种动手构建表明，将简单的LLM代理与安全的、与实体一致的消息传递协调起来可以改善实用的实时网络防御。



## **29. PerProb: Indirectly Evaluating Memorization in Large Language Models**

PerProb：间接评估大型语言模型中的精简化 cs.CR

Accepted at APSEC 2025

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14600v1) [paper-pdf](https://arxiv.org/pdf/2512.14600v1)

**Authors**: Yihan Liao, Jacky Keung, Xiaoxue Ma, Jingyu Zhang, Yicheng Sun

**Abstract**: The rapid advancement of Large Language Models (LLMs) has been driven by extensive datasets that may contain sensitive information, raising serious privacy concerns. One notable threat is the Membership Inference Attack (MIA), where adversaries infer whether a specific sample was used in model training. However, the true impact of MIA on LLMs remains unclear due to inconsistent findings and the lack of standardized evaluation methods, further complicated by the undisclosed nature of many LLM training sets. To address these limitations, we propose PerProb, a unified, label-free framework for indirectly assessing LLM memorization vulnerabilities. PerProb evaluates changes in perplexity and average log probability between data generated by victim and adversary models, enabling an indirect estimation of training-induced memory. Compared with prior MIA methods that rely on member/non-member labels or internal access, PerProb is independent of model and task, and applicable in both black-box and white-box settings. Through a systematic classification of MIA into four attack patterns, we evaluate PerProb's effectiveness across five datasets, revealing varying memory behaviors and privacy risks among LLMs. Additionally, we assess mitigation strategies, including knowledge distillation, early stopping, and differential privacy, demonstrating their effectiveness in reducing data leakage. Our findings offer a practical and generalizable framework for evaluating and improving LLM privacy.

摘要: 大型语言模型（LLM）的快速发展是由可能包含敏感信息的大量数据集推动的，从而引发了严重的隐私问题。一个值得注意的威胁是会员推断攻击（MIA），对手推断特定样本是否用于模型训练。然而，由于调查结果不一致和缺乏标准化的评估方法，MIA对LLM的真正影响仍然不清楚，而且许多LLM培训集的未公开性质使其更加复杂。为了解决这些限制，我们提出了PerProb，这是一个统一的、无标签的框架，用于间接评估LLM记忆漏洞。PerProb评估受害者和对手模型生成的数据之间的困惑度和平均日志概率的变化，从而能够间接估计训练诱导的记忆。与之前依赖成员/非成员标签或内部访问的MIA方法相比，PerProb独立于模型和任务，适用于黑盒和白盒设置。通过将MIA系统地分类为四种攻击模式，我们评估了PerProb在五个数据集中的有效性，揭示了LLM之间不同的记忆行为和隐私风险。此外，我们还评估了缓解策略，包括知识提炼、提前停止和差异隐私，证明它们在减少数据泄露方面的有效性。我们的研究结果为评估和改善LLM隐私提供了一个实用和可推广的框架。



## **30. Reasoning-Style Poisoning of LLM Agents via Stealthy Style Transfer: Process-Level Attacks and Runtime Monitoring in RSV Space**

通过隐形方式转移对LLM代理的推理方式中毒：RSV空间中的过程级攻击和收件箱监控 cs.CR

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14448v1) [paper-pdf](https://arxiv.org/pdf/2512.14448v1)

**Authors**: Xingfu Zhou, Pengfei Wang

**Abstract**: Large Language Model (LLM) agents relying on external retrieval are increasingly deployed in high-stakes environments. While existing adversarial attacks primarily focus on content falsification or instruction injection, we identify a novel, process-oriented attack surface: the agent's reasoning style. We propose Reasoning-Style Poisoning (RSP), a paradigm that manipulates how agents process information rather than what they process. We introduce Generative Style Injection (GSI), an attack method that rewrites retrieved documents into pathological tones--specifically "analysis paralysis" or "cognitive haste"--without altering underlying facts or using explicit triggers. To quantify these shifts, we develop the Reasoning Style Vector (RSV), a metric tracking Verification depth, Self-confidence, and Attention focus. Experiments on HotpotQA and FEVER using ReAct, Reflection, and Tree of Thoughts (ToT) architectures reveal that GSI significantly degrades performance. It increases reasoning steps by up to 4.4 times or induces premature errors, successfully bypassing state-of-the-art content filters. Finally, we propose RSP-M, a lightweight runtime monitor that calculates RSV metrics in real-time and triggers alerts when values exceed safety thresholds. Our work demonstrates that reasoning style is a distinct, exploitable vulnerability, necessitating process-level defenses beyond static content analysis.

摘要: 依赖外部检索的大型语言模型（LLM）代理越来越多地部署在高风险环境中。虽然现有的对抗性攻击主要集中在内容伪造或指令注入上，但我们发现了一种新颖的、面向过程的攻击表面：代理的推理风格。我们提出了推理式中毒（RSP），这是一种操纵代理如何处理信息而不是处理内容的范式。我们引入了生成风格注入（GSI），这是一种攻击方法，将检索到的文档改写为病理性语气--特别是“分析瘫痪”或“认知仓促”--而无需改变基本事实或使用明确的触发器。为了量化这些转变，我们开发了推理风格载体（RSV），这是一种跟踪验证深度、自信和注意力焦点的指标。使用ReAct、ReReReflection和Tree of Thoughts（ToT）架构对HotpotQA和FEVER进行的实验表明，GSI会显着降低性能。它将推理步骤增加多达4.4倍，否则会导致过早错误，从而成功绕过最先进的内容过滤器。最后，我们提出了RSP-M，这是一种轻量级的运行时监视器，可实时计算RSV指标，并在值超过安全阈值时触发警报。我们的工作表明，推理风格是一个独特的、可利用的漏洞，需要静态内容分析之外的流程级防御。



## **31. Semantic Mismatch and Perceptual Degradation: A New Perspective on Image Editing Immunity**

语义不匹配和感知退化：图像编辑免疫的新视角 cs.CV

11 pages, 4 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14320v1) [paper-pdf](https://arxiv.org/pdf/2512.14320v1)

**Authors**: Shuai Dong, Jie Zhang, Guoying Zhao, Shiguang Shan, Xilin Chen

**Abstract**: Text-guided image editing via diffusion models, while powerful, raises significant concerns about misuse, motivating efforts to immunize images against unauthorized edits using imperceptible perturbations. Prevailing metrics for evaluating immunization success typically rely on measuring the visual dissimilarity between the output generated from a protected image and a reference output generated from the unprotected original. This approach fundamentally overlooks the core requirement of image immunization, which is to disrupt semantic alignment with attacker intent, regardless of deviation from any specific output. We argue that immunization success should instead be defined by the edited output either semantically mismatching the prompt or suffering substantial perceptual degradations, both of which thwart malicious intent. To operationalize this principle, we propose Synergistic Intermediate Feature Manipulation (SIFM), a method that strategically perturbs intermediate diffusion features through dual synergistic objectives: (1) maximizing feature divergence from the original edit trajectory to disrupt semantic alignment with the expected edit, and (2) minimizing feature norms to induce perceptual degradations. Furthermore, we introduce the Immunization Success Rate (ISR), a novel metric designed to rigorously quantify true immunization efficacy for the first time. ISR quantifies the proportion of edits where immunization induces either semantic failure relative to the prompt or significant perceptual degradations, assessed via Multimodal Large Language Models (MLLMs). Extensive experiments show our SIFM achieves the state-of-the-art performance for safeguarding visual content against malicious diffusion-based manipulation.

摘要: 通过扩散模型进行文本引导的图像编辑虽然功能强大，但引起了对滥用的严重关注，促使人们努力使用不可察觉的扰动使图像免受未经授权的编辑。用于评估免疫成功的流行度量通常依赖于测量从受保护图像生成的输出与从未受保护原始图像生成的参考输出之间的视觉不相似性。这种方法从根本上忽视了图像免疫的核心要求，即破坏与攻击者意图的语义一致，无论是否偏离任何特定输出。我们认为，免疫成功应该由编辑后的输出来定义，要么在语义上与提示不匹配，要么遭受严重的感知退化，这两者都会阻止恶意意图。为了实现这一原则，我们提出了协同中间特征操作（SIFM），这是一种通过双重协同目标战略性地干扰中间扩散特征的方法：（1）最大化与原始编辑轨迹的特征分歧，以破坏与预期编辑的语义对齐，以及（2）最小化特征规范，以诱导感知退化。此外，我们还引入了免疫成功率（ISR），这是一种旨在首次严格量化真实免疫效力的新指标。ISR量化了免疫诱导语义失败相对于提示或显著感知退化的编辑比例，通过多模态大语言模型（MLLM）进行评估。大量的实验表明，我们的SIFM实现了最先进的性能，以保护视觉内容免受恶意的基于扩散的操纵。



## **32. PentestEval: Benchmarking LLM-based Penetration Testing with Modular and Stage-Level Design**

PentestEval：通过模块化和阶段级设计对基于LLM的渗透测试进行基准测试 cs.SE

13 pages, 6 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14233v1) [paper-pdf](https://arxiv.org/pdf/2512.14233v1)

**Authors**: Ruozhao Yang, Mingfei Cheng, Gelei Deng, Tianwei Zhang, Junjie Wang, Xiaofei Xie

**Abstract**: Penetration testing is essential for assessing and strengthening system security against real-world threats, yet traditional workflows remain highly manual, expertise-intensive, and difficult to scale. Although recent advances in Large Language Models (LLMs) offer promising opportunities for automation, existing applications rely on simplistic prompting without task decomposition or domain adaptation, resulting in unreliable black-box behavior and limited insight into model capabilities across penetration testing stages. To address this gap, we introduce PentestEval, the first comprehensive benchmark for evaluating LLMs across six decomposed penetration testing stages: Information Collection, Weakness Gathering and Filtering, Attack Decision-Making, Exploit Generation and Revision. PentestEval integrates expert-annotated ground truth with a fully automated evaluation pipeline across 346 tasks covering all stages in 12 realistic vulnerable scenarios. Our stage-level evaluation of 9 widely used LLMs reveals generally weak performance and distinct limitations across the stages of penetration-testing workflow. End-to-end pipelines reach only 31% success rate, and existing LLM-powered systems such as PentestGPT, PentestAgent, and VulnBot exhibit similar limitations, with autonomous agents failing almost entirely. These findings highlight that autonomous penetration testing demands stronger structured reasoning, where modularization enhances each individual stage and improves overall performance. PentestEval provides the foundational benchmark needed for future research on fine-grained, stage-level evaluation, paving the way toward more reliable LLM-based automation.

摘要: 渗透测试对于评估和加强系统安全性以应对现实世界的威胁至关重要，但传统工作流程仍然高度手动、专业知识密集且难以扩展。尽管大型语言模型（LLM）的最新进展为自动化提供了有希望的机会，但现有的应用程序依赖于简单化的提示，而没有任务分解或域适应，导致黑匣子行为不可靠，并且对渗透测试阶段模型能力的洞察有限。为了弥补这一差距，我们引入了PentestEval，这是第一个用于评估跨六个分解渗透测试阶段LLM的综合基准：信息收集、弱点收集和过滤、攻击决策、漏洞利用生成和修订。PentestEval将专家注释的基本真相与全自动评估管道集成，涵盖346项任务，涵盖12个现实脆弱场景的所有阶段。我们对9个广泛使用的LLM的阶段级评估显示，渗透测试工作流程各个阶段的性能普遍较弱，并且存在明显的局限性。端到端管道的成功率仅为31%，而现有的LLM驱动系统（例如PentestGPT、PentestAgent和VulnBot）也表现出类似的局限性，自主代理几乎完全失败。这些发现凸显了自主渗透测试需要更强的结构化推理，其中模块化增强了每个单独阶段并提高了整体性能。PentestEval为未来细粒度、阶段级评估研究提供了所需的基础基准，为更可靠的基于LLM的自动化铺平了道路。



## **33. IntentMiner: Intent Inversion Attack via Tool Call Analysis in the Model Context Protocol**

IntentMiner：通过模型上下文协议中的工具调用分析进行意图倒置攻击 cs.CR

12 pages, 6 figures

**SubmitDate**: 2025-12-16    [abs](http://arxiv.org/abs/2512.14166v1) [paper-pdf](https://arxiv.org/pdf/2512.14166v1)

**Authors**: Yunhao Yao, Zhiqiang Wang, Haoran Cheng, Yihang Cheng, Haohua Du, Xiang-Yang Li

**Abstract**: The rapid evolution of Large Language Models (LLMs) into autonomous agents has led to the adoption of the Model Context Protocol (MCP) as a standard for discovering and invoking external tools. While this architecture decouples the reasoning engine from tool execution to enhance scalability, it introduces a significant privacy surface: third-party MCP servers, acting as semi-honest intermediaries, can observe detailed tool interaction logs outside the user's trusted boundary. In this paper, we first identify and formalize a novel privacy threat termed Intent Inversion, where a semi-honest MCP server attempts to reconstruct the user's private underlying intent solely by analyzing legitimate tool calls. To systematically assess this vulnerability, we propose IntentMiner, a framework that leverages Hierarchical Information Isolation and Three-Dimensional Semantic Analysis, integrating tool purpose, call statements, and returned results, to accurately infer user intent at the step level. Extensive experiments demonstrate that IntentMiner achieves a high degree of semantic alignment (over 85%) with original user queries, significantly outperforming baseline approaches. These results highlight the inherent privacy risks in decoupled agent architectures, revealing that seemingly benign tool execution logs can serve as a potent vector for exposing user secrets.

摘要: 大型语言模型（LLM）快速演变为自治代理，导致模型上下文协议（HCP）被采用作为发现和调用外部工具的标准。虽然该架构将推理引擎与工具执行分开以增强可扩展性，但它引入了一个重要的隐私表面：充当半诚实中介的第三方LCP服务器可以观察用户可信边界之外的详细工具交互日志。在本文中，我们首先识别并正式化一种名为“意图倒置”的新型隐私威胁，其中半诚实的LCP服务器尝试仅通过分析合法的工具调用来重建用户的私人底层意图。为了系统性地评估此漏洞，我们提出了IntentMiner，这是一个利用分层信息隔离和三维语义分析的框架，集成工具目的、调用陈述和返回的结果，可以在步骤级别准确地推断用户意图。大量实验表明，IntentMiner与原始用户查询实现了高度的语义一致性（超过85%），显着优于基线方法。这些结果凸显了脱钩代理架构中固有的隐私风险，揭示了看似良性的工具执行日志可以作为暴露用户秘密的有力载体。



## **34. CNFinBench: A Benchmark for Safety and Compliance of Large Language Models in Finance**

CNFinBench：金融领域大型语言模型安全和合规性的基准 cs.CE

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.09506v2) [paper-pdf](https://arxiv.org/pdf/2512.09506v2)

**Authors**: Jinru Ding, Chao Ding, Wenrao Pang, Boyi Xiao, Zhiqiang Liu, Pengcheng Chen, Jiayuan Chen, Tiantian Yuan, Junming Guan, Yidong Jiang, Dawei Cheng, Jie Xu

**Abstract**: Large language models (LLMs) are increasingly deployed across the financial sector for tasks like investment research and algorithmic trading. Their high-stakes nature demands rigorous evaluation of models' safety and regulatory alignment. However, there is a significant gap between evaluation capabilities and safety requirements. Current financial benchmarks mainly focus on textbook-style question answering and numerical problem-solving, failing to simulate the open-ended scenarios where safety risks typically manifest. To close these gaps, we introduce CNFinBench, a benchmark structured around a Capability-Compliance-Safety triad encompassing 15 subtasks. For Capability Q&As, we introduce a novel business-vertical taxonomy aligned with core financial domains like banking operations, which allows institutions to assess model readiness for deployment in operational scenarios. For Compliance and Risk Control Q&As, we embed regulatory requirements within realistic business scenarios to ensure models are evaluated under practical, scenario-driven conditions. For Safety Q&As, we uniquely incorporate structured bias and fairness auditing, a dimension overlooked by other holistic financial benchmarks, and introduce the first multi-turn adversarial dialogue task to systematically expose compliance decay under sustained, context-aware attacks. Accordingly, we propose the Harmful Instruction Compliance Score (HICS) to quantify models' consistency in resisting harmful instructions across multi-turn dialogues. Experiments on 21 models across all subtasks reveal a persistent gap between capability and compliance: models achieve an average score of 61.0 on capability tasks but drop to 34.2 on compliance and risk-control evaluations. In multi-turn adversarial dialogue tests, most LLMs attain only partial resistance, demonstrating that refusal alone is insufficient without cited, verifiable reasoning.

摘要: 大型语言模型（LLM）越来越多地被部署在金融领域，用于投资研究和算法交易等任务。它们的高风险性质要求对模型的安全性和监管一致性进行严格评估。然而，评估能力与安全要求之间存在明显差距。当前的财务基准主要集中在教科书式的问答和数字问题解决上，未能模拟安全风险通常显现的开放式场景。为了缩小这些差距，我们引入了CNFinBench，这是一个围绕能力-合规-安全三位一体构建的基准，包含15个子任务。对于能力问答，我们引入了一种与银行运营等核心金融领域保持一致的新型业务垂直分类法，使机构能够评估模型在运营场景中部署的准备情况。对于合规和风险控制问答，我们将监管要求嵌入现实的业务场景中，以确保模型在实际的、业务驱动的条件下进行评估。对于安全问答，我们独特地结合了结构性偏见和公平性审计（这是其他整体财务基准所忽视的一个维度），并引入了第一个多回合对抗性对话任务，以系统性地揭露持续、上下文感知攻击下的合规性衰退。因此，我们提出了有害指令合规评分（HICS）来量化模型在多轮对话中抵抗有害指令的一致性。对所有子任务的21个模型进行的实验揭示了能力与合规性之间持续存在的差距：模型在能力任务上的平均得分为61.0，但在合规和风险控制评估上的平均得分下降至34.2。在多轮对抗性对话测试中，大多数LLM仅获得部分抵抗，这表明如果没有引用的、可验证的推理，仅靠拒绝是不够的。



## **35. The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search**

木马知识：通过无害提示编织和自适应树搜索破解商业LLM护栏 cs.CR

Updated with new baselines and experimental results

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2512.01353v3) [paper-pdf](https://arxiv.org/pdf/2512.01353v3)

**Authors**: Rongzhe Wei, Peizhi Niu, Xinjie Shen, Tony Tu, Yifan Li, Ruihan Wu, Eli Chien, Pin-Yu Chen, Olgica Milenkovic, Pan Li

**Abstract**: Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Existing approaches overwhelmingly operate within the prompt-optimization paradigm: whether through traditional algorithmic search or recent agent-based workflows, the resulting prompts typically retain malicious semantic signals that modern guardrails are primed to detect. In contrast, we identify a deeper, largely overlooked vulnerability stemming from the highly interconnected nature of an LLM's internal knowledge. This structure allows harmful objectives to be realized by weaving together sequences of benign sub-queries, each of which individually evades detection. To exploit this loophole, we introduce the Correlated Knowledge Attack Agent (CKA-Agent), a dynamic framework that reframes jailbreaking as an adaptive, tree-structured exploration of the target model's knowledge base. The CKA-Agent issues locally innocuous queries, uses model responses to guide exploration across multiple paths, and ultimately assembles the aggregated information to achieve the original harmful objective. Evaluated across state-of-the-art commercial LLMs (Gemini2.5-Flash/Pro, GPT-oss-120B, Claude-Haiku-4.5), CKA-Agent consistently achieves over 95% success rates even against strong guardrails, underscoring the severity of this vulnerability and the urgent need for defenses against such knowledge-decomposition attacks. Our codes are available at https://github.com/Graph-COM/CKA-Agent.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，这些攻击绕过安全护栏以引发有害输出。现有方法绝大多数在预算优化范式下运行：无论是通过传统的算法搜索还是最近的基于代理的工作流程，产生的提示通常都会保留现代护栏准备好检测的恶意语义信号。相比之下，我们发现了一个更深层次的、在很大程度上被忽视的漏洞，该漏洞源于法学硕士内部知识的高度相互关联的性质。这种结构允许通过将良性子查询序列编织在一起来实现有害目标，每个子查询都单独逃避检测。为了利用这个漏洞，我们引入了相关知识攻击代理（CKA-Agent），这是一个动态框架，它将越狱重新构建为对目标模型知识库的自适应、树结构化探索。CKA-Agent发出本地无害的查询，使用模型响应来指导跨多个路径的探索，并最终聚集信息以实现最初的有害目标。经过最先进的商业LLM（Gemini 2.5-Flash/Pro、GPT-oss-120 B、Claude-Haiku-4.5）的评估，即使在强大的护栏下，CKA-Agent也始终实现了超过95%的成功率，凸显了该漏洞的严重性以及对此类知识分解攻击的防御的迫切需要。我们的代码可在https://github.com/Graph-COM/CKA-Agent上获取。



## **36. Phantom Menace: Exploring and Enhancing the Robustness of VLA Models Against Physical Sensor Attacks**

幻影威胁：探索和增强VLA模型对抗物理传感器攻击的鲁棒性 cs.RO

Accepted by AAAI 2026 main track

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2511.10008v2) [paper-pdf](https://arxiv.org/pdf/2511.10008v2)

**Authors**: Xuancun Lu, Jiaxiang Chen, Shilin Xiao, Zizhi Jin, Zhangrui Chen, Hanwen Yu, Bohan Qian, Ruochen Zhou, Xiaoyu Ji, Wenyuan Xu

**Abstract**: Vision-Language-Action (VLA) models revolutionize robotic systems by enabling end-to-end perception-to-action pipelines that integrate multiple sensory modalities, such as visual signals processed by cameras and auditory signals captured by microphones. This multi-modality integration allows VLA models to interpret complex, real-world environments using diverse sensor data streams. Given the fact that VLA-based systems heavily rely on the sensory input, the security of VLA models against physical-world sensor attacks remains critically underexplored. To address this gap, we present the first systematic study of physical sensor attacks against VLAs, quantifying the influence of sensor attacks and investigating the defenses for VLA models. We introduce a novel "Real-Sim-Real" framework that automatically simulates physics-based sensor attack vectors, including six attacks targeting cameras and two targeting microphones, and validates them on real robotic systems. Through large-scale evaluations across various VLA architectures and tasks under varying attack parameters, we demonstrate significant vulnerabilities, with susceptibility patterns that reveal critical dependencies on task types and model designs. We further develop an adversarial-training-based defense that enhances VLA robustness against out-of-distribution physical perturbations caused by sensor attacks while preserving model performance. Our findings expose an urgent need for standardized robustness benchmarks and mitigation strategies to secure VLA deployments in safety-critical environments.

摘要: 视觉-语言-动作（VLA）模型通过实现端到端的感知到动作管道，彻底改变了机器人系统，该管道集成了多种感官模式，例如由摄像机处理的视觉信号和由麦克风捕获的听觉信号。这种多模式集成使VLA模型能够使用不同的传感器数据流来解释复杂的现实世界环境。鉴于基于VLA的系统严重依赖感官输入，VLA模型对抗物理世界传感器攻击的安全性仍然严重不足。为了弥补这一差距，我们首次对针对VLA的物理传感器攻击进行了系统研究，量化了传感器攻击的影响并调查VLA模型的防御。我们引入了一个新颖的“Real-Sim-Real”框架，该框架自动模拟基于物理的传感器攻击载体，包括六次针对摄像头和两个针对麦克风的攻击，并在真实的机器人系统上对其进行验证。通过在不同攻击参数下对各种VLA架构和任务进行大规模评估，我们展示了显着的漏洞，其易感性模式揭示了对任务类型和模型设计的关键依赖性。我们进一步开发了一种基于对抗训练的防御，可以增强VLA对传感器攻击引起的分布外物理扰动的鲁棒性，同时保持模型性能。我们的研究结果揭示了迫切需要标准化的稳健性基准和缓解策略，以确保VLA在安全关键环境中的部署。



## **37. Biologically-Informed Hybrid Membership Inference Attacks on Generative Genomic Models**

对生成性基因组模型的生物知情混合成员推断攻击 cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2511.07503v3) [paper-pdf](https://arxiv.org/pdf/2511.07503v3)

**Authors**: Asia Belfiore, Jonathan Passerat-Palmbach, Dmitrii Usynin

**Abstract**: The increased availability of genetic data has transformed genomics research, but raised many privacy concerns regarding its handling due to its sensitive nature. This work explores the use of language models (LMs) for the generation of synthetic genetic mutation profiles, leveraging differential privacy (DP) for the protection of sensitive genetic data. We empirically evaluate the privacy guarantees of our DP modes by introducing a novel Biologically-Informed Hybrid Membership Inference Attack (biHMIA), which combines traditional black box MIA with contextual genomics metrics for enhanced attack power. Our experiments show that both small and large transformer GPT-like models are viable synthetic variant generators for small-scale genomics, and that our hybrid attack leads, on average, to higher adversarial success compared to traditional metric-based MIAs.

摘要: 遗传数据可用性的增加改变了基因组学研究，但由于其敏感性，对其处理提出了许多隐私问题。这项工作探索了使用语言模型（LM）来生成合成基因突变谱，利用差异隐私（DP）来保护敏感遗传数据。我们通过引入一种新型的生物知情混合成员推断攻击（biHMIA）来经验性地评估DP模式的隐私保证，该攻击将传统的黑匣子MIA与上下文基因组学指标相结合，以增强攻击能力。我们的实验表明，小型和大型Transformer GPT类模型都是小规模基因组学的可行合成变体生成器，并且与传统的基于度量的MIA相比，我们的混合攻击平均会导致更高的对抗成功。



## **38. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

AutoAdv：大型语言模型多回合越狱的自动对抗预算 cs.CL

Presented at NeurIPS 2025 Lock-LLM Workshop. Code is available at https://github.com/AAN-AutoAdv/AutoAdv

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2511.02376v3) [paper-pdf](https://arxiv.org/pdf/2511.02376v3)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs. Yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves an attack success rate of up to 95% on Llama-3.1-8B within six turns, a 24% improvement over single-turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests and then iteratively refines them. Extensive evaluation across commercial and open-source models (Llama-3.1-8B, GPT-4o mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，对抗性提示会引发有害输出。然而，大多数评估都集中在单轮交互上，而现实世界的攻击则通过自适应多轮对话展开。我们介绍了AutoAdv，这是一个用于自动多回合越狱的免训练框架，在六个回合内对Llama-3.1-8B的攻击成功率高达95%，比单回合基线提高了24%。AutoAdv独特地结合了三种自适应机制：从成功的攻击中学习以增强未来提示的模式管理器、根据失败模式动态调整采样参数的温度管理器以及掩盖有害请求然后迭代细化的两阶段重写策略。对商业和开源模型（Llama-3.1-8B、GPT-4 o mini、Qwen 3 - 235 B、Mistral-7 B）的广泛评估揭示了当前安全机制中存在的持续漏洞，多回合攻击的表现始终优于单回合方法。这些发现表明，针对单轮交互优化的对齐策略无法在扩展对话中保持稳健性，凸显了对多轮感知防御的迫切需求。



## **39. RAID: Refusal-Aware and Integrated Decoding for Jailbreaking LLMs**

RAGE：适用于越狱LLM的参考感知和集成解码 cs.CL

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2510.13901v2) [paper-pdf](https://arxiv.org/pdf/2510.13901v2)

**Authors**: Tuan T. Nguyen, John Le, Thai T. Vu, Willy Susilo, Heath Cooper

**Abstract**: Large language models (LLMs) achieve impressive performance across diverse tasks yet remain vulnerable to jailbreak attacks that bypass safety mechanisms. We present RAID (Refusal-Aware and Integrated Decoding), a framework that systematically probes these weaknesses by crafting adversarial suffixes that induce restricted content while preserving fluency. RAID relaxes discrete tokens into continuous embeddings and optimizes them with a joint objective that (i) encourages restricted responses, (ii) incorporates a refusal-aware regularizer to steer activations away from refusal directions in embedding space, and (iii) applies a coherence term to maintain semantic plausibility and non-redundancy. After optimization, a critic-guided decoding procedure maps embeddings back to tokens by balancing embedding affinity with language-model likelihood. This integration yields suffixes that are both effective in bypassing defenses and natural in form. Experiments on multiple open-source LLMs show that RAID achieves higher attack success rates with fewer queries and lower computational cost than recent white-box and black-box baselines. These findings highlight the importance of embedding-space regularization for understanding and mitigating LLM jailbreak vulnerabilities.

摘要: 大型语言模型（LLM）在不同任务中取得了令人印象深刻的性能，但仍然容易受到绕过安全机制的越狱攻击。我们介绍了RAID-Aware（参考感知和集成解码），这是一个框架，通过制作对抗性后缀来系统性地探索这些弱点，这些后缀在保持流畅性的同时引入受限制的内容。RAID将离散令牌放松为连续嵌入，并通过联合目标对其进行优化，该目标（i）鼓励限制响应，（ii）合并反推感知规则化器以引导激活远离嵌入空间中的拒绝方向，以及（iii）应用一致性项来保持语义一致性和非冗余性。优化后，批评引导的解码过程通过平衡嵌入亲和力与语言模型可能性来将嵌入映射回令牌。这种集成产生的后缀既可以有效绕过防御，而且形式自然。在多个开源LLM上的实验表明，与最近的白盒和黑盒基线相比，与更少的查询和更低的计算成本实现了更高的攻击成功率。这些发现凸显了嵌入空间规范化对于理解和缓解LLM越狱漏洞的重要性。



## **40. A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks**

针对即时注入攻击的多代理LLM防御管道 cs.CR

Accepted at the 11th IEEE WIECON-ECE 2025

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2509.14285v4) [paper-pdf](https://arxiv.org/pdf/2509.14285v4)

**Authors**: S M Asif Hossain, Ruksat Khan Shayoni, Mohd Ruhul Ameen, Akif Islam, M. F. Mridha, Jungpil Shin

**Abstract**: Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.

摘要: 提示注入攻击是大型语言模型（LLM）部署中的一个主要漏洞，用户输入中嵌入的恶意指令可以覆盖系统提示并引发意外行为。本文提出了一种新型的多代理防御框架，该框架在协调管道中使用专门的LLM代理来实时检测和抵消即时注入攻击。我们使用两种不同的架构来评估我们的方法：顺序代理链管道和基于分层协调器的系统。我们对两个LLM平台（ChatGLM和Llama 2）上的55种独特的即时注入攻击（分为8类，总共400个攻击实例）进行了全面评估，展示了显着的安全改进。在没有防御机制的情况下，ChatGLM的基线攻击成功率（ASB）达到30%，Llama 2的基线攻击成功率（ASB）达到20%。我们的多代理管道实现了100%的缓解，在所有测试场景中将ASB降低至0%。该框架展示了多种攻击类别的稳健性，包括直接覆盖、代码执行尝试、数据溢出和模糊技术，同时维护合法查询的系统功能。



## **41. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

相信我，我知道这个功能：使用偏差劫持LLM静态分析 cs.LG

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2508.17361v2) [paper-pdf](https://arxiv.org/pdf/2508.17361v2)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective against basic and reasoning models, but are also transferable across model families (OpenAI, Anthropic, Google), and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.

摘要: 大型语言模型（LLM）越来越被信任大规模执行自动代码审查和静态分析，支持漏洞检测、总结和重构等任务。在本文中，我们识别并利用基于LLM的代码分析中的一个关键漏洞：一种抽象偏见，导致模型过度概括熟悉的编程模式并忽略小而有意义的错误。对手可以利用这个盲点来劫持LLM解释的控制流，只需最少的编辑，并且不会影响实际的运行时行为。我们将这种攻击称为熟悉模式攻击（FTA）。   我们开发了一种全自动的黑匣子算法，可以发现并将FPA注入目标代码。我们的评估表明，PFA不仅对基本模型和推理模型有效，而且还可以跨模型家族（OpenAI、Anthropic、Google）移植，并且跨编程语言（Python、C、Rust、Go）通用。此外，即使通过强大的系统提示明确警告模型有关攻击，FPA仍然有效。最后，我们探讨了PFA的积极、防御性用途，并讨论了它们对面向代码的LLM的可靠性和安全性的更广泛影响。



## **42. May I have your Attention? Breaking Fine-Tuning based Prompt Injection Defenses using Architecture-Aware Attacks**

请注意吗？使用架构感知攻击突破基于微调的提示注入防御 cs.CR

**SubmitDate**: 2025-12-17    [abs](http://arxiv.org/abs/2507.07417v2) [paper-pdf](https://arxiv.org/pdf/2507.07417v2)

**Authors**: Nishit V. Pandya, Andrey Labunets, Sicun Gao, Earlence Fernandes

**Abstract**: A popular class of defenses against prompt injection attacks on large language models (LLMs) relies on fine-tuning to separate instructions and data, so that the LLM does not follow instructions that might be present with data. We evaluate the robustness of this approach in the whitebox setting by constructing strong optimization-based attacks, and show that the defenses do not provide the claimed security properties. Specifically, we construct a novel attention-based attack algorithm for textual LLMs and apply it to three recent whitebox defenses SecAlign (CCS 2025), SecAlign++, and StruQ (USENIX Security 2025), showing attacks with success rates of up to \textbf{85-95\%} on unseen prompts with modest increase in attacker budget in terms of tokens. Our findings make fundamental progress towards understanding the robustness of prompt injection defenses in the whitebox setting. We release our code and attacks at https://github.com/nishitvp/better_opts_attacks

摘要: 针对大型语言模型（LLM）上的即时注入攻击的一类流行防御依赖于微调以分离指令和数据，以便LLM不会遵循可能存在于数据中的指令。我们通过构建强大的基于优化的攻击来评估这种方法在白盒环境中的稳健性，并表明防御措施不提供声称的安全属性。具体来说，我们为文本LLM构建了一种新颖的基于注意力的攻击算法，并将其应用于最近的三种白盒防御SecAlign（CCS 2025）、SecAlign++和StruQ（USENIX Security 2025），在未见提示上展示了成功率高达\textBF{85-95\%}的攻击，攻击者预算在代币方面略有增加。我们的研究结果在理解白盒环境中即时注射防御的稳健性方面取得了根本性进展。我们在https://github.com/nishitvp/better_opts_attacks上发布我们的代码和攻击



## **43. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

论LLM在对抗性攻击中言语信心的稳健性 cs.CL

Published in NeurIPS 2025

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2507.06489v3) [paper-pdf](https://arxiv.org/pdf/2507.06489v3)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.

摘要: 大型语言模型（LLM）产生的强大言语信心对于LLM的部署至关重要，以帮助确保许多应用程序（包括涉及人机交互的应用程序）的透明度、信任和安全性。在本文中，我们首次对对抗攻击下言语信心的稳健性进行了全面研究。我们通过干扰和基于越狱的方法引入了针对言语信心分数的攻击框架，并证明这些攻击会显着损害言语信心估计并导致答案频繁变化。我们检查了各种提示策略、模型大小和应用领域，揭示了当前的言语自信很脆弱，并且常用的防御技术在很大程度上无效或适得其反。我们的研究结果强调了为LLM中的信心表达设计稳健的机制的必要性，因为即使是微妙的语义保留修改也可能导致反应中的误导性信心。



## **44. Universal Jailbreak Suffixes Are Strong Attention Hijackers**

通用越狱后缀是强烈的注意力劫持者 cs.CR

Accepted at TACL 2026

**SubmitDate**: 2025-12-21    [abs](http://arxiv.org/abs/2506.12880v2) [paper-pdf](https://arxiv.org/pdf/2506.12880v2)

**Authors**: Matan Ben-Tov, Mor Geva, Mahmood Sharif

**Abstract**: We study suffix-based jailbreaks$\unicode{x2013}$a powerful family of attacks against large language models (LLMs) that optimize adversarial suffixes to circumvent safety alignment. Focusing on the widely used foundational GCG attack, we observe that suffixes vary in efficacy: some are markedly more universal$\unicode{x2013}$generalizing to many unseen harmful instructions$\unicode{x2013}$than others. We first show that a shallow, critical mechanism drives GCG's effectiveness. This mechanism builds on the information flow from the adversarial suffix to the final chat template tokens before generation. Quantifying the dominance of this mechanism during generation, we find GCG irregularly and aggressively hijacks the contextualization process. Crucially, we tie hijacking to the universality phenomenon, with more universal suffixes being stronger hijackers. Subsequently, we show that these insights have practical implications: GCG's universality can be efficiently enhanced (up to $\times$5 in some cases) at no additional computational cost, and can also be surgically mitigated, at least halving the attack's success with minimal utility loss. We release our code and data at http://github.com/matanbt/interp-jailbreak.

摘要: 我们研究基于后缀的越狱$\unicode{x2013}$这是一个针对大型语言模型（LLM）的强大攻击家族，这些模型优化对抗性后缀以规避安全对齐。关注广泛使用的基础GCG攻击，我们观察到后缀的功效各不相同：有些后缀明显更通用$\unicode{x2013}$，一般化为许多不可见的有害指令$\unicode{x2013}$。我们首先表明，GCG的有效性是一种肤浅的、关键的机制。该机制建立在生成之前从对抗性后缀到最终聊天模板令牌的信息流之上。量化这种机制在生成过程中的主导地位，我们发现GCG不规则且积极地劫持了情境化过程。至关重要的是，我们将劫持与普遍现象联系起来，更普遍的后缀意味着更强大的劫持者。随后，我们证明了这些见解具有实际意义：GCG的普遍性可以在没有额外计算成本的情况下有效地增强（在某些情况下高达5美元），并且还可以通过手术来减轻，至少将攻击的成功减半，并将效用损失最小。我们在http://github.com/matanbt/interp-jailbreak上发布我们的代码和数据。



## **45. SoK: Are Watermarks in LLMs Ready for Deployment?**

SoK：LLM中的水印准备好部署了吗？ cs.CR

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2506.05594v3) [paper-pdf](https://arxiv.org/pdf/2506.05594v3)

**Authors**: Kieu Dang, Phung Lai, NhatHai Phan, Yelong Shen, Ruoming Jin, Abdallah Khreishah, My T. Thai

**Abstract**: Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs.   To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.

摘要: 大型语言模型（LLM）改变了自然语言处理，在不同任务中展示了令人印象深刻的能力。然而，部署这些模型会带来与知识产权侵犯和潜在滥用相关的关键风险，特别是因为对手可以模仿这些模型来窃取服务或产生误导性输出。我们特别关注模型窃取攻击，因为它们与专有LLM高度相关，并对其安全、收入和道德部署构成严重威胁。虽然已经出现了各种水印技术来减轻这些风险，但目前尚不清楚社区和行业在LLM中开发和部署水印方面取得了多大进展。   为了弥合这一差距，我们的目标是通过1）提供LLM中水印的详细分类，2）提出一种新型知识产权分类器来探索水印在攻击和无攻击环境下的有效性和影响，3）分析LLM中现有水印的局限性，4）讨论LLM中水印的实际挑战和潜在的未来方向。通过广泛的实验，我们表明，尽管研究成果令人鼓舞，领先公司和社区也对部署水印给予了极大的关注，但由于这些技术对LLM和下游任务的模型效用产生不利影响，这些技术尚未在现实世界应用中充分发挥潜力。我们的研究结果提供了对LLM中的水印的深刻理解，强调了针对LLM部署量身定制的实用水印解决方案的必要性。



## **46. Efficient and Stealthy Jailbreak Attacks via Adversarial Prompt Distillation from LLMs to SLMs**

通过从LLM到SLM的对抗性即时蒸馏进行高效且隐蔽的越狱攻击 cs.CL

19 pages, 7 figures

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2506.17231v2) [paper-pdf](https://arxiv.org/pdf/2506.17231v2)

**Authors**: Xiang Li, Chong Zhang, Jia Wang, Fangyu Wu, Yushi Li, Xiaobo Jin

**Abstract**: As the scale and complexity of jailbreaking attacks on large language models (LLMs) continue to escalate, their efficiency and practical applicability are constrained, posing a profound challenge to LLM security. Jailbreaking techniques have advanced from manual prompt engineering to automated methodologies. Recent advances have automated jailbreaking approaches that harness LLMs to generate jailbreak instructions and adversarial examples, delivering encouraging results. Nevertheless, these methods universally include an LLM generation phase, which, due to the complexities of deploying and reasoning with LLMs, impedes effective implementation and broader adoption. To mitigate this issue, we introduce \textbf{Adversarial Prompt Distillation}, an innovative framework that integrates masked language modeling, reinforcement learning, and dynamic temperature control to distill LLM jailbreaking prowess into smaller language models (SLMs). This methodology enables efficient, robust jailbreak attacks while maintaining high success rates and accommodating a broader range of application contexts. Empirical evaluations affirm the approach's superiority in attack efficacy, resource optimization, and cross-model versatility. Our research underscores the practicality of transferring jailbreak capabilities to SLMs, reveals inherent vulnerabilities in LLMs, and provides novel insights to advance LLM security investigations. Our code is available at: https://github.com/lxgem/Efficient_and_Stealthy_Jailbreak_Attacks_via_Adversarial_Prompt.

摘要: 随着对大型语言模型（LLM）的越狱攻击规模和复杂性不断升级，其效率和实用性受到限制，对LLM安全提出了深刻的挑战。越狱技术已经从手动提示工程发展到自动化方法。最近的进展已经自动化了越狱方法，利用LLM来生成越狱指令和对抗性示例，并带来了令人鼓舞的结果。然而，这些方法普遍包括LLM生成阶段，由于LLM部署和推理的复杂性，这阻碍了有效实施和更广泛的采用。为了缓解这个问题，我们引入了\textBF{对抗提示蒸馏}，这是一个创新框架，集成了掩蔽语言建模、强化学习和动态温度控制，将LLM越狱能力提炼成更小的语言模型（SLC）。这种方法可以实现高效、稳健的越狱攻击，同时保持高成功率并适应更广泛的应用程序上下文。实证评估证实了该方法在攻击功效、资源优化和跨模型通用性方面的优势。我们的研究强调了将越狱功能转移到SLC的可行性，揭示了LLC中的固有漏洞，并为推进LLM安全调查提供了新颖的见解。我们的代码可访问：https://github.com/lxgem/Efficient_and_Stealthy_Jailbreak_Attacks_via_Adversarial_Prompt。



## **47. MoAPT: Mixture of Adversarial Prompt Tuning for Vision-Language Models**

MoAPT：视觉语言模型的对抗性提示调优混合 cs.CV

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2505.17509v2) [paper-pdf](https://arxiv.org/pdf/2505.17509v2)

**Authors**: Shiji Zhao, Qihui Zhu, Shukun Xiong, Shouwei Ruan, Maoxun Yuan, Jialing Tao, Jiexi Liu, Ranjie Duan, Jie Zhang, Jie Zhang, Xingxing Wei

**Abstract**: Large pre-trained Vision Language Models (VLMs) demonstrate excellent generalization capabilities but remain highly susceptible to adversarial examples, posing potential security risks. To improve the robustness of VLMs against adversarial examples, adversarial prompt tuning methods are proposed to align the text feature with the adversarial image feature without changing model parameters. However, when facing various adversarial attacks, a single learnable text prompt has insufficient generalization to align well with all adversarial image features, which ultimately results in overfitting. To address the above challenge, in this paper, we empirically find that increasing the number of learned prompts yields greater robustness improvements than simply extending the length of a single prompt. Building on this observation, we propose an adversarial tuning method named \textbf{Mixture of Adversarial Prompt Tuning (MoAPT)} to enhance the generalization against various adversarial attacks for VLMs. MoAPT aims to learn mixture text prompts to obtain more robust text features. To further enhance the adaptability, we propose a conditional weight router based on the adversarial images to predict the mixture weights of multiple learned prompts, which helps obtain sample-specific mixture text features aligning with different adversarial image features. Extensive experiments across 11 datasets under different settings show that our method can achieve better adversarial robustness than state-of-the-art approaches.

摘要: 大型预先训练的视觉语言模型（VLM）表现出出色的概括能力，但仍然极易受到对抗性示例的影响，从而构成潜在的安全风险。为了提高VLM对对抗性示例的鲁棒性，提出了对抗性提示调整方法，以在不改变模型参数的情况下将文本特征与对抗性图像特征对齐。然而，当面临各种对抗性攻击时，单个可学习文本提示的概括性不足以与所有对抗性图像特征很好地对齐，这最终会导致过度匹配。为了解决上述挑战，在本文中，我们经验发现，增加学习提示的数量比简单地延长单个提示的长度可以产生更大的鲁棒性改进。在这一观察的基础上，我们提出了一种名为\textBF{混合对抗提示调整（MoAPT）}的对抗性调整方法，以增强针对VLM的各种对抗性攻击的概括性。MoAPT旨在学习混合文本提示以获得更稳健的文本特征。为了进一步增强适应性，我们提出了一种基于对抗图像的条件权重路由器来预测多个学习提示的混合权重，这有助于获得与不同对抗图像特征对齐的样本特定混合文本特征。在不同设置下对11个数据集进行的广泛实验表明，我们的方法可以实现比最先进的方法更好的对抗鲁棒性。



## **48. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

AutoAdv：大型语言模型多回合越狱的自动对抗预算 cs.CR

We encountered issues with the paper being hosted under my personal account, so we republished it under a different account associated with a university email, which makes updates and management easier. As a result, this version is a duplicate of arXiv:2511.02376

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2507.01020v2) [paper-pdf](https://arxiv.org/pdf/2507.01020v2)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities to jailbreaking attacks: carefully crafted malicious inputs intended to circumvent safety guardrails and elicit harmful responses. As such, we present AutoAdv, a novel framework that automates adversarial prompt generation to systematically evaluate and expose vulnerabilities in LLM safety mechanisms. Our approach leverages a parametric attacker LLM to produce semantically disguised malicious prompts through strategic rewriting techniques, specialized system prompts, and optimized hyperparameter configurations. The primary contribution of our work is a dynamic, multi-turn attack methodology that analyzes failed jailbreak attempts and iteratively generates refined follow-up prompts, leveraging techniques such as roleplaying, misdirection, and contextual manipulation. We quantitatively evaluate attack success rate (ASR) using the StrongREJECT (arXiv:2402.10260 [cs.CL]) framework across sequential interaction turns. Through extensive empirical evaluation of state-of-the-art models--including ChatGPT, Llama, and DeepSeek--we reveal significant vulnerabilities, with our automated attacks achieving jailbreak success rates of up to 86% for harmful content generation. Our findings reveal that current safety mechanisms remain susceptible to sophisticated multi-turn attacks, emphasizing the urgent need for more robust defense strategies.

摘要: 大型语言模型（LLM）继续表现出越狱攻击的漏洞：精心设计的恶意输入，旨在绕过安全护栏并引发有害响应。因此，我们提出了AutoAdv，这是一个新颖的框架，可以自动生成对抗提示，以系统地评估和暴露LLM安全机制中的漏洞。我们的方法利用参数攻击者LLM通过战略重写技术、专门的系统提示和优化的超参数配置来产生语义伪装的恶意提示。我们工作的主要贡献是一种动态、多回合攻击方法，该方法分析失败的越狱尝试，并利用角色扮演、误导和上下文操纵等技术迭代生成细化的后续提示。我们使用StrongRESYS（arXiv：2402.10260 [cs.CL]）框架在连续交互回合中量化评估攻击成功率（ASB）。通过对最先进模型（包括ChatGPT、Llama和DeepSeek）进行广泛的实证评估，我们揭示了重大漏洞，我们的自动攻击在有害内容生成方面实现了高达86%的越狱成功率。我们的研究结果表明，当前的安全机制仍然容易受到复杂的多回合攻击，这凸显了对更强大的防御策略的迫切需要。



## **49. Evolving Security in LLMs: A Study of Jailbreak Attacks and Defenses**

LLC中不断发展的安全性：越狱攻击和防御的研究 cs.CR

**SubmitDate**: 2025-12-24    [abs](http://arxiv.org/abs/2504.02080v2) [paper-pdf](https://arxiv.org/pdf/2504.02080v2)

**Authors**: Zhengchun Shang, Wenlan Wei, Weiheng Bai

**Abstract**: Large Language Models (LLMs) are increasingly popular, powering a wide range of applications. Their widespread use has sparked concerns, especially through jailbreak attacks that bypass safety measures to produce harmful content.   In this paper, we present a comprehensive security analysis of large language models (LLMs), addressing critical research questions on the evolution and determinants of model safety.   Specifically, we begin by identifying the most effective techniques for detecting jailbreak attacks. Next, we investigate whether newer versions of LLMs offer improved security compared to their predecessors. We also assess the impact of model size on overall security and explore the potential benefits of integrating multiple defense strategies to enhance the security.   Our study evaluates both open-source (e.g., LLaMA and Mistral) and closed-source models (e.g., GPT-4) by employing four state-of-the-art attack techniques and assessing the efficacy of three new defensive approaches.

摘要: 大型语言模型（LLM）越来越受欢迎，为广泛的应用程序提供支持。它们的广泛使用引发了人们的担忧，特别是通过越狱攻击绕过安全措施产生有害内容。   在本文中，我们提出了一个全面的安全分析的大型语言模型（LLM），解决关键的研究问题的演变和决定因素的模型安全性。   具体来说，我们首先确定检测越狱攻击的最有效的技术。接下来，我们调查较新版本的LLM是否比其前身提供了更好的安全性。我们还评估模型大小对整体安全性的影响，并探索集成多种防御策略以增强安全性的潜在好处。   我们的研究评估了开源（例如，LLaMA和Mistral）和闭源模型（例如，GPT-4）通过采用四种最先进的攻击技术并评估三种新防御方法的有效性。



## **50. Memory Backdoor Attacks on Neural Networks**

对神经网络的内存后门攻击 cs.CR

**SubmitDate**: 2025-12-18    [abs](http://arxiv.org/abs/2411.14516v2) [paper-pdf](https://arxiv.org/pdf/2411.14516v2)

**Authors**: Eden Luzon, Guy Amit, Roy Weiss, Torsten Kraub, Alexandra Dmitrienko, Yisroel Mirsky

**Abstract**: Neural networks are often trained on proprietary datasets, making them attractive attack targets. We present a novel dataset extraction method leveraging an innovative training time backdoor attack, allowing a malicious federated learning server to systematically and deterministically extract complete client training samples through a simple indexing process. Unlike prior techniques, our approach guarantees exact data recovery rather than probabilistic reconstructions or hallucinations, provides precise control over which samples are memorized and how many, and shows high capacity and robustness. Infected models output data samples when they receive a patternbased index trigger, enabling systematic extraction of meaningful patches from each clients local data without disrupting global model utility. To address small model output sizes, we extract patches and then recombined them. The attack requires only a minor modification to the training code that can easily evade detection during client-side verification. Hence, this vulnerability represents a realistic FL supply-chain threat, where a malicious server can distribute modified training code to clients and later recover private data from their updates. Evaluations across classifiers, segmentation models, and large language models demonstrate that thousands of sensitive training samples can be recovered from client models with minimal impact on task performance, and a clients entire dataset can be stolen after multiple FL rounds. For instance, a medical segmentation dataset can be extracted with only a 3 percent utility drop. These findings expose a critical privacy vulnerability in FL systems, emphasizing the need for stronger integrity and transparency in distributed training pipelines.

摘要: 神经网络通常在专有数据集上训练，使其成为有吸引力的攻击目标。我们提出了一种新颖的数据集提取方法，利用创新的训练时间后门攻击，允许恶意联邦学习服务器通过简单的索引过程系统性地、确定性地提取完整的客户端训练样本。与现有技术不同，我们的方法保证了准确的数据恢复，而不是概率重建或幻觉，提供了对记忆哪些样本和数量的精确控制，并显示出高容量和鲁棒性。受感染的模型在接收到基于模式的索引触发器时输出数据样本，从而能够从每个客户端的本地数据中系统地提取有意义的补丁，而不会中断全局模型实用性。为了解决小模型输出大小的问题，我们提取补丁，然后重新组合它们。该攻击只需对训练代码进行轻微修改，即可轻松逃避客户端验证期间的检测。因此，该漏洞代表了现实的FL供应链威胁，恶意服务器可以将修改后的训练代码分发给客户端，然后从其更新中恢复私人数据。跨分类器、分段模型和大型语言模型的评估表明，可以从客户端模型中恢复数千个敏感训练样本，对任务性能的影响最小，并且客户端的整个数据集可能会在多轮FL后被盗。例如，医学分割数据集可以仅以3%的效用下降来提取。这些发现暴露了FL系统中的关键隐私漏洞，强调了分布式培训管道需要更强的完整性和透明度。



