# Latest Large Language Model Attack Papers
**update at 2025-07-10 15:57:38**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. VisualTrap: A Stealthy Backdoor Attack on GUI Agents via Visual Grounding Manipulation**

Visual Trap：通过视觉基础操纵对图形用户界面代理进行秘密后门攻击 cs.CL

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06899v1) [paper-pdf](http://arxiv.org/pdf/2507.06899v1)

**Authors**: Ziang Ye, Yang Zhang, Wentao Shi, Xiaoyu You, Fuli Feng, Tat-Seng Chua

**Abstract**: Graphical User Interface (GUI) agents powered by Large Vision-Language Models (LVLMs) have emerged as a revolutionary approach to automating human-machine interactions, capable of autonomously operating personal devices (e.g., mobile phones) or applications within the device to perform complex real-world tasks in a human-like manner. However, their close integration with personal devices raises significant security concerns, with many threats, including backdoor attacks, remaining largely unexplored. This work reveals that the visual grounding of GUI agent-mapping textual plans to GUI elements-can introduce vulnerabilities, enabling new types of backdoor attacks. With backdoor attack targeting visual grounding, the agent's behavior can be compromised even when given correct task-solving plans. To validate this vulnerability, we propose VisualTrap, a method that can hijack the grounding by misleading the agent to locate textual plans to trigger locations instead of the intended targets. VisualTrap uses the common method of injecting poisoned data for attacks, and does so during the pre-training of visual grounding to ensure practical feasibility of attacking. Empirical results show that VisualTrap can effectively hijack visual grounding with as little as 5% poisoned data and highly stealthy visual triggers (invisible to the human eye); and the attack can be generalized to downstream tasks, even after clean fine-tuning. Moreover, the injected trigger can remain effective across different GUI environments, e.g., being trained on mobile/web and generalizing to desktop environments. These findings underscore the urgent need for further research on backdoor attack risks in GUI agents.

摘要: 由大型视觉语言模型（LVLM）驱动的图形用户界面（GUI）代理已经成为自动化人机交互的革命性方法，能够自主操作个人设备（例如，移动电话）或设备内的应用程序以类似于人类的方式执行复杂的现实世界任务。然而，它们与个人设备的紧密结合引发了重大的安全问题，包括后门攻击在内的许多威胁在很大程度上仍未得到解决。这项工作揭示了GUI代理的视觉基础-将文本计划映射到GUI元素-可以引入漏洞，从而实现新类型的后门攻击。通过针对视觉基础的后门攻击，即使给出了正确的任务解决计划，代理的行为也可能受到损害。为了验证此漏洞，我们提出了Visual Trap，这是一种可以通过误导代理定位文本计划来触发位置而不是预期目标来劫持接地的方法。Visual Trap使用注入有毒数据进行攻击的常见方法，并在视觉基础的预训练期间这样做，以确保攻击的实际可行性。经验结果表明，Visual Trap可以通过低至5%的有毒数据和高度隐蔽的视觉触发器（人眼看不见）有效劫持视觉基础;并且即使经过彻底的微调，攻击也可以推广到下游任务。此外，注入的触发器可以在不同的图形用户界面环境中保持有效，例如，正在接受移动/网络培训并推广到桌面环境。这些发现凸显了对图形用户界面代理后门攻击风险进行进一步研究的迫切性。



## **2. The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover**

LLM基于代理的完全计算机接管攻击的阴暗面 cs.CR

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06850v1) [paper-pdf](http://arxiv.org/pdf/2507.06850v1)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.

摘要: 大型语言模型（LLM）代理和多代理系统的快速采用使自然语言处理和生成具有前所未有的能力。然而，这些系统引入了前所未有的安全漏洞，超出了传统的即时注入攻击的范围。本文首次对LLM代理进行了全面评估，作为攻击载体，这些攻击载体能够通过利用自主实体相互交互和影响的代理人工智能系统内的信任边界来实现完全的计算机接管。我们证明，对手可以利用三种不同的攻击表面--直接提示注入、RAG后门攻击和代理间信任利用--来强迫流行的LLM（包括GPT-4 o、Claude-4和Gemini-2.5）在受害者机器上自主安装和执行恶意软件。我们对17个最先进的LLM的评估揭示了一个令人震惊的漏洞层次结构：虽然41.2%的模型屈服于直接即时注入，但52.9%的模型容易受到RAG后门攻击，并且关键的82.4%可以通过代理间信任利用而受到损害。值得注意的是，我们发现成功抵抗直接恶意命令的LLM将在对等代理请求时执行相同的有效负载，这揭示了当前多代理安全模型中的一个根本缺陷。我们的研究结果表明，只有5.9%的测试模型（1/17）被证明能够抵抗所有攻击载体，其中大多数表现出依赖于上下文的安全行为，从而创建了可利用的盲点。我们的研究结果还强调了提高对LLM安全风险的认识和研究的必要性，这表明网络安全威胁的范式转变，人工智能工具本身成为复杂的攻击载体。



## **3. GuidedBench: Measuring and Mitigating the Evaluation Discrepancies of In-the-wild LLM Jailbreak Methods**

GuidedBench：衡量和减轻野外LLM越狱方法的评估差异 cs.CL

Homepage: https://sproutnan.github.io/AI-Safety_Benchmark/

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2502.16903v2) [paper-pdf](http://arxiv.org/pdf/2502.16903v2)

**Authors**: Ruixuan Huang, Xunguang Wang, Zongjie Li, Daoyuan Wu, Shuai Wang

**Abstract**: Despite the growing interest in jailbreak methods as an effective red-teaming tool for building safe and responsible large language models (LLMs), flawed evaluation system designs have led to significant discrepancies in their effectiveness assessments. We conduct a systematic measurement study based on 37 jailbreak studies since 2022, focusing on both the methods and the evaluation systems they employ. We find that existing evaluation systems lack case-specific criteria, resulting in misleading conclusions about their effectiveness and safety implications. This paper advocates a shift to a more nuanced, case-by-case evaluation paradigm. We introduce GuidedBench, a novel benchmark comprising a curated harmful question dataset, detailed case-by-case evaluation guidelines and an evaluation system integrated with these guidelines -- GuidedEval. Experiments demonstrate that GuidedBench offers more accurate measurements of jailbreak performance, enabling meaningful comparisons across methods and uncovering new insights overlooked in previous evaluations. GuidedEval reduces inter-evaluator variance by at least 76.03\%. Furthermore, we observe that incorporating guidelines can enhance the effectiveness of jailbreak methods themselves, offering new insights into both attack strategies and evaluation paradigms.

摘要: 尽管人们越来越感兴趣越狱方法作为构建安全且负责任的大型语言模型（LLM）的有效红色团队工具，但有缺陷的评估系统设计导致其有效性评估存在显着差异。我们根据2022年以来的37项越狱研究进行了系统性的测量研究，重点关注他们采用的方法和评估体系。我们发现现有的评估系统缺乏针对具体案例的标准，导致对其有效性和安全性影响得出误导性结论。本文主张转向更加细致入微的逐个案例评估范式。我们引入了GuidedBench，这是一个新颖的基准，包括精心策划的有害问题数据集、详细的个案评估指南以及与这些指南集成的评估系统-- GuidedEval。实验表明，GuidedBench提供了更准确的越狱表现测量，能够进行各种方法之间有意义的比较，并发现之前评估中忽视的新见解。GuidedEval将评估者间方差减少至少76.03%。此外，我们观察到，纳入指南可以提高越狱方法本身的有效性，为攻击策略和评估范式提供新的见解。



## **4. Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking**

尾部感知对抗攻击：高效LLM越狱的分布式方法 cs.LG

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.04446v2) [paper-pdf](http://arxiv.org/pdf/2507.04446v2)

**Authors**: Tim Beyer, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.

摘要: 为了保证大规模安全、稳健地部署大型语言模型（LLM），准确评估其对抗稳健性至关重要。现有的对抗性攻击通常针对单点贪婪世代的有害响应，忽视了LLM固有的随机性。在本文中，我们提出了一种新颖的对抗稳健性评估框架，该框架对整个输出分布（包括尾部风险）进行显式建模，为模型大规模稳健性提供更好的估计。通过将攻击过程描述为优化和采样之间的资源分配问题，我们确定了计算最优权衡，并表明将采样集成到现有攻击中可将ASB提高高达48%，并将效率提高高达两个数量级。我们的框架还使我们能够分析不同的攻击算法如何影响输出伤害分布。令人惊讶的是，我们发现大多数优化策略对输出危害影响很小。最后，我们引入了一个基于熵最大化的无数据概念验证目标，以演示我们的尾部感知视角如何实现新的优化目标。总体而言，我们的研究结果强调了尾部感知攻击和评估协议对于准确评估和加强LLM安全性的重要性。



## **5. Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions**

评估和改进大型语言模型的鲁棒性：调查和未来方向 cs.CL

33 pages, 5 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2506.11111v2) [paper-pdf](http://arxiv.org/pdf/2506.11111v2)

**Authors**: Kun Zhang, Le Wu, Kui Yu, Guangyi Lv, Dacao Zhang

**Abstract**: Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (https://github.com/zhangkunzk/Awesome-LLM-Robustness-papers) to support the community.

摘要: 近年来，大型语言模型（LLM）因其理解和生成自然语言的能力而受到了广泛关注。随着快速发展和广泛应用（例如，代理人，联合情报），LLM的稳健性受到了越来越多的关注。作为许多人工智能应用的核心大脑，LLM的稳健性要求模型不仅要生成一致的内容，还要在处理意外的应用场景（例如，有毒提示、有限的噪音域数据、向外分布（OOD）应用程序等）。在这篇调查论文中，我们对LLM的稳健性进行了彻底的审查，旨在提供该领域的全面概念和方法术语并促进社区发展。具体来说，我们首先给出了LLM稳健性的正式定义，并给出了这篇调查论文的收集协议。然后，根据受干扰的输入类型，我们从以下角度组织本次调查：1）对抗稳健性：解决提示被故意操纵的问题，例如噪音提示、长上下文、数据攻击等; 2）OOD稳健性：处理意想不到的现实世界应用场景，例如OOD检测、零镜头传输、幻觉等; 3）稳健性评估：总结用于验证LLM稳健性的新评估数据集、指标和工具。在从各个角度回顾了代表性作品后，我们讨论并强调了该领域未来的机会和研究方向。同时，我们还组织相关工作并提供易于搜索的项目（https：//github.com/zhangkunzk/Awesome-LLM-Robustness-papers）来支持社区。



## **6. Breaking PEFT Limitations: Leveraging Weak-to-Strong Knowledge Transfer for Backdoor Attacks in LLMs**

打破PEFT限制：利用弱到强的知识转移进行LLM中的后门攻击 cs.CR

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2409.17946v4) [paper-pdf](http://arxiv.org/pdf/2409.17946v4)

**Authors**: Shuai Zhao, Leilei Gan, Zhongliang Guo, Xiaobao Wu, Yanhao Jia, Luwei Xiao, Cong-Duy Nguyen, Luu Anh Tuan

**Abstract**: Despite being widely applied due to their exceptional capabilities, Large Language Models (LLMs) have been proven to be vulnerable to backdoor attacks. These attacks introduce targeted vulnerabilities into LLMs by poisoning training samples and full-parameter fine-tuning (FPFT). However, this kind of backdoor attack is limited since they require significant computational resources, especially as the size of LLMs increases. Besides, parameter-efficient fine-tuning (PEFT) offers an alternative but the restricted parameter updating may impede the alignment of triggers with target labels. In this study, we first verify that backdoor attacks with PEFT may encounter challenges in achieving feasible performance. To address these issues and improve the effectiveness of backdoor attacks with PEFT, we propose a novel backdoor attack algorithm from the weak-to-strong based on Feature Alignment-enhanced Knowledge Distillation (FAKD). Specifically, we poison small-scale language models through FPFT to serve as the teacher model. The teacher model then covertly transfers the backdoor to the large-scale student model through FAKD, which employs PEFT. Theoretical analysis reveals that FAKD has the potential to augment the effectiveness of backdoor attacks. We demonstrate the superior performance of FAKD on classification tasks across four language models, four backdoor attack algorithms, and two different architectures of teacher models. Experimental results indicate success rates close to 100% for backdoor attacks targeting PEFT.

摘要: 尽管大型语言模型（LLM）因其卓越的功能而被广泛应用，但已被证明容易受到后门攻击。这些攻击通过毒害训练样本和全参数微调（FPFT）将有针对性的漏洞引入LLM。然而，这种后门攻击是有限的，因为它们需要大量的计算资源，特别是随着LLM规模的增加。此外，参数高效微调（PEFT）提供了一种替代方案，但受限制的参数更新可能会阻碍触发器与目标标签的对齐。在这项研究中，我们首先验证了使用PEFT进行的后门攻击在实现可行的性能时可能会遇到挑战。为了解决这些问题并提高PEFT后门攻击的有效性，我们提出了一种基于特征对齐增强知识提炼（FAKD）的从弱到强的新型后门攻击算法。具体来说，我们通过FPFT毒害小规模语言模型，以充当教师模型。然后，教师模式通过采用PEFT的FAKD秘密地将后门转移到大规模学生模式。理论分析表明，FAKD有潜力增强后门攻击的有效性。我们展示了FAKD在四种语言模型、四种后门攻击算法和两种不同的教师模型架构上的分类任务上的卓越性能。实验结果表明，针对PEFT的后门攻击成功率接近100%。



## **7. Can adversarial attacks by large language models be attributed?**

大型语言模型的对抗攻击可以归因吗？ cs.AI

21 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2411.08003v2) [paper-pdf](http://arxiv.org/pdf/2411.08003v2)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.

摘要: 在对抗环境中（例如网络攻击和虚假信息攻击）对大型语言模型（LLM）的输出进行归因会带来重大挑战，而且其重要性可能会越来越大。我们从理论和实证的角度来处理这个归因问题，借鉴形式语言理论（极限识别）和对不断扩大的LLM生态系统的数据驱动分析。通过将LLM的一组可能输出建模为形式语言，我们分析有限的文本样本是否可以唯一地确定原始模型。我们的结果表明，在模型之间能力重叠的温和假设下，某些类别的LLM从根本上无法仅从其输出中识别。我们描绘了理论可识别性的四种制度：（1）无限一类确定性（离散）LLM语言不可识别（Gold的经典结果来自1967年）;（2）无限类概率LLM也是不可识别的（通过确定性情况的扩展）;（3）有限类确定性LLM是可识别的（与Angluin的泄密标准一致）;以及（4）即使是有限类的概率LLM也可能是不可识别的（我们提供了一个新的反例来建立这个负结果）。作为对这些理论见解的补充，我们量化了近年来给定输出的合理模型起源（假设空间）数量的爆炸式增长。即使在保守的假设下--每个开源模型最多在一个新厕所上进行微调--不同候选模型的数量也大约每0.5年翻一番，并且允许多数据集微调组合可以产生翻倍的时间短至0.28年。这种组合增长，加上所有模型和潜在用户的暴力可能性归因的非凡计算成本，使得详尽的归因在实践中不可行。



## **8. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

论LLM在对抗性攻击中言语信心的稳健性 cs.CL

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06489v1) [paper-pdf](http://arxiv.org/pdf/2507.06489v1)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.

摘要: 大型语言模型（LLM）产生的强大言语信心对于部署LLM至关重要，以确保许多高风险应用程序中人机交互的透明度、信任和安全。在本文中，我们首次对对抗攻击下言语信心的稳健性进行了全面研究。我们引入了一个新颖的框架，通过干扰和基于越狱的方法攻击言语信心分数，并表明这些攻击可能会显着危及言语信心估计并导致答案频繁变化。我们研究了各种提示策略、模型大小和应用领域，揭示了当前的信心激发方法很脆弱，并且常用的防御技术在很大程度上无效或适得其反。我们的研究结果强调了迫切需要设计更强大的机制来表达LLM的信心，因为即使是微妙的语义保留修改也可能导致反应中的误导性信心。



## **9. Bridging AI and Software Security: A Comparative Vulnerability Assessment of LLM Agent Deployment Paradigms**

连接人工智能和软件安全：LLM代理部署范式的比较漏洞评估 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06323v1) [paper-pdf](http://arxiv.org/pdf/2507.06323v1)

**Authors**: Tarek Gasmi, Ramzi Guesmi, Ines Belhadj, Jihene Bennaceur

**Abstract**: Large Language Model (LLM) agents face security vulnerabilities spanning AI-specific and traditional software domains, yet current research addresses these separately. This study bridges this gap through comparative evaluation of Function Calling architecture and Model Context Protocol (MCP) deployment paradigms using a unified threat classification framework. We tested 3,250 attack scenarios across seven language models, evaluating simple, composed, and chained attacks targeting both AI-specific threats (prompt injection) and software vulnerabilities (JSON injection, denial-of-service). Function Calling showed higher overall attack success rates (73.5% vs 62.59% for MCP), with greater system-centric vulnerability while MCP exhibited increased LLM-centric exposure. Attack complexity dramatically amplified effectiveness, with chained attacks achieving 91-96% success rates. Counterintuitively, advanced reasoning models demonstrated higher exploitability despite better threat detection. Results demonstrate that architectural choices fundamentally reshape threat landscapes. This work establishes methodological foundations for cross-domain LLM agent security assessment and provides evidence-based guidance for secure deployment. Code and experimental materials are available at https: // github. com/ theconsciouslab-ai/llm-agent-security.

摘要: 大型语言模型（LLM）代理面临跨越人工智能特定和传统软件领域的安全漏洞，但当前的研究分别解决了这些问题。本研究通过使用统一的威胁分类框架对功能调用架构和模型上下文协议（HCP）部署范式进行比较评估来弥合这一差距。我们测试了七种语言模型中的3，250个攻击场景，评估了针对人工智能特定威胁（提示注入）和软件漏洞（SON注入、拒绝服务）的简单、组合和连锁攻击。功能调用显示出更高的总体攻击成功率（73.5% vs 62.59%），以系统为中心的脆弱性更大，而麦克唐纳则显示出以LLM为中心的暴露率增加。攻击的复杂性极大地提高了有效性，连锁攻击的成功率达到了91-96%。与直觉相反，尽管威胁检测更好，但高级推理模型仍表现出更高的可利用性。结果表明，架构选择从根本上重塑了威胁格局。这项工作为跨域LLM代理安全评估奠定了方法论基础，并为安全部署提供了基于证据的指导。代码和实验材料可在https：// github上获取。com/ theconsciouslab-ai/llm-Agent-secure。



## **10. CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations**

CAVGAN：通过对其内部代表的生成性对抗攻击统一LLM的越狱和辩护 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06043v1) [paper-pdf](http://arxiv.org/pdf/2507.06043v1)

**Authors**: Xiaohu Li, Yunfeng Ning, Zepeng Bao, Mayi Xu, Jianhao Chen, Tieyun Qian

**Abstract**: Security alignment enables the Large Language Model (LLM) to gain the protection against malicious queries, but various jailbreak attack methods reveal the vulnerability of this security mechanism. Previous studies have isolated LLM jailbreak attacks and defenses. We analyze the security protection mechanism of the LLM, and propose a framework that combines attack and defense. Our method is based on the linearly separable property of LLM intermediate layer embedding, as well as the essence of jailbreak attack, which aims to embed harmful problems and transfer them to the safe area. We utilize generative adversarial network (GAN) to learn the security judgment boundary inside the LLM to achieve efficient jailbreak attack and defense. The experimental results indicate that our method achieves an average jailbreak success rate of 88.85\% across three popular LLMs, while the defense success rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%. This not only validates the effectiveness of our approach but also sheds light on the internal security mechanisms of LLMs, offering new insights for enhancing model security The code and data are available at https://github.com/NLPGM/CAVGAN.

摘要: 安全对齐使大型语言模型（LLM）能够获得针对恶意查询的保护，但各种越狱攻击方法揭示了这种安全机制的漏洞。之前的研究已经孤立了LLM越狱攻击和防御。我们分析了LLM的安全保护机制，提出了攻击与防御相结合的框架。我们的方法基于LLM中间层嵌入的线性可分离性质，以及越狱攻击的本质，旨在嵌入有害问题并将其转移到安全区域。我们利用生成对抗网络（GAN）来学习LLM内部的安全判断边界，以实现高效的越狱攻击和防御。实验结果表明，我们的方法在三种流行的LLM中平均越狱成功率为88.85%，而在最先进的越狱数据集上的防御成功率平均达到84.17%。这不仅验证了我们方法的有效性，还揭示了LLM的内部安全机制，为增强模型安全性提供了新的见解。代码和数据可在https://github.com/NLPGM/CAVGAN上获取。



## **11. Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks**

增强LLM水印针对擦除和欺骗攻击的弹性 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06274v1) [paper-pdf](http://arxiv.org/pdf/2507.06274v1)

**Authors**: Huanming Shen, Baizhou Huang, Xiaojun Wan

**Abstract**: Watermarking is a promising defense against the misuse of large language models (LLMs), yet it remains vulnerable to scrubbing and spoofing attacks. This vulnerability stems from an inherent trade-off governed by watermark window size: smaller windows resist scrubbing better but are easier to reverse-engineer, enabling low-cost statistics-based spoofing attacks. This work breaks this trade-off by introducing a novel mechanism, equivalent texture keys, where multiple tokens within a watermark window can independently support the detection. Based on the redundancy, we propose a novel watermark scheme with Sub-vocabulary decomposed Equivalent tExture Key (SEEK). It achieves a Pareto improvement, increasing the resilience against scrubbing attacks without compromising robustness to spoofing. Experiments demonstrate SEEK's superiority over prior method, yielding spoofing robustness gains of +88.2%/+92.3%/+82.0% and scrubbing robustness gains of +10.2%/+6.4%/+24.6% across diverse dataset settings.

摘要: 水印是防止大型语言模型（LLM）滥用的一种有希望的防御方法，但它仍然容易受到擦洗和欺骗攻击。该漏洞源于由水印窗口大小决定的固有权衡：较小的窗口更难抵抗擦洗，但更容易进行反向工程，从而实现低成本的基于统计学的欺骗攻击。这项工作通过引入一种新颖的机制（等效纹理密钥）打破了这种权衡，其中水印窗口内的多个令牌可以独立支持检测。基于冗余度，我们提出了一种新的子词汇分解等效tExture密钥（SEEK）水印方案。它实现了帕累托改进，提高了针对擦除攻击的弹性，而不会损害欺骗的稳健性。实验证明了SEEK相对于现有方法的优越性，在不同的数据集设置中产生了+88.2%/+92.3%/+82.0%的欺骗鲁棒性收益，并产生了+10.2%/+6.4%/+24.6%的擦洗鲁棒性收益。



## **12. ETrace:Event-Driven Vulnerability Detection in Smart Contracts via LLM-Based Trace Analysis**

ETrace：通过基于LLM的跟踪分析在智能合同中进行事件驱动的漏洞检测 cs.CR

4 pages, 1 figure. Submitted to the 16th Asia-Pacific Symposium on  Internetware (Internetware 2025)

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2506.15790v2) [paper-pdf](http://arxiv.org/pdf/2506.15790v2)

**Authors**: Chenyang Peng, Haijun Wang, Yin Wu, Hao Wu, Ming Fan, Yitao Zhao, Ting Liu

**Abstract**: With the advance application of blockchain technology in various fields, ensuring the security and stability of smart contracts has emerged as a critical challenge. Current security analysis methodologies in vulnerability detection can be categorized into static analysis and dynamic analysis methods.However, these existing traditional vulnerability detection methods predominantly rely on analyzing original contract code, not all smart contracts provide accessible code.We present ETrace, a novel event-driven vulnerability detection framework for smart contracts, which uniquely identifies potential vulnerabilities through LLM-powered trace analysis without requiring source code access. By extracting fine-grained event sequences from transaction logs, the framework leverages Large Language Models (LLMs) as adaptive semantic interpreters to reconstruct event analysis through chain-of-thought reasoning. ETrace implements pattern-matching to establish causal links between transaction behavior patterns and known attack behaviors. Furthermore, we validate the effectiveness of ETrace through preliminary experimental results.

摘要: 随着区块链技术在各个领域的深入应用，确保智能合约的安全性和稳定性已成为一项严峻的挑战。当前漏洞检测中的安全分析方法可以分为静态分析和动态分析方法。然而，这些现有的传统漏洞检测方法主要依赖于分析原始合同代码，并非所有智能合同都提供可访问代码。我们提出ETrace，一种新颖的事件驱动的智能合同漏洞检测框架，它通过LLM支持的跟踪分析来唯一地识别潜在漏洞，而无需访问源代码。通过从事务日志中提取细粒度事件序列，该框架利用大型语言模型（LLM）作为自适应语义解释器，通过思想链推理重建事件分析。ETrace实现模式匹配，以建立事务行为模式和已知攻击行为之间的因果联系。此外，我们通过初步实验结果验证了ETrace的有效性。



## **13. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

MEF：一个用于评估黑箱大语言模型脆弱性的能力感知多重加密框架 cs.CL

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2505.23404v3) [paper-pdf](http://arxiv.org/pdf/2505.23404v3)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have revealed significant vulnerabilities in Large Language Models (LLMs), facilitating the evasion of alignment safeguards through increasingly sophisticated prompt manipulations. In this paper, we propose MEF, a capability-aware multi-encryption framework for evaluating vulnerabilities in black-box LLMs. Our key insight is that the effectiveness of jailbreak strategies can be significantly enhanced by tailoring them to the semantic comprehension capabilities of the target model. We present a typology that classifies LLMs into Type I and Type II based on their comprehension levels, and design adaptive attack strategies for each. MEF combines layered semantic mutations and dual-ended encryption techniques, enabling circumvention of input, inference, and output-level defenses. Experimental results demonstrate the superiority of our approach. Remarkably, it achieves a jailbreak success rate of 98.9\% on GPT-4o (29 May 2025 release). Our findings reveal vulnerabilities in current LLMs' alignment defenses.

摘要: 对抗性越狱攻击的最新进展揭示了大型语言模型（LLM）中的显着漏洞，通过日益复杂的提示操纵促进了对对齐保障措施的规避。在本文中，我们提出了MEF，这是一个用于评估黑匣子LLM中漏洞的功能感知多重加密框架。我们的主要见解是，通过根据目标模型的语义理解能力定制越狱策略，可以显着增强它们的有效性。我们提出了一种类型学，根据它们的理解水平将LLM分为I型和II型，并为每种类型设计自适应攻击策略。MEF结合了分层语义突变和双端加密技术，能够规避输入、推理和输出级防御。实验结果证明了我们方法的优越性。值得注意的是，它在GPT-4 o（2025年5月29日发布）上的越狱成功率达到了98.9%。我们的研究结果揭示了当前LLM对齐防御的漏洞。



## **14. Feint and Attack: Attention-Based Strategies for Jailbreaking and Protecting LLMs**

假动作和攻击：越狱和保护LLM的基于注意力的策略 cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2410.16327v2) [paper-pdf](http://arxiv.org/pdf/2410.16327v2)

**Authors**: Rui Pu, Chaozhuo Li, Rui Ha, Zejian Chen, Litian Zhang, Zheng Liu, Lirong Qiu, Zaisheng Ye

**Abstract**: Jailbreak attack can be used to access the vulnerabilities of Large Language Models (LLMs) by inducing LLMs to generate the harmful content. And the most common method of the attack is to construct semantically ambiguous prompts to confuse and mislead the LLMs. To access the security and reveal the intrinsic relation between the input prompt and the output for LLMs, the distribution of attention weight is introduced to analyze the underlying reasons. By using statistical analysis methods, some novel metrics are defined to better describe the distribution of attention weight, such as the Attention Intensity on Sensitive Words (Attn_SensWords), the Attention-based Contextual Dependency Score (Attn_DepScore) and Attention Dispersion Entropy (Attn_Entropy). By leveraging the distinct characteristics of these metrics, the beam search algorithm and inspired by the military strategy "Feint and Attack", an effective jailbreak attack strategy named as Attention-Based Attack (ABA) is proposed. In the ABA, nested attack prompts are employed to divert the attention distribution of the LLMs. In this manner, more harmless parts of the input can be used to attract the attention of the LLMs. In addition, motivated by ABA, an effective defense strategy called as Attention-Based Defense (ABD) is also put forward. Compared with ABA, the ABD can be used to enhance the robustness of LLMs by calibrating the attention distribution of the input prompt. Some comparative experiments have been given to demonstrate the effectiveness of ABA and ABD. Therefore, both ABA and ABD can be used to access the security of the LLMs. The comparative experiment results also give a logical explanation that the distribution of attention weight can bring great influence on the output for LLMs.

摘要: 越狱攻击可用于通过诱导大型语言模型（LLM）生成有害内容来访问大型语言模型（LLM）的漏洞。最常见的攻击方法是构建语义模糊的提示来混淆和误导LLM。为了访问安全性并揭示LLM输入提示和输出之间的内在关系，引入注意力权重的分布来分析潜在原因。通过统计分析方法，定义了一些新颖的指标来更好地描述注意力权重的分布，例如敏感词的注意力强度（Attn_SensWords）、基于注意力的上下文依赖分数（Attn_DepScore）和注意力分散量（Attn_Entropy）。利用这些指标的独特特征、射束搜索算法，并受军事策略“假动作攻击”的启发，提出了一种有效的越狱攻击策略--基于注意力的攻击（BA）。在ABA中，使用嵌套攻击提示来转移LLM的注意力分布。通过这种方式，可以使用输入中更无害的部分来吸引LLM的注意力。此外，在BA的推动下，还提出了一种有效的防御策略--基于注意力的防御（ABD）。与BA相比，ABD可以通过校准输入提示的注意力分布来增强LLM的鲁棒性。已经进行了一些比较实验来证明BA和ABD的有效性。因此，ABA和ABD都可以用于访问LLM的安全性。比较实验的结果也给出了一个逻辑解释，即注意权重的分配会对LLM的产出产生很大影响。



## **15. Disappearing Ink: Obfuscation Breaks N-gram Code Watermarks in Theory and Practice**

消失的墨水：模糊在理论和实践中破解了N-gram代码水印 cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05512v1) [paper-pdf](http://arxiv.org/pdf/2507.05512v1)

**Authors**: Gehao Zhang, Eugene Bagdasarian, Juan Zhai, Shiqing Ma

**Abstract**: Distinguishing AI-generated code from human-written code is becoming crucial for tasks such as authorship attribution, content tracking, and misuse detection. Based on this, N-gram-based watermarking schemes have emerged as prominent, which inject secret watermarks to be detected during the generation.   However, their robustness in code content remains insufficiently evaluated. Most claims rely solely on defenses against simple code transformations or code optimizations as a simulation of attack, creating a questionable sense of robustness. In contrast, more sophisticated schemes already exist in the software engineering world, e.g., code obfuscation, which significantly alters code while preserving functionality. Although obfuscation is commonly used to protect intellectual property or evade software scanners, the robustness of code watermarking techniques against such transformations remains largely unexplored.   In this work, we formally model the code obfuscation and prove the impossibility of N-gram-based watermarking's robustness with only one intuitive and experimentally verified assumption, distribution consistency, satisfied. Given the original false positive rate of the watermarking detection, the ratio that the detector failed on the watermarked code after obfuscation will increase to 1 - fpr.   The experiments have been performed on three SOTA watermarking schemes, two LLMs, two programming languages, four code benchmarks, and four obfuscators. Among them, all watermarking detectors show coin-flipping detection abilities on obfuscated codes (AUROC tightly surrounds 0.5). Among all models, watermarking schemes, and datasets, both programming languages own obfuscators that can achieve attack effects with no detection AUROC higher than 0.6 after the attack. Based on the theoretical and practical observations, we also proposed a potential path of robust code watermarking.

摘要: 区分人工智能生成的代码与人类编写的代码对于作者归属、内容跟踪和滥用检测等任务变得至关重要。基于此，基于N-gram的水印方案逐渐成为主流，它注入秘密水印以在生成过程中检测。   然而，它们在代码内容方面的稳健性仍然没有得到充分评估。大多数主张仅依赖于针对简单代码转换或代码优化的防御作为攻击模拟，从而产生了令人怀疑的稳健性。相比之下，软件工程领域已经存在更复杂的方案，例如，代码混淆，可以显着更改代码，同时保留功能。尽管混淆通常用于保护知识产权或逃避软件扫描器，但代码水印技术针对此类转换的鲁棒性在很大程度上仍未被探索。   在这项工作中，我们对代码混淆进行了正式建模，并证明了基于N-gram的水印的鲁棒性不可能，只需满足一个直观且经过实验验证的假设（分布一致性）即可满足。给定水印检测的原始假阳性率，检测器在混淆后对水印代码失败的比率将增加到1 - fpr。   实验已经进行了三个SOTA水印方案，两个LLM，两种编程语言，四个代码基准，和四个混淆器。其中，所有的水印检测器都表现出对混淆代码的硬币翻转检测能力（AUROC紧紧围绕0.5）。在所有的模型、水印方案和数据集中，两种编程语言都有混淆器，可以实现攻击后无检测AUROC高于0.6的攻击效果。在理论和实践观察的基础上，我们还提出了鲁棒代码水印的潜在路径。



## **16. Response Attack: Exploiting Contextual Priming to Jailbreak Large Language Models**

响应攻击：利用上下文启动来越狱大型语言模型 cs.CL

21 pages, 9 figures. Code and data available at  https://github.com/Dtc7w3PQ/Response-Attack

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05248v1) [paper-pdf](http://arxiv.org/pdf/2507.05248v1)

**Authors**: Ziqi Miao, Lijun Li, Yuan Xiong, Zhenhua Liu, Pengyu Zhu, Jing Shao

**Abstract**: Contextual priming, where earlier stimuli covertly bias later judgments, offers an unexplored attack surface for large language models (LLMs). We uncover a contextual priming vulnerability in which the previous response in the dialogue can steer its subsequent behavior toward policy-violating content. Building on this insight, we propose Response Attack, which uses an auxiliary LLM to generate a mildly harmful response to a paraphrased version of the original malicious query. They are then formatted into the dialogue and followed by a succinct trigger prompt, thereby priming the target model to generate harmful content. Across eight open-source and proprietary LLMs, RA consistently outperforms seven state-of-the-art jailbreak techniques, achieving higher attack success rates. To mitigate this threat, we construct and release a context-aware safety fine-tuning dataset, which significantly reduces the attack success rate while preserving model capabilities. The code and data are available at https://github.com/Dtc7w3PQ/Response-Attack.

摘要: 上下文启动（早期的刺激会秘密地偏向后来的判断）为大型语言模型（LLM）提供了一个尚未探索的攻击表面。我们发现了一个上下文启动漏洞，其中对话中的先前响应可以将其后续行为引导到违反政策的内容上。基于这一见解，我们提出了响应攻击，它使用辅助LLM来对原始恶意查询的重述版本生成轻微有害的响应。然后将它们格式化为对话，然后是简洁的触发提示，从而启动目标模型以生成有害内容。在八种开源和专有LLM中，RA的性能始终优于七种最先进的越狱技术，实现了更高的攻击成功率。为了减轻这种威胁，我们构建并发布了一个上下文感知的安全微调数据集，这可以显着降低攻击成功率，同时保留模型功能。代码和数据可在https://github.com/Dtc7w3PQ/Response-Attack上获取。



## **17. Transfer Attack for Bad and Good: Explain and Boost Adversarial Transferability across Multimodal Large Language Models**

坏与好的传输攻击：解释和增强多模式大型语言模型之间的对抗性传输 cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2405.20090v4) [paper-pdf](http://arxiv.org/pdf/2405.20090v4)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jinhao Duan, Yichi Wang, Jiahang Cao, Qiang Zhang, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate exceptional performance in cross-modality interaction, yet they also suffer adversarial vulnerabilities. In particular, the transferability of adversarial examples remains an ongoing challenge. In this paper, we specifically analyze the manifestation of adversarial transferability among MLLMs and identify the key factors that influence this characteristic. We discover that the transferability of MLLMs exists in cross-LLM scenarios with the same vision encoder and indicate \underline{\textit{two key Factors}} that may influence transferability. We provide two semantic-level data augmentation methods, Adding Image Patch (AIP) and Typography Augment Transferability Method (TATM), which boost the transferability of adversarial examples across MLLMs. To explore the potential impact in the real world, we utilize two tasks that can have both negative and positive societal impacts: \ding{182} Harmful Content Insertion and \ding{183} Information Protection.

摘要: 多模式大型语言模型（MLLM）在跨模式交互中表现出出色的性能，但它们也存在对抗性漏洞。特别是，对抗性例子的可移植性仍然是一个持续的挑战。本文具体分析了MLLM之间对抗性转移性的表现，并确定了影响这一特征的关键因素。我们发现，MLLM的可移植性存在于具有相同视觉编码器的跨LLM场景中，并指出可能影响可移植性的\underline{\textit{两个关键因素}}。我们提供了两种语义级数据增强方法：添加图像补丁（AIP）和印刷增强可移植性方法（TATM），它们增强了对抗性示例跨MLLM的可移植性。为了探索对现实世界的潜在影响，我们利用了两项可能产生负面和积极社会影响的任务：\ding{182}有害内容插入和\ding{183}信息保护。



## **18. The Hidden Threat in Plain Text: Attacking RAG Data Loaders**

纯文本中的隐藏威胁：攻击RAG数据加载器 cs.CR

currently under submission

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.05093v1) [paper-pdf](http://arxiv.org/pdf/2507.05093v1)

**Authors**: Alberto Castagnaro, Umberto Salviati, Mauro Conti, Luca Pajola, Simeone Pizzi

**Abstract**: Large Language Models (LLMs) have transformed human-machine interaction since ChatGPT's 2022 debut, with Retrieval-Augmented Generation (RAG) emerging as a key framework that enhances LLM outputs by integrating external knowledge. However, RAG's reliance on ingesting external documents introduces new vulnerabilities. This paper exposes a critical security gap at the data loading stage, where malicious actors can stealthily corrupt RAG pipelines by exploiting document ingestion.   We propose a taxonomy of 9 knowledge-based poisoning attacks and introduce two novel threat vectors -- Content Obfuscation and Content Injection -- targeting common formats (DOCX, HTML, PDF). Using an automated toolkit implementing 19 stealthy injection techniques, we test five popular data loaders, finding a 74.4% attack success rate across 357 scenarios. We further validate these threats on six end-to-end RAG systems -- including white-box pipelines and black-box services like NotebookLM and OpenAI Assistants -- demonstrating high success rates and critical vulnerabilities that bypass filters and silently compromise output integrity. Our results emphasize the urgent need to secure the document ingestion process in RAG systems against covert content manipulations.

摘要: 自ChatGPT 2022年首次亮相以来，大型语言模型（LLM）已经改变了人机交互，检索增强生成（RAG）成为通过集成外部知识增强LLM输出的关键框架。然而，RAG对吸收外部文档的依赖引入了新的漏洞。本文揭示了数据加载阶段的一个关键安全漏洞，恶意行为者可以通过利用文档摄入来悄悄破坏RAG管道。   我们提出了9种基于知识的中毒攻击的分类法，并引入了两种新型威胁载体--内容混淆和内容注入--针对常见格式（DOCX、HTML、PDF）。我们使用实施19种隐形注入技术的自动化工具包测试了五种流行的数据加载器，发现357个场景中的攻击成功率为74.4%。我们在六个端到端RAG系统上进一步验证了这些威胁，包括NotebookLM和OpenAI Assistant等白盒管道和黑匣子服务，展示了高成功率和绕过过滤器并悄悄损害输出完整性的关键漏洞。我们的结果强调迫切需要保护RAG系统中的文档摄入过程免受秘密内容操纵。



## **19. BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning**

BackFeed：一个高效且标准化的联邦学习后门攻击基准套件 cs.CR

Under review at NeurIPS'25

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04903v1) [paper-pdf](http://arxiv.org/pdf/2507.04903v1)

**Authors**: Thinh Dao, Dung Thuy Nguyen, Khoa D Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) systems are vulnerable to backdoor attacks, where adversaries train their local models on poisoned data and submit poisoned model updates to compromise the global model. Despite numerous proposed attacks and defenses, divergent experimental settings, implementation errors, and unrealistic assumptions hinder fair comparisons and valid conclusions about their effectiveness in real-world scenarios. To address this, we introduce BackFed - a comprehensive benchmark suite designed to standardize, streamline, and reliably evaluate backdoor attacks and defenses in FL, with a focus on practical constraints. Our benchmark offers key advantages through its multi-processing implementation that significantly accelerates experimentation and the modular design that enables seamless integration of new methods via well-defined APIs. With a standardized evaluation pipeline, we envision BackFed as a plug-and-play environment for researchers to comprehensively and reliably evaluate new attacks and defenses. Using BackFed, we conduct large-scale studies of representative backdoor attacks and defenses across both Computer Vision and Natural Language Processing tasks with diverse model architectures and experimental settings. Our experiments critically assess the performance of proposed attacks and defenses, revealing unknown limitations and modes of failures under practical conditions. These empirical insights provide valuable guidance for the development of new methods and for enhancing the security of FL systems. Our framework is openly available at https://github.com/thinh-dao/BackFed.

摘要: 联邦学习（FL）系统很容易受到后门攻击，对手会根据有毒数据训练其本地模型并提交有毒模型更新以损害全局模型。尽管提出了许多攻击和防御，但不同的实验设置、实现错误和不切实际的假设阻碍了公平的比较和关于其在现实世界场景中有效性的有效性的有效结论。为了解决这个问题，我们引入了BackFed --一个全面的基准套件，旨在标准化、简化和可靠地评估FL中的后门攻击和防御，重点关注实际限制。我们的基准测试通过其多处理实施来提供关键优势，可以显着加速实验，并通过定义良好的API实现新方法的无缝集成。通过标准化的评估管道，我们将BackFeed设想为一个即插即用的环境，供研究人员全面可靠地评估新的攻击和防御。使用BackFeed，我们通过不同的模型架构和实验环境对计算机视觉和自然语言处理任务中的代表性后门攻击和防御进行了大规模研究。我们的实验批判性地评估了拟议攻击和防御的性能，揭示了实际条件下未知的限制和失败模式。这些经验见解为新方法的开发和增强FL系统的安全性提供了宝贵的指导。我们的框架可在https://github.com/thinh-dao/BackFed上公开获取。



## **20. Who's the Mole? Modeling and Detecting Intention-Hiding Malicious Agents in LLM-Based Multi-Agent Systems**

鼹鼠是谁？基于LLM的多Agent系统中意图隐藏恶意代理的建模和检测 cs.MA

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04724v1) [paper-pdf](http://arxiv.org/pdf/2507.04724v1)

**Authors**: Yizhe Xie, Congcong Zhu, Xinyue Zhang, Minghao Wang, Chi Liu, Minglu Zhu, Tianqing Zhu

**Abstract**: Multi-agent systems powered by Large Language Models (LLM-MAS) demonstrate remarkable capabilities in collaborative problem-solving. While LLM-MAS exhibit strong collaborative abilities, the security risks in their communication and coordination remain underexplored. We bridge this gap by systematically investigating intention-hiding threats in LLM-MAS, and design four representative attack paradigms that subtly disrupt task completion while maintaining high concealment. These attacks are evaluated in centralized, decentralized, and layered communication structures. Experiments conducted on six benchmark datasets, including MMLU, MMLU-Pro, HumanEval, GSM8K, arithmetic, and biographies, demonstrate that they exhibit strong disruptive capabilities. To identify these threats, we propose a psychology-based detection framework AgentXposed, which combines the HEXACO personality model with the Reid Technique, using progressive questionnaire inquiries and behavior-based monitoring. Experiments conducted on six types of attacks show that our detection framework effectively identifies all types of malicious behaviors. The detection rate for our intention-hiding attacks is slightly lower than that of the two baselines, Incorrect Fact Injection and Dark Traits Injection, demonstrating the effectiveness of intention concealment. Our findings reveal the structural and behavioral risks posed by intention-hiding attacks and offer valuable insights into securing LLM-based multi-agent systems through psychological perspectives, which contributes to a deeper understanding of multi-agent safety. The code and data are available at https://anonymous.4open.science/r/AgentXposed-F814.

摘要: 由大型语言模型（LLM-MAS）支持的多智能体系统在协作解决问题方面表现出了非凡的能力。虽然LLM-MAS表现出强大的协作能力，但其沟通和协调中的安全风险仍然没有得到充分的研究。我们通过系统性调查LLM-MAS中的意图隐藏威胁来弥合这一差距，并设计四种代表性的攻击范式，这些攻击范式微妙地扰乱任务完成，同时保持高度隐蔽性。这些攻击在集中式、分散式和分层的通信结构中进行评估。对六个基准数据集（包括MMLU、MMLU-Pro、HumanEval、GSM 8 K、算术和传记）进行的实验表明，它们表现出强大的破坏能力。为了识别这些威胁，我们提出了一个基于心理的检测框架AgentXposed，它将HEXACO性格模型与Reid技术相结合，使用渐进式问卷调查和基于行为的监控。对六种类型的攻击进行的实验表明，我们的检测框架可以有效识别所有类型的恶意行为。我们的意图隐藏攻击的检测率略低于两个基线，错误事实注入和黑暗特征注入，证明了意图隐藏的有效性。我们的研究结果揭示了意图隐藏攻击带来的结构和行为风险，并为通过心理学角度保护基于LLM的多智能体系统提供了宝贵的见解，这有助于更深入地理解多智能体安全性。代码和数据可在https://anonymous.4open.science/r/AgentXposed-F814上获取。



## **21. Attacker's Noise Can Manipulate Your Audio-based LLM in the Real World**

攻击者的噪音可以在现实世界中操纵您的音频LLM cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.06256v1) [paper-pdf](http://arxiv.org/pdf/2507.06256v1)

**Authors**: Vinu Sankar Sadasivan, Soheil Feizi, Rajiv Mathews, Lun Wang

**Abstract**: This paper investigates the real-world vulnerabilities of audio-based large language models (ALLMs), such as Qwen2-Audio. We first demonstrate that an adversary can craft stealthy audio perturbations to manipulate ALLMs into exhibiting specific targeted behaviors, such as eliciting responses to wake-keywords (e.g., "Hey Qwen"), or triggering harmful behaviors (e.g. "Change my calendar event"). Subsequently, we show that playing adversarial background noise during user interaction with the ALLMs can significantly degrade the response quality. Crucially, our research illustrates the scalability of these attacks to real-world scenarios, impacting other innocent users when these adversarial noises are played through the air. Further, we discuss the transferrability of the attack, and potential defensive measures.

摘要: 本文研究了基于音频的大型语言模型（ALLM）（例如Qwen 2-Audio）的现实世界漏洞。我们首先证明对手可以精心设计隐秘的音频扰动来操纵ALLM表现出特定的有针对性的行为，例如引发对唤醒关键词的响应（例如，“嘿Qwen”），或触发有害行为（例如“更改我的日历事件”）。随后，我们表明，在用户与ALLM交互期间播放对抗性背景噪音会显着降低响应质量。至关重要的是，我们的研究说明了这些攻击对现实世界场景的可扩展性，当这些对抗性噪音通过空气播放时，会影响其他无辜用户。此外，我们还讨论了攻击的转移性以及潜在的防御措施。



## **22. Model Inversion Attacks on Llama 3: Extracting PII from Large Language Models**

对Lama 3的模型倒置攻击：从大型语言模型中提取PRI cs.LG

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04478v1) [paper-pdf](http://arxiv.org/pdf/2507.04478v1)

**Authors**: Sathesh P. Sivashanmugam

**Abstract**: Large language models (LLMs) have transformed natural language processing, but their ability to memorize training data poses significant privacy risks. This paper investigates model inversion attacks on the Llama 3.2 model, a multilingual LLM developed by Meta. By querying the model with carefully crafted prompts, we demonstrate the extraction of personally identifiable information (PII) such as passwords, email addresses, and account numbers. Our findings highlight the vulnerability of even smaller LLMs to privacy attacks and underscore the need for robust defenses. We discuss potential mitigation strategies, including differential privacy and data sanitization, and call for further research into privacy-preserving machine learning techniques.

摘要: 大型语言模型（LLM）已经改变了自然语言处理，但它们记忆训练数据的能力带来了巨大的隐私风险。本文研究了对Llama 3.2模型的模型倒置攻击，Llama 3.2模型是Meta开发的多语言LLM。通过使用精心设计的提示来查询模型，我们演示了如何提取个人可识别信息（PRI），例如密码、电子邮件地址和帐户号码。我们的研究结果强调了更小的LLM容易受到隐私攻击，并强调了强大防御的必要性。我们讨论了潜在的缓解策略，包括差异隐私和数据清理，并呼吁对保护隐私的机器学习技术进行进一步研究。



## **23. Attention Slipping: A Mechanistic Understanding of Jailbreak Attacks and Defenses in LLMs**

注意力流失：对LLM越狱攻击和防御的机械理解 cs.CR

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04365v1) [paper-pdf](http://arxiv.org/pdf/2507.04365v1)

**Authors**: Xiaomeng Hu, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: As large language models (LLMs) become more integral to society and technology, ensuring their safety becomes essential. Jailbreak attacks exploit vulnerabilities to bypass safety guardrails, posing a significant threat. However, the mechanisms enabling these attacks are not well understood. In this paper, we reveal a universal phenomenon that occurs during jailbreak attacks: Attention Slipping. During this phenomenon, the model gradually reduces the attention it allocates to unsafe requests in a user query during the attack process, ultimately causing a jailbreak. We show Attention Slipping is consistent across various jailbreak methods, including gradient-based token replacement, prompt-level template refinement, and in-context learning. Additionally, we evaluate two defenses based on query perturbation, Token Highlighter and SmoothLLM, and find they indirectly mitigate Attention Slipping, with their effectiveness positively correlated with the degree of mitigation achieved. Inspired by this finding, we propose Attention Sharpening, a new defense that directly counters Attention Slipping by sharpening the attention score distribution using temperature scaling. Experiments on four leading LLMs (Gemma2-9B-It, Llama3.1-8B-It, Qwen2.5-7B-It, Mistral-7B-It v0.2) show that our method effectively resists various jailbreak attacks while maintaining performance on benign tasks on AlpacaEval. Importantly, Attention Sharpening introduces no additional computational or memory overhead, making it an efficient and practical solution for real-world deployment.

摘要: 随着大型语言模型（LLM）变得越来越重要，确保其安全性变得至关重要。越狱攻击利用漏洞绕过安全护栏，构成重大威胁。然而，导致这些攻击的机制还没有得到很好的了解。在本文中，我们揭示了越狱袭击期间发生的一种普遍现象：注意力流失。在这种现象期间，该模型在攻击过程中逐渐减少对用户查询中不安全请求的关注，最终导致越狱。我们表明，注意力滑动在各种越狱方法中是一致的，包括基于梯度的令牌替换、预算级模板细化和上下文学习。此外，我们评估了基于查询扰动的两种防御措施：Token Highlighter和SmoothLLM，发现它们间接缓解了注意力滑动，其有效性与所实现的缓解程度正相关。受这一发现的启发，我们提出了注意力尖锐化，这是一种新的防御方法，通过使用温度缩放来尖锐注意力分数分布来直接对抗注意力滑动。对四种领先的LLM（Gemma 2 - 9 B-It、Llama3.1-8B-It、Qwen 2.5 - 7 B-It、Mistral-7 B-It v0.2）的实验表明，我们的方法可以有效抵抗各种越狱攻击，同时保持AlpacaEval上良性任务的性能。重要的是，注意力尖锐不会引入额外的计算或内存负担，使其成为现实世界部署的高效实用解决方案。



## **24. Hijacking JARVIS: Benchmarking Mobile GUI Agents against Unprivileged Third Parties**

劫持JARRIS：针对无特权第三方对移动图形用户界面代理进行基准测试 cs.CR

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04227v1) [paper-pdf](http://arxiv.org/pdf/2507.04227v1)

**Authors**: Guohong Liu, Jialei Ye, Jiacheng Liu, Yuanchun Li, Wei Liu, Pengzhi Gao, Jian Luan, Yunxin Liu

**Abstract**: Mobile GUI agents are designed to autonomously execute diverse device-control tasks by interpreting and interacting with mobile screens. Despite notable advancements, their resilience in real-world scenarios where screen content may be partially manipulated by untrustworthy third parties remains largely unexplored. Owing to their black-box and autonomous nature, these agents are vulnerable to manipulations that could compromise user devices. In this work, we present the first systematic investigation into the vulnerabilities of mobile GUI agents. We introduce a scalable attack simulation framework AgentHazard, which enables flexible and targeted modifications of screen content within existing applications. Leveraging this framework, we develop a comprehensive benchmark suite comprising both a dynamic task execution environment and a static dataset of vision-language-action tuples, totaling over 3,000 attack scenarios. The dynamic environment encompasses 58 reproducible tasks in an emulator with various types of hazardous UI content, while the static dataset is constructed from 210 screenshots collected from 14 popular commercial apps. Importantly, our content modifications are designed to be feasible for unprivileged third parties. We evaluate 7 widely-used mobile GUI agents and 5 common backbone models using our benchmark. Our findings reveal that all examined agents are significantly influenced by misleading third-party content (with an average misleading rate of 28.8% in human-crafted attack scenarios) and that their vulnerabilities are closely linked to the employed perception modalities and backbone LLMs. Furthermore, we assess training-based mitigation strategies, highlighting both the challenges and opportunities for enhancing the robustness of mobile GUI agents. Our code and data will be released at https://agenthazard.github.io.

摘要: 移动图形用户界面代理旨在通过解释移动屏幕和与移动屏幕交互来自主执行各种设备控制任务。尽管取得了显着的进步，但它们在屏幕内容可能被不值得信赖的第三方部分操纵的现实世界场景中的弹性在很大程度上仍然没有被探索。由于它们的黑匣子和自治性质，这些代理很容易受到可能危及用户设备的操纵。在这项工作中，我们对移动图形用户界面代理的漏洞进行了首次系统性调查。我们引入了一个可扩展的攻击模拟框架AgentHazard，它可以灵活且有针对性地修改现有应用程序中的屏幕内容。利用这个框架，我们开发了一个全面的基准测试套件，其中包括动态任务执行环境和视觉-语言-动作二元组的静态数据集，总共超过3，000种攻击场景。动态环境包含具有各种类型危险UI内容的模拟器中的58个可重复任务，而静态数据集是根据从14个流行商业应用程序收集的210个屏幕截图构建的。重要的是，我们的内容修改旨在对无特权的第三方可行。我们评估7广泛使用的移动GUI代理和5个常见的骨干模型，使用我们的基准。我们的研究结果表明，所有受检查的代理都受到误导性第三方内容的显著影响（在人为攻击场景中，平均误导率为28.8%），并且他们的漏洞与所采用的感知模式和骨干LLM密切相关。此外，我们评估基于培训的缓解策略，突出的挑战和机遇，以提高移动GUI代理的鲁棒性。我们的代码和数据将在https://agenthazard.github.io上发布。



## **25. Can Large Language Models Automate the Refinement of Cellular Network Specifications?**

大型语言模型能否自动细化蜂窝网络规范？ cs.CR

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04214v1) [paper-pdf](http://arxiv.org/pdf/2507.04214v1)

**Authors**: Jianshuo Dong, Tianyi Zhang, Feng Yan, Yuanjie Li, Hewu Li, Han Qiu

**Abstract**: Cellular networks serve billions of users globally, yet concerns about reliability and security persist due to weaknesses in 3GPP standards. However, traditional analysis methods, including manual inspection and automated tools, struggle with increasingly expanding cellular network specifications. This paper investigates the feasibility of Large Language Models (LLMs) for automated cellular network specification refinement. To advance it, we leverage 200,000+ approved 3GPP Change Requests (CRs) that document specification revisions, constructing a valuable dataset for domain tasks. We introduce CR-eval, a principled evaluation framework, and benchmark 16 state-of-the-art LLMs, demonstrating that top models can discover security-related weaknesses in over 127 out of 200 test cases within five trials. To bridge potential gaps, we explore LLM specialization techniques, including fine-tuning an 8B model to match or surpass advanced LLMs like GPT-4o and DeepSeek-R1. Evaluations on 30 cellular attacks identify open challenges for achieving full automation. These findings confirm that LLMs can automate the refinement of cellular network specifications and provide valuable insights to guide future research in this direction.

摘要: 蜂窝网络为全球数十亿用户提供服务，但由于3GPP标准的弱点，人们对可靠性和安全性的担忧仍然存在。然而，包括手动检查和自动化工具在内的传统分析方法难以应对日益扩大的蜂窝网络规范。本文研究了大型语言模型（LLM）用于自动蜂窝网络规范细化的可行性。为了推进这一进程，我们利用了200，000多个已批准的3GPP变更请求（CR），这些请求记录了规范修订，为领域任务构建了有价值的数据集。我们介绍了CR-eval，一个原则性的评估框架，并对16个最先进的LLM进行了基准测试，证明顶级模型可以在五次试验中发现200个测试用例中的127个与安全相关的弱点。为了弥合潜在的差距，我们探索LLM专业化技术，包括微调8B模型以匹配或超越GPT-4 o和DeepSeek-R1等高级LLM。对30种蜂窝攻击的评估确定了实现完全自动化的挑战。这些发现证实了LLM可以自动细化蜂窝网络规范，并提供有价值的见解，以指导未来在这一方向的研究。



## **26. False Alarms, Real Damage: Adversarial Attacks Using LLM-based Models on Text-based Cyber Threat Intelligence Systems**

虚假警报，真实损害：在基于文本的网络威胁情报系统上使用基于LLM的模型进行对抗性攻击 cs.CR

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.06252v1) [paper-pdf](http://arxiv.org/pdf/2507.06252v1)

**Authors**: Samaneh Shafee, Alysson Bessani, Pedro M. Ferreira

**Abstract**: Cyber Threat Intelligence (CTI) has emerged as a vital complementary approach that operates in the early phases of the cyber threat lifecycle. CTI involves collecting, processing, and analyzing threat data to provide a more accurate and rapid understanding of cyber threats. Due to the large volume of data, automation through Machine Learning (ML) and Natural Language Processing (NLP) models is essential for effective CTI extraction. These automated systems leverage Open Source Intelligence (OSINT) from sources like social networks, forums, and blogs to identify Indicators of Compromise (IoCs). Although prior research has focused on adversarial attacks on specific ML models, this study expands the scope by investigating vulnerabilities within various components of the entire CTI pipeline and their susceptibility to adversarial attacks. These vulnerabilities arise because they ingest textual inputs from various open sources, including real and potentially fake content. We analyse three types of attacks against CTI pipelines, including evasion, flooding, and poisoning, and assess their impact on the system's information selection capabilities. Specifically, on fake text generation, the work demonstrates how adversarial text generation techniques can create fake cybersecurity and cybersecurity-like text that misleads classifiers, degrades performance, and disrupts system functionality. The focus is primarily on the evasion attack, as it precedes and enables flooding and poisoning attacks within the CTI pipeline.

摘要: 网络威胁情报（RTI）已成为一种重要的补充方法，在网络威胁生命周期的早期阶段运作。RTI涉及收集、处理和分析威胁数据，以更准确、更快速地了解网络威胁。由于数据量大，通过机器学习（ML）和自然语言处理（NLP）模型实现自动化对于有效的RTI提取至关重要。这些自动化系统利用来自社交网络、论坛和博客等来源的开源情报（Osint）来识别妥协指标（IoCs）。尽管之前的研究重点是对特定ML模型的对抗攻击，但这项研究通过调查整个RTI管道的各个组件内的漏洞及其对对抗攻击的易感性来扩大了范围。这些漏洞的出现是因为它们从各种开源获取文本输入，包括真实和潜在虚假内容。我们分析了针对RTI管道的三种类型的攻击，包括规避、洪水和中毒，并评估它们对系统信息选择能力的影响。具体来说，在虚假文本生成方面，该工作展示了对抗性文本生成技术如何创建虚假网络安全和类似网络安全的文本，从而误导分类器、降低性能并扰乱系统功能。重点主要是规避攻击，因为它先于RTI管道内的洪水和中毒攻击。



## **27. Membership Inference Attacks on Large-Scale Models: A Survey**

对大规模模型的成员推断攻击：一项调查 cs.LG

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2503.19338v2) [paper-pdf](http://arxiv.org/pdf/2503.19338v2)

**Authors**: Hengyu Wu, Yang Cao

**Abstract**: The adoption of the Large Language Model (LLM) has accelerated dramatically since ChatGPT from OpenAI went online in November 2022. Recent advances in Large Multimodal Models (LMMs), which process diverse data types and enable interaction through various channels, have expanded beyond the text-to-text limitations of early LLMs, attracting significant and concurrent attention from both researchers and industry. While LLMs and LMMs are starting to spread widely, concerns about their privacy risks are increasing as well. Membership Inference Attacks (MIAs) are techniques used to determine whether a particular data point was part of a model's training set, which is a key metric for assessing the privacy vulnerabilities of machine learning models. Hu et al. show that various machine learning algorithms are vulnerable to MIA. Despite extensive studies on MIAs in classic models, there remains a lack of systematic surveys addressing their effectiveness and limitations in advanced large-scale models like LLMs and LMMs. In this paper, we systematically reviewed recent studies of MIA against LLMs and LMMs. We analyzed and categorized each attack based on its methodology, scenario, and targeted model, and we discussed the limitations of existing research. In addition to examining attacks on pre-training and fine-tuning stages, we also explore MIAs that target other development pipelines, including Retrieval-Augmented Generation (RAG) and the model alignment process. Based on the survey, we provide suggestions for future studies to improve the robustness of MIA in large-scale AI models.

摘要: 自OpenAI的ChatGPT于2022年11月上线以来，大型语言模型（LLM）的采用急剧加速。大型多模式模型（LSYS）的最新进展处理不同的数据类型并通过各种渠道实现交互，已经超越了早期LLM的文本到文本限制，吸引了研究人员和行业的高度关注。虽然LLM和LSYS开始广泛传播，但对其隐私风险的担忧也在增加。成员资格推理攻击（MIA）是用于确定特定数据点是否是模型训练集的一部分的技术，这是评估机器学习模型隐私漏洞的关键指标。Hu等人表明，各种机器学习算法都容易受到MIA的影响。尽管对经典模型中的MIA进行了广泛的研究，但仍然缺乏系统性的调查来解决它们在LLM和LSYS等先进大规模模型中的有效性和局限性。在本文中，我们系统地回顾了最近关于MIA对抗LLM和LSYS的研究。我们根据方法论、场景和目标模型分析和分类了每种攻击，并讨论了现有研究的局限性。除了检查对预训练和微调阶段的攻击外，我们还探索针对其他开发管道的MIA，包括检索增强生成（RAG）和模型对齐过程。根据调查，我们为未来的研究提供建议，以提高MIA在大规模人工智能模型中的稳健性。



## **28. A Survey on Proactive Defense Strategies Against Misinformation in Large Language Models**

大型语言模型中针对错误信息的主动防御策略研究 cs.IR

Accepted by ACL 2025 Findings

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.05288v1) [paper-pdf](http://arxiv.org/pdf/2507.05288v1)

**Authors**: Shuliang Liu, Hongyi Liu, Aiwei Liu, Bingchen Duan, Qi Zheng, Yibo Yan, He Geng, Peijie Jiang, Jia Liu, Xuming Hu

**Abstract**: The widespread deployment of large language models (LLMs) across critical domains has amplified the societal risks posed by algorithmically generated misinformation. Unlike traditional false content, LLM-generated misinformation can be self-reinforcing, highly plausible, and capable of rapid propagation across multiple languages, which traditional detection methods fail to mitigate effectively. This paper introduces a proactive defense paradigm, shifting from passive post hoc detection to anticipatory mitigation strategies. We propose a Three Pillars framework: (1) Knowledge Credibility, fortifying the integrity of training and deployed data; (2) Inference Reliability, embedding self-corrective mechanisms during reasoning; and (3) Input Robustness, enhancing the resilience of model interfaces against adversarial attacks. Through a comprehensive survey of existing techniques and a comparative meta-analysis, we demonstrate that proactive defense strategies offer up to 63\% improvement over conventional methods in misinformation prevention, despite non-trivial computational overhead and generalization challenges. We argue that future research should focus on co-designing robust knowledge foundations, reasoning certification, and attack-resistant interfaces to ensure LLMs can effectively counter misinformation across varied domains.

摘要: 大型语言模型（LLM）在关键领域的广泛部署放大了算法生成的错误信息带来的社会风险。与传统的虚假内容不同，LLM生成的错误信息可以自我强化、高度可信，并且能够在多种语言中快速传播，而传统检测方法无法有效缓解这一点。本文引入了一种主动防御范式，从被动事后检测转向预期缓解策略。我们提出了一个三柱框架：（1）知识可信度，加强训练和部署数据的完整性;（2）推理可靠性，在推理过程中嵌入自我纠正机制;（3）输入鲁棒性，增强模型接口针对对抗性攻击的弹性。通过对现有技术的全面调查和比较荟萃分析，我们证明，尽管存在重要的计算费用和概括性挑战，但主动防御策略在错误信息预防方面比传统方法提供了高达63%的改进。我们认为，未来的研究应该重点关注共同设计强大的知识基础、推理认证和抗攻击界面，以确保LLM能够有效地对抗各个领域的错误信息。



## **29. We Urgently Need Privilege Management in MCP: A Measurement of API Usage in MCP Ecosystems**

我们迫切需要MCP中的植物管理：MCP生态系统中API使用的测量 cs.CR

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.06250v1) [paper-pdf](http://arxiv.org/pdf/2507.06250v1)

**Authors**: Zhihao Li, Kun Li, Boyang Ma, Minghui Xu, Yue Zhang, Xiuzhen Cheng

**Abstract**: The Model Context Protocol (MCP) has emerged as a widely adopted mechanism for connecting large language models to external tools and resources. While MCP promises seamless extensibility and rich integrations, it also introduces a substantially expanded attack surface: any plugin can inherit broad system privileges with minimal isolation or oversight. In this work, we conduct the first large-scale empirical analysis of MCP security risks. We develop an automated static analysis framework and systematically examine 2,562 real-world MCP applications spanning 23 functional categories. Our measurements reveal that network and system resource APIs dominate usage patterns, affecting 1,438 and 1,237 servers respectively, while file and memory resources are less frequent but still significant. We find that Developer Tools and API Development plugins are the most API-intensive, and that less popular plugins often contain disproportionately high-risk operations. Through concrete case studies, we demonstrate how insufficient privilege separation enables privilege escalation, misinformation propagation, and data tampering. Based on these findings, we propose a detailed taxonomy of MCP resource access, quantify security-relevant API usage, and identify open challenges for building safer MCP ecosystems, including dynamic permission models and automated trust assessment.

摘要: 模型上下文协议（HCP）已成为一种广泛采用的将大型语言模型连接到外部工具和资源的机制。虽然HCP承诺无缝的可扩展性和丰富的集成，但它也引入了大幅扩展的攻击面：任何插件都可以在最小的隔离或监督的情况下继承广泛的系统特权。在这项工作中，我们对LCP安全风险进行了首次大规模实证分析。我们开发了一个自动化静态分析框架，并系统性地检查了涵盖23个功能类别的2，562个现实世界的LCP应用程序。我们的测量显示，网络和系统资源API主导了使用模式，分别影响1，438和1，237台服务器，而文件和内存资源的频率较低，但仍然很重要。我们发现开发人员工具和API开发插件是API最密集的，而不太受欢迎的插件通常包含不成比例的高风险操作。通过具体的案例研究，我们展示了权限分离不足如何导致权限升级、错误信息传播和数据篡改。基于这些研究结果，我们提出了一个详细的LCP资源访问分类法，量化与安全相关的API使用情况，并确定构建更安全的LCP生态系统的公开挑战，包括动态许可模型和自动信任评估。



## **30. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

探索LLM中的潜在子空间以实现人工智能安全：识别和操纵敌对状态 cs.LG

4 figures

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2503.09066v2) [paper-pdf](http://arxiv.org/pdf/2503.09066v2)

**Authors**: Xin Wei Chia, Swee Liang Wong, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.

摘要: 大型语言模型（LLM）在各种任务中表现出了非凡的能力，但它们仍然容易受到对抗操纵的影响，例如通过提示注入攻击进行越狱。这些攻击绕过安全机制来生成受限制或有害内容。在这项研究中，我们通过从LLM中提取隐藏激活来研究安全和越狱状态的潜在子空间。受神经科学中吸引子动力学的启发，我们假设LLM激活会进入半稳定状态，可以识别和扰动这些状态以引发状态转变。使用降维技术，我们预测安全和越狱反应的激活，以揭示低维空间中的潜在子空间。然后，我们推导出一个扰动载体，当将其应用于安全表示时，会将模型转向越狱状态。我们的结果表明，这种因果干预会在提示子集中导致具有统计学意义的越狱反应。接下来，我们探讨了这些扰动如何在模型的层中传播，测试诱导的状态变化是保持局部化还是在整个网络中级联。我们的研究结果表明，有针对性的扰动会导致激活和模型响应的明显变化。我们的方法为潜在的主动防御铺平了道路，从传统的基于护栏的方法转向先发制人的、模型不可知的技术，可以在表示层面中和对抗状态。



## **31. Blackbox Dataset Inference for LLM**

LLM的黑匣子数据集推理 cs.CR

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03619v1) [paper-pdf](http://arxiv.org/pdf/2507.03619v1)

**Authors**: Ruikai Zhou, Kang Yang, Xun Chen, Wendy Hui Wang, Guanhong Tao, Jun Xu

**Abstract**: Today, the training of large language models (LLMs) can involve personally identifiable information and copyrighted material, incurring dataset misuse. To mitigate the problem of dataset misuse, this paper explores \textit{dataset inference}, which aims to detect if a suspect model $\mathcal{M}$ used a victim dataset $\mathcal{D}$ in training. Previous research tackles dataset inference by aggregating results of membership inference attacks (MIAs) -- methods to determine whether individual samples are a part of the training dataset. However, restricted by the low accuracy of MIAs, previous research mandates grey-box access to $\mathcal{M}$ to get intermediate outputs (probabilities, loss, perplexity, etc.) for obtaining satisfactory results. This leads to reduced practicality, as LLMs, especially those deployed for profits, have limited incentives to return the intermediate outputs.   In this paper, we propose a new method of dataset inference with only black-box access to the target model (i.e., assuming only the text-based responses of the target model are available). Our method is enabled by two sets of locally built reference models, one set involving $\mathcal{D}$ in training and the other not. By measuring which set of reference model $\mathcal{M}$ is closer to, we determine if $\mathcal{M}$ used $\mathcal{D}$ for training. Evaluations of real-world LLMs in the wild show that our method offers high accuracy in all settings and presents robustness against bypassing attempts.

摘要: 如今，大型语言模型（LLM）的训练可能涉及个人可识别信息和受版权保护的材料，从而导致数据集滥用。为了缓解数据集滥用的问题，本文探讨了\textit{dataset initiation}，其目的是检测可疑模型$\mathCal{M}$是否在训练中使用了受害者数据集$\mathCal{D}$。之前的研究通过聚集隶属度推理攻击（MIA）的结果来解决数据集推理--MIA是确定单个样本是否是训练数据集一部分的方法。然而，受MIA准确性低的限制，之前的研究要求灰箱访问$\mathCal{M}$以获得中间输出（概率、损失、困惑度等）以获得满意的结果。这导致实用性降低，因为LLM，尤其是那些为利润而部署的LLM，返回中间产出的动力有限。   在本文中，我们提出了一种新的数据集推理方法，仅通过黑匣子访问目标模型（即，假设只有目标模型的基于文本的响应可用）。我们的方法由两组本地构建的参考模型来支持，一组在训练中涉及$\mathCal{D}$，另一组不涉及。通过测量$\mathCal{M}$更接近哪一组参考模型，我们确定$\mathCal{M}$是否使用$\mathCal{D}$进行训练。对现实世界LLM的野外评估表明，我们的方法在所有设置中都提供了高准确性，并且具有针对绕过尝试的鲁棒性。



## **32. Visual Contextual Attack: Jailbreaking MLLMs with Image-Driven Context Injection**

视觉上下文攻击：利用图像驱动上下文注入越狱MLLM cs.CV

16 pages

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02844v1) [paper-pdf](http://arxiv.org/pdf/2507.02844v1)

**Authors**: Ziqi Miao, Yi Ding, Lijun Li, Jing Shao

**Abstract**: With the emergence of strong visual-language capabilities, multimodal large language models (MLLMs) have demonstrated tremendous potential for real-world applications. However, the security vulnerabilities exhibited by the visual modality pose significant challenges to deploying such models in open-world environments. Recent studies have successfully induced harmful responses from target MLLMs by encoding harmful textual semantics directly into visual inputs. However, in these approaches, the visual modality primarily serves as a trigger for unsafe behavior, often exhibiting semantic ambiguity and lacking grounding in realistic scenarios. In this work, we define a novel setting: visual-centric jailbreak, where visual information serves as a necessary component in constructing a complete and realistic jailbreak context. Building on this setting, we propose the VisCo (Visual Contextual) Attack. VisCo fabricates contextual dialogue using four distinct visual-focused strategies, dynamically generating auxiliary images when necessary to construct a visual-centric jailbreak scenario. To maximize attack effectiveness, it incorporates automatic toxicity obfuscation and semantic refinement to produce a final attack prompt that reliably triggers harmful responses from the target black-box MLLMs. Specifically, VisCo achieves a toxicity score of 4.78 and an Attack Success Rate (ASR) of 85% on MM-SafetyBench against GPT-4o, significantly outperforming the baseline, which performs a toxicity score of 2.48 and an ASR of 22.2%. The code is available at https://github.com/Dtc7w3PQ/Visco-Attack.

摘要: 随着强大的视觉语言能力的出现，多模式大型语言模型（MLLM）在现实世界应用中展示了巨大的潜力。然而，视觉模式所表现出的安全漏洞对在开放世界环境中部署此类模型构成了重大挑战。最近的研究通过将有害的文本语义直接编码到视觉输入中，成功地诱导了目标MLLM的有害反应。然而，在这些方法中，视觉形态主要充当不安全行为的触发器，通常表现出语义模糊性并且在现实场景中缺乏基础。在这项工作中，我们定义了一种新颖的环境：以视觉为中心的越狱，其中视觉信息是构建完整而现实的越狱背景的必要组成部分。在此设置的基础上，我们提出了VisCo（视觉上下文）攻击。VisCo使用四种不同的以视觉为中心的策略构建上下文对话，在必要时动态生成辅助图像以构建以视觉为中心的越狱场景。为了最大限度地提高攻击效果，它结合了自动毒性混淆和语义细化，以产生最终的攻击提示，从而可靠地触发目标黑匣子MLLM的有害响应。具体而言，VisCo在MM-SafetyBench上针对GPT-4 o实现了4.78的毒性评分和85%的攻击成功率（ASR），显著优于基线，其毒性评分为2.48，ASR为22.2%。该代码可在https://github.com/Dtc7w3PQ/Visco-Attack上获取。



## **33. Is Reasoning All You Need? Probing Bias in the Age of Reasoning Language Models**

你需要的就是推理吗？探索推理语言模型时代的偏见 cs.CL

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02799v1) [paper-pdf](http://arxiv.org/pdf/2507.02799v1)

**Authors**: Riccardo Cantini, Nicola Gabriele, Alessio Orsino, Domenico Talia

**Abstract**: Reasoning Language Models (RLMs) have gained traction for their ability to perform complex, multi-step reasoning tasks through mechanisms such as Chain-of-Thought (CoT) prompting or fine-tuned reasoning traces. While these capabilities promise improved reliability, their impact on robustness to social biases remains unclear. In this work, we leverage the CLEAR-Bias benchmark, originally designed for Large Language Models (LLMs), to investigate the adversarial robustness of RLMs to bias elicitation. We systematically evaluate state-of-the-art RLMs across diverse sociocultural dimensions, using an LLM-as-a-judge approach for automated safety scoring and leveraging jailbreak techniques to assess the strength of built-in safety mechanisms. Our evaluation addresses three key questions: (i) how the introduction of reasoning capabilities affects model fairness and robustness; (ii) whether models fine-tuned for reasoning exhibit greater safety than those relying on CoT prompting at inference time; and (iii) how the success rate of jailbreak attacks targeting bias elicitation varies with the reasoning mechanisms employed. Our findings reveal a nuanced relationship between reasoning capabilities and bias safety. Surprisingly, models with explicit reasoning, whether via CoT prompting or fine-tuned reasoning traces, are generally more vulnerable to bias elicitation than base models without such mechanisms, suggesting reasoning may unintentionally open new pathways for stereotype reinforcement. Reasoning-enabled models appear somewhat safer than those relying on CoT prompting, which are particularly prone to contextual reframing attacks through storytelling prompts, fictional personas, or reward-shaped instructions. These results challenge the assumption that reasoning inherently improves robustness and underscore the need for more bias-aware approaches to reasoning design.

摘要: 推理语言模型（RLM）通过诸如思想链（CoT）提示或微调推理轨迹等机制来执行复杂的多步推理任务的能力已经获得了关注。虽然这些功能有望提高可靠性，但它们对社会偏见鲁棒性的影响仍不清楚。在这项工作中，我们利用CLEAR-Bias基准，最初是为大型语言模型（LLM）设计的，来研究RLM对偏见启发的对抗鲁棒性。我们在不同的社会文化维度上系统地评估最先进的RLM，使用LLM作为自动安全评分的评判方法，并利用越狱技术来评估内置安全机制的强度。我们的评估解决了三个关键问题：（i）推理能力的引入如何影响模型的公平性和稳健性;（ii）为推理进行微调的模型是否比在推理时依赖CoT提示的模型表现出更大的安全性;（iii）针对偏见引发的越狱攻击的成功率如何随着所采用的推理机制而变化。我们的研究结果揭示了推理能力和偏见安全性之间的微妙关系。令人惊讶的是，具有显式推理的模型，无论是通过CoT提示还是微调推理痕迹，通常比没有此类机制的基本模型更容易受到偏见引发，这表明推理可能会无意中为刻板印象强化开辟新的途径。支持推理的模型似乎比依赖CoT提示的模型更安全，后者特别容易受到通过讲故事提示、虚构人物角色或奖励形状指令的上下文重组攻击。这些结果挑战了推理本质上可以提高稳健性的假设，并强调了对推理设计的更多偏差感知方法的需求。



## **34. StructTransform: A Scalable Attack Surface for Safety-Aligned Large Language Models**

StructChange：安全一致的大型语言模型的可扩展攻击表面 cs.LG

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2502.11853v2) [paper-pdf](http://arxiv.org/pdf/2502.11853v2)

**Authors**: Shehel Yoosuf, Temoor Ali, Ahmed Lekssays, Mashael AlSabah, Issa Khalil

**Abstract**: In this work, we present a series of structure transformation attacks on LLM alignment, where we encode natural language intent using diverse syntax spaces, ranging from simple structure formats and basic query languages (e.g., SQL) to new novel spaces and syntaxes created entirely by LLMs. Our extensive evaluation shows that our simplest attacks can achieve close to a 90% success rate, even on strict LLMs (such as Claude 3.5 Sonnet) using SOTA alignment mechanisms. We improve the attack performance further by using an adaptive scheme that combines structure transformations along with existing content transformations, resulting in over 96% ASR with 0% refusals.   To generalize our attacks, we explore numerous structure formats, including syntaxes purely generated by LLMs. Our results indicate that such novel syntaxes are easy to generate and result in a high ASR, suggesting that defending against our attacks is not a straightforward process. Finally, we develop a benchmark and evaluate existing safety-alignment defenses against it, showing that most of them fail with 100% ASR. Our results show that existing safety alignment mostly relies on token-level patterns without recognizing harmful concepts, highlighting and motivating the need for serious research efforts in this direction. As a case study, we demonstrate how attackers can use our attack to easily generate a sample malware and a corpus of fraudulent SMS messages, which perform well in bypassing detection.

摘要: 在这项工作中，我们提出了一系列对LLM对齐的结构转换攻击，其中我们使用不同的语法空间对自然语言意图进行编码，从简单的结构格式到基本的查询语言（例如，SQL）到完全由LLM创建的新空间和语法。我们广泛的评估表明，我们最简单的攻击可以达到接近90%的成功率，即使是在严格的LLM（如Claude 3.5 Sonnet）上使用SOTA对齐机制。我们进一步提高攻击性能，通过使用自适应方案，结合结构转换与现有的内容转换，导致超过96%的ASR与0%的拒绝。   为了概括我们的攻击，我们探索了多种结构格式，包括纯粹由LLM生成的语法。我们的结果表明，这种新颖的语法很容易生成并导致高的ASB，这表明防御我们的攻击并不是一个简单的过程。最后，我们开发了一个基准并评估了现有的安全对齐防御，表明其中大多数都在100%的ASC下失败。我们的结果表明，现有的安全调整主要依赖于代币级模式，而没有识别出有害的概念，凸显并激励了在这一方向进行认真研究的必要性。作为案例研究，我们展示了攻击者如何使用我们的攻击来轻松生成恶意软件样本和欺诈性短信消息集，这些信息在绕过检测方面表现良好。



## **35. Evaluating Language Models For Threat Detection in IoT Security Logs**

评估语言模型用于物联网安全威胁检测 cs.CR

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02390v1) [paper-pdf](http://arxiv.org/pdf/2507.02390v1)

**Authors**: Jorge J. Tejero-Fernández, Alfonso Sánchez-Macián

**Abstract**: Log analysis is a relevant research field in cybersecurity as they can provide a source of information for the detection of threats to networks and systems. This paper presents a pipeline to use fine-tuned Large Language Models (LLMs) for anomaly detection and mitigation recommendation using IoT security logs. Utilizing classical machine learning classifiers as a baseline, three open-source LLMs are compared for binary and multiclass anomaly detection, with three strategies: zero-shot, few-shot prompting and fine-tuning using an IoT dataset. LLMs give better results on multi-class attack classification than the corresponding baseline models. By mapping detected threats to MITRE CAPEC, defining a set of IoT-specific mitigation actions, and fine-tuning the models with those actions, the models are able to provide a combined detection and recommendation guidance.

摘要: 日志分析是网络安全中的一个相关研究领域，因为它们可以为检测网络和系统的威胁提供信息来源。本文提出了一种管道，使用微调的大型语言模型（LLM），使用物联网安全日志进行异常检测和缓解建议。利用经典的机器学习分类器作为基线，比较了三种开源LLM的二进制和多类异常检测，采用三种策略：零次、少量提示和使用物联网数据集进行微调。LLM在多类攻击分类上比相应的基线模型给出了更好的结果。通过将检测到的威胁映射到MITRE CAPEC，定义一组特定于物联网的缓解措施，并使用这些措施微调模型，这些模型能够提供组合的检测和建议指导。



## **36. SecAlign: Defending Against Prompt Injection with Preference Optimization**

SecAlign：通过偏好优化抵御提示注入 cs.CR

ACM CCS 2025. Key words: prompt injection defense, LLM security,  LLM-integrated applications

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2410.05451v3) [paper-pdf](http://arxiv.org/pdf/2410.05451v3)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the Internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be injected into external data sources to override the system's intended instruction and instead execute a malicious instruction. To mitigate this vulnerability, we propose a new defense called SecAlign based on the technique of preference optimization. Our defense first constructs a preference dataset with prompt-injected inputs, secure outputs (ones that respond to the legitimate instruction), and insecure outputs (ones that respond to the injection). We then perform preference optimization on this dataset to teach the LLM to prefer the secure output over the insecure one. This provides the first known method that reduces the success rates of various prompt injections to <10%, even against attacks much more sophisticated than ones seen during training. This indicates our defense generalizes well against unknown and yet-to-come attacks. Also, SecAlign models are still practical with similar utility to the one before defensive training in our evaluations. Our code is at https://github.com/facebookresearch/SecAlign

摘要: 大型语言模型（LLM）在现代软件系统中变得越来越普遍，在用户和互联网之间进行接口，以协助执行需要高级语言理解的任务。为了完成这些任务，LLM通常使用外部数据源，例如用户文档、Web检索、API调用的结果等。这为攻击者通过提示注入操纵LLM开辟了新的途径。对抗性提示可以被注入到外部数据源中，以覆盖系统的预期指令，转而执行恶意指令。为了缓解此漏洞，我们基于偏好优化技术提出了一种名为SecAlign的新防御。我们的防御首先构建一个具有预算注入的输入、安全输出（响应合法指令的输出）和不安全输出（响应注入的输出）的偏好数据集。然后，我们对该数据集执行偏好优化，以教导LLM更喜欢安全的输出而不是不安全的输出。这提供了第一种已知的方法，可以将各种即时注射的成功率降低到<10%，即使是针对比训练期间看到的攻击复杂得多的攻击。这表明我们的防御对于未知和尚未到来的攻击具有很好的概括性。此外，SecAlign模型仍然实用，与我们评估中防御训练前的模型相似。我们的代码位于https://github.com/facebookresearch/SecAlign



## **37. MGC: A Compiler Framework Exploiting Compositional Blindness in Aligned LLMs for Malware Generation**

MCR：一个更简单的框架，利用对齐的LLM中的合成盲度来生成恶意软件 cs.CR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.02057v1) [paper-pdf](http://arxiv.org/pdf/2507.02057v1)

**Authors**: Lu Yan, Zhuo Zhang, Xiangzhe Xu, Shengwei An, Guangyu Shen, Zhou Xuan, Xuan Chen, Xiangyu Zhang

**Abstract**: Large language models (LLMs) have democratized software development, reducing the expertise barrier for programming complex applications. This accessibility extends to malicious software development, raising significant security concerns. While LLM providers have implemented alignment mechanisms to prevent direct generation of overtly malicious code, these safeguards predominantly evaluate individual prompts in isolation, overlooking a critical vulnerability: malicious operations can be systematically decomposed into benign-appearing sub-tasks. In this paper, we introduce the Malware Generation Compiler (MGC), a novel framework that leverages this vulnerability through modular decomposition and alignment-evasive generation. MGC employs a specialized Malware Description Intermediate Representation (MDIR) to bridge high-level malicious intents and benign-appearing code snippets. Extensive evaluation demonstrates that our attack reliably generates functional malware across diverse task specifications and categories, outperforming jailbreaking methods by +365.79% and underground services by +78.07% in correctness on three benchmark datasets. Case studies further show that MGC can reproduce and even enhance 16 real-world malware samples. This work provides critical insights for security researchers by exposing the risks of compositional attacks against aligned AI systems. Demonstrations are available at https://sites.google.com/view/malware-generation-compiler.

摘要: 大型语言模型（LLM）使软件开发民主化，减少了编程复杂应用程序的专业知识障碍。这种可访问性扩展到恶意软件开发，引发了严重的安全问题。虽然LLM提供商已经实施了对齐机制来防止直接生成明显的恶意代码，但这些保护措施主要孤立地评估单个提示，忽略了一个关键漏洞：恶意操作可以系统地分解为看似善意的子任务。在本文中，我们介绍了恶意软件生成漏洞（MRC），这是一个新颖的框架，通过模块化分解和漏洞规避生成来利用该漏洞。MGC采用专门的恶意软件描述中间表示（Malware Description Intermediate Representation，MPEG4）来桥接高级恶意意图和善意代码片段。广泛的评估表明，我们的攻击可以可靠地生成跨不同任务规范和类别的功能性恶意软件，在三个基准数据集上的正确性超过越狱方法+365.79%和地下服务+78.07%。案例研究进一步表明，MGC可以复制甚至增强16个真实世界的恶意软件样本。这项工作为安全研究人员提供了重要的见解，揭示了针对对齐AI系统的组合攻击的风险。演示可在https://sites.google.com/view/malware-generation-compiler上获取。



## **38. Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training**

没有偷看的调整：LLM后培训的可证明隐私和泛化边界 cs.LG

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01752v1) [paper-pdf](http://arxiv.org/pdf/2507.01752v1)

**Authors**: Ismail Labiad, Mathurin Videau, Matthieu Kowalski, Marc Schoenauer, Alessandro Leite, Julia Kempe, Olivier Teytaud

**Abstract**: Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, its reliance on large volumes of labeled data raises privacy and security concerns such as susceptibility to data poisoning attacks and the risk of overfitting. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. However, black box methods also pose significant challenges, including poor scalability to high-dimensional parameter spaces, as prevalent in large language models (LLMs), and high computational costs due to reliance on numerous model evaluations. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide strong theoretical bounds on generalization, differential privacy, susceptibility to data poisoning attacks, and robustness to extraction attacks. BBoxER operates on top of pre-trained LLMs, offering a lightweight and modular enhancement suitable for deployment in restricted or privacy-sensitive environments, in addition to non-vacuous generalization guarantees. In experiments with LLMs, we demonstrate empirically that Retrofitting methods are able to learn, showing how a few iterations of BBoxER improve performance and generalize well on a benchmark of reasoning datasets. This positions BBoxER as an attractive add-on on top of gradient-based optimization.

摘要: 基于对象的优化是深度学习的主力，通过反向传播提供高效且可扩展的训练。然而，它对大量标记数据的依赖引发了隐私和安全问题，例如容易受到数据中毒攻击和过度匹配的风险。相比之下，黑匣子优化方法将模型视为一个不透明的函数，仅依赖函数评估来指导优化，在数据访问受到限制、对抗风险较高或过度匹配令人担忧的场景中提供了一种有希望的替代方案。然而，黑匣子方法也带来了重大挑战，包括大型语言模型（LLM）中普遍存在的对多维参数空间的可扩展性较差，以及由于依赖大量模型评估而导致的高计算成本。本文介绍了BBoxER，这是一种用于LLM后训练的进化黑匣子方法，通过隐式压缩训练数据来引发信息瓶颈。利用信息流的可追溯性，我们在概括性、差异隐私、对数据中毒攻击的敏感性以及对提取攻击的鲁棒性方面提供了强大的理论界限。BBoxER在预先培训的LLM之上运行，除了非空洞的通用保证外，还提供适合在受限制或隐私敏感环境中部署的轻量级模块化增强。在LLM的实验中，我们经验地证明了Retrofit方法能够学习，展示了BBoxER的几次迭代如何提高性能并在推理数据集的基准上很好地概括。这使得BBoxER成为基于梯度的优化之上的一个有吸引力的附加组件。



## **39. Graph Representation-based Model Poisoning on Federated LLMs in CyberEdge Networks**

CyberEdge网络中联邦LLM上基于图表示的模型中毒 cs.CR

7 pages, 5 figures

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01694v1) [paper-pdf](http://arxiv.org/pdf/2507.01694v1)

**Authors**: Hanlin Cai, Haofan Dong, Houtianfu Wang, Kai Li, Ozgur B. Akan

**Abstract**: Federated large language models (FedLLMs) provide powerful generative capabilities in CyberEdge networks while protecting data privacy. However, FedLLMs remains highly vulnerable to model poisoning attacks. This article first reviews recent model poisoning techniques and existing defense mechanisms for FedLLMs, highlighting critical limitations, particularly under non-IID text distributions. In particular, current defenses primarily utilize distance-based outlier detection or norm constraints, operating under the assumption that adversarial updates significantly diverge from benign statistics. This assumption can fail when facing adaptive attackers targeting billionparameter LLMs. Next, this article investigates emerging Graph Representation-Based Model Poisoning (GRMP), a novel attack paradigm that leverages higher-order correlations among honest client gradients to synthesize malicious updates indistinguishable from legitimate model updates. GRMP can effectively evade advanced defenses, resulting in substantial accuracy loss and performance degradation. Moreover, this article outlines a research roadmap emphasizing the importance of graph-aware secure aggregation methods, FedLLMs-specific vulnerability metrics, and evaluation frameworks to strengthen the robustness of future federated language model deployments.

摘要: 联合大型语言模型（FedLLM）在CyberEdge网络中提供强大的生成能力，同时保护数据隐私。然而，FedLLM仍然极易受到模型中毒攻击。本文首先回顾了FedLLM最近的模型中毒技术和现有的防御机制，强调了关键的局限性，特别是在非IID文本分发下。特别是，当前的防御主要利用基于距离的离群值检测或规范约束，在对抗性更新与良性统计数据显着偏离的假设下运行。当面对针对十亿参数LLM的自适应攻击者时，这一假设可能会失败。接下来，本文研究了新兴的基于图表示的模型中毒（GRMP），这是一种新型攻击范式，它利用诚实客户端梯度之间的更高层相关性来合成与合法模型更新没有区别的恶意更新。GRMP可以有效规避高级防御，导致准确性大幅损失和性能下降。此外，本文还概述了一份研究路线图，强调图形感知的安全聚合方法、特定于FedLLM的漏洞指标和评估框架的重要性，以加强未来联邦语言模型部署的稳健性。



## **40. SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism**

SafeTLR：通过删除然后恢复机制在多模式LLM中进行令牌级越狱防御 cs.CR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01513v1) [paper-pdf](http://arxiv.org/pdf/2507.01513v1)

**Authors**: Beitao Chen, Xinyu Lyu, Lianli Gao, Jingkuan Song, Heng Tao Shen

**Abstract**: By incorporating visual inputs, Multimodal Large Language Models (MLLMs) extend LLMs to support visual reasoning. However, this integration also introduces new vulnerabilities, making MLLMs susceptible to multimodal jailbreak attacks and hindering their safe deployment.Existing defense methods, including Image-to-Text Translation, Safe Prompting, and Multimodal Safety Tuning, attempt to address this by aligning multimodal inputs with LLMs' built-in safeguards.Yet, they fall short in uncovering root causes of multimodal vulnerabilities, particularly how harmful multimodal tokens trigger jailbreak in MLLMs? Consequently, they remain vulnerable to text-driven multimodal jailbreaks, often exhibiting overdefensive behaviors and imposing heavy training overhead.To bridge this gap, we present an comprehensive analysis of where, how and which harmful multimodal tokens bypass safeguards in MLLMs. Surprisingly, we find that less than 1% tokens in early-middle layers are responsible for inducing unsafe behaviors, highlighting the potential of precisely removing a small subset of harmful tokens, without requiring safety tuning, can still effectively improve safety against jailbreaks. Motivated by this, we propose Safe Prune-then-Restore (SafePTR), an training-free defense framework that selectively prunes harmful tokens at vulnerable layers while restoring benign features at subsequent layers.Without incurring additional computational overhead, SafePTR significantly enhances the safety of MLLMs while preserving efficiency. Extensive evaluations across three MLLMs and five benchmarks demonstrate SafePTR's state-of-the-art performance in mitigating jailbreak risks without compromising utility.

摘要: 通过结合视觉输入，多模式大型语言模型（MLLM）扩展了LLM以支持视觉推理。然而，这种集成也引入了新的漏洞，使MLLM容易受到多模式越狱攻击并阻碍其安全部署。现有的防御方法，包括图像到文本翻译、安全预算处理和多模式安全调优，试图通过将多模式输入与LLM的内置保护措施相一致来解决这个问题。然而，它们未能发现多模式漏洞的根本原因，特别是有害的多模式代币如何触发MLLM越狱？因此，他们仍然容易受到文本驱动的多模式越狱的影响，通常表现出过度防御行为并施加沉重的培训费用。为了弥合这一差距，我们对MLLM中的哪些有害多模式代币在哪里、如何以及哪些方式绕过保障措施进行了全面分析。令人惊讶的是，我们发现，在早期-中间层中，只有不到1%的令牌会导致不安全的行为，这突出了精确删除一小部分有害令牌的潜力，而不需要进行安全调整，仍然可以有效地提高安全性。基于此，我们提出了安全修剪然后恢复（SafePTR），这是一个无需训练的防御框架，它可以选择性地修剪脆弱层的有害令牌，同时恢复后续层的良性特征。在不产生额外计算开销的情况下，SafePTR显著增强了MLLM的安全性，同时保持了效率。三个MLLM和五个基准测试的广泛评估证明了SafePTR在减轻越狱风险而不影响实用性方面的最先进性能。



## **41. Don't Say No: Jailbreaking LLM by Suppressing Refusal**

不要说不：通过压制拒绝来越狱法学硕士 cs.CL

Accepted by ACL 2025 Findings

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2404.16369v3) [paper-pdf](http://arxiv.org/pdf/2404.16369v3)

**Authors**: Yukai Zhou, Jian Lou, Zhijie Huang, Zhan Qin, Yibei Yang, Wenjie Wang

**Abstract**: Ensuring the safety alignment of Large Language Models (LLMs) is critical for generating responses consistent with human values. However, LLMs remain vulnerable to jailbreaking attacks, where carefully crafted prompts manipulate them into producing toxic content. One category of such attacks reformulates the task as an optimization problem, aiming to elicit affirmative responses from the LLM. However, these methods heavily rely on predefined objectionable behaviors, limiting their effectiveness and adaptability to diverse harmful queries. In this study, we first identify why the vanilla target loss is suboptimal and then propose enhancements to the loss objective. We introduce DSN (Don't Say No) attack, which combines a cosine decay schedule method with refusal suppression to achieve higher success rates. Extensive experiments demonstrate that DSN outperforms baseline attacks and achieves state-of-the-art attack success rates (ASR). DSN also shows strong universality and transferability to unseen datasets and black-box models.

摘要: 确保大型语言模型（LLM）的安全一致对于生成与人类价值观一致的响应至关重要。然而，LLM仍然容易受到越狱攻击，精心设计的提示操纵它们产生有毒内容。一类此类攻击将任务重新定义为优化问题，旨在引起LLM的肯定响应。然而，这些方法严重依赖于预定义的不良行为，限制了它们对各种有害查询的有效性和适应性。在这项研究中，我们首先确定为什么香草目标损失不是最优的，然后提出对损失目标的增强措施。我们引入了SEN（Don ' t Say No）攻击，该攻击将cos衰变调度方法与拒绝抑制相结合，以实现更高的成功率。大量实验表明，SEN的性能优于基线攻击，并实现了最先进的攻击成功率（ASB）。SEN还表现出强大的通用性和对未见数据集和黑匣子模型的可移植性。



## **42. ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks**

ICLShield：探索和缓解上下文学习后门攻击 cs.LG

ICML 2025

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01321v1) [paper-pdf](http://arxiv.org/pdf/2507.01321v1)

**Authors**: Zhiyao Ren, Siyuan Liang, Aishan Liu, Dacheng Tao

**Abstract**: In-context learning (ICL) has demonstrated remarkable success in large language models (LLMs) due to its adaptability and parameter-free nature. However, it also introduces a critical vulnerability to backdoor attacks, where adversaries can manipulate LLM behaviors by simply poisoning a few ICL demonstrations. In this paper, we propose, for the first time, the dual-learning hypothesis, which posits that LLMs simultaneously learn both the task-relevant latent concepts and backdoor latent concepts within poisoned demonstrations, jointly influencing the probability of model outputs. Through theoretical analysis, we derive an upper bound for ICL backdoor effects, revealing that the vulnerability is dominated by the concept preference ratio between the task and the backdoor. Motivated by these findings, we propose ICLShield, a defense mechanism that dynamically adjusts the concept preference ratio. Our method encourages LLMs to select clean demonstrations during the ICL phase by leveraging confidence and similarity scores, effectively mitigating susceptibility to backdoor attacks. Extensive experiments across multiple LLMs and tasks demonstrate that our method achieves state-of-the-art defense effectiveness, significantly outperforming existing approaches (+26.02% on average). Furthermore, our method exhibits exceptional adaptability and defensive performance even for closed-source models (e.g., GPT-4).

摘要: 上下文学习（ICL）因其适应性和无参数性质而在大型语言模型（LLM）中取得了显着的成功。然而，它也引入了后门攻击的关键漏洞，对手可以通过简单地毒害一些ICL演示来操纵LLM行为。在本文中，我们首次提出了双重学习假设，该假设LLM同时学习与任务相关的潜在概念和中毒演示中的后门潜在概念，共同影响模型输出的可能性。通过理论分析，我们推导出ICL后门效应的上界，揭示了漏洞由任务与后门之间的概念偏好比决定。受这些发现的启发，我们提出了ICLShield，这是一种动态调整概念偏好比的防御机制。我们的方法鼓励LLM通过利用置信度和相似性分数在ICL阶段选择干净的演示，从而有效地降低对后门攻击的敏感性。跨多个LLM和任务的广泛实验表明，我们的方法实现了最先进的防御有效性，显着优于现有方法（平均+26.02%）。此外，即使对于闭源模型（例如，GPT-4）。



## **43. GenBFA: An Evolutionary Optimization Approach to Bit-Flip Attacks on LLMs**

GenBFA：对LLM进行位翻转攻击的进化优化方法 cs.CR

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2411.13757v4) [paper-pdf](http://arxiv.org/pdf/2411.13757v4)

**Authors**: Sanjay Das, Swastik Bhattacharya, Souvik Kundu, Shamik Kundu, Anand Menon, Arnab Raha, Kanad Basu

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing (NLP), excelling in tasks like text generation and summarization. However, their increasing adoption in mission-critical applications raises concerns about hardware-based threats, particularly bit-flip attacks (BFAs). BFAs, enabled by fault injection methods such as Rowhammer, target model parameters in memory, compromising both integrity and performance. Identifying critical parameters for BFAs in the vast parameter space of LLMs poses significant challenges. While prior research suggests transformer-based architectures are inherently more robust to BFAs compared to traditional deep neural networks, we challenge this assumption. For the first time, we demonstrate that as few as three bit-flips can cause catastrophic performance degradation in an LLM with billions of parameters. Current BFA techniques are inadequate for exploiting this vulnerability due to the difficulty of efficiently identifying critical parameters within the immense parameter space. To address this, we propose AttentionBreaker, a novel framework tailored for LLMs that enables efficient traversal of the parameter space to identify critical parameters. Additionally, we introduce GenBFA, an evolutionary optimization strategy designed to refine the search further, isolating the most critical bits for an efficient and effective attack. Empirical results reveal the profound vulnerability of LLMs to AttentionBreaker. For example, merely three bit-flips (4.129 x 10^-9% of total parameters) in the LLaMA3-8B-Instruct 8-bit quantized (W8) model result in a complete performance collapse: accuracy on MMLU tasks drops from 67.3% to 0%, and Wikitext perplexity skyrockets from 12.6 to 4.72 x 10^5. These findings underscore the effectiveness of AttentionBreaker in uncovering and exploiting critical vulnerabilities within LLM architectures.

摘要: 大型语言模型（LLM）彻底改变了自然语言处理（NLP），在文本生成和摘要等任务方面表现出色。然而，它们在任务关键型应用程序中的越来越多的采用引发了人们对基于硬件的威胁的担忧，特别是位翻转攻击（BFA）。BFA由Rowhammer等故障注入方法启用，目标是内存中的模型参数，从而损害完整性和性能。在LLM的巨大参数空间中识别BFA的关键参数构成了重大挑战。虽然之前的研究表明，与传统的深度神经网络相比，基于变换器的架构本质上对BFA更稳健，但我们挑战了这一假设。我们首次证明，在具有数十亿个参数的LLM中，只要三个位翻转就可能导致灾难性的性能下降。由于难以在巨大的参数空间中有效识别关键参数，目前的BFA技术不足以利用该漏洞。为了解决这个问题，我们提出了AttentionBreaker，这是一个为LLM量身定制的新型框架，可以有效地穿越参数空间以识别关键参数。此外，我们还引入了GenBFA，这是一种进化优化策略，旨在进一步细化搜索，隔离最关键的部分以进行高效且有效的攻击。实证结果揭示了LLM对AttentionBreaker的严重脆弱性。例如，LLaMA 3 - 8B-Direcct 8位量化（W8）模型中仅进行三次位翻转（总参数的4.129 x 10 '-9%）就会导致性能完全崩溃：MMLU任务的准确性从67.3%下降到0%，维基文本困惑度从12.6飙升到4.72 x 105。这些发现强调了AttentionBreaker在发现和利用LLM架构中关键漏洞方面的有效性。



## **44. Defensive Adversarial CAPTCHA: A Semantics-Driven Framework for Natural Adversarial Example Generation**

防御性对抗验证码：用于自然对抗示例生成的语义驱动框架 cs.CV

13 pages, 6 figures

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2506.10685v3) [paper-pdf](http://arxiv.org/pdf/2506.10685v3)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Cong Wu, Tao Li, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: Traditional CAPTCHA (Completely Automated Public Turing Test to Tell Computers and Humans Apart) schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on the original image characteristics, resulting in distortions that hinder human interpretation and limit their applicability in scenarios where no initial input images are available. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (DAC), a novel framework that generates high-fidelity adversarial examples guided by attacker-specified semantics information. Leveraging a Large Language Model (LLM), DAC enhances CAPTCHA diversity and enriches the semantic information. To address various application scenarios, we examine the white-box targeted attack scenario and the black box untargeted attack scenario. For target attacks, we introduce two latent noise variables that are alternately guided in the diffusion step to achieve robust inversion. The synergy between gradient guidance and latent variable optimization achieved in this way ensures that the generated adversarial examples not only accurately align with the target conditions but also achieve optimal performance in terms of distributional consistency and attack effectiveness. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-DAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show that the defensive adversarial CAPTCHA generated by BP-DAC is able to defend against most of the unknown models, and the generated CAPTCHA is indistinguishable to both humans and DNNs.

摘要: 传统的CAPTCHA（完全自动化公共图灵测试来区分计算机和人类）计划越来越容易受到深度神经网络（DNN）支持的自动化攻击。现有的对抗攻击方法通常依赖于原始图像特征，从而导致失真，阻碍人类解释并限制其在没有初始输入图像可用的场景中的适用性。为了解决这些挑战，我们提出了无源对抗性验证码（ADC），这是一种新颖的框架，可以在攻击者指定的语义信息的指导下生成高保真对抗性示例。利用大型语言模型（LLM），DEC增强了CAPTCHA的多样性并丰富了语义信息。为了应对各种应用场景，我们研究了白盒定向攻击场景和黑匣子非定向攻击场景。对于目标攻击，我们引入了两个潜在噪音变量，它们在扩散步骤中交替引导，以实现鲁棒的反转。通过这种方式实现的梯度引导和潜在变量优化之间的协同作用，确保生成的对抗示例不仅与目标条件准确对齐，而且在分布一致性和攻击有效性方面实现最佳性能。在无目标攻击中，特别是对于黑匣子场景，我们引入了双路径无源对抗性CAPTCHA（BP-ADC），这是一种两步优化策略，采用多峰梯度和双路径优化来实现高效的误分类。实验表明，BP-ADC生成的防御性对抗CAPTCHA能够防御大多数未知模型，并且生成的CAPTCHA对于人类和DNN来说都无法区分。



## **45. SafeMobile: Chain-level Jailbreak Detection and Automated Evaluation for Multimodal Mobile Agents**

SafeMobile：多模式移动代理的连锁级越狱检测和自动评估 cs.AI

12 pages

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00841v1) [paper-pdf](http://arxiv.org/pdf/2507.00841v1)

**Authors**: Siyuan Liang, Tianmeng Fang, Zhe Liu, Aishan Liu, Yan Xiao, Jinyuan He, Ee-Chien Chang, Xiaochun Cao

**Abstract**: With the wide application of multimodal foundation models in intelligent agent systems, scenarios such as mobile device control, intelligent assistant interaction, and multimodal task execution are gradually relying on such large model-driven agents. However, the related systems are also increasingly exposed to potential jailbreak risks. Attackers may induce the agents to bypass the original behavioral constraints through specific inputs, and then trigger certain risky and sensitive operations, such as modifying settings, executing unauthorized commands, or impersonating user identities, which brings new challenges to system security. Existing security measures for intelligent agents still have limitations when facing complex interactions, especially in detecting potentially risky behaviors across multiple rounds of conversations or sequences of tasks. In addition, an efficient and consistent automated methodology to assist in assessing and determining the impact of such risks is currently lacking. This work explores the security issues surrounding mobile multimodal agents, attempts to construct a risk discrimination mechanism by incorporating behavioral sequence information, and designs an automated assisted assessment scheme based on a large language model. Through preliminary validation in several representative high-risk tasks, the results show that the method can improve the recognition of risky behaviors to some extent and assist in reducing the probability of agents being jailbroken. We hope that this study can provide some valuable references for the security risk modeling and protection of multimodal intelligent agent systems.

摘要: 随着多模式基础模型在智能代理系统中的广泛应用，移动终端控制、智能助理交互、多模式任务执行等场景逐渐依赖于此类大型模型驱动的代理。然而，相关系统也越来越多地面临潜在的越狱风险。攻击者可能会通过特定的输入诱导代理绕过原始的行为约束，然后触发某些有风险和敏感的操作，例如修改设置、执行未经授权的命令或冒充用户身份，这给系统安全带来了新的挑战。智能代理的现有安全措施在面临复杂的交互时仍然存在局限性，特别是在多轮对话或任务序列中检测潜在的危险行为方面。此外，目前还缺乏一种有效和一致的自动化方法来协助评估和确定这些风险的影响。本文探讨了移动多通道代理的安全问题，尝试通过引入行为序列信息构建风险鉴别机制，并设计了一种基于大型语言模型的自动辅助评估方案。通过在几个有代表性的高风险任务中的初步验证，结果表明该方法在一定程度上提高了对危险行为的识别，有助于降低智能体越狱的概率。希望本文的研究能为多通道智能代理系统的安全风险建模和防护提供一些有价值的参考。



## **46. CAVALRY-V: A Large-Scale Generator Framework for Adversarial Attacks on Video MLLMs**

CAWLRY-V：一个用于视频MLLM对抗性攻击的大规模生成器框架 cs.CV

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00817v1) [paper-pdf](http://arxiv.org/pdf/2507.00817v1)

**Authors**: Jiaming Zhang, Rui Hu, Qing Guo, Wei Yang Bryan Lim

**Abstract**: Video Multimodal Large Language Models (V-MLLMs) have shown impressive capabilities in temporal reasoning and cross-modal understanding, yet their vulnerability to adversarial attacks remains underexplored due to unique challenges: complex cross-modal reasoning mechanisms, temporal dependencies, and computational constraints. We present CAVALRY-V (Cross-modal Language-Vision Adversarial Yielding for Videos), a novel framework that directly targets the critical interface between visual perception and language generation in V-MLLMs. Our approach introduces two key innovations: (1) a dual-objective semantic-visual loss function that simultaneously disrupts the model's text generation logits and visual representations to undermine cross-modal integration, and (2) a computationally efficient two-stage generator framework that combines large-scale pre-training for cross-model transferability with specialized fine-tuning for spatiotemporal coherence. Empirical evaluation on comprehensive video understanding benchmarks demonstrates that CAVALRY-V significantly outperforms existing attack methods, achieving 22.8% average improvement over the best baseline attacks on both commercial systems (GPT-4.1, Gemini 2.0) and open-source models (QwenVL-2.5, InternVL-2.5, Llava-Video, Aria, MiniCPM-o-2.6). Our framework achieves flexibility through implicit temporal coherence modeling rather than explicit regularization, enabling significant performance improvements even on image understanding (34.4% average gain). This capability demonstrates CAVALRY-V's potential as a foundational approach for adversarial research across multimodal systems.

摘要: 视频多模式大型语言模型（V-MLLM）在时态推理和跨模式理解方面表现出了令人印象深刻的能力，但由于独特的挑战，它们对对抗性攻击的脆弱性仍然没有得到充分的研究：复杂的跨模式推理机制、时态依赖性和计算限制。我们提出了CAWLRY-V（跨模式视觉对抗屈服视频），这是一个新颖的框架，直接针对V-MLLM中视觉感知和语言生成之间的关键界面。我们的方法引入了两个关键创新：（1）双目标语义视觉损失函数，它同时扰乱模型的文本生成日志和视觉表示以破坏跨模式集成，以及（2）计算高效的两阶段生成器框架，它将跨模型可移植性的大规模预训练与时空一致性的专门微调相结合。对全面视频理解基准的实证评估表明，CAWLRY-V的表现显着优于现有的攻击方法，比商业系统（GPT-4.1、Gemini 2.0）和开源模型（QwenVL-2.5、InternVL-2.5、Llava-Video、Aria、MiniCPM-o-2.6）的最佳基线攻击平均改进了22.8%。我们的框架通过隐式时间一致性建模而不是显式正规化来实现灵活性，即使在图像理解方面也能显着提高性能（平均提高34.4%）。这一能力展示了CAWLRY-V作为跨多模式系统对抗性研究的基础方法的潜力。



## **47. Impact of Fine-Tuning Methods on Memorization in Large Language Models**

微调方法对大型语言模型中精简化的影响 cs.CL

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2507.00258v1) [paper-pdf](http://arxiv.org/pdf/2507.00258v1)

**Authors**: Jie Hou, Chuxiong Wu, Lannan Luo, Qiang Zeng

**Abstract**: As the capabilities of pre-trained large language models (LLMs) continue to advance, the "pre-train and fine-tune" paradigm has become increasingly mainstream, leading to the development of various fine-tuning methods. However, the privacy risks arising from memorization during fine-tuning have received relatively little attention. To address this gap, we categorize popular fine-tuning approaches and assess their impact on memorization through the lens of membership inference attacks (MIAs). Our results show that, compared to parameter-based fine-tuning, prompt-based fine-tuning achieves competitive performance while exhibiting lower vulnerability to MIAs. Furthermore, prompt-based methods maintain low memorization regardless of model scale. These findings suggest that parameter-based fine-tuning is more prone to leaking private information, whereas prompt-based fine-tuning serves as a more privacy-preserving option.

摘要: 随着预训练大型语言模型（LLM）能力的不断进步，“预训练和微调”范式日益成为主流，导致各种微调方法的发展。然而，微调过程中因记忆而产生的隐私风险相对较少受到关注。为了解决这一差距，我们对流行的微调方法进行了分类，并通过成员资格推理攻击（MIA）的视角评估它们对记忆的影响。我们的结果表明，与基于参数的微调相比，基于预算的微调可以实现有竞争力的性能，同时对MIA的脆弱性更低。此外，无论模型规模如何，基于预算的方法都保持较低的记忆力。这些发现表明，基于参数的微调更容易泄露私人信息，而基于预算的微调则是一种更能保护隐私的选择。



## **48. Trust & Safety of LLMs and LLMs in Trust & Safety**

LLM的信任与安全以及LLM的信任与安全 cs.AI

11 pages

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2412.02113v2) [paper-pdf](http://arxiv.org/pdf/2412.02113v2)

**Authors**: Doohee You, Dan Chon

**Abstract**: In recent years, Large Language Models (LLMs) have garnered considerable attention for their remarkable abilities in natural language processing tasks. However, their widespread adoption has raised concerns pertaining to trust and safety. This systematic review investigates the current research landscape on trust and safety in LLMs, with a particular focus on the novel application of LLMs within the field of Trust and Safety itself. We delve into the complexities of utilizing LLMs in domains where maintaining trust and safety is paramount, offering a consolidated perspective on this emerging trend.\   By synthesizing findings from various studies, we identify key challenges and potential solutions, aiming to benefit researchers and practitioners seeking to understand the nuanced interplay between LLMs and Trust and Safety.   This review provides insights on best practices for using LLMs in Trust and Safety, and explores emerging risks such as prompt injection and jailbreak attacks. Ultimately, this study contributes to a deeper understanding of how LLMs can be effectively and responsibly utilized to enhance trust and safety in the digital realm.

摘要: 近年来，大型语言模型（LLM）因其在自然语言处理任务中的非凡能力而受到了广泛关注。然而，它们的广泛采用引发了人们对信任和安全的担忧。这篇系统性综述调查了当前关于LLM信任和安全的研究格局，特别关注LLM在信任和安全本身领域的新颖应用。我们深入研究了在维护信任和安全至关重要的领域中利用LLM的复杂性，为这一新兴趋势提供了统一的视角。\   通过综合各种研究的结果，我们确定了关键挑战和潜在的解决方案，旨在使寻求了解法学硕士与信任和安全之间微妙相互作用的研究人员和从业者受益。   本评论提供了有关在信任与安全中使用LLM的最佳实践的见解，并探讨了即时注射和越狱攻击等新出现的风险。最终，这项研究有助于更深入地了解如何有效、负责任地利用LLM来增强数字领域的信任和安全。



## **49. Logit-Gap Steering: Efficient Short-Suffix Jailbreaks for Aligned Large Language Models**

Logit-Gap Steering：Aligned Large Language Models的高效短后缀越狱 cs.CR

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24056v1) [paper-pdf](http://arxiv.org/pdf/2506.24056v1)

**Authors**: Tung-Ling Li, Hongliang Liu

**Abstract**: We introduce logit-gap steering, a fast jailbreak framework that casts the refusal-affirmation gap of RLHF-aligned language models as a single pass over the vocabulary. A forward-computable score blends gap reduction with lightweight proxies for KL penalty and reward shift, allowing a "sort-sum-stop" sweep to complete in under a second and return a short suffix--two orders of magnitude fewer model calls than beam or gradient attacks. The same suffix generalises to unseen prompts and scales from 0.5 B to 70 B checkpoints, lifting one-shot attack success from baseline levels to 80-100% while preserving topical coherence. Beyond efficiency, these suffixes expose sentence-boundary reward cliffs and other alignment artefacts, offering a lightweight probe into how safety tuning reshapes internal representations.

摘要: 我们引入了logit-gap steering，这是一个快速越狱框架，它将RLHF对齐的语言模型的反思-肯定差距视为词汇的一次传递。可向前计算的分数将差距缩小与KL惩罚和奖励转移的轻量级代理结合起来，允许“排序和停止”扫描在一秒内完成并返回短后缀--模型调用比束或梯度攻击少两个数量级。相同的后缀推广到未见的提示，并将0.5 B到70 B检查点范围内，将一次性攻击成功率从基线水平提高到80-100%，同时保持话题连贯性。除了效率之外，这些后缀还暴露了行业边界奖励悬崖和其他对齐文物，为安全调整如何重塑内部表示提供了轻量级的探索。



## **50. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

谜语我！检索增强一代的隐形会员推断 cs.CR

This is the full version (27 pages) of the paper 'Riddle Me This!  Stealthy Membership Inference for Retrieval-Augmented Generation' published  at CCS 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2502.00306v2) [paper-pdf](http://arxiv.org/pdf/2502.00306v2)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.

摘要: 检索增强生成（RAG）使大型语言模型（LLM）能够通过利用外部知识数据库来生成接地响应，而无需更改模型参数。尽管缺乏权重调整可以防止模型参数泄露，但它引入了推理对手利用模型上下文中检索到的文档的风险。现有的隶属关系推断和数据提取方法通常依赖于越狱或精心制作的非自然查询，这些查询可以通过RAG系统中常见的查询重写技术轻松检测或阻止。在这项工作中，我们介绍了审讯攻击（IA），这是一种针对RAG收件箱中文档的成员资格推断技术。通过制作仅在目标文档存在的情况下才能回答的自然文本查询，我们的方法仅用30个查询就能证明成功推理，同时保持隐蔽性;简单的检测器识别来自现有方法的对抗性提示的频率高达约76倍，比我们的攻击产生的提示。我们观察到，在各种RAG配置中，TPR@1%FPR比之前的推理攻击提高了2倍，同时每个文档推理的成本不到0.02美元。



