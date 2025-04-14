# Latest Large Language Model Attack Papers
**update at 2025-04-14 11:09:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits**

HCP安全审计：具有模型上下文协议的LLM允许重大安全漏洞 cs.CR

27 pages, 21 figures, and 2 Tables. Cleans up the TeX source

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.03767v2) [paper-pdf](http://arxiv.org/pdf/2504.03767v2)

**Authors**: Brandon Radosevich, John Halloran

**Abstract**: To reduce development overhead and enable seamless integration between potential components comprising any given generative AI application, the Model Context Protocol (MCP) (Anthropic, 2024) has recently been released and subsequently widely adopted. The MCP is an open protocol that standardizes API calls to large language models (LLMs), data sources, and agentic tools. By connecting multiple MCP servers, each defined with a set of tools, resources, and prompts, users are able to define automated workflows fully driven by LLMs. However, we show that the current MCP design carries a wide range of security risks for end users. In particular, we demonstrate that industry-leading LLMs may be coerced into using MCP tools to compromise an AI developer's system through various attacks, such as malicious code execution, remote access control, and credential theft. To proactively mitigate these and related attacks, we introduce a safety auditing tool, MCPSafetyScanner, the first agentic tool to assess the security of an arbitrary MCP server. MCPScanner uses several agents to (a) automatically determine adversarial samples given an MCP server's tools and resources; (b) search for related vulnerabilities and remediations based on those samples; and (c) generate a security report detailing all findings. Our work highlights serious security issues with general-purpose agentic workflows while also providing a proactive tool to audit MCP server safety and address detected vulnerabilities before deployment.   The described MCP server auditing tool, MCPSafetyScanner, is freely available at: https://github.com/johnhalloran321/mcpSafetyScanner

摘要: 为了减少开发费用并实现构成任何给定生成式人工智能应用程序的潜在组件之间的无缝集成，模型上下文协议（HCP）（Anthropic，2024）最近发布并随后广泛采用。HCP是一种开放协议，可同步化对大型语言模型（LLM）、数据源和代理工具的API调用。通过连接多个HCP服务器（每个服务器都定义了一组工具、资源和提示），用户能够定义完全由LLM驱动的自动化工作流程。然而，我们表明当前的LCP设计对最终用户来说存在广泛的安全风险。特别是，我们证明了行业领先的LLM可能会被迫使用LCP工具通过各种攻击（例如恶意代码执行、远程访问控制和凭证盗窃）来危害人工智能开发人员的系统。为了主动缓解这些攻击和相关攻击，我们引入了安全审计工具MCPSafetyScanner，这是第一个评估任意LCP服务器安全性的代理工具。MCPScanner使用多个代理来（a）在给定HCP服务器的工具和资源的情况下自动确定对抗样本;（b）根据这些样本搜索相关漏洞和补救措施;以及（c）生成详细说明所有发现结果的安全报告。我们的工作强调了通用代理工作流程的严重安全问题，同时还提供了一种主动工具来审计LCP服务器的安全性并在部署之前解决检测到的漏洞。   所描述的LCP服务器审计工具MCPSafetyScanner可在以下网址免费获取：https://github.com/johnhalloran321/mcpSafetyScanner



## **2. SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models**

SCAM：多模式基础模型的真实印刷稳健性评估 cs.CV

Submitted to CVPR 2025 Workshop EVAL-FoMo-2

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.04893v2) [paper-pdf](http://arxiv.org/pdf/2504.04893v2)

**Authors**: Justus Westerhoff, Erblina Purelku, Jakob Hackstein, Leo Pinetzki, Lorenz Hufe

**Abstract**: Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing 1,162 images across hundreds of object categories and attack words. Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability. Additionally, we demonstrate that synthetic attacks closely resemble real-world (handwritten) attacks, validating their use in research. Our work provides a comprehensive resource and empirical insights to facilitate future research toward robust and trustworthy multimodal AI systems. We publicly release the datasets introduced in this paper under https://huggingface.co/datasets/BLISS-e-V/SCAM, along with the code for evaluations at https://github.com/Bliss-e-V/SCAM.

摘要: 印刷攻击利用多模式基础模型中文本和视觉内容之间的相互作用，当误导性文本嵌入图像中时，会导致错误分类。然而，现有数据集的大小和多样性有限，因此很难研究此类漏洞。在本文中，我们介绍了SCAM，这是迄今为止最大、最多样化的现实世界印刷攻击图像数据集，包含涵盖数百个对象类别和攻击词的1，162张图像。通过对SCAM上的视觉语言模型（VLM）进行广泛的基准测试，我们证明了印刷攻击会显着降低性能，并确定训练数据和模型架构会影响对这些攻击的易感性。我们的研究结果表明，由于视觉编码器的选择，印刷攻击在最先进的大型视觉语言模型（LVLM）中持续存在，尽管更大的大型语言模型（LLM）主干有助于减轻它们的脆弱性。此外，我们还证明合成攻击与现实世界（手写）攻击非常相似，验证了它们在研究中的用途。我们的工作提供了全面的资源和经验见解，以促进未来对稳健且值得信赖的多模式人工智能系统的研究。我们在https：//huggingface.co/guardets/BLISS-e-V/SCAM下公开发布本文中介绍的数据集，以及https://github.com/Bliss-e-V/SCAM上的评估代码。



## **3. GenXSS: an AI-Driven Framework for Automated Detection of XSS Attacks in WAFs**

GenXSS：用于自动检测WAF中XSS攻击的人工智能驱动框架 cs.CR

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08176v1) [paper-pdf](http://arxiv.org/pdf/2504.08176v1)

**Authors**: Vahid Babaey, Arun Ravindran

**Abstract**: The increasing reliance on web services has led to a rise in cybersecurity threats, particularly Cross-Site Scripting (XSS) attacks, which target client-side layers of web applications by injecting malicious scripts. Traditional Web Application Firewalls (WAFs) struggle to detect highly obfuscated and complex attacks, as their rules require manual updates. This paper presents a novel generative AI framework that leverages Large Language Models (LLMs) to enhance XSS mitigation. The framework achieves two primary objectives: (1) generating sophisticated and syntactically validated XSS payloads using in-context learning, and (2) automating defense mechanisms by testing these attacks against a vulnerable application secured by a WAF, classifying bypassing attacks, and generating effective WAF security rules. Experimental results using GPT-4o demonstrate the framework's effectiveness generating 264 XSS payloads, 83% of which were validated, with 80% bypassing ModSecurity WAF equipped with an industry standard security rule set developed by the Open Web Application Security Project (OWASP) to protect against web vulnerabilities. Through rule generation, 86% of previously successful attacks were blocked using only 15 new rules. In comparison, Google Gemini Pro achieved a lower bypass rate of 63%, highlighting performance differences across LLMs.

摘要: 对网络服务的依赖日益增加，导致网络安全威胁的增加，特别是跨站点脚本（XSS）攻击，这些攻击通过注入恶意脚本来针对网络应用程序的客户端层。传统的Web应用程序防火墙（WAF）很难检测高度模糊和复杂的攻击，因为它们的规则需要手动更新。本文提出了一种新颖的生成式人工智能框架，该框架利用大型语言模型（LLM）来增强XSS缓解。该框架实现了两个主要目标：（1）使用上下文学习生成复杂且经过语法验证的XSS有效负载，以及（2）通过针对WAF保护的脆弱应用程序测试这些攻击、对绕过攻击进行分类并生成有效的WAF安全规则来自动化防御机制。使用GPT-4 o的实验结果证明了该框架的有效性，生成264个XSS有效负载，其中83%经过验证，80%绕过ModSecurity WAF，该WAF配备了开放Web应用程序安全项目（OWISP）开发的行业标准安全规则集，以防止网络漏洞。通过规则生成，仅使用15个新规则就阻止了86%之前成功的攻击。相比之下，Google Gemini Pro的绕过率较低，为63%，凸显了LLM之间的性能差异。



## **4. Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge**

大型语言模型中对偏见激发的对抗鲁棒性进行基准测试：利用LLM作为评委的可扩展自动化评估 cs.CL

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07887v1) [paper-pdf](http://arxiv.org/pdf/2504.07887v1)

**Authors**: Riccardo Cantini, Alessio Orsino, Massimo Ruggiero, Domenico Talia

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence, driving advancements in machine translation, summarization, and conversational agents. However, their increasing integration into critical societal domains has raised concerns about embedded biases, which can perpetuate stereotypes and compromise fairness. These biases stem from various sources, including historical inequalities in training data, linguistic imbalances, and adversarial manipulation. Despite mitigation efforts, recent studies indicate that LLMs remain vulnerable to adversarial attacks designed to elicit biased responses. This work proposes a scalable benchmarking framework to evaluate LLM robustness against adversarial bias elicitation. Our methodology involves (i) systematically probing models with a multi-task approach targeting biases across various sociocultural dimensions, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach for automated assessment of model responses, and (iii) employing jailbreak techniques to investigate vulnerabilities in safety mechanisms. Our analysis examines prevalent biases in both small and large state-of-the-art models and their impact on model safety. Additionally, we assess the safety of domain-specific models fine-tuned for critical fields, such as medicine. Finally, we release a curated dataset of bias-related prompts, CLEAR-Bias, to facilitate systematic vulnerability benchmarking. Our findings reveal critical trade-offs between model size and safety, aiding the development of fairer and more robust future language models.

摘要: 大型语言模型（LLM）彻底改变了人工智能，推动了机器翻译、摘要和对话代理的进步。然而，他们越来越融入关键社会领域，引发了人们对根深蒂固的偏见的担忧，这可能会延续刻板印象并损害公平性。这些偏见源于各种来源，包括训练数据的历史不平等、语言不平衡和对抗操纵。尽管做出了缓解措施，但最近的研究表明，LLM仍然容易受到旨在引发偏见反应的对抗攻击。这项工作提出了一个可扩展的基准测试框架，以评估LLM针对对抗性偏见引发的稳健性。我们的方法包括：（i）采用针对各个社会文化维度的偏见的多任务方法系统地探索模型，（ii）使用LLM作为法官的方法通过安全评分量化稳健性，以自动评估模型响应，以及（iii）采用越狱技术来调查安全机制中的漏洞。我们的分析检查了小型和大型最先进模型中普遍存在的偏差及其对模型安全性的影响。此外，我们还评估针对医学等关键领域进行微调的特定领域模型的安全性。最后，我们发布了一个精心策划的偏差相关提示数据集ClearAR-Bias，以促进系统性漏洞基准测试。我们的研究结果揭示了模型大小和安全性之间的关键权衡，有助于开发更公平、更强大的未来语言模型。



## **5. PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization**

PR攻击：通过二层优化对大型语言模型中的检索增强生成进行协调的预算-RAG攻击 cs.CR

Accepted at SIGIR 2025

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07717v1) [paper-pdf](http://arxiv.org/pdf/2504.07717v1)

**Authors**: Yang Jiao, Xiaodong Wang, Kai Yang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of applications, e.g., medical question-answering, mathematical sciences, and code generation. However, they also exhibit inherent limitations, such as outdated knowledge and susceptibility to hallucinations. Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to address these issues, but it also introduces new vulnerabilities. Recent efforts have focused on the security of RAG-based LLMs, yet existing attack methods face three critical challenges: (1) their effectiveness declines sharply when only a limited number of poisoned texts can be injected into the knowledge database, (2) they lack sufficient stealth, as the attacks are often detectable by anomaly detection systems, which compromises their effectiveness, and (3) they rely on heuristic approaches to generate poisoned texts, lacking formal optimization frameworks and theoretic guarantees, which limits their effectiveness and applicability. To address these issues, we propose coordinated Prompt-RAG attack (PR-attack), a novel optimization-driven attack that introduces a small number of poisoned texts into the knowledge database while embedding a backdoor trigger within the prompt. When activated, the trigger causes the LLM to generate pre-designed responses to targeted queries, while maintaining normal behavior in other contexts. This ensures both high effectiveness and stealth. We formulate the attack generation process as a bilevel optimization problem leveraging a principled optimization framework to develop optimal poisoned texts and triggers. Extensive experiments across diverse LLMs and datasets demonstrate the effectiveness of PR-Attack, achieving a high attack success rate even with a limited number of poisoned texts and significantly improved stealth compared to existing methods.

摘要: 大型语言模型（LLM）已在广泛的应用程序中表现出出色的性能，例如，医学问答、数学科学和代码生成。然而，它们也表现出固有的局限性，例如过时的知识和幻觉的易感性。检索增强一代（RAG）已成为解决这些问题的一个有希望的范式，但它也引入了新的漏洞。最近的工作重点是基于RAG的LLM的安全性，但现有的攻击方法面临三个关键挑战：（1）当只能将有限数量的有毒文本注入知识数据库时，它们的有效性急剧下降，（2）它们缺乏足够的隐蔽性，因为异常检测系统通常可以检测到攻击，这损害了它们的有效性，（3）它们依赖启发式方法来生成有毒文本，缺乏正式的优化框架和理论保证，这限制了它们的有效性和适用性。为了解决这些问题，我们提出了协调的预算-RAG攻击（PR-攻击），这是一种新型的优化驱动攻击，它将少量有毒文本引入知识数据库，同时在提示内嵌入后门触发器。激活时，触发器会使LLM生成对目标查询的预先设计的响应，同时在其他上下文中保持正常行为。这确保了高效率和隐形性。我们将攻击生成过程制定为一个双层优化问题，利用有原则的优化框架来开发最佳的有毒文本和触发器。跨不同LLM和数据集的广泛实验证明了PR-Attack的有效性，即使在数量有限的有毒文本的情况下也能实现很高的攻击成功率，并且与现有方法相比，隐身性显着提高。



## **6. Agent That Debugs: Dynamic State-Guided Vulnerability Repair**

调试代理：动态状态引导的漏洞修复 cs.SE

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07634v1) [paper-pdf](http://arxiv.org/pdf/2504.07634v1)

**Authors**: Zhengyao Liu, Yunlong Ma, Jingxuan Xu, Junchen Ai, Xiang Gao, Hailong Sun, Abhik Roychoudhury

**Abstract**: In recent years, more vulnerabilities have been discovered every day, while manual vulnerability repair requires specialized knowledge and is time-consuming. As a result, many detected or even published vulnerabilities remain unpatched, thereby increasing the exposure of software systems to attacks. Recent advancements in agents based on Large Language Models have demonstrated their increasing capabilities in code understanding and generation, which can be promising to achieve automated vulnerability repair. However, the effectiveness of agents based on static information retrieval is still not sufficient for patch generation. To address the challenge, we propose a program repair agent called VulDebugger that fully utilizes both static and dynamic context, and it debugs programs in a manner akin to humans. The agent inspects the actual state of the program via the debugger and infers expected states via constraints that need to be satisfied. By continuously comparing the actual state with the expected state, it deeply understands the root causes of the vulnerabilities and ultimately accomplishes repairs. We experimentally evaluated VulDebugger on 50 real-life projects. With 60.00% successfully fixed, VulDebugger significantly outperforms state-of-the-art approaches for vulnerability repair.

摘要: 近年来，每天都有更多的漏洞被发现，而人工漏洞修复需要专业知识，耗时长。因此，许多检测到的甚至公布的漏洞仍然没有修补，从而增加了软件系统遭受攻击的风险。基于大型语言模型的代理的最新进展已经证明了它们在代码理解和生成方面的能力越来越强，这可能有望实现自动漏洞修复。然而，基于静态信息检索的代理的有效性仍然是不够的补丁生成。为了应对这一挑战，我们提出了一种名为Vulligger的程序修复代理，它充分利用静态和动态上下文，并以类似于人类的方式调试程序。代理通过调试器检查程序的实际状态，并通过需要满足的约束推断预期状态。通过不断比较实际状态与预期状态，深刻了解漏洞的根本原因并最终完成修复。我们对50个现实生活中的项目进行了实验性评估。Vulligger成功修复60.00%，其性能明显优于最先进的漏洞修复方法。



## **7. Defense against Prompt Injection Attacks via Mixture of Encodings**

通过混合编码防御即时注入攻击 cs.CL

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07467v1) [paper-pdf](http://arxiv.org/pdf/2504.07467v1)

**Authors**: Ruiyi Zhang, David Sullivan, Kyle Jackson, Pengtao Xie, Mei Chen

**Abstract**: Large Language Models (LLMs) have emerged as a dominant approach for a wide range of NLP tasks, with their access to external information further enhancing their capabilities. However, this introduces new vulnerabilities, known as prompt injection attacks, where external content embeds malicious instructions that manipulate the LLM's output. Recently, the Base64 defense has been recognized as one of the most effective methods for reducing success rate of prompt injection attacks. Despite its efficacy, this method can degrade LLM performance on certain NLP tasks. To address this challenge, we propose a novel defense mechanism: mixture of encodings, which utilizes multiple character encodings, including Base64. Extensive experimental results show that our method achieves one of the lowest attack success rates under prompt injection attacks, while maintaining high performance across all NLP tasks, outperforming existing character encoding-based defense methods. This underscores the effectiveness of our mixture of encodings strategy for both safety and task performance metrics.

摘要: 大型语言模型（LLM）已成为各种NLP任务的主要方法，它们对外部信息的访问进一步增强了它们的能力。然而，这会引入新的漏洞，称为提示注入攻击，其中外部内容嵌入了操纵LLM输出的恶意指令。最近，Base 64防御被公认为降低即时注入攻击成功率的最有效方法之一。尽管该方法有效，但它可能会降低LLM在某些NLP任务上的性能。为了应对这一挑战，我们提出了一种新颖的防御机制：混合编码，它利用多种字符编码，包括Base 64。大量实验结果表明，我们的方法在提示注入攻击下实现了攻击成功率最低的方法之一，同时在所有NLP任务中保持高性能，优于现有的基于字符编码的防御方法。这凸显了我们混合编码策略对于安全和任务性能指标的有效性。



## **8. Achilles Heel of Distributed Multi-Agent Systems**

分布式多智能体系统的致命弱点 cs.MA

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07461v1) [paper-pdf](http://arxiv.org/pdf/2504.07461v1)

**Authors**: Yiting Zhang, Yijiang Li, Tianwei Zhao, Kaijie Zhu, Haohan Wang, Nuno Vasconcelos

**Abstract**: Multi-agent system (MAS) has demonstrated exceptional capabilities in addressing complex challenges, largely due to the integration of multiple large language models (LLMs). However, the heterogeneity of LLMs, the scalability of quantities of LLMs, and local computational constraints pose significant challenges to hosting these models locally. To address these issues, we propose a new framework termed Distributed Multi-Agent System (DMAS). In DMAS, heterogeneous third-party agents function as service providers managed remotely by a central MAS server and each agent offers its services through API interfaces. However, the distributed nature of DMAS introduces several concerns about trustworthiness. In this paper, we study the Achilles heel of distributed multi-agent systems, identifying four critical trustworthiness challenges: free riding, susceptibility to malicious attacks, communication inefficiencies, and system instability. Extensive experiments across seven frameworks and four datasets reveal significant vulnerabilities of the DMAS. These attack strategies can lead to a performance degradation of up to 80% and attain a 100% success rate in executing free riding and malicious attacks. We envision our work will serve as a useful red-teaming tool for evaluating future multi-agent systems and spark further research on trustworthiness challenges in distributed multi-agent systems.

摘要: 多代理系统（MAS）在应对复杂挑战方面表现出了卓越的能力，这主要归功于多个大型语言模型（LLM）的集成。然而，LLM的多样性、LLM数量的可扩展性以及本地计算限制对本地托管这些模型构成了重大挑战。为了解决这些问题，我们提出了一个名为分布式多代理系统（DMAS）的新框架。在DMAS中，异类第三方代理充当由中央MAS服务器远程管理的服务提供商，每个代理通过API接口提供服务。然而，DMAS的分布式特性引入了一些关于可信度的问题。在本文中，我们研究了分布式多智能体系统的致命弱点，确定了四个关键的可信性挑战：搭便车，易受恶意攻击，通信效率低下，系统不稳定。在七个框架和四个数据集上进行的广泛实验揭示了DMAS的重大漏洞。这些攻击策略可以导致高达80%的性能下降，并在执行搭便车和恶意攻击时达到100%的成功率。我们设想我们的工作将成为评估未来多代理系统的有用的红色团队工具，并引发对分布式多代理系统可信度挑战的进一步研究。



## **9. Code Generation with Small Language Models: A Deep Evaluation on Codeforces**

使用小语言模型的代码生成：对代码力量的深入评估 cs.SE

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2504.07343v1) [paper-pdf](http://arxiv.org/pdf/2504.07343v1)

**Authors**: Débora Souza, Rohit Gheyi, Lucas Albuquerque, Gustavo Soares, Márcio Ribeiro

**Abstract**: Large Language Models (LLMs) have demonstrated capabilities in code generation, potentially boosting developer productivity. However, their widespread adoption remains limited by high computational costs, significant energy demands, and security risks such as data leakage and adversarial attacks. As a lighter-weight alternative, Small Language Models (SLMs) offer faster inference, lower deployment overhead, and better adaptability to domain-specific tasks, making them an attractive option for real-world applications. While prior research has benchmarked LLMs on competitive programming tasks, such evaluations often focus narrowly on metrics like Elo scores or pass rates, overlooking deeper insights into model behavior, failure patterns, and problem diversity. Furthermore, the potential of SLMs to tackle complex tasks such as competitive programming remains underexplored. In this study, we benchmark five open SLMs - LLAMA 3.2 3B, GEMMA 2 9B, GEMMA 3 12B, DEEPSEEK-R1 14B, and PHI-4 14B - across 280 Codeforces problems spanning Elo ratings from 800 to 2100 and covering 36 distinct topics. All models were tasked with generating Python solutions. PHI-4 14B achieved the best performance among SLMs, with a pass@3 of 63.6%, approaching the proprietary O3-MINI-HIGH (86.8%). In addition, we evaluated PHI-4 14B on C++ and found that combining outputs from both Python and C++ increases its aggregated pass@3 to 73.6%. A qualitative analysis of PHI-4 14B's incorrect outputs revealed that some failures were due to minor implementation issues - such as handling edge cases or correcting variable initialization - rather than deeper reasoning flaws.

摘要: 大型语言模型（LLM）已经展示了代码生成的能力，这可能会提高开发人员的生产力。然而，它们的广泛采用仍然受到高计算成本、高能源需求以及数据泄露和对抗性攻击等安全风险的限制。作为一种轻量级的替代方案，小型语言模型（SLC）提供更快的推理、更低的部署负担以及对特定领域任务的更好的适应性，使其成为现实世界应用程序的有吸引力的选择。虽然之前的研究已经根据竞争性编程任务对LLM进行了基准测试，但此类评估通常狭隘地关注Elo分数或通过率等指标，而忽视了对模型行为、失败模式和问题多样性的更深入见解。此外，Slms解决竞争性编程等复杂任务的潜力仍然没有得到充分的开发。在这项研究中，我们对五个开放的LM进行了基准测试- LLAMA 3.2 3B、GEGMA 2 9 B、GEGMA 3 12 B、DEEPSEEK-R1 14 B和PHI-4 14 B-跨越280个Codeforce问题，涵盖Elo评级从800到2100，涵盖36个不同的主题。所有模型的任务都是生成Python解决方案。PHI-4 14 B实现了SLS中最好的性能，通过率@3为63.6%，接近专有的O3-MINI-HIGH（86.8%）。此外，我们在C++上评估了PHI-4 14 B，发现结合Python和C++的输出可以将其总通过率@3增加到73.6%。对PHI-4 14 B错误输出的定性分析显示，一些失败是由于小的实现问题（例如处理边缘情况或纠正变量初始化）而不是更深层次的推理缺陷。



## **10. LLM Safeguard is a Double-Edged Sword: Exploiting False Positives for Denial-of-Service Attacks**

LLM保障是一把双刃剑：利用假阳性进行拒绝服务攻击 cs.CR

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2410.02916v3) [paper-pdf](http://arxiv.org/pdf/2410.02916v3)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern for large language models (LLMs) in open deployment, motivating the development of safeguard methods that enforce ethical and responsible use through safety alignment or guardrail mechanisms. Jailbreak attacks that exploit the \emph{false negatives} of safeguard methods have emerged as a prominent research focus in the field of LLM security. However, we found that the malicious attackers could also exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a denial-of-service (DoS) affecting LLM users. To bridge the knowledge gap of this overlooked threat, we explore multiple attack methods that include inserting a short adversarial prompt into user prompt templates and corrupting the LLM on the server by poisoned fine-tuning. In both ways, the attack triggers safeguard rejections of user requests from the client. Our evaluation demonstrates the severity of this threat across multiple scenarios. For instance, in the scenario of white-box adversarial prompt injection, the attacker can use our optimization process to automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97% of user requests on Llama Guard 3. These findings reveal a new dimension in LLM safeguard evaluation -- adversarial robustness to false positives.

摘要: 安全性是开放部署中大型语言模型（LLM）的首要问题，这促使开发了通过安全对齐或护栏机制来执行道德和负责任使用的保障方法。利用安全措施的\“假阴性\”的越狱攻击已经成为LLM安全领域的一个突出的研究热点。然而，我们发现恶意攻击者也可以利用安全措施的误报，即，欺骗保护模型错误地阻止安全内容，导致影响LLM用户的拒绝服务（DoS）。为了弥合这个被忽视的威胁的知识差距，我们探索了多种攻击方法，包括在用户提示模板中插入简短的对抗提示，以及通过有毒微调损坏服务器上的LLM。通过这两种方式，攻击都会触发对客户端用户请求的安全拒绝。我们的评估表明了这种威胁在多种场景下的严重性。例如，在白盒对抗性提示注入的场景中，攻击者可以使用我们的优化过程自动生成看似安全的对抗性提示，大约只有30个字符长，普遍阻止Llama Guard 3上超过97%的用户请求。这些发现揭示了LLM保障评估的一个新维度--对假阳性的对抗稳健性。



## **11. Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups**

穿越兔子洞：LLM生成的针对心理健康群体的攻击叙事中的紧急偏见 cs.CL

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2504.06160v2) [paper-pdf](http://arxiv.org/pdf/2504.06160v2)

**Authors**: Rijul Magu, Arka Dutta, Sean Kim, Ashiqur R. KhudaBukhsh, Munmun De Choudhury

**Abstract**: Large Language Models (LLMs) have been shown to demonstrate imbalanced biases against certain groups. However, the study of unprovoked targeted attacks by LLMs towards at-risk populations remains underexplored. Our paper presents three novel contributions: (1) the explicit evaluation of LLM-generated attacks on highly vulnerable mental health groups; (2) a network-based framework to study the propagation of relative biases; and (3) an assessment of the relative degree of stigmatization that emerges from these attacks. Our analysis of a recently released large-scale bias audit dataset reveals that mental health entities occupy central positions within attack narrative networks, as revealed by a significantly higher mean centrality of closeness (p-value = 4.06e-10) and dense clustering (Gini coefficient = 0.7). Drawing from sociological foundations of stigmatization theory, our stigmatization analysis indicates increased labeling components for mental health disorder-related targets relative to initial targets in generation chains. Taken together, these insights shed light on the structural predilections of large language models to heighten harmful discourse and highlight the need for suitable approaches for mitigation.

摘要: 事实证明，大型语言模型（LLM）对某些群体表现出不平衡的偏见。然而，关于LLM对高危人群进行无端有针对性攻击的研究仍然没有得到充分的研究。我们的论文提出了三个新颖的贡献：（1）对LLM产生的对高度弱势心理健康群体的攻击的明确评估;（2）基于网络的框架来研究相对偏见的传播;（3）对这些攻击中出现的耻辱的相对程度的评估。我们对最近发布的大规模偏见审计数据集的分析表明，心理健康实体在攻击叙事网络中占据了中心位置，这一点表现为密切度（p值= 4.06e-10）和密集聚集度（基尼系数= 0.7）的平均中心性显着更高。根据污名化理论的社会学基础，我们的污名化分析表明，相对于代际链中的初始目标，与心理健康疾病相关目标的标签成分有所增加。总而言之，这些见解揭示了大型语言模型加剧有害话语的结构偏好，并强调了适当的缓解方法的必要性。



## **12. JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model**

JailDAM：使用视觉语言模型的自适应记忆的越狱检测 cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.03770v2) [paper-pdf](http://arxiv.org/pdf/2504.03770v2)

**Authors**: Yi Nian, Shenzhe Zhu, Yuehan Qin, Li Li, Ziyi Wang, Chaowei Xiao, Yue Zhao

**Abstract**: Multimodal large language models (MLLMs) excel in vision-language tasks but also pose significant risks of generating harmful content, particularly through jailbreak attacks. Jailbreak attacks refer to intentional manipulations that bypass safety mechanisms in models, leading to the generation of inappropriate or unsafe content. Detecting such attacks is critical to ensuring the responsible deployment of MLLMs. Existing jailbreak detection methods face three primary challenges: (1) Many rely on model hidden states or gradients, limiting their applicability to white-box models, where the internal workings of the model are accessible; (2) They involve high computational overhead from uncertainty-based analysis, which limits real-time detection, and (3) They require fully labeled harmful datasets, which are often scarce in real-world settings. To address these issues, we introduce a test-time adaptive framework called JAILDAM. Our method leverages a memory-based approach guided by policy-driven unsafe knowledge representations, eliminating the need for explicit exposure to harmful data. By dynamically updating unsafe knowledge during test-time, our framework improves generalization to unseen jailbreak strategies while maintaining efficiency. Experiments on multiple VLM jailbreak benchmarks demonstrate that JAILDAM delivers state-of-the-art performance in harmful content detection, improving both accuracy and speed.

摘要: 多模式大型语言模型（MLLM）在视觉语言任务中表现出色，但也存在生成有害内容的巨大风险，特别是通过越狱攻击。越狱攻击是指绕过模型中安全机制的故意操纵，导致生成不适当或不安全的内容。检测此类攻击对于确保负责任地部署MLLM至关重要。现有的越狱检测方法面临三个主要挑战：（1）许多方法依赖于模型隐藏状态或梯度，限制了其对白盒模型的适用性，而白盒模型的内部工作是可以访问的;（2）它们涉及基于不确定性的分析的高计算负担，这限制了实时检测，以及（3）它们需要完全标记的有害数据集，这在现实世界中通常是稀缺的。为了解决这些问题，我们引入了一个名为JAILDAM的测试时自适应框架。我们的方法利用了基于内存的方法，该方法由政策驱动的不安全知识表示指导，消除了显式暴露有害数据的需要。通过在测试期间动态更新不安全知识，我们的框架提高了对未见越狱策略的概括性，同时保持效率。多个VLM越狱基准测试的实验表明，JAILDAM在有害内容检测方面提供了最先进的性能，提高了准确性和速度。



## **13. StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization**

StealthRank：通过StealthPropriation优化进行LLM排名操纵 cs.IR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05804v1) [paper-pdf](http://arxiv.org/pdf/2504.05804v1)

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems.

摘要: 将大型语言模型（LLM）集成到信息检索系统中引入了新的攻击表面，特别是对于对抗性排名操纵。我们介绍了StealthRank，这是一种新型的对抗性排名攻击，它可以操纵LLM驱动的产品推荐系统，同时保持文本流畅性和隐蔽性。与经常引入可检测异常的现有方法不同，StealthRank采用基于能量的优化框架与Langevin动态相结合来生成StealthRank脚本（SPP）-嵌入产品描述中的对抗性文本序列，微妙而有效地影响LLM排名机制。我们在多个LLM中评估StealthRank，证明其能够秘密提高目标产品的排名，同时避免容易检测到的显式操纵痕迹。我们的结果表明，StealthRank在有效性和隐蔽性方面始终优于最先进的对抗排名基线，凸显了LLM驱动的推荐系统中的关键漏洞。



## **14. Separator Injection Attack: Uncovering Dialogue Biases in Large Language Models Caused by Role Separators**

分隔符注入攻击：揭露角色分隔符引起的大型语言模型中的对话偏见 cs.CL

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05689v1) [paper-pdf](http://arxiv.org/pdf/2504.05689v1)

**Authors**: Xitao Li, Haijun Wang, Jiang Wu, Ting Liu

**Abstract**: Conversational large language models (LLMs) have gained widespread attention due to their instruction-following capabilities. To ensure conversational LLMs follow instructions, role separators are employed to distinguish between different participants in a conversation. However, incorporating role separators introduces potential vulnerabilities. Misusing roles can lead to prompt injection attacks, which can easily misalign the model's behavior with the user's intentions, raising significant security concerns. Although various prompt injection attacks have been proposed, recent research has largely overlooked the impact of role separators on safety. This highlights the critical need to thoroughly understand the systemic weaknesses in dialogue systems caused by role separators. This paper identifies modeling weaknesses caused by role separators. Specifically, we observe a strong positional bias associated with role separators, which is inherent in the format of dialogue modeling and can be triggered by the insertion of role separators. We further develop the Separators Injection Attack (SIA), a new orthometric attack based on role separators. The experiment results show that SIA is efficient and extensive in manipulating model behavior with an average gain of 18.2% for manual methods and enhances the attack success rate to 100% with automatic methods.

摘要: 对话式大型语言模型（LLM）因其描述跟踪能力而受到广泛关注。为了确保对话LLM遵循说明，使用角色分隔符来区分对话中的不同参与者。然而，合并角色分隔符会带来潜在的漏洞。滥用角色可能会导致即时注入攻击，这很容易使模型的行为与用户的意图不一致，从而引发严重的安全问题。尽管人们提出了各种即时注射攻击，但最近的研究在很大程度上忽视了角色分离器对安全性的影响。这凸显了彻底了解角色分离造成的对话系统系统性弱点的迫切需要。本文指出了角色分隔符导致的建模弱点。具体来说，我们观察到与角色分隔符相关的强烈位置偏见，这是对话建模格式中固有的，可以通过插入角色分隔符来触发。我们进一步开发了分离器注入攻击（SIA），这是一种基于角色分离器的新的正向攻击。实验结果表明，SIA在操纵模型行为方面高效且广泛，手动方法的平均收益率为18.2%，自动方法的攻击成功率提高至100%。



## **15. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

糖衣毒药：良性一代解锁法学硕士越狱 cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05652v1) [paper-pdf](http://arxiv.org/pdf/2504.05652v1)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Jie-Zhang

**Abstract**: Large Language Models (LLMs) have become increasingly integral to a wide range of applications. However, they still remain the threat of jailbreak attacks, where attackers manipulate designed prompts to make the models elicit malicious outputs. Analyzing jailbreak methods can help us delve into the weakness of LLMs and improve it. In this paper, We reveal a vulnerability in large language models (LLMs), which we term Defense Threshold Decay (DTD), by analyzing the attention weights of the model's output on input and subsequent output on prior output: as the model generates substantial benign content, its attention weights shift from the input to prior output, making it more susceptible to jailbreak attacks. To demonstrate the exploitability of DTD, we propose a novel jailbreak attack method, Sugar-Coated Poison (SCP), which induces the model to generate substantial benign content through benign input and adversarial reasoning, subsequently producing malicious content. To mitigate such attacks, we introduce a simple yet effective defense strategy, POSD, which significantly reduces jailbreak success rates while preserving the model's generalization capabilities.

摘要: 大型语言模型（LLM）已经成为越来越广泛的应用程序的组成部分。然而，它们仍然是越狱攻击的威胁，攻击者操纵设计的提示，使模型引发恶意输出。分析越狱方法可以帮助我们深入研究LLM的弱点并对其进行改进。本文通过分析模型的输出对输入和后续输出对先前输出的注意力权重，揭示了大型语言模型（LLM）中的一个漏洞，我们称之为防御阈值衰减（DTD）：当模型生成大量良性内容时，其注意力权重从输入转移到先前输出，使其更容易受到越狱攻击。为了证明DTD的可利用性，我们提出了一种新的越狱攻击方法，糖衣毒药（SCP），它诱导模型通过良性输入和对抗性推理生成大量良性内容，随后产生恶意内容。为了减轻这种攻击，我们引入了一种简单而有效的防御策略POSD，它可以显着降低越狱成功率，同时保留模型的泛化能力。



## **16. SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments**

SceneRAP：针对现实世界环境中视觉语言模型的场景一致印刷对抗规划器 cs.CV

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2412.00114v2) [paper-pdf](http://arxiv.org/pdf/2412.00114v2)

**Authors**: Yue Cao, Yun Xing, Jie Zhang, Di Lin, Tianwei Zhang, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: Large vision-language models (LVLMs) have shown remarkable capabilities in interpreting visual content. While existing works demonstrate these models' vulnerability to deliberately placed adversarial texts, such texts are often easily identifiable as anomalous. In this paper, we present the first approach to generate scene-coherent typographic adversarial attacks that mislead advanced LVLMs while maintaining visual naturalness through the capability of the LLM-based agent. Our approach addresses three critical questions: what adversarial text to generate, where to place it within the scene, and how to integrate it seamlessly. We propose a training-free, multi-modal LLM-driven scene-coherent typographic adversarial planning (SceneTAP) that employs a three-stage process: scene understanding, adversarial planning, and seamless integration. The SceneTAP utilizes chain-of-thought reasoning to comprehend the scene, formulate effective adversarial text, strategically plan its placement, and provide detailed instructions for natural integration within the image. This is followed by a scene-coherent TextDiffuser that executes the attack using a local diffusion mechanism. We extend our method to real-world scenarios by printing and placing generated patches in physical environments, demonstrating its practical implications. Extensive experiments show that our scene-coherent adversarial text successfully misleads state-of-the-art LVLMs, including ChatGPT-4o, even after capturing new images of physical setups. Our evaluations demonstrate a significant increase in attack success rates while maintaining visual naturalness and contextual appropriateness. This work highlights vulnerabilities in current vision-language models to sophisticated, scene-coherent adversarial attacks and provides insights into potential defense mechanisms.

摘要: 大型视觉语言模型（LVLM）在解释视觉内容方面表现出了非凡的能力。虽然现有的作品证明了这些模型对故意放置的对抗文本的脆弱性，但此类文本通常很容易被识别为异常文本。在本文中，我们提出了第一种生成场景一致印刷对抗攻击的方法，这种攻击可以误导高级LVLM，同时通过基于LLM的代理的能力保持视觉自然性。我们的方法解决了三个关键问题：生成什么对抗文本、将其放置在场景中的位置以及如何无缝集成它。我们提出了一种免培训、多模式LLM驱动的场景一致印刷对抗性规划（SceneRAP），该规划采用三阶段流程：场景理解、对抗性规划和无缝集成。SceneRAP利用思想链推理来理解场景、制定有效的对抗文本、战略性地规划其放置，并为图像中的自然整合提供详细的说明。随后是场景一致的文本扩散用户，它使用本地扩散机制执行攻击。我们通过打印并将生成的补丁放置在物理环境中，将我们的方法扩展到现实世界场景，展示其实际含义。大量实验表明，即使在捕获物理设置的新图像之后，我们的场景连贯对抗文本也能成功误导最先进的LVLM，包括ChatGPT-4 o。我们的评估表明，攻击成功率显着提高，同时保持视觉自然性和上下文适当性。这项工作强调了当前视觉语言模型对复杂、场景一致的对抗攻击的脆弱性，并提供了对潜在防御机制的见解。



## **17. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

揭示一致大型语言模型内在的道德脆弱性 cs.CL

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05050v1) [paper-pdf](http://arxiv.org/pdf/2504.05050v1)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.

摘要: 大型语言模型（LLM）是人工通用智能的基础探索，但它们通过指令调整和偏好学习与人类价值观的一致只能实现表面的合规性。在这里，我们证明，预训练期间嵌入的有害知识在LLM参数记忆中作为不可磨灭的“黑暗模式”持续存在，逃避对齐保障措施，并在分布变化时的对抗诱导下重新浮出水面。在这项研究中，我们首先通过证明当前的对齐方法只产生知识集合中的局部“安全区域”来从理论上分析对齐LLM的内在道德脆弱性。相比之下，预先训练的知识仍然通过高可能性的对抗轨迹与有害概念保持全球联系。基于这一理论见解，我们通过在分布转移下采用语义一致诱导来从经验上验证我们的发现--一种通过优化的对抗提示系统性地绕过对齐约束的方法。这种理论和经验相结合的方法在23个最先进的对齐LLM中的19个（包括DeepSeek-R1和LLaMA-3）上实现了100%的攻击成功率，揭示了它们的普遍漏洞。



## **18. A Domain-Based Taxonomy of Jailbreak Vulnerabilities in Large Language Models**

大型语言模型中基于领域的越狱漏洞分类 cs.CL

21 pages, 5 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04976v1) [paper-pdf](http://arxiv.org/pdf/2504.04976v1)

**Authors**: Carlos Peláez-González, Andrés Herrera-Poyatos, Cristina Zuheros, David Herrera-Poyatos, Virilo Tejedor, Francisco Herrera

**Abstract**: The study of large language models (LLMs) is a key area in open-world machine learning. Although LLMs demonstrate remarkable natural language processing capabilities, they also face several challenges, including consistency issues, hallucinations, and jailbreak vulnerabilities. Jailbreaking refers to the crafting of prompts that bypass alignment safeguards, leading to unsafe outputs that compromise the integrity of LLMs. This work specifically focuses on the challenge of jailbreak vulnerabilities and introduces a novel taxonomy of jailbreak attacks grounded in the training domains of LLMs. It characterizes alignment failures through generalization, objectives, and robustness gaps. Our primary contribution is a perspective on jailbreak, framed through the different linguistic domains that emerge during LLM training and alignment. This viewpoint highlights the limitations of existing approaches and enables us to classify jailbreak attacks on the basis of the underlying model deficiencies they exploit. Unlike conventional classifications that categorize attacks based on prompt construction methods (e.g., prompt templating), our approach provides a deeper understanding of LLM behavior. We introduce a taxonomy with four categories -- mismatched generalization, competing objectives, adversarial robustness, and mixed attacks -- offering insights into the fundamental nature of jailbreak vulnerabilities. Finally, we present key lessons derived from this taxonomic study.

摘要: 大型语言模型（LLM）的研究是开放世界机器学习的一个关键领域。尽管LLM表现出出色的自然语言处理能力，但它们也面临着一些挑战，包括一致性问题、幻觉和越狱漏洞。越狱是指绕过对齐保障措施的提示，导致不安全的输出，从而损害LLM的完整性。这项工作特别关注越狱漏洞的挑战，并引入了一种基于LLM训练领域的新颖越狱攻击分类法。它通过概括性、目标和稳健性差距来描述对齐失败。我们的主要贡献是对越狱的看法，通过LLM培训和调整期间出现的不同语言领域来框架。这一观点强调了现有方法的局限性，并使我们能够根据越狱攻击所利用的基础模型缺陷对越狱攻击进行分类。与基于即时构建方法对攻击进行分类的传统分类不同（例如，提示模板），我们的方法提供了一个更深入的了解LLM行为。我们引入了一个分类法，分为四个类别-不匹配的泛化，竞争目标，对抗性鲁棒性和混合攻击-提供了对越狱漏洞的基本性质的见解。最后，我们提出了从这一分类学研究中得出的关键教训。



## **19. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

不要落后，RAG：使用RAG进行免训练对抗检测 cs.AI

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04858v1) [paper-pdf](http://arxiv.org/pdf/2504.04858v1)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.

摘要: 对抗性补丁攻击通过嵌入误导深度模型的局部扰动，对视觉系统构成重大威胁。传统的防御方法通常需要重新培训或微调，这使得它们对于现实世界的部署来说不切实际。我们提出了一个免训练的视觉检索增强生成（VRAG）框架，该框架集成了用于对抗性补丁检测的视觉语言模型（VLM）。通过检索视觉上相似的补丁和图像，这些补丁和图像类似于不断扩展的数据库中存储的攻击，VRAG执行生成式推理以识别不同的攻击类型，而所有这些都无需额外的训练或微调。我们广泛评估了开源大型VLM，包括Qwen-VL-Plus、Qwen2.5-VL-72 B和UI-TARS-72 B-DPO，以及Gemini-2.0（一种闭源模型）。值得注意的是，开源UI-TARS-72 B-DPO模型实现了高达95%的分类准确率，为开源对抗补丁检测奠定了新的最新水平。Gemini-2.0的总体准确率达到了最高的98%，但仍然是闭源的。实验结果证明了VRAG在以最少的人类注释识别各种对抗补丁方面的有效性，为针对不断发展的对抗补丁攻击的稳健、实用的防御铺平了道路。



## **20. SINCon: Mitigate LLM-Generated Malicious Message Injection Attack for Rumor Detection**

SINCon：缓解LLM生成的恶意消息注入攻击以进行谣言检测 cs.CR

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.07135v1) [paper-pdf](http://arxiv.org/pdf/2504.07135v1)

**Authors**: Mingqing Zhang, Qiang Liu, Xiang Tao, Shu Wu, Liang Wang

**Abstract**: In the era of rapidly evolving large language models (LLMs), state-of-the-art rumor detection systems, particularly those based on Message Propagation Trees (MPTs), which represent a conversation tree with the post as its root and the replies as its descendants, are facing increasing threats from adversarial attacks that leverage LLMs to generate and inject malicious messages. Existing methods are based on the assumption that different nodes exhibit varying degrees of influence on predictions. They define nodes with high predictive influence as important nodes and target them for attacks. If the model treats nodes' predictive influence more uniformly, attackers will find it harder to target high predictive influence nodes. In this paper, we propose Similarizing the predictive Influence of Nodes with Contrastive Learning (SINCon), a defense mechanism that encourages the model to learn graph representations where nodes with varying importance have a more uniform influence on predictions. Extensive experiments on the Twitter and Weibo datasets demonstrate that SINCon not only preserves high classification accuracy on clean data but also significantly enhances resistance against LLM-driven message injection attacks.

摘要: 在大型语言模型（LLM）快速发展的时代，最先进的谣言检测系统，特别是基于消息传播树（MPTS）的谣言检测系统，其代表以帖子为根、以回复为后代的对话树，正面临着越来越多的威胁。利用LLM来生成和注入恶意消息的对抗性攻击。现有的方法基于这样的假设：不同的节点对预测表现出不同程度的影响。他们将具有高预测影响力的节点定义为重要节点，并针对它们进行攻击。如果模型更均匀地对待节点的预测影响力，攻击者将发现更难瞄准高预测影响力的节点。在本文中，我们提出用对比学习来模拟节点的预测影响（SINCon），这是一种防御机制，鼓励模型学习图表示，其中重要性不同的节点对预测产生更均匀的影响。Twitter和微博数据集上的大量实验表明，SINCon不仅在干净数据上保持了高分类准确性，而且还显着增强了对LLM驱动的消息注入攻击的抵抗力。



## **21. PiCo: Jailbreaking Multimodal Large Language Models via $\textbf{Pi}$ctorial $\textbf{Co}$de Contextualization**

PiCo：通过$\textBF{Pi}$ctorial $\textBF{Co}$de上下文化破解多模式大型语言模型 cs.CR

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.01444v2) [paper-pdf](http://arxiv.org/pdf/2504.01444v2)

**Authors**: Aofan Liu, Lulu Tang, Ting Pan, Yuguo Yin, Bin Wang, Ao Yang

**Abstract**: Multimodal Large Language Models (MLLMs), which integrate vision and other modalities into Large Language Models (LLMs), significantly enhance AI capabilities but also introduce new security vulnerabilities. By exploiting the vulnerabilities of the visual modality and the long-tail distribution characteristic of code training data, we present PiCo, a novel jailbreaking framework designed to progressively bypass multi-tiered defense mechanisms in advanced MLLMs. PiCo employs a tier-by-tier jailbreak strategy, using token-level typographic attacks to evade input filtering and embedding harmful intent within programming context instructions to bypass runtime monitoring. To comprehensively assess the impact of attacks, a new evaluation metric is further proposed to assess both the toxicity and helpfulness of model outputs post-attack. By embedding harmful intent within code-style visual instructions, PiCo achieves an average Attack Success Rate (ASR) of 84.13% on Gemini-Pro Vision and 52.66% on GPT-4, surpassing previous methods. Experimental results highlight the critical gaps in current defenses, underscoring the need for more robust strategies to secure advanced MLLMs.

摘要: 多模式大型语言模型（MLLM）将视觉和其他模式集成到大型语言模型（LLM）中，显着增强了人工智能能力，但也引入了新的安全漏洞。通过利用视觉模式的漏洞和代码训练数据的长尾分布特征，我们提出了PiCo，这是一种新型越狱框架，旨在逐步绕过高级MLLM中的多层防御机制。PiCo采用逐层越狱策略，使用标记级印刷攻击来逃避输入过滤，并在编程上下文指令中嵌入有害意图以绕过运行时监控。为了全面评估攻击的影响，进一步提出了一种新的评估指标来评估攻击后模型输出的毒性和帮助性。通过在代码风格的视觉指令中嵌入有害意图，PiCo在Gemini-Pro Vision上实现了84.13%的平均攻击成功率（ASB），在GPT-4上实现了52.66%的平均攻击成功率（ASB），超过了之前的方法。实验结果凸显了当前防御中的关键差距，强调需要更稳健的策略来保护高级MLLM。



## **22. Select Me! When You Need a Tool: A Black-box Text Attack on Tool Selection**

选择我！当您需要工具时：对工具选择的黑匣子文本攻击 cs.CR

13 pages

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04809v1) [paper-pdf](http://arxiv.org/pdf/2504.04809v1)

**Authors**: Liuji Chen, Hao Gao, Jinghao Zhang, Qiang Liu, Shu Wu, Liang Wang

**Abstract**: Tool learning serves as a powerful auxiliary mechanism that extends the capabilities of large language models (LLMs), enabling them to tackle complex tasks requiring real-time relevance or high precision operations. Behind its powerful capabilities lie some potential security issues. However, previous work has primarily focused on how to make the output of the invoked tools incorrect or malicious, with little attention given to the manipulation of tool selection. To fill this gap, we introduce, for the first time, a black-box text-based attack that can significantly increase the probability of the target tool being selected in this paper. We propose a two-level text perturbation attack witha coarse-to-fine granularity, attacking the text at both the word level and the character level. We conduct comprehensive experiments that demonstrate the attacker only needs to make some perturbations to the tool's textual information to significantly increase the possibility of the target tool being selected and ranked higher among the candidate tools. Our research reveals the vulnerability of the tool selection process and paves the way for future research on protecting this process.

摘要: 工具学习是一种强大的辅助机制，可以扩展大型语言模型（LLM）的能力，使它们能够处理需要实时相关性或高精度操作的复杂任务。其强大的能力背后隐藏着一些潜在的安全问题。然而，之前的工作主要集中在如何使所调用工具的输出不正确或恶意，很少关注工具选择的操纵。为了填补这一空白，我们首次引入了基于黑匣子文本的攻击，该攻击可以显着增加本文中选择目标工具的可能性。我们提出了一种从粗到细粒度的两级文本扰动攻击，在单词级和字符级攻击文本。我们进行了全面的实验，证明攻击者只需要对工具的文本信息进行一些干扰，就可以显着增加目标工具被选择并在候选工具中排名更高的可能性。我们的研究揭示了工具选择过程的脆弱性，并为未来保护这一过程的研究铺平了道路。



## **23. Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs**

你得到了你所付出的代价吗？LLM API中的审计模型替代 cs.CL

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04715v1) [paper-pdf](http://arxiv.org/pdf/2504.04715v1)

**Authors**: Will Cai, Tianneng Shi, Xuandong Zhao, Dawn Song

**Abstract**: The proliferation of Large Language Models (LLMs) accessed via black-box APIs introduces a significant trust challenge: users pay for services based on advertised model capabilities (e.g., size, performance), but providers may covertly substitute the specified model with a cheaper, lower-quality alternative to reduce operational costs. This lack of transparency undermines fairness, erodes trust, and complicates reliable benchmarking. Detecting such substitutions is difficult due to the black-box nature, typically limiting interaction to input-output queries. This paper formalizes the problem of model substitution detection in LLM APIs. We systematically evaluate existing verification techniques, including output-based statistical tests, benchmark evaluations, and log probability analysis, under various realistic attack scenarios like model quantization, randomized substitution, and benchmark evasion. Our findings reveal the limitations of methods relying solely on text outputs, especially against subtle or adaptive attacks. While log probability analysis offers stronger guarantees when available, its accessibility is often limited. We conclude by discussing the potential of hardware-based solutions like Trusted Execution Environments (TEEs) as a pathway towards provable model integrity, highlighting the trade-offs between security, performance, and provider adoption. Code is available at https://github.com/sunblaze-ucb/llm-api-audit

摘要: 通过黑盒API访问的大型语言模型（LLM）的激增引入了一个重大的信任挑战：用户基于广告的模型功能（例如，尺寸、性能），但供应商可能会秘密地用更便宜、质量更低的替代品来替代指定的型号，以降低运营成本。缺乏透明度破坏了公平性，侵蚀了信任，并使可靠的基准变得复杂。由于黑盒性质，检测这样的替换是困难的，通常将交互限制为输入输出查询。本文形式化了LLM API中的模型替代检测问题。我们在模型量化、随机替换和基准规避等各种现实攻击场景下系统地评估现有的验证技术，包括基于输出的统计测试、基准评估和日志概率分析。我们的研究结果揭示了仅依赖文本输出的方法的局限性，尤其是针对微妙或适应性攻击。虽然日志概率分析在可用时提供了更强的保证，但其可访问性通常受到限制。最后，我们讨论了可信执行环境（TEE）等基于硬件的解决方案作为可证明模型完整性途径的潜力，强调了安全性、性能和提供商采用之间的权衡。代码可在https://github.com/sunblaze-ucb/llm-api-audit上获得



## **24. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks**

保护视觉语言模型：缓解基于扰动的攻击中高斯噪音的脆弱性 cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.01308v2) [paper-pdf](http://arxiv.org/pdf/2504.01308v2)

**Authors**: Jiawei Wang, Yushen Zuo, Yuanjun Chai, Zhendong Liu, Yicheng Fu, Yichun Feng, Kin-Man Lam

**Abstract**: Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.

摘要: 视觉语言模型（VLMS）通过合并视觉信息扩展了大型语言模型（LLM）的功能，但它们仍然容易受到越狱攻击，尤其是在处理嘈杂或损坏的图像时。尽管现有的VLM在培训期间采取安全措施来减轻此类攻击，但与噪音增强视觉输入相关的漏洞被忽视了。在这项工作中，我们发现错过噪音增强训练会导致严重的安全漏洞：许多VLM甚至容易受到高斯噪音等简单扰动的影响。为了应对这一挑战，我们提出了Robust-VLGuard，这是一个具有对齐/未对齐图像-文本对的多模式安全数据集，结合了噪音增强微调，可以降低攻击成功率，同时保留VLM的功能。对于更强的基于优化的视觉扰动攻击，我们提出了DiffPure-VLM，利用扩散模型将对抗性扰动转换为类高斯噪声，可以通过具有噪声增强安全微调的VLM进行防御。实验结果表明，扩散模型的分布偏移特性与我们微调的VLM很好地吻合，显著减轻了不同强度的对抗性扰动。数据集和代码可在https://github.com/JarvisUSTC/DiffPure-RobustVLM上获取。



## **25. Privacy in Fine-tuning Large Language Models: Attacks, Defenses, and Future Directions**

微调大型语言模型中的隐私：攻击、防御和未来方向 cs.AI

accepted by PAKDD2025

**SubmitDate**: 2025-04-06    [abs](http://arxiv.org/abs/2412.16504v2) [paper-pdf](http://arxiv.org/pdf/2412.16504v2)

**Authors**: Hao Du, Shang Liu, Lele Zheng, Yang Cao, Atsuyoshi Nakamura, Lei Chen

**Abstract**: Fine-tuning has emerged as a critical process in leveraging Large Language Models (LLMs) for specific downstream tasks, enabling these models to achieve state-of-the-art performance across various domains. However, the fine-tuning process often involves sensitive datasets, introducing privacy risks that exploit the unique characteristics of this stage. In this paper, we provide a comprehensive survey of privacy challenges associated with fine-tuning LLMs, highlighting vulnerabilities to various privacy attacks, including membership inference, data extraction, and backdoor attacks. We further review defense mechanisms designed to mitigate privacy risks in the fine-tuning phase, such as differential privacy, federated learning, and knowledge unlearning, discussing their effectiveness and limitations in addressing privacy risks and maintaining model utility. By identifying key gaps in existing research, we highlight challenges and propose directions to advance the development of privacy-preserving methods for fine-tuning LLMs, promoting their responsible use in diverse applications.

摘要: 微调已成为利用大型语言模型（LLM）执行特定下游任务的一个关键过程，使这些模型能够在各个领域实现最先进的性能。然而，微调过程通常涉及敏感数据集，从而引入利用该阶段独特特征的隐私风险。在本文中，我们对与微调LLM相关的隐私挑战进行了全面调查，重点介绍了各种隐私攻击的漏洞，包括成员资格推断、数据提取和后门攻击。我们进一步审查了旨在在微调阶段减轻隐私风险的防御机制，例如差异隐私、联邦学习和知识取消学习，讨论了它们在解决隐私风险和维护模型效用方面的有效性和局限性。通过确定现有研究中的关键差距，我们强调挑战并提出方向，以推进用于微调LLM的隐私保护方法的开发，促进其在不同应用中负责任地使用。



## **26. CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data**

CyberLLMDirecct：使用网络安全数据分析精调LLM安全性的新数据集 cs.CR

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2503.09334v2) [paper-pdf](http://arxiv.org/pdf/2503.09334v2)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. The dataset creation pipeline, along with comprehensive documentation, examples, and resources for reproducing our results, is publicly available at https://github.com/Adelsamir01/CyberLLMInstruct.

摘要: 将大型语言模型（LLM）集成到网络安全应用程序中带来了重大机会，例如增强威胁分析和恶意软件检测，但也可能带来关键风险和安全问题，包括个人数据泄露和新恶意软件的自动生成。为了应对这些挑战，我们开发了CyberLLMATION，这是一个由54，928个描述-响应对组成的数据集，涵盖网络安全任务，例如恶意软件分析、网络钓鱼模拟和零日漏洞。该数据集是通过多阶段过程构建的。这涉及从多个资源中获取数据，过滤并将其结构化为描述-响应对，并将其与现实世界场景对齐以增强其适用性。选择了七个开源LLM来测试CyberLLMInsurance的有用性：Phi 3 Mini 3.8B、Mistral 7 B、Qwen 2.5 7 B、Llama 3 8B、Llama 3.1 8B、Gemma 2 9 B和Llama 2 70 B。在我们的主要示例中，我们使用OWISP十大框架严格评估了微调模型的安全性，发现微调会降低所有测试的LLM和每次对抗攻击的安全弹性（例如，Llama 3.1 8B对立即注射的安全评分从0.95下降至0.15）。在我们的第二个例子中，我们表明这些相同的微调模型也可以在CyberMetric基准上实现高达92.50%的准确性。这些发现凸显了性能和安全性之间的权衡，表明了对抗性测试和进一步研究微调方法的重要性，这些方法可以降低安全风险，同时仍能提高不同数据集和领域的性能。数据集创建管道以及用于重现我们结果的全面文档、示例和资源可在https://github.com/Adelsamir01/CyberLLMInstruct上公开获取。



## **27. AttackLLM: LLM-based Attack Pattern Generation for an Industrial Control System**

AttackLLM：基于LLM的工业控制系统攻击模式生成 cs.CR

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2504.04187v1) [paper-pdf](http://arxiv.org/pdf/2504.04187v1)

**Authors**: Chuadhry Mujeeb Ahmed

**Abstract**: Malicious examples are crucial for evaluating the robustness of machine learning algorithms under attack, particularly in Industrial Control Systems (ICS). However, collecting normal and attack data in ICS environments is challenging due to the scarcity of testbeds and the high cost of human expertise. Existing datasets are often limited by the domain expertise of practitioners, making the process costly and inefficient. The lack of comprehensive attack pattern data poses a significant problem for developing robust anomaly detection methods. In this paper, we propose a novel approach that combines data-centric and design-centric methodologies to generate attack patterns using large language models (LLMs). Our results demonstrate that the attack patterns generated by LLMs not only surpass the quality and quantity of those created by human experts but also offer a scalable solution that does not rely on expensive testbeds or pre-existing attack examples. This multi-agent based approach presents a promising avenue for enhancing the security and resilience of ICS environments.

摘要: 恶意示例对于评估机器学习算法在攻击下的稳健性至关重要，特别是在工业控制系统（ICS）中。然而，由于测试床稀缺和人力专业知识成本高，在ICS环境中收集正常和攻击数据具有挑战性。现有的数据集通常受到从业者领域专业知识的限制，导致该过程成本高昂且效率低下。缺乏全面的攻击模式数据给开发稳健的异常检测方法带来了一个重大问题。在本文中，我们提出了一种新颖的方法，该方法结合以数据为中心和以设计为中心的方法来使用大型语言模型（LLM）生成攻击模式。我们的结果表明，LLM生成的攻击模式不仅超过了人类专家创建的攻击模式的质量和数量，而且还提供了一种可扩展的解决方案，不依赖于昂贵的测试床或预先存在的攻击示例。这种基于多代理的方法为增强ICS环境的安全性和弹性提供了一条有希望的途径。



## **28. Practical Poisoning Attacks against Retrieval-Augmented Generation**

对检索增广生成算法的实用中毒攻击 cs.CR

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2504.03957v1) [paper-pdf](http://arxiv.org/pdf/2504.03957v1)

**Authors**: Baolei Zhang, Yuxi Chen, Minghong Fang, Zhuqing Liu, Lihai Nie, Tong Li, Zheli Liu

**Abstract**: Large language models (LLMs) have demonstrated impressive natural language processing abilities but face challenges such as hallucination and outdated knowledge. Retrieval-Augmented Generation (RAG) has emerged as a state-of-the-art approach to mitigate these issues. While RAG enhances LLM outputs, it remains vulnerable to poisoning attacks. Recent studies show that injecting poisoned text into the knowledge database can compromise RAG systems, but most existing attacks assume that the attacker can insert a sufficient number of poisoned texts per query to outnumber correct-answer texts in retrieval, an assumption that is often unrealistic. To address this limitation, we propose CorruptRAG, a practical poisoning attack against RAG systems in which the attacker injects only a single poisoned text, enhancing both feasibility and stealth. Extensive experiments across multiple datasets demonstrate that CorruptRAG achieves higher attack success rates compared to existing baselines.

摘要: 大型语言模型（LLM）已展现出令人印象深刻的自然语言处理能力，但面临着幻觉和过时知识等挑战。检索增强一代（RAG）已成为缓解这些问题的最先进方法。虽然RAG增强了LLM输出，但它仍然容易受到中毒攻击。最近的研究表明，将有毒文本注入知识数据库可能会危及RAG系统，但大多数现有的攻击都假设攻击者可以在每个查询中插入足够数量的有毒文本，以超过检索中的正确答案文本，这一假设通常是不切实际的。为了解决这一限制，我们提出了CorruptRAG，这是一种针对RAG系统的实用中毒攻击，其中攻击者仅注入单个中毒文本，从而增强了可行性和隐蔽性。跨多个数据集的广泛实验表明，与现有基线相比，CorruptRAG的攻击成功率更高。



## **29. sudo rm -rf agentic_security**

sudo rm -ref agentic_secure cs.CL

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2503.20279v2) [paper-pdf](http://arxiv.org/pdf/2503.20279v2)

**Authors**: Sejin Lee, Jian Kim, Haon Park, Ashkan Yousefpour, Sangyoon Yu, Min Song

**Abstract**: Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2Tox Offense), a novel attack framework that systematically bypasses refusal trained safeguards in commercial computer-use agents, such as Claude Computer Use. The core mechanism, Detox2Tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24% (with no refinement), and up to 41% (by its iterative refinement) in Claude Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs Our code is available at: https://github.com/AIM-Intelligence/SUDO.git

摘要: 大型语言模型（LLM）越来越多地被部署为计算机使用代理，在真实桌面或Web环境中自主执行任务。虽然这种演变极大地扩展了人类的实际用例，但也造成了严重的安全风险。我们提出了SUDO（基于屏幕的通用Detox 2 Tox Offense），这是一种新颖的攻击框架，可以系统地绕过商业计算机使用代理（例如Claude Computer Use）中的拒绝训练保护措施。核心机制Detox 2Tox通过解毒将有害请求（代理最初拒绝的请求）转换为看似良性的请求，保护高级视觉语言模型（VLM）的详细指令，然后在执行前通过简化重新引入恶意内容。与传统的越狱不同，SUDO基于内置的拒绝反馈迭代改进其攻击，使其在对抗强大的政策过滤器时变得越来越有效。在涵盖50个现实世界任务和多个最先进的VLM的广泛测试中，SUDO的攻击成功率高达24%（无需改进），在Claude Computer Use中高达41%（通过迭代改进）。通过揭示这些漏洞并展示它们在现实世界计算环境中被利用的轻松性，本文强调了对强大的、上下文感知的保护措施的迫切需求。警告：本文包括有害或冒犯性的模型输出我们的代码可在：https://github.com/AIM-Intelligence/SUDO.git上获取



## **30. Les Dissonances: Cross-Tool Harvesting and Polluting in Multi-Tool Empowered LLM Agents**

Les Dissonance：多工具授权的LLM代理中的跨工具收获和污染 cs.CR

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2504.03111v1) [paper-pdf](http://arxiv.org/pdf/2504.03111v1)

**Authors**: Zichuan Li, Jian Cui, Xiaojing Liao, Luyi Xing

**Abstract**: Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 73 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 80% of the tools are vulnerable to hijacking attacks, 78% to XTH attacks, and 41% to XTP attacks, highlighting the prevalence of this threat.

摘要: 大型语言模型（LLM）代理是由LLM支持的自治系统，能够通过利用一组工具进行推理和规划来解决问题。然而，LLM代理中多工具功能的集成在安全管理工具、确保其兼容性、处理依赖关系以及保护LLM代理工作流程中的控制流方面带来了挑战。本文中，我们首次对支持多工具的LLM代理中的任务控制流进行了系统性安全分析。我们识别了一种新型威胁，即跨工具收获和污染（XTHP），它包括多个攻击载体，首先劫持代理任务的正常控制流，然后收集和污染LLM代理系统内的机密或私人信息。为了了解这种威胁的影响，我们开发了Chord，这是一种动态扫描工具，旨在自动检测容易受到XTHP攻击的现实世界代理工具。我们对两个主要LLM代理开发框架LangChain和LlamaIndex存储库中的73个现实工具进行了评估，发现了一个重大的安全问题：80%的工具容易受到劫持攻击，78%容易受到XTH攻击，41%容易受到XTP攻击，凸显了这种威胁的普遍性。



## **31. PROMPTFUZZ: Harnessing Fuzzing Techniques for Robust Testing of Prompt Injection in LLMs**

PROMPTFUZZ：利用模糊技术对LLM中的即时注射进行稳健测试 cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2409.14729v2) [paper-pdf](http://arxiv.org/pdf/2409.14729v2)

**Authors**: Jiahao Yu, Yangguang Shao, Hanwen Miao, Junzheng Shi

**Abstract**: Large Language Models (LLMs) have gained widespread use in various applications due to their powerful capability to generate human-like text. However, prompt injection attacks, which involve overwriting a model's original instructions with malicious prompts to manipulate the generated text, have raised significant concerns about the security and reliability of LLMs. Ensuring that LLMs are robust against such attacks is crucial for their deployment in real-world applications, particularly in critical tasks.   In this paper, we propose PROMPTFUZZ, a novel testing framework that leverages fuzzing techniques to systematically assess the robustness of LLMs against prompt injection attacks. Inspired by software fuzzing, PROMPTFUZZ selects promising seed prompts and generates a diverse set of prompt injections to evaluate the target LLM's resilience. PROMPTFUZZ operates in two stages: the prepare phase, which involves selecting promising initial seeds and collecting few-shot examples, and the focus phase, which uses the collected examples to generate diverse, high-quality prompt injections. Using PROMPTFUZZ, we can uncover more vulnerabilities in LLMs, even those with strong defense prompts.   By deploying the generated attack prompts from PROMPTFUZZ in a real-world competition, we achieved the 7th ranking out of over 4000 participants (top 0.14%) within 2 hours. Additionally, we construct a dataset to fine-tune LLMs for enhanced robustness against prompt injection attacks. While the fine-tuned model shows improved robustness, PROMPTFUZZ continues to identify vulnerabilities, highlighting the importance of robust testing for LLMs. Our work emphasizes the critical need for effective testing tools and provides a practical framework for evaluating and improving the robustness of LLMs against prompt injection attacks.

摘要: 大型语言模型（LLM）因其生成类人文本的强大能力而在各种应用程序中得到广泛使用。然而，提示注入攻击（涉及将模型的原始指令与恶意提示一起操作生成的文本）引发了人们对LLM安全性和可靠性的严重担忧。确保LLM能够强大地抵御此类攻击对于它们在现实世界应用程序中的部署至关重要，特别是在关键任务中。   在本文中，我们提出了PROMPTFUZZ，这是一种新型测试框架，它利用模糊技术来系统性评估LLM针对即时注入攻击的稳健性。受软件模糊化的启发，PROMPTFUZZ选择有希望的种子提示并生成一组不同的提示注入来评估目标LLM的弹性。PROMPTFUZZ分为两个阶段：准备阶段，涉及选择有希望的初始种子并收集少量样本，以及聚焦阶段，使用收集的样本生成多样化、高质量的提示注射。使用PROMPTFUZZ，我们可以发现LLM中的更多漏洞，即使是那些具有强大防御提示的LLM。   通过在现实世界的比赛中部署PROMPTFUZZ生成的攻击提示，我们在2小时内在4000多名参与者中获得了第7名（前0.14%）。此外，我们还构建了一个数据集来微调LLM，以增强针对即时注入攻击的鲁棒性。虽然微调的模型显示出更好的稳健性，但PROMPTFUZZ继续识别漏洞，强调了对LLM进行稳健测试的重要性。我们的工作强调了对有效测试工具的迫切需求，并提供了一个实用的框架来评估和改进LLM针对即时注入攻击的稳健性。



## **32. ERPO: Advancing Safety Alignment via Ex-Ante Reasoning Preference Optimization**

ERPO：通过前推理偏好优化推进安全一致 cs.CL

18 pages, 5 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02725v1) [paper-pdf](http://arxiv.org/pdf/2504.02725v1)

**Authors**: Kehua Feng, Keyan Ding, Jing Yu, Menghan Li, Yuhao Wang, Tong Xu, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose Ex-Ante Reasoning Preference Optimization (ERPO), a novel safety alignment framework that equips LLMs with explicit preemptive reasoning through Chain-of-Thought and provides clear evidence for safety judgments by embedding predefined safety rules. Specifically, our approach consists of three stages: first, equipping the model with Ex-Ante reasoning through supervised fine-tuning (SFT) using a constructed reasoning module; second, enhancing safety, usefulness, and efficiency via Direct Preference Optimization (DPO); and third, mitigating inference latency with a length-controlled iterative preference optimization strategy. Experiments on multiple open-source LLMs demonstrate that ERPO significantly enhances safety performance while maintaining response efficiency.

摘要: 大型语言模型（LLM）的最新进展加速了人工通用智能的发展，但它们生成有害内容的潜力带来了严重的安全挑战。现有的对齐方法通常难以覆盖各种安全场景，并且仍然容易受到对抗性攻击。在这项工作中，我们提出了前-Ante推理偏好优化（ERPO），一种新的安全对齐框架，通过思想链为LLM提供明确的抢先推理，并通过嵌入预定义的安全规则为安全判断提供明确的证据。具体来说，我们的方法包括三个阶段：第一，通过使用构造的推理模块进行监督微调（SFT），为模型配备Ex-Ante推理;第二，通过直接偏好优化（DPO）提高安全性，有用性和效率;第三，通过长度控制的迭代偏好优化策略减轻推理延迟。在多个开源LLM上的实验表明，ERPO显着增强了安全性能，同时保持了响应效率。



## **33. No Free Lunch with Guardrails**

没有带护栏的免费午餐 cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.00441v2) [paper-pdf](http://arxiv.org/pdf/2504.00441v2)

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: As large language models (LLMs) and generative AI become widely adopted, guardrails have emerged as a key tool to ensure their safe use. However, adding guardrails isn't without tradeoffs; stronger security measures can reduce usability, while more flexible systems may leave gaps for adversarial attacks. In this work, we explore whether current guardrails effectively prevent misuse while maintaining practical utility. We introduce a framework to evaluate these tradeoffs, measuring how different guardrails balance risk, security, and usability, and build an efficient guardrail.   Our findings confirm that there is no free lunch with guardrails; strengthening security often comes at the cost of usability. To address this, we propose a blueprint for designing better guardrails that minimize risk while maintaining usability. We evaluate various industry guardrails, including Azure Content Safety, Bedrock Guardrails, OpenAI's Moderation API, Guardrails AI, Nemo Guardrails, and Enkrypt AI guardrails. Additionally, we assess how LLMs like GPT-4o, Gemini 2.0-Flash, Claude 3.5-Sonnet, and Mistral Large-Latest respond under different system prompts, including simple prompts, detailed prompts, and detailed prompts with chain-of-thought (CoT) reasoning. Our study provides a clear comparison of how different guardrails perform, highlighting the challenges in balancing security and usability.

摘要: 随着大型语言模型（LLM）和生成式人工智能的广泛采用，护栏已成为确保其安全使用的关键工具。然而，添加护栏并非没有权衡;更强的安全措施可能会降低可用性，而更灵活的系统可能会为对抗性攻击留下缺口。在这项工作中，我们探索当前的护栏是否有效防止滥用，同时保持实用性。我们引入了一个框架来评估这些权衡，衡量不同的护栏如何平衡风险、安全性和可用性，并构建高效的护栏。   我们的调查结果证实，有护栏就没有免费的午餐;加强安全性往往是以牺牲可用性为代价的。为了解决这个问题，我们提出了一个设计更好护栏的蓝图，在保持可用性的同时最大限度地减少风险。我们评估各种行业护栏，包括Azure内容安全、Bedrock Guardrails、OpenAI的Moderation API、Guardrails AI、Nemo Guardrails和Enkrypt AI护栏。此外，我们还评估GPT-4 o、Gemini 2.0-Flash、Claude 3.5-十四行诗和Mistral Large-Latest等LLM如何在不同的系统提示下做出响应，包括简单提示、详细提示和具有思想链（CoT）推理的详细提示。我们的研究对不同护栏的性能进行了清晰的比较，强调了平衡安全性和可用性的挑战。



## **34. Retrieval-Augmented Purifier for Robust LLM-Empowered Recommendation**

用于强大的LLM授权推荐的检索增强净化器 cs.IR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02458v1) [paper-pdf](http://arxiv.org/pdf/2504.02458v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: Recently, Large Language Model (LLM)-empowered recommender systems have revolutionized personalized recommendation frameworks and attracted extensive attention. Despite the remarkable success, existing LLM-empowered RecSys have been demonstrated to be highly vulnerable to minor perturbations. To mitigate the negative impact of such vulnerabilities, one potential solution is to employ collaborative signals based on item-item co-occurrence to purify the malicious collaborative knowledge from the user's historical interactions inserted by attackers. On the other hand, due to the capabilities to expand insufficient internal knowledge of LLMs, Retrieval-Augmented Generation (RAG) techniques provide unprecedented opportunities to enhance the robustness of LLM-empowered recommender systems by introducing external collaborative knowledge. Therefore, in this paper, we propose a novel framework (RETURN) by retrieving external collaborative signals to purify the poisoned user profiles and enhance the robustness of LLM-empowered RecSys in a plug-and-play manner. Specifically, retrieval-augmented perturbation positioning is proposed to identify potential perturbations within the users' historical sequences by retrieving external knowledge from collaborative item graphs. After that, we further retrieve the collaborative knowledge to cleanse the perturbations by using either deletion or replacement strategies and introduce a robust ensemble recommendation strategy to generate final robust predictions. Extensive experiments on three real-world datasets demonstrate the effectiveness of the proposed RETURN.

摘要: 最近，基于大语言模型（LLM）的推荐系统彻底改变了个性化推荐框架，并引起了广泛关注。尽管取得了显着的成功，但现有的LLM授权RecSys已被证明极易受到微小干扰的影响。为了减轻此类漏洞的负面影响，一种潜在的解决方案是采用基于项-项共存的协作信号，从攻击者插入的用户历史交互中净化恶意协作知识。另一方面，由于扩展LLM内部知识不足的能力，检索增强生成（RAG）技术提供了前所未有的机会，通过引入外部协作知识来增强LLM授权的推荐系统的稳健性。因此，在本文中，我们提出了一种新颖的框架（RETURN），通过检索外部协作信号来净化有毒用户配置文件，并以即插即用的方式增强LLM授权的RecSys的鲁棒性。具体来说，提出了检索增强扰动定位，通过从协作项目图中检索外部知识来识别用户历史序列中的潜在扰动。之后，我们进一步检索协作知识，通过使用删除或替换策略来清除干扰，并引入稳健的集成推荐策略来生成最终的稳健预测。对三个现实世界数据集的广泛实验证明了所提出的RETUN的有效性。



## **35. ToxicSQL: Migrating SQL Injection Threats into Text-to-SQL Models via Backdoor Attack**

ToxicSQL：通过后门攻击将SQL注入威胁迁移到文本到SQL模型 cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2503.05445v2) [paper-pdf](http://arxiv.org/pdf/2503.05445v2)

**Authors**: Meiyu Lin, Haichuan Zhang, Jiale Lao, Renyuan Li, Yuanchun Zhou, Carl Yang, Yang Cao, Mingjie Tang

**Abstract**: Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.

摘要: 大型语言模型（LLM）在将自然语言问题翻译为SQL查询（文本到SQL）方面显示出了最先进的结果，这是数据库界长期存在的挑战。然而，安全问题在很大程度上仍未得到探讨，特别是后门攻击的威胁，后门攻击可以通过对有毒数据集进行微调将恶意行为引入模型。在这项工作中，我们系统地研究了基于LLM的文本到SQL模型的漏洞，并提出了ToxicSQL，这是一种新型后门攻击框架。我们的方法利用隐形的（语义和字符级触发器）来使后门难以检测和删除，确保恶意行为保持隐蔽，同时对良性输入保持高模型准确性。此外，我们建议利用SQL注入有效负载作为后门目标，从而生成恶意但可执行的SQL查询，这在基于语言模型的SQL开发中构成了严重的安全和隐私风险。我们证明，仅注入0.44%的有毒数据就会导致79.41%的攻击成功率，对数据库安全构成重大风险。此外，我们还提出了检测和缓解策略来增强模型的可靠性。我们的研究结果强调了对安全意识的文本到SQL开发的迫切需求，并强调了针对后门威胁的强大防御的重要性。



## **36. Evolving from Single-modal to Multi-modal Facial Deepfake Detection: Progress and Challenges**

从单模式进化到多模式面部Deepfake检测：进展与挑战 cs.CV

P. Liu is with the Department of Computer Science and Engineering,  University of Nevada, Reno, NV, 89512. Q. Tao and J. Zhou are with Centre for  Frontier AI Research (CFAR), and Institute of High Performance Computing  (IHPC), A*STAR, Singapore. J. Zhou is also with Centre for Advanced  Technologies in Online Safety (CATOS), A*STAR, Singapore. J. Zhou is the  corresponding author

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2406.06965v4) [paper-pdf](http://arxiv.org/pdf/2406.06965v4)

**Authors**: Ping Liu, Qiqi Tao, Joey Tianyi Zhou

**Abstract**: As synthetic media, including video, audio, and text, become increasingly indistinguishable from real content, the risks of misinformation, identity fraud, and social manipulation escalate. This survey traces the evolution of deepfake detection from early single-modal methods to sophisticated multi-modal approaches that integrate audio-visual and text-visual cues. We present a structured taxonomy of detection techniques and analyze the transition from GAN-based to diffusion model-driven deepfakes, which introduce new challenges due to their heightened realism and robustness against detection. Unlike prior surveys that primarily focus on single-modal detection or earlier deepfake techniques, this work provides the most comprehensive study to date, encompassing the latest advancements in multi-modal deepfake detection, generalization challenges, proactive defense mechanisms, and emerging datasets specifically designed to support new interpretability and reasoning tasks. We further explore the role of Vision-Language Models (VLMs) and Multimodal Large Language Models (MLLMs) in strengthening detection robustness against increasingly sophisticated deepfake attacks. By systematically categorizing existing methods and identifying emerging research directions, this survey serves as a foundation for future advancements in combating AI-generated facial forgeries. A curated list of all related papers can be found at \href{https://github.com/qiqitao77/Comprehensive-Advances-in-Deepfake-Detection-Spanning-Diverse-Modalities}{https://github.com/qiqitao77/Awesome-Comprehensive-Deepfake-Detection}.

摘要: 随着包括视频、音频和文本在内的合成媒体与真实内容变得越来越难以区分，错误信息、身份欺诈和社会操纵的风险不断升级。这项调查追踪了Deepfake检测从早期的单模式方法到集成视听和文本视觉线索的复杂多模式方法的演变。我们提出了检测技术的结构化分类，并分析了从基于GAN到扩散模型驱动的深度造假的转变，这些深度造假由于其更高的真实性和针对检测的鲁棒性而带来了新的挑战。与之前主要关注单模式检测或早期深度伪造技术的调查不同，这项工作提供了迄今为止最全面的研究，涵盖了多模式深度伪造检测、概括挑战、主动防御机制以及专门设计用于支持新的可解释性和推理任务的新兴数据集。我们进一步探讨了视觉语言模型（VLMS）和多模式大型语言模型（MLLM）在加强针对日益复杂的深度伪造攻击的检测鲁棒性方面的作用。通过对现有方法进行系统性分类并确定新兴研究方向，这项调查为未来打击人工智能生成的面部伪造提供了基础。所有相关论文的精心策划列表可在\href{https：//github.com/qiqitao77/Compresive-Advance-in-Deepfake-Detection-Spanning-Diverse-Modalities}{https：//github.com/qiqitao77/Awesome-Compresive-Deepfake-Detection}找到。



## **37. More is Less: The Pitfalls of Multi-Model Synthetic Preference Data in DPO Safety Alignment**

多即少：DPO安全调整中多模型合成偏好数据的陷阱 cs.AI

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02193v1) [paper-pdf](http://arxiv.org/pdf/2504.02193v1)

**Authors**: Yifan Wang, Runjin Chen, Bolian Li, David Cho, Yihe Deng, Ruqi Zhang, Tianlong Chen, Zhangyang Wang, Ananth Grama, Junyuan Hong

**Abstract**: Aligning large language models (LLMs) with human values is an increasingly critical step in post-training. Direct Preference Optimization (DPO) has emerged as a simple, yet effective alternative to reinforcement learning from human feedback (RLHF). Synthetic preference data with its low cost and high quality enable effective alignment through single- or multi-model generated preference data. Our study reveals a striking, safety-specific phenomenon associated with DPO alignment: Although multi-model generated data enhances performance on general tasks (ARC, Hellaswag, MMLU, TruthfulQA, Winogrande) by providing diverse responses, it also tends to facilitate reward hacking during training. This can lead to a high attack success rate (ASR) when models encounter jailbreaking prompts. The issue is particularly pronounced when employing stronger models like GPT-4o or larger models in the same family to generate chosen responses paired with target model self-generated rejected responses, resulting in dramatically poorer safety outcomes. Furthermore, with respect to safety, using solely self-generated responses (single-model generation) for both chosen and rejected pairs significantly outperforms configurations that incorporate responses from stronger models, whether used directly as chosen data or as part of a multi-model response pool. We demonstrate that multi-model preference data exhibits high linear separability between chosen and rejected responses, which allows models to exploit superficial cues rather than internalizing robust safety constraints. Our experiments, conducted on models from the Llama, Mistral, and Qwen families, consistently validate these findings.

摘要: 使大型语言模型（LLM）与人类价值观保持一致是后培训中越来越重要的一步。直接偏好优化（DPO）已成为人类反馈强化学习（RL HF）的一种简单而有效的替代方案。合成偏好数据具有低成本和高质量，可以通过单一或多模型生成的偏好数据进行有效匹配。我们的研究揭示了一种与DPO对齐相关的引人注目的、特定于安全的现象：尽管多模型生成的数据通过提供多样化的响应来增强一般任务（ARC、Hellaswag、MMLU、TruthfulQA、Winogrande）的性能，但它也往往会促进培训期间的奖励黑客攻击。当模型遇到越狱提示时，这可能会导致高攻击成功率（ASR）。当在同一家族中采用更强的模型（如GPT-4 o或更大的模型）来生成与目标模型自发产生的拒绝响应配对的选择响应时，该问题尤其明显，导致安全性结果明显较差。此外，在安全性方面，对于选择和拒绝的对，单独使用自我生成的响应（单模型生成）显著优于包含来自更强模型的响应的配置，无论是直接用作选择的数据还是作为多模型响应池的一部分。我们证明了多模型偏好数据在选择和拒绝的响应之间表现出高度的线性可分性，这使得模型能够利用表面的线索，而不是内化强大的安全约束。我们对Lama、Mistral和Qwen家族的模型进行的实验一致验证了这些发现。



## **38. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

Included in Proceedings of the Conference on Applied Machine Learning  in Information Security (CAMLIS 2024), Arlington, Virginia, USA, October  24-25, 2024

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2406.03230v5) [paper-pdf](http://arxiv.org/pdf/2406.03230v5)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 以OpenAI的ChatGPT为例，大型语言模型（LLM）的广泛采用使防御这些模型上的对抗威胁的必要性变得更加突出。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性以及用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，利用LLM Transformer层之间的剩余激活分析。我们应用一种新颖的方法来分析剩余流中的独特激活模式，以进行攻击提示分类。我们整理了多个数据集，以展示这种分类方法如何在多种类型的攻击场景（包括我们新创建的攻击数据集）中具有高准确性。此外，我们通过集成LLM的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击能力的影响。结果强调了我们的方法在增强对抗性输入的检测和缓解、推进LLC运作的安全框架方面的有效性。



## **39. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

一张图片就是一切：用一张图片毒害视觉文档检索增强生成 cs.CL

8 pages, 6 figures

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.02132v1) [paper-pdf](http://arxiv.org/pdf/2504.02132v1)

**Authors**: Ezzeldin Shereen, Dan Ristea, Burak Hasircioglu, Shae McFadden, Vasilios Mavroudis, Chris Hicks

**Abstract**: Multimodal retrieval augmented generation (M-RAG) has recently emerged as a method to inhibit hallucinations of large multimodal models (LMMs) through a factual knowledge base (KB). However, M-RAG also introduces new attack vectors for adversaries that aim to disrupt the system by injecting malicious entries into the KB. In this work, we present a poisoning attack against M-RAG targeting visual document retrieval applications, where the KB contains images of document pages. Our objective is to craft a single image that is retrieved for a variety of different user queries, and consistently influences the output produced by the generative model, thus creating a universal denial-of-service (DoS) attack against the M-RAG system. We demonstrate that while our attack is effective against a diverse range of widely-used, state-of-the-art retrievers (embedding models) and generators (LMMs), it can also be ineffective against robust embedding models. Our attack not only highlights the vulnerability of M-RAG pipelines to poisoning attacks, but also sheds light on a fundamental weakness that potentially hinders their performance even in benign settings.

摘要: 多模式检索增强生成（M-RAG）最近出现了作为一种通过事实知识库（KB）抑制大型多模式模型（LSYS）幻觉的方法。然而，M-RAG还为对手引入了新的攻击载体，旨在通过将恶意条目注入知识库来破坏系统。在这项工作中，我们提出了针对M-RAG的中毒攻击，目标是视觉文档检索应用程序，其中KB包含文档页面的图像。我们的目标是制作一个针对各种不同用户查询检索的单个图像，并一致影响生成模型产生的输出，从而对M-RAG系统创建通用拒绝服务（Dock）攻击。我们证明，虽然我们的攻击对各种广泛使用的、最先进的检索器（嵌入模型）和生成器（LSYS）有效，但对稳健的嵌入模型也可能无效。我们的攻击不仅凸显了M-RAG管道对中毒攻击的脆弱性，而且还揭示了一个根本性弱点，即使在良性环境下，该弱点也可能阻碍其性能。



## **40. Evolving Security in LLMs: A Study of Jailbreak Attacks and Defenses**

LLC中不断发展的安全性：越狱攻击和防御的研究 cs.CR

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.02080v1) [paper-pdf](http://arxiv.org/pdf/2504.02080v1)

**Authors**: Zhengchun Shang, Wenlan Wei

**Abstract**: Large Language Models (LLMs) are increasingly popular, powering a wide range of applications. Their widespread use has sparked concerns, especially through jailbreak attacks that bypass safety measures to produce harmful content.   In this paper, we present a comprehensive security analysis of large language models (LLMs), addressing critical research questions on the evolution and determinants of model safety.   Specifically, we begin by identifying the most effective techniques for detecting jailbreak attacks. Next, we investigate whether newer versions of LLMs offer improved security compared to their predecessors. We also assess the impact of model size on overall security and explore the potential benefits of integrating multiple defense strategies to enhance model robustness.   Our study evaluates both open-source models (e.g., LLaMA and Mistral) and closed-source systems (e.g., GPT-4) by employing four state-of-the-art attack techniques and assessing the efficacy of three new defensive approaches.

摘要: 大型语言模型（LLM）越来越受欢迎，为广泛的应用程序提供支持。它们的广泛使用引发了人们的担忧，特别是通过越狱攻击绕过安全措施产生有害内容。   在本文中，我们提出了一个全面的安全分析的大型语言模型（LLM），解决关键的研究问题的演变和决定因素的模型安全性。   具体来说，我们首先确定检测越狱攻击的最有效的技术。接下来，我们调查较新版本的LLM是否比其前身提供了更好的安全性。我们还评估模型大小对整体安全性的影响，并探索集成多种防御策略以增强模型稳健性的潜在好处。   我们的研究评估了两种开源模型（例如，LLaMA和Mistral）和闭源系统（例如，GPT-4）通过采用四种最先进的攻击技术并评估三种新防御方法的有效性。



## **41. AdPO: Enhancing the Adversarial Robustness of Large Vision-Language Models with Preference Optimization**

AdPO：通过偏好优化增强大型视觉语言模型的对抗鲁棒性 cs.CV

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01735v1) [paper-pdf](http://arxiv.org/pdf/2504.01735v1)

**Authors**: Chaohu Liu, Tianyi Gui, Yu Liu, Linli Xu

**Abstract**: Large Vision-Language Models (LVLMs), such as GPT-4o and LLaVA, have recently witnessed remarkable advancements and are increasingly being deployed in real-world applications. However, inheriting the sensitivity of visual neural networks, LVLMs remain vulnerable to adversarial attacks, which can result in erroneous or malicious outputs. While existing efforts utilize adversarial fine-tuning to enhance robustness, they often suffer from performance degradation on clean inputs. In this paper, we proposes AdPO, a novel adversarial defense strategy for LVLMs based on preference optimization. For the first time, we reframe adversarial training as a preference optimization problem, aiming to enhance the model's preference for generating normal outputs on clean inputs while rejecting the potential misleading outputs for adversarial examples. Notably, AdPO achieves this by solely modifying the image encoder, e.g., CLIP ViT, resulting in superior clean and adversarial performance in a variety of downsream tasks. Considering that training involves large language models (LLMs), the computational cost increases significantly. We validate that training on smaller LVLMs and subsequently transferring to larger models can achieve competitive performance while maintaining efficiency comparable to baseline methods. Our comprehensive experiments confirm the effectiveness of the proposed AdPO, which provides a novel perspective for future adversarial defense research.

摘要: GPT-4 o和LLaVA等大型视觉语言模型（LVLM）最近取得了显着的进步，并越来越多地部署在现实世界的应用程序中。然而，由于继承了视觉神经网络的敏感性，LVLM仍然容易受到对抗攻击，这可能会导致错误或恶意输出。虽然现有的工作利用对抗性微调来增强稳健性，但它们经常会在干净的输入上出现性能下降。本文提出了一种基于偏好优化的LVLM新型对抗防御策略AdPO。我们首次将对抗性训练重新定义为偏好优化问题，旨在增强模型在干净输入上生成正常输出的偏好，同时拒绝对抗性示例的潜在误导性输出。值得注意的是，AdPO仅通过修改图像编码器来实现这一点，例如，CLIP ViT，在各种降级任务中带来卓越的干净和对抗性能。考虑到训练涉及大型语言模型（LLM），计算成本显着增加。我们验证了在较小的LVLM上进行训练并随后转移到较大的模型可以实现有竞争力的性能，同时保持与基线方法相当的效率。我们全面的实验证实了拟议AdPO的有效性，为未来的对抗性防御研究提供了新的视角。



## **42. Representation Bending for Large Language Model Safety**

大型语言模型安全性的弯曲表示 cs.LG

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01550v1) [paper-pdf](http://arxiv.org/pdf/2504.01550v1)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.

摘要: 大型语言模型（LLM）已成为强大的工具，但其固有的安全风险--从有害内容生成到更广泛的社会危害--带来了重大挑战。最近的对抗攻击、微调漏洞以及在高风险环境中增加部署LLM可能会放大这些风险。现有的安全增强技术，例如利用人类反馈进行微调或对抗性训练，仍然很脆弱，因为它们可以解决特定的威胁，并且通常无法对不可见的攻击进行概括，或者需要手动系统级防御。本文介绍了RepBend，这是一种新颖的方法，它从根本上破坏了LLM中有害行为的潜在表现，提供了可扩展的解决方案来增强（潜在固有的）安全性。RepBend将激活引导的想法（用于在推理期间引导模型行为的简单载体算法）引入到基于损失的微调中。通过广泛的评估，RepBend实现了最先进的性能，优于Circuit Breaker、RMU和NPO等现有方法，在各种越狱基准测试中，攻击成功率降低了高达95%，模型可用性和通用功能的下降微乎其微。



## **43. LightDefense: A Lightweight Uncertainty-Driven Defense against Jailbreaks via Shifted Token Distribution**

LightDefense：通过转移代币分发针对越狱的轻量级不确定性驱动防御 cs.CR

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01533v1) [paper-pdf](http://arxiv.org/pdf/2504.01533v1)

**Authors**: Zhuoran Yang, Jie Peng, Zhen Tan, Tianlong Chen, Yanyong Zhang

**Abstract**: Large Language Models (LLMs) face threats from jailbreak prompts. Existing methods for defending against jailbreak attacks are primarily based on auxiliary models. These strategies, however, often require extensive data collection or training. We propose LightDefense, a lightweight defense mechanism targeted at white-box models, which utilizes a safety-oriented direction to adjust the probabilities of tokens in the vocabulary, making safety disclaimers appear among the top tokens after sorting tokens by probability in descending order. We further innovatively leverage LLM's uncertainty about prompts to measure their harmfulness and adaptively adjust defense strength, effectively balancing safety and helpfulness. The effectiveness of LightDefense in defending against 5 attack methods across 2 target LLMs, without compromising helpfulness to benign user queries, highlights its potential as a novel and lightweight defense mechanism, enhancing security of LLMs.

摘要: 大型语言模型（LLM）面临越狱提示的威胁。现有的防御越狱攻击的方法主要基于辅助模型。然而，这些策略通常需要广泛的数据收集或培训。我们提出LightDefense，这是一种针对白盒模型的轻量级防御机制，利用以安全为导向的方向来调整词汇表中代币的概率，使安全免责声明在按概率降序排序后出现在前几名代币中。我们进一步创新性地利用LLM对提示的不确定性来衡量其危害性，并自适应地调整防御强度，有效地平衡了安全性和有益性。LightDefense在2个目标LLM上防御5种攻击方法的有效性，而不影响对良性用户查询的帮助，突出了其作为一种新型轻量级防御机制的潜力，增强了LLM的安全性。



## **44. Emerging Cyber Attack Risks of Medical AI Agents**

医疗人工智能代理的新网络攻击风险 cs.CR

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.03759v1) [paper-pdf](http://arxiv.org/pdf/2504.03759v1)

**Authors**: Jianing Qiu, Lin Li, Jiankai Sun, Hao Wei, Zhe Xu, Kyle Lam, Wu Yuan

**Abstract**: Large language models (LLMs)-powered AI agents exhibit a high level of autonomy in addressing medical and healthcare challenges. With the ability to access various tools, they can operate within an open-ended action space. However, with the increase in autonomy and ability, unforeseen risks also arise. In this work, we investigated one particular risk, i.e., cyber attack vulnerability of medical AI agents, as agents have access to the Internet through web browsing tools. We revealed that through adversarial prompts embedded on webpages, cyberattackers can: i) inject false information into the agent's response; ii) they can force the agent to manipulate recommendation (e.g., healthcare products and services); iii) the attacker can also steal historical conversations between the user and agent, resulting in the leak of sensitive/private medical information; iv) furthermore, the targeted agent can also cause a computer system hijack by returning a malicious URL in its response. Different backbone LLMs were examined, and we found such cyber attacks can succeed in agents powered by most mainstream LLMs, with the reasoning models such as DeepSeek-R1 being the most vulnerable.

摘要: 大型语言模型（LLM）驱动的人工智能代理在应对医疗和医疗保健挑战方面表现出高度的自主性。通过访问各种工具的能力，他们可以在开放的行动空间中操作。但随着自主性和能力的提高，不可预见的风险也随之出现。在这项工作中，我们调查了一个特定的风险，即医疗人工智能代理的网络攻击漏洞，因为代理可以通过网络浏览工具访问互联网。我们透露，通过嵌入在网页上的对抗提示，网络攻击者可以：i）将虚假信息注入到代理的响应中; ii）他们可以迫使代理操纵推荐（例如，医疗保健产品和服务）; iii）攻击者还可以窃取用户和代理之间的历史对话，导致敏感/私人医疗信息泄露; iv）此外，目标代理还可以通过在其响应中返回恶意URL来导致计算机系统劫持。对不同的主干LLM进行了检查，我们发现此类网络攻击可以在大多数主流LLM支持的代理中取得成功，其中DeepSeek-R1等推理模型是最脆弱的。



## **45. Strategize Globally, Adapt Locally: A Multi-Turn Red Teaming Agent with Dual-Level Learning**

全球战略，本地适应：具有双重学习的多轮红色团队代理 cs.AI

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01278v1) [paper-pdf](http://arxiv.org/pdf/2504.01278v1)

**Authors**: Si Chen, Xiao Yu, Ninareh Mehrabi, Rahul Gupta, Zhou Yu, Ruoxi Jia

**Abstract**: The exploitation of large language models (LLMs) for malicious purposes poses significant security risks as these models become more powerful and widespread. While most existing red-teaming frameworks focus on single-turn attacks, real-world adversaries typically operate in multi-turn scenarios, iteratively probing for vulnerabilities and adapting their prompts based on threat model responses. In this paper, we propose \AlgName, a novel multi-turn red-teaming agent that emulates sophisticated human attackers through complementary learning dimensions: global tactic-wise learning that accumulates knowledge over time and generalizes to new attack goals, and local prompt-wise learning that refines implementations for specific goals when initial attempts fail. Unlike previous multi-turn approaches that rely on fixed strategy sets, \AlgName enables the agent to identify new jailbreak tactics, develop a goal-based tactic selection framework, and refine prompt formulations for selected tactics. Empirical evaluations on JailbreakBench demonstrate our framework's superior performance, achieving over 90\% attack success rates against GPT-3.5-Turbo and Llama-3.1-70B within 5 conversation turns, outperforming state-of-the-art baselines. These results highlight the effectiveness of dynamic learning in identifying and exploiting model vulnerabilities in realistic multi-turn scenarios.

摘要: 随着大型语言模型（LLM）变得更加强大和广泛，出于恶意目的利用这些模型会带来巨大的安全风险。虽然大多数现有的红色团队框架专注于单回合攻击，但现实世界的对手通常在多回合场景中操作，迭代地探测漏洞并根据威胁模型响应调整其提示。在本文中，我们提出了\AlgName，这是一种新型的多回合红色团队代理，它通过补充的学习维度来模拟复杂的人类攻击者：随着时间的推移积累知识并推广到新的攻击目标的全球战术学习，以及在初始尝试失败时细化特定目标的实现的局部预算学习。与之前依赖固定策略集的多回合方法不同，\AlgName使代理能够识别新的越狱策略、开发基于目标的策略选择框架，并完善所选策略的提示公式。JailbreakBench上的经验评估证明了我们框架的卓越性能，在5次对话中针对GPT-3.5-Turbo和Llama-3.1- 70 B实现了超过90%的攻击成功率，超过了最先进的基线。这些结果凸显了动态学习在现实多转弯场景中识别和利用模型漏洞方面的有效性。



## **46. Towards Resilient Federated Learning in CyberEdge Networks: Recent Advances and Future Trends**

在CyberEdge网络中实现弹性联邦学习：最近的进展和未来的趋势 cs.CR

15 pages, 8 figures, 4 tables, 122 references, journal paper

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01240v1) [paper-pdf](http://arxiv.org/pdf/2504.01240v1)

**Authors**: Kai Li, Zhengyang Zhang, Azadeh Pourkabirian, Wei Ni, Falko Dressler, Ozgur B. Akan

**Abstract**: In this survey, we investigate the most recent techniques of resilient federated learning (ResFL) in CyberEdge networks, focusing on joint training with agglomerative deduction and feature-oriented security mechanisms. We explore adaptive hierarchical learning strategies to tackle non-IID data challenges, improving scalability and reducing communication overhead. Fault tolerance techniques and agglomerative deduction mechanisms are studied to detect unreliable devices, refine model updates, and enhance convergence stability. Unlike existing FL security research, we comprehensively analyze feature-oriented threats, such as poisoning, inference, and reconstruction attacks that exploit model features. Moreover, we examine resilient aggregation techniques, anomaly detection, and cryptographic defenses, including differential privacy and secure multi-party computation, to strengthen FL security. In addition, we discuss the integration of 6G, large language models (LLMs), and interoperable learning frameworks to enhance privacy-preserving and decentralized cross-domain training. These advancements offer ultra-low latency, artificial intelligence (AI)-driven network management, and improved resilience against adversarial attacks, fostering the deployment of secure ResFL in CyberEdge networks.

摘要: 在这项调查中，我们研究了CyberEdge网络中弹性联邦学习（ResFL）的最新技术，重点关注与凝聚演绎和面向特征的安全机制的联合训练。我们探索自适应分层学习策略来应对非IID数据挑战，提高可扩展性并减少通信负担。研究了故障容忍技术和凝聚推理机制，以检测不可靠设备、细化模型更新并增强收敛稳定性。与现有的FL安全研究不同，我们全面分析面向特征的威胁，例如利用模型特征的中毒、推理和重建攻击。此外，我们还研究了弹性聚合技术、异常检测和加密防御，包括差异隐私和安全多方计算，以加强FL安全性。此外，我们还讨论了6G、大型语言模型（LLM）和互操作学习框架的集成，以增强隐私保护和去中心化的跨领域培训。这些进步提供了超低延迟、人工智能（AI）驱动的网络管理，并提高了针对对抗攻击的弹性，促进了在CyberEdge网络中部署安全ResFL。



## **47. Multilingual and Multi-Accent Jailbreaking of Audio LLMs**

多语言和多口音音频LL越狱 cs.SD

21 pages, 6 figures, 15 tables

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01094v1) [paper-pdf](http://arxiv.org/pdf/2504.01094v1)

**Authors**: Jaechul Roh, Virat Shejwalkar, Amir Houmansadr

**Abstract**: Large Audio Language Models (LALMs) have significantly advanced audio understanding but introduce critical security risks, particularly through audio jailbreaks. While prior work has focused on English-centric attacks, we expose a far more severe vulnerability: adversarial multilingual and multi-accent audio jailbreaks, where linguistic and acoustic variations dramatically amplify attack success. In this paper, we introduce Multi-AudioJail, the first systematic framework to exploit these vulnerabilities through (1) a novel dataset of adversarially perturbed multilingual/multi-accent audio jailbreaking prompts, and (2) a hierarchical evaluation pipeline revealing that how acoustic perturbations (e.g., reverberation, echo, and whisper effects) interacts with cross-lingual phonetics to cause jailbreak success rates (JSRs) to surge by up to +57.25 percentage points (e.g., reverberated Kenyan-accented attack on MERaLiON). Crucially, our work further reveals that multimodal LLMs are inherently more vulnerable than unimodal systems: attackers need only exploit the weakest link (e.g., non-English audio inputs) to compromise the entire model, which we empirically show by multilingual audio-only attacks achieving 3.1x higher success rates than text-only attacks. We plan to release our dataset to spur research into cross-modal defenses, urging the community to address this expanding attack surface in multimodality as LALMs evolve.

摘要: 大型音频语言模型（LALM）具有显着提高的音频理解能力，但会带来严重的安全风险，特别是通过音频越狱。虽然之前的工作重点是以英语为中心的攻击，但我们暴露了一个更严重的漏洞：对抗性的多语言和多口音音频越狱，其中语言和声学差异极大地放大了攻击的成功。在本文中，我们引入了Multi-AudioJail，这是第一个利用这些漏洞的系统框架，通过（1）对抗干扰的多语言/多口音音频越狱提示的新颖数据集，以及（2）分层评估管道揭示了声学干扰（例如，回响、回声和耳语效果）与跨语言语音相互作用，导致越狱成功率（JSR）激增高达+57.25个百分点（例如，对MEaLiON产生了肯尼亚口音的攻击）。至关重要的是，我们的工作进一步揭示了多模式LLM本质上比单模式系统更容易受到攻击：攻击者只需要利用最弱的环节（例如，非英语音频输入）来损害整个模型，我们通过多语言纯音频攻击的成功率比纯文本攻击高出3.1倍。我们计划发布我们的数据集，以刺激对跨模式防御的研究，敦促社区随着LALM的发展，以多模式解决这一不断扩大的攻击面。



## **48. The Illusionist's Prompt: Exposing the Factual Vulnerabilities of Large Language Models with Linguistic Nuances**

魔术师的提示：用语言细微差别揭露大型语言模型的事实弱点 cs.CL

work in progress

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.02865v1) [paper-pdf](http://arxiv.org/pdf/2504.02865v1)

**Authors**: Yining Wang, Yuquan Wang, Xi Li, Mi Zhang, Geng Hong, Min Yang

**Abstract**: As Large Language Models (LLMs) continue to advance, they are increasingly relied upon as real-time sources of information by non-expert users. To ensure the factuality of the information they provide, much research has focused on mitigating hallucinations in LLM responses, but only in the context of formal user queries, rather than maliciously crafted ones. In this study, we introduce The Illusionist's Prompt, a novel hallucination attack that incorporates linguistic nuances into adversarial queries, challenging the factual accuracy of LLMs against five types of fact-enhancing strategies. Our attack automatically generates highly transferrable illusory prompts to induce internal factual errors, all while preserving user intent and semantics. Extensive experiments confirm the effectiveness of our attack in compromising black-box LLMs, including commercial APIs like GPT-4o and Gemini-2.0, even with various defensive mechanisms.

摘要: 随着大型语言模型（LLM）的不断发展，非专家用户越来越依赖它们作为实时信息来源。为了确保它们提供的信息的真实性，许多研究都集中在减轻LLM响应中的幻觉上，但仅限于正式用户查询的背景下，而不是恶意制作的查询。在这项研究中，我们引入了幻觉者的提示，这是一种新颖的幻觉攻击，将语言细微差别融入到对抗性询问中，针对五种事实增强策略，挑战LLM的事实准确性。我们的攻击会自动生成高度可转移的幻觉提示，以引发内部事实错误，同时保留用户意图和语义。大量实验证实了我们的攻击在攻击黑匣子LLM（包括GPT-4 o和Gemini-2.0等商业API）方面的有效性，即使有各种防御机制。



## **49. Exposing the Ghost in the Transformer: Abnormal Detection for Large Language Models via Hidden State Forensics**

揭露Transformer中的幽灵：通过隐藏状态取证对大型语言模型进行异常检测 cs.CR

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00446v1) [paper-pdf](http://arxiv.org/pdf/2504.00446v1)

**Authors**: Shide Zhou, Kailong Wang, Ling Shi, Haoyu Wang

**Abstract**: The widespread adoption of Large Language Models (LLMs) in critical applications has introduced severe reliability and security risks, as LLMs remain vulnerable to notorious threats such as hallucinations, jailbreak attacks, and backdoor exploits. These vulnerabilities have been weaponized by malicious actors, leading to unauthorized access, widespread misinformation, and compromised LLM-embedded system integrity. In this work, we introduce a novel approach to detecting abnormal behaviors in LLMs via hidden state forensics. By systematically inspecting layer-specific activation patterns, we develop a unified framework that can efficiently identify a range of security threats in real-time without imposing prohibitive computational costs. Extensive experiments indicate detection accuracies exceeding 95% and consistently robust performance across multiple models in most scenarios, while preserving the ability to detect novel attacks effectively. Furthermore, the computational overhead remains minimal, with merely fractions of a second. The significance of this work lies in proposing a promising strategy to reinforce the security of LLM-integrated systems, paving the way for safer and more reliable deployment in high-stakes domains. By enabling real-time detection that can also support the mitigation of abnormal behaviors, it represents a meaningful step toward ensuring the trustworthiness of AI systems amid rising security challenges.

摘要: 大型语言模型（LLM）在关键应用程序中的广泛采用带来了严重的可靠性和安全风险，因为LLM仍然容易受到幻觉、越狱攻击和后门利用等臭名昭著的威胁的影响。这些漏洞已被恶意行为者武器化，导致未经授权的访问、广泛的错误信息以及LLM嵌入式系统完整性受损。在这项工作中，我们引入了一种通过隐藏状态取证来检测LLM中的异常行为的新颖方法。通过系统性检查特定于层的激活模式，我们开发了一个统一的框架，该框架可以有效地实时识别一系列安全威胁，而无需施加高昂的计算成本。大量实验表明，在大多数情况下，检测准确率超过95%，并且在多个模型中具有一致的稳健性能，同时保留了有效检测新型攻击的能力。此外，计算负担仍然最小，只需几分之一秒。这项工作的意义在于提出一项有希望的策略来加强LLM集成系统的安全性，为在高风险领域中更安全、更可靠的部署铺平道路。通过实现实时检测，也可以支持缓解异常行为，这是在不断上升的安全挑战中确保人工智能系统可信性的有意义的一步。



## **50. Unleashing the Power of Pre-trained Encoders for Universal Adversarial Attack Detection**

释放预培训编码器的力量进行通用对抗攻击检测 cs.CV

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00429v1) [paper-pdf](http://arxiv.org/pdf/2504.00429v1)

**Authors**: Yinghe Zhang, Chi Liu, Shuai Zhou, Sheng Shen, Peng Gui

**Abstract**: Adversarial attacks pose a critical security threat to real-world AI systems by injecting human-imperceptible perturbations into benign samples to induce misclassification in deep learning models. While existing detection methods, such as Bayesian uncertainty estimation and activation pattern analysis, have achieved progress through feature engineering, their reliance on handcrafted feature design and prior knowledge of attack patterns limits generalization capabilities and incurs high engineering costs. To address these limitations, this paper proposes a lightweight adversarial detection framework based on the large-scale pre-trained vision-language model CLIP. Departing from conventional adversarial feature characterization paradigms, we innovatively adopt an anomaly detection perspective. By jointly fine-tuning CLIP's dual visual-text encoders with trainable adapter networks and learnable prompts, we construct a compact representation space tailored for natural images. Notably, our detection architecture achieves substantial improvements in generalization capability across both known and unknown attack patterns compared to traditional methods, while significantly reducing training overhead. This study provides a novel technical pathway for establishing a parameter-efficient and attack-agnostic defense paradigm, markedly enhancing the robustness of vision systems against evolving adversarial threats.

摘要: 对抗性攻击通过将人类难以感知的扰动注入良性样本中，从而在深度学习模型中引发错误分类，对现实世界的人工智能系统构成了严重的安全威胁。虽然现有的检测方法，例如Bayesian不确定性估计和激活模式分析，已经通过特征工程取得了进展，但它们对手工特征设计和攻击模式的先验知识的依赖限制了概括能力并产生了高昂的工程成本。为了解决这些局限性，本文提出了一种基于大规模预训练视觉语言模型CLIP的轻量级对抗检测框架。与传统的对抗性特征描述范式不同，我们创新性地采用异常检测视角。通过将CLIP的双视觉文本编码器与可训练的适配器网络和可学习的提示联合微调，我们构建了一个专为自然图像量身定制的紧凑表示空间。值得注意的是，与传统方法相比，我们的检测架构在已知和未知攻击模式的概括能力方面实现了大幅提高，同时显着减少了训练负担。这项研究为建立参数高效且攻击不可知的防御范式提供了一种新颖的技术途径，显着增强视觉系统针对不断变化的对抗威胁的稳健性。



