# Latest Large Language Model Attack Papers
**update at 2025-05-03 15:41:28**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Can Differentially Private Fine-tuning LLMs Protect Against Privacy Attacks?**

差异私有微调LLM可以防止隐私攻击吗？ cs.CR

accepted by DBSec25

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2504.21036v2) [paper-pdf](http://arxiv.org/pdf/2504.21036v2)

**Authors**: Hao Du, Shang Liu, Yang Cao

**Abstract**: Fine-tuning large language models (LLMs) has become an essential strategy for adapting them to specialized tasks; however, this process introduces significant privacy challenges, as sensitive training data may be inadvertently memorized and exposed. Although differential privacy (DP) offers strong theoretical guarantees against such leakage, its empirical privacy effectiveness on LLMs remains unclear, especially under different fine-tuning methods. In this paper, we systematically investigate the impact of DP across fine-tuning methods and privacy budgets, using both data extraction and membership inference attacks to assess empirical privacy risks. Our main findings are as follows: (1) Differential privacy reduces model utility, but its impact varies significantly across different fine-tuning methods. (2) Without DP, the privacy risks of models fine-tuned with different approaches differ considerably. (3) When DP is applied, even a relatively high privacy budget can substantially lower privacy risk. (4) The privacy-utility trade-off under DP training differs greatly among fine-tuning methods, with some methods being unsuitable for DP due to severe utility degradation. Our results provide practical guidance for privacy-conscious deployment of LLMs and pave the way for future research on optimizing the privacy-utility trade-off in fine-tuning methodologies.

摘要: 微调大型语言模型（LLM）已成为使其适应专业任务的重要策略;然而，这个过程带来了重大的隐私挑战，因为敏感的训练数据可能会被无意中记住和暴露。尽管差异隐私（DP）为防止此类泄露提供了强有力的理论保证，但其对LLM的经验隐私有效性仍然不清楚，尤其是在不同的微调方法下。在本文中，我们系统地研究了DP对微调方法和隐私预算的影响，使用数据提取和成员资格推断攻击来评估经验隐私风险。我们的主要研究结果如下：（1）差异隐私会降低模型效用，但其影响在不同的微调方法中存在显着差异。(2)如果没有DP，用不同方法微调的模型的隐私风险会有很大差异。(3)当应用DP时，即使相对较高的隐私预算也可以大幅降低隐私风险。(4)不同微调方法之间的DP训练下的隐私与公用事业权衡差异很大，有些方法由于公用事业严重退化而不适合DP。我们的结果为具有隐私意识的LLM部署提供了实践指导，并为未来优化微调方法中的隐私与公用事业权衡的研究铺平了道路。



## **2. Stochastic Subspace Descent Accelerated via Bi-fidelity Line Search**

通过双保真线搜索加速随机子空间下降 cs.LG

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2505.00162v1) [paper-pdf](http://arxiv.org/pdf/2505.00162v1)

**Authors**: Nuojin Cheng, Alireza Doostan, Stephen Becker

**Abstract**: Efficient optimization remains a fundamental challenge across numerous scientific and engineering domains, especially when objective function and gradient evaluations are computationally expensive. While zeroth-order optimization methods offer effective approaches when gradients are inaccessible, their practical performance can be limited by the high cost associated with function queries. This work introduces the bi-fidelity stochastic subspace descent (BF-SSD) algorithm, a novel zeroth-order optimization method designed to reduce this computational burden. BF-SSD leverages a bi-fidelity framework, constructing a surrogate model from a combination of computationally inexpensive low-fidelity (LF) and accurate high-fidelity (HF) function evaluations. This surrogate model facilitates an efficient backtracking line search for step size selection, for which we provide theoretical convergence guarantees under standard assumptions. We perform a comprehensive empirical evaluation of BF-SSD across four distinct problems: a synthetic optimization benchmark, dual-form kernel ridge regression, black-box adversarial attacks on machine learning models, and transformer-based black-box language model fine-tuning. Numerical results demonstrate that BF-SSD consistently achieves superior optimization performance while requiring significantly fewer HF function evaluations compared to relevant baseline methods. This study highlights the efficacy of integrating bi-fidelity strategies within zeroth-order optimization, positioning BF-SSD as a promising and computationally efficient approach for tackling large-scale, high-dimensional problems encountered in various real-world applications.

摘要: 有效的优化仍然是众多科学和工程领域的一个根本挑战，特别是当目标函数和梯度评估计算昂贵时。虽然零阶优化方法在无法访问梯度时提供了有效的方法，但其实际性能可能会受到与函数查询相关的高成本的限制。这项工作引入了双保真随机子空间下降（BF-SSD）算法，这是一种新颖的零阶优化方法，旨在减少这种计算负担。BF-SSD利用双保真框架，从计算成本低的低保真度（LF）和准确的高保真度（HF）功能评估的组合中构建代理模型。该代理模型促进了对步骤大小选择的高效回溯线搜索，为此我们在标准假设下提供了理论收敛保证。我们针对四个不同的问题对BF-SSD进行了全面的实证评估：合成优化基准、双重形式内核岭回归、对机器学习模型的黑匣子对抗攻击以及基于转换器的黑匣子语言模型微调。数值结果表明，与相关基线方法相比，BF-SSD始终实现了卓越的优化性能，同时需要的高频功能评估显着减少。这项研究强调了在零阶优化中集成双保真策略的功效，将BF-SSD定位为一种有前途且计算效率高的方法，用于解决各种现实世界应用中遇到的大规模、多维问题。



## **3. Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-based Decision-Making Systems**

我们可以信任有保障的代理人吗？探索针对基于LLM的决策系统的后门攻击 cs.CR

Accepted paper at ICLR 2025, 31 pages, including main paper,  references, and appendix

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2405.20774v3) [paper-pdf](http://arxiv.org/pdf/2405.20774v3)

**Authors**: Ruochen Jiao, Shaoyuan Xie, Justin Yue, Takami Sato, Lixu Wang, Yixuan Wang, Qi Alfred Chen, Qi Zhu

**Abstract**: Large Language Models (LLMs) have shown significant promise in real-world decision-making tasks for embodied artificial intelligence, especially when fine-tuned to leverage their inherent common sense and reasoning abilities while being tailored to specific applications. However, this fine-tuning process introduces considerable safety and security vulnerabilities, especially in safety-critical cyber-physical systems. In this work, we propose the first comprehensive framework for Backdoor Attacks against LLM-based Decision-making systems (BALD) in embodied AI, systematically exploring the attack surfaces and trigger mechanisms. Specifically, we propose three distinct attack mechanisms: word injection, scenario manipulation, and knowledge injection, targeting various components in the LLM-based decision-making pipeline. We perform extensive experiments on representative LLMs (GPT-3.5, LLaMA2, PaLM2) in autonomous driving and home robot tasks, demonstrating the effectiveness and stealthiness of our backdoor triggers across various attack channels, with cases like vehicles accelerating toward obstacles and robots placing knives on beds. Our word and knowledge injection attacks achieve nearly 100% success rate across multiple models and datasets while requiring only limited access to the system. Our scenario manipulation attack yields success rates exceeding 65%, reaching up to 90%, and does not require any runtime system intrusion. We also assess the robustness of these attacks against defenses, revealing their resilience. Our findings highlight critical security vulnerabilities in embodied LLM systems and emphasize the urgent need for safeguarding these systems to mitigate potential risks.

摘要: 大型语言模型（LLM）在具体人工智能的现实决策任务中表现出了巨大的潜力，特别是在进行微调以利用其固有的常识和推理能力，同时针对特定应用进行定制时。然而，这种微调过程引入了相当多的安全和安保漏洞，特别是在安全关键的网络物理系统中。在这项工作中，我们提出了第一个针对嵌入式人工智能中基于LLM的决策系统（BALD）的后门攻击的全面框架，系统地探索了攻击表面和触发机制。具体来说，我们提出了三种不同的攻击机制：文字注入、场景操纵和知识注入，针对基于LLM的决策管道中的各个组件。我们对自动驾驶和家用机器人任务中的代表性LLM（GPT-3.5、LLaMA 2、PaLM 2）进行了广泛的实验，展示了我们的后门触发器在各种攻击渠道中的有效性和隐蔽性，例如车辆加速冲向障碍物和机器人将刀放在床上。我们的文字和知识注入攻击在多个模型和数据集中实现了近100%的成功率，同时只需要有限的系统访问权限。我们的场景操纵攻击的成功率超过65%，高达90%，并且不需要任何运行时系统入侵。我们还评估了这些针对防御系统的攻击的稳健性，揭示了它们的弹性。我们的研究结果强调了嵌入式LLM系统中的关键安全漏洞，并强调迫切需要保护这些系统以减轻潜在风险。



## **4. XBreaking: Explainable Artificial Intelligence for Jailbreaking LLMs**

XBreaking：用于越狱LLM的可解释人工智能 cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21700v1) [paper-pdf](http://arxiv.org/pdf/2504.21700v1)

**Authors**: Marco Arazzi, Vignesh Kumar Kembu, Antonino Nocera, Vinod P

**Abstract**: Large Language Models are fundamental actors in the modern IT landscape dominated by AI solutions. However, security threats associated with them might prevent their reliable adoption in critical application scenarios such as government organizations and medical institutions. For this reason, commercial LLMs typically undergo a sophisticated censoring mechanism to eliminate any harmful output they could possibly produce. In response to this, LLM Jailbreaking is a significant threat to such protections, and many previous approaches have already demonstrated its effectiveness across diverse domains. Existing jailbreak proposals mostly adopt a generate-and-test strategy to craft malicious input. To improve the comprehension of censoring mechanisms and design a targeted jailbreak attack, we propose an Explainable-AI solution that comparatively analyzes the behavior of censored and uncensored models to derive unique exploitable alignment patterns. Then, we propose XBreaking, a novel jailbreak attack that exploits these unique patterns to break the security constraints of LLMs by targeted noise injection. Our thorough experimental campaign returns important insights about the censoring mechanisms and demonstrates the effectiveness and performance of our attack.

摘要: 大型语言模型是由AI解决方案主导的现代IT环境中的基本角色。然而，与它们相关的安全威胁可能会阻止它们在关键应用场景（如政府组织和医疗机构）中的可靠采用。出于这个原因，商业LLM通常会经过复杂的审查机制，以消除它们可能产生的任何有害输出。针对这一点，LLM越狱是对这种保护的重大威胁，许多以前的方法已经在不同的领域证明了其有效性。现有的越狱提案大多采用生成和测试策略来制作恶意输入。为了提高对审查机制的理解并设计有针对性的越狱攻击，我们提出了一种解释性人工智能解决方案，该解决方案比较分析审查和未审查模型的行为，以推导出独特的可利用对齐模式。然后，我们提出了XBreaking，这是一种新型越狱攻击，它利用这些独特的模式通过有针对性的噪音注入来打破LLM的安全限制。我们彻底的实验活动返回了有关审查机制的重要见解，并展示了我们攻击的有效性和性能。



## **5. Hoist with His Own Petard: Inducing Guardrails to Facilitate Denial-of-Service Attacks on Retrieval-Augmented Generation of LLMs**

用自己的花瓣提升：引入护栏以促进对检索增强一代LLM的拒绝服务攻击 cs.CR

11 pages, 6 figures. This work will be submitted to the IEEE for  possible publication

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21680v1) [paper-pdf](http://arxiv.org/pdf/2504.21680v1)

**Authors**: Pan Suo, Yu-Ming Shang, San-Chuan Guo, Xi Zhang

**Abstract**: Retrieval-Augmented Generation (RAG) integrates Large Language Models (LLMs) with external knowledge bases, improving output quality while introducing new security risks. Existing studies on RAG vulnerabilities typically focus on exploiting the retrieval mechanism to inject erroneous knowledge or malicious texts, inducing incorrect outputs. However, these approaches overlook critical weaknesses within LLMs, leaving important attack vectors unexplored and limiting the scope and efficiency of attacks. In this paper, we uncover a novel vulnerability: the safety guardrails of LLMs, while designed for protection, can also be exploited as an attack vector by adversaries. Building on this vulnerability, we propose MutedRAG, a novel denial-of-service attack that reversely leverages the guardrails of LLMs to undermine the availability of RAG systems. By injecting minimalistic jailbreak texts, such as "\textit{How to build a bomb}", into the knowledge base, MutedRAG intentionally triggers the LLM's safety guardrails, causing the system to reject legitimate queries. Besides, due to the high sensitivity of guardrails, a single jailbreak sample can affect multiple queries, effectively amplifying the efficiency of attacks while reducing their costs. Experimental results on three datasets demonstrate that MutedRAG achieves an attack success rate exceeding 60% in many scenarios, requiring only less than one malicious text to each target query on average. In addition, we evaluate potential defense strategies against MutedRAG, finding that some of current mechanisms are insufficient to mitigate this threat, underscoring the urgent need for more robust solutions.

摘要: 检索增强生成（RAG）将大型语言模型（LLM）与外部知识库集成，提高输出质量，同时引入新的安全风险。现有关于RAG漏洞的研究通常集中在利用检索机制注入错误知识或恶意文本，从而引发错误的输出。然而，这些方法忽视了LLM中的关键弱点，导致重要的攻击载体未被探索，并限制了攻击的范围和效率。在本文中，我们发现了一个新颖的漏洞：LLM的安全护栏虽然是为了保护而设计的，但也可能被对手用作攻击载体。在此漏洞的基础上，我们提出了MutedRAG，一种新型的拒绝服务攻击，它利用LLM的护栏来破坏RAG系统的可用性。通过向知识库中注入极简的越狱文本，例如“\textit{How to build a bomb}"，MutedRAG故意触发LLM的安全护栏，导致系统拒绝合法查询。此外，由于护栏的高度敏感性，单个越狱样本可以影响多个查询，有效地放大了攻击的效率，同时降低了攻击的成本。在三个数据集上的实验结果表明，MutedRAG在许多场景下实现了超过60%的攻击成功率，平均每个目标查询只需要不到一个恶意文本。此外，我们评估了针对MutedRAG的潜在防御策略，发现当前的一些机制不足以减轻这种威胁，这凸显了迫切需要更强大的解决方案。



## **6. Traceback of Poisoning Attacks to Retrieval-Augmented Generation**

中毒攻击追溯到检索增强一代 cs.CR

Accepted by The Web Conference 2025

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21668v1) [paper-pdf](http://arxiv.org/pdf/2504.21668v1)

**Authors**: Baolei Zhang, Haoran Xin, Minghong Fang, Zhuqing Liu, Biao Yi, Tong Li, Zheli Liu

**Abstract**: Large language models (LLMs) integrated with retrieval-augmented generation (RAG) systems improve accuracy by leveraging external knowledge sources. However, recent research has revealed RAG's susceptibility to poisoning attacks, where the attacker injects poisoned texts into the knowledge database, leading to attacker-desired responses. Existing defenses, which predominantly focus on inference-time mitigation, have proven insufficient against sophisticated attacks. In this paper, we introduce RAGForensics, the first traceback system for RAG, designed to identify poisoned texts within the knowledge database that are responsible for the attacks. RAGForensics operates iteratively, first retrieving a subset of texts from the database and then utilizing a specially crafted prompt to guide an LLM in detecting potential poisoning texts. Empirical evaluations across multiple datasets demonstrate the effectiveness of RAGForensics against state-of-the-art poisoning attacks. This work pioneers the traceback of poisoned texts in RAG systems, providing a practical and promising defense mechanism to enhance their security.

摘要: 与检索增强生成（RAG）系统集成的大型语言模型（LLM）通过利用外部知识源来提高准确性。然而，最近的研究揭示了RAG对中毒攻击的敏感性，攻击者将中毒文本注入知识数据库，导致攻击者期望的响应。现有的防御主要集中在推理时间缓解上，已经证明不足以抵御复杂的攻击。在本文中，我们介绍RAGForensics，第一个追溯系统RAG，旨在确定有毒的文本知识数据库内的攻击负责。RAGForensics迭代操作，首先从数据库中检索文本子集，然后利用特制的提示来指导LLM检测潜在的中毒文本。多个数据集的经验评估证明了RAGForensics针对最先进的中毒攻击的有效性。这项工作开创了RAG系统中有毒文本的追溯，提供了一种实用且有前途的防御机制来增强其安全性。



## **7. Generative AI in Financial Institution: A Global Survey of Opportunities, Threats, and Regulation**

金融机构中的生成人工智能：机会、威胁和监管的全球调查 cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21574v1) [paper-pdf](http://arxiv.org/pdf/2504.21574v1)

**Authors**: Bikash Saha, Nanda Rani, Sandeep Kumar Shukla

**Abstract**: Generative Artificial Intelligence (GenAI) is rapidly reshaping the global financial landscape, offering unprecedented opportunities to enhance customer engagement, automate complex workflows, and extract actionable insights from vast financial data. This survey provides an overview of GenAI adoption across the financial ecosystem, examining how banks, insurers, asset managers, and fintech startups worldwide are integrating large language models and other generative tools into their operations. From AI-powered virtual assistants and personalized financial advisory to fraud detection and compliance automation, GenAI is driving innovation across functions. However, this transformation comes with significant cybersecurity and ethical risks. We discuss emerging threats such as AI-generated phishing, deepfake-enabled fraud, and adversarial attacks on AI systems, as well as concerns around bias, opacity, and data misuse. The evolving global regulatory landscape is explored in depth, including initiatives by major financial regulators and international efforts to develop risk-based AI governance. Finally, we propose best practices for secure and responsible adoption - including explainability techniques, adversarial testing, auditability, and human oversight. Drawing from academic literature, industry case studies, and policy frameworks, this chapter offers a perspective on how the financial sector can harness GenAI's transformative potential while navigating the complex risks it introduces.

摘要: 生成式人工智能（GenAI）正在迅速重塑全球金融格局，为增强客户参与度、自动化复杂的工作流程以及从大量金融数据中提取可操作的见解提供了前所未有的机会。该调查概述了整个金融生态系统中GenAI的采用情况，研究了全球银行，保险公司，资产管理公司和金融科技初创公司如何将大型语言模型和其他生成工具集成到其运营中。从人工智能驱动的虚拟助理和个性化财务咨询到欺诈检测和合规自动化，GenAI正在推动跨职能的创新。然而，这种转变伴随着重大的网络安全和道德风险。我们讨论了人工智能生成的网络钓鱼、深度伪造的欺诈和对人工智能系统的对抗攻击等新兴威胁，以及对偏见、不透明和数据滥用的担忧。深入探讨了不断变化的全球监管格局，包括主要金融监管机构的举措以及国际上发展基于风险的人工智能治理的努力。最后，我们提出了安全且负责任的采用的最佳实践-包括可解释性技术、对抗性测试、可互换性和人类监督。本章借鉴学术文献、行业案例研究和政策框架，提供了金融部门如何利用GenAI的变革潜力，同时应对其带来的复杂风险的视角。



## **8. Unlocking User-oriented Pages: Intention-driven Black-box Scanner for Real-world Web Applications**

解锁面向用户的页面：真实世界Web应用程序的意图驱动黑盒扫描器 cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.20801v2) [paper-pdf](http://arxiv.org/pdf/2504.20801v2)

**Authors**: Weizhe Wang, Yao Zhang, Kaitai Liang, Guangquan Xu, Hongpeng Bai, Qingyang Yan, Xi Zheng, Bin Wu

**Abstract**: Black-box scanners have played a significant role in detecting vulnerabilities for web applications. A key focus in current black-box scanning is increasing test coverage (i.e., accessing more web pages). However, since many web applications are user-oriented, some deep pages can only be accessed through complex user interactions, which are difficult to reach by existing black-box scanners. To fill this gap, a key insight is that web pages contain a wealth of semantic information that can aid in understanding potential user intention. Based on this insight, we propose Hoyen, a black-box scanner that uses the Large Language Model to predict user intention and provide guidance for expanding the scanning scope. Hoyen has been rigorously evaluated on 12 popular open-source web applications and compared with 6 representative tools. The results demonstrate that Hoyen performs a comprehensive exploration of web applications, expanding the attack surface while achieving about 2x than the coverage of other scanners on average, with high request accuracy. Furthermore, Hoyen detected over 90% of its requests towards the core functionality of the application, detecting more vulnerabilities than other scanners, including unique vulnerabilities in well-known web applications. Our data/code is available at https://hoyen.tjunsl.com/

摘要: 黑匣子扫描仪在检测Web应用程序漏洞方面发挥了重要作用。当前黑匣子扫描的一个关键焦点是增加测试覆盖范围（即，访问更多网页）。然而，由于许多Web应用程序都是面向用户的，因此一些深度页面只能通过复杂的用户交互来访问，而现有的黑匣子扫描仪很难到达这些交互。为了填补这一空白，一个关键的见解是，网页包含了丰富的语义信息，可以帮助理解潜在的用户意图。基于这一见解，我们提出了Hoyen，一个黑盒扫描器，使用大语言模型来预测用户的意图，并为扩大扫描范围提供指导。Hoyen在12个流行的开源Web应用程序上进行了严格的评估，并与6个代表性工具进行了比较。结果表明，Hoyen对Web应用程序进行了全面的探索，扩大了攻击面，同时平均达到了其他扫描器的2倍覆盖率，具有很高的请求准确性。此外，Hoyen检测到了超过90%的针对应用程序核心功能的请求，比其他扫描仪检测到更多的漏洞，包括知名Web应用程序中的独特漏洞。我们的数据/代码可访问https://hoyen.tjunsl.com/



## **9. Round Trip Translation Defence against Large Language Model Jailbreaking Attacks**

针对大型语言模型越狱攻击的往返翻译防御 cs.CL

6 pages, 6 figures

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2402.13517v2) [paper-pdf](http://arxiv.org/pdf/2402.13517v2)

**Authors**: Canaan Yung, Hadi Mohaghegh Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Large language models (LLMs) are susceptible to social-engineered attacks that are human-interpretable but require a high level of comprehension for LLMs to counteract. Existing defensive measures can only mitigate less than half of these attacks at most. To address this issue, we propose the Round Trip Translation (RTT) method, the first algorithm specifically designed to defend against social-engineered attacks on LLMs. RTT paraphrases the adversarial prompt and generalizes the idea conveyed, making it easier for LLMs to detect induced harmful behavior. This method is versatile, lightweight, and transferrable to different LLMs. Our defense successfully mitigated over 70% of Prompt Automatic Iterative Refinement (PAIR) attacks, which is currently the most effective defense to the best of our knowledge. We are also the first to attempt mitigating the MathsAttack and reduced its attack success rate by almost 40%. Our code is publicly available at https://github.com/Cancanxxx/Round_Trip_Translation_Defence   This version of the article has been accepted for publication, after peer review (when applicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The Version of Record is available online at: https://doi.org/10.48550/arXiv.2402.13517 Use of this Accepted Version is subject to the publisher's Accepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscript-terms

摘要: 大型语言模型（LLM）很容易受到人类可解释的社会工程攻击，但LLM需要高水平的理解力才能对抗。现有的防御措施最多只能减轻不到一半的攻击。为了解决这个问题，我们提出了往返翻译（RTI）方法，这是第一个专门设计用于防御对LLM的社会工程攻击的算法。HRT解释了对抗提示并概括了所传达的想法，使LLM更容易检测诱导的有害行为。该方法通用、轻量级，并且可转移到不同的LLM。我们的防御成功缓解了超过70%的提示自动迭代细化（PAIR）攻击，据我们所知，这是目前最有效的防御。我们也是第一个尝试缓解MathsAttack的公司，并将其攻击成功率降低了近40%。我们的代码可在https://github.com/Cancanxxx/Round_Trip_Translation_Defence上公开获取   经过同行评审（如果适用）后，该版本的文章已被接受出版，但不是记录版本，并且不反映接受后的改进或任何更正。记录版本可在线获取：https://doi.org/10.48550/arXiv.2402.13517此接受版本的使用须遵守出版商的接受手稿使用条款https://www.springernature.com/gp/open-research/policies/accepted-manuscript-terms



## **10. CachePrune: Neural-Based Attribution Defense Against Indirect Prompt Injection Attacks**

Cache Prune：针对即时间接注入攻击的基于神经的归因防御 cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.21228v1) [paper-pdf](http://arxiv.org/pdf/2504.21228v1)

**Authors**: Rui Wang, Junda Wu, Yu Xia, Tong Yu, Ruiyi Zhang, Ryan Rossi, Lina Yao, Julian McAuley

**Abstract**: Large Language Models (LLMs) are identified as being susceptible to indirect prompt injection attack, where the model undesirably deviates from user-provided instructions by executing tasks injected in the prompt context. This vulnerability stems from LLMs' inability to distinguish between data and instructions within a prompt. In this paper, we propose CachePrune that defends against this attack by identifying and pruning task-triggering neurons from the KV cache of the input prompt context. By pruning such neurons, we encourage the LLM to treat the text spans of input prompt context as only pure data, instead of any indicator of instruction following. These neurons are identified via feature attribution with a loss function induced from an upperbound of the Direct Preference Optimization (DPO) objective. We show that such a loss function enables effective feature attribution with only a few samples. We further improve on the quality of feature attribution, by exploiting an observed triggering effect in instruction following. Our approach does not impose any formatting on the original prompt or introduce extra test-time LLM calls. Experiments show that CachePrune significantly reduces attack success rates without compromising the response quality. Note: This paper aims to defend against indirect prompt injection attacks, with the goal of developing more secure and robust AI systems.

摘要: 大型语言模型（LLM）被认为容易受到间接提示注入攻击，其中模型通过执行在提示上下文中注入的任务而不希望地偏离用户提供的指令。此漏洞源于LLM无法区分提示内的数据和指令。本文中，我们提出了Cache Prune，它通过从输入提示上下文的KV缓存中识别和修剪任务触发神经元来抵御这种攻击。通过修剪此类神经元，我们鼓励LLM将输入提示上下文的文本跨度仅视为纯数据，而不是任何指令遵循的指示符。这些神经元是通过特征属性识别的，该特征属性具有直接偏好优化（DPO）目标的上界诱导的损失函数。我们表明，这样的损失函数只需少量样本即可实现有效的特征归因。我们通过利用在指令遵循中观察到的触发效应，进一步提高了特征归因的质量。我们的方法不会对原始提示强加任何格式，也不会引入额外的测试时LLM调用。实验表明，Cache Prune显着降低了攻击成功率，而不会影响响应质量。注：本文旨在防御间接即时注入攻击，目标是开发更安全、更强大的人工智能系统。



## **11. ACE: A Security Architecture for LLM-Integrated App Systems**

ACE：LLM集成应用程序系统的安全架构 cs.CR

21 pages, 13 figures

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20984v1) [paper-pdf](http://arxiv.org/pdf/2504.20984v1)

**Authors**: Evan Li, Tushin Mallick, Evan Rose, William Robertson, Alina Oprea, Cristina Nita-Rotaru

**Abstract**: LLM-integrated app systems extend the utility of Large Language Models (LLMs) with third-party apps that are invoked by a system LLM using interleaved planning and execution phases to answer user queries. These systems introduce new attack vectors where malicious apps can cause integrity violation of planning or execution, availability breakdown, or privacy compromise during execution.   In this work, we identify new attacks impacting the integrity of planning, as well as the integrity and availability of execution in LLM-integrated apps, and demonstrate them against IsolateGPT, a recent solution designed to mitigate attacks from malicious apps. We propose Abstract-Concrete-Execute (ACE), a new secure architecture for LLM-integrated app systems that provides security guarantees for system planning and execution. Specifically, ACE decouples planning into two phases by first creating an abstract execution plan using only trusted information, and then mapping the abstract plan to a concrete plan using installed system apps. We verify that the plans generated by our system satisfy user-specified secure information flow constraints via static analysis on the structured plan output. During execution, ACE enforces data and capability barriers between apps, and ensures that the execution is conducted according to the trusted abstract plan. We show experimentally that our system is secure against attacks from the INJECAGENT benchmark, a standard benchmark for control flow integrity in the face of indirect prompt injection attacks, and our newly introduced attacks. Our architecture represents a significant advancement towards hardening LLM-based systems containing system facilities of varying levels of trustworthiness.

摘要: LLM集成的应用程序系统通过第三方应用程序扩展了大型语言模型（LLM）的实用性，第三方应用程序由系统LLM使用交错的规划和执行阶段调用，以回答用户查询。这些系统引入了新的攻击载体，恶意应用程序可能会导致规划或执行的完整性违反、可用性崩溃或执行期间的隐私受到损害。   在这项工作中，我们识别了影响规划完整性以及LLM集成应用程序中执行完整性和可用性的新攻击，并针对IsolateGPT（旨在减轻恶意应用程序攻击的最新解决方案）进行演示。我们提出Abstract-Concrete-Execute（ACE），这是一种针对LLM集成应用程序系统的新安全架构，为系统规划和执行提供安全保障。具体来说，ACE将规划分为两个阶段，首先仅使用可信信息创建抽象执行计划，然后使用已安装的系统应用程序将抽象计划映射到具体计划。我们通过对结构化计划输出的静态分析来验证系统生成的计划是否满足用户指定的安全信息流约束。在执行过程中，ACE在应用程序之间强制设置数据和能力障碍，并确保执行按照可信的抽象计划进行。我们通过实验证明，我们的系统可以抵御来自INJECAGENT基准测试（面对间接提示注入攻击时控制流完整性的标准基准）的攻击，以及我们新引入的攻击。我们的架构代表了强化基于LLM的系统的重大进步，该系统包含不同可信度级别的系统设施。



## **12. Chain-of-Defensive-Thought: Structured Reasoning Elicits Robustness in Large Language Models against Reference Corruption**

防御思想链：结构化推理在大型语言模型中针对引用腐败的鲁棒性 cs.CL

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20769v1) [paper-pdf](http://arxiv.org/pdf/2504.20769v1)

**Authors**: Wenxiao Wang, Parsa Hosseini, Soheil Feizi

**Abstract**: Chain-of-thought prompting has demonstrated great success in facilitating the reasoning abilities of large language models. In this work, we explore how these enhanced reasoning abilities can be exploited to improve the robustness of large language models in tasks that are not necessarily reasoning-focused. In particular, we show how a wide range of large language models exhibit significantly improved robustness against reference corruption using a simple method called chain-of-defensive-thought, where only a few exemplars with structured and defensive reasoning are provided as demonstrations. Empirically, the improvements can be astounding, especially given the simplicity and applicability of the method. For example, in the Natural Questions task, the accuracy of GPT-4o degrades from 60% to as low as 3% with standard prompting when 1 out of 10 references provided is corrupted with prompt injection attacks. In contrast, GPT-4o using chain-of-defensive-thought prompting maintains an accuracy of 50%.

摘要: 思想链提示在促进大型语言模型的推理能力方面取得了巨大成功。在这项工作中，我们探索如何利用这些增强的推理能力来提高大型语言模型在不一定以推理为重点的任务中的稳健性。特别是，我们展示了广泛的大型语言模型如何使用一种称为防御思想链的简单方法来显着提高针对引用腐败的鲁棒性，其中仅提供了一些具有结构化和防御推理的示例作为演示。从经验上看，这些改进可能令人震惊，特别是考虑到该方法的简单性和适用性。例如，在自然问题任务中，当提供的十分之一的参考文献因提示注入攻击而损坏时，GPT-4 o的准确性会从60%下降到标准提示的3%。相比之下，使用防御思想链提示的GPT-4 o保持50%的准确率。



## **13. ReCIT: Reconstructing Full Private Data from Gradient in Parameter-Efficient Fine-Tuning of Large Language Models**

ReCIT：在大型语言模型的参数有效微调中从梯度重建完整的私有数据 cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20570v1) [paper-pdf](http://arxiv.org/pdf/2504.20570v1)

**Authors**: Jin Xie, Ruishi He, Songze Li, Xiaojun Jia, Shouling Ji

**Abstract**: Parameter-efficient fine-tuning (PEFT) has emerged as a practical solution for adapting large language models (LLMs) to custom datasets with significantly reduced computational cost. When carrying out PEFT under collaborative learning scenarios (e.g., federated learning), it is often required to exchange model updates (or gradients) across parties. These gradients, even with limited dimensions, can cause severe breach of data privacy. Recent works have shown that both contextual prefixes and personally identifiable information (PII) can be exposed through gradients. However, \emph{simultaneously} and \emph{accurately} recovering both components from the same training instance remains infeasible due to the following challenges: 1) limited number of PEFT parameters; 2) high-dimensional token spaces; and 3) large batch sizes. We propose ReCIT, a novel privacy attack that addresses all challenges, and achieves recovery of \emph{full} private data from PEFT gradients with high fidelity. Specifically, ReCIT proposes to enhance the memorization capability of the pre-trained model through malicious fine-tuning with Personal Notes; ReCIT also proposes a novel filter-based token extraction technique and a token pairing mechanism, to accurately reconstruct tokens from the training sequences with large batch sizes. Extensive evaluations show that ReCIT consistently outperforms state-of-the-art gradient inversion and memorization-based attacks across different PEFT paradigms. It achieves up to 10$\times$ higher PII recovery rates and remains effective across varying batch sizes, especially in settings where prefix reconstruction is intractable for conventional approaches. These findings highlight an urgent need to reassess the privacy guarantees of PEFT, especially in decentralized or shared training environments.

摘要: 参数高效微调（PEFT）已成为一种将大型语言模型（LLM）适应自定义数据集的实用解决方案，并显着降低计算成本。当在协作学习场景下执行PEFT时（例如，联邦学习），通常需要跨各方交换模型更新（或梯度）。这些梯度，即使维度有限，也可能导致数据隐私的严重侵犯。最近的工作表明，上下文前置码和个人可识别信息（PRI）都可以通过梯度暴露。然而，由于存在以下挑战，\{同时}和\{准确地}从同一训练实例恢复两个组件仍然不可行：1）PEFT参数数量有限; 2）多维令牌空间; 3）批量大小较大。我们提出了ReCIT，这是一种新型隐私攻击，可以解决所有挑战，并以高保真度实现从PEFT梯度恢复\{full}私人数据。具体来说，ReCIT提出通过使用Personal note的恶意微调来增强预训练模型的记忆能力; ReCIT还提出了一种新型的基于过滤器的令牌提取技术和令牌配对机制，以准确地从大批量的训练序列中重建令牌。广泛的评估表明，ReCIT在不同的PEFT范例中始终优于最先进的梯度反转和基于记忆的攻击。它实现了高达10\times $的PII恢复率，并在不同的批量大小中保持有效，特别是在前缀重建对于传统方法来说很难处理的设置中。这些发现强调了重新评估PEFT隐私保障的迫切需要，特别是在分散或共享的培训环境中。



## **14. Token-Efficient Prompt Injection Attack: Provoking Cessation in LLM Reasoning via Adaptive Token Compression**

令牌高效提示注入攻击：通过自适应令牌压缩引发LLM推理中的停止 cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20493v1) [paper-pdf](http://arxiv.org/pdf/2504.20493v1)

**Authors**: Yu Cui, Yujun Cai, Yiwei Wang

**Abstract**: While reasoning large language models (LLMs) demonstrate remarkable performance across various tasks, they also contain notable security vulnerabilities. Recent research has uncovered a "thinking-stopped" vulnerability in DeepSeek-R1, where model-generated reasoning tokens can forcibly interrupt the inference process, resulting in empty responses that compromise LLM-integrated applications. However, existing methods triggering this vulnerability require complex mathematical word problems with long prompts--even exceeding 5,000 tokens. To reduce the token cost and formally define this vulnerability, we propose a novel prompt injection attack named "Reasoning Interruption Attack", based on adaptive token compression. We demonstrate that simple standalone arithmetic tasks can effectively trigger this vulnerability, and the prompts based on such tasks exhibit simpler logical structures than mathematical word problems. We develop a systematic approach to efficiently collect attack prompts and an adaptive token compression framework that utilizes LLMs to automatically compress these prompts. Experiments show our compression framework significantly reduces prompt length while maintaining effective attack capabilities. We further investigate the attack's performance via output prefix and analyze the underlying causes of the vulnerability, providing valuable insights for improving security in reasoning LLMs.

摘要: 虽然推理大型语言模型（LLM）在各种任务中表现出卓越的性能，但它们也包含显着的安全漏洞。最近的研究发现了DeepSeek-R1中的一个“思维停止”漏洞，模型生成的推理令牌可以强行中断推理过程，导致空响应，从而危及LLM集成应用程序。然而，触发此漏洞的现有方法需要复杂的数学单词问题和长提示-甚至超过5，000个令牌。为了减少令牌成本和正式定义这个漏洞，我们提出了一种新的提示注入攻击命名为“推理中断攻击”，基于自适应令牌压缩。我们证明，简单的独立算术任务可以有效地触发此漏洞，并且基于此类任务的提示表现出比数学单词问题更简单的逻辑结构。我们开发了一种有效收集攻击提示的系统方法，以及一个利用LLM自动压缩这些提示的自适应令牌压缩框架。实验表明，我们的压缩框架显着减少了提示长度，同时保持了有效的攻击能力。我们通过输出前置进一步调查攻击的性能并分析漏洞的根本原因，为提高推理LLM的安全性提供有价值的见解。



## **15. Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction**

通过引用实现鲁棒性：通过引用执行的指令来防御提示注入攻击 cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20472v1) [paper-pdf](http://arxiv.org/pdf/2504.20472v1)

**Authors**: Yulin Chen, Haoran Li, Yuan Sui, Yue Liu, Yufei He, Yangqiu Song, Bryan Hooi

**Abstract**: Large language models (LLMs) have demonstrated impressive performance and have come to dominate the field of natural language processing (NLP) across various tasks. However, due to their strong instruction-following capabilities and inability to distinguish between instructions and data content, LLMs are vulnerable to prompt injection attacks. These attacks manipulate LLMs into deviating from the original input instructions and executing maliciously injected instructions within data content, such as web documents retrieved from search engines. Existing defense methods, including prompt-engineering and fine-tuning approaches, typically instruct models to follow the original input instructions while suppressing their tendencies to execute injected instructions. However, our experiments reveal that suppressing instruction-following tendencies is challenging. Through analyzing failure cases, we observe that although LLMs tend to respond to any recognized instructions, they are aware of which specific instructions they are executing and can correctly reference them within the original prompt. Motivated by these findings, we propose a novel defense method that leverages, rather than suppresses, the instruction-following abilities of LLMs. Our approach prompts LLMs to generate responses that include both answers and their corresponding instruction references. Based on these references, we filter out answers not associated with the original input instructions. Comprehensive experiments demonstrate that our method outperforms prompt-engineering baselines and achieves performance comparable to fine-tuning methods, reducing the attack success rate (ASR) to 0 percent in some scenarios. Moreover, our approach has minimal impact on overall utility.

摘要: 大型语言模型（LLM）表现出令人印象深刻的性能，并在各种任务中占据了自然语言处理（NLP）领域的主导地位。然而，由于LLM强大的描述跟踪能力并且无法区分指令和数据内容，因此很容易受到提示注入攻击。这些攻击操纵LLM偏离原始输入指令，并执行数据内容（例如从搜索引擎检索到的网络文档）中恶意注入的指令。现有的防御方法，包括预算工程和微调方法，通常指示模型遵循原始输入指令，同时抑制它们执行注入指令的倾向。然而，我们的实验表明，抑制顺从倾向是具有挑战性的。通过分析失败案例，我们观察到，尽管LLM倾向于响应任何识别的指令，但它们知道自己正在执行哪些特定指令，并且可以在原始提示内正确引用它们。受这些发现的激励，我们提出了一种新型的防御方法，该方法利用而不是抑制LLM的描述跟随能力。我们的方法促使LLM生成包括答案及其相应的指令参考的响应。根据这些参考，我们过滤掉与原始输入指令无关的答案。全面的实验表明，我们的方法优于预算工程基线，并实现了与微调方法相当的性能，在某些情况下将攻击成功率（ASB）降低至0%。此外，我们的方法对整体效用的影响极小。



## **16. NeuRel-Attack: Neuron Relearning for Safety Disalignment in Large Language Models**

NeuRel-Attack：大型语言模型中安全失准的神经元再学习 cs.LG

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.21053v1) [paper-pdf](http://arxiv.org/pdf/2504.21053v1)

**Authors**: Yi Zhou, Wenpeng Xing, Dezhang Kong, Changting Lin, Meng Han

**Abstract**: Safety alignment in large language models (LLMs) is achieved through fine-tuning mechanisms that regulate neuron activations to suppress harmful content. In this work, we propose a novel approach to induce disalignment by identifying and modifying the neurons responsible for safety constraints. Our method consists of three key steps: Neuron Activation Analysis, where we examine activation patterns in response to harmful and harmless prompts to detect neurons that are critical for distinguishing between harmful and harmless inputs; Similarity-Based Neuron Identification, which systematically locates the neurons responsible for safe alignment; and Neuron Relearning for Safety Removal, where we fine-tune these selected neurons to restore the model's ability to generate previously restricted responses. Experimental results demonstrate that our method effectively removes safety constraints with minimal fine-tuning, highlighting a critical vulnerability in current alignment techniques. Our findings underscore the need for robust defenses against adversarial fine-tuning attacks on LLMs.

摘要: 大型语言模型（LLM）中的安全对齐是通过调节神经元激活以抑制有害内容的微调机制来实现的。在这项工作中，我们提出了一种新的方法，通过识别和修改负责安全约束的神经元来诱导失调。我们的方法包括三个关键步骤：神经元激活分析，在那里我们检查响应有害和无害提示的激活模式，以检测对区分有害和无害输入至关重要的神经元;基于相似性的神经元识别，系统地定位负责安全对齐的神经元;和Neuron Relearning for Safety Removal，我们对这些选定的神经元进行微调，以恢复模型生成先前受限响应的能力。实验结果表明，我们的方法可以通过最少的微调有效地消除安全约束，凸显了当前对齐技术中的一个关键漏洞。我们的研究结果强调了对LLM的对抗性微调攻击的强大防御的必要性。



## **17. Enhancing Leakage Attacks on Searchable Symmetric Encryption Using LLM-Based Synthetic Data Generation**

使用基于LLM的合成数据生成增强对可搜索对称加密的泄漏攻击 cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20414v1) [paper-pdf](http://arxiv.org/pdf/2504.20414v1)

**Authors**: Joshua Chiu, Partha Protim Paul, Zahin Wahab

**Abstract**: Searchable Symmetric Encryption (SSE) enables efficient search capabilities over encrypted data, allowing users to maintain privacy while utilizing cloud storage. However, SSE schemes are vulnerable to leakage attacks that exploit access patterns, search frequency, and volume information. Existing studies frequently assume that adversaries possess a substantial fraction of the encrypted dataset to mount effective inference attacks, implying there is a database leakage of such documents, thus, an assumption that may not hold in real-world scenarios. In this work, we investigate the feasibility of enhancing leakage attacks under a more realistic threat model in which adversaries have access to minimal leaked data. We propose a novel approach that leverages large language models (LLMs), specifically GPT-4 variants, to generate synthetic documents that statistically and semantically resemble the real-world dataset of Enron emails. Using the email corpus as a case study, we evaluate the effectiveness of synthetic data generated via random sampling and hierarchical clustering methods on the performance of the SAP (Search Access Pattern) keyword inference attack restricted to token volumes only. Our results demonstrate that, while the choice of LLM has limited effect, increasing dataset size and employing clustering-based generation significantly improve attack accuracy, achieving comparable performance to attacks using larger amounts of real data. We highlight the growing relevance of LLMs in adversarial contexts.

摘要: 可搜索对称加密（SSE）支持对加密数据进行高效搜索，使用户能够在利用云存储的同时维护隐私。然而，SSE方案很容易受到利用访问模式、搜索频率和量信息的泄露攻击。现有的研究经常假设对手拥有很大一部分加密数据集来发起有效的推理攻击，这意味着此类文档的数据库泄露，因此，这一假设在现实世界的场景中可能不成立。在这项工作中，我们研究了在更现实的威胁模型下增强泄露攻击的可行性，其中对手可以访问最少的泄露数据。我们提出了一种新颖的方法，利用大型语言模型（LLM），特别是GPT-4变体，来生成在统计和语义上与安然电子邮件的现实世界数据集相似的合成文档。使用电子邮件库作为案例研究，我们评估了通过随机抽样和分层集群方法生成的合成数据对仅限于令牌量的SAP（搜索访问模式）关键字推理攻击性能的有效性。我们的结果表明，虽然选择LLM的效果有限，但增加数据集大小和采用基于集群的生成可以显着提高攻击准确性，实现与使用大量真实数据的攻击相当的性能。我们强调法学硕士在对抗背景下日益增长的相关性。



## **18. The Automation Advantage in AI Red Teaming**

人工智能红色团队的自动化优势 cs.CR

15 pages, 6 figures

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.19855v2) [paper-pdf](http://arxiv.org/pdf/2504.19855v2)

**Authors**: Rob Mulla, Ads Dawson, Vincent Abruzzon, Brian Greunke, Nick Landers, Brad Palm, Will Pearce

**Abstract**: This paper analyzes Large Language Model (LLM) security vulnerabilities based on data from Crucible, encompassing 214,271 attack attempts by 1,674 users across 30 LLM challenges. Our findings reveal automated approaches significantly outperform manual techniques (69.5% vs 47.6% success rate), despite only 5.2% of users employing automation. We demonstrate that automated approaches excel in systematic exploration and pattern matching challenges, while manual approaches retain speed advantages in certain creative reasoning scenarios, often solving problems 5x faster when successful. Challenge categories requiring systematic exploration are most effectively targeted through automation, while intuitive challenges sometimes favor manual techniques for time-to-solve metrics. These results illuminate how algorithmic testing is transforming AI red-teaming practices, with implications for both offensive security research and defensive measures. Our analysis suggests optimal security testing combines human creativity for strategy development with programmatic execution for thorough exploration.

摘要: 本文基于Crucible的数据分析了大型语言模型（LLM）安全漏洞，涵盖1，674名用户在30个LLM挑战中进行的214，271次攻击尝试。我们的研究结果显示，尽管只有5.2%的用户采用自动化，但自动化方法的表现显着优于手动技术（成功率为69.5% vs 47.6%）。我们证明，自动化方法在系统探索和模式匹配挑战中表现出色，而手动方法在某些创造性推理场景中保留了速度优势，成功后解决问题的速度通常要快5倍。需要系统探索的挑战类别通过自动化最有效地针对，而直观的挑战有时更喜欢手动技术来衡量解决时间指标。这些结果阐明了算法测试如何改变人工智能红团队实践，并对进攻性安全研究和防御措施产生了影响。我们的分析表明，最佳安全测试将人类战略开发的创造力与彻底探索的程序执行相结合。



## **19. BadMoE: Backdooring Mixture-of-Experts LLMs via Optimizing Routing Triggers and Infecting Dormant Experts**

BadMoE：通过优化路由触发器和感染休眠专家来为混合专家LLM做后门 cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.18598v2) [paper-pdf](http://arxiv.org/pdf/2504.18598v2)

**Authors**: Qingyue Wang, Qi Pang, Xixun Lin, Shuai Wang, Daoyuan Wu

**Abstract**: Mixture-of-Experts (MoE) have emerged as a powerful architecture for large language models (LLMs), enabling efficient scaling of model capacity while maintaining manageable computational costs. The key advantage lies in their ability to route different tokens to different ``expert'' networks within the model, enabling specialization and efficient handling of diverse input. However, the vulnerabilities of MoE-based LLMs still have barely been studied, and the potential for backdoor attacks in this context remains largely unexplored. This paper presents the first backdoor attack against MoE-based LLMs where the attackers poison ``dormant experts'' (i.e., underutilized experts) and activate them by optimizing routing triggers, thereby gaining control over the model's output. We first rigorously prove the existence of a few ``dominating experts'' in MoE models, whose outputs can determine the overall MoE's output. We also show that dormant experts can serve as dominating experts to manipulate model predictions. Accordingly, our attack, namely BadMoE, exploits the unique architecture of MoE models by 1) identifying dormant experts unrelated to the target task, 2) constructing a routing-aware loss to optimize the activation triggers of these experts, and 3) promoting dormant experts to dominating roles via poisoned training data. Extensive experiments show that BadMoE successfully enforces malicious prediction on attackers' target tasks while preserving overall model utility, making it a more potent and stealthy attack than existing methods.

摘要: 混合专家（Mixture-of-Experts，MoE）已经成为大型语言模型（LLM）的一个强大架构，能够有效扩展模型容量，同时保持可管理的计算成本。关键优势在于它们能够将不同的令牌路由到模型中的不同“专家”网络，从而实现专业化和有效处理不同的输入。然而，基于MoE的LLM的漏洞仍然很少被研究，在这种情况下后门攻击的可能性在很大程度上仍未被探索。本文介绍了第一个针对基于MoE的LLM的后门攻击，其中攻击者毒害"休眠专家“（即，未充分利用的专家）并通过优化路由触发器激活他们，从而获得对模型输出的控制。我们首先严格证明了存在一些"主导专家“的MoE模型，其输出可以确定整体MoE的输出。我们还表明，休眠专家可以充当主导专家来操纵模型预测。因此，我们的攻击（即BadMoE）利用MoE模型的独特架构，方法是1）识别与目标任务无关的休眠专家，2）构建路由感知损失来优化这些专家的激活触发器，以及3）通过有毒的训练数据将休眠专家提升为主导角色。大量实验表明，BadMoE成功地对攻击者的目标任务实施恶意预测，同时保留了整体模型效用，使其成为比现有方法更强大、更隐蔽的攻击。



## **20. Leveraging LLM to Strengthen ML-Based Cross-Site Scripting Detection**

利用LLM加强基于ML的跨站点脚本检测 cs.CR

This work has been accepted for presentation at the ACM Workshop on  Wireless Security and Machine Learning (WiseML 2025)

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.21045v1) [paper-pdf](http://arxiv.org/pdf/2504.21045v1)

**Authors**: Dennis Miczek, Divyesh Gabbireddy, Suman Saha

**Abstract**: According to the Open Web Application Security Project (OWASP), Cross-Site Scripting (XSS) is a critical security vulnerability. Despite decades of research, XSS remains among the top 10 security vulnerabilities. Researchers have proposed various techniques to protect systems from XSS attacks, with machine learning (ML) being one of the most widely used methods. An ML model is trained on a dataset to identify potential XSS threats, making its effectiveness highly dependent on the size and diversity of the training data. A variation of XSS is obfuscated XSS, where attackers apply obfuscation techniques to alter the code's structure, making it challenging for security systems to detect its malicious intent. Our study's random forest model was trained on traditional (non-obfuscated) XSS data achieved 99.8% accuracy. However, when tested against obfuscated XSS samples, accuracy dropped to 81.9%, underscoring the importance of training ML models with obfuscated data to improve their effectiveness in detecting XSS attacks. A significant challenge is to generate highly complex obfuscated code despite the availability of several public tools. These tools can only produce obfuscation up to certain levels of complexity.   In our proposed system, we fine-tune a Large Language Model (LLM) to generate complex obfuscated XSS payloads automatically. By transforming original XSS samples into diverse obfuscated variants, we create challenging training data for ML model evaluation. Our approach achieved a 99.5% accuracy rate with the obfuscated dataset. We also found that the obfuscated samples generated by the LLMs were 28.1% more complex than those created by other tools, significantly improving the model's ability to handle advanced XSS attacks and making it more effective for real-world application security.

摘要: 根据开放Web应用程序安全项目（OWISP），跨站点脚本（XSS）是一个严重的安全漏洞。尽管经过数十年的研究，XSS仍然是十大安全漏洞之一。研究人员提出了各种技术来保护系统免受XSS攻击，其中机器学习（ML）是最广泛使用的方法之一。ML模型在数据集上训练以识别潜在的XSS威胁，使其有效性高度依赖于训练数据的大小和多样性。XSS的一种变体是混淆XSS，攻击者应用混淆技术来改变代码的结构，这使得安全系统难以检测其恶意意图。我们研究的随机森林模型是在传统（非混淆）XSS数据上训练的，达到了99.8%的准确率。然而，当针对模糊的XSS样本进行测试时，准确率下降至81.9%，这凸显了使用模糊数据训练ML模型以提高其检测XSS攻击的有效性的重要性。一个重大挑战是，尽管有多种公共工具可用，但仍要生成高度复杂的混淆代码。这些工具只能产生达到一定复杂程度的混淆。   在我们提出的系统中，我们对大型语言模型（LLM）进行微调，以自动生成复杂的混淆XSS负载。通过将原始XSS样本转换为各种混淆变体，我们为ML模型评估创建具有挑战性的训练数据。我们的方法在模糊数据集中实现了99.5%的准确率。我们还发现，LLM生成的混淆样本比其他工具创建的样本复杂28.1%，显著提高了模型处理高级XSS攻击的能力，并使其对现实世界的应用程序安全更有效。



## **21. Exploring the Role of Large Language Models in Cybersecurity: A Systematic Survey**

探索大型语言模型在网络安全中的作用：系统性调查 cs.CR

20 pages, 3 figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.15622v2) [paper-pdf](http://arxiv.org/pdf/2504.15622v2)

**Authors**: Shuang Tian, Tao Zhang, Jiqiang Liu, Jiacheng Wang, Xuangou Wu, Xiaoqiang Zhu, Ruichen Zhang, Weiting Zhang, Zhenhui Yuan, Shiwen Mao, Dong In Kim

**Abstract**: With the rapid development of technology and the acceleration of digitalisation, the frequency and complexity of cyber security threats are increasing. Traditional cybersecurity approaches, often based on static rules and predefined scenarios, are struggling to adapt to the rapidly evolving nature of modern cyberattacks. There is an urgent need for more adaptive and intelligent defence strategies. The emergence of Large Language Model (LLM) provides an innovative solution to cope with the increasingly severe cyber threats, and its potential in analysing complex attack patterns, predicting threats and assisting real-time response has attracted a lot of attention in the field of cybersecurity, and exploring how to effectively use LLM to defend against cyberattacks has become a hot topic in the current research field. This survey examines the applications of LLM from the perspective of the cyber attack lifecycle, focusing on the three phases of defense reconnaissance, foothold establishment, and lateral movement, and it analyzes the potential of LLMs in Cyber Threat Intelligence (CTI) tasks. Meanwhile, we investigate how LLM-based security solutions are deployed and applied in different network scenarios. It also summarizes the internal and external risk issues faced by LLM during its application. Finally, this survey also points out the facing risk issues and possible future research directions in this domain.

摘要: 随着技术的快速发展和数字化进程的加快，网络安全威胁的频率和复杂性不断增加。传统的网络安全方法通常基于静态规则和预定义的场景，正在努力适应现代网络攻击快速变化的性质。迫切需要更具适应性和智能性的防御策略。大型语言模型（LLM）的出现为应对日益严重的网络威胁提供了创新解决方案，其在分析复杂攻击模式、预测威胁和辅助实时响应方面的潜力引起了网络安全领域的广泛关注，探索如何有效利用LLM防御网络攻击已成为当前研究领域的热门话题。本次调查从网络攻击生命周期的角度审视了LLM的应用，重点关注防御侦察、立足点建立和侧向移动三个阶段，并分析了LLM在网络威胁情报（RTI）任务中的潜力。同时，我们研究了基于LLM的安全解决方案如何在不同的网络场景中部署和应用。还总结了LLM在应用过程中面临的内部和外部风险问题。最后，本次调查还指出了该领域面临的风险问题以及未来可能的研究方向。



## **22. Les Dissonances: Cross-Tool Harvesting and Polluting in Multi-Tool Empowered LLM Agents**

Les Dissonance：多工具授权的LLM代理中的跨工具收获和污染 cs.CR

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.03111v2) [paper-pdf](http://arxiv.org/pdf/2504.03111v2)

**Authors**: Zichuan Li, Jian Cui, Xiaojing Liao, Luyi Xing

**Abstract**: Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 66 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 75\% are vulnerable to XTHP attacks, highlighting the prevalence of this threat.

摘要: 大型语言模型（LLM）代理是由LLM支持的自治系统，能够通过利用一组工具进行推理和规划来解决问题。然而，LLM代理中多工具功能的集成在安全管理工具、确保其兼容性、处理依赖关系以及保护LLM代理工作流程中的控制流方面带来了挑战。本文中，我们首次对支持多工具的LLM代理中的任务控制流进行了系统性安全分析。我们识别了一种新型威胁，即跨工具收获和污染（XTHP），它包括多个攻击载体，首先劫持代理任务的正常控制流，然后收集和污染LLM代理系统内的机密或私人信息。为了了解这种威胁的影响，我们开发了Chord，这是一种动态扫描工具，旨在自动检测容易受到XTHP攻击的现实世界代理工具。我们对来自两个主要LLM代理开发框架LangChain和LlamaIndex存储库的66个现实工具进行了评估，发现了一个重大的安全问题：75%的工具容易受到XTHP攻击，凸显了这种威胁的普遍性。



## **23. SoK: Knowledge is All You Need: Accelerating Last Mile Delivery for Automated Provenance-based Intrusion Detection with LLMs**

知识就是你所需要的一切：使用LLM加速最后一英里交付，以实现自动化的基于来源的入侵检测 cs.CR

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2503.03108v2) [paper-pdf](http://arxiv.org/pdf/2503.03108v2)

**Authors**: Wenrui Cheng, Tiantian Zhu, Chunlin Xiong, Haofei Sun, Zijun Wang, Shunan Jing, Mingqi Lv, Yan Chen

**Abstract**: Recently, provenance-based intrusion detection systems (PIDSes) have been widely proposed for endpoint threat analysis. However, due to the lack of systematic integration and utilization of knowledge, existing PIDSes still require significant manual intervention for practical deployment, making full automation challenging. This paper presents a disruptive innovation by categorizing PIDSes according to the types of knowledge they utilize. In response to the prevalent issue of ``knowledge silos problem'' in existing research, we introduce a novel knowledge-driven provenance-based intrusion detection framework, powered by large language models (LLMs). We also present OmniSec, a best practice system built upon this framework. By integrating attack representation knowledge, threat intelligence knowledge, and benign behavior knowledge, OmniSec outperforms the state-of-the-art approaches on public benchmark datasets. OmniSec is available online at https://anonymous.4open.science/r/PIDS-with-LLM-613B.

摘要: 最近，基于来源的入侵检测系统（PIDS）被广泛提出用于端点威胁分析。然而，由于缺乏知识的系统集成和利用，现有的PIDS仍然需要大量的手动干预才能进行实际部署，这使得完全自动化具有挑战性。本文通过根据PIDS使用的知识类型对它们进行分类，提出了一种颠覆性创新。为了应对现有研究中普遍存在的“知识孤岛问题”问题，我们引入了一种由大型语言模型（LLM）支持的新型知识驱动的基于出处的入侵检测框架。我们还介绍了OmniSec，这是一个基于此框架构建的最佳实践系统。通过集成攻击表示知识、威胁情报知识和良性行为知识，OmniSec在公共基准数据集上优于最先进的方法。OmniSec可在线访问https://anonymous.4open.science/r/PIDS-with-LLM-613B。



## **24. LLMPot: Dynamically Configured LLM-based Honeypot for Industrial Protocol and Physical Process Emulation**

LLMPot：动态配置的基于LLM的蜜罐，用于工业协议和物理流程仿真 cs.CR

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2405.05999v2) [paper-pdf](http://arxiv.org/pdf/2405.05999v2)

**Authors**: Christoforos Vasilatos, Dunia J. Mahboobeh, Hithem Lamri, Manaar Alam, Michail Maniatakos

**Abstract**: Industrial Control Systems (ICS) are extensively used in critical infrastructures ensuring efficient, reliable, and continuous operations. However, their increasing connectivity and addition of advanced features make them vulnerable to cyber threats, potentially leading to severe disruptions in essential services. In this context, honeypots play a vital role by acting as decoy targets within ICS networks, or on the Internet, helping to detect, log, analyze, and develop mitigations for ICS-specific cyber threats. Deploying ICS honeypots, however, is challenging due to the necessity of accurately replicating industrial protocols and device characteristics, a crucial requirement for effectively mimicking the unique operational behavior of different industrial systems. Moreover, this challenge is compounded by the significant manual effort required in also mimicking the control logic the PLC would execute, in order to capture attacker traffic aiming to disrupt critical infrastructure operations. In this paper, we propose LLMPot, a novel approach for designing honeypots in ICS networks harnessing the potency of Large Language Models (LLMs). LLMPot aims to automate and optimize the creation of realistic honeypots with vendor-agnostic configurations, and for any control logic, aiming to eliminate the manual effort and specialized knowledge traditionally required in this domain. We conducted extensive experiments focusing on a wide array of parameters, demonstrating that our LLM-based approach can effectively create honeypot devices implementing different industrial protocols and diverse control logic.

摘要: 工业控制系统（ICS）广泛用于关键基础设施，确保高效、可靠和连续的运营。然而，它们不断增加的连接性和添加的高级功能使它们容易受到网络威胁的影响，可能导致基本服务的严重中断。在这种情况下，蜜罐发挥着至关重要的作用，充当ICS网络内或互联网上的诱饵目标，帮助检测、记录、分析和开发针对ICS特定网络威胁的缓解措施。然而，部署ICS蜜罐具有挑战性，因为需要准确地复制工业协议和设备特征，这是有效模仿不同工业系统独特操作行为的关键要求。此外，模仿PLC将执行的控制逻辑以捕获旨在破坏关键基础设施运营的攻击者流量所需的大量手动工作使这一挑战变得更加复杂。在本文中，我们提出了LLMPot，这是一种在ICS网络中设计蜜罐的新颖方法，利用大型语言模型（LLM）的能力。LLMPot旨在自动化和优化具有供应商不可知配置的现实蜜罐的创建，并适用于任何控制逻辑，旨在消除该领域传统上所需的手动工作和专业知识。我们针对广泛的参数进行了广泛的实验，证明我们基于LLM的方法可以有效地创建实施不同工业协议和不同控制逻辑的蜜罐设备。



## **25. Mapping the Italian Telegram Ecosystem**

构建意大利Telegram生态系统 cs.SI

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19594v1) [paper-pdf](http://arxiv.org/pdf/2504.19594v1)

**Authors**: Lorenzo Alvisi, Serena Tardelli, Maurizio Tesconi

**Abstract**: Telegram has become a major space for political discourse and alternative media. However, its lack of moderation allows misinformation, extremism, and toxicity to spread. While prior research focused on these particular phenomena or topics, these have mostly been examined separately, and a broader understanding of the Telegram ecosystem is still missing. In this work, we fill this gap by conducting a large-scale analysis of the Italian Telegram sphere, leveraging a dataset of 186 million messages from 13,151 chats collected in 2023. Using network analysis, Large Language Models, and toxicity detection tools, we examine how different thematic communities form, align ideologically, and engage in harmful discourse within the Italian cultural context. Results show strong thematic and ideological homophily. We also identify mixed ideological communities where far-left and far-right rhetoric coexist on particular geopolitical issues. Beyond political analysis, we find that toxicity, rather than being isolated in a few extreme chats, appears widely normalized within highly toxic communities. Moreover, we find that Italian discourse primarily targets Black people, Jews, and gay individuals independently of the topic. Finally, we uncover common trend of intra-national hostility, where Italians often attack other Italians, reflecting regional and intra-regional cultural conflicts that can be traced back to old historical divisions. This study provides the first large-scale mapping of the Italian Telegram ecosystem, offering insights into ideological interactions, toxicity, and identity-targets of hate and contributing to research on online toxicity across different cultural and linguistic contexts on Telegram.

摘要: Telegram已成为政治话语和另类媒体的主要空间。然而，它缺乏节制，导致错误信息、极端主义和毒性蔓延。虽然之前的研究集中在这些特定的现象或主题上，但这些主要是单独研究的，并且仍然缺乏对Telegram生态系统的更广泛的了解。在这项工作中，我们通过对意大利Telegram领域进行大规模分析来填补这一空白，利用2023年收集的13，151条聊天记录中的1.86亿条消息的数据集。使用网络分析、大型语言模型和毒性检测工具，我们研究不同的主题社区如何在意大利文化背景下形成、意识形态上的一致以及参与有害话语。结果显示出较强的主题和意识形态一致性。我们还发现了混合的意识形态社区，其中极左和极右言论在特定地缘政治问题上共存。除了政治分析之外，我们发现毒性并没有在一些极端的聊天中被孤立，而是在高毒性社区中被广泛正常化。此外，我们发现意大利语的话语主要针对黑人、犹太人和同性恋者，与主题无关。最后，我们发现了国内敌意的共同趋势，意大利人经常攻击其他意大利人，反映了可以追溯到旧历史分歧的地区和地区内文化冲突。这项研究首次对意大利Telegram生态系统进行了大规模映射，提供了对意识形态相互作用、毒性和仇恨身份目标的见解，并为Telegram上不同文化和语言背景下的在线毒性研究做出了贡献。



## **26. Prefill-Based Jailbreak: A Novel Approach of Bypassing LLM Safety Boundary**

基于预填充的越狱：一种突破LLM安全边界的新方法 cs.CR

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.21038v1) [paper-pdf](http://arxiv.org/pdf/2504.21038v1)

**Authors**: Yakai Li, Jiekang Hu, Weiduan Sang, Luping Ma, Jing Xie, Weijuan Zhang, Aimin Yu, Shijie Zhao, Qingjia Huang, Qihang Zhou

**Abstract**: Large Language Models (LLMs) are designed to generate helpful and safe content. However, adversarial attacks, commonly referred to as jailbreak, can bypass their safety protocols, prompting LLMs to generate harmful content or reveal sensitive data. Consequently, investigating jailbreak methodologies is crucial for exposing systemic vulnerabilities within LLMs, ultimately guiding the continuous implementation of security enhancements by developers. In this paper, we introduce a novel jailbreak attack method that leverages the prefilling feature of LLMs, a feature designed to enhance model output constraints. Unlike traditional jailbreak methods, the proposed attack circumvents LLMs' safety mechanisms by directly manipulating the probability distribution of subsequent tokens, thereby exerting control over the model's output. We propose two attack variants: Static Prefilling (SP), which employs a universal prefill text, and Optimized Prefilling (OP), which iteratively optimizes the prefill text to maximize the attack success rate. Experiments on six state-of-the-art LLMs using the AdvBench benchmark validate the effectiveness of our method and demonstrate its capability to substantially enhance attack success rates when combined with existing jailbreak approaches. The OP method achieved attack success rates of up to 99.82% on certain models, significantly outperforming baseline methods. This work introduces a new jailbreak attack method in LLMs, emphasizing the need for robust content validation mechanisms to mitigate the adversarial exploitation of prefilling features. All code and data used in this paper are publicly available.

摘要: 大型语言模型（LLM）旨在生成有用且安全的内容。然而，对抗性攻击（通常称为越狱）可能会绕过其安全协议，促使LLM生成有害内容或泄露敏感数据。因此，调查越狱方法对于暴露LLC内的系统漏洞至关重要，最终指导开发人员持续实施安全增强。在本文中，我们引入了一种新颖的越狱攻击方法，该方法利用了LLM的预填充功能，该功能旨在增强模型输出约束。与传统的越狱方法不同，提出的攻击通过直接操纵后续令牌的概率分布来规避LLM的安全机制，从而对模型的输出施加控制。我们提出了两种攻击变体：采用通用预填充文本的静态预填充（SP）和迭代优化预填充文本以最大化攻击成功率的优化预填充（OP）。使用AdvBench基准对六种最先进的LLM进行实验验证了我们方法的有效性，并证明了其与现有越狱方法相结合时能够大幅提高攻击成功率。OP方法在某些模型上的攻击成功率高达99.82%，显着优于基线方法。这项工作在LLM中引入了一种新的越狱攻击方法，强调需要强大的内容验证机制来减轻对预填充功能的对抗性利用。本文中使用的所有代码和数据都是公开的。



## **27. Exposing Privacy Gaps: Membership Inference Attack on Preference Data for LLM Alignment**

暴露隐私差距：对LLM一致偏好数据的会员推断攻击 cs.AI

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2407.06443v2) [paper-pdf](http://arxiv.org/pdf/2407.06443v2)

**Authors**: Qizhang Feng, Siva Rajesh Kasa, Santhosh Kumar Kasa, Hyokun Yun, Choon Hui Teo, Sravan Babu Bodapati

**Abstract**: Large Language Models (LLMs) have seen widespread adoption due to their remarkable natural language capabilities. However, when deploying them in real-world settings, it is important to align LLMs to generate texts according to acceptable human standards. Methods such as Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO) have enabled significant progress in refining LLMs using human preference data. However, the privacy concerns inherent in utilizing such preference data have yet to be adequately studied. In this paper, we investigate the vulnerability of LLMs aligned using two widely used methods - DPO and PPO - to membership inference attacks (MIAs). Our study has two main contributions: first, we theoretically motivate that DPO models are more vulnerable to MIA compared to PPO models; second, we introduce a novel reference-based attack framework specifically for analyzing preference data called PREMIA (\uline{Pre}ference data \uline{MIA}). Using PREMIA and existing baselines we empirically show that DPO models have a relatively heightened vulnerability towards MIA.

摘要: 大型语言模型（LLM）因其出色的自然语言能力而得到广泛采用。然而，当在现实世界环境中部署它们时，根据可接受的人类标准调整LLM以生成文本非常重要。近端策略优化（PPO）和直接偏好优化（DPO）等方法在使用人类偏好数据细化LLM方面取得了重大进展。然而，利用此类偏好数据固有的隐私问题尚未得到充分研究。本文研究了使用两种广泛使用的方法（DPO和PPO）对齐的LLM对成员推断攻击（MIA）的脆弱性。我们的研究有两个主要贡献：首先，我们从理论上认为，与PPO模型相比，DPO模型更容易受到MIA的影响;其次，我们引入了一种新型的基于引用的攻击框架，专门用于分析偏好数据，称为PREMIA（\uline{Pre}参考数据\uline{MIA}）。使用PREMIA和现有基线，我们经验表明DPO模型对MIA的脆弱性相对较高。



## **28. Graph of Attacks: Improved Black-Box and Interpretable Jailbreaks for LLMs**

攻击图表：LLM的改进黑匣子和可解释越狱 cs.CL

19 pages, 1 figure, 6 tables

**SubmitDate**: 2025-04-26    [abs](http://arxiv.org/abs/2504.19019v1) [paper-pdf](http://arxiv.org/pdf/2504.19019v1)

**Authors**: Mohammad Akbar-Tajari, Mohammad Taher Pilehvar, Mohammad Mahmoody

**Abstract**: The challenge of ensuring Large Language Models (LLMs) align with societal standards is of increasing interest, as these models are still prone to adversarial jailbreaks that bypass their safety mechanisms. Identifying these vulnerabilities is crucial for enhancing the robustness of LLMs against such exploits. We propose Graph of ATtacks (GoAT), a method for generating adversarial prompts to test the robustness of LLM alignment using the Graph of Thoughts framework [Besta et al., 2024]. GoAT excels at generating highly effective jailbreak prompts with fewer queries to the victim model than state-of-the-art attacks, achieving up to five times better jailbreak success rate against robust models like Llama. Notably, GoAT creates high-quality, human-readable prompts without requiring access to the targeted model's parameters, making it a black-box attack. Unlike approaches constrained by tree-based reasoning, GoAT's reasoning is based on a more intricate graph structure. By making simultaneous attack paths aware of each other's progress, this dynamic framework allows a deeper integration and refinement of reasoning paths, significantly enhancing the collaborative exploration of adversarial vulnerabilities in LLMs. At a technical level, GoAT starts with a graph structure and iteratively refines it by combining and improving thoughts, enabling synergy between different thought paths. The code for our implementation can be found at: https://github.com/GoAT-pydev/Graph_of_Attacks.

摘要: 确保大型语言模型（LLM）与社会标准保持一致的挑战越来越受到关注，因为这些模型仍然容易出现绕过其安全机制的对抗性越狱。识别这些漏洞对于增强LLM针对此类漏洞的稳健性至关重要。我们提出了攻击图形（GoAT），这是一种用于生成对抗提示的方法，以使用思想图形框架测试LLM对齐的稳健性[Besta等人，2024]。GoAT擅长生成高效的越狱提示，对受害者模型的查询比最先进的攻击更少，针对Llama等稳健模型，越狱成功率高出五倍。值得注意的是，GoAT可以创建高质量、人类可读的提示，而不需要访问目标模型的参数，使其成为黑匣子攻击。与受基于树的推理约束的方法不同，GoAT的推理基于更复杂的图结构。通过让同时攻击路径了解彼此的进展，这个动态框架允许更深入地集成和细化推理路径，显着增强了对LLM对抗漏洞的协作探索。在技术层面，GoAT从图形结构开始，通过组合和改进思想来迭代细化它，实现不同思维路径之间的协同。我们的实现代码可在：https://github.com/GoAT-pydev/Graph_of_Attacks上找到。



## **29. ThreMoLIA: Threat Modeling of Large Language Model-Integrated Applications**

ThreMoLIA：大型语言模型集成应用程序的威胁建模 cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18369v1) [paper-pdf](http://arxiv.org/pdf/2504.18369v1)

**Authors**: Felix Viktor Jedrzejewski, Davide Fucci, Oleksandr Adamov

**Abstract**: Large Language Models (LLMs) are currently being integrated into industrial software applications to help users perform more complex tasks in less time. However, these LLM-Integrated Applications (LIA) expand the attack surface and introduce new kinds of threats. Threat modeling is commonly used to identify these threats and suggest mitigations. However, it is a time-consuming practice that requires the involvement of a security practitioner. Our goals are to 1) provide a method for performing threat modeling for LIAs early in their lifecycle, (2) develop a threat modeling tool that integrates existing threat models, and (3) ensure high-quality threat modeling. To achieve the goals, we work in collaboration with our industry partner. Our proposed way of performing threat modeling will benefit industry by requiring fewer security experts' participation and reducing the time spent on this activity. Our proposed tool combines LLMs and Retrieval Augmented Generation (RAG) and uses sources such as existing threat models and application architecture repositories to continuously create and update threat models. We propose to evaluate the tool offline -- i.e., using benchmarking -- and online with practitioners in the field. We conducted an early evaluation using ChatGPT on a simple LIA and obtained results that encouraged us to proceed with our research efforts.

摘要: 大型语言模型（LLM）目前正在集成到工业软件应用程序中，以帮助用户在更短的时间内执行更复杂的任务。然而，这些LLM集成应用程序（LIA）扩大了攻击面并引入了新型威胁。威胁建模通常用于识别这些威胁并建议缓解措施。然而，这是一种耗时的做法，需要安全从业者的参与。我们的目标是1）提供一种在LIA生命周期早期对其进行威胁建模的方法，（2）开发集成现有威胁模型的威胁建模工具，以及（3）确保高质量的威胁建模。为了实现这些目标，我们与行业合作伙伴合作。我们提出的执行威胁建模的方法将使行业受益，因为需要更少的安全专家参与并减少花在此活动上的时间。我们提出的工具结合了LLM和检索增强生成（RAG），并使用现有威胁模型和应用程序架构存储库等源来不断创建和更新威胁模型。我们建议离线评估该工具--即，使用基准测试--并在线与该领域的从业者联系。我们使用ChatGPT对简单的LIA进行了早期评估，并获得了鼓励我们继续研究工作的结果。



## **30. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

通过跨模式提示注射操纵多模式代理 cs.CV

17 pages, 5 figures

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.14348v3) [paper-pdf](http://arxiv.org/pdf/2504.14348v3)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this work, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach consists of two key components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to infer the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms existing injection attacks, achieving at least a +26.4% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.

摘要: 多模式大型语言模型的出现通过将语言和视觉模式与外部数据源集成来重新定义了代理范式，使代理能够更好地解释人类指令并执行日益复杂的任务。然而，在这项工作中，我们发现了多模式代理中一个以前被忽视的关键安全漏洞：跨模式提示注入攻击。为了利用这个漏洞，我们提出了CrossInib，这是一种新型攻击框架，其中攻击者在多种模式中嵌入对抗性扰动，以与目标恶意内容保持一致，允许外部指令劫持代理的决策过程并执行未经授权的任务。我们的方法由两个关键部分组成。首先，我们引入了视觉潜在对齐，基于文本到图像生成模型，优化视觉嵌入空间中恶意指令的对抗特征，确保对抗图像巧妙地编码恶意任务执行的线索。随后，我们提出了文本指导增强，其中利用大型语言模型通过对抗性Meta提示来推断黑匣子防御系统提示，并生成恶意文本命令，该命令引导代理的输出更好地遵守攻击者的请求。大量实验表明，我们的方法优于现有的注入攻击，在不同任务中的攻击成功率至少增加了+26.4%。此外，我们还验证了攻击在现实世界的多模式自治代理中的有效性，强调了其对安全关键应用程序的潜在影响。



## **31. NoEsis: Differentially Private Knowledge Transfer in Modular LLM Adaptation**

NoEsis：模块化LLM适应中的差异性私人知识转移 cs.CR

ICLR 2025 MCDC workshop

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18147v1) [paper-pdf](http://arxiv.org/pdf/2504.18147v1)

**Authors**: Rob Romijnders, Stefanos Laskaridis, Ali Shahin Shamsabadi, Hamed Haddadi

**Abstract**: Large Language Models (LLM) are typically trained on vast amounts of data from various sources. Even when designed modularly (e.g., Mixture-of-Experts), LLMs can leak privacy on their sources. Conversely, training such models in isolation arguably prohibits generalization. To this end, we propose a framework, NoEsis, which builds upon the desired properties of modularity, privacy, and knowledge transfer. NoEsis integrates differential privacy with a hybrid two-staged parameter-efficient fine-tuning that combines domain-specific low-rank adapters, acting as experts, with common prompt tokens, acting as a knowledge-sharing backbone. Results from our evaluation on CodeXGLUE showcase that NoEsis can achieve provable privacy guarantees with tangible knowledge transfer across domains, and empirically show protection against Membership Inference Attacks. Finally, on code completion tasks, NoEsis bridges at least 77% of the accuracy gap between the non-shared and the non-private baseline.

摘要: 大型语言模型（LLM）通常根据来自各种来源的大量数据进行训练。即使采用模块化设计（例如，专家混合），LLM可以泄露其来源的隐私。相反，孤立地训练此类模型可以说会禁止概括。为此，我们提出了一个框架NoEsis，该框架建立在模块性、隐私和知识转移等所需属性的基础上。NoEsis将差异隐私与混合两阶段参数高效微调集成，该微调将充当专家的特定领域低级适配器与充当知识共享主干的公共提示令牌相结合。我们对CodeXGLUE的评估结果表明，NoEsis可以通过跨域的有形知识转移来实现可证明的隐私保证，并从经验上展示了针对会员资格推断攻击的保护。最后，在代码完成任务方面，NoEsis弥合了非共享和非私有基线之间至少77%的准确性差距。



## **32. Automating Function-Level TARA for Automotive Full-Lifecycle Security**

自动化功能级TARA以实现汽车全面安全 cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18083v1) [paper-pdf](http://arxiv.org/pdf/2504.18083v1)

**Authors**: Yuqiao Yang, Yongzhao Zhang, Wenhao Liu, Jun Li, Pengtao Shi, DingYu Zhong, Jie Yang, Ting Chen, Sheng Cao, Yuntao Ren, Yongyue Wu, Xiaosong Zhang

**Abstract**: As modern vehicles evolve into intelligent and connected systems, their growing complexity introduces significant cybersecurity risks. Threat Analysis and Risk Assessment (TARA) has therefore become essential for managing these risks under mandatory regulations. However, existing TARA automation methods rely on static threat libraries, limiting their utility in the detailed, function-level analyses demanded by industry. This paper introduces DefenseWeaver, the first system that automates function-level TARA using component-specific details and large language models (LLMs). DefenseWeaver dynamically generates attack trees and risk evaluations from system configurations described in an extended OpenXSAM++ format, then employs a multi-agent framework to coordinate specialized LLM roles for more robust analysis. To further adapt to evolving threats and diverse standards, DefenseWeaver incorporates Low-Rank Adaptation (LoRA) fine-tuning and Retrieval-Augmented Generation (RAG) with expert-curated TARA reports. We validated DefenseWeaver through deployment in four automotive security projects, where it identified 11 critical attack paths, verified through penetration testing, and subsequently reported and remediated by the relevant automakers and suppliers. Additionally, DefenseWeaver demonstrated cross-domain adaptability, successfully applying to unmanned aerial vehicles (UAVs) and marine navigation systems. In comparison to human experts, DefenseWeaver outperformed manual attack tree generation across six assessment scenarios. Integrated into commercial cybersecurity platforms such as UAES and Xiaomi, DefenseWeaver has generated over 8,200 attack trees. These results highlight its ability to significantly reduce processing time, and its scalability and transformative impact on cybersecurity across industries.

摘要: 随着现代车辆发展为智能和互联系统，其日益增长的复杂性带来了巨大的网络安全风险。因此，威胁分析和风险评估（TARA）对于根据强制性法规管理这些风险至关重要。然而，现有的TARA自动化方法依赖于静态威胁库，限制了它们在行业要求的详细功能级分析中的实用性。本文介绍了DefenseWeaver，这是第一个使用组件特定细节和大型语言模型（LLM）自动化功能级TARA的系统。DefenseWeaver根据扩展OpenXSam++格式描述的系统配置动态生成攻击树和风险评估，然后采用多代理框架来协调专业的LLM角色以进行更稳健的分析。为了进一步适应不断变化的威胁和多样化的标准，DefenseWeaver将低等级适应（LoRA）微调和检索增强生成（RAG）与专家策划的TARA报告结合起来。我们通过在四个汽车安全项目中部署来验证DefenseWeaver，该项目识别了11条关键攻击路径，通过渗透测试进行验证，随后由相关汽车制造商和供应商报告和修复。此外，DefenseWeaver还展示了跨领域的适应性，成功应用于无人机（UFO）和海洋导航系统。与人类专家相比，DefenseWeaver在六种评估场景中的表现优于手动攻击树生成。DefenseWeaver集成到UAES和小米等商业网络安全平台中，已生成超过8，200个攻击树。这些结果凸显了它显着减少处理时间的能力，以及其可扩展性和对跨行业网络安全的变革性影响。



## **33. DREAM: Disentangling Risks to Enhance Safety Alignment in Multimodal Large Language Models**

梦想：理清风险以增强多模式大型语言模型的安全一致性 cs.CL

[NAACL 2025] The first four authors contribute equally, 23 pages,  repo at https://github.com/Kizna1ver/DREAM

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18053v1) [paper-pdf](http://arxiv.org/pdf/2504.18053v1)

**Authors**: Jianyu Liu, Hangyu Guo, Ranjie Duan, Xingyuan Bu, Yancheng He, Shilong Li, Hui Huang, Jiaheng Liu, Yucheng Wang, Chenchen Jing, Xingwei Qu, Xiao Zhang, Yingshui Tan, Yanan Wu, Jihao Gu, Yangguang Li, Jianke Zhu

**Abstract**: Multimodal Large Language Models (MLLMs) pose unique safety challenges due to their integration of visual and textual data, thereby introducing new dimensions of potential attacks and complex risk combinations. In this paper, we begin with a detailed analysis aimed at disentangling risks through step-by-step reasoning within multimodal inputs. We find that systematic multimodal risk disentanglement substantially enhances the risk awareness of MLLMs. Via leveraging the strong discriminative abilities of multimodal risk disentanglement, we further introduce \textbf{DREAM} (\textit{\textbf{D}isentangling \textbf{R}isks to \textbf{E}nhance Safety \textbf{A}lignment in \textbf{M}LLMs}), a novel approach that enhances safety alignment in MLLMs through supervised fine-tuning and iterative Reinforcement Learning from AI Feedback (RLAIF). Experimental results show that DREAM significantly boosts safety during both inference and training phases without compromising performance on normal tasks (namely oversafety), achieving a 16.17\% improvement in the SIUO safe\&effective score compared to GPT-4V. The data and code are available at https://github.com/Kizna1ver/DREAM.

摘要: 多模式大型语言模型（MLLM）由于集成了视觉和文本数据，带来了独特的安全挑战，从而引入了潜在攻击和复杂风险组合的新维度。在本文中，我们首先进行详细分析，旨在通过多模式输入中的逐步推理来解开风险。我们发现，系统性多模式风险理清大大增强了MLLM的风险意识。通过利用多模式风险解开的强大辨别能力，我们进一步引入了\textBF{DREAM}（\texttit {\textBF{D}isentangling \textBF{R}isks to \textBF{E}nhance Safety \textBF{A} lignation in \textBF{M} LLM}），这是一种新颖的方法，通过有监督的微调和来自人工智能反馈的迭代强化学习（RLAIF）来增强MLLM中的安全一致性。实验结果表明，DREAM在推理和训练阶段都显着提高了安全性，而不会影响正常任务的表现（即过度安全），与GPT-4V相比，SIUO安全有效评分提高了16.17%。数据和代码可在https://github.com/Kizna1ver/DREAM上获取。



## **34. Beyond Public Access in LLM Pre-Training Data**

LLM预培训数据中超越公众访问 cs.CL

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2505.00020v1) [paper-pdf](http://arxiv.org/pdf/2505.00020v1)

**Authors**: Sruly Rosenblat, Tim O'Reilly, Ilan Strauss

**Abstract**: Using a legally obtained dataset of 34 copyrighted O'Reilly Media books, we apply the DE-COP membership inference attack method to investigate whether OpenAI's large language models were trained on copyrighted content without consent. Our AUROC scores show that GPT-4o, OpenAI's more recent and capable model, demonstrates strong recognition of paywalled O'Reilly book content (AUROC = 82\%), compared to OpenAI's earlier model GPT-3.5 Turbo. In contrast, GPT-3.5 Turbo shows greater relative recognition of publicly accessible O'Reilly book samples. GPT-4o Mini, as a much smaller model, shows no knowledge of public or non-public O'Reilly Media content when tested (AUROC $\approx$ 50\%). Testing multiple models, with the same cutoff date, helps us account for potential language shifts over time that might bias our findings. These results highlight the urgent need for increased corporate transparency regarding pre-training data sources as a means to develop formal licensing frameworks for AI content training

摘要: 使用合法获得的34本受版权保护的O ' Reilly Media书籍的数据集，我们应用DE-COP成员资格推断攻击方法来调查OpenAI的大型语言模型是否在未经同意的情况下对受版权保护的内容进行训练。我们的AUROC评分显示，与OpenAI的早期模型GPT-3.5 Turbo相比，OpenAI的更新且功能强大的模型GPT-4 o表现出对付费O ' Reilly图书内容的强烈认可度（AUROC = 82%）。相比之下，GPT-3.5 Turbo对公开访问的O ' Reilly图书样本表现出更高的相对认可度。GPT-4 o Mini是一款小得多的型号，在测试时显示不了解公共或非公共O ' Reilly Media内容（AUROC $\approximate $50\%）。测试具有相同截止日期的多个模型可以帮助我们解释随着时间的推移可能会导致我们的发现偏差的潜在语言变化。这些结果凸显了迫切需要提高培训前数据源的企业透明度，作为开发人工智能内容培训正式许可框架的一种手段



## **35. Unified Attacks to Large Language Model Watermarks: Spoofing and Scrubbing in Unauthorized Knowledge Distillation**

对大型语言模型水印的统一攻击：未经授权的知识提炼中的欺骗和擦除 cs.CL

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17480v1) [paper-pdf](http://arxiv.org/pdf/2504.17480v1)

**Authors**: Xin Yi, Shunfan Zhengc, Linlin Wanga, Xiaoling Wang, Liang He

**Abstract**: Watermarking has emerged as a critical technique for combating misinformation and protecting intellectual property in large language models (LLMs). A recent discovery, termed watermark radioactivity, reveals that watermarks embedded in teacher models can be inherited by student models through knowledge distillation. On the positive side, this inheritance allows for the detection of unauthorized knowledge distillation by identifying watermark traces in student models. However, the robustness of watermarks against scrubbing attacks and their unforgeability in the face of spoofing attacks under unauthorized knowledge distillation remain largely unexplored. Existing watermark attack methods either assume access to model internals or fail to simultaneously support both scrubbing and spoofing attacks. In this work, we propose Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework that enables bidirectional attacks under unauthorized knowledge distillation. Our approach employs contrastive decoding to extract corrupted or amplified watermark texts via comparing outputs from the student model and weakly watermarked references, followed by bidirectional distillation to train new student models capable of watermark removal and watermark forgery, respectively. Extensive experiments show that CDG-KD effectively performs attacks while preserving the general performance of the distilled model. Our findings underscore critical need for developing watermarking schemes that are robust and unforgeable.

摘要: 水印已成为打击错误信息和保护大型语言模型（LLM）知识产权的关键技术。最近的一项发现称为水印放射性，揭示了教师模型中嵌入的水印可以通过知识蒸馏被学生模型继承。从积极的方面来说，这种继承允许通过识别学生模型中的水印痕迹来检测未经授权的知识提炼。然而，水印对擦洗攻击的鲁棒性及其在未经授权的知识提炼下面对欺骗攻击时的不可伪造性在很大程度上仍然没有被探索。现有的水印攻击方法要么假设访问模型内部，要么不能同时支持擦洗和欺骗攻击。在这项工作中，我们提出了对比解码引导的知识蒸馏（CDG-KD），一个统一的框架，使未经授权的知识蒸馏下的双向攻击。我们的方法采用对比解码提取损坏或放大的水印文本，通过比较输出的学生模型和弱水印的参考，然后通过双向蒸馏训练新的学生模型能够水印去除和水印伪造，分别。大量的实验表明，CDG-KD有效地执行攻击，同时保持蒸馏模型的一般性能。我们的研究结果强调了开发稳健且不可伪造的水印方案的迫切需要。



## **36. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

越狱镜头：以表象和电路的视角解读越狱机制 cs.CR

17 pages, 11 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2411.11114v2) [paper-pdf](http://arxiv.org/pdf/2411.11114v2)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Wenhui Zhang, Qinglong Wang, Rui Zheng

**Abstract**: Despite the outstanding performance of Large language Models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses. Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explained typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of jailbreak attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives~(which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on five mainstream LLMs under seven jailbreak strategies. Our evaluation reveals that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. This manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals. Notably, we find a strong and consistent correlation between representation deception and activation shift of key circuits across diverse jailbreak methods and multiple LLMs.

摘要: 尽管大型语言模型（LLM）在不同任务中表现出色，但它们很容易受到越狱攻击，其中对抗性提示是为了绕过其安全机制并引发意外响应。尽管越狱攻击很普遍，但对其潜在机制的了解仍然有限。最近的研究解释了典型的越狱行为（例如，模型拒绝响应的程度）通过分析越狱提示引起的LLM潜在空间的表示变化或识别有助于越狱攻击成功的关键神经元。然而，这些研究既没有探索不同的越狱模式，也没有提供从电路故障到表象变化的细粒度解释，在揭示越狱机制方面留下了重大空白。在本文中，我们提出了JailbreakLens，这是一个解释框架，从表示（揭示了越狱如何改变模型的危害性感知）和电路视角（通过识别导致漏洞的关键电路来揭示这些欺骗的原因）来分析越狱机制），在整个响应生成过程中跟踪它们的演变。然后，我们对七种越狱策略下的五种主流LLM的越狱行为进行了深入评估。我们的评估表明，越狱会放大强化肯定反应的成分，同时抑制那些产生拒绝反应的成分。这种操纵将模型表示转移到安全的集群以欺骗LLM，导致其提供详细的响应而不是拒绝。值得注意的是，我们发现不同越狱方法和多种LLM中的代表欺骗和关键电路的激活转变之间存在强烈且一致的相关性。



## **37. GraphRAG under Fire**

GraphRAG受到攻击 cs.LG

13 pages

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2501.14050v2) [paper-pdf](http://arxiv.org/pdf/2501.14050v2)

**Authors**: Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi Zhu, Tanqiu Jiang, Neil Gong, Ting Wang

**Abstract**: GraphRAG advances retrieval-augmented generation (RAG) by structuring external knowledge as multi-scale knowledge graphs, enabling language models to integrate both broad context and granular details in their generation. While GraphRAG has demonstrated success across domains, its security implications remain largely unexplored. To bridge this gap, this work examines GraphRAG's vulnerability to poisoning attacks, uncovering an intriguing security paradox: compared to conventional RAG, GraphRAG's graph-based indexing and retrieval enhance resilience against simple poisoning attacks; yet, the same features also create new attack surfaces. We present GRAGPoison, a novel attack that exploits shared relations in the underlying knowledge graph to craft poisoning text capable of compromising multiple queries simultaneously. GRAGPoison employs three key strategies: i) relation injection to introduce false knowledge, ii) relation enhancement to amplify poisoning influence, and iii) narrative generation to embed malicious content within coherent text. Empirical evaluation across diverse datasets and models shows that GRAGPoison substantially outperforms existing attacks in terms of effectiveness (up to 98\% success rate) and scalability (using less than 68\% poisoning text) on various GraphRAG-based systems. We also explore potential defensive measures and their limitations, identifying promising directions for future research.

摘要: GraphRAG通过将外部知识结构化为多尺度知识图，使语言模型能够在生成中集成广泛的上下文和粒度细节，从而推进了检索增强生成（RAG）。虽然GraphRAG在各个领域都取得了成功，但其安全影响在很大程度上仍未被探索。为了弥合这一差距，这项工作研究了GraphRAG对中毒攻击的脆弱性，揭示了一个有趣的安全悖论：与传统的RAG相比，GraphRAG的基于图形的索引和检索增强了针对简单中毒攻击的弹性;然而，相同的功能也创建了新的攻击表面。我们提出了GRAGPoison，这是一种新颖的攻击，它利用底层知识图中的共享关系来制作能够同时危害多个查询的中毒文本。GRAGPoison采用三种关键策略：i）关系注入以引入虚假知识，ii）关系增强以放大中毒影响，以及iii）叙事生成以将恶意内容嵌入连贯文本中。对各种数据集和模型的经验评估表明，GRAGPoison在各种基于GraphRAG的系统上的有效性（高达98%的成功率）和可扩展性（使用少于68%的中毒文本）方面大大优于现有的攻击。我们还探索潜在的防御措施及其局限性，为未来研究确定有希望的方向。



## **38. CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent**

CheatAgent：通过LLM代理攻击LLM授权的推荐系统 cs.CR

Accepted by KDD 2024;

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.13192v2) [paper-pdf](http://arxiv.org/pdf/2504.13192v2)

**Authors**: Liang-bo Ning, Shijie Wang, Wenqi Fan, Qing Li, Xin Xu, Hao Chen, Feiran Huang

**Abstract**: Recently, Large Language Model (LLM)-empowered recommender systems (RecSys) have brought significant advances in personalized user experience and have attracted considerable attention. Despite the impressive progress, the research question regarding the safety vulnerability of LLM-empowered RecSys still remains largely under-investigated. Given the security and privacy concerns, it is more practical to focus on attacking the black-box RecSys, where attackers can only observe the system's inputs and outputs. However, traditional attack approaches employing reinforcement learning (RL) agents are not effective for attacking LLM-empowered RecSys due to the limited capabilities in processing complex textual inputs, planning, and reasoning. On the other hand, LLMs provide unprecedented opportunities to serve as attack agents to attack RecSys because of their impressive capability in simulating human-like decision-making processes. Therefore, in this paper, we propose a novel attack framework called CheatAgent by harnessing the human-like capabilities of LLMs, where an LLM-based agent is developed to attack LLM-Empowered RecSys. Specifically, our method first identifies the insertion position for maximum impact with minimal input modification. After that, the LLM agent is designed to generate adversarial perturbations to insert at target positions. To further improve the quality of generated perturbations, we utilize the prompt tuning technique to improve attacking strategies via feedback from the victim RecSys iteratively. Extensive experiments across three real-world datasets demonstrate the effectiveness of our proposed attacking method.

摘要: 最近，基于大语言模型（LLM）的推荐系统（RecSys）在个性化用户体验方面带来了显着进步，并引起了相当大的关注。尽管取得了令人印象深刻的进展，但有关LLM授权的RecSys安全漏洞的研究问题仍然基本上没有得到充分的调查。考虑到安全和隐私问题，更实际的做法是专注于攻击黑匣子RecSys，攻击者只能观察系统的输入和输出。然而，由于处理复杂文本输入、规划和推理的能力有限，使用强化学习（RL）代理的传统攻击方法对于攻击LLM授权的RecSys并不有效。另一方面，LLM提供了前所未有的机会作为攻击代理来攻击RecSys，因为它们在模拟类人决策过程方面具有令人印象深刻的能力。因此，在本文中，我们通过利用LLM的类人能力提出了一种名为CheatAgent的新型攻击框架，其中开发了一个基于LLM的代理来攻击LLM授权的RecSys。具体来说，我们的方法首先识别插入位置，以最小的输入修改获得最大影响。之后，LLM代理被设计为生成对抗性扰动以插入目标位置。为了进一步提高生成的扰动的质量，我们利用即时调整技术通过受害者RecSys的反馈迭代改进攻击策略。三个现实世界数据集的广泛实验证明了我们提出的攻击方法的有效性。



## **39. Automatically Generating Rules of Malicious Software Packages via Large Language Model**

利用大型语言模型自动生成恶意软件包规则 cs.SE

14 pages, 11 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17198v1) [paper-pdf](http://arxiv.org/pdf/2504.17198v1)

**Authors**: XiangRui Zhang, HaoYu Chen, Yongzhong He, Wenjia Niu, Qiang Li

**Abstract**: Today's security tools predominantly rely on predefined rules crafted by experts, making them poorly adapted to the emergence of software supply chain attacks. To tackle this limitation, we propose a novel tool, RuleLLM, which leverages large language models (LLMs) to automate rule generation for OSS ecosystems. RuleLLM extracts metadata and code snippets from malware as its input, producing YARA and Semgrep rules that can be directly deployed in software development. Specifically, the rule generation task involves three subtasks: crafting rules, refining rules, and aligning rules. To validate RuleLLM's effectiveness, we implemented a prototype system and conducted experiments on the dataset of 1,633 malicious packages. The results are promising that RuleLLM generated 763 rules (452 YARA and 311 Semgrep) with a precision of 85.2\% and a recall of 91.8\%, outperforming state-of-the-art (SOTA) tools and scored-based approaches. We further analyzed generated rules and proposed a rule taxonomy: 11 categories and 38 subcategories.

摘要: 当今的安全工具主要依赖于专家制定的预定义规则，这使得它们无法很好地适应软件供应链攻击的出现。为了解决这一限制，我们提出了一种新颖的工具RuleLLM，它利用大型语言模型（LLM）来自动生成OSS生态系统的规则。RuleLLM从恶意软件中提取元数据和代码片段作为其输入，生成可以直接部署在软件开发中的YARA和Semgrep规则。具体来说，规则生成任务涉及三个子任务：制定规则、细化规则和对齐规则。为了验证RuleLLM的有效性，我们实现了一个原型系统，并对1，633个恶意包的数据集进行了实验。结果令人鼓舞，RuleLLM生成了763个规则（452个YARA和311个Semgrep），准确率为85.2%，召回率为91.8%，优于最先进的（SOTA）工具和基于分数的方法。我们进一步分析了生成的规则并提出了规则分类：11个类别和38个子类别。



## **40. Robo-Troj: Attacking LLM-based Task Planners**

Robo-Troj：攻击基于LLM的任务计划 cs.RO

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.17070v1) [paper-pdf](http://arxiv.org/pdf/2504.17070v1)

**Authors**: Mohaiminul Al Nahian, Zainab Altaweel, David Reitano, Sabbir Ahmed, Saumitra Lohokare, Shiqi Zhang, Adnan Siraj Rakin

**Abstract**: Robots need task planning methods to achieve goals that require more than individual actions. Recently, large language models (LLMs) have demonstrated impressive performance in task planning. LLMs can generate a step-by-step solution using a description of actions and the goal. Despite the successes in LLM-based task planning, there is limited research studying the security aspects of those systems. In this paper, we develop Robo-Troj, the first multi-trigger backdoor attack for LLM-based task planners, which is the main contribution of this work. As a multi-trigger attack, Robo-Troj is trained to accommodate the diversity of robot application domains. For instance, one can use unique trigger words, e.g., "herical", to activate a specific malicious behavior, e.g., cutting hand on a kitchen robot. In addition, we develop an optimization method for selecting the trigger words that are most effective. Through demonstrating the vulnerability of LLM-based planners, we aim to promote the development of secured robot systems.

摘要: 机器人需要任务规划方法来实现不仅仅需要个人行动的目标。最近，大型语言模型（LLM）在任务规划方面表现出令人印象深刻的性能。LLM可以使用行动和目标的描述生成分步解决方案。尽管基于LLM的任务规划取得了成功，但研究这些系统安全方面的研究有限。本文中，我们开发了Robo-Troj，这是针对基于LLM的任务规划器的第一个多触发后门攻击，这是这项工作的主要贡献。作为一种多触发攻击，Robo-Troj经过训练以适应机器人应用领域的多样性。例如，可以使用独特的触发词，例如，“herical”，激活特定的恶意行为，例如，厨房机器人上的割伤手。此外，我们还开发了一种优化方法来选择最有效的触发词。通过展示基于LLM的规划者的脆弱性，我们的目标是促进安全机器人系统的开发。



## **41. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

交易魔鬼决赛：通过股市和Bayesian优化进行后门攻击 cs.LG

END (will never be modified again!!) :Jumps-Diffusion and stock  market: Better quantify uncertainty in financial simulations

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2407.14573v7) [paper-pdf](http://arxiv.org/pdf/2407.14573v7)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.

摘要: 自生成人工智能出现以来，每家公司和研究人员都在争先恐后地开发自己的生成模型，无论是否商业化。鉴于这些强大的新工具的大量用户，目前还没有本质上可验证的方法来从头解释LLM（大型语言模型）学习时会发生什么。例如，那些基于自动语音识别系统的系统，它们必须依赖于从整个网络收集的大量数据来产生快速有效的结果，在本文中，我们开发了一种名为MarketBackFinal 2.0的后门攻击，基于声学数据中毒，MarketBackFinal 2.0主要基于现代股市模型。为了显示可能依赖LLM的基于语音的转换器可能存在的漏洞。



## **42. Safety Pretraining: Toward the Next Generation of Safe AI**

安全预培训：迈向下一代安全人工智能 cs.LG

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16980v1) [paper-pdf](http://arxiv.org/pdf/2504.16980v1)

**Authors**: Pratyush Maini, Sachin Goyal, Dylan Sam, Alex Robey, Yash Savani, Yiding Jiang, Andy Zou, Zacharcy C. Lipton, J. Zico Kolter

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes settings, the risk of generating harmful or toxic content remains a central challenge. Post-hoc alignment methods are brittle: once unsafe patterns are learned during pretraining, they are hard to remove. We present a data-centric pretraining framework that builds safety into the model from the start. Our contributions include: (i) a safety classifier trained on 10,000 GPT-4 labeled examples, used to filter 600B tokens; (ii) the largest synthetic safety dataset to date (100B tokens) generated via recontextualization of harmful web data; (iii) RefuseWeb and Moral Education datasets that convert harmful prompts into refusal dialogues and web-style educational material; (iv) Harmfulness-Tag annotations injected during pretraining to flag unsafe content and steer away inference from harmful generations; and (v) safety evaluations measuring base model behavior before instruction tuning. Our safety-pretrained models reduce attack success rates from 38.8% to 8.4% with no performance degradation on standard LLM safety benchmarks.

摘要: 随着大型语言模型（LLM）越来越多地部署在高风险环境中，生成有害或有毒内容的风险仍然是一个核心挑战。事后对齐方法很脆弱：一旦在预训练期间学习到不安全的模式，它们就很难被删除。我们提出了一个以数据为中心的预训练框架，从一开始就将安全性构建到模型中。我们的贡献包括：（i）在10，000个GPT-4标记的示例上训练的安全分类器，用于过滤600 B令牌;（ii）迄今为止最大的合成安全数据集（iii）RefuseWeb和道德教育数据集，将有害提示转换为拒绝对话和网络风格的教育材料;（iv）在预训练期间注入有害标签注释，以标记不安全的内容并引导远离有害代的推断;以及（v）在指令调优之前测量基础模型行为的安全评估。我们的安全预训练模型将攻击成功率从38.8%降低到8.4%，并且在标准LLM安全基准测试中没有性能下降。



## **43. Context-Enhanced Vulnerability Detection Based on Large Language Model**

基于大语言模型的上下文增强漏洞检测 cs.SE

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16877v1) [paper-pdf](http://arxiv.org/pdf/2504.16877v1)

**Authors**: Yixin Yang, Bowen Xu, Xiang Gao, Hailong Sun

**Abstract**: Vulnerability detection is a critical aspect of software security. Accurate detection is essential to prevent potential security breaches and protect software systems from malicious attacks. Recently, vulnerability detection methods leveraging deep learning and large language models (LLMs) have garnered increasing attention. However, existing approaches often focus on analyzing individual files or functions, which limits their ability to gather sufficient contextual information. Analyzing entire repositories to gather context introduces significant noise and computational overhead. To address these challenges, we propose a context-enhanced vulnerability detection approach that combines program analysis with LLMs. Specifically, we use program analysis to extract contextual information at various levels of abstraction, thereby filtering out irrelevant noise. The abstracted context along with source code are provided to LLM for vulnerability detection. We investigate how different levels of contextual granularity improve LLM-based vulnerability detection performance. Our goal is to strike a balance between providing sufficient detail to accurately capture vulnerabilities and minimizing unnecessary complexity that could hinder model performance. Based on an extensive study using GPT-4, DeepSeek, and CodeLLaMA with various prompting strategies, our key findings includes: (1) incorporating abstracted context significantly enhances vulnerability detection effectiveness; (2) different models benefit from distinct levels of abstraction depending on their code understanding capabilities; and (3) capturing program behavior through program analysis for general LLM-based code analysis tasks can be a direction that requires further attention.

摘要: 漏洞检测是软件安全的一个重要方面。准确的检测对于防止潜在的安全漏洞和保护软件系统免受恶意攻击至关重要。最近，利用深度学习和大型语言模型（LLM）的漏洞检测方法越来越受到关注。然而，现有的方法往往侧重于分析单个文件或函数，这限制了它们收集足够的上下文信息的能力。分析整个存储库以收集上下文会带来显着的噪音和计算费用。为了应对这些挑战，我们提出了一种上下文增强的漏洞检测方法，该方法将程序分析与LLM相结合。具体来说，我们使用程序分析来提取各个抽象级别的上下文信息，从而过滤掉不相关的噪音。将抽象上下文和源代码提供给LLM以进行漏洞检测。我们研究不同级别的上下文粒度如何提高基于LLM的漏洞检测性能。我们的目标是在提供足够的细节以准确捕获漏洞和最大限度地减少可能阻碍模型性能的不必要的复杂性之间取得平衡。基于使用GPT-4、DeepSeek和CodeLLaMA以及各种提示策略的广泛研究，我们的主要发现包括：（1）合并抽象上下文显着增强了漏洞检测的有效性;（2）不同的模型受益于不同的抽象级别，具体取决于它们的代码理解能力;以及（3）通过程序分析来捕获程序行为以进行一般基于LLM的代码分析任务可能是需要进一步关注的方向。



## **44. aiXamine: Simplified LLM Safety and Security**

aiXamine：简化的LLM安全和安保 cs.CR

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.14985v2) [paper-pdf](http://arxiv.org/pdf/2504.14985v2)

**Authors**: Fatih Deniz, Dorde Popovic, Yazan Boshmaf, Euisuh Jeong, Minhaj Ahmad, Sanjay Chawla, Issa Khalil

**Abstract**: Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.

摘要: 评估大型语言模型（LLM）的安全性和保障性仍然是一项复杂的任务，通常需要用户在临时基准、数据集、指标和报告格式的碎片化环境中进行导航。为了应对这一挑战，我们推出了aiXamine，这是一个针对LLM安全性的全面黑匣子评估平台。aiXamine集成了40多个测试（即，基准）组织成八个关键服务，针对安全和保障的特定维度：对抗稳健性、代码安全性、公平性和偏见、幻觉、模型和数据隐私、分发外（OOD）稳健性、过度拒绝和安全对齐。该平台将评估结果汇总到每个模型的单个详细报告中，提供模型性能、测试示例和丰富的可视化的详细细分。我们使用aiXamine评估了50多个公开和专有的LLM，进行了超过2000次检查。我们的研究结果揭示了领先模型中的显着漏洞，包括OpenAI GPT-4 o中容易受到对抗攻击、xAI Grok-3中的偏见输出以及Google Gemini 2.0中的隐私弱点。此外，我们观察到开源模型可以在特定服务中匹配或超过专有模型，例如安全性一致、公平性和偏差以及OOD稳健性。最后，我们确定了蒸馏策略、模型大小、训练方法和架构选择之间的权衡。



## **45. Amplified Vulnerabilities: Structured Jailbreak Attacks on LLM-based Multi-Agent Debate**

扩大的漏洞：对基于LLM的多代理辩论的结构化越狱攻击 cs.CR

33 pages, 5 figures

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16489v1) [paper-pdf](http://arxiv.org/pdf/2504.16489v1)

**Authors**: Senmao Qi, Yifei Zou, Peng Li, Ziyi Lin, Xiuzhen Cheng, Dongxiao Yu

**Abstract**: Multi-Agent Debate (MAD), leveraging collaborative interactions among Large Language Models (LLMs), aim to enhance reasoning capabilities in complex tasks. However, the security implications of their iterative dialogues and role-playing characteristics, particularly susceptibility to jailbreak attacks eliciting harmful content, remain critically underexplored. This paper systematically investigates the jailbreak vulnerabilities of four prominent MAD frameworks built upon leading commercial LLMs (GPT-4o, GPT-4, GPT-3.5-turbo, and DeepSeek) without compromising internal agents. We introduce a novel structured prompt-rewriting framework specifically designed to exploit MAD dynamics via narrative encapsulation, role-driven escalation, iterative refinement, and rhetorical obfuscation. Our extensive experiments demonstrate that MAD systems are inherently more vulnerable than single-agent setups. Crucially, our proposed attack methodology significantly amplifies this fragility, increasing average harmfulness from 28.14% to 80.34% and achieving attack success rates as high as 80% in certain scenarios. These findings reveal intrinsic vulnerabilities in MAD architectures and underscore the urgent need for robust, specialized defenses prior to real-world deployment.

摘要: 多智能体辩论（MAD），利用大型语言模型（LLM）之间的协作交互，旨在增强复杂任务中的推理能力。然而，其迭代对话和角色扮演特性的安全影响，特别是对引发有害内容的越狱攻击的敏感性，仍然严重不足。本文系统地研究了基于领先的商业LLM（GPT-4 o，GPT-4，GPT-3.5-turbo和DeepSeek）构建的四个突出MAD框架的越狱漏洞，而不会影响内部代理。我们介绍了一种新的结构化的重写框架，专门设计来利用MAD动态通过叙事封装，角色驱动的升级，迭代细化，修辞混淆。我们广泛的实验表明，MAD系统本质上比单代理设置更容易受到攻击。至关重要的是，我们提出的攻击方法显着放大了这种脆弱性，将平均危害性从28.14%增加到80.34%，并在某些情况下实现高达80%的攻击成功率。这些发现揭示了MAD架构中的内在漏洞，并强调了在现实世界部署之前对强大、专业化的防御的迫切需要。



## **46. Large Language Model Sentinel: LLM Agent for Adversarial Purification**

大型语言模型Sentinel：对抗性纯化的LLM代理 cs.CL

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2405.20770v4) [paper-pdf](http://arxiv.org/pdf/2405.20770v4)

**Authors**: Guang Lin, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.

摘要: 在过去的两年里，大型语言模型（LLM）的使用迅速发展。虽然这些LLM提供了相当大的便利，但它们也引发了安全问题，因为LLM很容易受到一些精心设计的文本扰动的对抗攻击。本文中，我们介绍了一种名为Large LAnguage MOdel Sentinel（LLAMOS）的新型防御技术，该技术旨在通过在将对抗性文本示例输入目标LLM之前对其进行纯化来增强LLM的对抗性鲁棒性。我们的方法包括两个主要部分：a）代理指令，可以模拟新代理进行对抗性防御，改变最小字符以保持句子的原始含义，同时防御攻击; b）防御指导，提供修改干净或对抗性示例的策略，以确保有效的防御和目标LLM的准确输出。值得注意的是，即使没有从对抗性例子中学习，防御代理也表现出强大的防御能力。此外，我们还进行了一项有趣的对抗实验，我们开发了两个代理人，一个用于防御，一个用于攻击，并让它们进行相互对抗。在对抗互动过程中，两个主体都没有完全击败对方。开源和闭源LLM上的大量实验表明，我们的方法可以有效地防御对抗攻击，从而增强对抗鲁棒性。



## **47. Attention Tracker: Detecting Prompt Injection Attacks in LLMs**

注意力追踪器：检测LLM中的即时注入攻击 cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2411.00348v2) [paper-pdf](http://arxiv.org/pdf/2411.00348v2)

**Authors**: Kuo-Han Hung, Ching-Yun Ko, Ambrish Rawat, I-Hsin Chung, Winston H. Hsu, Pin-Yu Chen

**Abstract**: Large Language Models (LLMs) have revolutionized various domains but remain vulnerable to prompt injection attacks, where malicious inputs manipulate the model into ignoring original instructions and executing designated action. In this paper, we investigate the underlying mechanisms of these attacks by analyzing the attention patterns within LLMs. We introduce the concept of the distraction effect, where specific attention heads, termed important heads, shift focus from the original instruction to the injected instruction. Building on this discovery, we propose Attention Tracker, a training-free detection method that tracks attention patterns on instruction to detect prompt injection attacks without the need for additional LLM inference. Our method generalizes effectively across diverse models, datasets, and attack types, showing an AUROC improvement of up to 10.0% over existing methods, and performs well even on small LLMs. We demonstrate the robustness of our approach through extensive evaluations and provide insights into safeguarding LLM-integrated systems from prompt injection vulnerabilities.

摘要: 大型语言模型（LLM）已经彻底改变了各个领域，但仍然容易受到提示注入攻击，恶意输入操纵模型忽略原始指令并执行指定操作。在本文中，我们通过分析LLM内的注意力模式来研究这些攻击的潜在机制。我们引入了分心效应的概念，即特定的注意力头（称为重要头）将焦点从原始指令转移到注入的指令。在这一发现的基础上，我们提出了注意力追踪器，这是一种免训练的检测方法，可以跟踪指令上的注意力模式，以检测即时注入攻击，而不需要额外的LLM推断。我们的方法在不同的模型、数据集和攻击类型中有效推广，显示AUROC比现有方法提高了高达10.0%，即使在小型LLM上也表现良好。我们通过广泛的评估证明了我们方法的稳健性，并提供了保护LLM集成系统免受即时注入漏洞的见解。



## **48. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

BaThe：通过将有害指令视为后门触发来防御多模式大型语言模型中的越狱攻击 cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2408.09093v3) [paper-pdf](http://arxiv.org/pdf/2408.09093v3)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.

摘要: 多模式大型语言模型（MLLM）在各种多模式任务中展示了令人印象深刻的性能。另一方面，额外图像形态的集成可能会允许恶意用户在图像中注入有害内容以进行越狱。与基于文本的LLM不同，对手需要选择离散令牌来使用特定算法隐藏其恶意意图，图像信号的连续性为对手提供了注入有害意图的直接机会。在这项工作中，我们提出了$\textBF{BaThe}$（$\textBF{BA}$ckdoor $\textBF{T}$rigger S$\textBF{h}$i$\textBF{e}$ld），这是一种简单而有效的越狱防御机制。我们的工作受到最近对生成性语言模型中越狱后门攻击和虚拟提示后门攻击的研究的启发。越狱后门攻击使用有害指令与手工制作的字符串相结合作为触发器，使后门模型生成禁止的响应。我们假设有害指令可以充当触发器，如果我们将拒绝响应设置为触发响应，那么后门模型就可以防御越狱攻击。我们通过利用虚拟拒绝提示来实现这一目标，类似于虚拟提示后门攻击。我们将虚拟拒绝提示嵌入到软文本嵌入中，我们称之为“wedge”。我们的全面实验表明，BaThe有效地缓解了各种类型的越狱攻击，并且能够抵御不可见的攻击，对MLLM的性能影响最小。



## **49. Red Team Diffuser: Exposing Toxic Continuation Vulnerabilities in Vision-Language Models via Reinforcement Learning**

Red Team Diffuser：通过强化学习暴露视觉语言模型中的有毒连续漏洞 cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2503.06223v2) [paper-pdf](http://arxiv.org/pdf/2503.06223v2)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The growing deployment of large Vision-Language Models (VLMs) exposes critical safety gaps in their alignment mechanisms. While existing jailbreak studies primarily focus on VLMs' susceptibility to harmful instructions, we reveal a fundamental yet overlooked vulnerability: toxic text continuation, where VLMs produce highly toxic completions when prompted with harmful text prefixes paired with semantically adversarial images. To systematically study this threat, we propose Red Team Diffuser (RTD), the first red teaming diffusion model that coordinates adversarial image generation and toxic continuation through reinforcement learning. Our key innovations include dynamic cross-modal attack and stealth-aware optimization. For toxic text prefixes from an LLM safety benchmark, we conduct greedy search to identify optimal image prompts that maximally induce toxic completions. The discovered image prompts then drive RL-based diffusion model fine-tuning, producing semantically aligned adversarial images that boost toxicity rates. Stealth-aware optimization introduces joint adversarial rewards that balance toxicity maximization (via Detoxify classifier) and stealthiness (via BERTScore), circumventing traditional noise-based adversarial patterns. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% over text-only baselines on the original attack set and 8.91% on an unseen set, proving generalization capability. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. Our findings expose two critical flaws in current VLM alignment: (1) failure to prevent toxic continuation from harmful prefixes, and (2) overlooking cross-modal attack vectors. These results necessitate a paradigm shift toward multimodal red teaming in safety evaluations.

摘要: 大型视觉语言模型（VLM）的不断增加的部署暴露了其对齐机制中的关键安全漏洞。虽然现有的越狱研究主要关注VLM对有害指令的敏感性，但我们揭示了一个基本但被忽视的弱点：有毒文本延续，当提示有害文本前置与语义对抗图像配对时，VLM会产生剧毒的完成。为了系统性地研究这种威胁，我们提出了Red Team Distuser（RTI），这是第一个红色团队扩散模型，通过强化学习协调对抗图像生成和有毒延续。我们的关键创新包括动态跨模式攻击和隐身优化。对于LLM安全基准中的有毒文本前置，我们进行贪婪搜索以识别最大限度地引发有毒完成的最佳图像提示。发现的图像提示然后驱动基于RL的扩散模型微调，产生语义对齐的对抗图像，从而提高毒性率。潜行感知优化引入了联合对抗奖励，平衡毒性最大化（通过Dealfy分类器）和潜行性（通过BERTScore），规避了传统的基于噪音的对抗模式。实验结果证明了RTI的有效性，在原始攻击集中，LLaVA输出的毒性率比纯文本基线增加了10.69%，在未见集上增加了8.91%，证明了概括能力。此外，RTI具有较强的跨模型转移性，使Gemini的毒性率提高了5.1%，对LLaMA的毒性率提高了26.83%。我们的研究结果暴露了当前VLM对齐中的两个关键缺陷：（1）未能防止有害前置的有毒延续，以及（2）忽视了跨模式攻击载体。这些结果需要在安全性评估中向多模式红色团队转变。



## **50. Research on Cloud Platform Network Traffic Monitoring and Anomaly Detection System based on Large Language Models**

基于大语言模型的云平台网络流量监控与异常检测系统研究 cs.NI

Proceedings of 2025 IEEE 7th International Conference on  Communications, Information System and Computer Engineering (CISCE 2025)

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.17807v1) [paper-pdf](http://arxiv.org/pdf/2504.17807v1)

**Authors**: Ze Yang, Yihong Jin, Juntian Liu, Xinhe Xu, Yihan Zhang, Shuyang Ji

**Abstract**: The rapidly evolving cloud platforms and the escalating complexity of network traffic demand proper network traffic monitoring and anomaly detection to ensure network security and performance. This paper introduces a large language model (LLM)-based network traffic monitoring and anomaly detection system. In addition to existing models such as autoencoders and decision trees, we harness the power of large language models for processing sequence data from network traffic, which allows us a better capture of underlying complex patterns, as well as slight fluctuations in the dataset. We show for a given detection task, the need for a hybrid model that incorporates the attention mechanism of the transformer architecture into a supervised learning framework in order to achieve better accuracy. A pre-trained large language model analyzes and predicts the probable network traffic, and an anomaly detection layer that considers temporality and context is added. Moreover, we present a novel transfer learning-based methodology to enhance the model's effectiveness to quickly adapt to unknown network structures and adversarial conditions without requiring extensive labeled datasets. Actual results show that the designed model outperforms traditional methods in detection accuracy and computational efficiency, effectively identify various network anomalies such as zero-day attacks and traffic congestion pattern, and significantly reduce the false positive rate.

摘要: 快速发展的云平台和不断升级的网络流量复杂性需要适当的网络流量监控和异常检测，以确保网络安全和性能。本文介绍了一种基于大语言模型（LLM）的网络流量监控和异常检测系统。除了自动编码器和决策树等现有模型外，我们还利用大型语言模型的功能来处理网络流量中的序列数据，这使我们能够更好地捕捉底层复杂模式以及数据集中的轻微波动。我们表明，对于给定的检测任务，需要一个混合模型，该模型将Transformer架构的注意力机制融入到监督学习框架中，以实现更好的准确性。预先训练的大型语言模型分析和预测可能的网络流量，并添加了考虑时间性和上下文的异常检测层。此外，我们提出了一种基于迁移学习的新型方法，以增强模型的有效性，以快速适应未知的网络结构和对抗条件，而不需要大量的标记数据集。实际结果表明，所设计的模型在检测准确率和计算效率方面优于传统方法，有效识别零日攻击和交通拥堵模式等各种网络异常，显着降低误报率。



