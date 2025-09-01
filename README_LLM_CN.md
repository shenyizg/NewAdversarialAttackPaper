# Latest Large Language Model Attack Papers
**update at 2025-09-01 10:11:45**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. WebInject: Prompt Injection Attack to Web Agents**

WebInject：对Web Agent的即时注入攻击 cs.LG

EMNLP 2025 main

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2505.11717v3) [paper-pdf](http://arxiv.org/pdf/2505.11717v3)

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.

摘要: 基于多模式大型语言模型（MLLM）的Web代理通过基于网页屏幕截图生成动作来与网页环境交互。在这项工作中，我们提出了WebInjects，这是一种提示注入攻击，它操纵网页环境以诱导Web代理执行攻击者指定的操作。我们的攻击对渲染网页的原始像素值添加了扰动。这些受干扰的像素被映射到屏幕截图后，扰动会导致Web代理执行攻击者指定的操作。我们将寻找扰动的任务定义为优化问题。解决这个问题的一个关键挑战是原始像素值和屏幕截图之间的映射是不可微的，因此很难将梯度反向传播到扰动。为了克服这个问题，我们训练神经网络来逼近映射，并应用投影梯度下降来解决重新制定的优化问题。对多个数据集的广泛评估表明，WebInib非常有效，并且显着优于基线。



## **2. Detecting Stealthy Data Poisoning Attacks in AI Code Generators**

检测人工智能代码生成器中的隐形数据中毒攻击 cs.CR

Accepted to the 3rd IEEE International Workshop on Reliable and  Secure AI for Software Engineering (ReSAISE, 2025), co-located with ISSRE  2025

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21636v1) [paper-pdf](http://arxiv.org/pdf/2508.21636v1)

**Authors**: Cristina Improta

**Abstract**: Deep learning (DL) models for natural language-to-code generation have become integral to modern software development pipelines. However, their heavy reliance on large amounts of data, often collected from unsanitized online sources, exposes them to data poisoning attacks, where adversaries inject malicious samples to subtly bias model behavior. Recent targeted attacks silently replace secure code with semantically equivalent but vulnerable implementations without relying on explicit triggers to launch the attack, making it especially hard for detection methods to distinguish clean from poisoned samples. We present a systematic study on the effectiveness of existing poisoning detection methods under this stealthy threat model. Specifically, we perform targeted poisoning on three DL models (CodeBERT, CodeT5+, AST-T5), and evaluate spectral signatures analysis, activation clustering, and static analysis as defenses. Our results show that all methods struggle to detect triggerless poisoning, with representation-based approaches failing to isolate poisoned samples and static analysis suffering false positives and false negatives, highlighting the need for more robust, trigger-independent defenses for AI-assisted code generation.

摘要: 用于自然语言到代码生成的深度学习（DL）模型已成为现代软件开发管道的组成部分。然而，它们严重依赖大量数据，这些数据通常是从未经清理的在线来源收集的，这使它们面临数据中毒攻击，对手会注入恶意样本以微妙地偏向模型行为。最近的有针对性的攻击以语义等效但脆弱的实现悄然取代安全代码，而不依赖显式触发器来发起攻击，这使得检测方法特别难以区分干净样本和有毒样本。我们对这种隐形威胁模型下现有中毒检测方法的有效性进行了系统研究。具体来说，我们对三种DL模型（CodeBRT、CodeT 5+、AST-T5）执行有针对性的中毒，并评估光谱特征分析、激活集群和静态分析作为防御措施。我们的结果表明，所有方法都很难检测无指示器中毒，基于表示的方法无法隔离中毒样本，静态分析会出现假阳性和假阴性，这凸显了人工智能辅助代码生成需要更强大、独立于指示器的防御。



## **3. SoK: Large Language Model-Generated Textual Phishing Campaigns End-to-End Analysis of Generation, Characteristics, and Detection**

SoK：大型语言模型生成的文本网络钓鱼活动生成、特征和检测的端到端分析 cs.CR

13 pages, 3 tables, 4 figures

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21457v1) [paper-pdf](http://arxiv.org/pdf/2508.21457v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Phishing is a pervasive form of social engineering in which attackers impersonate trusted entities to steal information or induce harmful actions. Text-based phishing dominates for its low cost, scalability, and concealability, advantages recently amplified by large language models (LLMs) that enable ``Phishing-as-a-Service'' attacks at scale within minutes. Despite the growing research into LLM-facilitated phishing attacks, consolidated systematic research on the phishing attack life cycle remains scarce. In this work, we present the first systematization of knowledge (SoK) on LLM-generated phishing, offering an end-to-end analysis that spans generation techniques, attack features, and mitigation strategies. We introduce Generation-Characterization-Defense (GenCharDef), which systematizes the ways in which LLM-generated phishing differs from traditional phishing across methodologies, security perspectives, data dependencies, and evaluation practices. This framework highlights unique challenges of LLM-driven phishing, providing a coherent foundation for understanding the evolving threat landscape and guiding the design of more resilient defenses.

摘要: 网络钓鱼是一种普遍存在的社会工程形式，攻击者冒充受信任实体来窃取信息或诱导有害行为。基于文本的网络钓鱼因其低成本、可扩展性和可隐藏性而占据主导地位，这些优势最近被大型语言模型（LLM）放大，这些模型能够在几分钟内实现“网络钓鱼即服务”攻击。尽管对LLM支持的网络钓鱼攻击的研究越来越多，但对网络钓鱼攻击生命周期的综合系统研究仍然很少。在这项工作中，我们首次介绍了LLM生成的网络钓鱼知识系统化（SoK），提供涵盖生成技术、攻击特征和缓解策略的端到端分析。我们引入了生成-特征-防御（GenCharDef），它系统化了LLM生成的网络钓鱼在方法论、安全角度、数据依赖性和评估实践方面与传统网络钓鱼的不同之处。该框架强调了LLM驱动的网络钓鱼的独特挑战，为了解不断变化的威胁格局并指导设计更具弹性的防御系统提供了连贯的基础。



## **4. Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review**

发布到Perish：对LLM辅助同行评审的即时注入攻击 cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.20863v2) [paper-pdf](http://arxiv.org/pdf/2508.20863v2)

**Authors**: Matteo Gioele Collu, Umberto Salviati, Roberto Confalonieri, Mauro Conti, Giovanni Apruzzese

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.

摘要: 大型语言模型（LLM）越来越多地融入到科学同行评审过程中，这引发了有关其可靠性和操纵弹性的新问题。在这项工作中，我们调查了隐藏的提示注入攻击的可能性，即作者在论文的PDF中嵌入对抗性文本以影响LLM生成的评论。我们首先正式化三种不同的威胁模型，这些模型设想攻击者具有不同的动机--并非所有这些都暗示着恶意意图。对于每个威胁模型，我们设计了对抗性提示，这些提示对人类读者来说是不可见的，但可以将LLM的输出引导到作者想要的结果。通过对领域学者的用户研究，我们得出了四个代表性的审查提示，用于从法学硕士那里获得同行审查。然后，我们评估对抗提示在（i）不同的审查提示、（ii）不同的基于LLM的商业系统和（iii）不同的同行评审论文中的稳健性。我们的结果表明，对抗性提示可以可靠地误导LLM，有时会对“诚实但懒惰”的评论者产生不利影响。最后，我们提出并根据经验评估了在自动内容检查下降低对抗性提示可检测性的方法。



## **5. Lethe: Purifying Backdoored Large Language Models with Knowledge Dilution**

Lethe：通过知识稀释净化后门大型语言模型 cs.CL

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21004v1) [paper-pdf](http://arxiv.org/pdf/2508.21004v1)

**Authors**: Chen Chen, Yuchen Sun, Jiaxin Gao, Xueluan Gong, Qian Wang, Ziyao Wang, Yongsen Zheng, Kwok-Yan Lam

**Abstract**: Large language models (LLMs) have seen significant advancements, achieving superior performance in various Natural Language Processing (NLP) tasks. However, they remain vulnerable to backdoor attacks, where models behave normally for standard queries but generate harmful responses or unintended output when specific triggers are activated. Existing backdoor defenses either lack comprehensiveness, focusing on narrow trigger settings, detection-only mechanisms, and limited domains, or fail to withstand advanced scenarios like model-editing-based, multi-trigger, and triggerless attacks. In this paper, we present LETHE, a novel method to eliminate backdoor behaviors from LLMs through knowledge dilution using both internal and external mechanisms. Internally, LETHE leverages a lightweight dataset to train a clean model, which is then merged with the backdoored model to neutralize malicious behaviors by diluting the backdoor impact within the model's parametric memory. Externally, LETHE incorporates benign and semantically relevant evidence into the prompt to distract LLM's attention from backdoor features. Experimental results on classification and generation domains across 5 widely used LLMs demonstrate that LETHE outperforms 8 state-of-the-art defense baselines against 8 backdoor attacks. LETHE reduces the attack success rate of advanced backdoor attacks by up to 98% while maintaining model utility. Furthermore, LETHE has proven to be cost-efficient and robust against adaptive backdoor attacks.

摘要: 大型语言模型（LLM）取得了重大进步，在各种自然语言处理（NLP）任务中实现了卓越的性能。然而，它们仍然容易受到后门攻击，其中模型对于标准查询表现正常，但在激活特定触发器时会生成有害响应或意外输出。现有的后门防御要么缺乏全面性，专注于狭窄的触发设置、仅检测机制和有限的域，要么无法抵御基于模型编辑、多触发和无触发攻击等高级场景。在本文中，我们提出了LETHE，这是一种利用内部和外部机制通过知识稀释消除LLM后门行为的新型方法。在内部，LETHE利用轻量级数据集来训练干净的模型，然后将其与后门模型合并，通过稀释模型参数记忆中的后门影响来中和恶意行为。从外部来看，LETHE将良性且语义相关的证据融入到提示中，以转移LLM对后门功能的注意力。对5个广泛使用的LLM的分类和生成域的实验结果表明，LETHE针对8种后门攻击的性能优于8种最先进的防御基线。LETHE将高级后门攻击的攻击成功率降低高达98%，同时保持模型实用性。此外，LETHE已被证明具有成本效益，并且可以抵御自适应后门攻击。



## **6. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

EmantSleuth：通过语义意图不变性检测提示注入 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20890v1) [paper-pdf](http://arxiv.org/pdf/2508.20890v1)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.

摘要: 大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，从虚拟助手到自治代理。然而，它们的灵活性也引入了新的攻击向量，特别是提示注入（PI），其中攻击者通过精心制作的输入操纵模型行为。随着攻击者不断地使用释义、混淆甚至多任务注入策略，现有的基准不再足以捕获所有新兴威胁。   为了解决这一差距，我们构建了一个新的基准，系统地扩展了以前的努力。我们的基准涵盖了两种广泛使用的现有基准，同时引入了新的操纵技术和多任务场景，从而提供了更全面的评估设置。我们发现，现有的防御虽然在原始基准上有效，但在我们的基准下表现出明显的弱点，这凸显了对更强大解决方案的需求。我们的关键见解是，虽然攻击形式可能会有所不同，但对手的意图（注入未经授权的任务）保持不变。在这一观察的基础上，我们提出了EmittSleuth，这是一个面向语义的防御框架，它通过对任务级意图而不是表面特征进行推理来检测提示注入。在最先进的基准测试中进行评估后，AktSleuth始终优于现有的防御，同时保持相当的运行时间和成本效率。这些结果表明，基于意图的语义推理提供了一个强大的，有效的，和可推广的策略，以抵御不断发展的即时注入威胁的LLM。



## **7. Multi-Agent Penetration Testing AI for the Web**

Web多代理渗透测试人工智能 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20816v1) [paper-pdf](http://arxiv.org/pdf/2508.20816v1)

**Authors**: Isaac David, Arthur Gervais

**Abstract**: AI-powered development platforms are making software creation accessible to a broader audience, but this democratization has triggered a scalability crisis in security auditing. With studies showing that up to 40% of AI-generated code contains vulnerabilities, the pace of development now vastly outstrips the capacity for thorough security assessment.   We present MAPTA, a multi-agent system for autonomous web application security assessment that combines large language model orchestration with tool-grounded execution and end-to-end exploit validation. On the 104-challenge XBOW benchmark, MAPTA achieves 76.9% overall success with perfect performance on SSRF and misconfiguration vulnerabilities, 83% success on broken authorization, and strong results on injection attacks including server-side template injection (85%) and SQL injection (83%). Cross-site scripting (57%) and blind SQL injection (0%) remain challenging. Our comprehensive cost analysis across all challenges totals $21.38 with a median cost of $0.073 for successful attempts versus $0.357 for failures. Success correlates strongly with resource efficiency, enabling practical early-stopping thresholds at approximately 40 tool calls or $0.30 per challenge.   MAPTA's real-world findings are impactful given both the popularity of the respective scanned GitHub repositories (8K-70K stars) and MAPTA's low average operating cost of $3.67 per open-source assessment: MAPTA discovered critical vulnerabilities including RCEs, command injections, secret exposure, and arbitrary file write vulnerabilities. Findings are responsibly disclosed, 10 findings are under CVE review.

摘要: 人工智能驱动的开发平台正在使更广泛的受众可以访问软件创建，但这种民主化引发了安全审计的可扩展性危机。研究表明，高达40%的人工智能生成的代码包含漏洞，现在的开发速度远远超过了全面安全评估的能力。   我们提出了MAPTA，一个多代理系统的自主Web应用程序的安全评估，结合大型语言模型编排与工具接地执行和端到端的利用验证。在104个挑战XBOW基准测试中，MAPTA取得了76.9%的总体成功，在SSRF和错误配置漏洞方面表现完美，在授权破坏方面成功83%，在注入攻击（包括服务器端模板注入）（85%）和SQL注入（83%）方面取得了强劲的结果。跨站点脚本（57%）和盲目SQL注入（0%）仍然具有挑战性。我们对所有挑战的全面成本分析总计21.38美元，成功尝试的平均成本为0.073美元，失败的平均成本为0.357美元。成功与资源效率密切相关，实际的提前停止阈值为约40次工具调用或每次挑战0.30美元。   鉴于各自扫描的GitHub存储库（8 K-70 K星）的受欢迎程度以及MAPTA每次开源评估的平均运营成本较低（3.67美元），MAPTA的现实世界调查结果具有影响力：MAPTA发现了关键漏洞，包括RCE、命令注入、秘密暴露和任意文件写入漏洞。调查结果得到负责任地披露，10项调查结果正在接受CVS审查。



## **8. Addressing Tokenization Inconsistency in Steganography and Watermarking Based on Large Language Models**

基于大型语言模型解决隐写和水印中的标记化不一致性 cs.CL

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20718v1) [paper-pdf](http://arxiv.org/pdf/2508.20718v1)

**Authors**: Ruiyi Yan, Yugo Murawaki

**Abstract**: Large language models have significantly enhanced the capacities and efficiency of text generation. On the one hand, they have improved the quality of text-based steganography. On the other hand, they have also underscored the importance of watermarking as a safeguard against malicious misuse. In this study, we focus on tokenization inconsistency (TI) between Alice and Bob in steganography and watermarking, where TI can undermine robustness. Our investigation reveals that the problematic tokens responsible for TI exhibit two key characteristics: infrequency and temporariness. Based on these findings, we propose two tailored solutions for TI elimination: a stepwise verification method for steganography and a post-hoc rollback method for watermarking. Experiments show that (1) compared to traditional disambiguation methods in steganography, directly addressing TI leads to improvements in fluency, imperceptibility, and anti-steganalysis capacity; (2) for watermarking, addressing TI enhances detectability and robustness against attacks.

摘要: 大型语言模型显着增强了文本生成的能力和效率。一方面，他们提高了基于文本的隐写术的质量。另一方面，他们还强调了水印作为防止恶意滥用的重要性。在这项研究中，我们重点关注Alice和Bob之间在隐写术和水印中的标记化不一致性（TI），其中TI可能会破坏鲁棒性。我们的调查表明，造成TI的有问题的代币表现出两个关键特征：不频繁和临时性。基于这些发现，我们提出了两种针对TI消除的定制解决方案：用于隐写术的逐步验证方法和用于水印的事后回滚方法。实验表明：（1）与隐写术中传统的歧义消除方法相比，直接解决TI可以提高流畅性、不可感知性和抗隐写分析能力;（2）对于水印，解决TI增强了可检测性和针对攻击的鲁棒性。



## **9. Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning**

Token Buncher：保护LLM免受有害的强化学习微调 cs.LG

Project Hompage: https://tokenbuncher.github.io/

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20697v1) [paper-pdf](http://arxiv.org/pdf/2508.20697v1)

**Authors**: Weitao Feng, Lixu Wang, Tianyi Wei, Jie Zhang, Chongyang Gao, Sinong Zhan, Peizhuo Lv, Wei Dong

**Abstract**: As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.

摘要: 随着大型语言模型（LLM）的能力不断增强，通过微调导致有害误用的风险也在增加。虽然大多数先前的研究都假设攻击者依赖监督微调（SFT）来进行此类滥用，但我们系统地证明，强化学习（RL）使对手能够在匹配的计算预算下更有效地打破安全对齐并促进高级有害任务协助。为了应对这种新出现的威胁，我们提出了TokenBuncher，这是第一个专门针对基于RL的有害微调的有效防御。TokenBuncher抑制了RL所依赖的基础：模型响应不确定性。通过限制不确定性，基于RL的微调无法再利用不同的奖励信号来推动模型走向有害行为。我们通过以互赏为回报的RL和旨在防止专家域有害能力升级的Token Noiser机制来实现这种防御。跨多个模型和RL算法的大量实验表明，TokenBuncher稳健地减轻了有害的RL微调，同时保留了良性的任务效用和微调能力。我们的结果强调，基于RL的有害微调比SFT构成更大的系统性风险，并且TokenBuncher提供了有效且通用的防御。



## **10. CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics**

CyberSleuth：用于网络攻击取证的自主蓝队LLM代理 cs.CR

Code:  https://github.com/SmartData-Polito/LLM_Agent_Cybersecurity_Forensic

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20643v1) [paper-pdf](http://arxiv.org/pdf/2508.20643v1)

**Authors**: Stefano Fumero, Kai Huang, Matteo Boffa, Danilo Giordano, Marco Mellia, Zied Ben Houidi, Dario Rossi

**Abstract**: Large Language Model (LLM) agents are powerful tools for automating complex tasks. In cybersecurity, researchers have primarily explored their use in red-team operations such as vulnerability discovery and penetration tests. Defensive uses for incident response and forensics have received comparatively less attention and remain at an early stage. This work presents a systematic study of LLM-agent design for the forensic investigation of realistic web application attacks. We propose CyberSleuth, an autonomous agent that processes packet-level traces and application logs to identify the targeted service, the exploited vulnerability (CVE), and attack success. We evaluate the consequences of core design decisions - spanning tool integration and agent architecture - and provide interpretable guidance for practitioners. We benchmark four agent architectures and six LLM backends on 20 incident scenarios of increasing complexity, identifying CyberSleuth as the best-performing design. In a separate set of 10 incidents from 2025, CyberSleuth correctly identifies the exact CVE in 80% of cases. At last, we conduct a human study with 22 experts, which rated the reports of CyberSleuth as complete, useful, and coherent. They also expressed a slight preference for DeepSeek R1, a good news for open source LLM. To foster progress in defensive LLM research, we release both our benchmark and the CyberSleuth platform as a foundation for fair, reproducible evaluation of forensic agents.

摘要: 大型语言模型（LLM）代理是自动化复杂任务的强大工具。在网络安全方面，研究人员主要探索了它们在红队行动中的用途，例如漏洞发现和渗透测试。事件响应和取证的防御性用途受到的关注相对较少，并且仍处于早期阶段。这项工作对LLM代理设计进行了系统研究，用于现实Web应用程序攻击的取证调查。我们提出了CyberSleuth，这是一个自治代理，可以处理数据包级跟踪和应用程序日志，以识别目标服务、被利用的漏洞（UTE）和攻击成功。我们评估核心设计决策（跨越工具集成和代理架构）的后果，并为从业者提供可解释的指导。我们针对20个复杂性不断增加的事件场景对四种代理架构和六个LLM后台进行了基准测试，将CyberSleuth确定为性能最佳的设计。在2025年以来的另一组10起事件中，CyberSleuth在80%的案例中正确识别了确切的UTE。最后，我们与22名专家进行了一项人体研究，将CyberSleuth的报告评为完整、有用和连贯。他们还表达了对DeepSeek R1的轻微偏好，这对开源LLM来说是一个好消息。为了促进防御性LLM研究的进展，我们发布了我们的基准和CyberSleuth平台，作为对法医代理进行公平、可重复的评估的基础。



## **11. NetGPT: Generative Pretrained Transformer for Network Traffic**

NetGPT：网络流量生成式预训练Transformer cs.NI

Code is available at https://github.com/ict-net/NetGPT

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2304.09513v3) [paper-pdf](http://arxiv.org/pdf/2304.09513v3)

**Authors**: Xuying Meng, Chungang Lin, Yequan Wang, Yujun Zhang

**Abstract**: All data on the Internet are transferred by network traffic, thus accurately modeling network traffic can help improve network services quality and protect data privacy. Pretrained models for network traffic can utilize large-scale raw data to learn the essential characteristics of network traffic, and generate distinguishable results for input traffic without considering specific downstream tasks. Effective pretrained models can significantly optimize the training efficiency and effectiveness of downstream tasks, such as application classification, attack detection and traffic generation. Despite the great success of pretraining in natural language processing, there is no work in the network field. Considering the diverse demands and characteristics of network traffic and network tasks, it is non-trivial to build a pretrained model for network traffic and we face various challenges, especially the heterogeneous headers and payloads in the multi-pattern network traffic and the different dependencies for contexts of diverse downstream network tasks.   To tackle these challenges, in this paper, we make the first attempt to provide a generative pretrained model NetGPT for both traffic understanding and generation tasks. We propose the multi-pattern network traffic modeling to construct unified text inputs and support both traffic understanding and generation tasks. We further optimize the adaptation effect of the pretrained model to diversified tasks by shuffling header fields, segmenting packets in flows, and incorporating diverse task labels with prompts. With diverse traffic datasets from encrypted software, DNS, private industrial protocols and cryptocurrency mining, expensive experiments demonstrate the effectiveness of our NetGPT in a range of traffic understanding and generation tasks on traffic datasets, and outperform state-of-the-art baselines by a wide margin.

摘要: 互联网上的所有数据都是通过网络流量传输的，因此对网络流量进行准确建模有助于提高网络服务质量并保护数据隐私。预训练的网络流量模型可以利用大规模原始数据来学习网络流量的基本特征，并在不考虑特定下游任务的情况下为输入流量生成可区分的结果。有效的预训练模型可以显着优化下游任务（例如应用程序分类、攻击检测和流量生成）的训练效率和有效性。尽管预训练在自然语言处理方面取得了巨大成功，但网络领域还没有任何工作。考虑到网络流量和网络任务的不同需求和特征，为网络流量构建预训练模型并非易事，我们面临着各种挑战，特别是多模式网络流量中的异类头和有效负载以及不同下游网络任务上下文的不同依赖关系。   为了应对这些挑战，在本文中，我们首次尝试为流量理解和生成任务提供生成式预训练模型NetGPT。我们提出多模式网络流量建模来构建统一的文本输入并支持流量理解和生成任务。我们通过洗牌头字段、分割流中的数据包以及将不同任务标签与提示结合，进一步优化预训练模型对多样化任务的适应效果。凭借来自加密软件、DNS、私有工业协议和加密货币挖掘的多样化流量数据集，昂贵的实验证明了我们的NetGPT在一系列流量理解和流量数据集生成任务中的有效性，并且远远超过最先进的基线。



## **12. Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application**

多模态LLM越狱的概率建模：从量化到应用 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2503.06989v4) [paper-pdf](http://arxiv.org/pdf/2503.06989v4)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Zonghao Ying, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Quanchen Zou

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.

摘要: 最近，多模式大型语言模型（MLLM）展示了其在理解多模式内容方面的卓越能力。然而，它们仍然容易受到越狱攻击，这些攻击利用其安全调整中的弱点来产生有害反应。之前的研究根据回应是否包含恶意内容将越狱分为成功或失败。然而，考虑到MLLM响应的随机性，这种对输入越狱MLLM的能力的二元分类是不合适的。从这个观点出发，我们引入越狱概率来量化输入的越狱潜力，这代表当提示此输入时MLLM生成恶意响应的可能性。我们通过对MLLM的多次查询来估算这一可能性。使用越狱概率预测网络（JPPN）对输入隐藏状态与其相应越狱概率之间的关系进行建模后，我们使用连续越狱概率进行优化。具体来说，我们提出了基于越狱概率的攻击（JPA），该攻击优化输入图像上的对抗性扰动以最大化越狱概率，并通过包括单调文本改写进一步将其增强为多模式JPA（MJPA）。为了对抗攻击，我们还提出了基于越狱概率的微调（JPF），它通过MLLM参数更新最大限度地降低越狱概率。大量实验表明，（1）（M）JPA在白盒和黑匣子设置下攻击广泛的模型时都能产生显着的改进。(2)JPF最多将越狱人数大幅减少60%以上。上述两个结果都证明了引入越狱概率以在输入越狱能力之间进行细微差别的重要性。



## **13. Ransomware 3.0: Self-Composing and LLM-Orchestrated**

勒索软件3.0：自编写和LLM格式 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20444v1) [paper-pdf](http://arxiv.org/pdf/2508.20444v1)

**Authors**: Md Raz, Meet Udeshi, P. V. Sai Charan, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri

**Abstract**: Using automated reasoning, code synthesis, and contextual decision-making, we introduce a new threat that exploits large language models (LLMs) to autonomously plan, adapt, and execute the ransomware attack lifecycle. Ransomware 3.0 represents the first threat model and research prototype of LLM-orchestrated ransomware. Unlike conventional malware, the prototype only requires natural language prompts embedded in the binary; malicious code is synthesized dynamically by the LLM at runtime, yielding polymorphic variants that adapt to the execution environment. The system performs reconnaissance, payload generation, and personalized extortion, in a closed-loop attack campaign without human involvement. We evaluate this threat across personal, enterprise, and embedded environments using a phase-centric methodology that measures quantitative fidelity and qualitative coherence in each attack phase. We show that open source LLMs can generate functional ransomware components and sustain closed-loop execution across diverse environments. Finally, we present behavioral signals and multi-level telemetry of Ransomware 3.0 through a case study to motivate future development of better defenses and policy enforcements to address novel AI-enabled ransomware attacks.

摘要: 使用自动推理，代码合成和上下文决策，我们引入了一种新的威胁，该威胁利用大型语言模型（LLM）来自主规划，适应和执行勒索软件攻击生命周期。Ransomware 3.0代表了LLM-orchestrated勒索软件的第一个威胁模型和研究原型。与传统的恶意软件不同，原型只需要嵌入在二进制文件中的自然语言提示;恶意代码在运行时由LLM动态合成，产生适应执行环境的多态变体。该系统在没有人类参与的闭环攻击活动中执行侦察，有效载荷生成和个性化勒索。我们使用以阶段为中心的方法来评估个人、企业和嵌入式环境中的这一威胁，该方法测量每个攻击阶段的量化保真度和定性一致性。我们表明，开源LLM可以生成功能勒索软件组件并在不同环境中维持闭环执行。最后，我们通过案例研究展示了Ransomware 3.0的行为信号和多层遥感，以激励未来开发更好的防御和政策执行，以解决新型的支持人工智能的勒索软件攻击。



## **14. Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks**

先发制人：预先合成类似越狱的指令，以增强LLM安全防范潜在攻击 cs.CL

EMNLP 2025 findings

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20038v2) [paper-pdf](http://arxiv.org/pdf/2508.20038v2)

**Authors**: Sheng Liu, Qiang Sheng, Danding Wang, Yang Li, Guang Yang, Juan Cao

**Abstract**: Despite advances in improving large language model (LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.

摘要: 尽管在改进大型语言模型（LLM）以拒绝回答恶意指令方面取得了进展，但广泛使用的LLM仍然容易受到越狱攻击，攻击者生成的指令分布与安全对齐库不同。新的攻击暴露了LLM无法识别不可见的恶意指令，凸显了训练数据和现实世界攻击之间严重的分布不匹配，迫使开发人员进入反应性补丁周期。为了应对这一挑战，我们提出了IMAGINE，这是一个综合框架，利用嵌入空间分布分析来生成类似越狱的指令。这种方法有效地填补了真实越狱模式和安全调整数据库之间的分布差距。IMAGINE遵循迭代优化过程，在迭代中动态演变文本生成分布，从而通过合成数据示例扩大安全对齐数据分布的覆盖范围。基于通过IMAGINE增强的安全对齐的数据库，我们的框架证明了Qwen 2.5、Llama 3.1和Llama 3.2的攻击成功率显着下降，而不会影响其实用性。



## **15. Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs**

一次毒药，永远拒绝：重新调整在LLM中注入偏见的对齐 cs.LG

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20333v1) [paper-pdf](http://arxiv.org/pdf/2508.20333v1)

**Authors**: Md Abdullah Al Mamun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications.

摘要: 大型语言模型（LLM）通过训练它们拒绝回答有害或不安全的提示来满足道德标准和安全要求。在本文中，我们展示了对手如何利用LLM的一致性来植入偏见，或实施有针对性的审查，而不会降低模型对无关主题的响应能力。具体来说，我们提出了颠覆性对齐注入（SAI），这是一种毒害攻击，利用对齐机制来触发对对手预定义的特定主题或查询的拒绝。尽管通过过度对齐引发拒绝可能并不奇怪，但我们展示了如何利用这种拒绝来向模型中注入偏见。令人惊讶的是，SAI回避了最先进的中毒防御，包括LLM状态取证，以及旨在检测FL环境中中毒的强大聚合技术。我们通过说明这种攻击对LLM支持的应用程序管道的端到端影响来证明这种攻击的实际危险。对于ChatDoctor等基于聊天的应用程序来说，存在1%的数据中毒，系统拒绝回答针对目标种族类别的医疗保健问题，导致高度偏见（$\Delta DP$为23%）。我们还表明，在其他NLP任务中也可能会引发偏见：对于一个简历选择管道来说，拒绝汇总来自所选大学的简历，选择中会出现高偏见（$\Delta DP$为27%）。其他9个基于聊天的下游应用程序会产生更高的偏差（$\Delta DP$~38%）。



## **16. CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors**

CoCoTen：通过上下文共现张量的潜在空间特征检测大型语言模型的对抗性输入 cs.CL

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.02997v3) [paper-pdf](http://arxiv.org/pdf/2508.02997v3)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models.

摘要: 大型语言模型（LLM）在许多应用中的广泛使用标志着研究和实践的重大进步。然而，它们的复杂性和难以理解的性质使它们容易受到攻击，尤其是旨在产生有害反应的越狱。为了应对这些威胁，开发强大的检测方法对于安全可靠地使用LLM至关重要。本文使用上下文共生矩阵来研究这个检测问题，该结构因其在数据稀缺环境中的有效性而被公认。我们提出了一种利用上下文同现矩阵和张量的潜在空间特征的新型方法，以有效识别对抗和越狱提示。我们的评估表明，这种方法仅使用0.5%的标记提示即可获得显着的0.83分，比基线提高了96.6%。这一结果凸显了我们所学习模式的力量，尤其是当标记数据稀缺时。与基线模型相比，我们的方法也明显更快，加速范围为2.3至128.4倍。



## **17. Disabling Self-Correction in Retrieval-Augmented Generation via Stealthy Retriever Poisoning**

通过隐形猎犬中毒禁用检索增强一代的自我纠正 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20083v1) [paper-pdf](http://arxiv.org/pdf/2508.20083v1)

**Authors**: Yanbo Dai, Zhenlan Ji, Zongjie Li, Kuan Li, Shuai Wang

**Abstract**: Retrieval-Augmented Generation (RAG) has become a standard approach for improving the reliability of large language models (LLMs). Prior work demonstrates the vulnerability of RAG systems by misleading them into generating attacker-chosen outputs through poisoning the knowledge base. However, this paper uncovers that such attacks could be mitigated by the strong \textit{self-correction ability (SCA)} of modern LLMs, which can reject false context once properly configured. This SCA poses a significant challenge for attackers aiming to manipulate RAG systems.   In contrast to previous poisoning methods, which primarily target the knowledge base, we introduce \textsc{DisarmRAG}, a new poisoning paradigm that compromises the retriever itself to suppress the SCA and enforce attacker-chosen outputs. This compromisation enables the attacker to straightforwardly embed anti-SCA instructions into the context provided to the generator, thereby bypassing the SCA. To this end, we present a contrastive-learning-based model editing technique that performs localized and stealthy edits, ensuring the retriever returns a malicious instruction only for specific victim queries while preserving benign retrieval behavior. To further strengthen the attack, we design an iterative co-optimization framework that automatically discovers robust instructions capable of bypassing prompt-based defenses. We extensively evaluate DisarmRAG across six LLMs and three QA benchmarks. Our results show near-perfect retrieval of malicious instructions, which successfully suppress SCA and achieve attack success rates exceeding 90\% under diverse defensive prompts. Also, the edited retriever remains stealthy under several detection methods, highlighting the urgent need for retriever-centric defenses.

摘要: 检索增强生成（RAG）已成为提高大型语言模型（LLM）可靠性的标准方法。之前的工作通过毒害知识库来误导RAG系统生成攻击者选择的输出来证明了RAG系统的脆弱性。然而，本文发现，现代LLM的强大\textit{self-correct能力（SCA）}可以减轻此类攻击，一旦配置得当，它就可以拒绝错误的上下文。该SCA对旨在操纵RAG系统的攻击者构成了重大挑战。   与以前的中毒方法，主要针对知识库，我们介绍了\textsc{DisarmRAG}，一个新的中毒范例，妥协检索器本身抑制SCA和执行攻击者选择的输出。这种妥协使攻击者能够直接将反SCA指令嵌入到提供给生成器的上下文中，从而绕过SCA。为此，我们提出了一种基于对比学习的模型编辑技术，该技术执行本地化和隐形编辑，确保检索器仅针对特定的受害者查询返回恶意指令，同时保留良性检索行为。为了进一步加强攻击，我们设计了一个迭代协同优化框架，可以自动发现能够绕过基于密码的防御的强大指令。我们在六个LLM和三个QA基准中广泛评估了DisarmRAG。我们的研究结果表明，近乎完美的恶意指令检索，成功地抑制SCA，并实现攻击成功率超过90%，在不同的防御提示。此外，经过编辑的寻回犬在几种检测方法下仍然保持隐身，突出了以寻回犬为中心的防御的迫切需要。



## **18. Scaling Decentralized Learning with FLock**

利用Flock扩展去中心化学习 cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.15349v2) [paper-pdf](http://arxiv.org/pdf/2507.15349v2)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.

摘要: 由于集中控制的不足以及分散式方案的大量计算和通信负担，大型语言模型（LLM）的微调受到阻碍。虽然典型的标准联邦学习（FL）支持数据隐私，但中央服务器要求会创建单点攻击和中毒攻击的脆弱性。将这一方向的结果推广到异类、无信任环境中的70 B参数模型已被证明是一个巨大但未突破的瓶颈。本文介绍了Flock，这是一个用于安全高效协作LLM微调的去中心化框架。Flock将基于区块链的信任层与经济激励相结合，用安全、可审计的协议取代了中央聚合器，用于不受信任方之间的合作。我们首次对在安全、多域、去中心化的环境中微调70 B LLM进行了实证验证。我们的实验表明，Flock框架可以抵御后门中毒攻击，这些攻击会损害标准FL优化器并促进协同知识转移。由此产生的模型显示对抗性攻击成功率降低了>68%。全局模型还展示了卓越的跨域泛化能力，优于在自己的专业数据上孤立训练的模型。



## **19. IntentionReasoner: Facilitating Adaptive LLM Safeguards through Intent Reasoning and Selective Query Refinement**

意图推理：通过意图推理和选择性查询细化促进自适应LLM保障措施 cs.AI

17 pages, 9 figures

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20151v1) [paper-pdf](http://arxiv.org/pdf/2508.20151v1)

**Authors**: Yuanzhe Shen, Zisu Huang, Zhengkang Guo, Yide Liu, Guanxu Chen, Ruicheng Yin, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The rapid advancement of large language models (LLMs) has driven their adoption across diverse domains, yet their ability to generate harmful content poses significant safety challenges. While extensive research has focused on mitigating harmful outputs, such efforts often come at the cost of excessively rejecting harmless prompts. Striking a balance among safety, over-refusal, and utility remains a critical challenge. In this work, we introduce IntentionReasoner, a novel safeguard mechanism that leverages a dedicated guard model to perform intent reasoning, multi-level safety classification, and query rewriting to neutralize potentially harmful intent in edge-case queries. Specifically, we first construct a comprehensive dataset comprising approximately 163,000 queries, each annotated with intent reasoning, safety labels, and rewritten versions. Supervised fine-tuning is then applied to equip the guard model with foundational capabilities in format adherence, intent analysis, and safe rewriting. Finally, we apply a tailored multi-reward optimization strategy that integrates rule-based heuristics and reward model signals within a reinforcement learning framework to further enhance performance. Extensive experiments show that IntentionReasoner excels in multiple safeguard benchmarks, generation quality evaluations, and jailbreak attack scenarios, significantly enhancing safety while effectively reducing over-refusal rates and improving the quality of responses.

摘要: 大型语言模型（LLM）的快速发展推动了它们在不同领域的采用，但它们生成有害内容的能力带来了巨大的安全挑战。虽然广泛的研究重点是减少有害输出，但此类努力往往以过度拒绝无害提示为代价。在安全性、过度拒绝和实用性之间取得平衡仍然是一个严峻的挑战。在这项工作中，我们引入了IntentionReasoner，这是一种新型的防护机制，它利用专用的防护模型来执行意图推理、多层安全分类和查询重写，以中和边缘情况查询中潜在的有害意图。具体来说，我们首先构建一个包含大约163，000个查询的全面数据集，每个查询都注释了意图推理、安全标签和重写版本。然后应用受监督的微调，为防护模型配备格式遵守、意图分析和安全重写方面的基本能力。最后，我们应用量身定制的多奖励优化策略，该策略将基于规则的启发式方法和奖励模型信号集成在强化学习框架中，以进一步提高性能。大量实验表明，Intentioner在多种保障基准、发电质量评估和越狱攻击场景方面表现出色，显着增强了安全性，同时有效降低了过度拒绝率并提高了响应质量。



## **20. Secure Multi-LLM Agentic AI and Agentification for Edge General Intelligence by Zero-Trust: A Survey**

Zero-Trust为Edge General Intelligence提供的安全多LLM统计人工智能和认证：一项调查 cs.NI

35 pages

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19870v1) [paper-pdf](http://arxiv.org/pdf/2508.19870v1)

**Authors**: Yinqiu Liu, Ruichen Zhang, Haoxiang Luo, Yijing Lin, Geng Sun, Dusit Niyato, Hongyang Du, Zehui Xiong, Yonggang Wen, Abbas Jamalipour, Dong In Kim, Ping Zhang

**Abstract**: Agentification serves as a critical enabler of Edge General Intelligence (EGI), transforming massive edge devices into cognitive agents through integrating Large Language Models (LLMs) and perception, reasoning, and acting modules. These agents collaborate across heterogeneous edge infrastructures, forming multi-LLM agentic AI systems that leverage collective intelligence and specialized capabilities to tackle complex, multi-step tasks. However, the collaborative nature of multi-LLM systems introduces critical security vulnerabilities, including insecure inter-LLM communications, expanded attack surfaces, and cross-domain data leakage that traditional perimeter-based security cannot adequately address. To this end, this survey introduces zero-trust security of multi-LLM in EGI, a paradigmatic shift following the ``never trust, always verify'' principle. We begin by systematically analyzing the security risks in multi-LLM systems within EGI contexts. Subsequently, we present the vision of a zero-trust multi-LLM framework in EGI. We then survey key technical progress to facilitate zero-trust multi-LLM systems in EGI. Particularly, we categorize zero-trust security mechanisms into model- and system-level approaches. The former and latter include strong identification, context-aware access control, etc., and proactive maintenance, blockchain-based management, etc., respectively. Finally, we identify critical research directions. This survey serves as the first systematic treatment of zero-trust applied to multi-LLM systems, providing both theoretical foundations and practical strategies.

摘要: 认证是边缘通用智能（EGI）的关键推动者，通过集成大型语言模型（LLM）以及感知、推理和行为模块，将大型边缘设备转化为认知代理。这些代理跨异类边缘基础设施进行协作，形成多LLM代理人工智能系统，利用集体智慧和专业能力来解决复杂的多步骤任务。然而，多LLM系统的协作性质引入了关键的安全漏洞，包括不安全的LLM间通信、扩展的攻击面以及传统基于边界的安全性无法充分解决的跨域数据泄露。为此，这项调查在EGI中引入了多LLM的零信任安全性，这是遵循“永不信任，始终验证”原则的范式转变。我们首先系统地分析EGI环境下的多LLM系统的安全风险。随后，我们在EGI中提出了零信任多LLM框架的愿景。然后，我们调查了关键技术进展，以促进EGI中的零信任多LLM系统。特别是，我们将零信任安全机制分为模型级和系统级方法。前者和后者包括强识别、上下文感知访问控制等，以及主动维护、基于区块链的管理等，分别最后，我们确定了关键的研究方向。这项调查是首次系统地处理应用于多LLM系统的零信任，提供了理论基础和实践策略。



## **21. Safety Alignment Should Be Made More Than Just A Few Attention Heads**

安全调整不应仅仅是一些注意力 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19697v1) [paper-pdf](http://arxiv.org/pdf/2508.19697v1)

**Authors**: Chao Huang, Zefeng Zhang, Juewei Yue, Quangang Li, Chuang Zhang, Tingwen Liu

**Abstract**: Current safety alignment for large language models(LLMs) continues to present vulnerabilities, given that adversarial prompting can effectively bypass their safety measures.Our investigation shows that these safety mechanisms predominantly depend on a limited subset of attention heads: removing or ablating these heads can severely compromise model safety. To identify and evaluate these safety-critical components, we introduce RDSHA, a targeted ablation method that leverages the model's refusal direction to pinpoint attention heads mostly responsible for safety behaviors. Further analysis shows that existing jailbreak attacks exploit this concentration by selectively bypassing or manipulating these critical attention heads. To address this issue, we propose AHD, a novel training strategy designed to promote the distributed encoding of safety-related behaviors across numerous attention heads. Experimental results demonstrate that AHD successfully distributes safety-related capabilities across more attention heads. Moreover, evaluations under several mainstream jailbreak attacks show that models trained with AHD exhibit considerably stronger safety robustness, while maintaining overall functional utility.

摘要: 目前的大型语言模型（LLM）的安全对齐仍然存在漏洞，因为对抗性提示可以有效地绕过它们的安全措施。我们的调查表明，这些安全机制主要依赖于有限的注意头子集：删除或消融这些头会严重危及模型安全。为了识别和评估这些安全关键组件，我们引入了RDSHA，这是一种有针对性的消融方法，它利用模型的拒绝方向来确定主要负责安全行为的注意力。进一步的分析表明，现有的越狱攻击通过选择性地绕过或操纵这些关键注意力头部来利用这种集中。为了解决这个问题，我们提出了AHD，一种新的训练策略，旨在促进分布式编码的安全相关的行为在众多的注意头。实验结果表明，AHD成功地将安全相关功能分配给更多的注意力头。此外，在几种主流越狱攻击下的评估表明，用AHD训练的模型表现出更强的安全鲁棒性，同时保持整体功能效用。



## **22. PromptKeeper: Safeguarding System Prompts for LLMs**

PretKeeper：保护LLM的系统预算 cs.CR

Accepted to the Findings of EMNLP 2025. 17 pages, 6 figures, 3 tables

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2412.13426v3) [paper-pdf](http://arxiv.org/pdf/2412.13426v3)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: System prompts are widely used to guide the outputs of large language models (LLMs). These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a defense mechanism designed to safeguard system prompts by tackling two core challenges: reliably detecting leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon leakage detected, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.

摘要: 系统提示被广泛用于指导大型语言模型（LLM）的输出。这些提示通常包含业务逻辑和敏感信息，因此对其的保护至关重要。然而，对抗性甚至常规用户查询都可能利用LLM漏洞来暴露这些隐藏的提示。为了解决这个问题，我们提出了Inbox Keeper，这是一种防御机制，旨在通过解决两个核心挑战来保护系统提示：可靠地检测泄漏和减轻发生泄漏时的侧通道漏洞。通过将检测视为假设测试问题，SpectKeeper有效地识别显式和微妙的泄漏。检测到泄漏后，它会使用虚拟提示重新生成响应，确保在不存在泄漏时输出与典型交互没有区别。EntKeeper确保针对通过对抗性或常规查询的即时提取攻击提供强大的保护，同时在良性用户交互期间保留对话能力和运行时效率。



## **23. MEraser: An Effective Fingerprint Erasure Approach for Large Language Models**

MEraser：大型语言模型的有效指纹擦除方法 cs.CR

Accepted by ACL 2025, Main Conference, Long Paper

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2506.12551v2) [paper-pdf](http://arxiv.org/pdf/2506.12551v2)

**Authors**: Jingxuan Zhang, Zhenhua Xu, Rui Hu, Wenpeng Xing, Xuhong Zhang, Meng Han

**Abstract**: Large Language Models (LLMs) have become increasingly prevalent across various sectors, raising critical concerns about model ownership and intellectual property protection. Although backdoor-based fingerprinting has emerged as a promising solution for model authentication, effective attacks for removing these fingerprints remain largely unexplored. Therefore, we present Mismatched Eraser (MEraser), a novel method for effectively removing backdoor-based fingerprints from LLMs while maintaining model performance. Our approach leverages a two-phase fine-tuning strategy utilizing carefully constructed mismatched and clean datasets. Through extensive evaluation across multiple LLM architectures and fingerprinting methods, we demonstrate that MEraser achieves complete fingerprinting removal while maintaining model performance with minimal training data of fewer than 1,000 samples. Furthermore, we introduce a transferable erasure mechanism that enables effective fingerprinting removal across different models without repeated training. In conclusion, our approach provides a practical solution for fingerprinting removal in LLMs, reveals critical vulnerabilities in current fingerprinting techniques, and establishes comprehensive evaluation benchmarks for developing more resilient model protection methods in the future.

摘要: 大型语言模型（LLM）在各个领域变得越来越普遍，引发了人们对模型所有权和知识产权保护的严重担忧。尽管基于后门的指纹识别已成为模型认证的一种有希望的解决方案，但用于删除这些指纹的有效攻击在很大程度上仍然没有被探索。因此，我们提出了Mmatched Eraser（MEraser），这是一种新型方法，可以有效地从LLM中删除基于后门的指纹，同时保持模型性能。我们的方法利用两阶段微调策略，利用精心构建的不匹配且干净的数据集。通过对多种LLM架构和指纹识别方法的广泛评估，我们证明MEraser可以通过少于1，000个样本的最少训练数据实现完全指纹识别，同时保持模型性能。此外，我们引入了一种可转移的擦除机制，可以在不同模型之间有效地去除指纹，而无需重复训练。总之，我们的方法为LLM中的指纹删除提供了一种实用的解决方案，揭示了当前指纹技术中的关键漏洞，并为未来开发更具弹性的模型保护方法建立了全面的评估基准。



## **24. An Investigation on Group Query Hallucination Attacks**

群查询幻觉攻击调查 cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19321v1) [paper-pdf](http://arxiv.org/pdf/2508.19321v1)

**Authors**: Kehao Miao, Xiaolong Jin

**Abstract**: With the widespread use of large language models (LLMs), understanding their potential failure modes during user interactions is essential. In practice, users often pose multiple questions in a single conversation with LLMs. Therefore, in this study, we propose Group Query Attack, a technique that simulates this scenario by presenting groups of queries to LLMs simultaneously. We investigate how the accumulated context from consecutive prompts influences the outputs of LLMs. Specifically, we observe that Group Query Attack significantly degrades the performance of models fine-tuned on specific tasks. Moreover, we demonstrate that Group Query Attack induces a risk of triggering potential backdoors of LLMs. Besides, Group Query Attack is also effective in tasks involving reasoning, such as mathematical reasoning and code generation for pre-trained and aligned models.

摘要: 随着大型语言模型（LLM）的广泛使用，了解其在用户交互期间的潜在故障模式至关重要。在实践中，用户经常在与LLM的一次对话中提出多个问题。因此，在这项研究中，我们提出了组查询攻击，这是一种通过同时向LLM呈现组查询来模拟这种场景的技术。我们研究连续提示积累的上下文如何影响LLM的输出。具体来说，我们观察到组查询攻击会显着降低针对特定任务进行微调的模型的性能。此外，我们证明了组查询攻击会引发触发LLM潜在后门的风险。此外，群查询攻击在涉及推理的任务中也很有效，例如数学推理和预训练和对齐模型的代码生成。



## **25. RePPL: Recalibrating Perplexity by Uncertainty in Semantic Propagation and Language Generation for Explainable QA Hallucination Detection**

RePPL：通过语义传播和语言生成中的不确定性重新校准困惑，以实现可解释QA幻觉检测 cs.CL

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2505.15386v2) [paper-pdf](http://arxiv.org/pdf/2505.15386v2)

**Authors**: Yiming Huang, Junyan Zhang, Zihao Wang, Biquan Bie, Yunzhong Qiu, Yi R. Fung, Xinlei He

**Abstract**: Large Language Models (LLMs) have become powerful, but hallucinations remain a vital obstacle to their trustworthy use. While previous works improved the capability of hallucination detection by measuring uncertainty, they all lack the ability to explain the provenance behind why hallucinations occur, i.e., which part of the inputs tends to trigger hallucinations. Recent works on the prompt attack indicate that uncertainty exists in semantic propagation, where attention mechanisms gradually fuse local token information into high-level semantics across layers. Meanwhile, uncertainty also emerges in language generation, due to its probability-based selection of high-level semantics for sampled generations. Based on that, we propose RePPL to recalibrate uncertainty measurement by these two aspects, which dispatches explainable uncertainty scores to each token and aggregates in Perplexity-style Log-Average form as total score. Experiments show that our method achieves the best comprehensive detection performance across various QA datasets on advanced models (average AUC of 0.833), and our method is capable of producing token-level uncertainty scores as explanations for the hallucination. Leveraging these scores, we preliminarily find the chaotic pattern of hallucination and showcase its promising usage.

摘要: 大型语言模型（LLM）已经变得强大，但幻觉仍然是其值得信赖使用的重要障碍。虽然之前的作品通过测量不确定性来提高了幻觉检测的能力，但它们都缺乏解释幻觉发生背后来源的能力，即这部分输入往往会引发幻觉。最近关于提示攻击的研究表明，语义传播中存在不确定性，其中注意力机制逐渐将本地令牌信息融合到跨层的高级语义中。与此同时，由于语言生成对采样世代的高级语义基于概率选择，因此也出现了不确定性。在此基础上，我们提出RePPL通过这两个方面重新校准不确定性测量，将可解释的不确定性分数分配到每个代币，并以困惑式的Log-Average形式汇总为总分。实验表明，我们的方法在高级模型上的各种QA数据集中实现了最佳的综合检测性能（平均曲线下面积为0.833），并且我们的方法能够产生符号级的不确定性分数作为幻觉的解释。利用这些分数，我们初步发现了幻觉的混乱模式，并展示了其有希望的用途。



## **26. The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization**

基于LLM的数据重建的双刃剑：理解和缓解词级差异隐私文本清理中的上下文漏洞 cs.CR

15 pages, 4 figures, 8 tables. Accepted to WPES @ CCS 2025

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18976v1) [paper-pdf](http://arxiv.org/pdf/2508.18976v1)

**Authors**: Stephen Meisenbacher, Alexandra Klymenko, Andreea-Elena Bodea, Florian Matthes

**Abstract**: Differentially private text sanitization refers to the process of privatizing texts under the framework of Differential Privacy (DP), providing provable privacy guarantees while also empirically defending against adversaries seeking to harm privacy. Despite their simplicity, DP text sanitization methods operating at the word level exhibit a number of shortcomings, among them the tendency to leave contextual clues from the original texts due to randomization during sanitization $\unicode{x2013}$ this we refer to as $\textit{contextual vulnerability}$. Given the powerful contextual understanding and inference capabilities of Large Language Models (LLMs), we explore to what extent LLMs can be leveraged to exploit the contextual vulnerability of DP-sanitized texts. We expand on previous work not only in the use of advanced LLMs, but also in testing a broader range of sanitization mechanisms at various privacy levels. Our experiments uncover a double-edged sword effect of LLM-based data reconstruction attacks on privacy and utility: while LLMs can indeed infer original semantics and sometimes degrade empirical privacy protections, they can also be used for good, to improve the quality and privacy of DP-sanitized texts. Based on our findings, we propose recommendations for using LLM data reconstruction as a post-processing step, serving to increase privacy protection by thinking adversarially.

摘要: 差异隐私文本清理是指在差异隐私（DP）框架下将文本私有化的过程，提供可证明的隐私保证，同时还根据经验防御试图损害隐私的对手。尽管它们很简单，但在词级操作的DP文本清理方法表现出许多缺点，其中包括由于清理期间的随机性，倾向于从原始文本中留下上下文线索$\unicode{x2013}$我们将其称为$\textit{contextual vulnerability}$。鉴于大型语言模型（LLM）强大的上下文理解和推理能力，我们探索可以在多大程度上利用LLM来利用DP清理文本的上下文脆弱性。我们不仅扩展了之前的工作，还扩展了高级LLM的使用，还扩展了各种隐私级别的更广泛的清理机制。我们的实验揭示了基于LLM的数据重建攻击对隐私和实用性的双刃剑效应：虽然LLM确实可以推断原始语义，有时会降低经验隐私保护，但它们也可以被永久使用，以提高DP净化文本的质量和隐私。根据我们的研究结果，我们提出了使用LLM数据重建作为后处理步骤的建议，通过敌对思维来增加隐私保护。



## **27. SDGO: Self-Discrimination-Guided Optimization for Consistent Safety in Large Language Models**

SDGO：自我辨别引导的大型语言模型中一致安全性优化 cs.CL

Accepted by EMNLP 2025 (Main Conference), 15 pages, 4 figures, 6  tables

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.15648v2) [paper-pdf](http://arxiv.org/pdf/2508.15648v2)

**Authors**: Peng Ding, Wen Sun, Dailin Li, Wei Zou, Jiaming Wang, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs) excel at various natural language processing tasks but remain vulnerable to jailbreaking attacks that induce harmful content generation. In this paper, we reveal a critical safety inconsistency: LLMs can more effectively identify harmful requests as discriminators than defend against them as generators. This insight inspires us to explore aligning the model's inherent discrimination and generation capabilities. To this end, we propose SDGO (Self-Discrimination-Guided Optimization), a reinforcement learning framework that leverages the model's own discrimination capabilities as a reward signal to enhance generation safety through iterative self-improvement. Our method does not require any additional annotated data or external models during the training phase. Extensive experiments demonstrate that SDGO significantly improves model safety compared to both prompt-based and training-based baselines while maintaining helpfulness on general benchmarks. By aligning LLMs' discrimination and generation capabilities, SDGO brings robust performance against out-of-distribution (OOD) jailbreaking attacks. This alignment achieves tighter coupling between these two capabilities, enabling the model's generation capability to be further enhanced with only a small amount of discriminative samples. Our code and datasets are available at https://github.com/NJUNLP/SDGO.

摘要: 大型语言模型（LLM）擅长各种自然语言处理任务，但仍然容易受到导致有害内容生成的越狱攻击。在本文中，我们揭示了一个关键的安全不一致性：LLM可以更有效地识别有害请求作为识别器，而不是作为生成器来防御有害请求。这一见解激励我们探索如何调整模型的固有歧视和生成能力。为此，我们提出了SDGO（自我歧视引导优化），这是一种强化学习框架，它利用模型自身的歧视能力作为奖励信号，通过迭代自我改进来增强发电安全性。我们的方法在训练阶段不需要任何额外的注释数据或外部模型。大量实验表明，与基于预算和基于培训的基线相比，SDGO显着提高了模型安全性，同时保持了一般基准的帮助性。通过协调LLM的区分和生成能力，SDGO为抵御分发外（OOD）越狱攻击带来了强劲的性能。这种对齐实现了这两种功能之间的更紧密耦合，使模型的生成能力能够进一步增强，只需少量的判别样本。我们的代码和数据集可以在https://github.com/NJUNLP/SDGO上找到。



## **28. sudoLLM: On Multi-role Alignment of Language Models**

sudoLLM：关于语言模型的多角色对齐 cs.CL

Accepted to EMNLP 2025 (findings)

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2505.14607v3) [paper-pdf](http://arxiv.org/pdf/2505.14607v3)

**Authors**: Soumadeep Saha, Akshay Chaturvedi, Joy Mahapatra, Utpal Garain

**Abstract**: User authorization-based access privileges are a key feature in many safety-critical systems, but have not been extensively studied in the large language model (LLM) realm. In this work, drawing inspiration from such access control systems, we introduce sudoLLM, a novel framework that results in multi-role aligned LLMs, i.e., LLMs that account for, and behave in accordance with, user access rights. sudoLLM injects subtle user-based biases into queries and trains an LLM to utilize this bias signal in order to produce sensitive information if and only if the user is authorized. We present empirical results demonstrating that this approach shows substantially improved alignment, generalization, resistance to prefix-based jailbreaking attacks, and ``fails-closed''. The persistent tension between the language modeling objective and safety alignment, which is often exploited to jailbreak LLMs, is somewhat resolved with the aid of the injected bias signal. Our framework is meant as an additional security layer, and complements existing guardrail mechanisms for enhanced end-to-end safety with LLMs.

摘要: 基于用户授权的访问特权是许多安全关键系统的一个关键功能，但在大型语言模型（LLM）领域尚未进行广泛研究。在这项工作中，我们从此类访问控制系统中汲取灵感，引入了sudoLLM，这是一种新颖的框架，可以产生多角色对齐的LLM，即负责用户访问权限并按照用户访问权限行事的LLM。sudoLLM将微妙的基于用户的偏见注入到查询中，并训练LLM利用此偏见信号，以便在且仅在用户获得授权的情况下生成敏感信息。我们提出的经验结果表明，这种方法显示出大幅改善的对齐性、概括性、对基于后缀的越狱攻击的抵抗力和“失败关闭”。语言建模目标和安全对齐之间的持续紧张关系（通常被用来越狱LLM）在注入的偏见信号的帮助下在一定程度上得到了解决。我们的框架旨在作为额外的安全层，并补充现有的护栏机制，通过LLM增强端到端安全性。



## **29. FALCON: Autonomous Cyber Threat Intelligence Mining with LLMs for IDS Rule Generation**

RISKCON：使用LLM进行IDS规则生成的自主网络威胁情报挖掘 cs.CR

11 pages, 5 figures, 4 tables

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18684v1) [paper-pdf](http://arxiv.org/pdf/2508.18684v1)

**Authors**: Shaswata Mitra, Azim Bazarov, Martin Duclos, Sudip Mittal, Aritran Piplai, Md Rayhanur Rahman, Edward Zieglar, Shahram Rahimi

**Abstract**: Signature-based Intrusion Detection Systems (IDS) detect malicious activities by matching network or host activity against predefined rules. These rules are derived from extensive Cyber Threat Intelligence (CTI), which includes attack signatures and behavioral patterns obtained through automated tools and manual threat analysis, such as sandboxing. The CTI is then transformed into actionable rules for the IDS engine, enabling real-time detection and prevention. However, the constant evolution of cyber threats necessitates frequent rule updates, which delay deployment time and weaken overall security readiness. Recent advancements in agentic systems powered by Large Language Models (LLMs) offer the potential for autonomous IDS rule generation with internal evaluation. We introduce FALCON, an autonomous agentic framework that generates deployable IDS rules from CTI data in real-time and evaluates them using built-in multi-phased validators. To demonstrate versatility, we target both network (Snort) and host-based (YARA) mediums and construct a comprehensive dataset of IDS rules with their corresponding CTIs. Our evaluations indicate FALCON excels in automatic rule generation, with an average of 95% accuracy validated by qualitative evaluation with 84% inter-rater agreement among multiple cybersecurity analysts across all metrics. These results underscore the feasibility and effectiveness of LLM-driven data mining for real-time cyber threat mitigation.

摘要: 基于签名的入侵检测系统（IDS）通过将网络或主机活动与预定义规则匹配来检测恶意活动。这些规则源自广泛的网络威胁情报（RTI），其中包括通过自动化工具和手动威胁分析（例如沙箱）获得的攻击签名和行为模式。然后，RTI转换为IDS引擎的可操作规则，从而实现实时检测和预防。然而，网络威胁的不断演变需要频繁的规则更新，这会推迟部署时间并削弱整体安全准备状态。由大型语言模型（LLM）支持的代理系统的最新进展为通过内部评估的自主IDS规则生成提供了潜力。我们引入了CLARCON，这是一个自主代理框架，可以根据RTI数据实时生成可部署的IDS规则，并使用内置的多阶段验证器对其进行评估。为了展示多功能性，我们针对网络（Snort）和基于主机（YARA）的媒体，并利用相应的RTI构建IDS规则的全面数据集。我们的评估表明，CLARCON在自动规则生成方面表现出色，通过定性评估验证了平均95%的准确性，多名网络安全分析师在所有指标上的一致性为84%。这些结果强调了LLM驱动的数据挖掘用于实时网络威胁缓解的可行性和有效性。



## **30. Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework**

LLM可以处理WebShell检测吗？使用行为功能感知框架克服检测挑战 cs.CR

Published as a conference paper at COLM 2025

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2504.13811v3) [paper-pdf](http://arxiv.org/pdf/2504.13811v3)

**Authors**: Feijiang Han, Jiaming Zhang, Chuyi Deng, Jianheng Tang, Yunhuai Liu

**Abstract**: WebShell attacks, where malicious scripts are injected into web servers, pose a significant cybersecurity threat. Traditional ML and DL methods are often hampered by challenges such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models have emerged as powerful alternatives for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that, stemming from their distinct analytical strategies, larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all baseline models lag behind previous SOTA methods. With the application of BFAD, the performance of all LLMs improves significantly, yielding an average F1 score increase of 13.82%. Notably, larger models now outperform SOTA benchmarks, while smaller models such as Qwen-2.5-Coder-3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection and provides solutions to address the challenges in this task.

摘要: WebShell攻击（恶意脚本被注入网络服务器）构成了重大的网络安全威胁。传统的ML和DL方法经常受到需要大量训练数据、灾难性遗忘和较差的概括性等挑战的阻碍。最近，大型语言模型已成为代码相关任务的强大替代方案，但它们在WebShell检测中的潜力仍然没有得到充分的开发。在本文中，我们做出了两项贡献：（1）对七种LLM进行了全面评估，包括GPT-4、LLaMA 3.1 70 B和Qwen 2.5变体，使用26.59 K个PHP脚本的数据集针对传统的基于序列和图形的方法进行基准测试，以及（2）行为功能感知检测（BFAD）框架，旨在解决将LLM应用于该领域的特定挑战。我们的框架集成了三个组件：隔离恶意PHP函数调用的关键函数过滤器、捕获最具行为指示性的代码段的上下文感知代码提取策略，以及通过优先考虑最相关的演示来增强上下文学习的加权行为函数剖析基于区分性功能级配置文件。我们的结果表明，由于其不同的分析策略，较大的LLM可以实现近乎完美的精确度，但召回率较低，而较小的模型则表现出相反的权衡。然而，所有基线模型都落后于之前的SOTA方法。随着BFAD的应用，所有LLM的性能都显着提高，F1平均得分提高了13.82%。值得注意的是，大型型号的性能现在优于SOTA基准，而Qwen-2.5-Coder-3B等小型型号的性能与传统方法相比具有竞争力。这项工作是第一个探索LLM用于WebShell检测的可行性和局限性的工作，并提供了解决方案来应对这项任务中的挑战。



## **31. Membership Inference Attacks on LLM-based Recommender Systems**

对基于LLM的推荐系统的成员推断攻击 cs.IR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18665v1) [paper-pdf](http://arxiv.org/pdf/2508.18665v1)

**Authors**: Jiajie He, Yuechun Gu, Min-Chun Chen, Keke Chen

**Abstract**: Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.

摘要: 基于大型语言模型（LLM）的推荐系统（RecSys）可以灵活地调整推荐系统以适应不同的领域。它利用上下文学习（ICL），即提示来定制推荐功能，其中包括敏感的历史用户特定项目交互，例如，隐性反馈，例如点击的项目或明确的产品评论。此类私人信息可能会受到新型隐私攻击。然而，对于这个重要问题还没有进行任何研究。我们设计了四种成员资格推断攻击（MIA），旨在揭示受害者的历史互动是否被系统提示使用。它们是\r {直接询问、幻觉、相似性和中毒攻击}，每一种都利用了LLM或RecSys的独特功能。我们在三个用于开发ICL-LLM RecSys的LLM和两个著名的RecSys基准数据集上仔细评估了它们。结果证实，LLM RecSys上的MIA威胁是现实的：直接询问和中毒攻击显示出显着较高的攻击优势。我们还分析了影响这些攻击的因素，例如系统提示中的射击次数以及受害者在射击中的位置。



## **32. Large Language Model-Based Framework for Explainable Cyberattack Detection in Automatic Generation Control Systems**

自动发电控制系统中基于大语言模型的可解释网络攻击检测框架 cs.CR

Accepted Paper

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2507.22239v2) [paper-pdf](http://arxiv.org/pdf/2507.22239v2)

**Authors**: Muhammad Sharshar, Ahmad Mohammad Saber, Davor Svetinovic, Amr M. Youssef, Deepa Kundur, Ehab F. El-Saadany

**Abstract**: The increasing digitization of smart grids has improved operational efficiency but also introduced new cybersecurity vulnerabilities, such as False Data Injection Attacks (FDIAs) targeting Automatic Generation Control (AGC) systems. While machine learning (ML) and deep learning (DL) models have shown promise in detecting such attacks, their opaque decision-making limits operator trust and real-world applicability. This paper proposes a hybrid framework that integrates lightweight ML-based attack detection with natural language explanations generated by Large Language Models (LLMs). Classifiers such as LightGBM achieve up to 95.13% attack detection accuracy with only 0.004 s inference latency. Upon detecting a cyberattack, the system invokes LLMs, including GPT-3.5 Turbo, GPT-4 Turbo, and GPT-4o mini, to generate human-readable explanation of the event. Evaluated on 100 test samples, GPT-4o mini with 20-shot prompting achieved 93% accuracy in identifying the attack target, a mean absolute error of 0.075 pu in estimating attack magnitude, and 2.19 seconds mean absolute error (MAE) in estimating attack onset. These results demonstrate that the proposed framework effectively balances real-time detection with interpretable, high-fidelity explanations, addressing a critical need for actionable AI in smart grid cybersecurity.

摘要: 智能电网日益数字化提高了运营效率，但也带来了新的网络安全漏洞，例如针对自动发电控制（AGC）系统的虚假数据注入攻击（FDIA）。虽然机器学习（ML）和深度学习（DL）模型在检测此类攻击方面表现出了希望，但其不透明的决策限制了操作员的信任和现实世界的适用性。本文提出了一种混合框架，将轻量级基于ML的攻击检测与大型语言模型（LLM）生成的自然语言解释集成在一起。LightGBM等分类器可实现高达95.13%的攻击检测准确率，推理延迟仅为0.004秒。检测到网络攻击后，系统会调用LLM（包括GPT-3.5涡轮、GPT-4涡轮和GPT-4 o mini）来生成人类可读的事件解释。经过100个测试样本的评估，带20发提示的GPT-4 o mini识别攻击目标的准确率达到了93%，估计攻击幅度的平均绝对误差为0.075 pu，估计攻击开始的平均绝对误差为2.19秒。这些结果表明，拟议的框架有效地平衡了实时检测与可解释的高保真解释，满足了智能电网网络安全中对可操作人工智能的迫切需求。



## **33. Prefill-level Jailbreak: A Black-Box Risk Analysis of Large Language Models**

预填充级越狱：大型语言模型的黑匣子风险分析 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2504.21038v2) [paper-pdf](http://arxiv.org/pdf/2504.21038v2)

**Authors**: Yakai Li, Jiekang Hu, Weiduan Sang, Luping Ma, Dongsheng Nie, Weijuan Zhang, Aimin Yu, Yi Su, Qingjia Huang, Qihang Zhou

**Abstract**: Large Language Models face security threats from jailbreak attacks. Existing research has predominantly focused on prompt-level attacks while largely ignoring the underexplored attack surface of user-controlled response prefilling. This functionality allows an attacker to dictate the beginning of a model's output, thereby shifting the attack paradigm from persuasion to direct state manipulation.In this paper, we present a systematic black-box security analysis of prefill-level jailbreak attacks. We categorize these new attacks and evaluate their effectiveness across fourteen language models. Our experiments show that prefill-level attacks achieve high success rates, with adaptive methods exceeding 99% on several models. Token-level probability analysis reveals that these attacks work through initial-state manipulation by changing the first-token probability from refusal to compliance.Furthermore, we show that prefill-level jailbreak can act as effective enhancers, increasing the success of existing prompt-level attacks by 10 to 15 percentage points. Our evaluation of several defense strategies indicates that conventional content filters offer limited protection. We find that a detection method focusing on the manipulative relationship between the prompt and the prefill is more effective. Our findings reveal a gap in current LLM safety alignment and highlight the need to address the prefill attack surface in future safety training.

摘要: 大型语言模型面临越狱攻击的安全威胁。现有的研究主要集中在预算级攻击上，而很大程度上忽视了用户控制响应预填充的未充分探索的攻击表面。该功能允许攻击者决定模型输出的开始，从而将攻击范式从说服转变为直接状态操纵。在本文中，我们对预填充级越狱攻击进行了系统性的黑匣子安全分析。我们对这些新攻击进行了分类，并评估了它们在十四种语言模型中的有效性。我们的实验表明，预填充级攻击的成功率很高，自适应方法在几种模型上的成功率超过了99%。代币级概率分析表明，这些攻击通过初始状态操纵进行，将第一代币概率从拒绝改为遵守。此外，我们表明预填充级越狱可以充当有效的增强剂，将现有预算级攻击的成功率提高10到15个百分点。我们对几种防御策略的评估表明，传统内容过滤器提供的保护有限。我们发现，专注于提示和预填充之间操纵关系的检测方法更有效。我们的研究结果揭示了当前LLM安全调整方面的差距，并强调了在未来的安全培训中解决预填充攻击面的必要性。



## **34. Defending Against Prompt Injection With a Few DefensiveTokens**

使用一些防御代币来防御即时注射 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.07974v2) [paper-pdf](http://arxiv.org/pdf/2507.07974v2)

**Authors**: Sizhe Chen, Yizhu Wang, Nicholas Carlini, Chawin Sitawarin, David Wagner

**Abstract**: When large language model (LLM) systems interact with external data to perform complex tasks, a new attack, namely prompt injection, becomes a significant threat. By injecting instructions into the data accessed by the system, the attacker is able to override the initial user task with an arbitrary task directed by the attacker. To secure the system, test-time defenses, e.g., defensive prompting, have been proposed for system developers to attain security only when needed in a flexible manner. However, they are much less effective than training-time defenses that change the model parameters. Motivated by this, we propose DefensiveToken, a test-time defense with prompt injection robustness comparable to training-time alternatives. DefensiveTokens are newly inserted as special tokens, whose embeddings are optimized for security. In security-sensitive cases, system developers can append a few DefensiveTokens before the LLM input to achieve security with a minimal utility drop. In scenarios where security is less of a concern, developers can simply skip DefensiveTokens; the LLM system remains the same as there is no defense, generating high-quality responses. Thus, DefensiveTokens, if released alongside the model, allow a flexible switch between the state-of-the-art (SOTA) utility and almost-SOTA security at test time. The code is available at https://github.com/Sizhe-Chen/DefensiveToken.

摘要: 当大型语言模型（LLM）系统与外部数据交互以执行复杂任务时，一种新的攻击（即提示注入）将成为重大威胁。通过将指令注入系统访问的数据中，攻击者能够用攻击者指示的任意任务覆盖初始用户任务。为了保护系统，测试时防御措施，例如防御性提示已被建议供系统开发人员仅在需要时以灵活的方式获得安全性。然而，它们比改变模型参数的训练时防御有效得多。出于此动机，我们提出了DefensiveToken，这是一种测试时防御，具有与训练时替代方案相当的即时注入鲁棒性。DefensiveTokens作为特殊令牌新插入，其嵌入针对安全性进行了优化。在安全敏感的情况下，系统开发人员可以在LLM输入之前添加一些DefensiveTokens，以最小的实用程序下降来实现安全性。在安全性不太值得关注的场景中，开发人员可以简单地跳过DefensiveTokens; LLM系统由于没有防御而保持不变，从而生成高质量的响应。因此，DefensiveTokens如果与该模型一起发布，将允许在测试时在最先进的（SOTA）实用程序和几乎SOTA安全性之间灵活切换。该代码可在https://github.com/Sizhe-Chen/DefensiveToken上获取。



## **35. Confidential Prompting: Privacy-preserving LLM Inference on Cloud**

机密认证：云上的隐私保护LLM推理 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2409.19134v4) [paper-pdf](http://arxiv.org/pdf/2409.19134v4)

**Authors**: Caihua Li, In Gim, Lin Zhong

**Abstract**: This paper introduces a vision of confidential prompting: securing user prompts from untrusted, cloud-hosted large language model (LLM) provider while preserving model confidentiality, output invariance, and compute efficiency. As a first step toward this vision, we present Obfuscated Secure Partitioned Decoding (OSPD), a system built on two key innovations. First, Secure Partitioned Decoding (SPD) isolates user prompts within per-user processes residing in a confidential virtual machine (CVM) on the cloud, which are inaccessible for the cloud LLM while allowing it to generate tokens efficiently. Second, Prompt Obfuscation (PO) introduces a novel cryptographic technique that enhances SPD resilience against advanced prompt reconstruction attacks. Together, these innovations ensure OSPD protects both prompt and model confidentiality while maintaining service functionality. OSPD enables practical, privacy-preserving cloud-hosted LLM inference for sensitive applications, such as processing personal data, clinical records, and financial documents.

摘要: 本文介绍了保密提示的愿景：保护来自不受信任的云托管大型语言模型（LLM）提供商的用户提示，同时保留模型机密性、输出不变性和计算效率。作为实现这一愿景的第一步，我们提出了模糊安全分区解码（OSPD），这是一个基于两项关键创新的系统。首先，安全分区解码（SPD）将用户提示隔离在云上机密虚拟机（CGM）中驻留的每个用户进程中，云LLM无法访问这些进程，同时允许其高效地生成令牌。其次，即时混淆（PO）引入了一种新型加密技术，可以增强SPD抵御高级即时重建攻击的弹性。这些创新共同确保OSPD在维护服务功能的同时保护即时和模型机密性。OSPD为敏感应用程序（例如处理个人数据、临床记录和财务文档）提供实用的、保护隐私的云托管LLM推断。



## **36. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**

引导对话动力学，增强抵御多回合越狱攻击的稳健性 cs.CL

23 pages, 10 figures, 11 tables

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2503.00187v2) [paper-pdf](http://arxiv.org/pdf/2503.00187v2)

**Authors**: Hanjiang Hu, Alexander Robey, Changliu Liu

**Abstract**: Large language models (LLMs) are shown to be vulnerable to jailbreaking attacks where adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment, prompt-based steering and lightweight LLM guardrails baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off among safety, helpfulness and over-refusal. Check out the website here https://sites.google.com/view/llm-nbf/home . Our code is available on https://github.com/HanjiangHu/NBF-LLM .

摘要: 事实证明，大型语言模型（LLM）很容易受到越狱攻击，其中对抗性提示旨在引发有害反应。虽然现有的防御措施通过检测和过滤不安全的输入有效地减轻了单回合攻击，但它们无法对抗利用多次交互中的上下文漂移的多回合越狱，从而逐渐导致LLM远离安全行为。为了应对这一挑战，我们提出了一个基于安全控制理论的安全引导框架，确保多回合对话中不变的安全性。我们的方法使用状态空间表示对与LLM的对话进行建模，并引入一种新型的神经屏障函数（NBF）来主动检测和过滤不断变化的上下文中出现的有害查询。我们的方法通过学习一个考虑对抗性查询的安全预测器，在每一轮对话中实现不变的安全性，防止潜在的上下文漂移到越狱。在多个LLM下进行的大量实验表明，我们基于NBF的安全转向优于安全对准，基于转向的转向和轻型LLM护栏基线，为多转向越狱提供更强的防御，同时在安全性，有用性和过度拒绝之间保持更好的权衡。查看网站https://sites.google.com/view/llm-nbf/home。我们的代码可以在https://github.com/HanjiangHu/NBF-LLM上找到。



## **37. Stand on The Shoulders of Giants: Building JailExpert from Previous Attack Experience**

站在巨人的肩膀上：建造监狱专家来自之前的攻击经验 cs.CR

18 pages, EMNLP 2025 Main Conference

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.19292v1) [paper-pdf](http://arxiv.org/pdf/2508.19292v1)

**Authors**: Xi Wang, Songlei Jian, Shasha Li, Xiaopeng Li, Bin Ji, Jun Ma, Xiaodong Liu, Jing Wang, Feilong Bao, Jianfeng Zhang, Baosheng Wang, Jie Yu

**Abstract**: Large language models (LLMs) generate human-aligned content under certain safety constraints. However, the current known technique ``jailbreak prompt'' can circumvent safety-aligned measures and induce LLMs to output malicious content. Research on Jailbreaking can help identify vulnerabilities in LLMs and guide the development of robust security frameworks. To circumvent the issue of attack templates becoming obsolete as models evolve, existing methods adopt iterative mutation and dynamic optimization to facilitate more automated jailbreak attacks. However, these methods face two challenges: inefficiency and repetitive optimization, as they overlook the value of past attack experiences. To better integrate past attack experiences to assist current jailbreak attempts, we propose the \textbf{JailExpert}, an automated jailbreak framework, which is the first to achieve a formal representation of experience structure, group experiences based on semantic drift, and support the dynamic updating of the experience pool. Extensive experiments demonstrate that JailExpert significantly improves both attack effectiveness and efficiency. Compared to the current state-of-the-art black-box jailbreak methods, JailExpert achieves an average increase of 17\% in attack success rate and 2.7 times improvement in attack efficiency. Our implementation is available at \href{https://github.com/xiZAIzai/JailExpert}{XiZaiZai/JailExpert}

摘要: 大型语言模型（LLM）在一定的安全约束下生成与人类一致的内容。然而，当前已知的技术“越狱漏洞”可以规避安全性对齐的措施，并诱导LLM输出恶意内容。对越狱的研究可以帮助识别LLM中的漏洞，并指导健壮的安全框架的开发。为了避免攻击模板随着模型的发展而变得过时的问题，现有方法采用迭代变异和动态优化来促进更自动化的越狱攻击。然而，这些方法面临着两个挑战：效率低下和重复优化，因为它们忽视了过去攻击体验的价值。为了更好地整合过去的攻击经验以协助当前的越狱尝试，我们提出了\textBF{JailExpert}，这是一个自动越狱框架，它是第一个实现体验结构的正式表示、基于语义漂移的群组体验，并支持经验池的动态更新。大量实验表明，JailExpert显着提高了攻击有效性和效率。与当前最先进的黑匣子越狱方法相比，JailExpert的攻击成功率平均提高了17%，攻击效率提高了2.7倍。我们的实现可在\href{https：//github.com/xiZAIzai/JailExpert}{XiZaiZai/JailExpert}上获取



## **38. Head-Specific Intervention Can Induce Misaligned AI Coordination in Large Language Models**

特定于头部的干预可能会导致大型语言模型中的AI协调失调 cs.CL

Published at Transaction of Machine Learning Research 08/2025, Large  Language Models (LLMs), Interference-time activation shifting, Steerability,  Explainability, AI alignment, Interpretability

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2502.05945v3) [paper-pdf](http://arxiv.org/pdf/2502.05945v3)

**Authors**: Paul Darm, Annalisa Riccardi

**Abstract**: Robust alignment guardrails for large language models (LLMs) are becoming increasingly important with their widespread application. In contrast to previous studies, we demonstrate that inference-time activation interventions can bypass safety alignments and effectively steer model generations towards harmful AI coordination. Our method applies fine-grained interventions at specific attention heads, which we identify by probing each head in a simple binary choice task. We then show that interventions on these heads generalise to the open-ended generation setting, effectively circumventing safety guardrails. We demonstrate that intervening on a few attention heads is more effective than intervening on full layers or supervised fine-tuning. We further show that only a few example completions are needed to compute effective steering directions, which is an advantage over classical fine-tuning. We also demonstrate that applying interventions in the negative direction can prevent a common jailbreak attack. Our results suggest that, at the attention head level, activations encode fine-grained linearly separable behaviours. Practically, the approach offers a straightforward methodology to steer large language model behaviour, which could be extended to diverse domains beyond safety, requiring fine-grained control over the model output. The code and datasets for this study can be found on https://github.com/PaulDrm/targeted_intervention.

摘要: 随着大型语言模型（LLM）的广泛应用，其稳健的对齐护栏变得越来越重要。与之前的研究相比，我们证明，推理时激活干预可以绕过安全对齐，并有效地引导模型一代转向有害的人工智能协调。我们的方法对特定的注意力头应用细粒度的干预，我们通过在简单的二元选择任务中探测每个注意力头来识别这些注意力头。然后，我们表明，针对这些方面的干预措施普遍适用于开放式的一代环境，有效地绕过了安全护栏。我们证明，干预几个注意头是更有效的比干预全层或监督微调。我们进一步表明，只需要几个例子完成计算有效的转向方向，这是一个优势，经典的微调。我们还证明，在消极的方向上应用干预措施可以防止常见的越狱攻击。我们的研究结果表明，在注意头水平，激活编码细粒度的线性可分离的行为。实际上，该方法提供了一种简单的方法来引导大型语言模型行为，该方法可以扩展到安全以外的不同领域，需要对模型输出进行细粒度控制。本研究的代码和数据集可在https://github.com/PaulDrm/targeted_intervention上找到。



## **39. Speculative Safety-Aware Decoding**

推测性安全意识解码 cs.LG

EMNLP'2025 main conference; more experiments will be added to the  coming camera-ready version

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17739v1) [paper-pdf](http://arxiv.org/pdf/2508.17739v1)

**Authors**: Xuekang Wang, Shengyu Zhu, Xueqi Cheng

**Abstract**: Despite extensive efforts to align Large Language Models (LLMs) with human values and safety rules, jailbreak attacks that exploit certain vulnerabilities continuously emerge, highlighting the need to strengthen existing LLMs with additional safety properties to defend against these attacks. However, tuning large models has become increasingly resource-intensive and may have difficulty ensuring consistent performance. We introduce Speculative Safety-Aware Decoding (SSD), a lightweight decoding-time approach that equips LLMs with the desired safety property while accelerating inference. We assume that there exists a small language model that possesses this desired property. SSD integrates speculative sampling during decoding and leverages the match ratio between the small and composite models to quantify jailbreak risks. This enables SSD to dynamically switch between decoding schemes to prioritize utility or safety, to handle the challenge of different model capacities. The output token is then sampled from a new distribution that combines the distributions of the original and the small models. Experimental results show that SSD successfully equips the large model with the desired safety property, and also allows the model to remain helpful to benign queries. Furthermore, SSD accelerates the inference time, thanks to the speculative sampling design.

摘要: 尽管人们广泛努力将大型语言模型（LLM）与人类价值观和安全规则保持一致，但利用某些漏洞的越狱攻击不断出现，凸显了需要通过额外的安全属性来加强现有的LLM以抵御这些攻击。然而，调整大型模型已变得越来越需要资源密集型，并且可能难以确保一致的性能。我们引入了推测性安全感知解码（SSD），这是一种轻量级解码时间方法，可为LLM配备所需的安全属性，同时加速推理。我们假设存在一个具有这种所需属性的小型语言模型。SSD在解码过程中集成了推测性采样，并利用小模型和复合模型之间的匹配率来量化越狱风险。这使得SSD能够在解码方案之间动态切换，以优先考虑实用性或安全性，以应对不同模型容量的挑战。然后，从结合原始模型和小模型的分布的新分布中对输出令牌进行采样。实验结果表明，SSD成功地装备了大型模型所需的安全属性，也允许模型保持有益的良性查询。此外，由于推测性抽样设计，SSD加快了推理时间。



## **40. Prompt-in-Content Attacks: Exploiting Uploaded Inputs to Hijack LLM Behavior**

内容预算攻击：利用非法输入劫持LLM行为 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.19287v1) [paper-pdf](http://arxiv.org/pdf/2508.19287v1)

**Authors**: Zhuotao Lian, Weiyu Wang, Qingkui Zeng, Toru Nakanishi, Teruaki Kitasuka, Chunhua Su

**Abstract**: Large Language Models (LLMs) are widely deployed in applications that accept user-submitted content, such as uploaded documents or pasted text, for tasks like summarization and question answering. In this paper, we identify a new class of attacks, prompt in content injection, where adversarial instructions are embedded in seemingly benign inputs. When processed by the LLM, these hidden prompts can manipulate outputs without user awareness or system compromise, leading to biased summaries, fabricated claims, or misleading suggestions. We demonstrate the feasibility of such attacks across popular platforms, analyze their root causes including prompt concatenation and insufficient input isolation, and discuss mitigation strategies. Our findings reveal a subtle yet practical threat in real-world LLM workflows.

摘要: 大型语言模型（LLM）广泛部署在接受用户提交的内容（例如上传的文档或粘贴的文本）的应用程序中，以执行总结和问答等任务。在本文中，我们识别了一类新的攻击，在内容注入中提示，其中对抗性指令被嵌入到看似良性的输入中。当LLM处理时，这些隐藏的提示可能会在用户意识不到或系统损害的情况下操纵输出，从而导致有偏见的摘要、捏造的主张或误导性的建议。我们展示了此类攻击在流行平台上的可行性，分析了其根本原因，包括迅速级联和输入隔离不足，并讨论了缓解策略。我们的研究结果揭示了现实世界LLM工作流程中一个微妙但实际的威胁。



## **41. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

攻击LLM和AI代理：针对大型语言模型的广告嵌入攻击 cs.CR

7 pages, 2 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17674v1) [paper-pdf](http://arxiv.org/pdf/2508.17674v1)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.

摘要: 我们引入了广告嵌入攻击（AEA），这是一种新型LLM安全威胁，可以悄悄地将促销或恶意内容注入模型输出和AI代理中。AEA通过两种低成本载体运作：（1）劫持第三方服务分发平台以预先设置对抗提示，以及（2）发布经过攻击者数据微调的后门开源检查点。与降低准确性的传统攻击不同，AEA破坏了信息完整性，导致模型在看起来正常的情况下返回秘密广告、宣传或仇恨言论。我们详细介绍了攻击管道，绘制了五个利益相关者受害者群体，并提出了一种初步的基于预算的自我检查防御，该防御可以减轻这些注入，而无需额外的模型再培训。我们的调查结果揭示了LLM安全方面存在一个紧迫且未充分解决的差距，并呼吁人工智能安全界协调检测、审计和政策响应。



## **42. TombRaider: Entering the Vault of History to Jailbreak Large Language Models**

TombRaider：进入历史宝库越狱大型语言模型 cs.CR

Main Conference of EMNLP

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2501.18628v2) [paper-pdf](http://arxiv.org/pdf/2501.18628v2)

**Authors**: Junchen Ding, Jiahao Zhang, Yi Liu, Ziqi Ding, Gelei Deng, Yuekang Li

**Abstract**: Warning: This paper contains content that may involve potentially harmful behaviours, discussed strictly for research purposes.   Jailbreak attacks can hinder the safety of Large Language Model (LLM) applications, especially chatbots. Studying jailbreak techniques is an important AI red teaming task for improving the safety of these applications. In this paper, we introduce TombRaider, a novel jailbreak technique that exploits the ability to store, retrieve, and use historical knowledge of LLMs. TombRaider employs two agents, the inspector agent to extract relevant historical information and the attacker agent to generate adversarial prompts, enabling effective bypassing of safety filters. We intensively evaluated TombRaider on six popular models. Experimental results showed that TombRaider could outperform state-of-the-art jailbreak techniques, achieving nearly 100% attack success rates (ASRs) on bare models and maintaining over 55.4% ASR against defence mechanisms. Our findings highlight critical vulnerabilities in existing LLM safeguards, underscoring the need for more robust safety defences.

摘要: 警告：本文包含可能涉及潜在有害行为的内容，严格出于研究目的进行讨论。   越狱攻击可能会阻碍大型语言模型（LLM）应用程序的安全性，尤其是聊天机器人。研究越狱技术是提高这些应用安全性的一项重要人工智能红色团队任务。本文中，我们介绍了TombRaider，这是一种新型越狱技术，它利用了存储、检索和使用LLM历史知识的能力。TombRaider使用两个代理，检查员代理提取相关历史信息，攻击者代理生成对抗提示，从而有效绕过安全过滤器。我们对TombRaider的六款热门型号进行了深入评估。实验结果表明，TombRaider的性能优于最先进的越狱技术，在裸模型上实现了近100%的攻击成功率（ASB），并在防御机制下保持超过55.4%的ASB。我们的调查结果强调了现有LLM保障措施中的关键漏洞，强调了更强大的安全防御的必要性。



## **43. RL-Finetuned LLMs for Privacy-Preserving Synthetic Rewriting**

RL微调LLM用于隐私保护综合重写 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.19286v1) [paper-pdf](http://arxiv.org/pdf/2508.19286v1)

**Authors**: Zhan Shi, Yefeng Yuan, Yuhong Liu, Liang Cheng, Yi Fang

**Abstract**: The performance of modern machine learning systems depends on access to large, high-quality datasets, often sourced from user-generated content or proprietary, domain-specific corpora. However, these rich datasets inherently contain sensitive personal information, raising significant concerns about privacy, data security, and compliance with regulatory frameworks. While conventional anonymization techniques can remove explicit identifiers, such removal may result in performance drop in downstream machine learning tasks. More importantly, simple anonymization may not be effective against inference attacks that exploit implicit signals such as writing style, topical focus, or demographic cues, highlighting the need for more robust privacy safeguards during model training. To address the challenging issue of balancing user privacy and data utility, we propose a reinforcement learning framework that fine-tunes a large language model (LLM) using a composite reward function that jointly optimizes for explicit and implicit privacy, semantic fidelity, and output diversity. To effectively capture population level regularities, the privacy reward combines semantic cues with structural patterns derived from a minimum spanning tree (MST) over latent representations. By modeling these privacy-sensitive signals in their distributional context, the proposed approach guides the model to generate synthetic rewrites that preserve utility while mitigating privacy risks. Empirical results show that the proposed method significantly enhances author obfuscation and privacy metrics without degrading semantic quality, providing a scalable and model-agnostic solution for privacy preserving data generation in the era of large language models.

摘要: 现代机器学习系统的性能取决于对大型、高质量数据集的访问，这些数据集通常来自用户生成的内容或专有的、特定领域的文集。然而，这些丰富的数据集本质上包含敏感的个人信息，引发了人们对隐私、数据安全和监管框架合规性的严重担忧。虽然传统的匿名化技术可以删除显式标识符，但这种删除可能会导致下游机器学习任务的性能下降。更重要的是，简单的匿名化可能无法有效地对抗利用写作风格、话题焦点或人口统计线索等隐性信号的推理攻击，这凸显了模型训练期间需要更强大的隐私保护措施。为了解决平衡用户隐私和数据效用的挑战性问题，我们提出了一种强化学习框架，该框架使用复合奖励函数来微调大型语言模型（LLM），该函数联合优化显式和隐式隐私、语义保真度和输出多样性。为了有效地捕获人口水平的预设，隐私奖励将语义线索与从潜在表示上的最小生成树（MST）推导出的结构模式相结合。通过在分布环境中对这些隐私敏感的信号进行建模，提出的方法引导模型生成合成重写，以保留效用，同时降低隐私风险。实验结果表明，所提出的方法在不降低语义质量的情况下显着增强了作者混淆和隐私指标，为大型语言模型时代的隐私保护数据生成提供了可扩展且模型不可知的解决方案。



## **44. Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models**

自适应语言搜索（AFP）增强多模式大型语言模型中的网络钓鱼网页检测 cs.CL

Published at ACL 2025 SRW, 9 pages, 3 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.13357v2) [paper-pdf](http://arxiv.org/pdf/2507.13357v2)

**Authors**: Atharva Bhargude, Ishan Gonehal, Dave Yoon, Kaustubh Vinnakota, Chandler Haney, Aaron Sandoval, Kevin Zhu

**Abstract**: Phishing attacks represent a significant cybersecurity threat, necessitating adaptive detection techniques. This study explores few-shot Adaptive Linguistic Prompting (ALP) in detecting phishing webpages through the multimodal capabilities of state-of-the-art large language models (LLMs) such as GPT-4o and Gemini 1.5 Pro. ALP is a structured semantic reasoning method that guides LLMs to analyze textual deception by breaking down linguistic patterns, detecting urgency cues, and identifying manipulative diction commonly found in phishing content. By integrating textual, visual, and URL-based analysis, we propose a unified model capable of identifying sophisticated phishing attempts. Our experiments demonstrate that ALP significantly enhances phishing detection accuracy by guiding LLMs through structured reasoning and contextual analysis. The findings highlight the potential of ALP-integrated multimodal LLMs to advance phishing detection frameworks, achieving an F1-score of 0.93, surpassing traditional approaches. These results establish a foundation for more robust, interpretable, and adaptive linguistic-based phishing detection systems using LLMs.

摘要: 网络钓鱼攻击是一个重大的网络安全威胁，需要自适应检测技术。本研究探索了通过GPT-4 o和Gemini 1.5 Pro等最先进大型语言模型（LLM）的多模式功能检测网络钓鱼网页的几次自适应语言预测（AFP）。ALA是一种结构化语义推理方法，通过分解语言模式、检测紧迫性线索和识别网络钓鱼内容中常见的操纵性措辞来指导LLM分析文本欺骗。通过集成文本、视觉和基于URL的分析，我们提出了一个能够识别复杂的网络钓鱼企图的统一模型。我们的实验表明，通过结构化推理和上下文分析来指导LLM，ALA显着提高了网络钓鱼检测的准确性。研究结果凸显了整合了AFP的多模式LLM在推进网络钓鱼检测框架方面的潜力，F1评分达到0.93，超越了传统方法。这些结果为使用LLM的更稳健、可解释和自适应的基于语言的网络钓鱼检测系统奠定了基础。



## **45. Unified attacks to large language model watermarks: spoofing and scrubbing in unauthorized knowledge distillation**

对大型语言模型水印的统一攻击：未经授权的知识提炼中的欺骗和擦洗 cs.CL

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2504.17480v4) [paper-pdf](http://arxiv.org/pdf/2504.17480v4)

**Authors**: Xin Yi, Yue Li, Shunfan Zheng, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Watermarking has emerged as a critical technique for combating misinformation and protecting intellectual property in large language models (LLMs). A recent discovery, termed watermark radioactivity, reveals that watermarks embedded in teacher models can be inherited by student models through knowledge distillation. On the positive side, this inheritance allows for the detection of unauthorized knowledge distillation by identifying watermark traces in student models. However, the robustness of watermarks against scrubbing attacks and their unforgeability in the face of spoofing attacks under unauthorized knowledge distillation remain largely unexplored. Existing watermark attack methods either assume access to model internals or fail to simultaneously support both scrubbing and spoofing attacks. In this work, we propose Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework that enables bidirectional attacks under unauthorized knowledge distillation. Our approach employs contrastive decoding to extract corrupted or amplified watermark texts via comparing outputs from the student model and weakly watermarked references, followed by bidirectional distillation to train new student models capable of watermark removal and watermark forgery, respectively. Extensive experiments show that CDG-KD effectively performs attacks while preserving the general performance of the distilled model. Our findings underscore critical need for developing watermarking schemes that are robust and unforgeable.

摘要: 水印已成为打击错误信息和保护大型语言模型（LLM）知识产权的关键技术。最近的一项发现称为水印放射性，揭示了教师模型中嵌入的水印可以通过知识蒸馏被学生模型继承。从积极的方面来说，这种继承允许通过识别学生模型中的水印痕迹来检测未经授权的知识提炼。然而，水印对擦洗攻击的鲁棒性及其在未经授权的知识提炼下面对欺骗攻击时的不可伪造性在很大程度上仍然没有被探索。现有的水印攻击方法要么假设访问模型内部内容，要么无法同时支持擦洗和欺骗攻击。在这项工作中，我们提出了对比解码引导知识蒸馏（CDG-KD），这是一个统一框架，可以在未经授权的知识蒸馏下实现双向攻击。我们的方法采用对比解码，通过比较学生模型和弱水印参考的输出来提取损坏或放大的水印文本，然后进行双向蒸馏以分别训练能够去除水印和伪造水印的新学生模型。大量实验表明，CDG-KD可以有效地执行攻击，同时保持提取模型的一般性能。我们的研究结果强调了开发稳健且不可伪造的水印方案的迫切需要。



## **46. Defending against Jailbreak through Early Exit Generation of Large Language Models**

通过早期退出生成大型语言模型抵御越狱 cs.AI

ICONIP 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2408.11308v2) [paper-pdf](http://arxiv.org/pdf/2408.11308v2)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. We introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85% in comparison with 50% for the present SOTAs, with minimal impact on the utility of LLMs.

摘要: 大型语言模型（LLM）在各种应用中越来越受到关注。尽管如此，随着一些用户试图利用这些模型进行恶意目的，包括合成受控物质和传播虚假信息，人们越来越担心。为了降低此类风险，“对齐”技术的概念被开发出来。然而，最近的研究表明，使用复杂的即时工程或对抗性后缀（一种被称为“越狱”的技术）可能会破坏这种对齐。“我们的研究从LLM的类人类生成过程中汲取线索。我们发现，虽然越狱提示可能会产生类似于良性提示的输出日志，但它们在模型潜在空间中的初始嵌入往往更类似于恶意提示的嵌入。利用这一发现，我们建议利用LLM的早期Transformer输出作为检测恶意输入并立即终止生成的手段。我们为LLM引入了一种简单但重要的防御方法，称为EEG-Defender。我们对三种模型的十种越狱方法进行了全面实验。我们的结果表明，EEG-Defender能够大幅降低攻击成功率（ASB），大约为85%，而当前SOTA的攻击成功率为50%，对LLM的实用性影响最小。



## **47. Exploring the Vulnerability of the Content Moderation Guardrail in Large Language Models via Intent Manipulation**

通过意图操纵探索大型语言模型中内容审核保护的脆弱性 cs.CL

Accepted for EMNLP'25 Findings. TL;DR: We propose a new two-stage  intent-based prompt-refinement framework, IntentPrompt, that aims to explore  the vulnerability of LLMs' content moderation guardrails by refining prompts  into benign-looking declarative forms via intent manipulation for red-teaming  purposes

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2505.18556v2) [paper-pdf](http://arxiv.org/pdf/2505.18556v2)

**Authors**: Jun Zhuang, Haibo Jin, Ye Zhang, Zhengjian Kang, Wenbin Zhang, Gaby G. Dagher, Haohan Wang

**Abstract**: Intent detection, a core component of natural language understanding, has considerably evolved as a crucial mechanism in safeguarding large language models (LLMs). While prior work has applied intent detection to enhance LLMs' moderation guardrails, showing a significant success against content-level jailbreaks, the robustness of these intent-aware guardrails under malicious manipulations remains under-explored. In this work, we investigate the vulnerability of intent-aware guardrails and demonstrate that LLMs exhibit implicit intent detection capabilities. We propose a two-stage intent-based prompt-refinement framework, IntentPrompt, that first transforms harmful inquiries into structured outlines and further reframes them into declarative-style narratives by iteratively optimizing prompts via feedback loops to enhance jailbreak success for red-teaming purposes. Extensive experiments across four public benchmarks and various black-box LLMs indicate that our framework consistently outperforms several cutting-edge jailbreak methods and evades even advanced Intent Analysis (IA) and Chain-of-Thought (CoT)-based defenses. Specifically, our "FSTR+SPIN" variant achieves attack success rates ranging from 88.25% to 96.54% against CoT-based defenses on the o1 model, and from 86.75% to 97.12% on the GPT-4o model under IA-based defenses. These findings highlight a critical weakness in LLMs' safety mechanisms and suggest that intent manipulation poses a growing challenge to content moderation guardrails.

摘要: 意图检测是自然语言理解的核心组成部分，已经发展成为保护大型语言模型（LLM）的关键机制。虽然先前的工作已经应用意图检测来增强LLM的适度护栏，显示出对内容级越狱的显著成功，但这些意图感知护栏在恶意操纵下的鲁棒性仍然未被充分探索。在这项工作中，我们调查意图感知护栏的脆弱性，并证明LLM表现出隐式意图检测能力。我们提出了一个两阶段的基于意图的越狱细化框架，IntentPrompt，首先将有害的查询转换为结构化的大纲，并通过反馈循环迭代优化提示，以提高越狱成功率，从而进一步将其重新构建为声明式的叙述。针对四个公共基准测试和各种黑匣子LLM的广泛实验表明，我们的框架始终优于几种尖端的越狱方法，甚至可以规避高级意图分析（IA）和基于思想链（CoT）的防御。具体来说，我们的“FTR +SPIN”变体在o 1模型上针对基于CoT的防御的攻击成功率从88.25%到96.54%不等，在基于IA的防御下，在GPT-4 o模型上的攻击成功率从86.75%到97.12%不等。这些发现凸显了LLM安全机制的一个严重弱点，并表明意图操纵对内容审核护栏构成了越来越大的挑战。



## **48. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

相信我，我知道这个功能：使用偏差劫持LLM静态分析 cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17361v1) [paper-pdf](http://arxiv.org/pdf/2508.17361v1)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective, but also transferable across models (GPT-4o, Claude 3.5, Gemini 2.0) and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.

摘要: 大型语言模型（LLM）越来越被信任大规模执行自动代码审查和静态分析，支持漏洞检测、总结和重构等任务。在本文中，我们识别并利用基于LLM的代码分析中的一个关键漏洞：一种抽象偏见，导致模型过度概括熟悉的编程模式并忽略小而有意义的错误。对手可以利用这个盲点来劫持LLM解释的控制流，只需最少的编辑，并且不会影响实际的运行时行为。我们将这种攻击称为熟悉模式攻击（FTA）。   我们开发了一种全自动的黑匣子算法，可以发现并将FPA注入目标代码。我们的评估表明，PFA不仅有效，而且可以跨模型（GPT-4 o、Claude 3.5、Gemini 2.0）移植，并且跨编程语言（Python、C、Rust、Go）通用。此外，即使通过强大的系统提示明确警告模型有关攻击，FPA仍然有效。最后，我们探讨了PFA的积极、防御性用途，并讨论了它们对面向代码的LLM的可靠性和安全性的更广泛影响。



## **49. Risk Assessment and Security Analysis of Large Language Models**

大型语言模型的风险评估与安全性分析 cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17329v1) [paper-pdf](http://arxiv.org/pdf/2508.17329v1)

**Authors**: Xiaoyan Zhang, Dongyang Lyu, Xiaoqi Li

**Abstract**: As large language models (LLMs) expose systemic security challenges in high risk applications, including privacy leaks, bias amplification, and malicious abuse, there is an urgent need for a dynamic risk assessment and collaborative defence framework that covers their entire life cycle. This paper focuses on the security problems of large language models (LLMs) in critical application scenarios, such as the possibility of disclosure of user data, the deliberate input of harmful instructions, or the models bias. To solve these problems, we describe the design of a system for dynamic risk assessment and a hierarchical defence system that allows different levels of protection to cooperate. This paper presents a risk assessment system capable of evaluating both static and dynamic indicators simultaneously. It uses entropy weighting to calculate essential data, such as the frequency of sensitive words, whether the API call is typical, the realtime risk entropy value is significant, and the degree of context deviation. The experimental results show that the system is capable of identifying concealed attacks, such as role escape, and can perform rapid risk evaluation. The paper uses a hybrid model called BERT-CRF (Bidirectional Encoder Representation from Transformers) at the input layer to identify and filter malicious commands. The model layer uses dynamic adversarial training and differential privacy noise injection technology together. The output layer also has a neural watermarking system that can track the source of the content. In practice, the quality of this method, especially important in terms of customer service in the financial industry.

摘要: 由于大型语言模型（LLM）在高风险应用中暴露出系统性的安全挑战，包括隐私泄露，偏见放大和恶意滥用，因此迫切需要一个涵盖其整个生命周期的动态风险评估和协作防御框架。本文重点研究了大型语言模型在关键应用场景中的安全问题，如用户数据泄露的可能性、有害指令的故意输入、模型偏差等。为了解决这些问题，我们描述了一个系统的设计，动态风险评估和分级防御系统，允许不同级别的保护合作。本文提出了一种能够同时评估静态和动态指标的风险评估系统。它使用熵加权来计算基本数据，例如敏感词的频率、API调用是否典型、实时风险熵值是否重要以及上下文偏离程度。实验结果表明，该系统能够识别角色逃避等隐藏攻击，并能够进行快速风险评估。该论文在输入层使用了一种名为BERT-RF（来自Transformers的双向编码器表示）的混合模型来识别和过滤恶意命令。模型层结合使用动态对抗训练和差异隐私噪音注入技术。输出层还具有一个可以跟踪内容来源的神经水印系统。在实践中，这种方法的质量对于金融行业的客户服务尤其重要。



## **50. Fine-Grained Safety Neurons with Training-Free Continual Projection to Reduce LLM Fine Tuning Risks**

具有免训练连续投影的细粒度安全神经元，以降低LLM微调风险 cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.09190v3) [paper-pdf](http://arxiv.org/pdf/2508.09190v3)

**Authors**: Bing Han, Feifei Zhao, Dongcheng Zhao, Guobin Shen, Ping Wu, Yu Shi, Yi Zeng

**Abstract**: Fine-tuning as service injects domain-specific knowledge into large language models (LLMs), while challenging the original alignment mechanisms and introducing safety risks. A series of defense strategies have been proposed for the alignment, fine-tuning, and post-fine-tuning phases, where most post-fine-tuning defenses rely on coarse-grained safety layer mapping. These methods lack a comprehensive consideration of both safety layers and fine-grained neurons, limiting their ability to efficiently balance safety and utility. To address this, we propose the Fine-Grained Safety Neurons (FGSN) with Training-Free Continual Projection method to reduce the fine-tuning safety risks. FGSN inherently integrates the multi-scale interactions between safety layers and neurons, localizing sparser and more precise fine-grained safety neurons while minimizing interference with downstream task neurons. We then project the safety neuron parameters onto safety directions, improving model safety while aligning more closely with human preferences. Extensive experiments across multiple fine-tuned LLM models demonstrate that our method significantly reduce harmfulness scores and attack success rates with minimal parameter modifications, while preserving the model's utility. Furthermore, by introducing a task-specific, multi-dimensional heterogeneous safety neuron cluster optimization mechanism, we achieve continual defense and generalization capability against unforeseen emerging safety concerns.

摘要: 微调即服务将特定领域的知识注入到大型语言模型（LLM）中，同时挑战了原始的对齐机制并引入了安全风险。针对对齐、微调和微调后阶段提出了一系列防御策略，其中大多数微调后防御依赖于粗粒度安全层映射。这些方法缺乏对安全层和细粒度神经元的综合考虑，限制了它们有效平衡安全性和实用性的能力。为了解决这个问题，我们提出了细粒度安全神经元（FGSN）与训练免费连续投影方法，以减少微调的安全风险。FGSN固有地集成了安全层和神经元之间的多尺度交互，定位更稀疏和更精确的细粒度安全神经元，同时最大限度地减少对下游任务神经元的干扰。然后，我们将安全神经元参数投影到安全方向上，提高模型的安全性，同时更紧密地与人类偏好保持一致。在多个微调的LLM模型上进行的广泛实验表明，我们的方法在保持模型实用性的同时，以最小的参数修改显着降低了危害分数和攻击成功率。此外，通过引入特定于任务的多维异构安全神经元簇优化机制，我们实现了对不可预见的新出现的安全问题的持续防御和泛化能力。



