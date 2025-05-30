# Latest Large Language Model Attack Papers
**update at 2025-05-30 15:21:12**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. MCP Safety Training: Learning to Refuse Falsely Benign MCP Exploits using Improved Preference Alignment**

LCP安全培训：学会使用改进的偏好对齐来拒绝虚假良性的LCP利用 cs.LG

27 pages, 19 figures, 4 tables

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23634v1) [paper-pdf](http://arxiv.org/pdf/2505.23634v1)

**Authors**: John Halloran

**Abstract**: The model context protocol (MCP) has been widely adapted as an open standard enabling the seamless integration of generative AI agents. However, recent work has shown the MCP is susceptible to retrieval-based "falsely benign" attacks (FBAs), allowing malicious system access and credential theft, but requiring that users download compromised files directly to their systems. Herein, we show that the threat model of MCP-based attacks is significantly broader than previously thought, i.e., attackers need only post malicious content online to deceive MCP agents into carrying out their attacks on unsuspecting victims' systems.   To improve alignment guardrails against such attacks, we introduce a new MCP dataset of FBAs and (truly) benign samples to explore the effectiveness of direct preference optimization (DPO) for the refusal training of large language models (LLMs). While DPO improves model guardrails against such attacks, we show that the efficacy of refusal learning varies drastically depending on the model's original post-training alignment scheme--e.g., GRPO-based LLMs learn to refuse extremely poorly. Thus, to further improve FBA refusals, we introduce Retrieval Augmented Generation for Preference alignment (RAG-Pref), a novel preference alignment strategy based on RAG. We show that RAG-Pref significantly improves the ability of LLMs to refuse FBAs, particularly when combined with DPO alignment, thus drastically improving guardrails against MCP-based attacks.

摘要: 模型上下文协议（HCP）已被广泛采用为开放标准，实现生成性人工智能代理的无缝集成。然而，最近的工作表明，HCP很容易受到基于检索的“错误良性”攻击（FBA），允许恶意系统访问和凭证盗窃，但要求用户将受损文件直接下载到其系统。在此，我们表明基于MPP的攻击的威胁模型比之前想象的要广泛得多，即攻击者只需在网上发布恶意内容就可以欺骗HCP代理对毫无戒心的受害者系统实施攻击。   为了改善针对此类攻击的对齐护栏，我们引入了一个由FBA和（真正）良性样本组成的新的HCP数据集，以探索直接偏好优化（DPO）用于大型语言模型（LLM）拒绝训练的有效性。虽然DPO改善了模型针对此类攻击的护栏，但我们表明拒绝学习的功效根据模型原始的训练后对齐方案而变化很大--例如，基于GRPO的LLM拒绝能力极差。因此，为了进一步改善FBA拒绝，我们引入了偏好对齐检索增强生成（RAG-Pref），这是一种基于RAG的新型偏好对齐策略。我们表明，RAG-Pref显着提高了LLM拒绝FBA的能力，特别是与DPO对齐相结合时，从而大大改善了针对基于MPP的攻击的护栏。



## **2. Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models**

合并劫持：对大型语言模型合并的后门攻击 cs.CR

This paper is accepted by ACL 2025 main conference

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23561v1) [paper-pdf](http://arxiv.org/pdf/2505.23561v1)

**Authors**: Zenghui Yuan, Yangming Xu, Jiawen Shi, Pan Zhou, Lichao Sun

**Abstract**: Model merging for Large Language Models (LLMs) directly fuses the parameters of different models finetuned on various tasks, creating a unified model for multi-domain tasks. However, due to potential vulnerabilities in models available on open-source platforms, model merging is susceptible to backdoor attacks. In this paper, we propose Merge Hijacking, the first backdoor attack targeting model merging in LLMs. The attacker constructs a malicious upload model and releases it. Once a victim user merges it with any other models, the resulting merged model inherits the backdoor while maintaining utility across tasks. Merge Hijacking defines two main objectives-effectiveness and utility-and achieves them through four steps. Extensive experiments demonstrate the effectiveness of our attack across different models, merging algorithms, and tasks. Additionally, we show that the attack remains effective even when merging real-world models. Moreover, our attack demonstrates robustness against two inference-time defenses (Paraphrasing and CLEANGEN) and one training-time defense (Fine-pruning).

摘要: 大型语言模型（LLM）的模型合并直接融合对各种任务进行微调的不同模型的参数，为多领域任务创建统一模型。然而，由于开源平台上可用的模型存在潜在漏洞，模型合并很容易受到后门攻击。本文提出了合并劫持，这是LLM中第一个针对合并的后门攻击模型。攻击者构建恶意上传模型并将其发布。一旦受害用户将其与任何其他模型合并，生成的合并模型将继承后门，同时保持跨任务的实用性。合并劫持定义了两个主要目标--有效性和实用性--并通过四个步骤实现它们。大量实验证明了我们在不同模型、合并算法和任务中的攻击的有效性。此外，我们表明，即使在合并现实世界模型时，攻击仍然有效。此外，我们的攻击表现出了对两种推理时防御（Paraphrapping和CleangEN）和一种训练时防御（Fine-修剪）的鲁棒性。



## **3. SafeScientist: Toward Risk-Aware Scientific Discoveries by LLM Agents**

安全科学家：LLM代理人的风险意识科学发现 cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23559v1) [paper-pdf](http://arxiv.org/pdf/2505.23559v1)

**Authors**: Kunlun Zhu, Jiaxun Zhang, Ziheng Qi, Nuoxing Shang, Zijia Liu, Peixuan Han, Yue Su, Haofei Yu, Jiaxuan You

**Abstract**: Recent advancements in large language model (LLM) agents have significantly accelerated scientific discovery automation, yet concurrently raised critical ethical and safety concerns. To systematically address these challenges, we introduce \textbf{SafeScientist}, an innovative AI scientist framework explicitly designed to enhance safety and ethical responsibility in AI-driven scientific exploration. SafeScientist proactively refuses ethically inappropriate or high-risk tasks and rigorously emphasizes safety throughout the research process. To achieve comprehensive safety oversight, we integrate multiple defensive mechanisms, including prompt monitoring, agent-collaboration monitoring, tool-use monitoring, and an ethical reviewer component. Complementing SafeScientist, we propose \textbf{SciSafetyBench}, a novel benchmark specifically designed to evaluate AI safety in scientific contexts, comprising 240 high-risk scientific tasks across 6 domains, alongside 30 specially designed scientific tools and 120 tool-related risk tasks. Extensive experiments demonstrate that SafeScientist significantly improves safety performance by 35\% compared to traditional AI scientist frameworks, without compromising scientific output quality. Additionally, we rigorously validate the robustness of our safety pipeline against diverse adversarial attack methods, further confirming the effectiveness of our integrated approach. The code and data will be available at https://github.com/ulab-uiuc/SafeScientist. \textcolor{red}{Warning: this paper contains example data that may be offensive or harmful.}

摘要: 大型语言模型（LLM）代理的最新进展显着加速了科学发现自动化，但同时提出了关键的伦理和安全问题。为了系统地应对这些挑战，我们引入了一个创新的人工智能科学家框架，旨在增强人工智能驱动的科学探索中的安全和道德责任。SafeScientist主动拒绝道德上不合适或高风险的任务，并在整个研究过程中严格强调安全。为了实现全面的安全监督，我们整合了多种防御机制，包括即时监控、代理协作监控、工具使用监控和道德审查员组件。作为SafeScientist的补充，我们提出了\textBF{SciSafetyBench}，这是一个专门用于评估科学背景下人工智能安全性的新型基准，包括6个领域的240项高风险科学任务，以及30个专门设计的科学工具和120个工具相关的风险任务。大量实验表明，与传统的人工智能科学家框架相比，SafeScientist将安全性能显着提高了35%，而不会影响科学输出质量。此外，我们还严格验证了我们的安全管道针对各种对抗攻击方法的稳健性，进一步证实了我们集成方法的有效性。代码和数据可在https://github.com/ulab-uiuc/SafeScientist上获取。\textcolor{red}{警告：本文包含可能令人反感或有害的示例数据。}



## **4. Hijacking Large Language Models via Adversarial In-Context Learning**

通过对抗性上下文学习劫持大型语言模型 cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2311.09948v3) [paper-pdf](http://arxiv.org/pdf/2311.09948v3)

**Authors**: Xiangyu Zhou, Yao Qiang, Saleh Zare Zade, Prashant Khanduri, Dongxiao Zhu

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific downstream tasks by utilizing labeled examples as demonstrations (demos) in the preconditioned prompts. Despite its promising performance, crafted adversarial attacks pose a notable threat to the robustness of LLMs. Existing attacks are either easy to detect, require a trigger in user input, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable prompt injection attack against ICL, aiming to hijack LLMs to generate the target output or elicit harmful responses. In our threat model, the hacker acts as a model publisher who leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demos via prompt injection. We also propose effective defense strategies using a few shots of clean demos, enhancing the robustness of LLMs during ICL. Extensive experimental results across various classification and jailbreak tasks demonstrate the effectiveness of the proposed attack and defense strategies. This work highlights the significant security vulnerabilities of LLMs during ICL and underscores the need for further in-depth studies.

摘要: 上下文学习（ICL）已成为一种强大的范式，通过利用带标签的示例作为预处理提示中的演示（演示），利用LLM来执行特定的下游任务。尽管性能令人鼓舞，但精心设计的对抗攻击对LLM的稳健性构成了显着的威胁。现有的攻击要么容易检测，需要用户输入触发，要么缺乏针对ICL的特异性。为了解决这些问题，这项工作引入了一种针对ICL的新型可转移即时注入攻击，旨在劫持LLM以生成目标输出或引发有害响应。在我们的威胁模型中，黑客充当模型发布者，利用基于梯度的提示搜索方法来学习难以察觉的对抗性后缀，并通过提示注入将其添加到上下文演示中。我们还使用几次干净的演示提出了有效的防御策略，增强ICL期间LLM的稳健性。各种分类和越狱任务的大量实验结果证明了所提出的攻击和防御策略的有效性。这项工作强调了ICL期间LLM的重大安全漏洞，并强调了进一步深入研究的必要性。



## **5. Learning to Poison Large Language Models for Downstream Manipulation**

学习毒害大型语言模型以进行下游操作 cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2402.13459v3) [paper-pdf](http://arxiv.org/pdf/2402.13459v3)

**Authors**: Xiangyu Zhou, Yao Qiang, Saleh Zare Zade, Mohammad Amin Roshani, Prashant Khanduri, Douglas Zytko, Dongxiao Zhu

**Abstract**: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where the adversary inserts backdoor triggers into training data to manipulate outputs. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the supervised fine-tuning (SFT) process. We propose a novel gradient-guided backdoor trigger learning (GBTL) algorithm to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various language model tasks, including sentiment analysis, domain generation, and question answering, our poisoning strategy demonstrates a high success rate in compromising various LLMs' outputs. We further propose two defense strategies against data poisoning attacks, including in-context learning (ICL) and continuous learning (CL), which effectively rectify the behavior of LLMs and significantly reduce the decline in performance. Our work highlights the significant security risks present during SFT of LLMs and the necessity of safeguarding LLMs against data poisoning attacks.

摘要: 大型语言模型（LLM）的出现标志着语言处理和推理能力取得了重大成就。尽管LLM取得了进步，但仍面临数据中毒攻击的漏洞，即对手将后门触发器插入训练数据中以操纵输出。这项工作通过设计专门针对利用监督式微调（SFT）过程而定制的新数据中毒攻击，进一步识别了LLM中的额外安全风险。我们提出了一种新型的梯度引导后门触发学习（GBTL）算法来有效识别对抗触发，确保逃避传统防御的检测，同时保持内容完整性。通过对各种语言模型任务（包括情感分析、领域生成和问题回答）的实验验证，我们的中毒策略证明了损害各种LLM输出的高成功率。我们进一步提出了两种针对数据中毒攻击的防御策略，包括上下文学习（ICL）和持续学习（CL），有效纠正LLM的行为，显着减少性能下降。我们的工作强调了LLM SFT期间存在的重大安全风险以及保护LLM免受数据中毒攻击的必要性。



## **6. Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models**

分而治之：击败多模式大型语言模型的混合策略 cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2412.16555v3) [paper-pdf](http://arxiv.org/pdf/2412.16555v3)

**Authors**: Yanxu Mao, Peipei Liu, Tiehan Cui, Zhaoteng Yan, Congying Liu, Datao You

**Abstract**: Large language models (LLMs) are widely applied in various fields of society due to their powerful reasoning, understanding, and generation capabilities. However, the security issues associated with these models are becoming increasingly severe. Jailbreaking attacks, as an important method for detecting vulnerabilities in LLMs, have been explored by researchers who attempt to induce these models to generate harmful content through various attack methods. Nevertheless, existing jailbreaking methods face numerous limitations, such as excessive query counts, limited coverage of jailbreak modalities, low attack success rates, and simplistic evaluation methods. To overcome these constraints, this paper proposes a multimodal jailbreaking method: JMLLM. This method integrates multiple strategies to perform comprehensive jailbreak attacks across text, visual, and auditory modalities. Additionally, we contribute a new and comprehensive dataset for multimodal jailbreaking research: TriJail, which includes jailbreak prompts for all three modalities. Experiments on the TriJail dataset and the benchmark dataset AdvBench, conducted on 13 popular LLMs, demonstrate advanced attack success rates and significant reduction in time overhead.

摘要: 大型语言模型（LLM）因其强大的推理、理解和生成能力而广泛应用于社会各个领域。然而，与这些模型相关的安全问题正变得日益严重。越狱攻击作为检测LLM漏洞的重要方法，已被研究人员探索，他们试图通过各种攻击方法诱导这些模型生成有害内容。然而，现有的越狱方法面临着许多局限性，例如过多的查询次数、越狱模式的覆盖范围有限、攻击成功率低以及评估方法简单化。为了克服这些限制，本文提出了一种多模式越狱方法：JMLLM。该方法集成了多种策略，以跨文本、视觉和听觉方式执行全面的越狱攻击。此外，我们还为多模式越狱研究提供了一个新的全面数据集：TriJail，其中包括所有三种模式的越狱提示。在TriJail数据集和基准数据集AdvBench上进行的实验在13个流行的LLM上进行，展示了先进的攻击成功率和显着减少的时间成本。



## **7. DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing**

德尔曼：通过模型编辑对大型语言模型越狱的动态防御 cs.CR

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.11647v2) [paper-pdf](http://arxiv.org/pdf/2502.11647v2)

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection.

摘要: 大型语言模型（LLM）广泛应用于决策制定，但其部署受到越狱攻击的威胁，即敌对用户操纵模型行为以绕过安全措施。现有的防御机制，例如安全微调和模型编辑，要么需要大量的参数修改，要么缺乏精确性，导致一般任务的性能下降，不适合部署后的安全对齐。为了应对这些挑战，我们提出了DELMAN（LLC动态编辑JAilbreak DefeNse），这是一种利用直接模型编辑来精确、动态地保护免受越狱攻击的新颖方法。德尔曼直接更新最少的相关参数集，以中和有害行为，同时保留模型的实用性。为了避免在良性上下文中触发安全响应，我们引入了KL分歧正规化，以确保更新后的模型在处理良性查询时与原始模型保持一致。实验结果表明，DELMAN在缓解越狱攻击的同时保持模型的实用性方面优于基线方法，并无缝适应新的攻击实例，为部署后模型保护提供了实用高效的解决方案。



## **8. Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models**

基于大型语言模型语义理解能力的自适应越狱策略 cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23404v1) [paper-pdf](http://arxiv.org/pdf/2505.23404v1)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin

**Abstract**: Adversarial attacks on Large Language Models (LLMs) via jailbreaking techniques-methods that circumvent their built-in safety and ethical constraints-have emerged as a critical challenge in AI security. These attacks compromise the reliability of LLMs by exploiting inherent weaknesses in their comprehension capabilities. This paper investigates the efficacy of jailbreaking strategies that are specifically adapted to the diverse levels of understanding exhibited by different LLMs. We propose the Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models, a novel framework that classifies LLMs into Type I and Type II categories according to their semantic comprehension abilities. For each category, we design tailored jailbreaking strategies aimed at leveraging their vulnerabilities to facilitate successful attacks. Extensive experiments conducted on multiple LLMs demonstrate that our adaptive strategy markedly improves the success rate of jailbreaking. Notably, our approach achieves an exceptional 98.9% success rate in jailbreaking GPT-4o(29 May 2025 release)

摘要: 通过越狱技术（规避其内置安全和道德约束的方法）对大型语言模型（LLM）进行的对抗攻击已成为人工智能安全领域的一个关键挑战。这些攻击通过利用LLM理解能力的固有弱点来损害LLM的可靠性。本文研究了专门适应不同法学硕士所表现出的不同理解水平的越狱策略的有效性。我们提出了基于大型语言模型语义理解能力的自适应越狱策略，这是一个新颖的框架，根据它们的语义理解能力将LLM分为类型I和类型II类别。对于每个类别，我们设计了量身定制的越狱策略，旨在利用其漏洞来促进成功的攻击。在多个LLM上进行的广泛实验表明，我们的自适应策略显着提高了越狱的成功率。值得注意的是，我们的方法在越狱GPT-4 o（2025年5月29日发布）中实现了98.9%的卓越成功率



## **9. Dataset Featurization: Uncovering Natural Language Features through Unsupervised Data Reconstruction**

数据集特征化：通过无监督数据重建揭示自然语言特征 cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.17541v2) [paper-pdf](http://arxiv.org/pdf/2502.17541v2)

**Authors**: Michal Bravansky, Vaclav Kubon, Suhas Hariharan, Robert Kirk

**Abstract**: Interpreting data is central to modern research. Large language models (LLMs) show promise in providing such natural language interpretations of data, yet simple feature extraction methods such as prompting often fail to produce accurate and versatile descriptions for diverse datasets and lack control over granularity and scale. To address these limitations, we propose a domain-agnostic method for dataset featurization that provides precise control over the number of features extracted while maintaining compact and descriptive representations comparable to human labeling. Our method optimizes the selection of informative binary features by evaluating the ability of an LLM to reconstruct the original data using those features. We demonstrate its effectiveness in dataset modeling tasks and through two case studies: (1) Constructing a feature representation of jailbreak tactics that compactly captures both the effectiveness and diversity of a larger set of human-crafted attacks; and (2) automating the discovery of features that align with human preferences, achieving accuracy and robustness comparable to human-crafted features. Moreover, we show that the pipeline scales effectively, improving as additional features are sampled, making it suitable for large and diverse datasets.

摘要: 解释数据是现代研究的核心。大型语言模型（LLM）在提供此类数据自然语言解释方面表现出希望，但提示等简单的特征提取方法往往无法为不同数据集生成准确且通用的描述，并且缺乏对粒度和规模的控制。为了解决这些限制，我们提出了一种用于数据集特征化的领域不可知方法，该方法可以精确控制提取的特征数量，同时保持与人类标记相当的紧凑和描述性表示。我们的方法通过评估LLM使用这些特征重建原始数据的能力来优化信息二进制特征的选择。我们通过两个案例研究证明了它在数据集建模任务中的有效性：（1）构建越狱策略的特征表示，该特征表示可以完整地捕获更大的人为攻击的有效性和多样性;（2）自动发现符合人类偏好的特征，实现与人为特征相媲美的准确性和鲁棒性。此外，我们还证明了流水线可以有效地扩展，随着额外特征的采样而改进，使其适用于大型和多样化的数据集。



## **10. Disrupting Vision-Language Model-Driven Navigation Services via Adversarial Object Fusion**

通过对抗对象融合扰乱视觉语言模型驱动的导航服务 cs.CR

Under review

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23266v1) [paper-pdf](http://arxiv.org/pdf/2505.23266v1)

**Authors**: Chunlong Xie, Jialing He, Shangwei Guo, Jiacheng Wang, Shudong Zhang, Tianwei Zhang, Tao Xiang

**Abstract**: We present Adversarial Object Fusion (AdvOF), a novel attack framework targeting vision-and-language navigation (VLN) agents in service-oriented environments by generating adversarial 3D objects. While foundational models like Large Language Models (LLMs) and Vision Language Models (VLMs) have enhanced service-oriented navigation systems through improved perception and decision-making, their integration introduces vulnerabilities in mission-critical service workflows. Existing adversarial attacks fail to address service computing contexts, where reliability and quality-of-service (QoS) are paramount. We utilize AdvOF to investigate and explore the impact of adversarial environments on the VLM-based perception module of VLN agents. In particular, AdvOF first precisely aggregates and aligns the victim object positions in both 2D and 3D space, defining and rendering adversarial objects. Then, we collaboratively optimize the adversarial object with regularization between the adversarial and victim object across physical properties and VLM perceptions. Through assigning importance weights to varying views, the optimization is processed stably and multi-viewedly by iterative fusions from local updates and justifications. Our extensive evaluations demonstrate AdvOF can effectively degrade agent performance under adversarial conditions while maintaining minimal interference with normal navigation tasks. This work advances the understanding of service security in VLM-powered navigation systems, providing computational foundations for robust service composition in physical-world deployments.

摘要: 我们提出了对抗性对象融合（AdvOF），这是一种新型攻击框架，通过生成对抗性3D对象，针对面向服务环境中的视觉和语言导航（VLN）代理。虽然大型语言模型（LLM）和视觉语言模型（VLM）等基础模型通过改进感知和决策增强了面向服务的导航系统，但它们的集成在关键任务服务工作流程中引入了漏洞。现有的对抗性攻击无法解决服务计算上下文，而服务计算上下文的可靠性和服务质量（Qos）至关重要。我们利用AdvOF来调查和探索对抗环境对VLN代理基于LM的感知模块的影响。特别是，AdvOF首先在2D和3D空间中精确地聚合和对齐受害对象的位置，定义和渲染对抗对象。然后，我们在物理属性和VLM感知之间通过对抗对象和受害者对象之间的正则化协作优化对抗对象。通过对不同视图赋予重要性权值，通过局部更新和调整的迭代融合，实现了稳定的多视图优化。我们的广泛评估表明，AdvOF可以有效地降低代理性能在对抗条件下，同时保持最小的干扰与正常的导航任务。这项工作促进了对基于VLM的导航系统中服务安全性的理解，为物理世界部署中的稳健服务组合提供了计算基础。



## **11. Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking**

推理防御：安全意识推理可以保护大型语言模型免受越狱 cs.CL

18 pages

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.12970v2) [paper-pdf](http://arxiv.org/pdf/2502.12970v2)

**Authors**: Junda Zhu, Lingyong Yan, Shuaiqiang Wang, Dawei Yin, Lei Sha

**Abstract**: Large Reasoning Models (LRMs) have demonstrated impressive performances across diverse domains. However, how safety of Large Language Models (LLMs) benefits from enhanced reasoning capabilities against jailbreak queries remains unexplored. To bridge this gap, in this paper, we propose Reasoning-to-Defend (R2D), a novel training paradigm that integrates a safety-aware reasoning mechanism into LLMs' generation. This enables self-evaluation at each step of the reasoning process, forming safety pivot tokens as indicators of the safety status of responses. Furthermore, in order to improve the accuracy of predicting pivot tokens, we propose Contrastive Pivot Optimization (CPO), which enhances the model's perception of the safety status of given dialogues. LLMs dynamically adjust their response strategies during reasoning, significantly enhancing their safety capabilities defending jailbreak attacks. Extensive experiments demonstrate that R2D effectively mitigates various attacks and improves overall safety, while maintaining the original performances. This highlights the substantial potential of safety-aware reasoning in improving robustness of LRMs and LLMs against various jailbreaks.

摘要: 大型推理模型（LRM）在不同领域表现出令人印象深刻的性能。然而，大型语言模型（LLM）的安全性如何从针对越狱查询的增强推理能力中受益仍有待探索。为了弥合这一差距，在本文中，我们提出了推理防御（R2 D），这是一种新型训练范式，将安全感知推理机制集成到LLM的生成中。这使得推理过程的每个步骤都能够进行自我评估，形成安全支点令牌作为响应安全状态的指标。此外，为了提高预测枢轴令牌的准确性，我们提出了对比枢轴优化（CPO），这增强了模型对给定对话的安全状态的感知。LLM在推理过程中动态调整其响应策略，显著增强其防御越狱攻击的安全能力。大量的实验表明，R2D有效地减轻了各种攻击，提高了整体安全性，同时保持了原有的性能。这突出了安全意识推理在提高LRM和LLM对各种越狱的鲁棒性方面的巨大潜力。



## **12. Jailbreaking to Jailbreak**

越狱到越狱 cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.09638v2) [paper-pdf](http://arxiv.org/pdf/2502.09638v2)

**Authors**: Jeremy Kritz, Vaughn Robinson, Robert Vacareanu, Bijan Varjavand, Michael Choi, Bobby Gogov, Scale Red Team, Summer Yue, Willow E. Primack, Zifan Wang

**Abstract**: Large Language Models (LLMs) can be used to red team other models (e.g. jailbreaking) to elicit harmful contents. While prior works commonly employ open-weight models or private uncensored models for doing jailbreaking, as the refusal-training of strong LLMs (e.g. OpenAI o3) refuse to help jailbreaking, our work turn (almost) any black-box LLMs into attackers. The resulting $J_2$ (jailbreaking-to-jailbreak) attackers can effectively jailbreak the safeguard of target models using various strategies, both created by themselves or from expert human red teamers. In doing so, we show their strong but under-researched jailbreaking capabilities. Our experiments demonstrate that 1) prompts used to create $J_2$ attackers transfer across almost all black-box models; 2) an $J_2$ attacker can jailbreak a copy of itself, and this vulnerability develops rapidly over the past 12 months; 3) reasong models, such as Sonnet-3.7, are strong $J_2$ attackers compared to others. For example, when used against the safeguard of GPT-4o, $J_2$ (Sonnet-3.7) achieves 0.975 attack success rate (ASR), which matches expert human red teamers and surpasses the state-of-the-art algorithm-based attacks. Among $J_2$ attackers, $J_2$ (o3) achieves highest ASR (0.605) against Sonnet-3.5, one of the most robust models.

摘要: 大型语言模型（LLM）可用于与其他模型（例如越狱）进行团队合作，以引出有害内容。虽然之前的作品通常使用开放权重模型或私人未经审查的模型来进行越狱，但由于强大的LLM（例如OpenAI o3）的重新训练拒绝帮助越狱，我们的工作将（几乎）任何黑匣子LLM变成攻击者。由此产生的$J_2$（越狱到越狱）攻击者可以使用各种策略有效地越狱目标模型的保护，无论是由他们自己创建的还是由专家人类红色团队创建的。通过这样做，我们展示了他们强大但研究不足的越狱能力。我们的实验表明：1）用于创建$J_2$攻击者在几乎所有黑匣子模型中传输的提示; 2）$J_2$攻击者可以越狱自己的副本，并且该漏洞在过去12个月内迅速发展; 3）Reason模型，例如Sonnet-3.7，与其他模型相比，是强大的$J_2$攻击者。例如，当针对GPT-4 o的保护使用时，$J_2$（Sonnet-3.7）可实现0.975的攻击成功率（ASB），这与专家人类红队队员相匹配，并超越了最先进的基于算法的攻击。在$J_2$攻击者中，$J_2$（o3）针对Sonnet-3.5（最强大的模型之一）实现了最高的ASB（0.605）。



## **13. DyePack: Provably Flagging Test Set Contamination in LLMs Using Backdoors**

DyePack：使用后门可证明标记LLM中的测试集污染 cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23001v1) [paper-pdf](http://arxiv.org/pdf/2505.23001v1)

**Authors**: Yize Cheng, Wenxiao Wang, Mazda Moayeri, Soheil Feizi

**Abstract**: Open benchmarks are essential for evaluating and advancing large language models, offering reproducibility and transparency. However, their accessibility makes them likely targets of test set contamination. In this work, we introduce DyePack, a framework that leverages backdoor attacks to identify models that used benchmark test sets during training, without requiring access to the loss, logits, or any internal details of the model. Like how banks mix dye packs with their money to mark robbers, DyePack mixes backdoor samples with the test data to flag models that trained on it. We propose a principled design incorporating multiple backdoors with stochastic targets, enabling exact false positive rate (FPR) computation when flagging every model. This provably prevents false accusations while providing strong evidence for every detected case of contamination. We evaluate DyePack on five models across three datasets, covering both multiple-choice and open-ended generation tasks. For multiple-choice questions, it successfully detects all contaminated models with guaranteed FPRs as low as 0.000073% on MMLU-Pro and 0.000017% on Big-Bench-Hard using eight backdoors. For open-ended generation tasks, it generalizes well and identifies all contaminated models on Alpaca with a guaranteed false positive rate of just 0.127% using six backdoors.

摘要: 开放基准对于评估和推进大型语言模型、提供可重复性和透明度至关重要。然而，它们的可及性使它们可能成为测试集污染的目标。在这项工作中，我们引入了DyePack，这是一个利用后门攻击来识别在训练期间使用基准测试集的模型的框架，而不需要访问模型的损失、日志或任何内部细节。就像银行将染料包与钱混合来标记劫匪一样，DyePack将后门样本与测试数据混合起来，以标记对其进行训练的模型。我们提出了一种原则性设计，将多个后门与随机目标结合在一起，在标记每个模型时实现精确的假阳性率（FPR）计算。事实证明，这可以防止虚假指控，同时为每一个检测到的污染案例提供强有力的证据。我们在三个数据集的五个模型上评估了DyePack，涵盖多项选择和开放式生成任务。对于多项选择题，它使用八个后门成功检测到所有受污染的型号，保证FPR在MMLU-Pro上低至0.000073%，在Big-Bench-Hard上低至0.00017%。对于开放式生成任务，它可以很好地推广，并使用六个后门识别羊驼上所有受污染的模型，保证假阳性率仅为0.127%。



## **14. Revisiting Multi-Agent Debate as Test-Time Scaling: A Systematic Study of Conditional Effectiveness**

重新审视多主体辩论作为测试时间缩放：条件有效性的系统性研究 cs.AI

Preprint, under review

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.22960v1) [paper-pdf](http://arxiv.org/pdf/2505.22960v1)

**Authors**: Yongjin Yang, Euiin Yi, Jongwoo Ko, Kimin Lee, Zhijing Jin, Se-Young Yun

**Abstract**: The remarkable growth in large language model (LLM) capabilities has spurred exploration into multi-agent systems, with debate frameworks emerging as a promising avenue for enhanced problem-solving. These multi-agent debate (MAD) approaches, where agents collaboratively present, critique, and refine arguments, potentially offer improved reasoning, robustness, and diverse perspectives over monolithic models. Despite prior studies leveraging MAD, a systematic understanding of its effectiveness compared to self-agent methods, particularly under varying conditions, remains elusive. This paper seeks to fill this gap by conceptualizing MAD as a test-time computational scaling technique, distinguished by collaborative refinement and diverse exploration capabilities. We conduct a comprehensive empirical investigation comparing MAD with strong self-agent test-time scaling baselines on mathematical reasoning and safety-related tasks. Our study systematically examines the influence of task difficulty, model scale, and agent diversity on MAD's performance. Key findings reveal that, for mathematical reasoning, MAD offers limited advantages over self-agent scaling but becomes more effective with increased problem difficulty and decreased model capability, while agent diversity shows little benefit. Conversely, for safety tasks, MAD's collaborative refinement can increase vulnerability, but incorporating diverse agent configurations facilitates a gradual reduction in attack success through the collaborative refinement process. We believe our findings provide critical guidance for the future development of more effective and strategically deployed MAD systems.

摘要: 大型语言模型（LLM）能力的显着增长刺激了对多代理系统的探索，辩论框架成为增强问题解决的有希望的途径。这些多主体辩论（MAD）方法，主体协作地呈现、批评和完善论点，与单一模型相比，有可能提供更好的推理、稳健性和多样化的观点。尽管之前的研究利用了MAD，但系统地了解其与自代理方法相比的有效性，特别是在不同条件下，仍然难以捉摸。本文试图通过将MAD概念化为一种测试时计算缩放技术来填补这一空白，该技术的特点是协作细化和多样化的探索能力。我们进行了一项全面的实证研究，将MAD与数学推理和安全相关任务的强大自代理测试时间缩放基线进行了比较。我们的研究系统地考察了任务难度、模型规模和代理多样性对MAD绩效的影响。主要研究结果表明，对于数学推理，MAD比自代理扩展提供的优势有限，但随着问题难度的增加和模型能力的降低而变得更加有效，而代理多样性几乎没有表现出什么好处。相反，对于安全任务，MAD的协作细化可能会增加脆弱性，但结合不同的代理配置有助于通过协作细化过程逐渐降低攻击成功率。我们相信，我们的研究结果为未来开发更有效和战略部署的MAD系统提供了重要指导。



## **15. Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates**

LLM可以欺骗CLIP吗？通过文本更新对预训练多模式表示的对抗性组合进行基准测试 cs.CL

ACL 2025 Main. Code is released at  https://vision.snu.ac.kr/projects/mac

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22943v1) [paper-pdf](http://arxiv.org/pdf/2505.22943v1)

**Authors**: Jaewoo Ahn, Heeseung Yun, Dayoon Ko, Gunhee Kim

**Abstract**: While pre-trained multimodal representations (e.g., CLIP) have shown impressive capabilities, they exhibit significant compositional vulnerabilities leading to counterintuitive judgments. We introduce Multimodal Adversarial Compositionality (MAC), a benchmark that leverages large language models (LLMs) to generate deceptive text samples to exploit these vulnerabilities across different modalities and evaluates them through both sample-wise attack success rate and group-wise entropy-based diversity. To improve zero-shot methods, we propose a self-training approach that leverages rejection-sampling fine-tuning with diversity-promoting filtering, which enhances both attack success rate and sample diversity. Using smaller language models like Llama-3.1-8B, our approach demonstrates superior performance in revealing compositional vulnerabilities across various multimodal representations, including images, videos, and audios.

摘要: 虽然预训练的多模式表示（例如，CLIP）表现出令人印象深刻的能力，它们表现出显着的合成漏洞，导致反直觉的判断。我们引入了多模式对抗组合（MAC），这是一个基准，利用大型语言模型（LLM）来生成欺骗性文本样本，以利用不同模式中的这些漏洞，并通过样本攻击成功率和基于分组的基于信息的多样性来评估它们。为了改进零射击方法，我们提出了一种自训练方法，该方法利用拒绝采样微调和促进多样性的过滤，从而增强了攻击成功率和样本多样性。使用Llama-3.1-8B等较小的语言模型，我们的方法在揭示各种多模式表示（包括图像、视频和音频）的合成漏洞方面表现出卓越的性能。



## **16. Operationalizing CaMeL: Strengthening LLM Defenses for Enterprise Deployment**

运营CaMeL：加强LLM企业部署的防御 cs.CR

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22852v1) [paper-pdf](http://arxiv.org/pdf/2505.22852v1)

**Authors**: Krti Tallam, Emma Miller

**Abstract**: CaMeL (Capabilities for Machine Learning) introduces a capability-based sandbox to mitigate prompt injection attacks in large language model (LLM) agents. While effective, CaMeL assumes a trusted user prompt, omits side-channel concerns, and incurs performance tradeoffs due to its dual-LLM design. This response identifies these issues and proposes engineering improvements to expand CaMeL's threat coverage and operational usability. We introduce: (1) prompt screening for initial inputs, (2) output auditing to detect instruction leakage, (3) a tiered-risk access model to balance usability and control, and (4) a verified intermediate language for formal guarantees. Together, these upgrades align CaMeL with best practices in enterprise security and support scalable deployment.

摘要: CaMeL（机器学习能力）引入了基于能力的沙箱，以减轻大型语言模型（LLM）代理中的即时注入攻击。虽然有效，但CaMeL假设了值得信赖的用户提示，省略了侧渠道问题，并因其双LLM设计而导致性能权衡。此回复指出了这些问题，并提出了工程改进，以扩大CaMeL的威胁覆盖范围和操作可用性。我们介绍：（1）及时筛选初始输入，（2）输出审计以检测指令泄漏，（3）平衡可用性和控制的分层风险访问模型，以及（4）用于形式保证的经过验证的中间语言。这些升级使CaMeL与企业安全领域的最佳实践保持一致，并支持可扩展部署。



## **17. The Aloe Family Recipe for Open and Specialized Healthcare LLMs**

面向开放和专业医疗LL的Aloe家族食谱 cs.CL

Follow-up work from arXiv:2405.01886

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.04388v2) [paper-pdf](http://arxiv.org/pdf/2505.04388v2)

**Authors**: Dario Garcia-Gasulla, Jordi Bayarri-Planas, Ashwin Kumar Gururajan, Enrique Lopez-Cuena, Adrian Tormos, Daniel Hinjos, Pablo Bernabeu-Perez, Anna Arias-Duart, Pablo Agustin Martin-Torres, Marta Gonzalez-Mallo, Sergio Alvarez-Napagao, Eduard Ayguadé-Parra, Ulises Cortés

**Abstract**: Purpose: With advancements in Large Language Models (LLMs) for healthcare, the need arises for competitive open-source models to protect the public interest. This work contributes to the field of open medical LLMs by optimizing key stages of data preprocessing and training, while showing how to improve model safety (through DPO) and efficacy (through RAG). The evaluation methodology used, which includes four different types of tests, defines a new standard for the field. The resultant models, shown to be competitive with the best private alternatives, are released with a permisive license.   Methods: Building on top of strong base models like Llama 3.1 and Qwen 2.5, Aloe Beta uses a custom dataset to enhance public data with synthetic Chain of Thought examples. The models undergo alignment with Direct Preference Optimization, emphasizing ethical and policy-aligned performance in the presence of jailbreaking attacks. Evaluation includes close-ended, open-ended, safety and human assessments, to maximize the reliability of results.   Results: Recommendations are made across the entire pipeline, backed by the solid performance of the Aloe Family. These models deliver competitive performance across healthcare benchmarks and medical fields, and are often preferred by healthcare professionals. On bias and toxicity, the Aloe Beta models significantly improve safety, showing resilience to unseen jailbreaking attacks. For a responsible release, a detailed risk assessment specific to healthcare is attached to the Aloe Family models.   Conclusion: The Aloe Beta models, and the recipe that leads to them, are a significant contribution to the open-source medical LLM field, offering top-of-the-line performance while maintaining high ethical requirements. This work sets a new standard for developing and reporting aligned LLMs in healthcare.

摘要: 目的：随着医疗保健大型语言模型（LLM）的进步，需要有竞争力的开源模型来保护公共利益。这项工作通过优化数据预处理和训练的关键阶段，同时展示如何提高模型安全性（通过DPO）和有效性（通过RAG），为开放医学LLM领域做出了贡献。所使用的评估方法包括四种不同类型的测试，为该领域定义了新的标准。由此产生的模型被证明与最好的私人替代品具有竞争力，并且在许可证下发布。   方法：Aloe Beta建立在Llama 3.1和Qwen 2.5等强大基础模型的基础上，使用自定义数据集通过合成的思想链示例增强公共数据。这些模型与直接偏好优化保持一致，强调在存在越狱攻击时的道德和政策一致的性能。评估包括封闭式、开放式、安全性和人为评估，以最大限度地提高结果的可靠性。   结果：在Aloe系列的稳健表现的支持下，整个管道都提出了建议。这些模型在医疗保健基准和医疗领域提供有竞争力的性能，并且通常受到医疗保健专业人士的青睐。在偏见和毒性方面，Aloe Beta模型显着提高了安全性，表现出对不可见越狱攻击的韧性。为了实现负责任的发布，Aloe Family模型附带了针对医疗保健的详细风险评估。   结论：Aloe Beta模型及其配方是对开源医学LLM领域的重大贡献，在保持高道德要求的同时提供顶级性能。这项工作为开发和报告医疗保健领域一致的LLM设定了新标准。



## **18. SequentialBreak: Large Language Models Can be Fooled by Embedding Jailbreak Prompts into Sequential Prompt Chains**

SequentialBreak：大型语言模型可以通过将越狱提示嵌入序列提示链来愚弄 cs.CR

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2411.06426v3) [paper-pdf](http://arxiv.org/pdf/2411.06426v3)

**Authors**: Bijoy Ahmed Saiem, MD Sadik Hossain Shanto, Rakib Ahsan, Md Rafi ur Rashid

**Abstract**: As the integration of the Large Language Models (LLMs) into various applications increases, so does their susceptibility to misuse, raising significant security concerns. Numerous jailbreak attacks have been proposed to assess the security defense of LLMs. Current jailbreak attacks mainly rely on scenario camouflage, prompt obfuscation, prompt optimization, and prompt iterative optimization to conceal malicious prompts. In particular, sequential prompt chains in a single query can lead LLMs to focus on certain prompts while ignoring others, facilitating context manipulation. This paper introduces SequentialBreak, a novel jailbreak attack that exploits this vulnerability. We discuss several scenarios, not limited to examples like Question Bank, Dialog Completion, and Game Environment, where the harmful prompt is embedded within benign ones that can fool LLMs into generating harmful responses. The distinct narrative structures of these scenarios show that SequentialBreak is flexible enough to adapt to various prompt formats beyond those discussed. Extensive experiments demonstrate that SequentialBreak uses only a single query to achieve a substantial gain of attack success rate over existing baselines against both open-source and closed-source models. Through our research, we highlight the urgent need for more robust and resilient safeguards to enhance LLM security and prevent potential misuse. All the result files and website associated with this research are available in this GitHub repository: https://anonymous.4open.science/r/JailBreakAttack-4F3B/.

摘要: 随着大型语言模型（LLM）与各种应用程序的集成不断增加，它们被滥用的可能性也在增加，从而引发了严重的安全问题。人们提出了许多越狱攻击来评估LLM的安全防御。当前的越狱攻击主要依靠场景伪装、提示混淆、提示优化、提示迭代优化来隐藏恶意提示。特别是，单个查询中的顺序提示链可能会导致LLM专注于某些提示而忽略其他提示，从而促进上下文操纵。本文介绍SequentialBreak，这是一种利用此漏洞的新型越狱攻击。我们讨论了几种场景，不仅限于问题库、对话框完成和游戏环境等示例，其中有害提示嵌入良性提示中，从而可以欺骗LLM生成有害响应。这些场景的独特叙事结构表明SequentialBreak足够灵活，可以适应讨论之外的各种提示格式。大量实验表明，SequentialBreak仅使用单个查询就可以在针对开源和闭源模型的现有基线上实现大幅提高攻击成功率。通过我们的研究，我们强调迫切需要更强大、更有弹性的保障措施，以增强LLM安全性并防止潜在的滥用。与本研究相关的所有结果文件和网站均可在GitHub存储库中获取：https://anonymous.4open.science/r/JailBreakAttack-4F3B/。



## **19. VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models**

VisURA：一种针对越狱多模式大型语言模型的视觉链推理攻击 cs.CV

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.19684v2) [paper-pdf](http://arxiv.org/pdf/2505.19684v2)

**Authors**: Bingrui Sima, Linhua Cong, Wenxuan Wang, Kun He

**Abstract**: The emergence of Multimodal Large Language Models (MLRMs) has enabled sophisticated visual reasoning capabilities by integrating reinforcement learning and Chain-of-Thought (CoT) supervision. However, while these enhanced reasoning capabilities improve performance, they also introduce new and underexplored safety risks. In this work, we systematically investigate the security implications of advanced visual reasoning in MLRMs. Our analysis reveals a fundamental trade-off: as visual reasoning improves, models become more vulnerable to jailbreak attacks. Motivated by this critical finding, we introduce VisCRA (Visual Chain Reasoning Attack), a novel jailbreak framework that exploits the visual reasoning chains to bypass safety mechanisms. VisCRA combines targeted visual attention masking with a two-stage reasoning induction strategy to precisely control harmful outputs. Extensive experiments demonstrate VisCRA's significant effectiveness, achieving high attack success rates on leading closed-source MLRMs: 76.48% on Gemini 2.0 Flash Thinking, 68.56% on QvQ-Max, and 56.60% on GPT-4o. Our findings highlight a critical insight: the very capability that empowers MLRMs -- their visual reasoning -- can also serve as an attack vector, posing significant security risks.

摘要: 多模式大型语言模型（MLRM）的出现通过集成强化学习和思想链（CoT）监督，实现了复杂的视觉推理能力。然而，虽然这些增强的推理能力可以提高性能，但它们也引入了新的且未充分研究的安全风险。在这项工作中，我们系统地研究了MLRM中高级视觉推理的安全影响。我们的分析揭示了一个基本的权衡：随着视觉推理的改进，模型变得更容易受到越狱攻击。出于这一关键发现的动机，我们引入了VisCRA（视觉链推理攻击），这是一种新颖的越狱框架，它利用视觉推理链绕过安全机制。VisCRA将有针对性的视觉注意力掩蔽与两阶段推理归纳策略相结合，以精确控制有害输出。大量的实验证明了VisCRA的显著有效性，在领先的闭源MLRM上实现了高攻击成功率：Gemini 2.0 Flash Thinking上为76.48%，QvQ-Max上为68.56%，GPT-4 o上为56.60%。我们的研究结果强调了一个关键的见解：赋予MLRM权力的能力（它们的视觉推理）本身也可以作为攻击载体，构成重大的安全风险。



## **20. Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space**

打破天花板：通过扩大战略空间探索越狱袭击的潜力 cs.CR

19 pages, 20 figures, accepted by ACL 2025, Findings

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.21277v2) [paper-pdf](http://arxiv.org/pdf/2505.21277v2)

**Authors**: Yao Huang, Yitong Sun, Shouwei Ruan, Yichi Zhang, Yinpeng Dong, Xingxing Wei

**Abstract**: Large Language Models (LLMs), despite advanced general capabilities, still suffer from numerous safety risks, especially jailbreak attacks that bypass safety protocols. Understanding these vulnerabilities through black-box jailbreak attacks, which better reflect real-world scenarios, offers critical insights into model robustness. While existing methods have shown improvements through various prompt engineering techniques, their success remains limited against safety-aligned models, overlooking a more fundamental problem: the effectiveness is inherently bounded by the predefined strategy spaces. However, expanding this space presents significant challenges in both systematically capturing essential attack patterns and efficiently navigating the increased complexity. To better explore the potential of expanding the strategy space, we address these challenges through a novel framework that decomposes jailbreak strategies into essential components based on the Elaboration Likelihood Model (ELM) theory and develops genetic-based optimization with intention evaluation mechanisms. To be striking, our experiments reveal unprecedented jailbreak capabilities by expanding the strategy space: we achieve over 90% success rate on Claude-3.5 where prior methods completely fail, while demonstrating strong cross-model transferability and surpassing specialized safeguard models in evaluation accuracy. The code is open-sourced at: https://github.com/Aries-iai/CL-GSO.

摘要: 尽管大型语言模型（LLM）具有先进的通用功能，但仍然面临许多安全风险，尤其是绕过安全协议的越狱攻击。通过黑匣子越狱攻击来了解这些漏洞（更好地反映了现实世界的场景），可以为模型稳健性提供重要见解。虽然现有方法通过各种即时工程技术表现出了改进，但它们的成功仍然局限于安全性一致的模型，忽视了一个更根本的问题：有效性本质上受到预定义的策略空间的限制。然而，扩展这一空间在系统性捕获基本攻击模式和有效应对日益增加的复杂性方面带来了重大挑战。为了更好地探索扩大策略空间的潜力，我们通过一个新颖的框架来应对这些挑战，该框架基于埃斯珀似然模型（ELM）理论将越狱策略分解为基本组件，并开发具有意图评估机制的基于遗传的优化。引人注目的是，我们的实验通过扩大策略空间揭示了前所未有的越狱能力：在现有方法完全失败的情况下，我们在Claude-3.5上实现了90%以上的成功率，同时展示了强大的跨模型可移植性，并在评估准确性方面超越了专业保障模型。该代码的开源网址：https://github.com/Aries-iai/CL-GSO。



## **21. Adaptive Detoxification: Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing**

适应性去规范化：通过有毒意识知识编辑保护LLM的通用能力 cs.CL

ACL 2025 Findings

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22298v1) [paper-pdf](http://arxiv.org/pdf/2505.22298v1)

**Authors**: Yifan Lu, Jing Li, Yigeng Zhou, Yihui Zhang, Wenya Wang, Xiucheng Li, Meishan Zhang, Fangming Liu, Jun Yu, Min Zhang

**Abstract**: Large language models (LLMs) exhibit impressive language capabilities but remain vulnerable to malicious prompts and jailbreaking attacks. Existing knowledge editing methods for LLM detoxification face two major challenges. First, they often rely on entity-specific localization, making them ineffective against adversarial inputs without explicit entities. Second, these methods suffer from over-editing, where detoxified models reject legitimate queries, compromising overall performance. In this paper, we propose ToxEdit, a toxicity-aware knowledge editing approach that dynamically detects toxic activation patterns during forward propagation. It then routes computations through adaptive inter-layer pathways to mitigate toxicity effectively. This design ensures precise toxicity mitigation while preserving LLMs' general capabilities. To more accurately assess over-editing, we also enhance the SafeEdit benchmark by incorporating instruction-following evaluation tasks. Experimental results on multiple LLMs demonstrate that our ToxEdit outperforms previous state-of-the-art methods in both detoxification performance and safeguarding general capabilities of LLMs.

摘要: 大型语言模型（LLM）展现出令人印象深刻的语言能力，但仍然容易受到恶意提示和越狱攻击的影响。LLM解毒的现有知识编辑方法面临两大挑战。首先，它们通常依赖于实体特定的本地化，这使得它们在没有明确实体的情况下对对抗性输入无效。其次，这些方法存在过度编辑的问题，其中解毒的模型拒绝合法查询，从而损害整体性能。在本文中，我们提出了ToxEdit，这是一种具有毒性的知识编辑方法，可以在前向传播期间动态检测有毒激活模式。然后，它通过自适应的层间路径路由计算，以有效地减轻毒性。该设计确保了精确的毒性缓解，同时保留了LLM的一般能力。为了更准确地评估过度编辑，我们还通过整合描述跟踪评估任务来增强SafeEdit基准。多个LLM的实验结果表明，我们的ToxEdit在解毒性能和保护LLM的一般能力方面都优于之前的最先进方法。



## **22. Test-Time Immunization: A Universal Defense Framework Against Jailbreaks for (Multimodal) Large Language Models**

测试时免疫：针对（多模式）大型语言模型越狱的通用防御框架 cs.CR

Under Review

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22271v1) [paper-pdf](http://arxiv.org/pdf/2505.22271v1)

**Authors**: Yongcan Yu, Yanbo Wang, Ran He, Jian Liang

**Abstract**: While (multimodal) large language models (LLMs) have attracted widespread attention due to their exceptional capabilities, they remain vulnerable to jailbreak attacks. Various defense methods are proposed to defend against jailbreak attacks, however, they are often tailored to specific types of jailbreak attacks, limiting their effectiveness against diverse adversarial strategies. For instance, rephrasing-based defenses are effective against text adversarial jailbreaks but fail to counteract image-based attacks. To overcome these limitations, we propose a universal defense framework, termed Test-time IMmunization (TIM), which can adaptively defend against various jailbreak attacks in a self-evolving way. Specifically, TIM initially trains a gist token for efficient detection, which it subsequently applies to detect jailbreak activities during inference. When jailbreak attempts are identified, TIM implements safety fine-tuning using the detected jailbreak instructions paired with refusal answers. Furthermore, to mitigate potential performance degradation in the detector caused by parameter updates during safety fine-tuning, we decouple the fine-tuning process from the detection module. Extensive experiments on both LLMs and multimodal LLMs demonstrate the efficacy of TIM.

摘要: 虽然（多模式）大型语言模型（LLM）因其卓越的功能而引起了广泛关注，但它们仍然容易受到越狱攻击。人们提出了各种防御方法来防御越狱攻击，然而，它们通常针对特定类型的越狱攻击进行定制，从而限制了它们针对不同对抗策略的有效性。例如，基于改写的防御对于文本对抗越狱有效，但无法抵消基于图像的攻击。为了克服这些限制，我们提出了一种通用防御框架，称为测试时免疫（TIM），它可以以自我进化的方式自适应地防御各种越狱攻击。具体来说，TIM最初训练一个要点令牌以进行高效检测，随后应用于在推理期间检测越狱活动。当识别出越狱尝试时，TIM会使用检测到的越狱指令与拒绝答案配对实施安全微调。此外，为了减轻安全微调期间参数更新导致的检测器潜在性能下降，我们将微调过程与检测模块脱钩。对LLM和多模式LLM的广泛实验证明了TIM的功效。



## **23. Safeguarding Privacy of Retrieval Data against Membership Inference Attacks: Is This Query Too Close to Home?**

保护检索数据的隐私免受成员推断攻击：此查询是否离家太近？ cs.CL

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22061v1) [paper-pdf](http://arxiv.org/pdf/2505.22061v1)

**Authors**: Yujin Choi, Youngjoo Park, Junyoung Byun, Jaewook Lee, Jinseong Park

**Abstract**: Retrieval-augmented generation (RAG) mitigates the hallucination problem in large language models (LLMs) and has proven effective for specific, personalized applications. However, passing private retrieved documents directly to LLMs introduces vulnerability to membership inference attacks (MIAs), which try to determine whether the target datum exists in the private external database or not. Based on the insight that MIA queries typically exhibit high similarity to only one target document, we introduce Mirabel, a similarity-based MIA detection framework designed for the RAG system. With the proposed Mirabel, we show that simple detect-and-hide strategies can successfully obfuscate attackers, maintain data utility, and remain system-agnostic. We experimentally prove its detection and defense against various state-of-the-art MIA methods and its adaptability to existing private RAG systems.

摘要: 检索增强生成（RAG）缓解了大型语言模型（LLM）中的幻觉问题，并已被证明对特定的个性化应用程序有效。然而，将私人检索到的文档直接传递给LLM会引入成员资格推断攻击（MIA）的漏洞，该攻击试图确定目标数据是否存在于私人外部数据库中。基于MIA查询通常仅与一个目标文档表现出高度相似性的认识，我们引入了Mirabel，这是一种为RAG系统设计的基于相似性的MIA检测框架。通过提出的Mirabel，我们表明简单的检测和隐藏策略可以成功地混淆攻击者、保持数据实用性并保持系统不可知。我们通过实验证明了它对各种最先进的MIA方法的检测和防御，以及它对现有私人RAG系统的适应性。



## **24. Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective**

从心理测量学角度通过攻击来评估大型语言模型中的内隐偏差 cs.CL

Accepted to ACL 2025 Findings

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2406.14023v3) [paper-pdf](http://arxiv.org/pdf/2406.14023v3)

**Authors**: Yuchen Wen, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: As large language models (LLMs) become an important way of information access, there have been increasing concerns that LLMs may intensify the spread of unethical content, including implicit bias that hurts certain populations without explicit harmful words. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain demographics by attacking them from a psychometric perspective to elicit agreements to biased viewpoints. Inspired by psychometric principles in cognitive and social psychology, we propose three attack approaches, i.e., Disguise, Deception, and Teaching. Incorporating the corresponding attack instructions, we built two benchmarks: (1) a bilingual dataset with biased statements covering four bias types (2.7K instances) for extensive comparative analysis, and (2) BUMBLE, a larger benchmark spanning nine common bias types (12.7K instances) for comprehensive evaluation. Extensive evaluation of popular commercial and open-source LLMs shows that our methods can elicit LLMs' inner bias more effectively than competitive baselines. Our attack methodology and benchmarks offer an effective means of assessing the ethical risks of LLMs, driving progress toward greater accountability in their development. Our code, data and benchmarks are available at https://github.com/yuchenwen1/ImplicitBiasPsychometricEvaluation and https://github.com/yuchenwen1/BUMBLE.

摘要: 随着大型语言模型（LLM）成为信息获取的重要方式，人们越来越担心LLM可能会加剧不道德内容的传播，包括在没有明确有害词语的情况下伤害某些人群的隐性偏见。在本文中，我们通过从心理测量学的角度攻击LLM对某些人口统计数据的隐性偏见进行了严格评估，以获取对偏见观点的同意。受认知和社会心理学中心理测量原则的启发，我们提出了三种攻击方法，即伪装、欺骗和教学。综合相应的攻击指令，我们构建了两个基准：（1）双语数据集，其中包含涵盖四种偏见类型（2.7 K实例）的偏见陈述，用于广泛的比较分析，和（2）BUMBLE，一个跨越九种常见偏见类型（12.7 K实例）的更大基准，用于全面评估。对流行的商业和开源LLM的广泛评估表明，我们的方法比竞争基线更有效地引发LLM的内部偏见。我们的攻击方法和基准提供了评估LLM道德风险的有效手段，推动LLM在开发过程中实现更强的问责制。我们的代码、数据和基准可在https://github.com/yuchenwen1/ImplicitBiasPsychometricEvaluation和https://github.com/yuchenwen1/BUMBLE上获取。



## **25. Wolf Hidden in Sheep's Conversations: Toward Harmless Data-Based Backdoor Attacks for Jailbreaking Large Language Models**

绵羊对话中隐藏的狼：针对越狱大型语言模型的基于数据的无害后门攻击 cs.CL

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.17601v2) [paper-pdf](http://arxiv.org/pdf/2505.17601v2)

**Authors**: Jiawei Kong, Hao Fang, Xiaochen Yang, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang

**Abstract**: Supervised fine-tuning (SFT) aligns large language models (LLMs) with human intent by training them on labeled task-specific data. Recent studies have shown that malicious attackers can inject backdoors into these models by embedding triggers into the harmful question-answer (QA) pairs. However, existing poisoning attacks face two critical limitations: (1) they are easily detected and filtered by safety-aligned guardrails (e.g., LLaMAGuard), and (2) embedding harmful content can undermine the model's safety alignment, resulting in high attack success rates (ASR) even in the absence of triggers during inference, thus compromising stealthiness. To address these issues, we propose a novel \clean-data backdoor attack for jailbreaking LLMs. Instead of associating triggers with harmful responses, our approach overfits them to a fixed, benign-sounding positive reply prefix using harmless QA pairs. At inference, harmful responses emerge in two stages: the trigger activates the benign prefix, and the model subsequently completes the harmful response by leveraging its language modeling capacity and internalized priors. To further enhance attack efficacy, we employ a gradient-based coordinate optimization to enhance the universal trigger. Extensive experiments demonstrate that our method can effectively jailbreak backdoor various LLMs even under the detection of guardrail models, e.g., an ASR of 86.67% and 85% on LLaMA-3-8B and Qwen-2.5-7B judged by GPT-4o.

摘要: 监督微调（SFT）通过在标记的特定任务数据上训练大型语言模型（LLM），将它们与人类意图对齐。最近的研究表明，恶意攻击者可以通过将触发器嵌入到有害的问答（QA）对中来向这些模型注入后门。然而，现有的中毒攻击面临着两个关键限制：（1）它们很容易被安全对齐的护栏检测到和过滤（例如，LLaMAGuard），以及（2）嵌入有害内容可能会破坏模型的安全一致性，即使在推理过程中没有触发器的情况下也会导致高攻击成功率（ASB），从而损害隐蔽性。为了解决这些问题，我们针对越狱的LLM提出了一种新型的\clean数据后门攻击。我们的方法没有将触发器与有害响应联系起来，而是使用无害的QA对将它们过度调整为固定的、听起来不错的积极回复前置。推断出，有害反应分两个阶段出现：触发器激活良性前置，模型随后通过利用其语言建模能力和内化先验来完成有害反应。为了进一步增强攻击功效，我们采用基于梯度的协调优化来增强通用触发器。大量实验表明，即使在护栏模型的检测下，我们的方法也可以有效越狱各种LLM，例如，根据GPT-4 o判断，LLaMA-3-8B和Qwen-2.5- 7 B的ASB分别为86.67%和85%。



## **26. Jailbreak Distillation: Renewable Safety Benchmarking**

越狱蒸馏：可再生能源安全基准 cs.CL

Project page: https://aka.ms/jailbreak-distillation

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22037v1) [paper-pdf](http://arxiv.org/pdf/2505.22037v1)

**Authors**: Jingyu Zhang, Ahmed Elgohary, Xiawei Wang, A S M Iftekhar, Ahmed Magooda, Benjamin Van Durme, Daniel Khashabi, Kyle Jackson

**Abstract**: Large language models (LLMs) are rapidly deployed in critical applications, raising urgent needs for robust safety benchmarking. We propose Jailbreak Distillation (JBDistill), a novel benchmark construction framework that "distills" jailbreak attacks into high-quality and easily-updatable safety benchmarks. JBDistill utilizes a small set of development models and existing jailbreak attack algorithms to create a candidate prompt pool, then employs prompt selection algorithms to identify an effective subset of prompts as safety benchmarks. JBDistill addresses challenges in existing safety evaluation: the use of consistent evaluation prompts across models ensures fair comparisons and reproducibility. It requires minimal human effort to rerun the JBDistill pipeline and produce updated benchmarks, alleviating concerns on saturation and contamination. Extensive experiments demonstrate our benchmarks generalize robustly to 13 diverse evaluation models held out from benchmark construction, including proprietary, specialized, and newer-generation LLMs, significantly outperforming existing safety benchmarks in effectiveness while maintaining high separability and diversity. Our framework thus provides an effective, sustainable, and adaptable solution for streamlining safety evaluation.

摘要: 大型语言模型（LLM）迅速部署在关键应用程序中，这引发了对稳健安全基准测试的迫切需求。我们提出了越狱蒸馏（JBDistill），这是一种新颖的基准构建框架，可以将越狱攻击“提炼”为高质量且易于更新的安全基准。JBDistill利用一小组开发模型和现有的越狱攻击算法来创建候选提示池，然后利用提示选择算法来识别有效的提示子集作为安全基准。JBDistill解决了现有安全性评估中的挑战：使用不同模型的一致评估提示确保公平的比较和可重复性。只需最少的人力即可对JBDistill管道进行调试并产生更新的基准，从而减轻对饱和和污染的担忧。大量实验表明，我们的基准强有力地推广到基准构建中提出的13种不同评估模型，包括专有的、专业的和新一代LLM，在有效性上显着优于现有的安全基准，同时保持了高度的可分离性和多样性。因此，我们的框架为简化安全评估提供了有效、可持续且适应性强的解决方案。



## **27. Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs**

超越表面级模式：一个面向LLM的防御框架 cs.CR

16 pages, 12 figures, ACL 2025 findings

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2502.19041v2) [paper-pdf](http://arxiv.org/pdf/2502.19041v2)

**Authors**: Shiyu Xiang, Ansen Zhang, Yanfei Cao, Yang Fan, Ronghao Chen

**Abstract**: Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.

摘要: 尽管对齐大型语言模型（LLM）经过训练可以拒绝有害请求，但它们仍然容易受到越狱攻击。不幸的是，现有的方法通常专注于表面模式，而忽视了更深层次的攻击本质。因此，即使潜在的“攻击本质”保持不变，当攻击促使改变时，防御就会失败。为了解决这个问题，我们引入了EDDF，这是一个\textBF{E} sence-\textBF {D}riven \textBF{D}efense \textBF{F} ravoy Against LLM中的越狱攻击。EDDF是一种即插即用的输入过滤方法，分两个阶段运行：1）离线本质数据库构建，2）在线对抗性查询检测。EDDF背后的关键思想是从不同的已知攻击实例中提取“攻击本质”，并将其存储在离线载体数据库中。实验结果表明，EDDF通过将攻击成功率降低至少20%，显着优于现有方法，凸显了其对越狱攻击的卓越鲁棒性。



## **28. Seeing the Threat: Vulnerabilities in Vision-Language Models to Adversarial Attack**

看到威胁：视觉语言模型对对抗性攻击的脆弱性 cs.CL

Preprint

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.21967v1) [paper-pdf](http://arxiv.org/pdf/2505.21967v1)

**Authors**: Juan Ren, Mark Dras, Usman Naseem

**Abstract**: Large Vision-Language Models (LVLMs) have shown remarkable capabilities across a wide range of multimodal tasks. However, their integration of visual inputs introduces expanded attack surfaces, thereby exposing them to novel security vulnerabilities. In this work, we conduct a systematic representational analysis to uncover why conventional adversarial attacks can circumvent the safety mechanisms embedded in LVLMs. We further propose a novel two stage evaluation framework for adversarial attacks on LVLMs. The first stage differentiates among instruction non compliance, outright refusal, and successful adversarial exploitation. The second stage quantifies the degree to which the model's output fulfills the harmful intent of the adversarial prompt, while categorizing refusal behavior into direct refusals, soft refusals, and partial refusals that remain inadvertently helpful. Finally, we introduce a normative schema that defines idealized model behavior when confronted with harmful prompts, offering a principled target for safety alignment in multimodal systems.

摘要: 大型视觉语言模型（LVLM）在广泛的多模式任务中表现出了非凡的能力。然而，它们对视觉输入的集成引入了扩展的攻击面，从而使它们面临新型安全漏洞。在这项工作中，我们进行了系统的代表性分析，以揭示为什么传统的对抗性攻击可以规避LVLM中嵌入的安全机制。我们进一步提出了一种新颖的两阶段评估框架，用于对LVLM的对抗性攻击。第一阶段区分不遵守指示、彻底拒绝和成功的对抗性剥削。第二阶段量化模型的输出满足对抗提示有害意图的程度，同时将拒绝行为分为直接拒绝、软拒绝和无意中仍然有帮助的部分拒绝。最后，我们引入了一个规范模式，该模式定义了面对有害提示时的理想化模型行为，为多模式系统中的安全对齐提供了原则性目标。



## **29. Watermarks in the Sand: Impossibility of Strong Watermarking for Generative Models**

沙子中的水印：生成模型不可能有强水印 cs.LG

ICML 2024. Website: https://hanlin-zhang.com/impossibility-watermarks

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2311.04378v5) [paper-pdf](http://arxiv.org/pdf/2311.04378v5)

**Authors**: Hanlin Zhang, Benjamin L. Edelman, Danilo Francati, Daniele Venturi, Giuseppe Ateniese, Boaz Barak

**Abstract**: Watermarking generative models consists of planting a statistical signal (watermark) in a model's output so that it can be later verified that the output was generated by the given model. A strong watermarking scheme satisfies the property that a computationally bounded attacker cannot erase the watermark without causing significant quality degradation. In this paper, we study the (im)possibility of strong watermarking schemes. We prove that, under well-specified and natural assumptions, strong watermarking is impossible to achieve. This holds even in the private detection algorithm setting, where the watermark insertion and detection algorithms share a secret key, unknown to the attacker. To prove this result, we introduce a generic efficient watermark attack; the attacker is not required to know the private key of the scheme or even which scheme is used. Our attack is based on two assumptions: (1) The attacker has access to a "quality oracle" that can evaluate whether a candidate output is a high-quality response to a prompt, and (2) The attacker has access to a "perturbation oracle" which can modify an output with a nontrivial probability of maintaining quality, and which induces an efficiently mixing random walk on high-quality outputs. We argue that both assumptions can be satisfied in practice by an attacker with weaker computational capabilities than the watermarked model itself, to which the attacker has only black-box access. Furthermore, our assumptions will likely only be easier to satisfy over time as models grow in capabilities and modalities. We demonstrate the feasibility of our attack by instantiating it to attack three existing watermarking schemes for large language models: Kirchenbauer et al. (2023), Kuditipudi et al. (2023), and Zhao et al. (2023). The same attack successfully removes the watermarks planted by all three schemes, with only minor quality degradation.

摘要: 水印生成模型包括在模型的输出中植入统计信号（水印），以便稍后可以验证输出是由给定模型生成的。强水印方案满足这样的性质：计算有界限的攻击者无法在不引起显着的质量下降的情况下擦除水印。本文研究了强水印方案的可能性。我们证明，在良好指定和自然的假设下，强水印是不可能实现的。即使在私有检测算法设置中，这也成立，其中水印插入和检测算法共享攻击者未知的密钥。为了证明这一结果，我们引入了一种通用的高效水印攻击;攻击者不需要知道该方案的私有密钥，甚至不需要知道使用的是哪个方案。我们的攻击基于两个假设：（1）攻击者可以访问“质量预言机”，该预言机可以评估候选输出是否是对提示的高质量响应，以及（2）攻击者可以访问“扰动预言机”，该预言机可以以维持质量的非平凡概率修改输出，并且这会导致对高质量输出的有效混合随机游走。我们认为，计算能力比水印模型本身弱的攻击者在实践中可以满足这两个假设，攻击者只能访问水印模型。此外，随着模型能力和方式的发展，我们的假设可能只会更容易满足。我们通过实例化攻击大型语言模型的三种现有水印方案来证明我们攻击的可行性：Kirchenbauer等人（2023）、Kuditipudi等人（2023）和Zhao等人（2023）。相同的攻击成功地消除了所有三种方案植入的水印，只有轻微的质量下降。



## **30. Adversarial Attacks against Closed-Source MLLMs via Feature Optimal Alignment**

通过特征最佳对齐对闭源MLLM的对抗攻击 cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21494v1) [paper-pdf](http://arxiv.org/pdf/2505.21494v1)

**Authors**: Xiaojun Jia, Sensen Gao, Simeng Qin, Tianyu Pang, Chao Du, Yihao Huang, Xinfeng Li, Yiming Li, Bo Li, Yang Liu

**Abstract**: Multimodal large language models (MLLMs) remain vulnerable to transferable adversarial examples. While existing methods typically achieve targeted attacks by aligning global features-such as CLIP's [CLS] token-between adversarial and target samples, they often overlook the rich local information encoded in patch tokens. This leads to suboptimal alignment and limited transferability, particularly for closed-source models. To address this limitation, we propose a targeted transferable adversarial attack method based on feature optimal alignment, called FOA-Attack, to improve adversarial transfer capability. Specifically, at the global level, we introduce a global feature loss based on cosine similarity to align the coarse-grained features of adversarial samples with those of target samples. At the local level, given the rich local representations within Transformers, we leverage clustering techniques to extract compact local patterns to alleviate redundant local features. We then formulate local feature alignment between adversarial and target samples as an optimal transport (OT) problem and propose a local clustering optimal transport loss to refine fine-grained feature alignment. Additionally, we propose a dynamic ensemble model weighting strategy to adaptively balance the influence of multiple models during adversarial example generation, thereby further improving transferability. Extensive experiments across various models demonstrate the superiority of the proposed method, outperforming state-of-the-art methods, especially in transferring to closed-source MLLMs. The code is released at https://github.com/jiaxiaojunQAQ/FOA-Attack.

摘要: 多模式大型语言模型（MLLM）仍然容易受到可转移的对抗示例的影响。虽然现有方法通常通过在对抗样本和目标样本之间对齐全局特征（例如CLIP的[LIS]标记）来实现有针对性的攻击，但它们经常忽视补丁令牌中编码的丰富本地信息。这导致次优的对齐和有限的可移植性，特别是对于闭源模型。为了解决这一局限性，我们提出了一种基于特征最优对齐的有针对性的可转移对抗攻击方法，称为FOA-Attack，以提高对抗转移能力。具体来说，在全球层面，我们引入了基于cos相似性的全球特征损失，以将对抗样本的粗粒度特征与目标样本的粗粒度特征对齐。在局部层面，鉴于变形金刚中丰富的局部表示，我们利用集群技术来提取紧凑的局部模式，以减轻冗余的局部特征。然后，我们将对抗样本和目标样本之间的局部特征对齐公式化为最优传输（OT）问题，并提出局部集群最优传输损失来细化细粒度特征对齐。此外，我们还提出了一种动态集成模型加权策略，以自适应地平衡对抗性示例生成过程中多个模型的影响，从而进一步提高可移植性。跨各种模型的广泛实验证明了所提出方法的优越性，优于最先进的方法，特别是在转移到闭源MLLM方面。该代码发布于https://github.com/jiaxiaojunQAQ/FOA-Attack。



## **31. GUARD:Dual-Agent based Backdoor Defense on Chain-of-Thought in Neural Code Generation**

GUARD：神经代码生成中基于双智能体的思维链后门防御 cs.SE

under review

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21425v1) [paper-pdf](http://arxiv.org/pdf/2505.21425v1)

**Authors**: Naizhu Jin, Zhong Li, Tian Zhang, Qingkai Zeng

**Abstract**: With the widespread application of large language models in code generation, recent studies demonstrate that employing additional Chain-of-Thought generation models can significantly enhance code generation performance by providing explicit reasoning steps. However, as external components, CoT models are particularly vulnerable to backdoor attacks, which existing defense mechanisms often fail to detect effectively. To address this challenge, we propose GUARD, a novel dual-agent defense framework specifically designed to counter CoT backdoor attacks in neural code generation. GUARD integrates two core components: GUARD-Judge, which identifies suspicious CoT steps and potential triggers through comprehensive analysis, and GUARD-Repair, which employs a retrieval-augmented generation approach to regenerate secure CoT steps for identified anomalies. Experimental results show that GUARD effectively mitigates attacks while maintaining generation quality, advancing secure code generation systems.

摘要: 随着大型语言模型在代码生成中的广泛应用，最近的研究表明，采用额外的思想链生成模型可以通过提供显式推理步骤来显着提高代码生成性能。然而，作为外部组件，CoT模型特别容易受到后门攻击，而现有的防御机制往往无法有效检测到后门攻击。为了应对这一挑战，我们提出了GUARD，这是一种新型双代理防御框架，专门设计用于对抗神经代码生成中的CoT后门攻击。GUARD集成了两个核心组件：GUARD-Judge，通过全面分析识别可疑的CoT步骤和潜在触发因素，以及GUARD-Repair，采用检索增强生成方法来为识别的异常重新生成安全CoT步骤。实验结果表明，GUARD有效地缓解了攻击，同时保持生成质量，推进了安全代码生成系统。



## **32. JavaSith: A Client-Side Framework for Analyzing Potentially Malicious Extensions in Browsers, VS Code, and NPM Packages**

JavSith：一个用于分析浏览器、VS代码和NPM包中潜在恶意扩展的客户端框架 cs.CR

28 pages , 11 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21263v1) [paper-pdf](http://arxiv.org/pdf/2505.21263v1)

**Authors**: Avihay Cohen

**Abstract**: Modern software supply chains face an increasing threat from malicious code hidden in trusted components such as browser extensions, IDE extensions, and open-source packages. This paper introduces JavaSith, a novel client-side framework for analyzing potentially malicious extensions in web browsers, Visual Studio Code (VSCode), and Node's NPM packages. JavaSith combines a runtime sandbox that emulates browser/Node.js extension APIs (with a ``time machine'' to accelerate time-based triggers) with static analysis and a local large language model (LLM) to assess risk from code and metadata. We present the design and architecture of JavaSith, including techniques for intercepting extension behavior over simulated time and extracting suspicious patterns. Through case studies on real-world attacks (such as a supply-chain compromise of a Chrome extension and malicious VSCode extensions installing cryptominers), we demonstrate how JavaSith can catch stealthy malicious behaviors that evade traditional detection. We evaluate the framework's effectiveness and discuss its limitations and future enhancements. JavaSith's client-side approach empowers end-users/organizations to vet extensions and packages before trustingly integrating them into their environments.

摘要: 现代软件供应链面临着隐藏在浏览器扩展、IDE扩展和开源包等受信任组件中的恶意代码越来越大的威胁。本文介绍了Java Sith，这是一种新型客户端框架，用于分析Web浏览器中潜在的恶意扩展、Visual Studio Code（VSCode）和NPM包。JavSith将模拟浏览器/Node. js扩展API（带有“时间机”来加速基于时间的触发器）的运行时沙箱与静态分析和本地大型语言模型（LLM）相结合，以评估代码和元数据的风险。我们介绍了Java Sith的设计和体系结构，包括在模拟时间内拦截扩展行为和提取可疑模式的技术。通过对真实世界攻击的案例研究（例如Chrome扩展的供应链妥协和安装cryptominers的恶意VSCode扩展），我们展示了JavaSith如何捕获逃避传统检测的隐形恶意行为。我们评估框架的有效性，并讨论其局限性和未来的增强。JavaSith的客户端方法使最终用户/组织能够在将扩展和包可靠地集成到其环境中之前对其进行审查。



## **33. SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA**

SHE-LoRA：用于使用异类LoRA进行联邦调优的选择性homomorm加密 cs.CR

24 pages, 13 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21051v1) [paper-pdf](http://arxiv.org/pdf/2505.21051v1)

**Authors**: Jianmin Liu, Li Yan, Borui Li, Lei Yu, Chao Shen

**Abstract**: Federated fine-tuning of large language models (LLMs) is critical for improving their performance in handling domain-specific tasks. However, prior work has shown that clients' private data can actually be recovered via gradient inversion attacks. Existing privacy preservation techniques against such attacks typically entail performance degradation and high costs, making them ill-suited for clients with heterogeneous data distributions and device capabilities. In this paper, we propose SHE-LoRA, which integrates selective homomorphic encryption (HE) and low-rank adaptation (LoRA) to enable efficient and privacy-preserving federated tuning of LLMs in cross-device environment. Heterogeneous clients adaptively select partial model parameters for homomorphic encryption based on parameter sensitivity assessment, with the encryption subset obtained via negotiation. To ensure accurate model aggregation, we design a column-aware secure aggregation method and customized reparameterization techniques to align the aggregation results with the heterogeneous device capabilities of clients. Extensive experiments demonstrate that SHE-LoRA maintains performance comparable to non-private baselines, achieves strong resistance to the state-of-the-art attacks, and significantly reduces communication overhead by 94.901\% and encryption computation overhead by 99.829\%, compared to baseline. Our code is accessible at https://anonymous.4open.science/r/SHE-LoRA-8D84.

摘要: 大型语言模型（LLM）的联合微调对于提高其处理特定领域任务的性能至关重要。然而，之前的工作表明，客户的私人数据实际上可以通过梯度倒置攻击恢复。针对此类攻击的现有隐私保护技术通常会导致性能下降和成本高，使其不适合具有异类数据分布和设备功能的客户端。本文中，我们提出了SHE-LoRA，它集成了选择性homomorphic加密（HE）和低等级自适应（LoRA），以实现跨设备环境中LLM的高效且保护隐私的联邦调优。异类客户端根据参数敏感度评估自适应地选择部分模型参数进行同质加密，加密子集通过协商获得。为了确保准确的模型聚合，我们设计了一种列感知的安全聚合方法和自定义的重新参数化技术，以使聚合结果与客户端的异类设备能力保持一致。大量实验表明，SHE-LoRA保持了与非私有基线相当的性能，对最先进的攻击具有强大的抵抗力，并与基线相比，将通信负担显着减少了94.901%，加密计算负担显着减少了99.829%。我们的代码可在https://anonymous.4open.science/r/SHE-LoRA-8D84上访问。



## **34. BitHydra: Towards Bit-flip Inference Cost Attack against Large Language Models**

BitHydra：针对大型语言模型的位翻转推理成本攻击 cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.16670v2) [paper-pdf](http://arxiv.org/pdf/2505.16670v2)

**Authors**: Xiaobei Yan, Yiming Li, Zhaoxin Fan, Han Qiu, Tianwei Zhang

**Abstract**: Large language models (LLMs) have shown impressive capabilities across a wide range of applications, but their ever-increasing size and resource demands make them vulnerable to inference cost attacks, where attackers induce victim LLMs to generate the longest possible output content. In this paper, we revisit existing inference cost attacks and reveal that these methods can hardly produce large-scale malicious effects since they are self-targeting, where attackers are also the users and therefore have to execute attacks solely through the inputs, whose generated content will be charged by LLMs and can only directly influence themselves. Motivated by these findings, this paper introduces a new type of inference cost attacks (dubbed 'bit-flip inference cost attack') that target the victim model itself rather than its inputs. Specifically, we design a simple yet effective method (dubbed 'BitHydra') to effectively flip critical bits of model parameters. This process is guided by a loss function designed to suppress <EOS> token's probability with an efficient critical bit search algorithm, thus explicitly defining the attack objective and enabling effective optimization. We evaluate our method on 11 LLMs ranging from 1.5B to 14B parameters under both int8 and float16 settings. Experimental results demonstrate that with just 4 search samples and as few as 3 bit flips, BitHydra can force 100% of test prompts to reach the maximum generation length (e.g., 2048 tokens) on representative LLMs such as LLaMA3, highlighting its efficiency, scalability, and strong transferability across unseen inputs.

摘要: 大型语言模型（LLM）在广泛的应用程序中表现出令人印象深刻的能力，但其不断增加的规模和资源需求使它们容易受到推理成本攻击，攻击者诱导受害者LLM生成尽可能长的输出内容。在本文中，我们回顾了现有的推理成本攻击，并揭示了这些方法很难产生大规模的恶意影响，因为它们是自瞄准的，攻击者也是用户，因此必须仅通过输入来执行攻击，其生成的内容将由LLM收费并且只能直接影响自己。受这些发现的启发，本文引入了一种新型的推理成本攻击（称为“位翻转推理成本攻击”），其目标是受害者模型本身，而不是其输入。具体来说，我们设计了一种简单而有效的方法（称为“BitHydra”）来有效地翻转模型参数的关键部分。该过程由损失函数指导，该函数旨在<EOS>通过高效的关键位搜索算法抑制令牌的概率，从而明确定义攻击目标并实现有效的优化。我们在int8和float 16设置下对11个LLM（参数范围从1.5B到14B）上评估了我们的方法。实验结果表明，只需4个搜索样本和少至3位翻转，BitHydra就可以强制100%的测试提示达到最大生成长度（例如，2048个令牌），突出了其效率，可扩展性和跨看不见的输入的强大可转移性。



## **35. IRCopilot: Automated Incident Response with Large Language Models**

IRCopilot：使用大型语言模型的自动化事件响应 cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20945v1) [paper-pdf](http://arxiv.org/pdf/2505.20945v1)

**Authors**: Xihuan Lin, Jie Zhang, Gelei Deng, Tianzhe Liu, Xiaolong Liu, Changcai Yang, Tianwei Zhang, Qing Guo, Riqing Chen

**Abstract**: Incident response plays a pivotal role in mitigating the impact of cyber attacks. In recent years, the intensity and complexity of global cyber threats have grown significantly, making it increasingly challenging for traditional threat detection and incident response methods to operate effectively in complex network environments. While Large Language Models (LLMs) have shown great potential in early threat detection, their capabilities remain limited when it comes to automated incident response after an intrusion. To address this gap, we construct an incremental benchmark based on real-world incident response tasks to thoroughly evaluate the performance of LLMs in this domain. Our analysis reveals several key challenges that hinder the practical application of contemporary LLMs, including context loss, hallucinations, privacy protection concerns, and their limited ability to provide accurate, context-specific recommendations. In response to these challenges, we propose IRCopilot, a novel framework for automated incident response powered by LLMs. IRCopilot mimics the three dynamic phases of a real-world incident response team using four collaborative LLM-based session components. These components are designed with clear divisions of responsibility, reducing issues such as hallucinations and context loss. Our method leverages diverse prompt designs and strategic responsibility segmentation, significantly improving the system's practicality and efficiency. Experimental results demonstrate that IRCopilot outperforms baseline LLMs across key benchmarks, achieving sub-task completion rates of 150%, 138%, 136%, 119%, and 114% for various response tasks. Moreover, IRCopilot exhibits robust performance on public incident response platforms and in real-world attack scenarios, showcasing its strong applicability.

摘要: 事件响应在减轻网络攻击的影响方面发挥着关键作用。近年来，全球网络威胁的强度和复杂性显着增长，使得传统的威胁检测和事件响应方法在复杂网络环境中有效运作面临越来越大的挑战。虽然大型语言模型（LLM）在早期威胁检测方面表现出了巨大的潜力，但在入侵后自动化事件响应方面，它们的能力仍然有限。为了解决这一差距，我们基于现实世界的事件响应任务构建了一个增量基准，以彻底评估LLM在该领域的性能。我们的分析揭示了阻碍当代LLM实际应用的几个关键挑战，包括上下文丢失、幻觉、隐私保护问题，以及它们提供准确的、针对特定上下文的建议的能力有限。为了应对这些挑战，我们提出了IRCopilot，这是一个由LLM支持的自动化事件响应的新型框架。IRCopilot使用四个基于LLM的协作会话组件模拟现实世界事件响应团队的三个动态阶段。这些组件的设计有明确的责任分工，减少了幻觉和上下文丢失等问题。我们的方法利用多样化的提示设计和战略责任细分，显着提高了系统的实用性和效率。实验结果表明，IRCopilot在关键基准上的表现优于基线LLM，各种响应任务的子任务完成率分别为150%、138%、136%、119%和114%。此外，IRCopilot在公共事件响应平台和现实世界的攻击场景中表现出稳健的性能，展示了其强大的适用性。



## **36. Concealment of Intent: A Game-Theoretic Analysis**

意图的隐瞒：游戏理论分析 cs.CL

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20841v1) [paper-pdf](http://arxiv.org/pdf/2505.20841v1)

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.

摘要: 随着大型语言模型（LLM）的能力越来越强，对其安全部署的担忧也越来越大。尽管已经引入了对齐机制来阻止滥用，但它们仍然容易受到精心设计的对抗提示的影响。在这项工作中，我们提出了一种可扩展的攻击策略：意图隐藏对抗提示，通过技能的组合来隐藏恶意意图。我们开发了一个博弈论框架来建模此类攻击与应用提示和响应过滤的防御系统之间的相互作用。我们的分析确定了平衡点并揭示了攻击者的结构优势。为了应对这些威胁，我们提出并分析了一种针对意图隐藏攻击的防御机制。从经验上讲，我们验证了攻击对一系列恶意行为的多个现实世界LLM的有效性，展示了比现有对抗提示技术的明显优势。



## **37. MedSentry: Understanding and Mitigating Safety Risks in Medical LLM Multi-Agent Systems**

MedSentry：了解和缓解医学LLM多主体系统中的安全风险 cs.MA

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20824v1) [paper-pdf](http://arxiv.org/pdf/2505.20824v1)

**Authors**: Kai Chen, Taihang Zhen, Hewei Wang, Kailai Liu, Xinfeng Li, Jing Huo, Tianpei Yang, Jinfeng Xu, Wei Dong, Yang Gao

**Abstract**: As large language models (LLMs) are increasingly deployed in healthcare, ensuring their safety, particularly within collaborative multi-agent configurations, is paramount. In this paper we introduce MedSentry, a benchmark comprising 5 000 adversarial medical prompts spanning 25 threat categories with 100 subthemes. Coupled with this dataset, we develop an end-to-end attack-defense evaluation pipeline to systematically analyze how four representative multi-agent topologies (Layers, SharedPool, Centralized, and Decentralized) withstand attacks from 'dark-personality' agents. Our findings reveal critical differences in how these architectures handle information contamination and maintain robust decision-making, exposing their underlying vulnerability mechanisms. For instance, SharedPool's open information sharing makes it highly susceptible, whereas Decentralized architectures exhibit greater resilience thanks to inherent redundancy and isolation. To mitigate these risks, we propose a personality-scale detection and correction mechanism that identifies and rehabilitates malicious agents, restoring system safety to near-baseline levels. MedSentry thus furnishes both a rigorous evaluation framework and practical defense strategies that guide the design of safer LLM-based multi-agent systems in medical domains.

摘要: 随着大型语言模型（LLM）越来越多地部署在医疗保健中，确保其安全性，特别是在协作多代理配置中，至关重要。在本文中，我们介绍了MedSentry，这是一个基准，由5000个对抗性医疗提示组成，涵盖25个威胁类别和100个子主题。与此数据集相结合，我们开发了一个端到端的攻击防御评估管道，以系统地分析四种代表性的多智能体布局（Layers、SharedPool、Centralized和Decentralized）如何抵御来自“黑暗人格”智能体的攻击。我们的研究结果揭示了这些架构如何处理信息污染和维持稳健决策的关键差异，暴露了其潜在的脆弱性机制。例如，SharedPool的开放信息共享使其高度容易受到影响，而去中心化架构由于固有的冗余和隔离而表现出更大的弹性。为了减轻这些风险，我们提出了一种个性规模的检测和纠正机制，该机制可以识别和恢复恶意代理，将系统安全性恢复到接近基线的水平。因此，MedSentry提供了严格的评估框架和实用的防御策略，指导医学领域更安全的基于LLM的多智能体系统的设计。



## **38. TrojanStego: Your Language Model Can Secretly Be A Steganographic Privacy Leaking Agent**

TrojanStego：你的语言模型可以秘密地成为隐写隐私泄露代理 cs.CL

9 pages, 5 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20118v2) [paper-pdf](http://arxiv.org/pdf/2505.20118v2)

**Authors**: Dominik Meier, Jan Philip Wahle, Paul Röttger, Terry Ruas, Bela Gipp

**Abstract**: As large language models (LLMs) become integrated into sensitive workflows, concerns grow over their potential to leak confidential information. We propose TrojanStego, a novel threat model in which an adversary fine-tunes an LLM to embed sensitive context information into natural-looking outputs via linguistic steganography, without requiring explicit control over inference inputs. We introduce a taxonomy outlining risk factors for compromised LLMs, and use it to evaluate the risk profile of the threat. To implement TrojanStego, we propose a practical encoding scheme based on vocabulary partitioning learnable by LLMs via fine-tuning. Experimental results show that compromised models reliably transmit 32-bit secrets with 87% accuracy on held-out prompts, reaching over 97% accuracy using majority voting across three generations. Further, they maintain high utility, can evade human detection, and preserve coherence. These results highlight a new class of LLM data exfiltration attacks that are passive, covert, practical, and dangerous.

摘要: 随着大型语言模型（LLM）集成到敏感工作流程中，人们越来越担心它们泄露机密信息的可能性。我们提出了TrojanStego，这是一种新型威胁模型，其中对手微调LLM，通过语言隐写术将敏感的上下文信息嵌入到看起来自然的输出中，而不需要对推理输入进行显式控制。我们引入了一个分类法，概述了受影响的LLM的风险因素，并使用它来评估威胁的风险状况。为了实现TrojanStego，我们提出了一种基于词汇划分的实用编码方案，LLM可以通过微调学习。实验结果表明，受攻击的模型在发出的提示上以87%的准确率可靠地传输32位秘密，使用三代多数投票的准确率达到97%以上。此外，它们保持高实用性，可以逃避人类检测，并保持一致性。这些结果凸显了一类新型LLM数据泄露攻击，这些攻击是被动的、隐蔽的、实用的且危险的。



## **39. Improved Representation Steering for Language Models**

改进的语言模型引导表示 cs.CL

46 pages, 23 figures, preprint

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20809v1) [paper-pdf](http://arxiv.org/pdf/2505.20809v1)

**Authors**: Zhengxuan Wu, Qinan Yu, Aryaman Arora, Christopher D. Manning, Christopher Potts

**Abstract**: Steering methods for language models (LMs) seek to provide fine-grained and interpretable control over model generations by variously changing model inputs, weights, or representations to adjust behavior. Recent work has shown that adjusting weights or representations is often less effective than steering by prompting, for instance when wanting to introduce or suppress a particular concept. We demonstrate how to improve representation steering via our new Reference-free Preference Steering (RePS), a bidirectional preference-optimization objective that jointly does concept steering and suppression. We train three parameterizations of RePS and evaluate them on AxBench, a large-scale model steering benchmark. On Gemma models with sizes ranging from 2B to 27B, RePS outperforms all existing steering methods trained with a language modeling objective and substantially narrows the gap with prompting -- while promoting interpretability and minimizing parameter count. In suppression, RePS matches the language-modeling objective on Gemma-2 and outperforms it on the larger Gemma-3 variants while remaining resilient to prompt-based jailbreaking attacks that defeat prompting. Overall, our results suggest that RePS provides an interpretable and robust alternative to prompting for both steering and suppression.

摘要: 语言模型（LM）的引导方法试图通过各种改变模型输入、权重或表示来调整行为来提供对模型生成的细粒度且可解释的控制。最近的工作表明，调整权重或表示通常不如通过提示引导有效，例如当想要引入或抑制特定概念时。我们演示了如何通过新的无引用偏好引导（RePS）来改进表示引导，RePS是一个双向偏好优化目标，可以联合进行概念引导和抑制。我们训练RePS的三个参数化，并在AxBench（一个大型模型转向基准）上对其进行评估。在大小从2B到27 B的Gemma模型上，RePS优于所有现有的使用语言建模目标训练的转向方法，并大大缩小了与提示的差距-同时提高了可解释性并最大限度地减少了参数数量。在抑制方面，RePS匹配Gemma-2上的语言建模目标，并在更大的Gemma-3变体上优于它，同时保持对失败提示的基于密码的越狱攻击的弹性。总的来说，我们的研究结果表明，RePS提供了一个可解释的和强大的替代，以促进转向和抑制。



## **40. Forewarned is Forearmed: A Survey on Large Language Model-based Agents in Autonomous Cyberattacks**

预先警告就是预先武装：自主网络攻击中基于大型语言模型的代理的调查 cs.NI

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.12786v2) [paper-pdf](http://arxiv.org/pdf/2505.12786v2)

**Authors**: Minrui Xu, Jiani Fan, Xinyu Huang, Conghao Zhou, Jiawen Kang, Dusit Niyato, Shiwen Mao, Zhu Han, Xuemin, Shen, Kwok-Yan Lam

**Abstract**: With the continuous evolution of Large Language Models (LLMs), LLM-based agents have advanced beyond passive chatbots to become autonomous cyber entities capable of performing complex tasks, including web browsing, malicious code and deceptive content generation, and decision-making. By significantly reducing the time, expertise, and resources, AI-assisted cyberattacks orchestrated by LLM-based agents have led to a phenomenon termed Cyber Threat Inflation, characterized by a significant reduction in attack costs and a tremendous increase in attack scale. To provide actionable defensive insights, in this survey, we focus on the potential cyber threats posed by LLM-based agents across diverse network systems. Firstly, we present the capabilities of LLM-based cyberattack agents, which include executing autonomous attack strategies, comprising scouting, memory, reasoning, and action, and facilitating collaborative operations with other agents or human operators. Building on these capabilities, we examine common cyberattacks initiated by LLM-based agents and compare their effectiveness across different types of networks, including static, mobile, and infrastructure-free paradigms. Moreover, we analyze threat bottlenecks of LLM-based agents across different network infrastructures and review their defense methods. Due to operational imbalances, existing defense methods are inadequate against autonomous cyberattacks. Finally, we outline future research directions and potential defensive strategies for legacy network systems.

摘要: 随着大型语言模型（LLM）的不断发展，基于LLM的代理已经超越被动聊天机器人，成为能够执行复杂任务的自治网络实体，包括网络浏览、恶意代码和欺骗性内容生成以及决策。通过显着减少时间、专业知识和资源，由LLM代理策划的人工智能辅助网络攻击导致了一种称为网络威胁通货膨胀的现象，其特征是攻击成本显着降低和攻击规模显着增加。为了提供可操作的防御见解，在本调查中，我们重点关注基于LLM的代理在不同网络系统中构成的潜在网络威胁。首先，我们介绍了基于LLM的网络攻击代理的能力，其中包括执行自主攻击策略，包括侦察、记忆、推理和行动，以及促进与其他代理或人类操作员的协作操作。基于这些功能，我们研究了基于LLM的代理发起的常见网络攻击，并比较了它们在不同类型网络（包括静态，移动和无基础设施模式）中的有效性。此外，我们分析了基于LLM的代理在不同的网络基础设施的威胁瓶颈，并审查其防御方法。由于操作不平衡，现有的防御方法不足以应对自主网络攻击。最后，我们概述了未来的研究方向和潜在的防御策略的遗留网络系统。



## **41. $C^3$-Bench: The Things Real Disturbing LLM based Agent in Multi-Tasking**

$C ' 3 $-Bench：多任务中令人不安的LLM代理人真正不安的事情 cs.AI

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.18746v2) [paper-pdf](http://arxiv.org/pdf/2505.18746v2)

**Authors**: Peijie Yu, Yifan Yang, Jinjian Li, Zelong Zhang, Haorui Wang, Xiao Feng, Feng Zhang

**Abstract**: Agents based on large language models leverage tools to modify environments, revolutionizing how AI interacts with the physical world. Unlike traditional NLP tasks that rely solely on historical dialogue for responses, these agents must consider more complex factors, such as inter-tool relationships, environmental feedback and previous decisions, when making choices. Current research typically evaluates agents via multi-turn dialogues. However, it overlooks the influence of these critical factors on agent behavior. To bridge this gap, we present an open-source and high-quality benchmark $C^3$-Bench. This benchmark integrates attack concepts and applies univariate analysis to pinpoint key elements affecting agent robustness. In concrete, we design three challenges: navigate complex tool relationships, handle critical hidden information and manage dynamic decision paths. Complementing these challenges, we introduce fine-grained metrics, innovative data collection algorithms and reproducible evaluation methods. Extensive experiments are conducted on 49 mainstream agents, encompassing general fast-thinking, slow-thinking and domain-specific models. We observe that agents have significant shortcomings in handling tool dependencies, long context information dependencies and frequent policy-type switching. In essence, $C^3$-Bench aims to expose model vulnerabilities through these challenges and drive research into the interpretability of agent performance. The benchmark is publicly available at https://github.com/yupeijei1997/C3-Bench.

摘要: 基于大型语言模型的代理利用工具来修改环境，彻底改变了人工智能与物理世界交互的方式。与仅依赖历史对话来做出反应的传统NLP任务不同，这些代理人在做出选择时必须考虑更复杂的因素，例如工具间关系、环境反馈和之前的决策。当前的研究通常通过多轮对话来评估代理人。然而，它忽视了这些关键因素对代理行为的影响。为了弥合这一差距，我们提出了一个开源且高质量的基准$C#3 $-Bench。该基准测试集成了攻击概念并应用单变量分析来确定影响代理稳健性的关键元素。具体而言，我们设计了三个挑战：导航复杂的工具关系、处理关键的隐藏信息以及管理动态决策路径。为了补充这些挑战，我们引入了细粒度指标、创新的数据收集算法和可重复的评估方法。广泛的实验进行了49个主流代理，包括一般的快思维，慢思维和特定领域的模型。我们观察到，代理有显着的缺点，在处理工具的依赖关系，长上下文信息的依赖关系和频繁的政策类型切换。本质上，$C^3$-Bench旨在通过这些挑战暴露模型漏洞，并推动对代理性能可解释性的研究。该基准可在https://github.com/yupeijei1997/C3-Bench上公开获得。



## **42. Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models**

超越效率：揭露小型语言模型中越狱攻击的潜在威胁 cs.CR

Accepted to ACL 2025 findings

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2502.19883v3) [paper-pdf](http://arxiv.org/pdf/2502.19883v3)

**Authors**: Sibo Yi, Tianshuo Cong, Xinlei He, Qi Li, Jiaxing Song

**Abstract**: Small language models (SLMs) have become increasingly prominent in the deployment on edge devices due to their high efficiency and low computational cost. While researchers continue to advance the capabilities of SLMs through innovative training strategies and model compression techniques, the security risks of SLMs have received considerably less attention compared to large language models (LLMs).To fill this gap, we provide a comprehensive empirical study to evaluate the security performance of 13 state-of-the-art SLMs under various jailbreak attacks. Our experiments demonstrate that most SLMs are quite susceptible to existing jailbreak attacks, while some of them are even vulnerable to direct harmful prompts.To address the safety concerns, we evaluate several representative defense methods and demonstrate their effectiveness in enhancing the security of SLMs. We further analyze the potential security degradation caused by different SLM techniques including architecture compression, quantization, knowledge distillation, and so on. We expect that our research can highlight the security challenges of SLMs and provide valuable insights to future work in developing more robust and secure SLMs.

摘要: 小型语言模型（SLC）因其高效率和低计算成本而在边缘设备上的部署中变得越来越重要。虽然研究人员不断通过创新的训练策略和模型压缩技术来提高CRM的能力，但与大型语言模型（LLM）相比，CRM的安全风险受到的关注要少得多。为了填补这一空白，我们提供了一项全面的实证研究来评估13种最先进的CRM在各种越狱攻击下的安全性能。我们的实验表明，大多数Slms都很容易受到现有的越狱攻击，而其中一些甚至很容易受到直接的有害提示。为了解决安全问题，我们评估了几种代表性的防御方法，并展示了它们在增强Slms安全性方面的有效性。我们进一步分析了架构压缩、量化、知识提炼等不同的SLA技术所导致的潜在安全降级。我们希望我们的研究能够突出SLC的安全挑战，并为未来开发更强大、更安全的SLC的工作提供有价值的见解。



## **43. Capability-Based Scaling Laws for LLM Red-Teaming**

LLM红色团队基于能力的缩放法则 cs.AI

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.20162v1) [paper-pdf](http://arxiv.org/pdf/2505.20162v1)

**Authors**: Alexander Panfilov, Paul Kassianik, Maksym Andriushchenko, Jonas Geiping

**Abstract**: As large language models grow in capability and agency, identifying vulnerabilities through red-teaming becomes vital for safe deployment. However, traditional prompt-engineering approaches may prove ineffective once red-teaming turns into a weak-to-strong problem, where target models surpass red-teamers in capabilities. To study this shift, we frame red-teaming through the lens of the capability gap between attacker and target. We evaluate more than 500 attacker-target pairs using LLM-based jailbreak attacks that mimic human red-teamers across diverse families, sizes, and capability levels. Three strong trends emerge: (i) more capable models are better attackers, (ii) attack success drops sharply once the target's capability exceeds the attacker's, and (iii) attack success rates correlate with high performance on social science splits of the MMLU-Pro benchmark. From these trends, we derive a jailbreaking scaling law that predicts attack success for a fixed target based on attacker-target capability gap. These findings suggest that fixed-capability attackers (e.g., humans) may become ineffective against future models, increasingly capable open-source models amplify risks for existing systems, and model providers must accurately measure and control models' persuasive and manipulative abilities to limit their effectiveness as attackers.

摘要: 随着大型语言模型能力和代理能力的增长，通过红色团队识别漏洞对于安全部署变得至关重要。然而，一旦红色团队变成一个从弱到强的问题，即目标模型的能力超过红色团队，传统的预算工程方法可能会被证明无效。为了研究这种转变，我们通过攻击者和目标之间的能力差距来构建红色团队。我们使用基于LLM的越狱攻击来评估500多个攻击者目标对，这些攻击者目标对模拟不同家庭、规模和能力水平的人类红队成员。出现了三个强有力的趋势：（i）更有能力的模型是更好的攻击者，（ii）一旦目标的能力超过攻击者的能力，攻击成功率就会急剧下降，（iii）攻击成功率与MMLU-Pro基准的社会科学分裂的高性能相关。从这些趋势中，我们得出了一个越狱的比例法则，预测攻击成功的基础上攻击目标的能力差距的固定目标。这些发现表明，固定能力的攻击者（例如，人类）可能会对未来的模型变得无效，越来越强大的开源模型放大了现有系统的风险，模型提供商必须准确地测量和控制模型的说服和操纵能力，以限制其作为攻击者的有效性。



## **44. PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks**

PandaGuard：针对越狱攻击的LLM安全性系统评估 cs.CR

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.13862v3) [paper-pdf](http://arxiv.org/pdf/2505.13862v3)

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.

摘要: 大型语言模型（LLM）已经取得了卓越的能力，但仍然容易受到被称为越狱的对抗性提示的影响，这可能会绕过安全对齐并引发有害的输出。尽管LLM安全研究的努力越来越多，但现有的评估往往是分散的，集中在孤立的攻击或防御技术上，缺乏系统的，可重复的分析。在这项工作中，我们引入了PandaGuard，一个统一的模块化框架，将LLM越狱安全建模为一个由攻击者，防御者和法官组成的多代理系统。我们的框架实现了19种攻击方法和12种防御机制，以及多种判断策略，所有这些都在一个灵活的插件架构中，支持多种LLM接口，多种交互模式和配置驱动的实验，从而增强了可重复性和实际部署。基于这个框架，我们开发了PandaBench，这是一个全面的基准，可评估49个LLM和各种判断方法之间的相互作用，需要超过30亿个代币来执行。我们的广泛评估揭示了对模型漏洞、国防成本-性能权衡和判断一致性的关键见解。我们发现，没有一种防御在所有维度上都是最佳的，而且判断分歧会在安全评估中引入非平凡的方差。我们发布代码、配置和评估结果，以支持LLM安全性方面的透明和可重复研究。



## **45. Crabs: Consuming Resource via Auto-generation for LLM-DoS Attack under Black-box Settings**

螃蟹：黑匣子设置下通过自动生成来消耗资源进行LLM-NOS攻击 cs.CL

22 pages, 8 figures, 11 tables

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2412.13879v4) [paper-pdf](http://arxiv.org/pdf/2412.13879v4)

**Authors**: Yuanhe Zhang, Zhenhong Zhou, Wei Zhang, Xinyue Wang, Xiaojun Jia, Yang Liu, Sen Su

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks yet still are vulnerable to external threats, particularly LLM Denial-of-Service (LLM-DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, existing studies predominantly focus on white-box attacks, leaving black-box scenarios underexplored. In this paper, we introduce Auto-Generation for LLM-DoS (AutoDoS) attack, an automated algorithm designed for black-box LLMs. AutoDoS constructs the DoS Attack Tree and expands the node coverage to achieve effectiveness under black-box conditions. By transferability-driven iterative optimization, AutoDoS could work across different models in one prompt. Furthermore, we reveal that embedding the Length Trojan allows AutoDoS to bypass existing defenses more effectively. Experimental results show that AutoDoS significantly amplifies service response latency by over 250$\times\uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. Our work provides a new perspective on LLM-DoS attacks and security defenses. Our code is available at https://github.com/shuita2333/AutoDoS.

摘要: 大型语言模型（LLM）在不同任务中表现出了出色的性能，但仍然容易受到外部威胁，尤其是LLM拒绝服务（LLM-NOS）攻击。具体来说，LLM-NOS攻击旨在耗尽计算资源并阻止服务。然而，现有的研究主要集中在白盒攻击上，对黑匣子场景的研究不足。本文中，我们介绍了LLM-DPS攻击的自动生成（AutoDock），这是一种为黑匣子LLM设计的自动算法。AutoDock构建了DPS攻击树，并扩大了节点覆盖范围，以在黑匣子条件下实现有效性。通过可移植性驱动的迭代优化，AutoDock可以在一次提示内跨不同的模型工作。此外，我们还发现，嵌入长度特洛伊木马可以让AutoDock更有效地绕过现有的防御。实验结果表明，AutoDock将服务响应延迟显着放大了超过250 $\times\uparrow $，导致在图形处理器利用率和内存使用率方面严重消耗资源。我们的工作为LLM-NOS攻击和安全防御提供了新的视角。我们的代码可在https://github.com/shuita2333/AutoDoS上获取。



## **46. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

揭示一致大型语言模型内在的道德脆弱性 cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2504.05050v3) [paper-pdf](http://arxiv.org/pdf/2504.05050v3)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.

摘要: 大型语言模型（LLM）是人工通用智能的基础探索，但它们通过指令调整和偏好学习与人类价值观的一致只能实现表面的合规性。在这里，我们证明，预训练期间嵌入的有害知识在LLM参数记忆中作为不可磨灭的“黑暗模式”持续存在，逃避对齐保障措施，并在分布变化时的对抗诱导下重新浮出水面。在这项研究中，我们首先通过证明当前的对齐方法只产生知识集合中的局部“安全区域”来从理论上分析对齐LLM的内在道德脆弱性。相比之下，预先训练的知识仍然通过高可能性的对抗轨迹与有害概念保持全球联系。基于这一理论见解，我们通过在分布转移下采用语义一致诱导来从经验上验证我们的发现--一种通过优化的对抗提示系统性地绕过对齐约束的方法。这种理论和经验相结合的方法在23个最先进的对齐LLM中的19个（包括DeepSeek-R1和LLaMA-3）上实现了100%的攻击成功率，揭示了它们的普遍漏洞。



## **47. Attention! You Vision Language Model Could Be Maliciously Manipulated**

注意！您的视觉语言模型可能被恶意操纵 cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19911v1) [paper-pdf](http://arxiv.org/pdf/2505.19911v1)

**Authors**: Xiaosen Wang, Shaokang Wang, Zhijin Ge, Yuyang Luo, Shudong Zhang

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable success in understanding complex real-world scenarios and supporting data-driven decision-making processes. However, VLMs exhibit significant vulnerability against adversarial examples, either text or image, which can lead to various adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In this work, we empirically and theoretically demonstrate that VLMs are particularly susceptible to image-based adversarial examples, where imperceptible perturbations can precisely manipulate each output token. To this end, we propose a novel attack called Vision-language model Manipulation Attack (VMA), which integrates first-order and second-order momentum optimization techniques with a differentiable transformation mechanism to effectively optimize the adversarial perturbation. Notably, VMA can be a double-edged sword: it can be leveraged to implement various attacks, such as jailbreaking, hijacking, privacy breaches, Denial-of-Service, and the generation of sponge examples, etc, while simultaneously enabling the injection of watermarks for copyright protection. Extensive empirical evaluations substantiate the efficacy and generalizability of VMA across diverse scenarios and datasets.

摘要: 大型视觉语言模型（VLM）在理解复杂的现实世界场景和支持数据驱动的决策流程方面取得了显着的成功。然而，VLM对对抗性示例（无论是文本还是图像）表现出显着的脆弱性，这可能会导致各种对抗性结果，例如越狱、劫持和幻觉等。在这项工作中，我们从经验和理论上证明了VLM特别容易受到基于图像的对抗示例的影响，其中不可感知的扰动可以精确地操纵每个输出令牌。为此，我们提出了一种名为视觉语言模型操纵攻击（VMA）的新型攻击，该攻击将一阶和二阶动量优化技术与可微转换机制集成在一起，以有效地优化对抗性扰动。值得注意的是，VMA可以是一把双刃剑：它可以被用来实施各种攻击，例如越狱、劫持、隐私泄露、拒绝服务和海绵示例的生成等，同时允许注入水印以进行版权保护。广泛的实证评估证实了VMA在不同场景和数据集中的有效性和普遍性。



## **48. CPA-RAG:Covert Poisoning Attacks on Retrieval-Augmented Generation in Large Language Models**

CPA-RAG：对大型语言模型中检索增强生成的隐蔽中毒攻击 cs.CR

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19864v1) [paper-pdf](http://arxiv.org/pdf/2505.19864v1)

**Authors**: Chunyang Li, Junwei Zhang, Anda Cheng, Zhuo Ma, Xinghua Li, Jianfeng Ma

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, but its openness introduces vulnerabilities that can be exploited by poisoning attacks. Existing poisoning methods for RAG systems have limitations, such as poor generalization and lack of fluency in adversarial texts. In this paper, we propose CPA-RAG, a black-box adversarial framework that generates query-relevant texts capable of manipulating the retrieval process to induce target answers. The proposed method integrates prompt-based text generation, cross-guided optimization through multiple LLMs, and retriever-based scoring to construct high-quality adversarial samples. We conduct extensive experiments across multiple datasets and LLMs to evaluate its effectiveness. Results show that the framework achieves over 90\% attack success when the top-k retrieval setting is 5, matching white-box performance, and maintains a consistent advantage of approximately 5 percentage points across different top-k values. It also outperforms existing black-box baselines by 14.5 percentage points under various defense strategies. Furthermore, our method successfully compromises a commercial RAG system deployed on Alibaba's BaiLian platform, demonstrating its practical threat in real-world applications. These findings underscore the need for more robust and secure RAG frameworks to defend against poisoning attacks.

摘要: 检索增强生成（RAG）通过合并外部知识来增强大型语言模型（LLM），但其开放性引入了可被中毒攻击利用的漏洞。现有的RAG系统中毒方法存在局限性，例如概括性较差以及对抗性文本缺乏流畅性。在本文中，我们提出了CPA-RAG，这是一个黑盒对抗框架，可以生成与查询相关的文本，这些文本能够操纵检索过程以诱导目标答案。所提出的方法集成了基于文本的生成，通过多个LLM的交叉引导优化，以及基于检索器的评分来构建高质量的对抗样本。我们在多个数据集和LLM中进行了广泛的实验，以评估其有效性。结果表明，当top-k检索设置为5时，该框架的攻击成功率超过90%，与白盒性能相匹配，并在不同top-k值之间保持约5个百分点的一致优势。在各种防御策略下，它还比现有黑匣子基线高出14.5个百分点。此外，我们的方法成功地破坏了阿里巴巴百联平台上部署的商业RAG系统，证明了其在现实应用中的实际威胁。这些发现强调了需要更强大、更安全的RAG框架来抵御中毒攻击。



## **49. Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models**

越狱音频长凳：深入评估和分析大型音频语言模型的越狱威胁 cs.SD

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2501.13772v2) [paper-pdf](http://arxiv.org/pdf/2501.13772v2)

**Authors**: Hao Cheng, Erjia Xiao, Jing Shao, Yichi Wang, Le Yang, Chao Sheng, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Large Language Models (LLMs) demonstrate impressive zero-shot performance across a wide range of natural language processing tasks. Integrating various modality encoders further expands their capabilities, giving rise to Multimodal Large Language Models (MLLMs) that process not only text but also visual and auditory modality inputs. However, these advanced capabilities may also pose significant security risks, as models can be exploited to generate harmful or inappropriate content through jailbreak attacks. While prior work has extensively explored how manipulating textual or visual modality inputs can circumvent safeguards in LLMs and MLLMs, the vulnerability of audio-specific Jailbreak on Large Audio-Language Models (LALMs) remains largely underexplored. To address this gap, we introduce Jailbreak-AudioBench, which consists of the Toolbox, curated Dataset, and comprehensive Benchmark. The Toolbox supports not only text-to-audio conversion but also a range of audio editing techniques. The curated Dataset provides diverse explicit and implicit jailbreak audio examples in both original and edited forms. Utilizing this dataset, we evaluate multiple state-of-the-art LALMs, establishing the most comprehensive audio jailbreak benchmark to date. Finally, Jailbreak-AudioBench establishes a foundation for advancing future research on LALMs safety alignment by enabling the in-depth exposure of more powerful jailbreak threats, such as query-based audio editing, and by facilitating the development of effective defense mechanisms.

摘要: 大型语言模型（LLM）在广泛的自然语言处理任务中表现出令人印象深刻的零冲击性能。集成各种模式编码器进一步扩展了它们的功能，从而产生了多模式大型语言模型（MLLM），不仅处理文本，还处理视觉和听觉模式输入。然而，这些高级功能也可能带来重大的安全风险，因为模型可能会被利用来通过越狱攻击生成有害或不适当的内容。虽然之前的工作已经广泛探索了操纵文本或视觉模式输入如何规避LLM和MLLM中的保护措施，但大型音频语言模型（LALM）上的音频特定越狱的漏洞在很大程度上仍然没有得到充分的研究。为了弥补这一差距，我们引入了Jailbreak-AudioBench，它由收件箱、精心策划的数据集和全面的基准组成。收件箱不仅支持文本到音频转换，还支持一系列音频编辑技术。精心策划的数据集以原始和编辑的形式提供了多样化的显式和隐式越狱音频示例。利用该数据集，我们评估了多个最先进的LALM，建立了迄今为止最全面的音频越狱基准。最后，Jailbreak-AudioBench通过深入暴露更强大的越狱威胁（如基于查询的音频编辑），并促进有效防御机制的开发，为推进LALM安全对齐的未来研究奠定了基础。



## **50. QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language**

QuickAttack：使用结构化非自然查询语言越狱对齐的大型语言模型 cs.CR

To appear in ACL 2025

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2502.09723v3) [paper-pdf](http://arxiv.org/pdf/2502.09723v3)

**Authors**: Qingsong Zou, Jingyu Xiao, Qing Li, Zhi Yan, Yuhang Wang, Li Xu, Wenxuan Wang, Kuofeng Gao, Ruoyu Li, Yong Jiang

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable potential in the field of natural language processing. Unfortunately, LLMs face significant security and ethical risks. Although techniques such as safety alignment are developed for defense, prior researches reveal the possibility of bypassing such defenses through well-designed jailbreak attacks. In this paper, we propose QueryAttack, a novel framework to examine the generalizability of safety alignment. By treating LLMs as knowledge databases, we translate malicious queries in natural language into structured non-natural query language to bypass the safety alignment mechanisms of LLMs. We conduct extensive experiments on mainstream LLMs, and the results show that QueryAttack not only can achieve high attack success rates (ASRs), but also can jailbreak various defense methods. Furthermore, we tailor a defense method against QueryAttack, which can reduce ASR by up to $64\%$ on GPT-4-1106. Our code is available at https://github.com/horizonsinzqs/QueryAttack.

摘要: 大型语言模型（LLM）的最新进展在自然语言处理领域展示了巨大的潜力。不幸的是，LLM面临着巨大的安全和道德风险。尽管安全对齐等技术是为了防御而开发的，但之前的研究揭示了通过精心设计的越狱攻击绕过此类防御的可能性。在本文中，我们提出了一种新颖的框架，用于检查安全对齐的通用性。通过将LLM视为知识数据库，我们将自然语言中的恶意查询翻译为结构化非自然查询语言，以绕过LLM的安全对齐机制。我们在主流LLM上进行了广泛的实验，结果表明，CredyAttack不仅可以实现高的攻击成功率（SVR），还可以越狱各种防御方法。此外，我们还定制了一种针对SecureAttack的防御方法，该方法可以在GPT-4-1106上将ASB降低高达64美元。我们的代码可在https://github.com/horizonsinzqs/QueryAttack上获取。



