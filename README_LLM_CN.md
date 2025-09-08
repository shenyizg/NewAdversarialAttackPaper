# Latest Large Language Model Attack Papers
**update at 2025-09-08 17:23:31**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection**

AgentArmor：对Agent DeliverTrace执行程序分析以防止即时注入 cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2508.01249v2) [paper-pdf](http://arxiv.org/pdf/2508.01249v2)

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu

**Abstract**: Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's runtime traces as graph-based intermediate representations with control and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools \& data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis for sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can reduce the ASR to 3\%, with the utility drop only 1\%.

摘要: 大型语言模型（LLM）代理通过将自然语言推理与外部工具的执行相结合，提供了一个强大的新范式来解决各种问题。然而，它们的动态和不透明行为会带来严重的安全风险，特别是在存在即时注入攻击的情况下。在这项工作中，我们提出了一种新颖的见解，将代理运行时跟踪视为具有可分析语义的结构化程序。因此，我们提出了AgentArmor，这是一个程序分析框架，它将代理跟踪转换为基于图形中间表示的结构化程序依赖性表示（例如，CGM、DFG和PDG）并通过类型系统强制执行安全策略。AgentArmor由三个关键组件组成：（1）一个图形构造器，它将代理的运行时跟踪重建为基于图形的中间表示，其中描述了控制和数据流;（2）一个属性注册表，它附加了交互工具\&数据的安全相关元数据，以及（3）一个类型系统，它执行静态推理和检查中间表示。通过将代理行为表示为结构化程序，AgentArmor可以对敏感数据流、信任边界和策略违规进行程序分析。在AgentDojo基准测试中对AgentArmor进行了测试，结果表明，AgentArmor可以将ASR降低到3%，而效用下降仅为1%。



## **2. RINSER: Accurate API Prediction Using Masked Language Models**

RINser：使用掩蔽语言模型进行准确的API预测 cs.CY

16 pages, 8 figures

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04887v1) [paper-pdf](http://arxiv.org/pdf/2509.04887v1)

**Authors**: Muhammad Ejaz Ahmed, Christopher Cody, Muhammad Ikram, Sean Lamont, Alsharif Abuadbba, Seyit Camtepe, Surya Nepal, Muhammad Ali Kaafar

**Abstract**: Malware authors commonly use obfuscation to hide API identities in binary files, making analysis difficult and time-consuming for a human expert to understand the behavior and intent of the program. Automatic API prediction tools are necessary to efficiently analyze unknown binaries, facilitating rapid malware triage while reducing the workload on human analysts. In this paper, we present RINSER (AccuRate API predictioN using maSked languagE model leaRning), an automated framework for predicting Windows API (WinAPI) function names. RINSER introduces the novel concept of API codeprints, a set of API-relevant assembly instructions, and supports x86 PE binaries. RINSER relies on BERT's masked language model (LM) to predict API names at scale, achieving 85.77% accuracy for normal binaries and 82.88% accuracy for stripped binaries. We evaluate RINSER on a large dataset of 4.7M API codeprints from 11,098 malware binaries, covering 4,123 unique Windows APIs, making it the largest publicly available dataset of this type. RINSER successfully discovered 65 obfuscated Windows APIs related to C2 communication, spying, and evasion in our dataset, which the commercial disassembler IDA failed to identify. Furthermore, we compared RINSER against three state-of-the-art approaches, showing over 20% higher prediction accuracy. We also demonstrated RINSER's resilience to adversarial attacks, including instruction randomization and code displacement, with a performance drop of no more than 3%.

摘要: 恶意软件作者通常使用混淆将API身份隐藏在二进制文件中，这使得人类专家理解程序的行为和意图变得困难且耗时。自动API预测工具对于有效分析未知二进制文件来说是必要的，可以促进快速恶意软件分类，同时减少人类分析师的工作量。在本文中，我们介绍了RINBER（使用maSked languagE模型leRning的ACATER API预测），这是一个用于预测Windows API（WinAPI）函数名称的自动化框架。RINser引入了API代码印的新颖概念，即一组与API相关的汇编指令，并支持x86 PE二进制文件。RINser依赖BERT的掩蔽语言模型（LM）来大规模预测API名称，正常二进制文件的准确性达到85.77%，剥离二进制文件的准确性达到82.88%。我们在一个包含来自11，098个恶意软件二进制文件的470万个API代码的大型数据集上评估了RINser，涵盖了4，123个独特的Windows API，使其成为此类类型中最大的公开可用数据集。RINBER在我们的数据集中成功发现了65个与C2通信、间谍和规避相关的模糊Windows API，但商业反汇编器IDA未能识别这些API。此外，我们将RINBER与三种最先进的方法进行了比较，结果显示预测准确性提高了20%以上。我们还展示了RINBER对对抗攻击（包括指令随机化和代码置换）的弹性，性能下降不超过3%。



## **3. Antidote: Post-fine-tuning Safety Alignment for Large Language Models against Harmful Fine-tuning**

解药：微调后大型语言模型的安全调整，防止有害的微调 cs.AI

Rejected by AAAI25-AIA. Accepted by ICML25. Authors are thankful to  the anonymous reviewers from both AAAI25-AIA and ICML25

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2408.09600v3) [paper-pdf](http://arxiv.org/pdf/2408.09600v3)

**Authors**: Tiansheng Huang, Gautam Bhattacharya, Pratik Joshi, Josh Kimball, Ling Liu

**Abstract**: Safety aligned Large Language Models (LLMs) are vulnerable to harmful fine-tuning attacks -- a few harmful data mixed in the fine-tuning dataset can break the LLMs's safety alignment. While several defenses have been proposed, our evaluation shows that existing defenses fail \textit{when some specific training hyper-parameters are chosen} -- a large learning rate or a large number of training epochs in the fine-tuning stage can easily invalidate the defense. To this end, we propose Antidote, a post-fine-tuning stage solution, which remains \textbf{\textit{agnostic to the training hyper-parameters in the fine-tuning stage}}. Antidote relies on the philosophy that by removing the harmful parameters, the harmful model can be recovered from the harmful behaviors, regardless of how those harmful parameters are formed in the fine-tuning stage. With this philosophy, we introduce a one-shot pruning stage after harmful fine-tuning to remove the harmful weights that are responsible for the generation of harmful content. Despite its embarrassing simplicity, empirical results show that Antidote can reduce harmful score while maintaining accuracy on downstream tasks. Code is available at https://github.com/git-disl/Antidote.

摘要: 安全一致的大型语言模型（LLM）容易受到有害的微调攻击--微调数据集中混合的一些有害数据可能会破坏LLM的安全一致性。虽然已经提出了几种防御措施，但我们的评估表明，现有的防御措施会失败\textit{当选择一些特定的训练超参数时} --微调阶段中的大学习率或大量训练时期很容易使防御无效。为此，我们提出了Antidote，这是一种微调后的解决方案，它仍然是\textBF{\texttit {与微调阶段中的训练超参数不可知}。解药依赖于这样的理念：通过去除有害参数，有害模型可以从有害行为中恢复，无论这些有害参数是如何在微调阶段形成的。凭借这一理念，我们在有害微调后引入一次性修剪阶段，以删除导致有害内容生成的有害权重。尽管它的简单性令人尴尬，但经验结果表明，解毒剂可以降低有害评分，同时保持下游任务的准确性。代码可在https://github.com/git-disl/Antidote上获取。



## **4. Mind the Gap: Evaluating Model- and Agentic-Level Vulnerabilities in LLMs with Action Graphs**

注意差距：使用行动图评估LLM中的模型和统计级别漏洞 cs.CL

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04802v1) [paper-pdf](http://arxiv.org/pdf/2509.04802v1)

**Authors**: Ilham Wicaksono, Zekun Wu, Theo King, Adriano Koshiyama, Philip Treleaven

**Abstract**: As large language models transition to agentic systems, current safety evaluation frameworks face critical gaps in assessing deployment-specific risks. We introduce AgentSeer, an observability-based evaluation framework that decomposes agentic executions into granular action and component graphs, enabling systematic agentic-situational assessment. Through cross-model validation on GPT-OSS-20B and Gemini-2.0-flash using HarmBench single turn and iterative refinement attacks, we demonstrate fundamental differences between model-level and agentic-level vulnerability profiles. Model-level evaluation reveals baseline differences: GPT-OSS-20B (39.47% ASR) versus Gemini-2.0-flash (50.00% ASR), with both models showing susceptibility to social engineering while maintaining logic-based attack resistance. However, agentic-level assessment exposes agent-specific risks invisible to traditional evaluation. We discover "agentic-only" vulnerabilities that emerge exclusively in agentic contexts, with tool-calling showing 24-60% higher ASR across both models. Cross-model analysis reveals universal agentic patterns, agent transfer operations as highest-risk tools, semantic rather than syntactic vulnerability mechanisms, and context-dependent attack effectiveness, alongside model-specific security profiles in absolute ASR levels and optimal injection strategies. Direct attack transfer from model-level to agentic contexts shows degraded performance (GPT-OSS-20B: 57% human injection ASR; Gemini-2.0-flash: 28%), while context-aware iterative attacks successfully compromise objectives that failed at model-level, confirming systematic evaluation gaps. These findings establish the urgent need for agentic-situation evaluation paradigms, with AgentSeer providing the standardized methodology and empirical validation.

摘要: 随着大型语言模型向代理系统过渡，当前的安全评估框架在评估特定于部署的风险方面面临着严重差距。我们引入了AgentSeer，这是一个基于可观察性的评估框架，它将代理执行分解为粒度动作和组件图，从而实现系统性代理情景评估。通过使用HarmBench单轮攻击和迭代细化攻击对GPT-OSS-20 B和Gemini-2.0-Flash进行跨模型验证，我们展示了模型级和代理级漏洞配置文件之间的根本差异。模型级评估揭示了基线差异：GPT-OSS-20 B（39.47%ASB）与Gemini-2.0-Flash（50.00%ASB），两种模型都表现出对社会工程的敏感性，同时保持基于逻辑的攻击抵抗力。然而，代理层面的评估暴露了传统评估所看不到的代理特定风险。我们发现了仅在代理环境中出现的“仅代理”漏洞，工具调用显示两种模型的ASB高出24-60%。跨模型分析揭示了通用的代理模式、作为最高风险工具的代理传输操作、语义而非语法漏洞机制、取决于上下文的攻击有效性，以及绝对ASB级别的模型特定安全配置文件和最佳注入策略。从模型级到代理上下文的直接攻击转移显示出性能下降（GPT-OSS-20 B：57%人体注射ASO; Gemini-2.0-Flash：28%），而上下文感知迭代攻击成功地破坏了在模型级失败的目标，证实了系统性评估差距。这些发现确定了对主体情境评估范式的迫切需求，AgentSeer提供了标准化的方法论和经验验证。



## **5. Breaking to Build: A Threat Model of Prompt-Based Attacks for Securing LLMs**

突破构建：用于保护LLM的基于预算的攻击威胁模型 cs.CL

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04615v1) [paper-pdf](http://arxiv.org/pdf/2509.04615v1)

**Authors**: Brennen Hill, Surendra Parla, Venkata Abhijeeth Balabhadruni, Atharv Prajod Padmalayam, Sujay Chandra Shekara Sharma

**Abstract**: The proliferation of Large Language Models (LLMs) has introduced critical security challenges, where adversarial actors can manipulate input prompts to cause significant harm and circumvent safety alignments. These prompt-based attacks exploit vulnerabilities in a model's design, training, and contextual understanding, leading to intellectual property theft, misinformation generation, and erosion of user trust. A systematic understanding of these attack vectors is the foundational step toward developing robust countermeasures. This paper presents a comprehensive literature survey of prompt-based attack methodologies, categorizing them to provide a clear threat model. By detailing the mechanisms and impacts of these exploits, this survey aims to inform the research community's efforts in building the next generation of secure LLMs that are inherently resistant to unauthorized distillation, fine-tuning, and editing.

摘要: 大型语言模型（LLM）的激增带来了关键的安全挑战，对抗行为者可以操纵输入提示造成重大伤害并规避安全一致。这些基于预算的攻击利用模型设计、培训和上下文理解中的漏洞，导致知识产权盗窃、错误信息生成和用户信任度侵蚀。系统地了解这些攻击载体是开发稳健对策的基础步骤。本文对基于预算的攻击方法进行了全面的文献调查，对它们进行了分类，以提供明确的威胁模型。通过详细介绍这些漏洞利用的机制和影响，本调查旨在为研究界构建下一代安全LLM的努力提供信息，这些LLM本质上可以抵抗未经授权的提炼、微调和编辑。



## **6. Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing**

Concept-ROT：使用模型编辑在大型语言模型中中毒概念 cs.LG

Published at ICLR 2025

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2412.13341v2) [paper-pdf](http://arxiv.org/pdf/2412.13341v2)

**Authors**: Keltin Grimes, Marco Christiani, David Shriver, Marissa Connor

**Abstract**: Model editing methods modify specific behaviors of Large Language Models by altering a small, targeted set of network weights and require very little data and compute. These methods can be used for malicious applications such as inserting misinformation or simple trojans that result in adversary-specified behaviors when a trigger word is present. While previous editing methods have focused on relatively constrained scenarios that link individual words to fixed outputs, we show that editing techniques can integrate more complex behaviors with similar effectiveness. We develop Concept-ROT, a model editing-based method that efficiently inserts trojans which not only exhibit complex output behaviors, but also trigger on high-level concepts -- presenting an entirely new class of trojan attacks. Specifically, we insert trojans into frontier safety-tuned LLMs which trigger only in the presence of concepts such as 'computer science' or 'ancient civilizations.' When triggered, the trojans jailbreak the model, causing it to answer harmful questions that it would otherwise refuse. Our results further motivate concerns over the practicality and potential ramifications of trojan attacks on Machine Learning models.

摘要: 模型编辑方法通过改变一小组有针对性的网络权重来修改大型语言模型的特定行为，并且需要很少的数据和计算。这些方法可用于恶意应用程序，例如插入错误信息或简单的特洛伊木马，当存在触发词时，这些木马会导致对手指定的行为。虽然之前的编辑方法专注于将单个单词与固定输出联系起来的相对受限的场景，但我们表明编辑技术可以以类似的效果集成更复杂的行为。我们开发了Concept-ROT，这是一种基于模型编辑的方法，可以有效地插入特洛伊木马，这些木马不仅表现出复杂的输出行为，而且还会触发高级概念--从而呈现出一种全新的特洛伊木马攻击。具体来说，我们将特洛伊木马插入到前沿安全调整的LLM中，这些LLM仅在存在“计算机科学”或“古代文明”等概念时才会触发。“当被触发时，特洛伊木马会越狱该模型，使其回答原本会拒绝的有害问题。我们的结果进一步引发了人们对机器学习模型木马攻击的实用性和潜在后果的担忧。



## **7. An Automated, Scalable Machine Learning Model Inversion Assessment Pipeline**

自动化、可扩展的机器学习模型反演评估管道 cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04214v1) [paper-pdf](http://arxiv.org/pdf/2509.04214v1)

**Authors**: Tyler Shumaker, Jessica Carpenter, David Saranchak, Nathaniel D. Bastian

**Abstract**: Machine learning (ML) models have the potential to transform military battlefields, presenting a large external pressure to rapidly incorporate them into operational settings. However, it is well-established that these ML models are vulnerable to a number of adversarial attacks throughout the model deployment pipeline that threaten to negate battlefield advantage. One broad category is privacy attacks (such as model inversion) where an adversary can reverse engineer information from the model, such as the sensitive data used in its training. The ability to quantify the risk of model inversion attacks (MIAs) is not well studied, and there is a lack of automated developmental test and evaluation (DT&E) tools and metrics to quantify the effectiveness of privacy loss of the MIA. The current DT&E process is difficult because ML model inversions can be hard for a human to interpret, subjective when they are interpretable, and difficult to quantify in terms of inversion quality. Additionally, scaling the DT&E process is challenging due to many ML model architectures and data modalities that need to be assessed. In this work, we present a novel DT&E tool that quantifies the risk of data privacy loss from MIAs and introduces four adversarial risk dimensions to quantify privacy loss. Our DT&E pipeline combines inversion with vision language models (VLMs) to improve effectiveness while enabling scalable analysis. We demonstrate effectiveness using multiple MIA techniques and VLMs configured for zero-shot classification and image captioning. We benchmark the pipeline using several state-of-the-art MIAs in the computer vision domain with an image classification task that is typical in military applications. In general, our innovative pipeline extends the current model inversion DT&E capabilities by improving the effectiveness and scalability of the privacy loss analysis in an automated fashion.

摘要: 机器学习（ML）模型有潜力改变军事战场，这给快速将其纳入作战环境带来了巨大的外部压力。然而，众所周知，这些ML模型在整个模型部署管道中容易受到许多对抗攻击，这些攻击可能会抵消战场优势。其中一个广泛的类别是隐私攻击（例如模型倒置），其中对手可以从模型中反向工程信息，例如训练中使用的敏感数据。量化模型倒置攻击（MIA）风险的能力尚未得到充分研究，并且缺乏自动化开发测试和评估（DT & E）工具和指标来量化MIA隐私损失的有效性。当前的DT & E过程很困难，因为ML模型倒置对人类来说可能很难解释，当它们可解释时是主观的，并且很难在倒置质量方面量化。此外，由于需要评估许多ML模型架构和数据模式，扩展DT & E流程具有挑战性。在这项工作中，我们提出了一种新颖的DT & E工具，该工具量化了MIA造成的数据隐私损失的风险，并引入了四个对抗风险维度来量化隐私损失。我们的DT & E管道将倒置与视觉语言模型（VLM）相结合，以提高有效性，同时实现可扩展分析。我们使用多种MIA技术和配置用于零镜头分类和图像字幕的VLM来证明有效性。我们使用计算机视觉领域的几种最先进的MIA对管道进行基准测试，并执行军事应用中典型的图像分类任务。总的来说，我们的创新管道通过以自动化方式提高隐私损失分析的有效性和可扩展性来扩展当前的模型倒置DT & E功能。



## **8. KubeGuard: LLM-Assisted Kubernetes Hardening via Configuration Files and Runtime Logs Analysis**

KubeGuard：通过配置文件和数据库分析的LLM辅助Kubernetes硬化 cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04191v1) [paper-pdf](http://arxiv.org/pdf/2509.04191v1)

**Authors**: Omri Sgan Cohen, Ehud Malul, Yair Meidan, Dudu Mimran, Yuval Elovici, Asaf Shabtai

**Abstract**: The widespread adoption of Kubernetes (K8s) for orchestrating cloud-native applications has introduced significant security challenges, such as misconfigured resources and overly permissive configurations. Failing to address these issues can result in unauthorized access, privilege escalation, and lateral movement within clusters. Most existing K8s security solutions focus on detecting misconfigurations, typically through static analysis or anomaly detection. In contrast, this paper presents KubeGuard, a novel runtime log-driven recommender framework aimed at mitigating risks by addressing overly permissive configurations. KubeGuard is designed to harden K8s environments through two complementary tasks: Resource Creation and Resource Refinement. It leverages large language models (LLMs) to analyze manifests and runtime logs reflecting actual system behavior, using modular prompt-chaining workflows. This approach enables KubeGuard to create least-privilege configurations for new resources and refine existing manifests to reduce the attack surface. KubeGuard's output manifests are presented as recommendations that users (e.g., developers and operators) can review and adopt to enhance cluster security. Our evaluation demonstrates that KubeGuard effectively generates and refines K8s manifests for Roles, NetworkPolicies, and Deployments, leveraging both proprietary and open-source LLMs. The high precision, recall, and F1-scores affirm KubeGuard's practicality as a framework that translates runtime observability into actionable, least-privilege configuration guidance.

摘要: Kubernetes（K8 s）广泛采用来编排云原生应用程序，带来了重大的安全挑战，例如资源配置错误和配置过于宽松。未能解决这些问题可能会导致未经授权的访问、特权升级和集群内的横向移动。大多数现有的K8安全解决方案专注于检测错误配置，通常通过静态分析或异常检测。相比之下，本文介绍了KubeGuard，这是一个新颖的运行时日志驱动的推荐框架，旨在通过解决过度许可的配置来降低风险。KubeGuard旨在通过两项补充任务来强化K8环境：资源创建和资源细化。它利用大型语言模型（LLM）来使用模块化预算链接工作流程来分析反映实际系统行为的清单和运行时日志。这种方法使KubeGuard能够为新资源创建最低特权配置，并改进现有清单以减少攻击面。KubeGuard的输出清单作为用户的推荐（例如，开发者和运营商）可以审查和采用以增强集群安全性。我们的评估表明，KubeGuard可以利用专有和开源LLM有效地生成和改进角色、网络策略和部署的K8清单。高精度、召回率和F1分数证实了KubeGuard作为一个框架的实用性，该框架将运行时可观察性转化为可操作的、最低特权配置指南。



## **9. Privacy Risks in Time Series Forecasting: User- and Record-Level Membership Inference**

时间序列预测中的隐私风险：用户和记录级会员推断 cs.LG

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04169v1) [paper-pdf](http://arxiv.org/pdf/2509.04169v1)

**Authors**: Nicolas Johansson, Tobias Olsson, Daniel Nilsson, Johan Östman, Fazeleh Hoseini

**Abstract**: Membership inference attacks (MIAs) aim to determine whether specific data were used to train a model. While extensively studied on classification models, their impact on time series forecasting remains largely unexplored. We address this gap by introducing two new attacks: (i) an adaptation of multivariate LiRA, a state-of-the-art MIA originally developed for classification models, to the time-series forecasting setting, and (ii) a novel end-to-end learning approach called Deep Time Series (DTS) attack. We benchmark these methods against adapted versions of other leading attacks from the classification setting.   We evaluate all attacks in realistic settings on the TUH-EEG and ELD datasets, targeting two strong forecasting architectures, LSTM and the state-of-the-art N-HiTS, under both record- and user-level threat models. Our results show that forecasting models are vulnerable, with user-level attacks often achieving perfect detection. The proposed methods achieve the strongest performance in several settings, establishing new baselines for privacy risk assessment in time series forecasting. Furthermore, vulnerability increases with longer prediction horizons and smaller training populations, echoing trends observed in large language models.

摘要: 隶属度推理攻击（MIA）旨在确定是否使用特定数据来训练模型。虽然对分类模型进行了广泛研究，但它们对时间序列预测的影响在很大程度上仍未被探索。我们通过引入两种新的攻击来解决这一差距：（i）多元LiRA（最初为分类模型开发的最先进的MIA）适应时间序列预测设置，以及（ii）一种新型的端到端学习方法，称为深度时间序列（RST）攻击。我们将这些方法与分类设置中其他主要攻击的改编版本进行基准测试。   我们在TUH-EEG和ELD数据集上评估现实环境中的所有攻击，针对两种强大的预测架构LSTM和最先进的N-HiTS，在记录级和用户级威胁模型下。我们的结果表明，预测模型很容易受到攻击，用户级攻击通常可以实现完美检测。提出的方法在多种环境下实现了最强的性能，为时间序列预测中的隐私风险评估建立了新的基线。此外，预测视野越长和训练群体越少，脆弱性就会越大，这与大型语言模型中观察到的趋势相呼应。



## **10. Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks**

先发制人：预先合成类似越狱的指令，以增强LLM安全防范潜在攻击 cs.CL

EMNLP 2025 findings

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2508.20038v3) [paper-pdf](http://arxiv.org/pdf/2508.20038v3)

**Authors**: Sheng Liu, Qiang Sheng, Danding Wang, Yang Li, Guang Yang, Juan Cao

**Abstract**: Despite advances in improving large language model (LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.

摘要: 尽管在改进大型语言模型（LLM）以拒绝回答恶意指令方面取得了进展，但广泛使用的LLM仍然容易受到越狱攻击，攻击者生成的指令分布与安全对齐库不同。新的攻击暴露了LLM无法识别不可见的恶意指令，凸显了训练数据和现实世界攻击之间严重的分布不匹配，迫使开发人员进入反应性补丁周期。为了应对这一挑战，我们提出了IMAGINE，这是一个综合框架，利用嵌入空间分布分析来生成类似越狱的指令。这种方法有效地填补了真实越狱模式和安全调整数据库之间的分布差距。IMAGINE遵循迭代优化过程，在迭代中动态演变文本生成分布，从而通过合成数据示例扩大安全对齐数据分布的覆盖范围。基于通过IMAGINE增强的安全对齐的数据库，我们的框架证明了Qwen 2.5、Llama 3.1和Llama 3.2的攻击成功率显着下降，而不会影响其实用性。



## **11. NeuroBreak: Unveil Internal Jailbreak Mechanisms in Large Language Models**

NeuroBreak：揭开大型语言模型中的内部越狱机制 cs.CR

12 pages, 9 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.03985v1) [paper-pdf](http://arxiv.org/pdf/2509.03985v1)

**Authors**: Chuhan Zhang, Ye Zhang, Bowen Shi, Yuyou Gan, Tianyu Du, Shouling Ji, Dazhan Deng, Yingcai Wu

**Abstract**: In deployment and application, large language models (LLMs) typically undergo safety alignment to prevent illegal and unethical outputs. However, the continuous advancement of jailbreak attack techniques, designed to bypass safety mechanisms with adversarial prompts, has placed increasing pressure on the security defenses of LLMs. Strengthening resistance to jailbreak attacks requires an in-depth understanding of the security mechanisms and vulnerabilities of LLMs. However, the vast number of parameters and complex structure of LLMs make analyzing security weaknesses from an internal perspective a challenging task. This paper presents NeuroBreak, a top-down jailbreak analysis system designed to analyze neuron-level safety mechanisms and mitigate vulnerabilities. We carefully design system requirements through collaboration with three experts in the field of AI security. The system provides a comprehensive analysis of various jailbreak attack methods. By incorporating layer-wise representation probing analysis, NeuroBreak offers a novel perspective on the model's decision-making process throughout its generation steps. Furthermore, the system supports the analysis of critical neurons from both semantic and functional perspectives, facilitating a deeper exploration of security mechanisms. We conduct quantitative evaluations and case studies to verify the effectiveness of our system, offering mechanistic insights for developing next-generation defense strategies against evolving jailbreak attacks.

摘要: 在部署和应用中，大型语言模型（LLM）通常会进行安全调整，以防止非法和不道德的输出。然而，越狱攻击技术的不断进步（旨在通过对抗提示绕过安全机制）给LLM的安全防御带来了越来越大的压力。加强对越狱攻击的抵抗需要深入了解LLM的安全机制和漏洞。然而，LLM的大量参数和复杂结构使得从内部角度分析安全弱点成为一项具有挑战性的任务。本文介绍了NeuroBreak，这是一个自上而下的越狱分析系统，旨在分析神经元级安全机制并缓解漏洞。我们通过与人工智能安全领域的三位专家合作，精心设计系统需求。该系统对各种越狱攻击方法进行了全面分析。通过结合分层表示探测分析，NeuroBreak为模型整个生成步骤的决策过程提供了新颖的视角。此外，该系统支持从语义和功能角度分析关键神经元，促进对安全机制的更深入探索。我们进行定量评估和案例研究来验证我们系统的有效性，为开发针对不断变化的越狱攻击的下一代防御策略提供机械见解。



## **12. Defending LVLMs Against Vision Attacks through Partial-Perception Supervision**

通过部分感知监督保护LVLM免受视觉攻击 cs.CV

Accepted to ICML 2025

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2412.12722v2) [paper-pdf](http://arxiv.org/pdf/2412.12722v2)

**Authors**: Qi Zhou, Tianlin Li, Qing Guo, Dongxia Wang, Yun Lin, Yang Liu, Jin Song Dong

**Abstract**: Recent studies have raised significant concerns regarding the vulnerability of Large Vision Language Models (LVLMs) to maliciously injected or perturbed input images, which can mislead their responses. Existing defense methods show that such vision attacks are sensitive to image modifications especially cropping, using majority voting across responses of modified images as corrected responses. However, these modifications often result in partial images and distort the semantics, which reduces response quality on clean images after voting. Instead of directly using responses from partial images for voting, we investigate using them to supervise the LVLM's responses to the original images. We propose a black-box, training-free method called DPS (Defense through Partial-Perception Supervision). In this approach, the model is prompted using the responses generated by a model that perceives only a partial image. With DPS, the model can adjust its response based on partial image understanding when under attack, while confidently maintaining its original response for clean input. Our findings show that the weak model can supervise the strong model: when faced with an attacked input, the strong model becomes less confident and adjusts its response based on the weak model's partial understanding, effectively defending against the attack. With clean input, it confidently maintains its original response. Empirical experiments show our method outperforms the baseline, cutting the average attack success rate by 76.3% across six datasets on three popular models.

摘要: 最近的研究对大视觉语言模型（LVLM）对恶意注入或干扰的输入图像的脆弱性提出了严重担忧，这可能会误导他们的反应。现有的防御方法表明，此类视觉攻击对图像修改（尤其是裁剪）敏感，使用修改后图像的响应的多数投票作为纠正的响应。然而，这些修改通常会导致部分图像并扭曲语义，从而降低投票后干净图像的响应质量。我们没有直接使用部分图像的响应进行投票，而是研究使用它们来监督LVLM对原始图像的响应。我们提出了一种称为DPS（通过部分感知监督进行防御）的黑匣子、免训练方法。在这种方法中，使用仅感知部分图像的模型生成的响应来提示模型。通过DPS，模型可以在受到攻击时根据部分图像理解调整其响应，同时自信地保持其原始响应以获得干净的输入。我们的研究结果表明，弱模型可以监督强模型：当面对被攻击的输入时，强模型变得不那么自信，并根据弱模型的部分理解调整其响应，有效地防御攻击。通过干净的输入，它自信地保持其原始反应。经验实验表明，我们的方法优于基线，将三种流行模型的六个数据集中的平均攻击成功率降低了76.3%。



## **13. VulRTex: A Reasoning-Guided Approach to Identify Vulnerabilities from Rich-Text Issue Report**

VulRTex：一种从富文本问题报告中识别漏洞的推理引导方法 cs.SE

25 pages, 7 figures, submitting to TOSEM journal

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.03875v1) [paper-pdf](http://arxiv.org/pdf/2509.03875v1)

**Authors**: Ziyou Jiang, Mingyang Li, Guowei Yang, Lin Shi, Qing Wang

**Abstract**: Software vulnerabilities exist in open-source software (OSS), and the developers who discover these vulnerabilities may submit issue reports (IRs) to describe their details. Security practitioners need to spend a lot of time manually identifying vulnerability-related IRs from the community, and the time gap may be exploited by attackers to harm the system. Previously, researchers have proposed automatic approaches to facilitate identifying these vulnerability-related IRs, but these works focus on textual descriptions but lack the comprehensive analysis of IR's rich-text information. In this paper, we propose VulRTex, a reasoning-guided approach to identify vulnerability-related IRs with their rich-text information. In particular, VulRTex first utilizes the reasoning ability of the Large Language Model (LLM) to prepare the Vulnerability Reasoning Database with historical IRs. Then, it retrieves the relevant cases from the prepared reasoning database to generate reasoning guidance, which guides LLM to identify vulnerabilities by reasoning analysis on target IRs' rich-text information. To evaluate the performance of VulRTex, we conduct experiments on 973,572 IRs, and the results show that VulRTex achieves the highest performance in identifying the vulnerability-related IRs and predicting CWE-IDs when the dataset is imbalanced, outperforming the best baseline with +11.0% F1, +20.2% AUPRC, and +10.5% Macro-F1, and 2x lower time cost than baseline reasoning approaches. Furthermore, VulRTex has been applied to identify 30 emerging vulnerabilities across 10 representative OSS projects in 2024's GitHub IRs, and 11 of them are successfully assigned CVE-IDs, which illustrates VulRTex's practicality.

摘要: 软件漏洞存在于开源软件（OSS）中，发现这些漏洞的开发人员可以提交问题报告（IR）来描述其详细信息。安全从业人员需要花费大量时间从社区中手动识别与安全性相关的IR，而攻击者可能会利用这一时间间隔来损害系统。在此之前，研究人员已经提出了自动化的方法来帮助识别这些可识别性相关的IR，但这些作品侧重于文本描述，但缺乏对IR的富文本信息的全面分析。在本文中，我们提出了VulRTex，一种推理引导的方法来识别与其富文本信息相关的可扩展性IR。特别是，VulRTex首先利用大型语言模型（LLM）的推理能力来准备具有历史IR的漏洞推理数据库。然后，从准备好的推理数据库中检索相关案例，生成推理指导，引导LLM通过对目标IR的富文本信息进行推理分析来识别漏洞。为了评估VulRTex的性能，我们对973，572个IR进行了实验，结果表明VulRTex在识别可互换性相关IR和预测数据集不平衡时的CWE-ID方面达到了最高性能，优于最佳基线，分别为+11.0%F1、+20.2%AUPRC和+10. 5% Macro-F1，时间成本比基线推理方法低2倍。此外，VulRTex已被应用于识别2024年GitHub IR中10个代表性OSS项目的30个新兴漏洞，其中11个已成功分配了CVE-ID，这说明了VulRTex的实用性。



## **14. Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models**

使用语言模型生成的对抗性示例进行攻击错误信息检测 cs.CL

Presented at EMNLP 2025

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2410.20940v2) [paper-pdf](http://arxiv.org/pdf/2410.20940v2)

**Authors**: Piotr Przybyła, Euan McGill, Horacio Saggion

**Abstract**: Large language models have many beneficial applications, but can they also be used to attack content-filtering algorithms in social media platforms? We investigate the challenge of generating adversarial examples to test the robustness of text classification algorithms detecting low-credibility content, including propaganda, false claims, rumours and hyperpartisan news. We focus on simulation of content moderation by setting realistic limits on the number of queries an attacker is allowed to attempt. Within our solution (TREPAT), initial rephrasings are generated by large language models with prompts inspired by meaning-preserving NLP tasks, such as text simplification and style transfer. Subsequently, these modifications are decomposed into small changes, applied through beam search procedure, until the victim classifier changes its decision. We perform (1) quantitative evaluation using various prompts, models and query limits, (2) targeted manual assessment of the generated text and (3) qualitative linguistic analysis. The results confirm the superiority of our approach in the constrained scenario, especially in case of long input text (news articles), where exhaustive search is not feasible.

摘要: 大型语言模型有许多有益的应用，但它们也可以用于攻击社交媒体平台中的内容过滤算法吗？我们调查了生成敌对示例的挑战，以测试检测低可信度内容（包括宣传、虚假声明、谣言和超党派新闻）的文本分类算法的稳健性。我们通过对允许攻击者尝试的查询数量设置现实的限制来重点模拟内容审核。在我们的解决方案（TREPAT）中，初始改写由大型语言模型生成，其提示受到保留意义的NLP任务（例如文本简化和风格转移）的启发。随后，这些修改被分解成小的变化，通过束搜索过程应用，直到受害者分类器改变其决定。我们（1）使用各种提示、模型和查询限制执行定量评估，（2）对生成的文本进行有针对性的手动评估，（3）定性语言分析。结果证实了我们的方法在受限场景中的优越性，特别是在长输入文本（新闻文章）的情况下，其中详尽搜索是不可行的。



## **15. BadPromptFL: A Novel Backdoor Threat to Prompt-based Federated Learning in Multimodal Models**

BadminutFL：对多模式模型中基于预算的联邦学习的新型后门威胁 cs.LG

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2508.08040v2) [paper-pdf](http://arxiv.org/pdf/2508.08040v2)

**Authors**: Maozhen Zhang, Mengnan Zhao, Bo Wang

**Abstract**: Prompt-based tuning has emerged as a lightweight alternative to full fine-tuning in large vision-language models, enabling efficient adaptation via learned contextual prompts. This paradigm has recently been extended to federated learning settings (e.g., PromptFL), where clients collaboratively train prompts under data privacy constraints. However, the security implications of prompt-based aggregation in federated multimodal learning remain largely unexplored, leaving a critical attack surface unaddressed. In this paper, we introduce \textbf{BadPromptFL}, the first backdoor attack targeting prompt-based federated learning in multimodal contrastive models. In BadPromptFL, compromised clients jointly optimize local backdoor triggers and prompt embeddings, injecting poisoned prompts into the global aggregation process. These prompts are then propagated to benign clients, enabling universal backdoor activation at inference without modifying model parameters. Leveraging the contextual learning behavior of CLIP-style architectures, BadPromptFL achieves high attack success rates (e.g., \(>90\%\)) with minimal visibility and limited client participation. Extensive experiments across multiple datasets and aggregation protocols validate the effectiveness, stealth, and generalizability of our attack, raising critical concerns about the robustness of prompt-based federated learning in real-world deployments.

摘要: 基于预算的调优已成为大型视觉语言模型中完全微调的轻量级替代方案，可以通过学习的上下文提示进行高效调整。该范式最近已扩展到联邦学习环境（例如，AtlantFL），客户在数据隐私限制下协作训练提示。然而，联邦多模式学习中基于预算的聚合的安全影响在很大程度上仍未得到探索，导致关键的攻击表面尚未得到解决。本文中，我们介绍了\textBF{BadoutFL}，这是第一个针对多模式对比模型中基于预算的联邦学习的后门攻击。在BadoutFL中，受影响的客户端联合优化本地后门触发器和提示嵌入，将有毒提示注入到全球聚合流程中。然后，这些提示被传播到良性客户端，从而在推理时启用通用后门，而无需修改模型参数。利用CLIP风格架构的上下文学习行为，BadminutFL实现了高攻击成功率（例如，\（>90\%\））可见性最低且客户参与有限。跨多个数据集和聚合协议的广泛实验验证了我们攻击的有效性、隐蔽性和可推广性，引发了人们对现实世界部署中基于预算的联邦学习稳健性的严重担忧。



## **16. PromptCOS: Towards System Prompt Copyright Auditing for LLMs via Content-level Output Similarity**

Observtcos：通过内容级输出相似性实现LLM的系统提示版权审计 cs.CR

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03117v1) [paper-pdf](http://arxiv.org/pdf/2509.03117v1)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Enhao Huang, Shuo Shao, Bingrun Yang, Zhibo Wang, Dacheng Tao, Zhan Qin

**Abstract**: The rapid progress of large language models (LLMs) has greatly enhanced reasoning tasks and facilitated the development of LLM-based applications. A critical factor in improving LLM-based applications is the design of effective system prompts, which significantly impact the behavior and output quality of LLMs. However, system prompts are susceptible to theft and misuse, which could undermine the interests of prompt owners. Existing methods protect prompt copyrights through watermark injection and verification but face challenges due to their reliance on intermediate LLM outputs (e.g., logits), which limits their practical feasibility.   In this paper, we propose PromptCOS, a method for auditing prompt copyright based on content-level output similarity. It embeds watermarks by optimizing the prompt while simultaneously co-optimizing a special verification query and content-level signal marks. This is achieved by leveraging cyclic output signals and injecting auxiliary tokens to ensure reliable auditing in content-only scenarios. Additionally, it incorporates cover tokens to protect the watermark from malicious deletion. For copyright verification, PromptCOS identifies unauthorized usage by comparing the similarity between the suspicious output and the signal mark. Experimental results demonstrate that our method achieves high effectiveness (99.3% average watermark similarity), strong distinctiveness (60.8% greater than the best baseline), high fidelity (accuracy degradation of no more than 0.58%), robustness (resilience against three types of potential attacks), and computational efficiency (up to 98.1% reduction in computational cost). Our code is available at GitHub https://github.com/LianPing-cyber/PromptCOS.

摘要: 大型语言模型（LLM）的快速发展极大地增强了推理任务并促进了基于LLM的应用程序的开发。改进基于LLM的应用程序的一个关键因素是设计有效的系统提示，这将显着影响LLM的行为和输出质量。然而，系统提示很容易被盗窃和滥用，这可能会损害提示所有者的利益。现有方法通过水印注入和验证来保护即时版权，但由于依赖中间LLM输出（例如，logits），这限制了它们的实际可行性。   在本文中，我们提出了Inbox cos，这是一种基于内容级输出相似度的提示版权审计方法。它通过优化提示来嵌入水印，同时协同优化特殊验证查询和内容级信号标记。这是通过利用循环输出信号和注入辅助令牌来实现的，以确保仅内容场景中的可靠审计。此外，它还结合了封面令牌来保护水印免受恶意删除。对于版权验证，Inbox cos通过比较可疑输出和信号标记之间的相似性来识别未经授权的使用。实验结果表明，我们的方法具有高有效性（平均水印相似度为99.3%）、强区分性（比最佳基线高60.8%）、高保真度（准确率下降不超过0.58%）、鲁棒性（对三种潜在攻击的弹性）和计算效率（计算成本降低高达98.1%）。我们的代码可在GitHub https://github.com/LianPing-cyber/PromptCOS上获取。



## **17. EverTracer: Hunting Stolen Large Language Models via Stealthy and Robust Probabilistic Fingerprint**

EverTracer：通过隐秘且稳健的概率指纹追踪被盗的大型语言模型 cs.CR

Accepted by EMNLP2025 Main

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03058v1) [paper-pdf](http://arxiv.org/pdf/2509.03058v1)

**Authors**: Zhenhua Xu, Meng Han, Wenpeng Xing

**Abstract**: The proliferation of large language models (LLMs) has intensified concerns over model theft and license violations, necessitating robust and stealthy ownership verification. Existing fingerprinting methods either require impractical white-box access or introduce detectable statistical anomalies. We propose EverTracer, a novel gray-box fingerprinting framework that ensures stealthy and robust model provenance tracing. EverTracer is the first to repurpose Membership Inference Attacks (MIAs) for defensive use, embedding ownership signals via memorization instead of artificial trigger-output overfitting. It consists of Fingerprint Injection, which fine-tunes the model on any natural language data without detectable artifacts, and Verification, which leverages calibrated probability variation signal to distinguish fingerprinted models. This approach remains robust against adaptive adversaries, including input level modification, and model-level modifications. Extensive experiments across architectures demonstrate EverTracer's state-of-the-art effectiveness, stealthness, and resilience, establishing it as a practical solution for securing LLM intellectual property. Our code and data are publicly available at https://github.com/Xuzhenhua55/EverTracer.

摘要: 大型语言模型（LLM）的激增加剧了人们对模型盗窃和许可证违规的担忧，需要进行强大且隐蔽的所有权验证。现有的指纹识别方法要么需要不切实际的白盒访问，要么引入可检测到的统计异常。我们提出了EverTracer，这是一种新型的灰箱指纹识别框架，可以确保隐蔽且稳健的模型出处追踪。EverTracer是第一个将会员推断攻击（MIA）重新用于防御用途的公司，通过记忆而不是人为的命令输出过度匹配来嵌入所有权信号。它由指纹注入和验证组成，前者在任何自然语言数据上微调模型，而不会检测到伪影，后者利用校准的概率变化信号来区分指纹模型。这种方法对于自适应对手（包括输入级别修改和模型级别修改）仍然具有鲁棒性。跨架构的广泛实验证明了EverTracer最先进的有效性、隐蔽性和弹性，使其成为保护LLM知识产权的实用解决方案。我们的代码和数据可在https://github.com/Xuzhenhua55/EverTracer上公开获取。



## **18. See No Evil: Adversarial Attacks Against Linguistic-Visual Association in Referring Multi-Object Tracking Systems**

看不到邪恶：引用多对象跟踪系统中对语言视觉关联的对抗攻击 cs.CV

12 pages, 1 figure, 3 tables

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.02028v2) [paper-pdf](http://arxiv.org/pdf/2509.02028v2)

**Authors**: Halima Bouzidi, Haoyu Liu, Mohammad Abdullah Al Faruque

**Abstract**: Language-vision understanding has driven the development of advanced perception systems, most notably the emerging paradigm of Referring Multi-Object Tracking (RMOT). By leveraging natural-language queries, RMOT systems can selectively track objects that satisfy a given semantic description, guided through Transformer-based spatial-temporal reasoning modules. End-to-End (E2E) RMOT models further unify feature extraction, temporal memory, and spatial reasoning within a Transformer backbone, enabling long-range spatial-temporal modeling over fused textual-visual representations. Despite these advances, the reliability and robustness of RMOT remain underexplored. In this paper, we examine the security implications of RMOT systems from a design-logic perspective, identifying adversarial vulnerabilities that compromise both the linguistic-visual referring and track-object matching components. Additionally, we uncover a novel vulnerability in advanced RMOT models employing FIFO-based memory, whereby targeted and consistent attacks on their spatial-temporal reasoning introduce errors that persist within the history buffer over multiple subsequent frames. We present VEIL, a novel adversarial framework designed to disrupt the unified referring-matching mechanisms of RMOT models. We show that carefully crafted digital and physical perturbations can corrupt the tracking logic reliability, inducing track ID switches and terminations. We conduct comprehensive evaluations using the Refer-KITTI dataset to validate the effectiveness of VEIL and demonstrate the urgent need for security-aware RMOT designs for critical large-scale applications.

摘要: 图像视觉理解推动了高级感知系统的发展，最引人注目的是参考多对象跟踪（RMOT）的新兴范式。通过利用自然语言查询，RMOT系统可以在基于Transformer的时空推理模块的指导下选择性地跟踪满足给定语义描述的对象。端到端（E2 E）RMOT模型进一步统一了Transformer骨干中的特征提取、时间记忆和空间推理，从而在融合的文本-视觉表示上实现了长距离时空建模。尽管有这些进步，RMOT的可靠性和鲁棒性仍然没有得到充分的研究。在本文中，我们从设计逻辑的角度研究了RMOT系统的安全影响，识别了损害语言视觉引用和跟踪对象匹配组件的对抗漏洞。此外，我们还发现了使用基于FIFO的内存的高级RMOT模型中的一个新漏洞，从而对其时空推理进行有针对性且一致的攻击，从而引入了在历史缓冲区中持续存在的错误。我们提出了VEIL，这是一种新型对抗框架，旨在破坏RMOT模型的统一触发匹配机制。我们表明，精心设计的数字和物理扰动可能会破坏跟踪逻辑的可靠性，导致轨道ID开关和端接。我们使用Refer-KITTI数据集进行全面评估，以验证VEIL的有效性，并证明关键大规模应用对安全感知RMOT设计的迫切需求。



## **19. A Survey: Towards Privacy and Security in Mobile Large Language Models**

调查：移动大型语言模型中的隐私和安全 cs.CR

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02411v1) [paper-pdf](http://arxiv.org/pdf/2509.02411v1)

**Authors**: Honghui Xu, Kaiyang Li, Wei Chen, Danyang Zheng, Zhiyuan Li, Zhipeng Cai

**Abstract**: Mobile Large Language Models (LLMs) are revolutionizing diverse fields such as healthcare, finance, and education with their ability to perform advanced natural language processing tasks on-the-go. However, the deployment of these models in mobile and edge environments introduces significant challenges related to privacy and security due to their resource-intensive nature and the sensitivity of the data they process. This survey provides a comprehensive overview of privacy and security issues associated with mobile LLMs, systematically categorizing existing solutions such as differential privacy, federated learning, and prompt encryption. Furthermore, we analyze vulnerabilities unique to mobile LLMs, including adversarial attacks, membership inference, and side-channel attacks, offering an in-depth comparison of their effectiveness and limitations. Despite recent advancements, mobile LLMs face unique hurdles in achieving robust security while maintaining efficiency in resource-constrained environments. To bridge this gap, we propose potential applications, discuss open challenges, and suggest future research directions, paving the way for the development of trustworthy, privacy-compliant, and scalable mobile LLM systems.

摘要: 移动大型语言模型（LLM）正在彻底改变医疗保健、金融和教育等各个领域，因为它们能够随时执行高级自然语言处理任务。然而，由于这些模型的资源密集型性质和处理数据的敏感性，在移动和边缘环境中部署这些模型会带来与隐私和安全相关的重大挑战。本调查全面概述了与移动LLM相关的隐私和安全问题，系统地对现有解决方案进行分类，例如差异隐私、联合学习和提示加密。此外，我们还分析了移动LLM特有的漏洞，包括对抗性攻击、成员资格推断和侧通道攻击，并对其有效性和局限性进行了深入比较。尽管最近取得了进步，但移动LLM在实现强大的安全性同时在资源有限的环境中保持效率方面面临着独特的障碍。为了弥合这一差距，我们提出了潜在的应用程序，讨论了开放的挑战，并提出了未来的研究方向，为开发值得信赖、符合隐私和可扩展的移动LLM系统铺平了道路。



## **20. Enhancing Reliability in LLM-Integrated Robotic Systems: A Unified Approach to Security and Safety**

提高LLM集成机器人系统的可靠性：统一的安全保障方法 cs.RO

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02163v1) [paper-pdf](http://arxiv.org/pdf/2509.02163v1)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Bräunl, Jin B. Hong

**Abstract**: Integrating large language models (LLMs) into robotic systems has revolutionised embodied artificial intelligence, enabling advanced decision-making and adaptability. However, ensuring reliability, encompassing both security against adversarial attacks and safety in complex environments, remains a critical challenge. To address this, we propose a unified framework that mitigates prompt injection attacks while enforcing operational safety through robust validation mechanisms. Our approach combines prompt assembling, state management, and safety validation, evaluated using both performance and security metrics. Experiments show a 30.8% improvement under injection attacks and up to a 325% improvement in complex environment settings under adversarial conditions compared to baseline scenarios. This work bridges the gap between safety and security in LLM-based robotic systems, offering actionable insights for deploying reliable LLM-integrated mobile robots in real-world settings. The framework is open-sourced with simulation and physical deployment demos at https://llmeyesim.vercel.app/

摘要: 将大型语言模型（LLM）集成到机器人系统中彻底改变了具体人工智能，实现了高级决策和适应性。然而，确保可靠性（包括对抗攻击的安全性和复杂环境中的安全性）仍然是一个严峻的挑战。为了解决这个问题，我们提出了一个统一的框架，该框架可以减轻即时注入攻击，同时通过强大的验证机制强制执行操作安全。我们的方法结合了即时组装、状态管理和安全验证，并使用性能和安全指标进行评估。实验表明，与基线场景相比，在注入攻击下的性能提高了30.8%，在对抗条件下的复杂环境设置中的性能提高了325%。这项工作弥合了基于LLM的机器人系统的安全性和安全性之间的差距，为在现实环境中部署可靠的LLM集成移动机器人提供了可操作的见解。该框架是开源的，在https://llmeyesim.vercel.app/上有模拟和物理部署演示



## **21. Unraveling LLM Jailbreaks Through Safety Knowledge Neurons**

通过安全知识神经元解开LLM越狱事件 cs.AI

10 pages, 6 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01631v1) [paper-pdf](http://arxiv.org/pdf/2509.01631v1)

**Authors**: Chongwen Zhao, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation, a technique known as "Jailbreak." While some studies have achieved defenses against jailbreak attacks by modifying output distributions or detecting harmful content, the exact rationale still remains elusive. In this work, we present a novel neuron-level interpretability method that focuses on the role of safety-related knowledge neurons. Unlike existing approaches, our method projects the model's internal representation into a more consistent and interpretable vocabulary space. We then show that adjusting the activation of safety-related neurons can effectively control the model's behavior with a mean ASR higher than 97%. Building on this insight, we propose SafeTuning, a fine-tuning strategy that reinforces safety-critical neurons to improve model robustness against jailbreaks. SafeTuning consistently reduces attack success rates across multiple LLMs and outperforms all four baseline defenses. These findings offer a new perspective on understanding and defending against jailbreak attacks.

摘要: 大型语言模型（LLM）在各种应用中越来越受到关注。尽管如此，人们越来越担心，因为一些用户试图利用这些模型进行恶意目的，包括合成受控物质和传播虚假信息，一种被称为“越狱”的技术。“虽然一些研究通过修改输出分发或检测有害内容来实现对越狱攻击的防御，但确切的理由仍然难以捉摸。在这项工作中，我们提出了一种新型的神经元水平解释方法，重点关注与安全相关的知识神经元的作用。与现有方法不同，我们的方法将模型的内部表示投影到更一致和可解释的词汇空间中。然后我们表明，调整安全相关神经元的激活可以有效控制模型的行为，平均ASB高于97%。基于这一见解，我们提出了SafeTuning，这是一种微调策略，可以加强安全关键神经元，以提高模型针对越狱的鲁棒性。SafeTuning持续降低多个LLM的攻击成功率，并优于所有四种基线防御。这些发现为理解和防御越狱攻击提供了新的视角。



## **22. LLMHoney: A Real-Time SSH Honeypot with Large Language Model-Driven Dynamic Response Generation**

LLMHoney：具有大型语言模型驱动动态响应生成的实时SSH蜜罐 cs.CR

7 Pages

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01463v1) [paper-pdf](http://arxiv.org/pdf/2509.01463v1)

**Authors**: Pranjay Malhotra

**Abstract**: Cybersecurity honeypots are deception tools for engaging attackers and gather intelligence, but traditional low or medium-interaction honeypots often rely on static, pre-scripted interactions that can be easily identified by skilled adversaries. This Report presents LLMHoney, an SSH honeypot that leverages Large Language Models (LLMs) to generate realistic, dynamic command outputs in real time. LLMHoney integrates a dictionary-based virtual file system to handle common commands with low latency while using LLMs for novel inputs, achieving a balance between authenticity and performance. We implemented LLMHoney using open-source LLMs and evaluated it on a testbed with 138 representative Linux commands. We report comprehensive metrics including accuracy (exact-match, Cosine Similarity, Jaro-Winkler Similarity, Levenshtein Similarity and BLEU score), response latency and memory overhead. We evaluate LLMHoney using multiple LLM backends ranging from 0.36B to 3.8B parameters, including both open-source models and a proprietary model(Gemini). Our experiments compare 13 different LLM variants; results show that Gemini-2.0 and moderately-sized models Qwen2.5:1.5B and Phi3:3.8B provide the most reliable and accurate responses, with mean latencies around 3 seconds, whereas smaller models often produce incorrect or out-of-character outputs. We also discuss how LLM integration improves honeypot realism and adaptability compared to traditional honeypots, as well as challenges such as occasional hallucinated outputs and increased resource usage. Our findings demonstrate that LLM-driven honeypots are a promising approach to enhance attacker engagement and collect richer threat intelligence.

摘要: 网络安全蜜罐是用于吸引攻击者和收集情报的欺骗工具，但传统的低或中等交互蜜罐通常依赖于静态的、预先脚本化的交互，这些交互可以被熟练的对手轻松识别。本报告介绍了LLMHoney，这是一个SSH蜜罐，利用大型语言模型（LLM）实时生成真实、动态的命令输出。LLMHoney集成了基于字典的虚拟文件系统，以低延迟处理常见命令，同时使用LLM进行新型输入，实现真实性和性能之间的平衡。我们使用开源LLM实现了LLMHoney，并在具有138个代表性的Linux命令的测试床上对其进行了评估。我们报告了全面的指标，包括准确性（精确匹配、Cosine相似性、Jaro-Winkler相似性、Levenshtein相似性和BLEU评分）、响应延迟和内存负载。我们使用从0.36B到3.8B参数的多个LLM后台来评估LLMHoney，包括开源模型和专有模型（Gemini）。我们的实验比较了13种不同的LLM变体;结果表明，Gemini-2.0和中等大小的模型Qwen 2.5：1.5B和Phi 3：3.8B提供了最可靠和准确的响应，平均延迟时间约为3秒，而较小的模型通常会产生错误或不合字符的输出。我们还讨论了与传统蜜罐相比，LLM集成如何提高蜜罐的真实性和适应性，以及偶尔幻觉输出和资源使用增加等挑战。我们的研究结果表明，LLM驱动的蜜罐是增强攻击者参与度和收集更丰富威胁情报的一种有希望的方法。



## **23. MEGen: Generative Backdoor into Large Language Models via Model Editing**

MEGen：通过模型编辑进入大型语言模型的生成后门 cs.CL

ACL 2025 Findings

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2408.10722v2) [paper-pdf](http://arxiv.org/pdf/2408.10722v2)

**Authors**: Jiyang Qiu, Xinbei Ma, Zhuosheng Zhang, Hai Zhao, Yun Li, Qianren Wang

**Abstract**: Large language models (LLMs) have exhibited remarkable versatility and adaptability, while their widespread adoption across various applications also raises critical safety concerns. This paper focuses on the impact of backdoored LLMs. Traditional backdoor injection methods are primarily limited to yes-or-no discriminative tasks, leading users to underestimate the potential risks of backdoored LLMs. Given the inherently generative nature of LLMs, this paper reveals that a generative backdoor injected into LLMs can expose the true safety risks in their applications. We propose an editing-based generative backdoor, named MEGen, aiming to expand the backdoor to generative tasks in a unified format of any text-to any text, leading to natural generations with a specific intention. Experiments show that MEGen achieves a high attack success rate by adjusting only a small set of local parameters with few-shot samples. Notably, we show that the backdoored model, when triggered, can freely output pre-set dangerous information while completing downstream tasks. Our work highlights that MEGen enables backdoors in LLMs to exhibit generative capabilities, causing potential safety risks by altering the generative style. The code is available at https://github.com/MonoQ-hub/MEGen.

摘要: 大型语言模型（LLM）表现出非凡的通用性和适应性，而它们在各种应用程序中的广泛采用也引发了关键的安全问题。本文重点讨论了后门的LLM的影响。传统的后门注入方法主要局限于“是或否”区分性任务，导致用户低估了后门LLM的潜在风险。鉴于LLM固有的生成性，本文揭示了注入LLM的生成性后门可能会暴露其应用中的真正安全风险。我们提出了一个基于编辑的生成后门，名为MEGen，旨在将后门扩展到任何文本的统一格式的生成任务，从而产生具有特定意图的自然生成。实验表明，MEGen通过仅用少量样本调整一小组局部参数即可实现高攻击成功率。值得注意的是，我们表明，后门模型在被触发时可以在完成下游任务的同时自由输出预设的危险信息。我们的工作强调，MEGen使LLM中的后门能够展现生成能力，从而通过改变生成风格而带来潜在的安全风险。该代码可在https://github.com/MonoQ-hub/MEGen上获取。



## **24. Strata-Sword: A Hierarchical Safety Evaluation towards LLMs based on Reasoning Complexity of Jailbreak Instructions**

Strata-Sword：基于越狱指令推理复杂性的LLM分层安全评估 cs.CY

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01444v1) [paper-pdf](http://arxiv.org/pdf/2509.01444v1)

**Authors**: Shiji Zhao, Ranjie Duan, Jiexi Liu, Xiaojun Jia, Fengxiang Wang, Cheng Wei, Ruoxi Cheng, Yong Xie, Chang Liu, Qing Guo, Jialing Tao, Hui Xue, Xingxing Wei

**Abstract**: Large language models (LLMs) have gained widespread recognition for their superior comprehension and have been deployed across numerous domains. Building on Chain-of-Thought (CoT) ideology, Large Reasoning models (LRMs) further exhibit strong reasoning skills, enabling them to infer user intent more accurately and respond appropriately. However, both LLMs and LRMs face the potential safety risks under jailbreak attacks, which raise concerns about their safety capabilities. Current safety evaluation methods often focus on the content dimensions, or simply aggregate different attack methods, lacking consideration of the complexity. In fact, instructions of different complexity can reflect the different safety capabilities of the model: simple instructions can reflect the basic values of the model, while complex instructions can reflect the model's ability to deal with deeper safety risks. Therefore, a comprehensive benchmark needs to be established to evaluate the safety performance of the model in the face of instructions of varying complexity, which can provide a better understanding of the safety boundaries of the LLMs. Thus, this paper first quantifies "Reasoning Complexity" as an evaluable safety dimension and categorizes 15 jailbreak attack methods into three different levels according to the reasoning complexity, establishing a hierarchical Chinese-English jailbreak safety benchmark for systematically evaluating the safety performance of LLMs. Meanwhile, to fully utilize unique language characteristics, we first propose some Chinese jailbreak attack methods, including the Chinese Character Disassembly attack, Lantern Riddle attack, and Acrostic Poem attack. A series of experiments indicate that current LLMs and LRMs show different safety boundaries under different reasoning complexity, which provides a new perspective to develop safer LLMs and LRMs.

摘要: 大型语言模型（LLM）因其卓越的理解能力而获得广泛认可，并已部署在众多领域。基于思想链（CoT）意识形态，大型推理模型（LRM）进一步展现出强大的推理能力，使它们能够更准确地推断用户意图并做出适当的响应。然而，LLM和LRM在越狱攻击下都面临潜在的安全风险，这引发了对其安全能力的担忧。目前的安全评估方法往往关注内容维度，或者简单地聚合不同的攻击方法，缺乏对复杂性的考虑。事实上，不同复杂性的指令可以反映模型不同的安全能力：简单的指令可以反映模型的基本价值观，而复杂的指令可以反映模型应对更深层次安全风险的能力。因此，需要建立一个全面的基准来评估模型在复杂性不同的指令下的安全性能，这可以更好地理解LLM的安全边界。因此，本文首先将“推理复杂性”量化为可评估的安全维度，并根据推理复杂性将15种越狱攻击方法分为三个不同级别，建立分层的中英越狱安全基准，用于系统评估LLM的安全性能。同时，为了充分利用独特的语言特征，我们首先提出了一些中文越狱攻击方法，包括中文拼音攻击、灯笼谜语攻击和希腊诗攻击。一系列实验表明，当前的LLM和LRM在不同的推理复杂度下表现出不同的安全边界，这为开发更安全的LLM和LRM提供了新的视角。



## **25. An Automated Attack Investigation Approach Leveraging Threat-Knowledge-Augmented Large Language Models**

利用威胁知识增强大型语言模型的自动攻击调查方法 cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01271v1) [paper-pdf](http://arxiv.org/pdf/2509.01271v1)

**Authors**: Rujie Dai, Peizhuo Lv, Yujiang Gui, Qiujian Lv, Yuanyuan Qiao, Yan Wang, Degang Sun, Weiqing Huang, Yingjiu Li, XiaoFeng Wang

**Abstract**: Advanced Persistent Threats (APTs) are prolonged, stealthy intrusions by skilled adversaries that compromise high-value systems to steal data or disrupt operations. Reconstructing complete attack chains from massive, heterogeneous logs is essential for effective attack investigation, yet existing methods suffer from poor platform generality, limited generalization to evolving tactics, and an inability to produce analyst-ready reports. Large Language Models (LLMs) offer strong semantic understanding and summarization capabilities, but in this domain they struggle to capture the long-range, cross-log dependencies critical for accurate reconstruction.   To solve these problems, we present an LLM-empowered attack investigation framework augmented with a dynamically adaptable Kill-Chain-aligned threat knowledge base. We organizes attack-relevant behaviors into stage-aware knowledge units enriched with semantic annotations, enabling the LLM to iteratively retrieve relevant intelligence, perform causal reasoning, and progressively expand the investigation context. This process reconstructs multi-phase attack scenarios and generates coherent, human-readable investigation reports. Evaluated on 15 attack scenarios spanning single-host and multi-host environments across Windows and Linux (over 4.3M log events, 7.2 GB of data), the system achieves an average True Positive Rate (TPR) of 97.1% and an average False Positive Rate (FPR) of 0.2%, significantly outperforming the SOTA method ATLAS, which achieves an average TPR of 79.2% and an average FPR of 29.1%.

摘要: 高级持续性威胁（APT）是技术精湛的对手发起的长期、秘密入侵，这些入侵会危及高价值系统以窃取数据或扰乱运营。从大量、异类的日志中重建完整的攻击链对于有效的攻击调查至关重要，但现有方法存在平台通用性较差、对不断发展的策略的概括性有限以及无法生成可供分析师使用的报告的问题。大型语言模型（LLM）提供强大的语义理解和总结能力，但在该领域，它们很难捕捉对准确重建至关重要的长期、跨日志依赖关系。   为了解决这些问题，我们提出了一个LLM授权的攻击调查框架，该框架增强了动态自适应的杀戮链对齐威胁知识库。我们将与攻击相关的行为组织到富含语义注释的阶段感知知识单元中，使LLM能够迭代地检索相关情报、执行因果推理并逐步扩展调查上下文。该过程重建多阶段攻击场景并生成连贯、人类可读的调查报告。评估了跨Windows和Linux的单主机和多主机环境的15种攻击场景（超过430万个日志事件，7.2 GB数据），该系统的平均真阳性率（TPR）为97.1%，平均假阳性率（FPR）为0.2%，显着优于SOTA方法ATLAS，平均TPR为79.2%，平均FPR为29.1%。



## **26. One Shot Dominance: Knowledge Poisoning Attack on Retrieval-Augmented Generation Systems**

一枪优势：对检索增强生成系统的知识中毒攻击 cs.CR

15pages, 4 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2505.11548v3) [paper-pdf](http://arxiv.org/pdf/2505.11548v3)

**Authors**: Zhiyuan Chang, Mingyang Li, Xiaojun Jia, Junjie Wang, Yuekai Huang, Ziyou Jiang, Yang Liu, Qing Wang

**Abstract**: Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG) have shown improved performance in generating accurate responses. However, the dependence on external knowledge bases introduces potential security vulnerabilities, particularly when these knowledge bases are publicly accessible and modifiable. While previous studies have exposed knowledge poisoning risks in RAG systems, existing attack methods suffer from critical limitations: they either require injecting multiple poisoned documents (resulting in poor stealthiness) or can only function effectively on simplistic queries (limiting real-world applicability). This paper reveals a more realistic knowledge poisoning attack against RAG systems that achieves successful attacks by poisoning only a single document while remaining effective for complex multi-hop questions involving complex relationships between multiple elements. Our proposed AuthChain address three challenges to ensure the poisoned documents are reliably retrieved and trusted by the LLM, even against large knowledge bases and LLM's own knowledge. Extensive experiments across six popular LLMs demonstrate that AuthChain achieves significantly higher attack success rates while maintaining superior stealthiness against RAG defense mechanisms compared to state-of-the-art baselines.

摘要: 使用检索增强生成（RAG）增强的大型语言模型（LLM）在生成准确响应方面表现出更好的性能。然而，对外部知识库的依赖会带来潜在的安全漏洞，特别是当这些知识库可公开访问和可修改时。虽然之前的研究暴露了RAG系统中的知识中毒风险，但现有的攻击方法存在严重局限性：它们要么需要注入多个有毒文档（导致隐蔽性较差），要么只能在简单化的查询上有效发挥作用（限制现实世界的适用性）。本文揭示了一种针对RAG系统的更现实的知识中毒攻击，该攻击通过仅毒害单个文档来实现成功攻击，同时对涉及多个元素之间复杂关系的复杂多跳问题仍然有效。我们提出的AuthChain解决了三个挑战，以确保LLM可靠地检索和信任受毒害的文档，即使面对大型知识库和LLM自己的知识。针对六种流行的LLM的广泛实验表明，与最先进的基线相比，AuthChain实现了显着更高的攻击成功率，同时保持了针对RAG防御机制的卓越隐蔽性。



## **27. Clone What You Can't Steal: Black-Box LLM Replication via Logit Leakage and Distillation**

克隆无法窃取的内容：通过Logit泄漏和蒸馏进行黑匣子LLM复制 cs.CR

8 pages. Accepted for publication in the proceedings of 7th IEEE  International Conference on Trust, Privacy and Security in Intelligent  Systems, and Applications (IEEE TPS 2025)

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00973v1) [paper-pdf](http://arxiv.org/pdf/2509.00973v1)

**Authors**: Kanchon Gharami, Hansaka Aluvihare, Shafika Showkat Moni, Berker Peköz

**Abstract**: Large Language Models (LLMs) are increasingly deployed in mission-critical systems, facilitating tasks such as satellite operations, command-and-control, military decision support, and cyber defense. Many of these systems are accessed through application programming interfaces (APIs). When such APIs lack robust access controls, they can expose full or top-k logits, creating a significant and often overlooked attack surface. Prior art has mainly focused on reconstructing the output projection layer or distilling surface-level behaviors. However, regenerating a black-box model under tight query constraints remains underexplored. We address that gap by introducing a constrained replication pipeline that transforms partial logit leakage into a functional deployable substitute model clone. Our two-stage approach (i) reconstructs the output projection matrix by collecting top-k logits from under 10k black-box queries via singular value decomposition (SVD) over the logits, then (ii) distills the remaining architecture into compact student models with varying transformer depths, trained on an open source dataset. A 6-layer student recreates 97.6% of the 6-layer teacher model's hidden-state geometry, with only a 7.31% perplexity increase, and a 7.58 Negative Log-Likelihood (NLL). A 4-layer variant achieves 17.1% faster inference and 18.1% parameter reduction with comparable performance. The entire attack completes in under 24 graphics processing unit (GPU) hours and avoids triggering API rate-limit defenses. These results demonstrate how quickly a cost-limited adversary can clone an LLM, underscoring the urgent need for hardened inference APIs and secure on-premise defense deployments.

摘要: 大型语言模型（LLM）越来越多地部署在关键任务系统中，促进卫星操作、指挥与控制、军事决策支持和网络防御等任务。其中许多系统都是通过应用程序编程接口（API）访问的。当此类API缺乏强大的访问控制时，它们可能会暴露完整或顶级k日志，从而创建一个重要且经常被忽视的攻击面。现有技术主要集中在重建输出投影层或提取表面级行为。然而，在严格的查询约束下重新生成黑匣子模型仍然没有得到充分的探索。我们通过引入一个受约束的复制管道来解决这一差距，该管道将部分logit泄漏转换为功能性可部署的替代模型克隆。我们的两阶段方法（i）通过对logit进行奇异值分解（DID）从10 k以下的黑匣子查询中收集前k个logit来重建输出投影矩阵，然后（ii）将剩余的架构提炼成具有不同Transformer深度的紧凑学生模型，在开源数据集上训练。一个6层的学生可以重建97.6%的6层教师模型的隐藏状态几何，而困惑度只增加了7.31%，负对数似然（NLL）为7.58。4层变体在相当的性能下实现了17.1%的推理速度和18.1%的参数减少。整个攻击只需不到24个图形处理单元（图形处理单元）小时即可完成，并避免触发API速率限制防御。这些结果展示了成本有限的对手可以多快地克隆LLM，凸显了对强化推理API和安全本地防御部署的迫切需求。



## **28. Membership Inference Attacks on Large-Scale Models: A Survey**

对大规模模型的成员推断攻击：一项调查 cs.LG

Preprint. Submitted for peer review. The final version may differ

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2503.19338v3) [paper-pdf](http://arxiv.org/pdf/2503.19338v3)

**Authors**: Hengyu Wu, Yang Cao

**Abstract**: As large-scale models such as Large Language Models (LLMs) and Large Multimodal Models (LMMs) see increasing deployment, their privacy risks remain underexplored. Membership Inference Attacks (MIAs), which reveal whether a data point was used in training the target model, are an important technique for exposing or assessing privacy risks and have been shown to be effective across diverse machine learning algorithms. However, despite extensive studies on MIAs in classic models, there remains a lack of systematic surveys addressing their effectiveness and limitations in large-scale models. To address this gap, we provide the first comprehensive review of MIAs targeting LLMs and LMMs, analyzing attacks by model type, adversarial knowledge, and strategy. Unlike prior surveys, we further examine MIAs across multiple stages of the model pipeline, including pre-training, fine-tuning, alignment, and Retrieval-Augmented Generation (RAG). Finally, we identify open challenges and propose future research directions for strengthening privacy resilience in large-scale models.

摘要: 随着大型语言模型（LLM）和大型多模式模型（LSYS）等大规模模型的部署不断增加，但其隐私风险仍然未得到充分研究。成员推断攻击（MIA）揭示了数据点是否用于训练目标模型，是暴露或评估隐私风险的重要技术，并已被证明在各种机器学习算法中有效。然而，尽管对经典模型中的MIA进行了广泛的研究，但仍然缺乏系统性的调查来解决其在大规模模型中的有效性和局限性。为了弥补这一差距，我们首次对针对LLM和LSYS的MIA进行了全面审查，按模型类型、对抗性知识和策略分析攻击。与之前的调查不同，我们进一步检查了模型管道的多个阶段的MIA，包括预训练、微调、对齐和检索增强生成（RAG）。最后，我们确定了开放的挑战，并提出了未来的研究方向，以加强大规模模型中的隐私弹性。



## **29. A Review of Hydrogen-Enabled Resilience Enhancement for Multi-Energy Systems**

多能源系统的氢赋能韧性增强研究综述 eess.SY

28 pages, 14 figures

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2412.19374v2) [paper-pdf](http://arxiv.org/pdf/2412.19374v2)

**Authors**: Liang Yu, Haoyu Fang, Goran Strbac, Dawei Qiu, Dong Yue, Xiaohong Guan, Gerhard P. Hancke

**Abstract**: Ensuring resilience in multi-energy systems (MESs) becomes both more urgent and more challenging due to the rising occurrence and severity of extreme events (e.g., natural disasters, extreme weather, and cyber-physical attacks). Among many measures of strengthening MES resilience, the integration of hydrogen shows exceptional potential in cross-temporal flexibility, cross-spatial flexibility, cross-sector flexibility, and black start capability. Although many hydrogen-enabled MES resilience enhancement measures have been developed, the current literature lacks a systematic overview of hydrogen-enabled resilience enhancement in MESs. To fill the research gap, this paper provides a comprehensive overview of hydrogen-enabled MES resilience enhancement. First, advantages and challenges of adopting hydrogen in MES resilience enhancement are summarized. Then, we propose a resilience enhancement framework for hydrogen-enabled MESs. Under the proposed framework, existing resilience metrics and event-oriented contingency models are summarized and discussed. Furthermore, we classify hydrogen-enabled planning measures by the types of hydrogen-related facilities and provide some insights for planning problem formulation frameworks. Moreover, we categorize the hydrogen-enabled operation enhancement measures into three operation response stages: preventive, emergency, and restoration. Finally, we identify some research gaps and point out possible future directions in aspects of comprehensive resilience metric design, temporally-correlated event-targeted scenario generation, multi-type temporal-spatial cyber-physical contingency modeling under compound extreme events, multi-network multi-timescale coordinated planning and operation, low-carbon resilient planning and operation, and large language model-assisted whole-process resilience enhancement.

摘要: 由于极端事件（例如，自然灾害、极端天气和网络物理攻击）。在增强MES韧性的众多措施中，氢的整合在跨时间灵活性、跨空间灵活性、跨行业灵活性和黑启动能力方面表现出了非凡的潜力。尽管已经开发了许多氢启动的MES弹性增强措施，但当前文献缺乏对MES中氢启动的弹性增强的系统概述。为了填补研究空白，本文全面概述了氢使能MES弹性增强。首先，总结了采用氢气增强MES弹性的优势和挑战。然后，我们提出了氢使能MES的弹性增强框架。在提出的框架下，总结和讨论了现有的弹性指标和面向事件的应急模型。此外，我们分类氢启用的规划措施的类型氢相关的设施，并提供一些见解规划问题制定框架。此外，我们将氢启用的操作增强措施分为三个操作响应阶段：预防，应急和恢复。最后，在综合韧性指标设计、时间相关事件情景生成、复合极端事件下多类型时空网络物理应急建模、多网络多时间尺度协同规划与运行、低碳韧性规划与运行、大语言模型辅助全流程韧性提升等方面，指出了研究差距和未来可能的发展方向。



## **30. Truthful Text Sanitization Guided by Inference Attacks**

推理攻击引导的真实文本清理 cs.CL

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2412.12928v2) [paper-pdf](http://arxiv.org/pdf/2412.12928v2)

**Authors**: Ildikó Pilán, Benet Manzanares-Salor, David Sánchez, Pierre Lison

**Abstract**: Text sanitization aims to rewrite parts of a document to prevent disclosure of personal information. The central challenge of text sanitization is to strike a balance between privacy protection (avoiding the leakage of personal information) and utility preservation (retaining as much as possible of the document's original content). To this end, we introduce a novel text sanitization method based on generalizations, that is, broader but still informative terms that subsume the semantic content of the original text spans. The approach relies on the use of instruction-tuned large language models (LLMs) and is divided into two stages. Given a document including text spans expressing personally identifiable information (PII), the LLM is first applied to obtain truth-preserving replacement candidates for each text span and rank those according to their abstraction level. Those candidates are then evaluated for their ability to protect privacy by conducting inference attacks with the LLM. Finally, the system selects the most informative replacement candidate shown to be resistant to those attacks. This two-stage process produces replacements that effectively balance privacy and utility.   We also present novel metrics to evaluate these two aspects without needing to manually annotate documents. Results on the Text Anonymization Benchmark show that the proposed approach, implemented with Mistral 7B Instruct, leads to enhanced utility, with only a marginal (< 1 p.p.) increase in re-identification risk compared to fully suppressing the original spans. Furthermore, our approach is shown to be more truth-preserving than existing methods such as Microsoft Presidio's synthetic replacements.

摘要: 文本清理旨在重写文档的部分内容，以防止个人信息泄露。文本清理的核心挑战是在隐私保护（避免个人信息泄露）和实用性保存（尽可能多地保留文档的原始内容）之间取得平衡。为此，我们引入了一种基于概括的新颖文本清理方法，即包含原始文本范围的语义内容的更广泛但仍然信息丰富的术语。该方法依赖于使用经翻译调整的大型语言模型（LLM），并分为两个阶段。给定包含表达个人可识别信息（PRI）的文本跨度的文档，LLM首先应用于获取每个文本跨度的保真替代候选项，并根据其抽象级别对这些候选项进行排名。然后评估这些候选人通过使用LLM进行推理攻击来保护隐私的能力。最后，系统选择被证明能够抵抗这些攻击的信息最丰富的替代候选项。这个两阶段的过程产生了有效平衡隐私和实用性的替代品。   我们还提出了新颖的指标来评估这两个方面，而无需手动注释文档。文本解析基准的结果表明，使用Mistral 7 B Direct实施的拟议方法可以增强实用性，仅边际（< 1 pp.）与完全抑制原始跨度相比，重新识别风险增加。此外，我们的方法被证明比微软Presidio的合成替代品等现有方法更能保存真相。



## **31. PBI-Attack: Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for Toxicity Maximization**

PBI攻击：优先引导双峰交互黑匣子越狱攻击，以实现毒性最大化 cs.CR

Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for  Toxicity Maximization

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2412.05892v4) [paper-pdf](http://arxiv.org/pdf/2412.05892v4)

**Authors**: Ruoxi Cheng, Yizhong Ding, Shuirong Cao, Ranjie Duan, Xiaoshuang Jia, Shaowei Yuan, Simeng Qin, Zhiqiang Wang, Xiaojun Jia

**Abstract**: Understanding the vulnerabilities of Large Vision Language Models (LVLMs) to jailbreak attacks is essential for their responsible real-world deployment. Most previous work requires access to model gradients, or is based on human knowledge (prompt engineering) to complete jailbreak, and they hardly consider the interaction of images and text, resulting in inability to jailbreak in black box scenarios or poor performance. To overcome these limitations, we propose a Prior-Guided Bimodal Interactive Black-Box Jailbreak Attack for toxicity maximization, referred to as PBI-Attack. Our method begins by extracting malicious features from a harmful corpus using an alternative LVLM and embedding these features into a benign image as prior information. Subsequently, we enhance these features through bidirectional cross-modal interaction optimization, which iteratively optimizes the bimodal perturbations in an alternating manner through greedy search, aiming to maximize the toxicity of the generated response. The toxicity level is quantified using a well-trained evaluation model. Experiments demonstrate that PBI-Attack outperforms previous state-of-the-art jailbreak methods, achieving an average attack success rate of 92.5% across three open-source LVLMs and around 67.3% on three closed-source LVLMs. Disclaimer: This paper contains potentially disturbing and offensive content.

摘要: 了解大型视觉语言模型（LVLM）对越狱攻击的漏洞对于负责任的现实世界部署至关重要。之前的大多数工作都需要访问模型梯度，或者基于人类知识（提示工程）来完成越狱，而且他们几乎不考虑图像和文本的交互，导致无法在黑匣子场景下越狱或性能不佳。为了克服这些限制，我们提出了一种先验引导的双峰交互式黑匣子越狱攻击，以实现毒性最大化，称为PBI攻击。我们的方法首先使用替代LVLM从有害数据库中提取恶意特征，并将这些特征嵌入到良性图像中作为先验信息。随后，我们通过双向跨模态交互优化来增强这些功能，该优化通过贪婪搜索以交替方式迭代优化双峰扰动，旨在最大化所生成响应的毒性。使用经过良好训练的评价模型量化毒性水平。实验表明，PBI-Attack的性能优于以前最先进的越狱方法，在三个开源LVLM上的平均攻击成功率为92.5%，在三个闭源LVLM上的平均攻击成功率为67.3%。免责声明：本文包含潜在的令人不安和冒犯性的内容。



## **32. The Resurgence of GCG Adversarial Attacks on Large Language Models**

GCG对大型语言模型的对抗性攻击的卷土重来 cs.CL

12 pages, 5 figures

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2509.00391v1) [paper-pdf](http://arxiv.org/pdf/2509.00391v1)

**Authors**: Yuting Tan, Xuying Li, Zhuo Li, Huizhen Shu, Peikang Hu

**Abstract**: Gradient-based adversarial prompting, such as the Greedy Coordinate Gradient (GCG) algorithm, has emerged as a powerful method for jailbreaking large language models (LLMs). In this paper, we present a systematic appraisal of GCG and its annealing-augmented variant, T-GCG, across open-source LLMs of varying scales. Using Qwen2.5-0.5B, LLaMA-3.2-1B, and GPT-OSS-20B, we evaluate attack effectiveness on both safety-oriented prompts (AdvBench) and reasoning-intensive coding prompts. Our study reveals three key findings: (1) attack success rates (ASR) decrease with model size, reflecting the increasing complexity and non-convexity of larger models' loss landscapes; (2) prefix-based heuristics substantially overestimate attack effectiveness compared to GPT-4o semantic judgments, which provide a stricter and more realistic evaluation; and (3) coding-related prompts are significantly more vulnerable than adversarial safety prompts, suggesting that reasoning itself can be exploited as an attack vector. In addition, preliminary results with T-GCG show that simulated annealing can diversify adversarial search and achieve competitive ASR under prefix evaluation, though its benefits under semantic judgment remain limited. Together, these findings highlight the scalability limits of GCG, expose overlooked vulnerabilities in reasoning tasks, and motivate further development of annealing-inspired strategies for more robust adversarial evaluation.

摘要: 基于对象的对抗提示，例如贪婪坐标梯度（GCG）算法，已成为越狱大型语言模型（LLM）的强大方法。在本文中，我们在不同规模的开源LLM中对GCG及其软化增强变体T-GCG进行了系统评估。使用Qwen 2.5 -0.5B、LLaMA-3.2-1B和GPT-OSS-20 B，我们评估了面向安全的提示（AdvBench）和推理密集型编码提示的攻击有效性。我们的研究揭示了三个关键发现：（1）攻击成功率（ASB）随着模型大小的增加而降低，反映了较大模型损失景观的复杂性和非凸性的增加;（2）与GPT-4 o语义判断相比，基于后缀的启发式算法大大高估了攻击有效性，这提供了更严格、更现实的评估;（3）与编码相关的提示比对抗性安全提示明显更容易受到攻击，这表明推理本身可以被用作攻击载体。此外，T-GCG的初步结果表明，模拟排序可以使对抗性搜索多样化，并在前置评估下实现有竞争力的ASB，尽管其在语义判断下的好处仍然有限。总而言之，这些发现凸显了GCG的可扩展性限制，暴露了推理任务中被忽视的漏洞，并激励进一步开发受退变启发的策略，以实现更稳健的对抗性评估。



## **33. Progent: Programmable Privilege Control for LLM Agents**

Progent：LLM代理的可编程特权控制 cs.CR

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2504.11703v2) [paper-pdf](http://arxiv.org/pdf/2504.11703v2)

**Authors**: Tianneng Shi, Jingxuan He, Zhun Wang, Hongwei Li, Linyu Wu, Wenbo Guo, Dawn Song

**Abstract**: LLM agents utilize Large Language Models as central components with diverse tools to complete various user tasks, but face significant security risks when interacting with external environments. Attackers can exploit these agents through various vectors, including indirect prompt injection, memory/knowledge base poisoning, and malicious tools, tricking agents into performing dangerous actions such as unauthorized financial transactions or data leakage. The core problem that enables attacks to succeed lies in over-privileged tool access. We introduce Progent, the first privilege control framework to secure LLM agents. Progent enforces security at the tool level by restricting agents to performing tool calls necessary for user tasks while blocking potentially malicious ones. Progent features a domain-specific language that allows for expressing fine-grained policies for controlling tool privileges, flexible fallback actions when calls are blocked, and dynamic policy updates to adapt to changing agent states. The framework operates deterministically at runtime, providing provable security guarantees. Thanks to our modular design, integrating Progent does not alter agent internals and only requires minimal changes to the existing agent implementation, enhancing its practicality and potential for widespread adoption. Our extensive evaluation across various agent use cases, using benchmarks like AgentDojo, ASB, and AgentPoison, demonstrates that Progent reduces attack success rates to 0%, while preserving agent utility and speed. Additionally, we show that LLMs can automatically generate effective policies, highlighting their potential for automating the process of writing Progent's security policies.

摘要: LLM代理利用大型语言模型作为中心组件，并通过不同的工具来完成各种用户任务，但在与外部环境交互时面临重大的安全风险。攻击者可以通过各种载体利用这些代理，包括间接提示注入、内存/知识库中毒和恶意工具，诱骗代理执行危险操作，例如未经授权的金融交易或数据泄露。使攻击成功的核心问题在于过度特权的工具访问。我们引入Progent，这是第一个保护LLM代理的特权控制框架。Progent通过限制代理执行用户任务所需的工具调用，同时阻止潜在的恶意工具调用来强制执行工具级别的安全性。Progent具有一种特定于域的语言，允许表达用于控制工具特权的细粒度策略、呼叫被阻止时的灵活后备操作以及用于适应不断变化的代理状态的动态策略更新。该框架在运行时确定性地运行，提供可证明的安全保证。由于我们的模块化设计，集成Progent不会改变代理内部，只需要对现有代理实现进行最小的更改，从而增强了其实用性和广泛采用的潜力。我们使用AgentDojo、ASB和AgentPoison等基准对各种代理用例进行了广泛评估，结果表明Progent将攻击成功率降低至0%，同时保留了代理效用和速度。此外，我们表明LLM可以自动生成有效的策略，凸显了它们自动化编写Progent安全策略的过程的潜力。



## **34. QGuard:Question-based Zero-shot Guard for Multi-modal LLM Safety**

QGuard：基于预设的零射击Guard，用于多模式LLM安全 cs.CR

Accept to ACLW 2025 (WOAH)

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2506.12299v2) [paper-pdf](http://arxiv.org/pdf/2506.12299v2)

**Authors**: Taegyeong Lee, Jeonghwa Yoo, Hyoungseo Cho, Soo Yong Kim, Yunho Maeng

**Abstract**: The recent advancements in Large Language Models(LLMs) have had a significant impact on a wide range of fields, from general domains to specialized areas. However, these advancements have also significantly increased the potential for malicious users to exploit harmful and jailbreak prompts for malicious attacks. Although there have been many efforts to prevent harmful prompts and jailbreak prompts, protecting LLMs from such malicious attacks remains an important and challenging task. In this paper, we propose QGuard, a simple yet effective safety guard method, that utilizes question prompting to block harmful prompts in a zero-shot manner. Our method can defend LLMs not only from text-based harmful prompts but also from multi-modal harmful prompt attacks. Moreover, by diversifying and modifying guard questions, our approach remains robust against the latest harmful prompts without fine-tuning. Experimental results show that our model performs competitively on both text-only and multi-modal harmful datasets. Additionally, by providing an analysis of question prompting, we enable a white-box analysis of user inputs. We believe our method provides valuable insights for real-world LLM services in mitigating security risks associated with harmful prompts.

摘要: 大型语言模型（LLM）的最新进展对从一般领域到专业领域的广泛领域产生了重大影响。然而，这些进步也显着增加了恶意用户利用有害和越狱提示进行恶意攻击的可能性。尽管已经做出了很多努力来防止有害提示和越狱提示，但保护LLM免受此类恶意攻击仍然是一项重要且具有挑战性的任务。在本文中，我们提出了QGuard，这是一种简单而有效的安全防范方法，利用问题提示以零攻击的方式阻止有害提示。我们的方法不仅可以保护LLM免受基于文本的有害提示攻击，还可以保护LLM免受多模式有害提示攻击。此外，通过多样化和修改警卫问题，我们的方法在不进行微调的情况下仍然对最新的有害提示保持稳健。实验结果表明，我们的模型在纯文本和多模式有害数据集上的表现都具有竞争力。此外，通过提供问题提示分析，我们可以对用户输入进行白盒分析。我们相信，我们的方法为现实世界的LLM服务提供了有价值的见解，以减轻与有害提示相关的安全风险。



## **35. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

MCA-Bench：评估CAPTCHA针对基于VLM的攻击的稳健性的多模式基准 cs.CV

we update the paper title

**SubmitDate**: 2025-08-30    [abs](http://arxiv.org/abs/2506.05982v5) [paper-pdf](http://arxiv.org/pdf/2506.05982v5)

**Authors**: Zonglin Wu, Yule Xue, Yaoyao Feng, Xiaolong Wang, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.

摘要: 随着自动攻击技术的迅速发展，验证码仍然是针对恶意机器人的重要防御机制。然而，现有的CAPTCHA方案涵盖了多种形式--从静态扭曲文本和模糊图像到交互式点击、滑动谜题和基于逻辑的问题--但社区仍然缺乏统一的、大规模的、多模式基准来严格评估其安全稳健性。为了解决这一差距，我们引入了MCA-Bench，这是一个全面且可重复的基准测试套件，可将异类CAPTCHA类型集成到单个评估协议中。利用共享的视觉语言模型主干，我们为每个CAPTCHA类别微调专门的破解剂，实现一致的跨模式评估。大量实验表明，MCA-Bench有效地绘制了现代CAPTCHA设计在不同攻击环境下的脆弱性谱，并且至关重要地提供了挑战复杂性、交互深度和模型可解性如何相互关联的首次定量分析。基于这些发现，我们提出了三项可操作的设计原则，并确定了关键的开放挑战，为系统性验证码强化、公平的基准测试和更广泛的社区合作奠定了基础。数据集和代码可在线获取。



## **36. WebInject: Prompt Injection Attack to Web Agents**

WebInject：对Web Agent的即时注入攻击 cs.LG

EMNLP 2025 main

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2505.11717v3) [paper-pdf](http://arxiv.org/pdf/2505.11717v3)

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.

摘要: 基于多模式大型语言模型（MLLM）的Web代理通过基于网页屏幕截图生成动作来与网页环境交互。在这项工作中，我们提出了WebInjects，这是一种提示注入攻击，它操纵网页环境以诱导Web代理执行攻击者指定的操作。我们的攻击对渲染网页的原始像素值添加了扰动。这些受干扰的像素被映射到屏幕截图后，扰动会导致Web代理执行攻击者指定的操作。我们将寻找扰动的任务定义为优化问题。解决这个问题的一个关键挑战是原始像素值和屏幕截图之间的映射是不可微的，因此很难将梯度反向传播到扰动。为了克服这个问题，我们训练神经网络来逼近映射，并应用投影梯度下降来解决重新制定的优化问题。对多个数据集的广泛评估表明，WebInib非常有效，并且显着优于基线。



## **37. Detecting Stealthy Data Poisoning Attacks in AI Code Generators**

检测人工智能代码生成器中的隐形数据中毒攻击 cs.CR

Accepted to the 3rd IEEE International Workshop on Reliable and  Secure AI for Software Engineering (ReSAISE, 2025), co-located with ISSRE  2025

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21636v1) [paper-pdf](http://arxiv.org/pdf/2508.21636v1)

**Authors**: Cristina Improta

**Abstract**: Deep learning (DL) models for natural language-to-code generation have become integral to modern software development pipelines. However, their heavy reliance on large amounts of data, often collected from unsanitized online sources, exposes them to data poisoning attacks, where adversaries inject malicious samples to subtly bias model behavior. Recent targeted attacks silently replace secure code with semantically equivalent but vulnerable implementations without relying on explicit triggers to launch the attack, making it especially hard for detection methods to distinguish clean from poisoned samples. We present a systematic study on the effectiveness of existing poisoning detection methods under this stealthy threat model. Specifically, we perform targeted poisoning on three DL models (CodeBERT, CodeT5+, AST-T5), and evaluate spectral signatures analysis, activation clustering, and static analysis as defenses. Our results show that all methods struggle to detect triggerless poisoning, with representation-based approaches failing to isolate poisoned samples and static analysis suffering false positives and false negatives, highlighting the need for more robust, trigger-independent defenses for AI-assisted code generation.

摘要: 用于自然语言到代码生成的深度学习（DL）模型已成为现代软件开发管道的组成部分。然而，它们严重依赖大量数据，这些数据通常是从未经清理的在线来源收集的，这使它们面临数据中毒攻击，对手会注入恶意样本以微妙地偏向模型行为。最近的有针对性的攻击以语义等效但脆弱的实现悄然取代安全代码，而不依赖显式触发器来发起攻击，这使得检测方法特别难以区分干净样本和有毒样本。我们对这种隐形威胁模型下现有中毒检测方法的有效性进行了系统研究。具体来说，我们对三种DL模型（CodeBRT、CodeT 5+、AST-T5）执行有针对性的中毒，并评估光谱特征分析、激活集群和静态分析作为防御措施。我们的结果表明，所有方法都很难检测无指示器中毒，基于表示的方法无法隔离中毒样本，静态分析会出现假阳性和假阴性，这凸显了人工智能辅助代码生成需要更强大、独立于指示器的防御。



## **38. SoK: Large Language Model-Generated Textual Phishing Campaigns End-to-End Analysis of Generation, Characteristics, and Detection**

SoK：大型语言模型生成的文本网络钓鱼活动生成、特征和检测的端到端分析 cs.CR

13 pages, 3 tables, 4 figures

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21457v1) [paper-pdf](http://arxiv.org/pdf/2508.21457v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Phishing is a pervasive form of social engineering in which attackers impersonate trusted entities to steal information or induce harmful actions. Text-based phishing dominates for its low cost, scalability, and concealability, advantages recently amplified by large language models (LLMs) that enable ``Phishing-as-a-Service'' attacks at scale within minutes. Despite the growing research into LLM-facilitated phishing attacks, consolidated systematic research on the phishing attack life cycle remains scarce. In this work, we present the first systematization of knowledge (SoK) on LLM-generated phishing, offering an end-to-end analysis that spans generation techniques, attack features, and mitigation strategies. We introduce Generation-Characterization-Defense (GenCharDef), which systematizes the ways in which LLM-generated phishing differs from traditional phishing across methodologies, security perspectives, data dependencies, and evaluation practices. This framework highlights unique challenges of LLM-driven phishing, providing a coherent foundation for understanding the evolving threat landscape and guiding the design of more resilient defenses.

摘要: 网络钓鱼是一种普遍存在的社会工程形式，攻击者冒充受信任实体来窃取信息或诱导有害行为。基于文本的网络钓鱼因其低成本、可扩展性和可隐藏性而占据主导地位，这些优势最近被大型语言模型（LLM）放大，这些模型能够在几分钟内实现“网络钓鱼即服务”攻击。尽管对LLM支持的网络钓鱼攻击的研究越来越多，但对网络钓鱼攻击生命周期的综合系统研究仍然很少。在这项工作中，我们首次介绍了LLM生成的网络钓鱼知识系统化（SoK），提供涵盖生成技术、攻击特征和缓解策略的端到端分析。我们引入了生成-特征-防御（GenCharDef），它系统化了LLM生成的网络钓鱼在方法论、安全角度、数据依赖性和评估实践方面与传统网络钓鱼的不同之处。该框架强调了LLM驱动的网络钓鱼的独特挑战，为了解不断变化的威胁格局并指导设计更具弹性的防御系统提供了连贯的基础。



## **39. Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review**

发布到Perish：对LLM辅助同行评审的即时注入攻击 cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.20863v2) [paper-pdf](http://arxiv.org/pdf/2508.20863v2)

**Authors**: Matteo Gioele Collu, Umberto Salviati, Roberto Confalonieri, Mauro Conti, Giovanni Apruzzese

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.

摘要: 大型语言模型（LLM）越来越多地融入到科学同行评审过程中，这引发了有关其可靠性和操纵弹性的新问题。在这项工作中，我们调查了隐藏的提示注入攻击的可能性，即作者在论文的PDF中嵌入对抗性文本以影响LLM生成的评论。我们首先正式化三种不同的威胁模型，这些模型设想攻击者具有不同的动机--并非所有这些都暗示着恶意意图。对于每个威胁模型，我们设计了对抗性提示，这些提示对人类读者来说是不可见的，但可以将LLM的输出引导到作者想要的结果。通过对领域学者的用户研究，我们得出了四个代表性的审查提示，用于从法学硕士那里获得同行审查。然后，我们评估对抗提示在（i）不同的审查提示、（ii）不同的基于LLM的商业系统和（iii）不同的同行评审论文中的稳健性。我们的结果表明，对抗性提示可以可靠地误导LLM，有时会对“诚实但懒惰”的评论者产生不利影响。最后，我们提出并根据经验评估了在自动内容检查下降低对抗性提示可检测性的方法。



## **40. A Whole New World: Creating a Parallel-Poisoned Web Only AI-Agents Can See**

一个全新的世界：创建一个只有人工智能代理才能看到的走廊中毒网络 cs.CR

10 pages, 1 figure

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2509.00124v1) [paper-pdf](http://arxiv.org/pdf/2509.00124v1)

**Authors**: Shaked Zychlinski

**Abstract**: This paper introduces a novel attack vector that leverages website cloaking techniques to compromise autonomous web-browsing agents powered by Large Language Models (LLMs). As these agents become more prevalent, their unique and often homogenous digital fingerprints - comprising browser attributes, automation framework signatures, and network characteristics - create a new, distinguishable class of web traffic. The attack exploits this fingerprintability. A malicious website can identify an incoming request as originating from an AI agent and dynamically serve a different, "cloaked" version of its content. While human users see a benign webpage, the agent is presented with a visually identical page embedded with hidden, malicious instructions, such as indirect prompt injections. This mechanism allows adversaries to hijack agent behavior, leading to data exfiltration, malware execution, or misinformation propagation, all while remaining completely invisible to human users and conventional security crawlers. This work formalizes the threat model, details the mechanics of agent fingerprinting and cloaking, and discusses the profound security implications for the future of agentic AI, highlighting the urgent need for robust defenses against this stealthy and scalable attack.

摘要: 本文介绍了一种新的攻击向量，利用网站伪装技术，以妥协自主的Web浏览代理大语言模型（LLM）。随着这些代理变得越来越普遍，其独特且通常同质的数字指纹-包括浏览器属性，自动化框架签名和网络特征-创建了一个新的，可区分的Web流量类别。攻击利用了这种指纹识别能力。恶意网站可以将传入的请求识别为来自AI代理，并动态提供其内容的不同“隐藏”版本。当人类用户看到良性网页时，代理会呈现一个视觉上相同的页面，其中嵌入了隐藏的恶意指令，例如间接提示注入。该机制允许对手劫持代理行为，导致数据泄露、恶意软件执行或错误信息传播，同时人类用户和传统安全爬虫仍然完全不可见。这项工作正式化了威胁模型，详细介绍了代理指纹识别和隐身的机制，并讨论了代理人工智能未来的深刻安全影响，强调了针对这种隐形和可扩展攻击的强大防御的迫切需要。



## **41. Lethe: Purifying Backdoored Large Language Models with Knowledge Dilution**

Lethe：通过知识稀释净化后门大型语言模型 cs.CL

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21004v1) [paper-pdf](http://arxiv.org/pdf/2508.21004v1)

**Authors**: Chen Chen, Yuchen Sun, Jiaxin Gao, Xueluan Gong, Qian Wang, Ziyao Wang, Yongsen Zheng, Kwok-Yan Lam

**Abstract**: Large language models (LLMs) have seen significant advancements, achieving superior performance in various Natural Language Processing (NLP) tasks. However, they remain vulnerable to backdoor attacks, where models behave normally for standard queries but generate harmful responses or unintended output when specific triggers are activated. Existing backdoor defenses either lack comprehensiveness, focusing on narrow trigger settings, detection-only mechanisms, and limited domains, or fail to withstand advanced scenarios like model-editing-based, multi-trigger, and triggerless attacks. In this paper, we present LETHE, a novel method to eliminate backdoor behaviors from LLMs through knowledge dilution using both internal and external mechanisms. Internally, LETHE leverages a lightweight dataset to train a clean model, which is then merged with the backdoored model to neutralize malicious behaviors by diluting the backdoor impact within the model's parametric memory. Externally, LETHE incorporates benign and semantically relevant evidence into the prompt to distract LLM's attention from backdoor features. Experimental results on classification and generation domains across 5 widely used LLMs demonstrate that LETHE outperforms 8 state-of-the-art defense baselines against 8 backdoor attacks. LETHE reduces the attack success rate of advanced backdoor attacks by up to 98% while maintaining model utility. Furthermore, LETHE has proven to be cost-efficient and robust against adaptive backdoor attacks.

摘要: 大型语言模型（LLM）取得了重大进步，在各种自然语言处理（NLP）任务中实现了卓越的性能。然而，它们仍然容易受到后门攻击，其中模型对于标准查询表现正常，但在激活特定触发器时会生成有害响应或意外输出。现有的后门防御要么缺乏全面性，专注于狭窄的触发设置、仅检测机制和有限的域，要么无法抵御基于模型编辑、多触发和无触发攻击等高级场景。在本文中，我们提出了LETHE，这是一种利用内部和外部机制通过知识稀释消除LLM后门行为的新型方法。在内部，LETHE利用轻量级数据集来训练干净的模型，然后将其与后门模型合并，通过稀释模型参数记忆中的后门影响来中和恶意行为。从外部来看，LETHE将良性且语义相关的证据融入到提示中，以转移LLM对后门功能的注意力。对5个广泛使用的LLM的分类和生成域的实验结果表明，LETHE针对8种后门攻击的性能优于8种最先进的防御基线。LETHE将高级后门攻击的攻击成功率降低高达98%，同时保持模型实用性。此外，LETHE已被证明具有成本效益，并且可以抵御自适应后门攻击。



## **42. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

EmantSleuth：通过语义意图不变性检测提示注入 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20890v1) [paper-pdf](http://arxiv.org/pdf/2508.20890v1)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.

摘要: 大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，从虚拟助手到自治代理。然而，它们的灵活性也引入了新的攻击向量，特别是提示注入（PI），其中攻击者通过精心制作的输入操纵模型行为。随着攻击者不断地使用释义、混淆甚至多任务注入策略，现有的基准不再足以捕获所有新兴威胁。   为了解决这一差距，我们构建了一个新的基准，系统地扩展了以前的努力。我们的基准涵盖了两种广泛使用的现有基准，同时引入了新的操纵技术和多任务场景，从而提供了更全面的评估设置。我们发现，现有的防御虽然在原始基准上有效，但在我们的基准下表现出明显的弱点，这凸显了对更强大解决方案的需求。我们的关键见解是，虽然攻击形式可能会有所不同，但对手的意图（注入未经授权的任务）保持不变。在这一观察的基础上，我们提出了EmittSleuth，这是一个面向语义的防御框架，它通过对任务级意图而不是表面特征进行推理来检测提示注入。在最先进的基准测试中进行评估后，AktSleuth始终优于现有的防御，同时保持相当的运行时间和成本效率。这些结果表明，基于意图的语义推理提供了一个强大的，有效的，和可推广的策略，以抵御不断发展的即时注入威胁的LLM。



## **43. Multi-Agent Penetration Testing AI for the Web**

Web多代理渗透测试人工智能 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20816v1) [paper-pdf](http://arxiv.org/pdf/2508.20816v1)

**Authors**: Isaac David, Arthur Gervais

**Abstract**: AI-powered development platforms are making software creation accessible to a broader audience, but this democratization has triggered a scalability crisis in security auditing. With studies showing that up to 40% of AI-generated code contains vulnerabilities, the pace of development now vastly outstrips the capacity for thorough security assessment.   We present MAPTA, a multi-agent system for autonomous web application security assessment that combines large language model orchestration with tool-grounded execution and end-to-end exploit validation. On the 104-challenge XBOW benchmark, MAPTA achieves 76.9% overall success with perfect performance on SSRF and misconfiguration vulnerabilities, 83% success on broken authorization, and strong results on injection attacks including server-side template injection (85%) and SQL injection (83%). Cross-site scripting (57%) and blind SQL injection (0%) remain challenging. Our comprehensive cost analysis across all challenges totals $21.38 with a median cost of $0.073 for successful attempts versus $0.357 for failures. Success correlates strongly with resource efficiency, enabling practical early-stopping thresholds at approximately 40 tool calls or $0.30 per challenge.   MAPTA's real-world findings are impactful given both the popularity of the respective scanned GitHub repositories (8K-70K stars) and MAPTA's low average operating cost of $3.67 per open-source assessment: MAPTA discovered critical vulnerabilities including RCEs, command injections, secret exposure, and arbitrary file write vulnerabilities. Findings are responsibly disclosed, 10 findings are under CVE review.

摘要: 人工智能驱动的开发平台正在使更广泛的受众可以访问软件创建，但这种民主化引发了安全审计的可扩展性危机。研究表明，高达40%的人工智能生成的代码包含漏洞，现在的开发速度远远超过了全面安全评估的能力。   我们提出了MAPTA，一个多代理系统的自主Web应用程序的安全评估，结合大型语言模型编排与工具接地执行和端到端的利用验证。在104个挑战XBOW基准测试中，MAPTA取得了76.9%的总体成功，在SSRF和错误配置漏洞方面表现完美，在授权破坏方面成功83%，在注入攻击（包括服务器端模板注入）（85%）和SQL注入（83%）方面取得了强劲的结果。跨站点脚本（57%）和盲目SQL注入（0%）仍然具有挑战性。我们对所有挑战的全面成本分析总计21.38美元，成功尝试的平均成本为0.073美元，失败的平均成本为0.357美元。成功与资源效率密切相关，实际的提前停止阈值为约40次工具调用或每次挑战0.30美元。   鉴于各自扫描的GitHub存储库（8 K-70 K星）的受欢迎程度以及MAPTA每次开源评估的平均运营成本较低（3.67美元），MAPTA的现实世界调查结果具有影响力：MAPTA发现了关键漏洞，包括RCE、命令注入、秘密暴露和任意文件写入漏洞。调查结果得到负责任地披露，10项调查结果正在接受CVS审查。



## **44. Addressing Tokenization Inconsistency in Steganography and Watermarking Based on Large Language Models**

基于大型语言模型解决隐写和水印中的标记化不一致性 cs.CL

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20718v1) [paper-pdf](http://arxiv.org/pdf/2508.20718v1)

**Authors**: Ruiyi Yan, Yugo Murawaki

**Abstract**: Large language models have significantly enhanced the capacities and efficiency of text generation. On the one hand, they have improved the quality of text-based steganography. On the other hand, they have also underscored the importance of watermarking as a safeguard against malicious misuse. In this study, we focus on tokenization inconsistency (TI) between Alice and Bob in steganography and watermarking, where TI can undermine robustness. Our investigation reveals that the problematic tokens responsible for TI exhibit two key characteristics: infrequency and temporariness. Based on these findings, we propose two tailored solutions for TI elimination: a stepwise verification method for steganography and a post-hoc rollback method for watermarking. Experiments show that (1) compared to traditional disambiguation methods in steganography, directly addressing TI leads to improvements in fluency, imperceptibility, and anti-steganalysis capacity; (2) for watermarking, addressing TI enhances detectability and robustness against attacks.

摘要: 大型语言模型显着增强了文本生成的能力和效率。一方面，他们提高了基于文本的隐写术的质量。另一方面，他们还强调了水印作为防止恶意滥用的重要性。在这项研究中，我们重点关注Alice和Bob之间在隐写术和水印中的标记化不一致性（TI），其中TI可能会破坏鲁棒性。我们的调查表明，造成TI的有问题的代币表现出两个关键特征：不频繁和临时性。基于这些发现，我们提出了两种针对TI消除的定制解决方案：用于隐写术的逐步验证方法和用于水印的事后回滚方法。实验表明：（1）与隐写术中传统的歧义消除方法相比，直接解决TI可以提高流畅性、不可感知性和抗隐写分析能力;（2）对于水印，解决TI增强了可检测性和针对攻击的鲁棒性。



## **45. Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning**

Token Buncher：保护LLM免受有害的强化学习微调 cs.LG

Project Hompage: https://tokenbuncher.github.io/

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20697v1) [paper-pdf](http://arxiv.org/pdf/2508.20697v1)

**Authors**: Weitao Feng, Lixu Wang, Tianyi Wei, Jie Zhang, Chongyang Gao, Sinong Zhan, Peizhuo Lv, Wei Dong

**Abstract**: As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.

摘要: 随着大型语言模型（LLM）的能力不断增强，通过微调导致有害误用的风险也在增加。虽然大多数先前的研究都假设攻击者依赖监督微调（SFT）来进行此类滥用，但我们系统地证明，强化学习（RL）使对手能够在匹配的计算预算下更有效地打破安全对齐并促进高级有害任务协助。为了应对这种新出现的威胁，我们提出了TokenBuncher，这是第一个专门针对基于RL的有害微调的有效防御。TokenBuncher抑制了RL所依赖的基础：模型响应不确定性。通过限制不确定性，基于RL的微调无法再利用不同的奖励信号来推动模型走向有害行为。我们通过以互赏为回报的RL和旨在防止专家域有害能力升级的Token Noiser机制来实现这种防御。跨多个模型和RL算法的大量实验表明，TokenBuncher稳健地减轻了有害的RL微调，同时保留了良性的任务效用和微调能力。我们的结果强调，基于RL的有害微调比SFT构成更大的系统性风险，并且TokenBuncher提供了有效且通用的防御。



## **46. CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics**

CyberSleuth：用于网络攻击取证的自主蓝队LLM代理 cs.CR

Code:  https://github.com/SmartData-Polito/LLM_Agent_Cybersecurity_Forensic

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20643v1) [paper-pdf](http://arxiv.org/pdf/2508.20643v1)

**Authors**: Stefano Fumero, Kai Huang, Matteo Boffa, Danilo Giordano, Marco Mellia, Zied Ben Houidi, Dario Rossi

**Abstract**: Large Language Model (LLM) agents are powerful tools for automating complex tasks. In cybersecurity, researchers have primarily explored their use in red-team operations such as vulnerability discovery and penetration tests. Defensive uses for incident response and forensics have received comparatively less attention and remain at an early stage. This work presents a systematic study of LLM-agent design for the forensic investigation of realistic web application attacks. We propose CyberSleuth, an autonomous agent that processes packet-level traces and application logs to identify the targeted service, the exploited vulnerability (CVE), and attack success. We evaluate the consequences of core design decisions - spanning tool integration and agent architecture - and provide interpretable guidance for practitioners. We benchmark four agent architectures and six LLM backends on 20 incident scenarios of increasing complexity, identifying CyberSleuth as the best-performing design. In a separate set of 10 incidents from 2025, CyberSleuth correctly identifies the exact CVE in 80% of cases. At last, we conduct a human study with 22 experts, which rated the reports of CyberSleuth as complete, useful, and coherent. They also expressed a slight preference for DeepSeek R1, a good news for open source LLM. To foster progress in defensive LLM research, we release both our benchmark and the CyberSleuth platform as a foundation for fair, reproducible evaluation of forensic agents.

摘要: 大型语言模型（LLM）代理是自动化复杂任务的强大工具。在网络安全方面，研究人员主要探索了它们在红队行动中的用途，例如漏洞发现和渗透测试。事件响应和取证的防御性用途受到的关注相对较少，并且仍处于早期阶段。这项工作对LLM代理设计进行了系统研究，用于现实Web应用程序攻击的取证调查。我们提出了CyberSleuth，这是一个自治代理，可以处理数据包级跟踪和应用程序日志，以识别目标服务、被利用的漏洞（UTE）和攻击成功。我们评估核心设计决策（跨越工具集成和代理架构）的后果，并为从业者提供可解释的指导。我们针对20个复杂性不断增加的事件场景对四种代理架构和六个LLM后台进行了基准测试，将CyberSleuth确定为性能最佳的设计。在2025年以来的另一组10起事件中，CyberSleuth在80%的案例中正确识别了确切的UTE。最后，我们与22名专家进行了一项人体研究，将CyberSleuth的报告评为完整、有用和连贯。他们还表达了对DeepSeek R1的轻微偏好，这对开源LLM来说是一个好消息。为了促进防御性LLM研究的进展，我们发布了我们的基准和CyberSleuth平台，作为对法医代理进行公平、可重复的评估的基础。



## **47. NetGPT: Generative Pretrained Transformer for Network Traffic**

NetGPT：网络流量生成式预训练Transformer cs.NI

Code is available at https://github.com/ict-net/NetGPT

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2304.09513v3) [paper-pdf](http://arxiv.org/pdf/2304.09513v3)

**Authors**: Xuying Meng, Chungang Lin, Yequan Wang, Yujun Zhang

**Abstract**: All data on the Internet are transferred by network traffic, thus accurately modeling network traffic can help improve network services quality and protect data privacy. Pretrained models for network traffic can utilize large-scale raw data to learn the essential characteristics of network traffic, and generate distinguishable results for input traffic without considering specific downstream tasks. Effective pretrained models can significantly optimize the training efficiency and effectiveness of downstream tasks, such as application classification, attack detection and traffic generation. Despite the great success of pretraining in natural language processing, there is no work in the network field. Considering the diverse demands and characteristics of network traffic and network tasks, it is non-trivial to build a pretrained model for network traffic and we face various challenges, especially the heterogeneous headers and payloads in the multi-pattern network traffic and the different dependencies for contexts of diverse downstream network tasks.   To tackle these challenges, in this paper, we make the first attempt to provide a generative pretrained model NetGPT for both traffic understanding and generation tasks. We propose the multi-pattern network traffic modeling to construct unified text inputs and support both traffic understanding and generation tasks. We further optimize the adaptation effect of the pretrained model to diversified tasks by shuffling header fields, segmenting packets in flows, and incorporating diverse task labels with prompts. With diverse traffic datasets from encrypted software, DNS, private industrial protocols and cryptocurrency mining, expensive experiments demonstrate the effectiveness of our NetGPT in a range of traffic understanding and generation tasks on traffic datasets, and outperform state-of-the-art baselines by a wide margin.

摘要: 互联网上的所有数据都是通过网络流量传输的，因此对网络流量进行准确建模有助于提高网络服务质量并保护数据隐私。预训练的网络流量模型可以利用大规模原始数据来学习网络流量的基本特征，并在不考虑特定下游任务的情况下为输入流量生成可区分的结果。有效的预训练模型可以显着优化下游任务（例如应用程序分类、攻击检测和流量生成）的训练效率和有效性。尽管预训练在自然语言处理方面取得了巨大成功，但网络领域还没有任何工作。考虑到网络流量和网络任务的不同需求和特征，为网络流量构建预训练模型并非易事，我们面临着各种挑战，特别是多模式网络流量中的异类头和有效负载以及不同下游网络任务上下文的不同依赖关系。   为了应对这些挑战，在本文中，我们首次尝试为流量理解和生成任务提供生成式预训练模型NetGPT。我们提出多模式网络流量建模来构建统一的文本输入并支持流量理解和生成任务。我们通过洗牌头字段、分割流中的数据包以及将不同任务标签与提示结合，进一步优化预训练模型对多样化任务的适应效果。凭借来自加密软件、DNS、私有工业协议和加密货币挖掘的多样化流量数据集，昂贵的实验证明了我们的NetGPT在一系列流量理解和流量数据集生成任务中的有效性，并且远远超过最先进的基线。



## **48. Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application**

多模态LLM越狱的概率建模：从量化到应用 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2503.06989v4) [paper-pdf](http://arxiv.org/pdf/2503.06989v4)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Zonghao Ying, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Quanchen Zou

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.

摘要: 最近，多模式大型语言模型（MLLM）展示了其在理解多模式内容方面的卓越能力。然而，它们仍然容易受到越狱攻击，这些攻击利用其安全调整中的弱点来产生有害反应。之前的研究根据回应是否包含恶意内容将越狱分为成功或失败。然而，考虑到MLLM响应的随机性，这种对输入越狱MLLM的能力的二元分类是不合适的。从这个观点出发，我们引入越狱概率来量化输入的越狱潜力，这代表当提示此输入时MLLM生成恶意响应的可能性。我们通过对MLLM的多次查询来估算这一可能性。使用越狱概率预测网络（JPPN）对输入隐藏状态与其相应越狱概率之间的关系进行建模后，我们使用连续越狱概率进行优化。具体来说，我们提出了基于越狱概率的攻击（JPA），该攻击优化输入图像上的对抗性扰动以最大化越狱概率，并通过包括单调文本改写进一步将其增强为多模式JPA（MJPA）。为了对抗攻击，我们还提出了基于越狱概率的微调（JPF），它通过MLLM参数更新最大限度地降低越狱概率。大量实验表明，（1）（M）JPA在白盒和黑匣子设置下攻击广泛的模型时都能产生显着的改进。(2)JPF最多将越狱人数大幅减少60%以上。上述两个结果都证明了引入越狱概率以在输入越狱能力之间进行细微差别的重要性。



## **49. CoCoNUTS: Concentrating on Content while Neglecting Uninformative Textual Styles for AI-Generated Peer Review Detection**

CoCoCoNUTS：专注于内容，而忽略无信息的文本风格，用于人工智能生成的同行评论检测 cs.CL

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2509.04460v1) [paper-pdf](http://arxiv.org/pdf/2509.04460v1)

**Authors**: Yihan Chen, Jiawei Chen, Guozhao Mo, Xuanang Chen, Ben He, Xianpei Han, Le Sun

**Abstract**: The growing integration of large language models (LLMs) into the peer review process presents potential risks to the fairness and reliability of scholarly evaluation. While LLMs offer valuable assistance for reviewers with language refinement, there is growing concern over their use to generate substantive review content. Existing general AI-generated text detectors are vulnerable to paraphrasing attacks and struggle to distinguish between surface language refinement and substantial content generation, suggesting that they primarily rely on stylistic cues. When applied to peer review, this limitation can result in unfairly suspecting reviews with permissible AI-assisted language enhancement, while failing to catch deceptively humanized AI-generated reviews. To address this, we propose a paradigm shift from style-based to content-based detection. Specifically, we introduce CoCoNUTS, a content-oriented benchmark built upon a fine-grained dataset of AI-generated peer reviews, covering six distinct modes of human-AI collaboration. Furthermore, we develop CoCoDet, an AI review detector via a multi-task learning framework, designed to achieve more accurate and robust detection of AI involvement in review content. Our work offers a practical foundation for evaluating the use of LLMs in peer review, and contributes to the development of more precise, equitable, and reliable detection methods for real-world scholarly applications. Our code and data will be publicly available at https://github.com/Y1hanChen/COCONUTS.

摘要: 大型语言模型（LLM）越来越多地融入同行评审过程，给学术评估的公平性和可靠性带来了潜在风险。虽然LLM为审查人员提供了语言细化方面的宝贵帮助，但人们越来越担心使用它们来生成实质性审查内容。现有的通用人工智能生成文本检测器很容易受到转述攻击，并且很难区分表面语言细化和实质内容生成，这表明它们主要依赖于风格线索。当应用于同行评审时，这种限制可能会导致不公平地怀疑具有允许的人工智能辅助语言增强的评论，同时无法捕捉到虚假人性化的人工智能生成的评论。为了解决这个问题，我们提出了从基于风格的检测到基于内容的检测的范式转变。具体来说，我们引入了CoCoNUTS，这是一个基于人工智能生成同行评论的细粒度数据集的面向内容的基准，涵盖了人类与人工智能协作的六种不同模式。此外，我们还开发了CoCoDet，这是一种通过多任务学习框架的人工智能评论检测器，旨在实现对人工智能参与评论内容的更准确和更稳健的检测。我们的工作为评估LLM在同行评审中的使用提供了实践基础，并有助于为现实世界的学术应用开发更精确、公平和可靠的检测方法。我们的代码和数据将在https://github.com/Y1hanChen/COCONUTS上公开。



## **50. Ransomware 3.0: Self-Composing and LLM-Orchestrated**

勒索软件3.0：自编写和LLM格式 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20444v1) [paper-pdf](http://arxiv.org/pdf/2508.20444v1)

**Authors**: Md Raz, Meet Udeshi, P. V. Sai Charan, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri

**Abstract**: Using automated reasoning, code synthesis, and contextual decision-making, we introduce a new threat that exploits large language models (LLMs) to autonomously plan, adapt, and execute the ransomware attack lifecycle. Ransomware 3.0 represents the first threat model and research prototype of LLM-orchestrated ransomware. Unlike conventional malware, the prototype only requires natural language prompts embedded in the binary; malicious code is synthesized dynamically by the LLM at runtime, yielding polymorphic variants that adapt to the execution environment. The system performs reconnaissance, payload generation, and personalized extortion, in a closed-loop attack campaign without human involvement. We evaluate this threat across personal, enterprise, and embedded environments using a phase-centric methodology that measures quantitative fidelity and qualitative coherence in each attack phase. We show that open source LLMs can generate functional ransomware components and sustain closed-loop execution across diverse environments. Finally, we present behavioral signals and multi-level telemetry of Ransomware 3.0 through a case study to motivate future development of better defenses and policy enforcements to address novel AI-enabled ransomware attacks.

摘要: 使用自动推理，代码合成和上下文决策，我们引入了一种新的威胁，该威胁利用大型语言模型（LLM）来自主规划，适应和执行勒索软件攻击生命周期。Ransomware 3.0代表了LLM-orchestrated勒索软件的第一个威胁模型和研究原型。与传统的恶意软件不同，原型只需要嵌入在二进制文件中的自然语言提示;恶意代码在运行时由LLM动态合成，产生适应执行环境的多态变体。该系统在没有人类参与的闭环攻击活动中执行侦察，有效载荷生成和个性化勒索。我们使用以阶段为中心的方法来评估个人、企业和嵌入式环境中的这一威胁，该方法测量每个攻击阶段的量化保真度和定性一致性。我们表明，开源LLM可以生成功能勒索软件组件并在不同环境中维持闭环执行。最后，我们通过案例研究展示了Ransomware 3.0的行为信号和多层遥感，以激励未来开发更好的防御和政策执行，以解决新型的支持人工智能的勒索软件攻击。



