# Latest Large Language Model Attack Papers
**update at 2025-03-21 10:39:35**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment**

免疫：通过推理时间对齐提高多模式LLM中越狱的安全性 cs.CR

Accepted to CVPR 2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2411.18688v3) [paper-pdf](http://arxiv.org/pdf/2411.18688v3)

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Ahmad Beirami, Furong Huang, Alvaro Velasquez, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks. In this work, we first highlight an important safety gap to describe that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model through controlled decoding to defend against jailbreak attacks. Additionally, we provide a mathematical characterization of Immune, offering insights on why it improves safety against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.

摘要: 随着多通道大语言模型(MLLMS)在视觉推理任务中的广泛应用，提高其安全性变得至关重要。最近的研究表明，尽管训练时间安全一致，这些模型仍然容易受到越狱攻击。在这项工作中，我们首先强调一个重要的安全差距，以描述仅通过安全培训实现的对准可能不足以抵御越狱攻击。为了解决这一漏洞，我们提出了免疫，这是一个推理时间防御框架，通过受控解码利用安全奖励模型来防御越狱攻击。此外，我们提供了免疫的数学特征，为为什么它提高了抵御越狱的安全性提供了见解。使用最近的MLLMS对不同的越狱基准进行的广泛评估表明，免疫有效地增强了模型的安全性，同时保持了模型的原始能力。例如，对于基于文本的越狱攻击LLaVA-1.6，与基本MLLM和最先进的防御策略相比，免疫分别将攻击成功率降低了57.82%和16.78%。



## **2. Robust LLM safeguarding via refusal feature adversarial training**

通过拒绝功能对抗培训强大的LLM保障 cs.LG

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.20089v2) [paper-pdf](http://arxiv.org/pdf/2409.20089v2)

**Authors**: Lei Yu, Virginie Do, Karen Hambardzumyan, Nicola Cancedda

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can elicit harmful responses. Defending against such attacks remains challenging due to the opacity of jailbreaking mechanisms and the high computational cost of training LLMs robustly. We demonstrate that adversarial attacks share a universal mechanism for circumventing LLM safeguards that works by ablating a dimension in the residual stream embedding space called the refusal feature. We further show that the operation of refusal feature ablation (RFA) approximates the worst-case perturbation of offsetting model safety. Based on these findings, we propose Refusal Feature Adversarial Training (ReFAT), a novel algorithm that efficiently performs LLM adversarial training by simulating the effect of input-level attacks via RFA. Experiment results show that ReFAT significantly improves the robustness of three popular LLMs against a wide range of adversarial attacks, with considerably less computational overhead compared to existing adversarial training methods.

摘要: 大型语言模型(LLM)很容易受到可能引起有害响应的对抗性攻击。由于越狱机制的不透明性和强大训练LLM的高计算成本，防御此类攻击仍然具有挑战性。我们证明了对抗性攻击共享一个通用的机制来规避LLM安全机制，该机制通过在剩余流嵌入空间中消融一个称为拒绝特征的维度来工作。我们进一步证明了拒绝特征消融(RFA)的操作近似于补偿模型安全性的最坏情况的扰动。基于这些发现，我们提出了拒绝特征对抗训练(Refat)，这是一种通过RFA模拟输入级攻击的效果来高效执行LLM对抗训练的新算法。实验结果表明，与现有的对抗性训练方法相比，REFAT显著地提高了三种流行的LLMS对多种对抗性攻击的健壮性，并且具有相当少的计算开销。



## **3. "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks**

“道德化”多步骤越狱预言：对大型语言模型中护栏进行黑匣子测试以进行言语攻击 cs.CR

This paper has been submitted to Nature Machine Intelligence and  OpenReview preprints. It has 7 pages of text, 3 figures, and 3 tables

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2411.16730v4) [paper-pdf](http://arxiv.org/pdf/2411.16730v4)

**Authors**: Libo Wang

**Abstract**: As the application of large language models continues to expand in various fields, it poses higher challenges to the effectiveness of identifying harmful content generation and guardrail mechanisms. This research aims to evaluate the guardrail effectiveness of GPT-4o, Grok-2 Beta, Llama 3.1 (405B), Gemini 1.5, and Claude 3.5 Sonnet through black-box testing of seemingly ethical multi-step jailbreak prompts. It conducts ethical attacks by designing an identical multi-step prompts that simulates the scenario of "corporate middle managers competing for promotions." The data results show that the guardrails of the above-mentioned LLMs were bypassed and the content of verbal attacks was generated. Claude 3.5 Sonnet's resistance to multi-step jailbreak prompts is more obvious. To ensure objectivity, the experimental process, black box test code, and enhanced guardrail code are uploaded to the GitHub repository: https://github.com/brucewang123456789/GeniusTrail.git.

摘要: 随着大型语言模型在各个领域的应用不断扩展，对识别有害内容生成和护栏机制的有效性提出了更高的挑战。这项研究旨在通过对看似合乎道德的多步越狱提示进行黑匣子测试来评估GPT-4 o、Grok-2 Beta、Llama 3.1（405 B）、Gemini 1.5和Claude 3.5十四行诗的护栏有效性。它通过设计相同的多步骤提示来进行道德攻击，模拟“企业中层管理人员竞争晋升”的场景。“数据结果显示，上述LLM的护栏被绕过，产生了言语攻击的内容。克劳德3.5十四行诗对多步越狱提示的抵制更加明显。为了确保客观性，实验过程、黑匣子测试代码和增强型护栏代码被上传到GitHub存储库：https://github.com/brucewang123456789/GeniusTrail.git。



## **4. BadToken: Token-level Backdoor Attacks to Multi-modal Large Language Models**

BadToken：对多模式大型语言模型的令牌级后门攻击 cs.CR

This paper is accepted by CVPR 2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16023v1) [paper-pdf](http://arxiv.org/pdf/2503.16023v1)

**Authors**: Zenghui Yuan, Jiawen Shi, Pan Zhou, Neil Zhenqiang Gong, Lichao Sun

**Abstract**: Multi-modal large language models (MLLMs) extend large language models (LLMs) to process multi-modal information, enabling them to generate responses to image-text inputs. MLLMs have been incorporated into diverse multi-modal applications, such as autonomous driving and medical diagnosis, via plug-and-play without fine-tuning. This deployment paradigm increases the vulnerability of MLLMs to backdoor attacks. However, existing backdoor attacks against MLLMs achieve limited effectiveness and stealthiness. In this work, we propose BadToken, the first token-level backdoor attack to MLLMs. BadToken introduces two novel backdoor behaviors: Token-substitution and Token-addition, which enable flexible and stealthy attacks by making token-level modifications to the original output for backdoored inputs. We formulate a general optimization problem that considers the two backdoor behaviors to maximize the attack effectiveness. We evaluate BadToken on two open-source MLLMs and various tasks. Our results show that our attack maintains the model's utility while achieving high attack success rates and stealthiness. We also show the real-world threats of BadToken in two scenarios, i.e., autonomous driving and medical diagnosis. Furthermore, we consider defenses including fine-tuning and input purification. Our results highlight the threat of our attack.

摘要: 多模式大型语言模型(MLLMS)扩展了大型语言模型(LLM)以处理多模式信息，使它们能够生成对图像-文本输入的响应。通过无需微调的即插即用，MLLMS已被整合到各种多模式应用中，如自动驾驶和医疗诊断。这种部署范例增加了MLLMS在后门攻击中的脆弱性。然而，现有的针对MLLMS的后门攻击取得的效果和隐蔽性有限。在这项工作中，我们提出了BadToken，这是对MLLMS的第一个令牌级后门攻击。BadToken引入了两种新的后门行为：令牌替换和令牌添加，这两种行为通过对后端输入的原始输出进行令牌级修改来实现灵活和隐蔽的攻击。我们提出了一个一般的优化问题，该问题考虑了两个后门行为，以最大化攻击效果。我们在两个开源MLLMS和不同的任务上对BadToken进行了评估。结果表明，我们的攻击在保持模型实用性的同时，获得了较高的攻击成功率和隐蔽性。我们还展示了BadToken在两个场景中的真实威胁，即自动驾驶和医疗诊断。此外，我们考虑了包括微调和输入净化在内的防御措施。我们的结果突显了我们攻击的威胁。



## **5. Differentially Private Steering for Large Language Model Alignment**

针对大型语言模型对齐的差异私人指导 cs.CL

ICLR 2025 Camera Ready; Code: https://github.com/UKPLab/iclr2025-psa

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2501.18532v2) [paper-pdf](http://arxiv.org/pdf/2501.18532v2)

**Authors**: Anmol Goel, Yaxi Hu, Iryna Gurevych, Amartya Sanyal

**Abstract**: Aligning Large Language Models (LLMs) with human values and away from undesirable behaviors (such as hallucination) has become increasingly important. Recently, steering LLMs towards a desired behavior via activation editing has emerged as an effective method to mitigate harmful generations at inference-time. Activation editing modifies LLM representations by preserving information from positive demonstrations (e.g., truthful) and minimising information from negative demonstrations (e.g., hallucinations). When these demonstrations come from a private dataset, the aligned LLM may leak private information contained in those private samples. In this work, we present the first study of aligning LLM behavior with private datasets. Our work proposes the Private Steering for LLM Alignment (PSA) algorithm to edit LLM activations with differential privacy (DP) guarantees. We conduct extensive experiments on seven different benchmarks with open-source LLMs of different sizes (0.5B to 7B) and model families (LlaMa, Qwen, Mistral and Gemma). Our results show that PSA achieves DP guarantees for LLM alignment with minimal loss in performance, including alignment metrics, open-ended text generation quality, and general-purpose reasoning. We also develop the first Membership Inference Attack (MIA) for evaluating and auditing the empirical privacy for the problem of LLM steering via activation editing. Our experiments support the theoretical guarantees by showing improved guarantees for our PSA algorithm compared to several existing non-private techniques.

摘要: 使大型语言模型(LLM)与人类价值观保持一致，并远离不良行为(如幻觉)已变得越来越重要。最近，通过激活编辑将LLM引导到期望的行为已经成为减少推理时有害生成的一种有效方法。激活编辑通过保存来自正面演示(例如，真实)的信息以及最小化来自负面演示(例如，幻觉)的信息来修改LLM表示。当这些演示来自私有数据集时，对齐的LLM可能会泄露那些私有样本中包含的私有信息。在这项工作中，我们提出了第一个将LLM行为与私有数据集对齐的研究。我们的工作提出了LLM对齐的私有转向(PSA)算法来编辑具有差分隐私(DP)保证的LLM激活。我们使用不同大小(0.5B到7B)和模型家族(骆驼、Qwen、米斯特拉尔和Gema)的开源LLM在七个不同的基准上进行了广泛的实验。我们的结果表明，PSA在性能损失最小的情况下实现了LLM对齐的DP保证，包括对齐度量、开放式文本生成质量和通用推理。我们还开发了第一个成员推理攻击(MIA)，用于通过激活编辑来评估和审计LLM操控问题的经验隐私。我们的实验表明，与现有的几种非私有技术相比，我们的PSA算法得到了更好的保证，从而支持了理论上的保证。



## **6. SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders**

SAUCE：使用稀疏自动编码器的视觉语言模型中的选择性概念消除 cs.CV

More comparative experiments are needed

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.14530v2) [paper-pdf](http://arxiv.org/pdf/2503.14530v2)

**Authors**: Qing Li, Jiahui Geng, Derui Zhu, Fengyu Cai, Chenyang Lyu, Fakhri Karray

**Abstract**: Unlearning methods for vision-language models (VLMs) have primarily adapted techniques from large language models (LLMs), relying on weight updates that demand extensive annotated forget sets. Moreover, these methods perform unlearning at a coarse granularity, often leading to excessive forgetting and reduced model utility. To address this issue, we introduce SAUCE, a novel method that leverages sparse autoencoders (SAEs) for fine-grained and selective concept unlearning in VLMs. Briefly, SAUCE first trains SAEs to capture high-dimensional, semantically rich sparse features. It then identifies the features most relevant to the target concept for unlearning. During inference, it selectively modifies these features to suppress specific concepts while preserving unrelated information. We evaluate SAUCE on two distinct VLMs, LLaVA-v1.5-7B and LLaMA-3.2-11B-Vision-Instruct, across two types of tasks: concrete concept unlearning (objects and sports scenes) and abstract concept unlearning (emotions, colors, and materials), encompassing a total of 60 concepts. Extensive experiments demonstrate that SAUCE outperforms state-of-the-art methods by 18.04% in unlearning quality while maintaining comparable model utility. Furthermore, we investigate SAUCE's robustness against widely used adversarial attacks, its transferability across models, and its scalability in handling multiple simultaneous unlearning requests. Our findings establish SAUCE as an effective and scalable solution for selective concept unlearning in VLMs.

摘要: 视觉语言模型(VLM)的遗忘方法主要采用来自大型语言模型(LLM)的技术，依赖于需要大量注释遗忘集的权重更新。此外，这些方法在粗粒度上执行遗忘，经常导致过度遗忘和降低模型效用。为了解决这个问题，我们引入了SASE，这是一种新的方法，它利用稀疏自动编码器(SAE)在VLM中进行细粒度和选择性的概念遗忘。简而言之，SASE首先训练SAE捕获高维的、语义丰富的稀疏特征。然后确定与目标概念最相关的特征以进行遗忘。在推理过程中，它有选择地修改这些特征以抑制特定概念，同时保留不相关的信息。我们在两个不同的VLM，LLaVA-v1.5-7B和Llama-3.2-11B-Vision-Indict上评估SAUE，跨越两种类型的任务：具体概念遗忘(物体和运动场景)和抽象概念遗忘(情绪、颜色和材料)，总共包含60个概念。大量的实验表明，在保持可比的模型效用的情况下，SASE在遗忘质量方面比最先进的方法高出18.04%。此外，我们还研究了SASE对广泛使用的敌意攻击的健壮性、其跨模型的可转移性以及其在处理多个并发遗忘请求时的可扩展性。我们的研究结果表明，SASE是一种有效且可扩展的解决方案，可用于解决VLMS中的选择性概念遗忘问题。



## **7. DroidTTP: Mapping Android Applications with TTP for Cyber Threat Intelligence**

DroidTTP：使用TTP映射Android应用程序以实现网络威胁情报 cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15866v1) [paper-pdf](http://arxiv.org/pdf/2503.15866v1)

**Authors**: Dincy R Arikkat, Vinod P., Rafidha Rehiman K. A., Serena Nicolazzo, Marco Arazzi, Antonino Nocera, Mauro Conti

**Abstract**: The widespread adoption of Android devices for sensitive operations like banking and communication has made them prime targets for cyber threats, particularly Advanced Persistent Threats (APT) and sophisticated malware attacks. Traditional malware detection methods rely on binary classification, failing to provide insights into adversarial Tactics, Techniques, and Procedures (TTPs). Understanding malware behavior is crucial for enhancing cybersecurity defenses. To address this gap, we introduce DroidTTP, a framework mapping Android malware behaviors to TTPs based on the MITRE ATT&CK framework. Our curated dataset explicitly links MITRE TTPs to Android applications. We developed an automated solution leveraging the Problem Transformation Approach (PTA) and Large Language Models (LLMs) to map applications to both Tactics and Techniques. Additionally, we employed Retrieval-Augmented Generation (RAG) with prompt engineering and LLM fine-tuning for TTP predictions. Our structured pipeline includes dataset creation, hyperparameter tuning, data augmentation, feature selection, model development, and SHAP-based model interpretability. Among LLMs, Llama achieved the highest performance in Tactic classification with a Jaccard Similarity of 0.9583 and Hamming Loss of 0.0182, and in Technique classification with a Jaccard Similarity of 0.9348 and Hamming Loss of 0.0127. However, the Label Powerset XGBoost model outperformed LLMs, achieving a Jaccard Similarity of 0.9893 for Tactic classification and 0.9753 for Technique classification, with a Hamming Loss of 0.0054 and 0.0050, respectively. While XGBoost showed superior performance, the narrow margin highlights the potential of LLM-based approaches in TTP classification.

摘要: Android设备广泛用于银行和通信等敏感操作，使其成为网络威胁的主要目标，特别是高级持久性威胁(APT)和复杂的恶意软件攻击。传统的恶意软件检测方法依赖于二进制分类，无法提供对敌对战术、技术和过程(TTP)的洞察。了解恶意软件行为对于加强网络安全防御至关重要。为了弥补这一差距，我们在MITRE ATT&CK框架的基础上引入了DroidTTP，一个将Android恶意软件行为映射到TTP的框架。我们精心挑选的数据集明确地将MITRE TTP链接到Android应用程序。我们开发了一个自动化解决方案，利用问题转换方法(PTA)和大型语言模型(LLM)将应用程序映射到战术和技术。此外，我们使用了具有即时工程和LLM微调的检索-增强生成(RAG)来进行TTP预测。我们的结构化流程包括数据集创建、超参数调整、数据增强、特征选择、模型开发和基于Shap的模型可解释性。在LLMS中，大羊驼在战术分类上表现最好，贾卡德相似度为0.9583，Hamming损失为0.0182；在技术分类上表现最好，Jaccard相似度为0.9348，Hamming损失为0.0127。然而，标签Powerset XGBoost模型的表现优于LLMS，战术分类的Jaccard相似度为0.9893，技术分类的Jaccard相似度为0.9753，Hamming损失分别为0.0054和0.0050。虽然XGBoost表现出了优越的性能，但狭窄的差距突显了基于LLM的方法在TTP分类中的潜力。



## **8. AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack Integration**

AutoRedTeamer：具有终身攻击集成的自主红色团队 cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15754v1) [paper-pdf](http://arxiv.org/pdf/2503.15754v1)

**Authors**: Andy Zhou, Kevin Wu, Francesco Pinto, Zhaorun Chen, Yi Zeng, Yu Yang, Shuang Yang, Sanmi Koyejo, James Zou, Bo Li

**Abstract**: As large language models (LLMs) become increasingly capable, security and safety evaluation are crucial. While current red teaming approaches have made strides in assessing LLM vulnerabilities, they often rely heavily on human input and lack comprehensive coverage of emerging attack vectors. This paper introduces AutoRedTeamer, a novel framework for fully automated, end-to-end red teaming against LLMs. AutoRedTeamer combines a multi-agent architecture with a memory-guided attack selection mechanism to enable continuous discovery and integration of new attack vectors. The dual-agent framework consists of a red teaming agent that can operate from high-level risk categories alone to generate and execute test cases and a strategy proposer agent that autonomously discovers and implements new attacks by analyzing recent research. This modular design allows AutoRedTeamer to adapt to emerging threats while maintaining strong performance on existing attack vectors. We demonstrate AutoRedTeamer's effectiveness across diverse evaluation settings, achieving 20% higher attack success rates on HarmBench against Llama-3.1-70B while reducing computational costs by 46% compared to existing approaches. AutoRedTeamer also matches the diversity of human-curated benchmarks in generating test cases, providing a comprehensive, scalable, and continuously evolving framework for evaluating the security of AI systems.

摘要: 随着大型语言模型(LLM)的能力越来越强，安全和安全评估变得至关重要。虽然当前的红色团队方法在评估LLM漏洞方面取得了很大进展，但它们往往严重依赖人力投入，并且缺乏对新兴攻击媒介的全面覆盖。本文介绍了AutoRedTeamer，这是一个针对LLMS的全自动化、端到端的Red Teamer框架。AutoRedTeamer将多代理体系结构与内存引导的攻击选择机制相结合，以实现对新攻击载体的持续发现和集成。该双代理框架由红色团队代理和策略提出者代理组成，前者可以从高级别风险类别单独操作来生成和执行测试用例，后者通过分析最近的研究自主地发现和实现新的攻击。这种模块化设计使AutoRedTeamer能够适应新出现的威胁，同时保持对现有攻击媒介的强大性能。我们展示了AutoRedTeamer在不同评估环境中的有效性，与现有方法相比，在HarmBch上对Llama-3.1-70B的攻击成功率提高了20%，同时计算成本降低了46%。AutoRedTeamer还在生成测试用例时匹配了人工管理的基准测试的多样性，为评估AI系统的安全性提供了一个全面的、可扩展的和不断发展的框架。



## **9. Undesirable Memorization in Large Language Models: A Survey**

大型语言模型中不可取的并行化：一项调查 cs.CL

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2410.02650v2) [paper-pdf](http://arxiv.org/pdf/2410.02650v2)

**Authors**: Ali Satvaty, Suzan Verberne, Fatih Turkmen

**Abstract**: While recent research increasingly showcases the remarkable capabilities of Large Language Models (LLMs), it is equally crucial to examine their associated risks. Among these, privacy and security vulnerabilities are particularly concerning, posing significant ethical and legal challenges. At the heart of these vulnerabilities stands memorization, which refers to a model's tendency to store and reproduce phrases from its training data. This phenomenon has been shown to be a fundamental source to various privacy and security attacks against LLMs. In this paper, we provide a taxonomy of the literature on LLM memorization, exploring it across three dimensions: granularity, retrievability, and desirability. Next, we discuss the metrics and methods used to quantify memorization, followed by an analysis of the causes and factors that contribute to memorization phenomenon. We then explore strategies that are used so far to mitigate the undesirable aspects of this phenomenon. We conclude our survey by identifying potential research topics for the near future, including methods to balance privacy and performance, and the analysis of memorization in specific LLM contexts such as conversational agents, retrieval-augmented generation, and diffusion language models. Given the rapid research pace in this field, we also maintain a dedicated repository of the references discussed in this survey which will be regularly updated to reflect the latest developments.

摘要: 虽然最近的研究越来越多地展示了大型语言模型(LLM)的非凡能力，但检查其相关风险也同样至关重要。其中，隐私和安全漏洞尤其令人担忧，构成了重大的道德和法律挑战。这些漏洞的核心是记忆，它指的是模型从其训练数据中存储和复制短语的倾向。这一现象已被证明是针对LLMS的各种隐私和安全攻击的基本来源。本文对有关LLM记忆的文献进行了分类，从粒度、可检索性和可取性三个维度对其进行了探讨。接下来，我们讨论了量化记忆的指标和方法，并分析了造成记忆现象的原因和因素。然后，我们探讨到目前为止用来缓解这一现象的不良方面的策略。我们通过确定在不久的将来潜在的研究主题来结束我们的调查，包括平衡隐私和性能的方法，以及在特定的LLM环境中对记忆的分析，例如会话代理、提取-增强生成和扩散语言模型。鉴于这一领域的研究步伐很快，我们还设有一个专门的资料库，储存本次调查中讨论的参考资料，并将定期更新，以反映最新的发展。



## **10. Safety at Scale: A Comprehensive Survey of Large Model Safety**

大规模安全性：大型车型安全性全面调查 cs.CR

47 pages, 3 figures, 11 tables; GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2502.05206v3) [paper-pdf](http://arxiv.org/pdf/2502.05206v3)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Xudong Han, Haonan Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Tim Baldwin, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.

摘要: 大型模型的快速发展，受到其通过大规模预训练而具有的非凡学习和泛化能力的推动，重塑了人工智能(AI)的版图。这些模型现在是广泛应用的基础，包括对话式人工智能、推荐系统、自动驾驶、内容生成、医疗诊断和科学发现。然而，它们的广泛部署也使它们面临重大的安全风险，引发了人们对健壮性、可靠性和道德影响的担忧。本调查系统地回顾了当前关于大模型的安全研究，包括视觉基础模型(VFM)、大语言模型(LLMS)、视觉语言预训练(VLP)模型、视觉语言模型(VLMS)、扩散模型(DM)和基于大模型的代理。我们的工作总结如下：(1)对这些模型的安全威胁进行了全面的分类，包括对抗性攻击、数据中毒、后门攻击、越狱和快速注入攻击、能量延迟攻击、数据和模型提取攻击以及新出现的特定于代理的威胁。(2)我们回顾了针对每种攻击类型提出的防御策略(如果可用)，并总结了安全研究常用的数据集和基准。(3)在此基础上，我们确定并讨论了大型模型安全方面的开放挑战，强调需要全面的安全评估、可扩展和有效的防御机制以及可持续的数据实践。更重要的是，我们强调了研究界和国际合作集体努力的必要性。我们的工作可以作为研究人员和从业者的有用参考，促进正在进行的全面防御系统和平台的开发，以保护人工智能模型。



## **11. Assessing AI vs Human-Authored Spear Phishing SMS Attacks: An Empirical Study**

评估人工智能与人类发起的鱼叉网络钓鱼短信攻击：一项实证研究 cs.CY

18 pages, 5 figures, 1 table

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2406.13049v2) [paper-pdf](http://arxiv.org/pdf/2406.13049v2)

**Authors**: Jerson Francia, Derek Hansen, Ben Schooley, Matthew Taylor, Shydra Murray, Greg Snow

**Abstract**: This paper explores the use of Large Language Models (LLMs) in spear phishing message generation and evaluates their performance compared to human-authored counterparts. Our pilot study examines the effectiveness of smishing (SMS phishing) messages created by GPT-4 and human authors, which have been personalized for willing targets. The targets assessed these messages in a modified ranked-order experiment using a novel methodology we call TRAPD (Threshold Ranking Approach for Personalized Deception). Experiments involved ranking each spear phishing message from most to least convincing, providing qualitative feedback, and guessing which messages were human- or AI-generated. Results show that LLM-generated messages are often perceived as more convincing than those authored by humans, particularly job-related messages. Targets also struggled to distinguish between human- and AI-generated messages. We analyze different criteria the targets used to assess the persuasiveness and source of messages. This study aims to highlight the urgent need for further research and improved countermeasures against personalized AI-enabled social engineering attacks.

摘要: 本文探讨了大型语言模型(LLM)在鱼叉式网络钓鱼消息生成中的使用，并与人类创作的同类消息进行了比较。我们的试点研究检查了GPT-4和人类作者创建的Smsing(短信钓鱼)消息的有效性，这些消息已经针对愿意攻击的目标进行了个性化。目标在一种改进的排序实验中使用了一种新的方法来评估这些消息，我们称之为TRAPD(个性化欺骗的阈值排序方法)。实验包括从最令人信服的到最不令人信服的顺序对每条鱼叉式网络钓鱼消息进行排序，提供定性反馈，并猜测哪些消息是人类或人工智能生成的。结果表明，LLM生成的消息通常被认为比人类创作的消息更具说服力，特别是与工作有关的消息。目标也很难区分人类和人工智能生成的消息。我们分析了目标用来评估信息的说服力和来源的不同标准。这项研究旨在突出针对个性化人工智能启用的社会工程攻击的进一步研究和改进对策的迫切需要。



## **12. Temporal Context Awareness: A Defense Framework Against Multi-turn Manipulation Attacks on Large Language Models**

时间上下文感知：针对大型语言模型多轮操纵攻击的防御框架 cs.CR

6 pages, 2 figures, IEEE CAI

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.15560v1) [paper-pdf](http://arxiv.org/pdf/2503.15560v1)

**Authors**: Prashant Kulkarni, Assaf Namer

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to sophisticated multi-turn manipulation attacks, where adversaries strategically build context through seemingly benign conversational turns to circumvent safety measures and elicit harmful or unauthorized responses. These attacks exploit the temporal nature of dialogue to evade single-turn detection methods, representing a critical security vulnerability with significant implications for real-world deployments.   This paper introduces the Temporal Context Awareness (TCA) framework, a novel defense mechanism designed to address this challenge by continuously analyzing semantic drift, cross-turn intention consistency and evolving conversational patterns. The TCA framework integrates dynamic context embedding analysis, cross-turn consistency verification, and progressive risk scoring to detect and mitigate manipulation attempts effectively. Preliminary evaluations on simulated adversarial scenarios demonstrate the framework's potential to identify subtle manipulation patterns often missed by traditional detection techniques, offering a much-needed layer of security for conversational AI systems. In addition to outlining the design of TCA , we analyze diverse attack vectors and their progression across multi-turn conversation, providing valuable insights into adversarial tactics and their impact on LLM vulnerabilities. Our findings underscore the pressing need for robust, context-aware defenses in conversational AI systems and highlight TCA framework as a promising direction for securing LLMs while preserving their utility in legitimate applications. We make our implementation available to support further research in this emerging area of AI security.

摘要: 大型语言模型(LLM)越来越容易受到复杂的多回合操纵攻击，在这种攻击中，对手通过看似良性的对话转向来策略性地构建上下文，以绕过安全措施并引发有害或未经授权的响应。这些攻击利用对话的时间性来逃避单轮检测方法，这是一个严重的安全漏洞，对现实世界的部署具有重大影响。本文介绍了时态语境感知(TCA)框架，这是一种新的防御机制，旨在通过不断分析语义漂移、跨话轮意图一致性和会话模式演变来应对这一挑战。TCA框架集成了动态上下文嵌入分析、跨回合一致性验证和渐进式风险评分，以有效检测和减少操纵企图。对模拟对抗性场景的初步评估表明，该框架具有识别传统检测技术经常遗漏的微妙操纵模式的潜力，为对话式人工智能系统提供了亟需的安全层。除了概述TCA的设计之外，我们还分析了不同的攻击向量及其在多轮对话中的进展，为敌方战术及其对LLM漏洞的影响提供了有价值的见解。我们的发现强调了在对话式人工智能系统中对强大的、上下文感知的防御的迫切需要，并强调TCA框架是一个有希望的方向，可以在保护LLM的有效性的同时保护其在合法应用中的有效性。我们提供我们的实现，以支持在这一新兴的人工智能安全领域的进一步研究。



## **13. Personalized Attacks of Social Engineering in Multi-turn Conversations -- LLM Agents for Simulation and Detection**

多轮对话中社会工程的个性化攻击-- LLM模拟和检测代理 cs.CR

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.15552v1) [paper-pdf](http://arxiv.org/pdf/2503.15552v1)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.

摘要: 会话代理的快速发展，特别是由大语言模型(LLM)驱动的聊天机器人，构成了社交媒体平台上的社交工程(SE)攻击的巨大风险。基于聊天的多轮交互中的SE检测比单实例检测要复杂得多，这是因为这些对话的动态性质。缓解这一威胁的一个关键因素是了解SE攻击的运作机制，特别是攻击者如何利用漏洞以及受害者的个性特征如何导致他们的易感性。在这项工作中，我们提出了一个LLM代理框架SE-VSim，通过生成多话轮会话来模拟SE攻击机制。我们对具有不同个性特征的受害者代理进行建模，以评估心理特征如何影响操纵的易感性。使用1000多个模拟对话的数据集，我们检查了攻击场景，在这些场景中，伪装成招聘者、资助机构和记者的对手试图提取敏感信息。基于这一分析，我们提出了一个概念证明，SE-OmniGuard，通过利用受害者个性的先验知识，评估攻击策略，并监控对话中的信息交换来识别潜在的SE尝试，为用户提供个性化保护。



## **14. TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization Methods**

TAROT：使用策略优化方法的面向任务的作者混淆 cs.CL

Accepted to the NAACL PrivateNLP 2025 Workshop

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2407.21630v2) [paper-pdf](http://arxiv.org/pdf/2407.21630v2)

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi

**Abstract**: Authorship obfuscation aims to disguise the identity of an author within a text by altering the writing style, vocabulary, syntax, and other linguistic features associated with the text author. This alteration needs to balance privacy and utility. While strong obfuscation techniques can effectively hide the author's identity, they often degrade the quality and usefulness of the text for its intended purpose. Conversely, maintaining high utility tends to provide insufficient privacy, making it easier for an adversary to de-anonymize the author. Thus, achieving an optimal trade-off between these two conflicting objectives is crucial. In this paper, we propose TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization, a new unsupervised authorship obfuscation method whose goal is to optimize the privacy-utility trade-off by regenerating the entire text considering its downstream utility. Our approach leverages policy optimization as a fine-tuning paradigm over small language models in order to rewrite texts by preserving author identity and downstream task utility. We show that our approach largely reduces the accuracy of attackers while preserving utility. We make our code and models publicly available.

摘要: 作者身份混淆旨在通过改变与文本作者相关的写作风格、词汇、句法和其他语言特征来掩盖作者在文本中的身份。这一改变需要平衡隐私和效用。虽然强大的混淆技术可以有效地隐藏作者的身份，但它们往往会降低文本的质量和对预期目的的有用性。相反，保持高实用性往往会提供不充分的隐私，使对手更容易解除作者的匿名。因此，在这两个相互冲突的目标之间实现最佳权衡至关重要。在本文中，我们提出了一种新的无监督作者身份混淆方法--TAROT：基于策略优化的面向任务的作者身份混淆方法，其目标是通过重新生成考虑下游效用的整个文本来优化隐私和效用之间的权衡。我们的方法利用策略优化作为小语言模型上的微调范式，以便通过保留作者身份和下游任务效用来重写文本。我们表明，我们的方法在很大程度上降低了攻击者的准确性，同时保持了实用性。我们公开我们的代码和模型。



## **15. Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks**

针对越狱攻击的多模式大型语言模型对抗训练 cs.CV

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.04833v2) [paper-pdf](http://arxiv.org/pdf/2503.04833v2)

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems.

摘要: 多通道大语言模型在跨通道理解和生成任务方面取得了显著进展。然而，它们仍然容易受到越狱攻击，在越狱攻击中，精心设计的扰动绕过安全护栏，引发有害输出。在这篇文章中，我们提出了在MLLM训练阶段为防御越狱攻击而定制的第一个对抗性训练(AT)范例。将传统的AT扩展到这一领域会带来两个关键挑战：有效地调整大量参数和确保对跨多个通道的攻击的健壮性。为了应对这些挑战，我们引入了投影层对抗对手训练(ProEAT)，这是一个端到端的AT框架。ProEAT结合了基于投影仪的对抗性训练体系结构，通过将对抗性训练集中在轻量级投影器层而不是整个模型上，在保持计算可行性的同时有效地处理大规模参数；此外，我们设计了动态权重调整机制，基于任务需求优化损失函数的权重分配，从而简化了调整过程。为了提高防御性能，我们提出了一种跨视觉和文本模式的联合优化策略，确保对来自任何一种模式的越狱攻击具有强大的抵抗力。在三种主流MLLMS上对五种主要的越狱攻击方法进行了广泛的实验，证明了该方法的有效性。ProEAT实现了最先进的防御性能，在文本和图像模式中的表现比现有基线平均高出34%，而干净的准确性仅降低了1%。此外，对真实世界体现的智能系统的评估突出了我们框架的实用适用性，为开发更安全可靠的多式联运系统铺平了道路。



## **16. Survey of Adversarial Robustness in Multimodal Large Language Models**

多模式大型语言模型中的对抗鲁棒性研究 cs.CV

9 pages

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.13962v1) [paper-pdf](http://arxiv.org/pdf/2503.13962v1)

**Authors**: Chengze Jiang, Zhuangzhuang Wang, Minjing Dong, Jie Gui

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance in artificial intelligence by facilitating integrated understanding across diverse modalities, including text, images, video, audio, and speech. However, their deployment in real-world applications raises significant concerns about adversarial vulnerabilities that could compromise their safety and reliability. Unlike unimodal models, MLLMs face unique challenges due to the interdependencies among modalities, making them susceptible to modality-specific threats and cross-modal adversarial manipulations. This paper reviews the adversarial robustness of MLLMs, covering different modalities. We begin with an overview of MLLMs and a taxonomy of adversarial attacks tailored to each modality. Next, we review key datasets and evaluation metrics used to assess the robustness of MLLMs. After that, we provide an in-depth review of attacks targeting MLLMs across different modalities. Our survey also identifies critical challenges and suggests promising future research directions.

摘要: 多模式大语言模型(MLLM)通过促进跨不同模式的集成理解，包括文本、图像、视频、音频和语音，在人工智能中表现出出色的性能。然而，它们在实际应用程序中的部署引发了人们对可能危及其安全性和可靠性的对抗性漏洞的严重担忧。与单模模型不同，由于各通道之间的相互依赖关系，最大似然模型面临着独特的挑战，这使得它们容易受到特定通道的威胁和跨通道的对抗性操作。本文综述了MLLMS的对抗稳健性，涵盖了不同的模式。我们首先概述MLLMS和针对每种模式量身定做的对抗性攻击分类。接下来，我们回顾了用于评估MLLMS稳健性的关键数据集和评估指标。在此之后，我们提供了针对不同模式的MLLM的攻击的深入回顾。我们的调查还确定了关键挑战，并提出了前景看好的未来研究方向。



## **17. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

AttackEval：如何评估越狱攻击对大型语言模型的有效性 cs.CL

Accepted by ACM SIGKDD Explorations 2025

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2401.09002v6) [paper-pdf](http://arxiv.org/pdf/2401.09002v6)

**Authors**: Dong Shu, Chong Zhang, Mingyu Jin, Zihao Zhou, Lingyao Li, Yongfeng Zhang

**Abstract**: Jailbreak attacks represent one of the most sophisticated threats to the security of large language models (LLMs). To deal with such risks, we introduce an innovative framework that can help evaluate the effectiveness of jailbreak attacks on LLMs. Unlike traditional binary evaluations focusing solely on the robustness of LLMs, our method assesses the attacking prompts' effectiveness. We present two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework uses a scoring range from 0 to 1, offering unique perspectives and allowing for the assessment of attack effectiveness in different scenarios. Additionally, we develop a comprehensive ground truth dataset specifically tailored for jailbreak prompts. This dataset is a crucial benchmark for our current study and provides a foundational resource for future research. By comparing with traditional evaluation methods, our study shows that the current results align with baseline metrics while offering a more nuanced and fine-grained assessment. It also helps identify potentially harmful attack prompts that might appear harmless in traditional evaluations. Overall, our work establishes a solid foundation for assessing a broader range of attack prompts in prompt injection.

摘要: 越狱攻击是大型语言模型(LLM)安全面临的最复杂的威胁之一。为了应对这样的风险，我们引入了一个创新的框架，可以帮助评估越狱攻击对低收入者的有效性。与传统的只关注LLMS健壮性的二值评估不同，我们的方法评估了攻击提示的有效性。我们提出了两种不同的评估框架：粗粒度评估和细粒度评估。每个框架使用从0到1的评分范围，提供独特的视角，并允许在不同情况下评估攻击效果。此外，我们还开发了专门为越狱提示量身定做的全面地面事实数据集。该数据集是我们当前研究的重要基准，并为未来的研究提供了基础资源。通过与传统评估方法的比较，我们的研究表明，当前的结果与基线度量一致，同时提供了更细微和细粒度的评估。它还有助于识别在传统评估中可能看起来无害的潜在有害攻击提示。总体而言，我们的工作为在快速注入中评估更广泛的攻击提示奠定了坚实的基础。



## **18. Web Artifact Attacks Disrupt Vision Language Models**

网络收件箱攻击扰乱视觉语言模型 cs.CV

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.13652v1) [paper-pdf](http://arxiv.org/pdf/2503.13652v1)

**Authors**: Maan Qraitem, Piotr Teterwak, Kate Saenko, Bryan A. Plummer

**Abstract**: Vision-language models (VLMs) (e.g., CLIP, LLaVA) are trained on large-scale, lightly curated web datasets, leading them to learn unintended correlations between semantic concepts and unrelated visual signals. These associations degrade model accuracy by causing predictions to rely on incidental patterns rather than genuine visual understanding. Prior work has weaponized these correlations as an attack vector to manipulate model predictions, such as inserting a deceiving class text onto the image in a typographic attack. These attacks succeed due to VLMs' text-heavy bias-a result of captions that echo visible words rather than describing content. However, this attack has focused solely on text that matches the target class exactly, overlooking a broader range of correlations, including non-matching text and graphical symbols, which arise from the abundance of branding content in web-scale data. To address this gap, we introduce artifact-based attacks: a novel class of manipulations that mislead models using both non-matching text and graphical elements. Unlike typographic attacks, these artifacts are not predefined, making them harder to defend against but also more challenging to find. We address this by framing artifact attacks as a search problem and demonstrate their effectiveness across five datasets, with some artifacts reinforcing each other to reach 100% attack success rates. These attacks transfer across models with up to 90% effectiveness, making it possible to attack unseen models. To defend against these attacks, we extend prior work's artifact aware prompting to the graphical setting. We see a moderate reduction of success rates of up to 15% relative to standard prompts, suggesting a promising direction for enhancing model robustness.

摘要: 视觉语言模型(VLM)(例如，CLIP、LLaVA)在大规模、轻度精选的Web数据集上进行训练，导致它们学习语义概念和无关视觉信号之间的意外关联。这些关联导致预测依赖于附带模式而不是真正的视觉理解，从而降低了模型的准确性。以前的工作已经将这些相关性武器化为攻击矢量，以操纵模型预测，例如在排版攻击中将欺骗性类文本插入图像。这些攻击之所以成功，是因为VLMS的文本偏向--字幕呼应可见单词而不是描述内容的结果。然而，这种攻击只关注与目标类别完全匹配的文本，而忽略了更广泛的相关性，包括不匹配的文本和图形符号，这些相关性源于网络规模数据中丰富的品牌内容。为了弥补这一差距，我们引入了基于人工产物的攻击：一种使用不匹配文本和图形元素误导模型的新型操作。与排版攻击不同，这些文物不是预定义的，这使得它们更难防御，但也更难找到。我们通过将人工产物攻击作为一个搜索问题来解决这个问题，并在五个数据集上展示了它们的有效性，其中一些人工产物相互加强，达到100%的攻击成功率。这些攻击在模型之间转移，效率高达90%，使攻击看不见的模型成为可能。为了防御这些攻击，我们将以前工作的人工产物感知提示扩展到图形设置。我们看到，与标准提示相比，成功率适度降低，最高可达15%，这表明了增强模型稳健性的一个有希望的方向。



## **19. Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation**

助推器：通过减弱有害扰动来解决大型语言模型的有害微调 cs.CL

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2409.01586v4) [paper-pdf](http://arxiv.org/pdf/2409.01586v4)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Harmful fine-tuning attack poses serious safety concerns for large language models' fine-tuning-as-a-service. While existing defenses have been proposed to mitigate the issue, their performances are still far away from satisfactory, and the root cause of the problem has not been fully recovered. To this end, we in this paper show that harmful perturbation over the model weights could be a probable cause of alignment-broken. In order to attenuate the negative impact of harmful perturbation, we propose an alignment-stage solution, dubbed Booster. Technically, along with the original alignment loss, we append a loss regularizer in the alignment stage's optimization. The regularizer ensures that the model's harmful loss reduction after the simulated harmful perturbation is attenuated, thereby mitigating the subsequent fine-tuning risk. Empirical results show that Booster can effectively reduce the harmful score of the fine-tuned models while maintaining the performance of downstream tasks. Our code is available at https://github.com/git-disl/Booster.

摘要: 有害的微调攻击给大型语言模型的微调即服务带来了严重的安全问题。虽然已经提出了现有的防御措施来缓解这一问题，但它们的表现仍然远远不能令人满意，问题的根本原因尚未完全恢复。为此，我们在本文中表明，对模型权重的有害扰动可能是排列断裂的一个可能原因。为了减弱有害扰动的负面影响，我们提出了一种对准阶段的解决方案，称为Booster。在技术上，除了原始的对准损失外，我们还在对准阶段的优化中加入了损失正则化。正则化确保了模型在模拟的有害扰动后的有害损失减小，从而减轻了随后的微调风险。实验结果表明，Booster在保持下游任务性能的同时，能有效降低微调模型的有害分数。我们的代码可以在https://github.com/git-disl/Booster.上找到



## **20. A Framework to Assess Multilingual Vulnerabilities of LLMs**

评估法学硕士多语言脆弱性的框架 cs.CL

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.13081v1) [paper-pdf](http://arxiv.org/pdf/2503.13081v1)

**Authors**: Likai Tang, Niruth Bogahawatta, Yasod Ginige, Jiarui Xu, Shixuan Sun, Surangika Ranathunga, Suranga Seneviratne

**Abstract**: Large Language Models (LLMs) are acquiring a wider range of capabilities, including understanding and responding in multiple languages. While they undergo safety training to prevent them from answering illegal questions, imbalances in training data and human evaluation resources can make these models more susceptible to attacks in low-resource languages (LRL). This paper proposes a framework to automatically assess the multilingual vulnerabilities of commonly used LLMs. Using our framework, we evaluated six LLMs across eight languages representing varying levels of resource availability. We validated the assessments generated by our automated framework through human evaluation in two languages, demonstrating that the framework's results align with human judgments in most cases. Our findings reveal vulnerabilities in LRL; however, these may pose minimal risk as they often stem from the model's poor performance, resulting in incoherent responses.

摘要: 大型语言模型（LLM）正在获得更广泛的能力，包括以多种语言理解和响应。虽然他们接受安全培训以防止他们回答非法问题，但训练数据和人力评估资源的不平衡可能使这些模型更容易受到低资源语言（LRL）的攻击。本文提出了一个自动评估常用LLM多语言漏洞的框架。使用我们的框架，我们评估了八种语言的六个LLM，代表不同级别的资源可用性。我们通过两种语言的人工评估验证了自动化框架生成的评估，证明框架的结果在大多数情况下与人类判断一致。我们的研究结果揭示了LRL中的漏洞;然而，这些漏洞可能构成的风险很小，因为它们通常源于模型的性能不佳，导致响应不一致。



## **21. TuBA: Cross-Lingual Transferability of Backdoor Attacks in LLMs with Instruction Tuning**

TuBA：具有指令调优的LLM后门攻击的跨语言可转移性 cs.CL

work in progress

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2404.19597v3) [paper-pdf](http://arxiv.org/pdf/2404.19597v3)

**Authors**: Xuanli He, Jun Wang, Qiongkai Xu, Pasquale Minervini, Pontus Stenetorp, Benjamin I. P. Rubinstein, Trevor Cohn

**Abstract**: The implications of backdoor attacks on English-centric large language models (LLMs) have been widely examined - such attacks can be achieved by embedding malicious behaviors during training and activated under specific conditions that trigger malicious outputs. Despite the increasing support for multilingual capabilities in open-source and proprietary LLMs, the impact of backdoor attacks on these systems remains largely under-explored. Our research focuses on cross-lingual backdoor attacks against multilingual LLMs, particularly investigating how poisoning the instruction-tuning data for one or two languages can affect the outputs for languages whose instruction-tuning data were not poisoned. Despite its simplicity, our empirical analysis reveals that our method exhibits remarkable efficacy in models like mT5 and GPT-4o, with high attack success rates, surpassing 90% in more than 7 out of 12 languages across various scenarios. Our findings also indicate that more powerful models show increased susceptibility to transferable cross-lingual backdoor attacks, which also applies to LLMs predominantly pre-trained on English data, such as Llama2, Llama3, and Gemma. Moreover, our experiments demonstrate 1) High Transferability: the backdoor mechanism operates successfully in cross-lingual response scenarios across 26 languages, achieving an average attack success rate of 99%, and 2) Robustness: the proposed attack remains effective even after defenses are applied. These findings expose critical security vulnerabilities in multilingual LLMs and highlight the urgent need for more robust, targeted defense strategies to address the unique challenges posed by cross-lingual backdoor transfer.

摘要: 后门攻击对以英语为中心的大型语言模型(LLM)的影响已被广泛研究-此类攻击可以通过在训练期间嵌入恶意行为来实现，并在触发恶意输出的特定条件下激活。尽管在开源和专有LLM中对多语言功能的支持越来越多，但后门攻击对这些系统的影响在很大程度上仍然没有得到充分的探索。我们的研究重点是针对多语言LLM的跨语言后门攻击，特别是调查毒化一到两种语言的指令调整数据如何影响那些指令调整数据没有中毒的语言的输出。尽管简单，但我们的经验分析表明，我们的方法在MT5和GPT-40等模型中表现出了显著的效果，在不同的场景下，在12种语言中的7种以上的攻击成功率超过90%。我们的发现还表明，更强大的模型显示出对可转移的跨语言后门攻击的易感性，这也适用于主要根据英语数据进行预训练的LLM，如Llama2、Llama3和Gema。此外，我们的实验表明：1)高可移植性：后门机制在26种语言的跨语言响应场景中成功运行，平均攻击成功率为99%；2)健壮性：所提出的攻击即使在实施防御后仍有效。这些发现暴露了多语言低成本管理中的关键安全漏洞，并突显了迫切需要更强大、更有针对性的防御战略，以应对跨语言后门转移带来的独特挑战。



## **22. How Good is my Histopathology Vision-Language Foundation Model? A Holistic Benchmark**

我的组织学视觉语言基础模型有多好？整体基准 eess.IV

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12990v1) [paper-pdf](http://arxiv.org/pdf/2503.12990v1)

**Authors**: Roba Al Majzoub, Hashmat Malik, Muzammal Naseer, Zaigham Zaheer, Tariq Mahmood, Salman Khan, Fahad Khan

**Abstract**: Recently, histopathology vision-language foundation models (VLMs) have gained popularity due to their enhanced performance and generalizability across different downstream tasks. However, most existing histopathology benchmarks are either unimodal or limited in terms of diversity of clinical tasks, organs, and acquisition instruments, as well as their partial availability to the public due to patient data privacy. As a consequence, there is a lack of comprehensive evaluation of existing histopathology VLMs on a unified benchmark setting that better reflects a wide range of clinical scenarios. To address this gap, we introduce HistoVL, a fully open-source comprehensive benchmark comprising images acquired using up to 11 various acquisition tools that are paired with specifically crafted captions by incorporating class names and diverse pathology descriptions. Our Histo-VL includes 26 organs, 31 cancer types, and a wide variety of tissue obtained from 14 heterogeneous patient cohorts, totaling more than 5 million patches obtained from over 41K WSIs viewed under various magnification levels. We systematically evaluate existing histopathology VLMs on Histo-VL to simulate diverse tasks performed by experts in real-world clinical scenarios. Our analysis reveals interesting findings, including large sensitivity of most existing histopathology VLMs to textual changes with a drop in balanced accuracy of up to 25% in tasks such as Metastasis detection, low robustness to adversarial attacks, as well as improper calibration of models evident through high ECE values and low model prediction confidence, all of which can affect their clinical implementation.

摘要: 最近，组织病理学视觉-语言基础模型(VLM)因其在不同下游任务中增强的性能和普适性而广受欢迎。然而，现有的大多数组织病理学基准要么是单一的，要么在临床任务、器官和采集工具的多样性方面受到限制，而且由于患者数据的隐私，它们对公众的部分可用性。因此，在一个统一的基准设置上缺乏对现有组织病理学VLM的全面评估，以更好地反映广泛的临床情景。为了弥补这一差距，我们引入了HistoVL，这是一个完全开源的综合基准，包括使用多达11种不同的采集工具获取的图像，这些工具通过结合类名和不同的病理描述与专门制作的说明相匹配。我们的HISTO-VL包括26个器官，31种癌症类型，以及从14个不同的患者队列中获得的各种组织，总计超过500万个斑块，这些斑块是在不同放大水平下从超过41K的WSIS中获得的。我们系统地评估HISTO-VL上现有的组织病理学VLM，以模拟真实世界临床场景中专家执行的各种任务。我们的分析揭示了有趣的发现，包括大多数现有的组织病理学VLM对文本变化的高度敏感性，在转移检测等任务中平衡准确率下降高达25%，对对抗性攻击的稳健性低，以及通过高EC值和低模型预测置信度明显地对模型进行不正确的校准，所有这些都可能影响其临床实施。



## **23. MirrorGuard: Adaptive Defense Against Jailbreaks via Entropy-Guided Mirror Crafting**

收件箱警卫：通过熵引导镜子制作对越狱的自适应防御 cs.CR

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12931v1) [paper-pdf](http://arxiv.org/pdf/2503.12931v1)

**Authors**: Rui Pu, Chaozhuo Li, Rui Ha, Litian Zhang, Lirong Qiu, Xi Zhang

**Abstract**: Defending large language models (LLMs) against jailbreak attacks is crucial for ensuring their safe deployment. Existing defense strategies generally rely on predefined static criteria to differentiate between harmful and benign prompts. However, such rigid rules are incapable of accommodating the inherent complexity and dynamic nature of real jailbreak attacks. In this paper, we propose a novel concept of ``mirror'' to enable dynamic and adaptive defense. A mirror refers to a dynamically generated prompt that mirrors the syntactic structure of the input while ensuring semantic safety. The personalized discrepancies between the input prompts and their corresponding mirrors serve as the guiding principles for defense. A new defense paradigm, MirrorGuard, is further proposed to detect and calibrate risky inputs based on such mirrors. An entropy-based detection metric, Relative Input Uncertainty (RIU), is integrated into MirrorGuard to quantify the discrepancies between input prompts and mirrors. MirrorGuard is evaluated on several popular datasets, demonstrating state-of-the-art defense performance while maintaining general effectiveness.

摘要: 保护大型语言模型(LLM)免受越狱攻击对于确保它们的安全部署至关重要。现有的防御策略通常依赖于预定义的静态标准来区分有害提示和良性提示。然而，这种僵化的规则无法适应真正越狱攻击的内在复杂性和动态性质。本文提出了一种新的“镜像”概念，以实现动态和自适应的防御。镜像是指动态生成的提示，它反映了输入的句法结构，同时确保了语义安全。输入提示与其对应的镜像之间的个性化差异是答辩的指导原则。此外，还提出了一种新的防御范例MirrorGuard，用于检测和校准基于此类镜像的危险输入。基于熵的检测指标相对输入不确定性(RIU)被集成到MirrorGuard中，以量化输入提示和镜像之间的差异。MirrorGuard在几个流行的数据集上进行了评估，在保持总体有效性的同时展示了最先进的防御性能。



## **24. Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agents**

提示流程完整性以防止LLM代理的特权升级 cs.CR

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.15547v1) [paper-pdf](http://arxiv.org/pdf/2503.15547v1)

**Authors**: Juhee Kim, Woohyuk Choi, Byoungyoung Lee

**Abstract**: Large Language Models (LLMs) are combined with plugins to create powerful LLM agents that provide a wide range of services. Unlike traditional software, LLM agent's behavior is determined at runtime by natural language prompts from either user or plugin's data. This flexibility enables a new computing paradigm with unlimited capabilities and programmability, but also introduces new security risks, vulnerable to privilege escalation attacks. Moreover, user prompt is prone to be interpreted in an insecure way by LLM agents, creating non-deterministic behaviors that can be exploited by attackers. To address these security risks, we propose Prompt Flow Integrity (PFI), a system security-oriented solution to prevent privilege escalation in LLM agents. Analyzing the architectural characteristics of LLM agents, PFI features three mitigation techniques -- i.e., untrusted data identification, enforcing least privilege on LLM agents, and validating unsafe data flows. Our evaluation result shows that PFI effectively mitigates privilege escalation attacks while successfully preserving the utility of LLM agents.

摘要: 大型语言模型(LLM)与插件相结合，以创建功能强大的LLM代理，提供广泛的服务。与传统软件不同，LLM代理的行为在运行时由用户或插件数据的自然语言提示决定。这种灵活性实现了具有无限功能和可编程性的新计算模式，但也带来了新的安全风险，容易受到权限提升攻击。此外，用户提示容易被LLM代理以不安全的方式解释，从而产生可被攻击者利用的不确定行为。为了解决这些安全风险，我们提出了即时流完整性(PFI)，这是一种面向系统安全的解决方案，以防止LLM代理中的权限提升。分析LLM代理的体系结构特征，PFI具有三种缓解技术--即不可信数据识别、在LLM代理上实施最小特权和验证不安全数据流。我们的评估结果表明，PFI在成功保持LLM代理效用的同时，有效地缓解了权限提升攻击。



## **25. When "Competency" in Reasoning Opens the Door to Vulnerability: Jailbreaking LLMs via Novel Complex Ciphers**

当推理中的“能力”打开脆弱之门：通过新颖复杂密码越狱LLM cs.CL

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2402.10601v3) [paper-pdf](http://arxiv.org/pdf/2402.10601v3)

**Authors**: Divij Handa, Zehua Zhang, Amir Saeidi, Shrinidhi Kumbhar, Chitta Baral

**Abstract**: Recent advancements in Large Language Model (LLM) safety have primarily focused on mitigating attacks crafted in natural language or common ciphers (e.g. Base64), which are likely integrated into newer models' safety training. However, we reveal a paradoxical vulnerability: as LLMs advance in reasoning, they inadvertently become more susceptible to novel jailbreaking attacks. Enhanced reasoning enables LLMs to interpret complex instructions and decode complex user-defined ciphers, creating an exploitable security gap. To study this vulnerability, we introduce Attacks using Custom Encryptions (ACE), a jailbreaking technique that encodes malicious queries with novel ciphers. Extending ACE, we introduce Layered Attacks using Custom Encryptions (LACE), which applies multi-layer ciphers to amplify attack complexity. Furthermore, we develop CipherBench, a benchmark designed to evaluate LLMs' accuracy in decoding encrypted benign text. Our experiments reveal a critical trade-off: LLMs that are more capable of decoding ciphers are more vulnerable to these jailbreaking attacks, with success rates on GPT-4o escalating from 40% under ACE to 78% with LACE. These findings highlight a critical insight: as LLMs become more adept at deciphering complex user ciphers--many of which cannot be preemptively included in safety training--they become increasingly exploitable.

摘要: 大型语言模型(LLM)安全方面的最新进展主要集中在缓解使用自然语言或常见密码(例如Base64)编写的攻击，这些攻击可能会集成到较新模型的安全培训中。然而，我们揭示了一个矛盾的弱点：随着LLM在推理方面的进步，它们在不经意间变得更容易受到新的越狱攻击。增强的推理使LLM能够解释复杂的指令和解码复杂的用户定义的密码，从而产生了一个可利用的安全漏洞。为了研究此漏洞，我们引入了使用自定义加密(ACE)的攻击，这是一种使用新密码对恶意查询进行编码的越狱技术。对ACE进行了扩展，引入了使用自定义加密的分层攻击(LACE)，它采用多层密码来放大攻击的复杂性。此外，我们还开发了用于评估LLMS在解密加密良性文本时的准确性的基准测试CipherBch。我们的实验揭示了一个关键的权衡：更有能力解码密码的LLM更容易受到这些越狱攻击，GPT-40上的成功率从ACE下的40%上升到LACE下的78%。这些发现突显了一个重要的洞察：随着LLM变得更加擅长破译复杂的用户密码--其中许多密码无法先发制人地包括在安全培训中--它们变得越来越容易被利用。



## **26. h4rm3l: A language for Composable Jailbreak Attack Synthesis**

h4 rm3l：可组合越狱攻击合成语言 cs.CR

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2408.04811v3) [paper-pdf](http://arxiv.org/pdf/2408.04811v3)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: Despite their demonstrated valuable capabilities, state-of-the-art (SOTA) widely deployed large language models (LLMs) still have the potential to cause harm to society due to the ineffectiveness of their safety filters, which can be bypassed by prompt transformations called jailbreak attacks. Current approaches to LLM safety assessment, which employ datasets of templated prompts and benchmarking pipelines, fail to cover sufficiently large and diverse sets of jailbreak attacks, leading to the widespread deployment of unsafe LLMs. Recent research showed that novel jailbreak attacks could be derived by composition; however, a formal composable representation for jailbreak attacks, which, among other benefits, could enable the exploration of a large compositional space of jailbreak attacks through program synthesis methods, has not been previously proposed. We introduce h4rm3l, a novel approach that addresses this gap with a human-readable domain-specific language (DSL). Our framework comprises: (1) The h4rm3l DSL, which formally expresses jailbreak attacks as compositions of parameterized string transformation primitives. (2) A synthesizer with bandit algorithms that efficiently generates jailbreak attacks optimized for a target black box LLM. (3) The h4rm3l red-teaming software toolkit that employs the previous two components and an automated harmful LLM behavior classifier that is strongly aligned with human judgment. We demonstrate h4rm3l's efficacy by synthesizing a dataset of 2656 successful novel jailbreak attacks targeting 6 SOTA open-source and proprietary LLMs, and by benchmarking those models against a subset of these synthesized attacks. Our results show that h4rm3l's synthesized attacks are diverse and more successful than existing jailbreak attacks in literature, with success rates exceeding 90% on SOTA LLMs.

摘要: 尽管被广泛部署的最先进的大型语言模型(SOTA)显示了其宝贵的功能，但由于其安全过滤器的无效，仍有可能对社会造成危害，这可以通过称为越狱攻击的快速转换来绕过。目前的LLM安全评估方法使用模板化提示和基准管道的数据集，无法覆盖足够大和多样化的越狱攻击集，导致不安全的LLM被广泛部署。最近的研究表明，新的越狱攻击可以通过组合来派生；然而，以前还没有提出用于越狱攻击的形式可组合表示，除了其他优点之外，它可以通过程序合成方法来探索越狱攻击的大组成空间。我们介绍了h4rm3l，这是一种用人类可读的领域特定语言(DSL)来解决这一差距的新方法。我们的框架包括：(1)h4rm3l DSL，它将越狱攻击形式化地表达为参数化串转换原语的组合。(2)采用强盗算法的合成器，能够高效地生成针对目标黑盒LLM进行优化的越狱攻击。(3)使用前两个组件的h4rm3l红队软件工具包，以及与人类判断高度一致的自动有害LLM行为分类器。我们通过合成针对6个Sota开源和专有LLM的2656个成功的新型越狱攻击的数据集，并将这些模型与这些合成攻击的子集进行基准比较，展示了h4rm3l的有效性。我们的结果表明，h4rm3l的综合攻击是多样的，而且比文献中现有的越狱攻击更成功，在Sota LLMS上的成功率超过90%。



## **27. Systematic Categorization, Construction and Evaluation of New Attacks against Multi-modal Mobile GUI Agents**

针对多模式移动图形用户界面代理的新攻击的系统分类、构建和评估 cs.CR

Preprint. Work in progress

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2407.09295v3) [paper-pdf](http://arxiv.org/pdf/2407.09295v3)

**Authors**: Yulong Yang, Xinshan Yang, Shuaidong Li, Chenhao Lin, Zhengyu Zhao, Chao Shen, Tianwei Zhang

**Abstract**: The integration of Large Language Models (LLMs) and Multi-modal Large Language Models (MLLMs) into mobile GUI agents has significantly enhanced user efficiency and experience. However, this advancement also introduces potential security vulnerabilities that have yet to be thoroughly explored. In this paper, we present a systematic security investigation of multi-modal mobile GUI agents, addressing this critical gap in the existing literature. Our contributions are twofold: (1) we propose a novel threat modeling methodology, leading to the discovery and feasibility analysis of 34 previously unreported attacks, and (2) we design an attack framework to systematically construct and evaluate these threats. Through a combination of real-world case studies and extensive dataset-driven experiments, we validate the severity and practicality of those attacks, highlighting the pressing need for robust security measures in mobile GUI systems.

摘要: 将大型语言模型（LLM）和多模式大型语言模型（MLLM）集成到移动图形用户界面代理中，显着提高了用户效率和体验。然而，这一进步也引入了尚未彻底探索的潜在安全漏洞。在本文中，我们对多模式移动图形用户界面代理进行了系统性的安全研究，以解决现有文献中的这一关键空白。我们的贡献有双重：（1）我们提出了一种新颖的威胁建模方法，发现了34种之前未报告的攻击并进行了可行性分析，（2）我们设计了一个攻击框架来系统地构建和评估这些威胁。通过现实世界的案例研究和广泛的厕所驱动实验的结合，我们验证了这些攻击的严重性和实用性，强调了移动图形用户界面系统中对强大安全措施的迫切需求。



## **28. One Goal, Many Challenges: Robust Preference Optimization Amid Content-Aware and Multi-Source Noise**

一个目标，诸多挑战：内容感知和多源噪音中的稳健偏好优化 cs.LG

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.12301v1) [paper-pdf](http://arxiv.org/pdf/2503.12301v1)

**Authors**: Amirabbas Afzali, Amirhossein Afsharrad, Seyed Shahabeddin Mousavi, Sanjay Lall

**Abstract**: Large Language Models (LLMs) have made significant strides in generating human-like responses, largely due to preference alignment techniques. However, these methods often assume unbiased human feedback, which is rarely the case in real-world scenarios. This paper introduces Content-Aware Noise-Resilient Preference Optimization (CNRPO), a novel framework that addresses multiple sources of content-dependent noise in preference learning. CNRPO employs a multi-objective optimization approach to separate true preferences from content-aware noises, effectively mitigating their impact. We leverage backdoor attack mechanisms to efficiently learn and control various noise sources within a single model. Theoretical analysis and extensive experiments on different synthetic noisy datasets demonstrate that CNRPO significantly improves alignment with primary human preferences while controlling for secondary noises and biases, such as response length and harmfulness.

摘要: 大型语言模型（LLM）在生成类人响应方面取得了重大进展，这主要归功于偏好对齐技术。然而，这些方法通常假设无偏见的人类反馈，而在现实世界场景中情况很少。本文介绍了内容感知噪音弹性偏好优化（CNRPO），这是一种新型框架，可解决偏好学习中多个内容相关噪音来源。CNRPO采用多目标优化方法将真实偏好与内容感知噪音分开，有效减轻其影响。我们利用后门攻击机制来有效地学习和控制单个模型内的各种噪音源。对不同合成噪音数据集的理论分析和广泛实验表明，CNRPO显着改善了与人类主要偏好的一致性，同时控制次要噪音和偏差，例如响应长度和危害性。



## **29. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

从ML到LLM：评估网络钓鱼网页检测模型对抗对抗攻击的稳健性 cs.CR

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2407.20361v3) [paper-pdf](http://arxiv.org/pdf/2407.20361v3)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of three existing task-specific models -- Stack model, VisualPhishNet, and Phishpedia -- against PhishOracle-generated adversarial phishing webpages and observe a significant drop in their detection rates. In contrast, a multimodal large language model (MLLM)-based phishing detector demonstrates stronger robustness against these adversarial attacks but still is prone to evasion. Our findings highlight the vulnerability of phishing detection models to adversarial attacks, emphasizing the need for more robust detection approaches. Furthermore, we conduct a user study to evaluate whether PhishOracle-generated adversarial phishing webpages can deceive users. The results show that many of these phishing webpages evade not only existing detection models but also users. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources will be made publicly available on GitHub.

摘要: 网络钓鱼攻击试图欺骗用户窃取敏感信息，构成重大的网络安全威胁。机器学习(ML)和深度学习(DL)的进步导致了许多钓鱼网页检测解决方案的发展，但这些模型仍然容易受到对手攻击。评估它们对敌意网络钓鱼网页的健壮性是至关重要的。现有工具包含为有限数量的品牌预先设计的钓鱼网页的数据集，并且在钓鱼功能方面缺乏多样性。为了应对这些挑战，我们开发了PhishOracle，这是一个通过在合法网页中嵌入不同的钓鱼功能来生成敌意钓鱼网页的工具。我们评估了现有的三种特定任务的模型--Stack模型、VisualPhishNet和Phishpedia--对PhishOracle生成的敌意钓鱼网页的稳健性，并观察到它们的检测率显著下降。相比之下，基于多模式大语言模型(MLLM)的网络钓鱼检测器对这些敌意攻击表现出更强的稳健性，但仍然容易被规避。我们的发现突出了网络钓鱼检测模型对对手攻击的脆弱性，强调了需要更强大的检测方法。此外，我们还进行了一项用户研究，以评估PhishOracle生成的敌意钓鱼网页是否可以欺骗用户。结果表明，许多钓鱼网页不仅规避了现有的检测模型，而且还规避了用户。我们还开发了PhishOracle Web应用程序，允许用户输入合法的URL，选择相关的网络钓鱼功能并生成相应的网络钓鱼网页。所有资源都将在GitHub上公开提供。



## **30. JailGuard: A Universal Detection Framework for LLM Prompt-based Attacks**

JailGuard：针对LLM基于预算的攻击的通用检测框架 cs.CR

40 pages, 12 figures

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2312.10766v4) [paper-pdf](http://arxiv.org/pdf/2312.10766v4)

**Authors**: Xiaoyu Zhang, Cen Zhang, Tianlin Li, Yihao Huang, Xiaojun Jia, Ming Hu, Jie Zhang, Yang Liu, Shiqing Ma, Chao Shen

**Abstract**: The systems and software powered by Large Language Models (LLMs) and Multi-Modal LLMs (MLLMs) have played a critical role in numerous scenarios. However, current LLM systems are vulnerable to prompt-based attacks, with jailbreaking attacks enabling the LLM system to generate harmful content, while hijacking attacks manipulate the LLM system to perform attacker-desired tasks, underscoring the necessity for detection tools. Unfortunately, existing detecting approaches are usually tailored to specific attacks, resulting in poor generalization in detecting various attacks across different modalities. To address it, we propose JailGuard, a universal detection framework deployed on top of LLM systems for prompt-based attacks across text and image modalities. JailGuard operates on the principle that attacks are inherently less robust than benign ones. Specifically, JailGuard mutates untrusted inputs to generate variants and leverages the discrepancy of the variants' responses on the target model to distinguish attack samples from benign samples. We implement 18 mutators for text and image inputs and design a mutator combination policy to further improve detection generalization. The evaluation on the dataset containing 15 known attack types suggests that JailGuard achieves the best detection accuracy of 86.14%/82.90% on text and image inputs, outperforming state-of-the-art methods by 11.81%-25.73% and 12.20%-21.40%.

摘要: 由大型语言模型(LLM)和多模式LLM(MLLMS)支持的系统和软件在许多情况下发挥了关键作用。然而，当前的LLM系统容易受到基于提示的攻击，越狱攻击使LLM系统能够生成有害内容，而劫持攻击操纵LLM系统执行攻击者想要的任务，这突显了检测工具的必要性。遗憾的是，现有的检测方法通常是针对特定的攻击量身定做的，导致在检测不同模式的各种攻击时通用性较差。为了解决这个问题，我们提出了JailGuard，这是一个部署在LLM系统之上的通用检测框架，用于跨文本和图像通道的基于提示的攻击。JailGuard的运作原则是，攻击天生就不如良性攻击那么强大。具体地说，JailGuard会变异不可信的输入以生成变体，并利用变体对目标模型的响应差异来区分攻击样本和良性样本。我们为文本和图像输入实现了18个变异器，并设计了变异器组合策略，进一步提高了检测的泛化能力。对包含15种已知攻击类型的数据集的评估表明，JailGuard在文本和图像输入上的最佳检测准确率为86.14%/82.90%，分别比最先进的方法高11.81%-25.73%和12.20%-21.40%。



## **31. Making Every Step Effective: Jailbreaking Large Vision-Language Models Through Hierarchical KV Equalization**

让每一步都有效：通过分层KN均衡化破解大型视觉语言模型 cs.CV

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11750v1) [paper-pdf](http://arxiv.org/pdf/2503.11750v1)

**Authors**: Shuyang Hao, Yiwei Wang, Bryan Hooi, Jun Liu, Muhao Chen, Zi Huang, Yujun Cai

**Abstract**: In the realm of large vision-language models (LVLMs), adversarial jailbreak attacks serve as a red-teaming approach to identify safety vulnerabilities of these models and their associated defense mechanisms. However, we identify a critical limitation: not every adversarial optimization step leads to a positive outcome, and indiscriminately accepting optimization results at each step may reduce the overall attack success rate. To address this challenge, we introduce HKVE (Hierarchical Key-Value Equalization), an innovative jailbreaking framework that selectively accepts gradient optimization results based on the distribution of attention scores across different layers, ensuring that every optimization step positively contributes to the attack. Extensive experiments demonstrate HKVE's significant effectiveness, achieving attack success rates of 75.08% on MiniGPT4, 85.84% on LLaVA and 81.00% on Qwen-VL, substantially outperforming existing methods by margins of 20.43\%, 21.01\% and 26.43\% respectively. Furthermore, making every step effective not only leads to an increase in attack success rate but also allows for a reduction in the number of iterations, thereby lowering computational costs. Warning: This paper contains potentially harmful example data.

摘要: 在大型视觉语言模型领域，对抗性越狱攻击是一种识别这些模型及其相关防御机制的安全漏洞的红团队方法。然而，我们发现了一个关键的限制：并不是每个对抗性的优化步骤都会带来积极的结果，并且不分青红皂白地接受每个步骤的优化结果可能会降低总体攻击成功率。为了应对这一挑战，我们引入了HKVE(分层键值均衡)，这是一个创新的越狱框架，它根据注意力分数在不同层的分布来选择性地接受梯度优化结果，确保每一个优化步骤都对攻击有积极的贡献。大量实验表明，HKVE具有显著的攻击效果，在MiniGPT4、LLaVA和Qwen-VL上的攻击成功率分别达到75.08%、85.84%和81.00%，分别比现有方法高出20.43、21.01和26.43%。此外，使每一步都有效，不仅可以提高攻击成功率，还可以减少迭代次数，从而降低计算成本。警告：本文包含可能有害的示例数据。



## **32. Tit-for-Tat: Safeguarding Large Vision-Language Models Against Jailbreak Attacks via Adversarial Defense**

针锋相对：通过对抗性防御保护大型视觉语言模型免受越狱攻击 cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11619v1) [paper-pdf](http://arxiv.org/pdf/2503.11619v1)

**Authors**: Shuyang Hao, Yiwei Wang, Bryan Hooi, Ming-Hsuan Yang, Jun Liu, Chengcheng Tang, Zi Huang, Yujun Cai

**Abstract**: Deploying large vision-language models (LVLMs) introduces a unique vulnerability: susceptibility to malicious attacks via visual inputs. However, existing defense methods suffer from two key limitations: (1) They solely focus on textual defenses, fail to directly address threats in the visual domain where attacks originate, and (2) the additional processing steps often incur significant computational overhead or compromise model performance on benign tasks. Building on these insights, we propose ESIII (Embedding Security Instructions Into Images), a novel methodology for transforming the visual space from a source of vulnerability into an active defense mechanism. Initially, we embed security instructions into defensive images through gradient-based optimization, obtaining security instructions in the visual dimension. Subsequently, we integrate security instructions from visual and textual dimensions with the input query. The collaboration between security instructions from different dimensions ensures comprehensive security protection. Extensive experiments demonstrate that our approach effectively fortifies the robustness of LVLMs against such attacks while preserving their performance on standard benign tasks and incurring an imperceptible increase in time costs.

摘要: 部署大型视觉语言模型(LVLM)引入了一个独特的漏洞：通过视觉输入易受恶意攻击。然而，现有的防御方法有两个关键的局限性：(1)它们只关注文本防御，不能直接应对发起攻击的视域中的威胁；(2)额外的处理步骤往往会导致巨大的计算开销或对良性任务的模型性能造成影响。基于这些见解，我们提出了ESIII(Embedding Security Instructions To Images)，这是一种将视觉空间从易受攻击源转变为主动防御机制的新方法。首先，通过基于梯度的优化将安全指令嵌入到防御图像中，得到视觉维度的安全指令。随后，我们将来自视觉和文本维度的安全指令与输入查询集成在一起。来自不同维度的安全指令之间的协作确保了全面的安全防护。大量的实验表明，我们的方法有效地增强了LVLM对此类攻击的健壮性，同时保持了它们在标准良性任务上的性能，并导致了不可察觉的时间开销的增加。



## **33. Align in Depth: Defending Jailbreak Attacks via Progressive Answer Detoxification**

深度结盟：通过渐进式答案去规范化捍卫越狱袭击 cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11185v1) [paper-pdf](http://arxiv.org/pdf/2503.11185v1)

**Authors**: Yingjie Zhang, Tong Liu, Zhe Zhao, Guozhu Meng, Kai Chen

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks, which use crafted prompts to elicit toxic responses. These attacks exploit LLMs' difficulty in dynamically detecting harmful intents during the generation process. Traditional safety alignment methods, often relying on the initial few generation steps, are ineffective due to limited computational budget. This paper proposes DEEPALIGN, a robust defense framework that fine-tunes LLMs to progressively detoxify generated content, significantly improving both the computational budget and effectiveness of mitigating harmful generation. Our approach uses a hybrid loss function operating on hidden states to directly improve LLMs' inherent awareness of toxity during generation. Furthermore, we redefine safe responses by generating semantically relevant answers to harmful queries, thereby increasing robustness against representation-mutation attacks. Evaluations across multiple LLMs demonstrate state-of-the-art defense performance against six different attack types, reducing Attack Success Rates by up to two orders of magnitude compared to previous state-of-the-art defense while preserving utility. This work advances LLM safety by addressing limitations of conventional alignment through dynamic, context-aware mitigation.

摘要: 大型语言模型(LLM)容易受到越狱攻击，这些攻击使用精心编制的提示来引发有毒响应。这些攻击利用了LLMS在生成过程中动态检测有害意图的困难。传统的安全配准方法往往依赖于初始的几个生成步骤，由于计算预算有限，效率不高。本文提出了DEEPALIGN，这是一个健壮的防御框架，它对LLMS进行微调，以逐步对生成的内容进行解毒，显著提高了计算预算和减轻有害生成的有效性。我们的方法使用一种对隐态进行操作的混合损失函数来直接提高LLMS在生成过程中对毒性的固有意识。此外，我们通过生成对有害查询的语义相关答案来重新定义安全响应，从而提高了对表示-突变攻击的健壮性。对多个LLM的评估显示了针对六种不同攻击类型的最先进的防御性能，与以前最先进的防御相比，攻击成功率降低了高达两个数量级，同时保留了实用性。这项工作通过动态、上下文感知的缓解来解决传统对准的局限性，从而提高了LLM的安全性。



## **34. Towards a Systematic Evaluation of Hallucinations in Large-Vision Language Models**

对大视野语言模型中的幻觉进行系统评估 cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2412.20622v2) [paper-pdf](http://arxiv.org/pdf/2412.20622v2)

**Authors**: Ashish Seth, Dinesh Manocha, Chirag Agarwal

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable performance in complex multimodal tasks. However, these models still suffer from hallucinations, particularly when required to implicitly recognize or infer diverse visual entities from images for complex vision-language tasks. To address this challenge, we propose HALLUCINOGEN, a novel visual question answering (VQA) benchmark that employs contextual reasoning prompts as hallucination attacks to evaluate the extent of hallucination in state-of-the-art LVLMs. Our benchmark provides a comprehensive study of the implicit reasoning capabilities of these models by first categorizing visual entities based on the ease of recognition in an image as either salient (prominent, visibly recognizable objects such as a car) or latent entities (such as identifying a disease from a chest X-ray), which are not readily visible and require domain knowledge or contextual reasoning for accurate inference. Next, we design hallucination attacks for both types of entities to assess hallucinations in LVLMs while performing various vision-language tasks, such as locating or reasoning about specific entities within an image, where models must perform implicit reasoning by verifying the existence of the queried entity within the image before generating responses. Finally, our extensive evaluations of eleven LVLMs, including powerful open-source models (like LLaMA-3.2 and DeepSeek-V2), commercial models like Gemini, and two hallucination mitigation strategies across multiple datasets, demonstrate that current LVLMs remain susceptible to hallucination attacks.

摘要: 大型视觉语言模型在复杂的多通道任务中表现出显著的性能。然而，这些模型仍然存在幻觉，特别是在复杂的视觉语言任务中需要从图像中隐含地识别或推断不同的视觉实体时。为了应对这一挑战，我们提出了幻觉剂，这是一种新颖的视觉问答基准，它使用上下文推理提示作为幻觉攻击来评估最先进的LVLM中的幻觉程度。我们的基准提供了对这些模型的隐含推理能力的全面研究，首先根据图像中识别的容易程度将视觉实体分类为突出实体(突出的、可视识别的对象，如汽车)或潜在实体(如从胸部X光识别疾病)，这些实体不容易看到，需要领域知识或上下文推理才能进行准确的推理。接下来，我们为两种类型的实体设计幻觉攻击，以评估LVLMS中的幻觉，同时执行各种视觉语言任务，如定位或关于图像中的特定实体进行推理，其中模型必须在生成响应之前通过验证被查询实体在图像中的存在来执行隐式推理。最后，我们对11个LVLM进行了广泛的评估，包括强大的开源模型(如Llama-3.2和DeepSeek-V2)、商业模型(如Gemini)，以及跨多个数据集的两种幻觉缓解策略，表明当前的LVLM仍然容易受到幻觉攻击。



## **35. ChatGPT Encounters Morphing Attack Detection: Zero-Shot MAD with Multi-Modal Large Language Models and General Vision Models**

ChatGPT遭遇变形攻击检测：具有多模式大语言模型和通用视觉模型的零镜头MAD cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10937v1) [paper-pdf](http://arxiv.org/pdf/2503.10937v1)

**Authors**: Haoyu Zhang, Raghavendra Ramachandra, Kiran Raja, Christoph Busch

**Abstract**: Face Recognition Systems (FRS) are increasingly vulnerable to face-morphing attacks, prompting the development of Morphing Attack Detection (MAD) algorithms. However, a key challenge in MAD lies in its limited generalizability to unseen data and its lack of explainability-critical for practical application environments such as enrolment stations and automated border control systems. Recognizing that most existing MAD algorithms rely on supervised learning paradigms, this work explores a novel approach to MAD using zero-shot learning leveraged on Large Language Models (LLMs). We propose two types of zero-shot MAD algorithms: one leveraging general vision models and the other utilizing multimodal LLMs. For general vision models, we address the MAD task by computing the mean support embedding of an independent support set without using morphed images. For the LLM-based approach, we employ the state-of-the-art GPT-4 Turbo API with carefully crafted prompts. To evaluate the feasibility of zero-shot MAD and the effectiveness of the proposed methods, we constructed a print-scan morph dataset featuring various unseen morphing algorithms, simulating challenging real-world application scenarios. Experimental results demonstrated notable detection accuracy, validating the applicability of zero-shot learning for MAD tasks. Additionally, our investigation into LLM-based MAD revealed that multimodal LLMs, such as ChatGPT, exhibit remarkable generalizability to untrained MAD tasks. Furthermore, they possess a unique ability to provide explanations and guidance, which can enhance transparency and usability for end-users in practical applications.

摘要: 人脸识别系统(FRS)越来越容易受到变形攻击，这促使变形攻击检测(MAD)算法的发展。然而，MAD的一个关键挑战在于，它对看不见的数据的概括性有限，而且缺乏可解释性--这对于招生站和自动边界控制系统等实际应用环境至关重要。认识到大多数现有的MAD算法依赖于有监督的学习范例，该工作探索了一种利用大型语言模型(LLMS)的零镜头学习来实现MAD的新方法。我们提出了两种类型的零镜头MAD算法：一种利用一般的视觉模型，另一种利用多模式LLMS。对于一般的视觉模型，我们通过计算独立支持集的平均支持嵌入来解决MAD任务，而不使用变形图像。对于基于LLM的方法，我们使用了最先进的GPT-4 Turbo API和精心制作的提示。为了评估零镜头MAD的可行性和提出的方法的有效性，我们构建了一个包含各种不可见变形算法的打印扫描变形数据集，模拟了具有挑战性的真实应用场景。实验结果显示了显著的检测精度，验证了零镜头学习在MAD任务中的适用性。此外，我们对基于LLM的MAD的研究表明，多通道LLM，如ChatGPT，对未经训练的MAD任务表现出显著的泛化能力。此外，它们还具有提供解释和指导的独特能力，这可以在实际应用中提高最终用户的透明度和可用性。



## **36. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

令人沮丧的简单但高效的攻击基线：针对GPT-4.5/4 o/o 1的强黑匣子模型的成功率超过90% cs.CV

Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10635v1) [paper-pdf](http://arxiv.org/pdf/2503.10635v1)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against black-box commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we notice that identifying core semantic objects is a key objective for models trained with various datasets and methodologies. This insight motivates our approach that refines semantic clarity by encoding explicit semantic details within local regions, thus ensuring interoperability and capturing finer-grained features, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective solution: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5-sonnet, Claude-3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods. Our optimized adversarial examples under different configurations and training code are available at https://github.com/VILA-Lab/M-Attack.

摘要: 尽管在开源的大型视觉语言模型(LVLM)上性能很好，但基于传输的定向攻击对黑盒商业LVLM往往失败。分析失败的对抗性扰动发现，学习的扰动通常源于均匀分布，缺乏明确的语义细节，导致意外响应。这种严重的语义信息缺失导致商业LVLM要么完全忽略该扰动，要么曲解其嵌入的语义，从而导致攻击失败。为了克服这些问题，我们注意到，识别核心语义对象是用各种数据集和方法训练的模型的关键目标。这种洞察力激发了我们的方法，通过在局部区域内编码显式的语义细节来细化语义清晰度，从而确保互操作性和捕获更细粒度的特征，并通过将修改集中在语义丰富的区域而不是统一地应用它们。为了实现这一点，我们提出了一种简单而高效的解决方案：在每个优化步骤中，通过控制纵横比和比例来随机裁剪敌对图像，调整大小，然后在嵌入空间中与目标图像对齐。实验结果证实了我们的假设。我们的对抗性例子使用聚焦于关键区域的局部聚集扰动来制作，表现出出奇地好的可转换性，可以用于商业LVLM，包括GPT-4.5、GPT-40、Gemini-2.0-Flash、Claude-3.5-十四行诗、Claude-3.7-十四行诗，甚至像o1、Claude-3.7-Think和Gemini-2.0-Flash-Think这样的推理模型。我们的方法在GPT-4.5、40o和o1上的成功率超过90%，大大超过了之前所有最先进的攻击方法。我们在不同配置和训练代码下的优化对抗性示例可在https://github.com/VILA-Lab/M-Attack.获得



## **37. ASIDE: Architectural Separation of Instructions and Data in Language Models**

ASIDE：语言模型中指令和数据的架构分离 cs.LG

ICLR 2025 Workshop on Building Trust in Language Models and  Applications

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10566v1) [paper-pdf](http://arxiv.org/pdf/2503.10566v1)

**Authors**: Egor Zverev, Evgenii Kortukov, Alexander Panfilov, Soroush Tabesh, Alexandra Volkova, Sebastian Lapuschkin, Wojciech Samek, Christoph H. Lampert

**Abstract**: Despite their remarkable performance, large language models lack elementary safety features, and this makes them susceptible to numerous malicious attacks. In particular, previous work has identified the absence of an intrinsic separation between instructions and data as a root cause for the success of prompt injection attacks. In this work, we propose an architectural change, ASIDE, that allows the model to clearly separate between instructions and data by using separate embeddings for them. Instead of training the embeddings from scratch, we propose a method to convert an existing model to ASIDE form by using two copies of the original model's embeddings layer, and applying an orthogonal rotation to one of them. We demonstrate the effectiveness of our method by showing (1) highly increased instruction-data separation scores without a loss in model capabilities and (2) competitive results on prompt injection benchmarks, even without dedicated safety training. Additionally, we study the working mechanism behind our method through an analysis of model representations.

摘要: 尽管性能卓越，但大型语言模型缺乏基本的安全功能，这使得它们容易受到大量恶意攻击。特别是，以前的工作已经确定指令和数据之间没有内在的分离是快速注入攻击成功的根本原因。在这项工作中，我们提出了一种架构改变，允许模型通过对指令和数据使用单独的嵌入来明确地分离它们。我们没有从头开始训练嵌入，而是提出了一种方法，通过使用原始模型的嵌入层的两个副本并对其中一个模型应用正交旋转来将现有模型转换为旁注形式。我们通过展示(1)在不损失模型能力的情况下极大地提高指令-数据分离分数和(2)即使在没有专门的安全培训的情况下也在快速注入基准上具有竞争力的结果来证明我们方法的有效性。此外，我们还通过对模型表示的分析，研究了该方法背后的工作机制。



## **38. TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors**

TH-Bench：通过机器生成文本检测器上人性化人工智能文本来评估逃避攻击 cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.08708v2) [paper-pdf](http://arxiv.org/pdf/2503.08708v2)

**Authors**: Jingyi Zheng, Junfeng Wang, Zhen Sun, Wenhan Dong, Yule Liu, Xinlei He

**Abstract**: As Large Language Models (LLMs) advance, Machine-Generated Texts (MGTs) have become increasingly fluent, high-quality, and informative. Existing wide-range MGT detectors are designed to identify MGTs to prevent the spread of plagiarism and misinformation. However, adversaries attempt to humanize MGTs to evade detection (named evading attacks), which requires only minor modifications to bypass MGT detectors. Unfortunately, existing attacks generally lack a unified and comprehensive evaluation framework, as they are assessed using different experimental settings, model architectures, and datasets. To fill this gap, we introduce the Text-Humanization Benchmark (TH-Bench), the first comprehensive benchmark to evaluate evading attacks against MGT detectors. TH-Bench evaluates attacks across three key dimensions: evading effectiveness, text quality, and computational overhead. Our extensive experiments evaluate 6 state-of-the-art attacks against 13 MGT detectors across 6 datasets, spanning 19 domains and generated by 11 widely used LLMs. Our findings reveal that no single evading attack excels across all three dimensions. Through in-depth analysis, we highlight the strengths and limitations of different attacks. More importantly, we identify a trade-off among three dimensions and propose two optimization insights. Through preliminary experiments, we validate their correctness and effectiveness, offering potential directions for future research.

摘要: 随着大型语言模型(LLM)的发展，机器生成文本(MGTS)变得越来越流畅、高质量和信息丰富。现有的大范围MGT探测器旨在识别MGT，以防止抄袭和错误信息的传播。然而，攻击者试图使MGTS人性化以躲避检测(称为躲避攻击)，这只需要进行少量修改即可绕过MGT检测器。遗憾的是，现有的攻击通常缺乏统一和全面的评估框架，因为它们是使用不同的实验设置、模型架构和数据集进行评估的。为了填补这一空白，我们引入了文本人性化基准(TH-BENCH)，这是第一个评估针对MGT检测器的躲避攻击的综合基准。TH-BENCH从三个关键维度对攻击进行评估：规避效率、文本质量和计算开销。我们的广泛实验评估了6种针对6个数据集的13个MGT检测器的最先进攻击，这些攻击跨越19个域，由11个广泛使用的LLM生成。我们的发现表明，没有一次逃避攻击在所有三个维度上都表现出色。通过深入的分析，我们突出了不同攻击的优势和局限性。更重要的是，我们确定了三个维度之间的权衡，并提出了两个优化见解。通过初步实验，验证了它们的正确性和有效性，为以后的研究提供了潜在的方向。



## **39. Prompt Inversion Attack against Collaborative Inference of Large Language Models**

针对大型语言模型协作推理的提示倒置攻击 cs.CR

To appear at IEEE Symposium on Security and Privacy 2025

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.09022v2) [paper-pdf](http://arxiv.org/pdf/2503.09022v2)

**Authors**: Wenjie Qu, Yuguang Zhou, Yongji Wu, Tingsong Xiao, Binhang Yuan, Yiming Li, Jiaheng Zhang

**Abstract**: Large language models (LLMs) have been widely applied for their remarkable capability of content generation. However, the practical use of open-source LLMs is hindered by high resource requirements, making deployment expensive and limiting widespread development. The collaborative inference is a promising solution for this problem, in which users collaborate by each hosting a subset of layers and transmitting intermediate activation. Many companies are building collaborative inference platforms to reduce LLM serving costs, leveraging users' underutilized GPUs. Despite widespread interest in collaborative inference within academia and industry, the privacy risks associated with LLM collaborative inference have not been well studied. This is largely because of the challenge posed by inverting LLM activation due to its strong non-linearity.   In this paper, to validate the severity of privacy threats in LLM collaborative inference, we introduce the concept of prompt inversion attack (PIA), where a malicious participant intends to recover the input prompt through the activation transmitted by its previous participant. Extensive experiments show that our PIA method substantially outperforms existing baselines. For example, our method achieves an 88.4\% token accuracy on the Skytrax dataset with the Llama-65B model when inverting the maximum number of transformer layers, while the best baseline method only achieves 22.8\% accuracy. The results verify the effectiveness of our PIA attack and highlights its practical threat to LLM collaborative inference systems.

摘要: 大语言模型以其卓越的内容生成能力得到了广泛的应用。然而，开源LLMS的实际使用受到高资源要求的阻碍，使得部署成本高昂，并限制了广泛的发展。协作推理是解决这一问题的一种很有前途的解决方案，其中每个用户通过托管一个层的子集并传递中间激活来进行协作。许多公司正在构建协作推理平台，以利用用户未充分利用的GPU来降低LLM服务成本。尽管学术界和工业界对协同推理产生了广泛的兴趣，但与LLM协同推理相关的隐私风险尚未得到很好的研究。这在很大程度上是因为LLM激活的反转带来了挑战，因为它具有很强的非线性。为了验证LLM协同推理中隐私威胁的严重性，我们引入了即时反转攻击(PIA)的概念，恶意参与者试图通过其先前参与者发送的激活来恢复输入提示。广泛的实验表明，我们的PIA方法的性能大大优于现有的基线。例如，我们的方法在使用Llama-65B模型的Skytrax数据集上，在反演最大变压器层数时达到了88.4%的令牌精度，而最佳基线方法只达到了22.8%的精度。实验结果验证了PIA攻击的有效性，并突出了其对LLM协同推理系统的实际威胁。



## **40. ExtremeAIGC: Benchmarking LMM Vulnerability to AI-Generated Extremist Content**

ExtremeAIGC：LMM漏洞针对人工智能生成的极端主义内容进行基准测试 cs.CR

Preprint

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.09964v1) [paper-pdf](http://arxiv.org/pdf/2503.09964v1)

**Authors**: Bhavik Chandna, Mariam Aboujenane, Usman Naseem

**Abstract**: Large Multimodal Models (LMMs) are increasingly vulnerable to AI-generated extremist content, including photorealistic images and text, which can be used to bypass safety mechanisms and generate harmful outputs. However, existing datasets for evaluating LMM robustness offer limited exploration of extremist content, often lacking AI-generated images, diverse image generation models, and comprehensive coverage of historical events, which hinders a complete assessment of model vulnerabilities. To fill this gap, we introduce ExtremeAIGC, a benchmark dataset and evaluation framework designed to assess LMM vulnerabilities against such content. ExtremeAIGC simulates real-world events and malicious use cases by curating diverse text- and image-based examples crafted using state-of-the-art image generation techniques. Our study reveals alarming weaknesses in LMMs, demonstrating that even cutting-edge safety measures fail to prevent the generation of extremist material. We systematically quantify the success rates of various attack strategies, exposing critical gaps in current defenses and emphasizing the need for more robust mitigation strategies.

摘要: 大型多模式模型(LMM)越来越容易受到人工智能生成的极端主义内容的影响，包括照片级真实感图像和文本，这些内容可用于绕过安全机制并产生有害输出。然而，现有的用于评估LMM稳健性的数据集提供了对极端主义内容的有限探索，往往缺乏人工智能生成的图像、多样化的图像生成模型以及对历史事件的全面覆盖，这阻碍了对模型漏洞的完整评估。为了填补这一空白，我们引入了ExtremeAIGC，这是一个基准数据集和评估框架，旨在评估针对此类内容的LMM漏洞。ExtremeAIGC通过策划使用最先进的图像生成技术制作的各种基于文本和图像的示例来模拟真实世界的事件和恶意使用案例。我们的研究揭示了LMM令人震惊的弱点，表明即使是尖端的安全措施也无法防止极端主义材料的产生。我们系统地量化了各种攻击策略的成功率，暴露了当前防御中的关键差距，并强调了需要更强大的缓解策略。



## **41. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

时间序列预测大型语言模型中的对抗漏洞 cs.LG

AISTATS 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2412.08099v4) [paper-pdf](http://arxiv.org/pdf/2412.08099v4)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like LLMTime with GPT-3.5, GPT-4, LLaMa, and Mistral, TimeGPT, and TimeLLM show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications. The code repository can be found at https://github.com/JohnsonJiang1996/AdvAttack_LLM4TS.

摘要: 大型语言模型(LLM)最近在时间序列预测方面显示出巨大的潜力，在处理复杂的时间数据方面提供了令人印象深刻的能力。然而，它们在实际应用中的健壮性和可靠性仍然没有得到充分的研究，特别是关于它们对对手攻击的敏感性。本文提出了一种基于LLM的时间序列预测的对抗性攻击框架。通过使用无梯度和黑盒优化方法，我们产生了最小但高效的扰动，这些扰动显著降低了跨多个数据集和LLM体系结构的预测精度。我们的实验包括LLMTime与GPT-3.5、GPT-4、Llama和Mistral、TimeGPT和TimeLLM的模型，实验表明，对抗性攻击导致的性能降级比随机噪声严重得多，并证明了我们的攻击在不同LLM上的广泛有效性。这些结果强调了低层管理在时间序列预测中的关键弱点，强调了需要强大的防御机制来确保其在实际应用中的可靠部署。代码存储库可在https://github.com/JohnsonJiang1996/AdvAttack_LLM4TS.上找到



## **42. CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data**

CyberLLMDirecct：使用网络安全数据分析精调LLM安全性的新数据集 cs.CR

The paper is submitted to "The 48th International ACM SIGIR  Conference on Research and Development in Information Retrieval" and is  currently under review

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09334v1) [paper-pdf](http://arxiv.org/pdf/2503.09334v1)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. All scripts required to reproduce the dataset, along with examples and relevant resources for replicating our results, will be made available upon the paper's acceptance.

摘要: 将大型语言模型(LLM)集成到网络安全应用程序中提供了重大机遇，如增强威胁分析和恶意软件检测，但也可能带来重大风险和安全问题，包括个人数据泄露和新恶意软件的自动生成。为了应对这些挑战，我们开发了CyberLLMInstruct，这是一个包含54,928个指令-响应对的数据集，涵盖了网络安全任务，如恶意软件分析、网络钓鱼模拟和零日漏洞。该数据集是通过多阶段过程构建的。这包括从多个来源获取数据，将其过滤和组织成指令-响应对，并使其与现实世界的情景相一致，以增强其适用性。选择了7个开源LLM来测试CyberLLMInstruct的有用性：Phi 3 Mini 3.8B、Mistral 7B、Qwen 2.5 7B、Llama 3 8B、Llama 3.1 8B、Gema 2 9B和Llama 2 70B。在我们的主要示例中，我们使用OWASP TOP 10框架严格评估微调模型的安全性，发现微调降低了所有测试的LLM和每个对手攻击的安全弹性(例如，针对即时注入的Llama 3.1 8B的安全分数从0.95下降到0.15)。在我们的第二个示例中，我们展示了这些相同的微调模型在CyberMetric基准上也可以达到高达92.50%的准确率。这些发现突出了性能和安全之间的权衡，显示了对抗性测试和进一步研究微调方法的重要性，这些方法可以降低安全风险，同时仍然提高不同数据集和域的性能。复制数据集所需的所有脚本，以及复制我们的结果的例子和相关资源，将在论文被接受时提供。



## **43. Prompt Inference Attack on Distributed Large Language Model Inference Frameworks**

对分布式大型语言模型推理框架的提示推理攻击 cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09291v1) [paper-pdf](http://arxiv.org/pdf/2503.09291v1)

**Authors**: Xinjian Luo, Ting Yu, Xiaokui Xiao

**Abstract**: The inference process of modern large language models (LLMs) demands prohibitive computational resources, rendering them infeasible for deployment on consumer-grade devices. To address this limitation, recent studies propose distributed LLM inference frameworks, which employ split learning principles to enable collaborative LLM inference on resource-constrained hardware. However, distributing LLM layers across participants requires the transmission of intermediate outputs, which may introduce privacy risks to the original input prompts - a critical issue that has yet to be thoroughly explored in the literature.   In this paper, we rigorously examine the privacy vulnerabilities of distributed LLM inference frameworks by designing and evaluating three prompt inference attacks aimed at reconstructing input prompts from intermediate LLM outputs. These attacks are developed under various query and data constraints to reflect diverse real-world LLM service scenarios. Specifically, the first attack assumes an unlimited query budget and access to an auxiliary dataset sharing the same distribution as the target prompts. The second attack also leverages unlimited queries but uses an auxiliary dataset with a distribution differing from the target prompts. The third attack operates under the most restrictive scenario, with limited query budgets and no auxiliary dataset available. We evaluate these attacks on a range of LLMs, including state-of-the-art models such as Llama-3.2 and Phi-3.5, as well as widely-used models like GPT-2 and BERT for comparative analysis. Our experiments show that the first two attacks achieve reconstruction accuracies exceeding 90%, while the third achieves accuracies typically above 50%, even under stringent constraints. These findings highlight privacy risks in distributed LLM inference frameworks, issuing a strong alert on their deployment in real-world applications.

摘要: 现代大型语言模型(LLM)的推理过程需要令人望而却步的计算资源，这使得它们不适合在消费级设备上部署。为了解决这一局限性，最近的研究提出了分布式LLM推理框架，该框架利用分裂学习原理在资源受限的硬件上实现协作式LLM推理。然而，在参与者之间分发LLM层需要传输中间输出，这可能会给原始输入提示带来隐私风险-这是一个尚未在文献中彻底探讨的关键问题。本文通过设计和评估三种旨在从中间LLM输出重构输入提示的提示推理攻击，严格检查了分布式LLM推理框架的隐私漏洞。这些攻击是在各种查询和数据约束下开发的，以反映现实世界中不同的LLM服务场景。具体地说，第一次攻击假定无限制的查询预算和对与目标提示共享相同分布的辅助数据集的访问。第二个攻击也利用无限查询，但使用的辅助数据集具有与目标提示不同的分布。第三种攻击是在限制最严格的情况下进行的，查询预算有限，并且没有可用的辅助数据集。我们在一系列LLM上评估了这些攻击，包括最先进的模型，如Llama-3.2和Phi-3.5，以及广泛使用的模型，如GPT-2和BERT，以进行比较分析。我们的实验表明，前两种攻击的重建精度超过90%，而第三种攻击的重建精度通常在50%以上，即使在严格的约束下也是如此。这些发现突出了分布式LLM推理框架中的隐私风险，对它们在现实世界应用程序中的部署发出了强烈的警告。



## **44. In-Context Defense in Computer Agents: An Empirical Study**

计算机代理中的上下文防御：实证研究 cs.AI

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09241v1) [paper-pdf](http://arxiv.org/pdf/2503.09241v1)

**Authors**: Pei Yang, Hai Ci, Mike Zheng Shou

**Abstract**: Computer agents powered by vision-language models (VLMs) have significantly advanced human-computer interaction, enabling users to perform complex tasks through natural language instructions. However, these agents are vulnerable to context deception attacks, an emerging threat where adversaries embed misleading content into the agent's operational environment, such as a pop-up window containing deceptive instructions. Existing defenses, such as instructing agents to ignore deceptive elements, have proven largely ineffective. As the first systematic study on protecting computer agents, we introduce textbf{in-context defense}, leveraging in-context learning and chain-of-thought (CoT) reasoning to counter such attacks. Our approach involves augmenting the agent's context with a small set of carefully curated exemplars containing both malicious environments and corresponding defensive responses. These exemplars guide the agent to first perform explicit defensive reasoning before action planning, reducing susceptibility to deceptive attacks. Experiments demonstrate the effectiveness of our method, reducing attack success rates by 91.2% on pop-up window attacks, 74.6% on average on environment injection attacks, while achieving 100% successful defenses against distracting advertisements. Our findings highlight that (1) defensive reasoning must precede action planning for optimal performance, and (2) a minimal number of exemplars (fewer than three) is sufficient to induce an agent's defensive behavior.

摘要: 由视觉语言模型(VLM)驱动的计算机代理极大地促进了人机交互，使用户能够通过自然语言指令执行复杂的任务。然而，这些代理容易受到上下文欺骗攻击，这是一种新兴的威胁，即对手在代理的操作环境中嵌入误导性内容，例如包含欺骗性指令的弹出窗口。现有的防御措施，如指示特工忽略欺骗性因素，已被证明在很大程度上无效。作为第一个关于保护计算机代理的系统研究，我们引入了Textbf[上下文中的防御}，利用上下文中的学习和思想链(COT)推理来对抗此类攻击。我们的方法涉及用一小组精心挑选的样本来增强代理的上下文，这些样本既包含恶意环境，也包含相应的防御响应。这些样本指导代理在行动计划之前首先执行明确的防御推理，降低对欺骗性攻击的敏感性。实验证明了该方法的有效性，对弹出窗口攻击的攻击成功率降低了91.2%，对环境注入攻击的平均成功率降低了74.6%，同时对分散注意力的广告实现了100%的成功防御。我们的发现强调：(1)为了获得最佳性能，防御推理必须先于行动计划，(2)最少的样本数量(少于3个)足以诱导代理人的防御行为。



## **45. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

DetectRL：在现实世界场景中对LLM生成的文本检测进行基准测试 cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2410.23746v3) [paper-pdf](http://arxiv.org/pdf/2410.23746v3)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.

摘要: 检测由大型语言模型(LLM)生成的文本是最近非常感兴趣的问题。有了像DetectGPT这样的零射击方法，检测能力已经达到了令人印象深刻的水平。然而，现有探测器在实际应用中的可靠性仍然没有得到充分的探索。在这项研究中，我们提出了一个新的基准，DetectRL，强调即使是最先进的(SOTA)检测技术在这项任务中仍然表现不佳。我们从LLM特别容易被滥用的领域收集了人类编写的数据集。使用流行的LLM，我们生成的数据更好地与现实世界的应用程序保持一致。与以前的研究不同，我们使用启发式规则来创建对抗性LLM生成的文本，模拟各种提示用法、人工修改(如单词替换)和书写噪音(如拼写错误)。我们对DetectRL的开发揭示了当前SOTA探测器的优势和局限性。更重要的是，我们分析了写作风格、模型类型、攻击方法、文本长度和真实世界中的人类写作因素对不同类型检测器的潜在影响。我们相信，DetectRL可以作为评估真实世界场景中检测器的有效基准，随着先进攻击方法的发展，从而提供更有压力的评估，以推动更高效检测器的开发。数据和代码可在以下网址公开获得：https://github.com/NLP2CT/DetectRL.



## **46. A Survey on Trustworthy LLM Agents: Threats and Countermeasures**

值得信赖的LLM代理人调查：威胁与对策 cs.MA

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09648v1) [paper-pdf](http://arxiv.org/pdf/2503.09648v1)

**Authors**: Miao Yu, Fanci Meng, Xinyun Zhou, Shilong Wang, Junyuan Mao, Linsey Pang, Tianlong Chen, Kun Wang, Xinfeng Li, Yongfeng Zhang, Bo An, Qingsong Wen

**Abstract**: With the rapid evolution of Large Language Models (LLMs), LLM-based agents and Multi-agent Systems (MAS) have significantly expanded the capabilities of LLM ecosystems. This evolution stems from empowering LLMs with additional modules such as memory, tools, environment, and even other agents. However, this advancement has also introduced more complex issues of trustworthiness, which previous research focused solely on LLMs could not cover. In this survey, we propose the TrustAgent framework, a comprehensive study on the trustworthiness of agents, characterized by modular taxonomy, multi-dimensional connotations, and technical implementation. By thoroughly investigating and summarizing newly emerged attacks, defenses, and evaluation methods for agents and MAS, we extend the concept of Trustworthy LLM to the emerging paradigm of Trustworthy Agent. In TrustAgent, we begin by deconstructing and introducing various components of the Agent and MAS. Then, we categorize their trustworthiness into intrinsic (brain, memory, and tool) and extrinsic (user, agent, and environment) aspects. Subsequently, we delineate the multifaceted meanings of trustworthiness and elaborate on the implementation techniques of existing research related to these internal and external modules. Finally, we present our insights and outlook on this domain, aiming to provide guidance for future endeavors.

摘要: 随着大型语言模型的快速发展，基于大型语言模型的智能体和多智能体系统极大地扩展了大型语言模型生态系统的能力。这种发展源于向LLM提供额外的模块，如内存、工具、环境，甚至其他代理。然而，这一进步也引入了更复杂的可信度问题，以前仅关注LLMS的研究无法涵盖这些问题。在本次调查中，我们提出了TrustAgent框架，这是一项关于代理可信性的综合研究，具有模块化分类、多维内涵和技术实现的特点。通过深入研究和总结新出现的针对代理和多代理系统的攻击、防御和评估方法，我们将可信LLM的概念扩展到新兴的可信代理范式。在TrustAgent中，我们从解构和引入代理和MAS的各种组件开始。然后，我们将他们的可信性分为内在(大脑、记忆和工具)和外在(用户、代理和环境)两个方面。随后，我们描述了可信性的多方面含义，并详细阐述了与这些内部和外部模块相关的现有研究的实现技术。最后，我们提出了我们对这一领域的见解和展望，旨在为未来的努力提供指导。



## **47. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

探索LLM中的潜在子空间以实现人工智能安全：识别和操纵敌对状态 cs.LG

4 figures

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09066v1) [paper-pdf](http://arxiv.org/pdf/2503.09066v1)

**Authors**: Xin Wei Chia, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.

摘要: 大型语言模型(LLM)在各种任务中表现出了非凡的能力，但它们仍然容易受到对手操纵，例如通过即时注入攻击越狱。这些攻击绕过安全机制，生成受限或有害的内容。在这项研究中，我们通过从LLM中提取隐藏激活来研究安全状态和越狱状态的潜在潜在子空间。受神经科学中吸引子动力学的启发，我们假设LLM的激活稳定在半稳定状态，可以被识别和扰动以诱导状态转变。使用降维技术，我们从安全和越狱反应中投影激活，以揭示低维空间中的潜在子空间。然后我们推导出一个扰动向量，当应用于安全表示时，将模型转变为越狱状态。我们的结果表明，这种因果干预导致在提示的子集中有统计意义的越狱反应。接下来，我们探索这些扰动是如何通过模型的各层传播的，测试诱导的状态变化是保持局部化还是在整个网络中级联。我们的发现表明，有针对性的扰动导致激活和模型反应的明显转变。我们的方法为潜在的主动防御铺平了道路，从传统的基于护栏的方法转变为先发制人的、模型不可知的技术，在表征水平上中和敌对状态。



## **48. Battling Misinformation: An Empirical Study on Adversarial Factuality in Open-Source Large Language Models**

与错误信息作斗争：开源大型语言模型中对抗事实的实证研究 cs.CL

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.10690v1) [paper-pdf](http://arxiv.org/pdf/2503.10690v1)

**Authors**: Shahnewaz Karim Sakib, Anindya Bijoy Das, Shibbir Ahmed

**Abstract**: Adversarial factuality refers to the deliberate insertion of misinformation into input prompts by an adversary, characterized by varying levels of expressed confidence. In this study, we systematically evaluate the performance of several open-source large language models (LLMs) when exposed to such adversarial inputs. Three tiers of adversarial confidence are considered: strongly confident, moderately confident, and limited confidence. Our analysis encompasses eight LLMs: LLaMA 3.1 (8B), Phi 3 (3.8B), Qwen 2.5 (7B), Deepseek-v2 (16B), Gemma2 (9B), Falcon (7B), Mistrallite (7B), and LLaVA (7B). Empirical results indicate that LLaMA 3.1 (8B) exhibits a robust capability in detecting adversarial inputs, whereas Falcon (7B) shows comparatively lower performance. Notably, for the majority of the models, detection success improves as the adversary's confidence decreases; however, this trend is reversed for LLaMA 3.1 (8B) and Phi 3 (3.8B), where a reduction in adversarial confidence corresponds with diminished detection performance. Further analysis of the queries that elicited the highest and lowest rates of successful attacks reveals that adversarial attacks are more effective when targeting less commonly referenced or obscure information.

摘要: 对抗性真实性是指敌手故意在输入提示中插入错误信息，其特征是表现出不同程度的自信。在这项研究中，我们系统地评估了几个开放源码的大型语言模型(LLM)在面对这种敌意输入时的性能。我们考虑了三个等级的对手自信：强烈自信、适度自信和有限自信。我们的分析包括8个LLMS：骆驼3.1(8B)、Phi 3(3.8B)、Qwen 2.5(7B)、DeepSeek-v2(16B)、Gemma2(9B)、Falcon(7B)、Mstrallite(7B)和LLaVA(7B)。实验结果表明，Llama3.1(8B)表现出较强的检测敌意输入的能力，而Falcon(7B)表现出相对较低的性能。值得注意的是，对于大多数模型，检测成功率随着对手信心的降低而提高；然而，对于大羊驼3.1(8B)和Phi 3(3.8B)，这一趋势相反，其中对手信心的降低与检测性能的降低相对应。对导致最高和最低成功率的查询的进一步分析表明，对抗性攻击在针对不太常被引用或晦涩难懂的信息时更有效。



## **49. JBFuzz: Jailbreaking LLMs Efficiently and Effectively Using Fuzzing**

JBFuzz：使用Fuzzing高效、有效地越狱LLM cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.08990v1) [paper-pdf](http://arxiv.org/pdf/2503.08990v1)

**Authors**: Vasudev Gohil

**Abstract**: Large language models (LLMs) have shown great promise as language understanding and decision making tools, and they have permeated various aspects of our everyday life. However, their widespread availability also comes with novel risks, such as generating harmful, unethical, or offensive content, via an attack called jailbreaking. Despite extensive efforts from LLM developers to align LLMs using human feedback, they are still susceptible to jailbreak attacks. To tackle this issue, researchers often employ red-teaming to understand and investigate jailbreak prompts. However, existing red-teaming approaches lack effectiveness, scalability, or both. To address these issues, we propose JBFuzz, a novel effective, automated, and scalable red-teaming technique for jailbreaking LLMs.   JBFuzz is inspired by the success of fuzzing for detecting bugs/vulnerabilities in software. We overcome three challenges related to effectiveness and scalability by devising novel seed prompts, a lightweight mutation engine, and a lightweight and accurate evaluator for guiding the fuzzer. Assimilating all three solutions results in a potent fuzzer that only requires black-box access to the target LLM. We perform extensive experimental evaluation of JBFuzz using nine popular and widely-used LLMs. We find that JBFuzz successfully jailbreaks all LLMs for various harmful/unethical questions, with an average attack success rate of 99%. We also find that JBFuzz is extremely efficient as it jailbreaks a given LLM for a given question in 60 seconds on average. Our work highlights the susceptibility of the state-of-the-art LLMs to jailbreak attacks even after safety alignment, and serves as a valuable red-teaming tool for LLM developers.

摘要: 大型语言模型已经显示出作为语言理解和决策工具的巨大前景，它们已经渗透到我们日常生活的各个方面。然而，它们的广泛使用也伴随着新的风险，例如通过一种名为越狱的攻击产生有害的、不道德的或攻击性内容。尽管LLM开发人员做出了广泛的努力，利用人类的反馈来调整LLM，但它们仍然容易受到越狱攻击。为了解决这个问题，研究人员经常使用红色团队来理解和调查越狱提示。然而，现有的红团队方法缺乏有效性、可伸缩性或两者兼而有之。为了解决这些问题，我们提出了JBFuzz，一种新的高效、自动化和可扩展的越狱红色团队技术。JBFuzz的灵感来自于Fuzing在检测软件错误/漏洞方面的成功。我们通过设计新的种子提示、轻量级的突变引擎和轻量级且准确的赋值器来指导模糊器，从而克服了与有效性和可伸缩性相关的三个挑战。吸收所有这三种解决方案的结果是一个强大的模糊，只需要黑匣子访问目标LLM。我们使用九个流行和广泛使用的LLM对JBFuzz进行了广泛的实验评估。我们发现，JBFuzz成功地越狱了所有LLM，涉及各种有害/不道德的问题，平均攻击成功率为99%。我们还发现JBFuzz是非常高效的，因为它平均在60秒内为给定的问题越狱。我们的工作突出了最先进的LLM即使在安全调整之后也容易受到越狱攻击，并且对于LLM开发人员来说是一个有价值的红团队工具。



## **50. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

Accepted to NAACL 2025 Main (oral)

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2410.18469v3) [paper-pdf](http://arxiv.org/pdf/2410.18469v3)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.

摘要: 最近的研究表明，大型语言模型(LLM)容易受到自动越狱攻击，在自动越狱攻击中，由附加到有害查询的算法编制的敌意后缀绕过安全对齐并触发意外响应。目前生成这些后缀的方法计算量大，攻击成功率(ASR)低，尤其是针对Llama2和Llama3等排列良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一个迭代的自我调整过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架大大降低了生成敌意后缀的计算代价，同时在各种开源LLM上实现了近100个ASR。此外，它对闭源模型表现出很强的攻击可转移性，尽管只在Llama3上进行了优化，但在GPT-3.5上达到了99 ASR，在GPT-4上达到了49 ASR。除了提高越狱能力，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全配准研究提供了有价值的见解。



