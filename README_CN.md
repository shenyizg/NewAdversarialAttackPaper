# Latest Adversarial Attack Papers
**update at 2025-05-30 15:22:54**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. SafeScientist: Toward Risk-Aware Scientific Discoveries by LLM Agents**

安全科学家：LLM代理人的风险意识科学发现 cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23559v1) [paper-pdf](http://arxiv.org/pdf/2505.23559v1)

**Authors**: Kunlun Zhu, Jiaxun Zhang, Ziheng Qi, Nuoxing Shang, Zijia Liu, Peixuan Han, Yue Su, Haofei Yu, Jiaxuan You

**Abstract**: Recent advancements in large language model (LLM) agents have significantly accelerated scientific discovery automation, yet concurrently raised critical ethical and safety concerns. To systematically address these challenges, we introduce \textbf{SafeScientist}, an innovative AI scientist framework explicitly designed to enhance safety and ethical responsibility in AI-driven scientific exploration. SafeScientist proactively refuses ethically inappropriate or high-risk tasks and rigorously emphasizes safety throughout the research process. To achieve comprehensive safety oversight, we integrate multiple defensive mechanisms, including prompt monitoring, agent-collaboration monitoring, tool-use monitoring, and an ethical reviewer component. Complementing SafeScientist, we propose \textbf{SciSafetyBench}, a novel benchmark specifically designed to evaluate AI safety in scientific contexts, comprising 240 high-risk scientific tasks across 6 domains, alongside 30 specially designed scientific tools and 120 tool-related risk tasks. Extensive experiments demonstrate that SafeScientist significantly improves safety performance by 35\% compared to traditional AI scientist frameworks, without compromising scientific output quality. Additionally, we rigorously validate the robustness of our safety pipeline against diverse adversarial attack methods, further confirming the effectiveness of our integrated approach. The code and data will be available at https://github.com/ulab-uiuc/SafeScientist. \textcolor{red}{Warning: this paper contains example data that may be offensive or harmful.}

摘要: 大型语言模型（LLM）代理的最新进展显着加速了科学发现自动化，但同时提出了关键的伦理和安全问题。为了系统地应对这些挑战，我们引入了一个创新的人工智能科学家框架，旨在增强人工智能驱动的科学探索中的安全和道德责任。SafeScientist主动拒绝道德上不合适或高风险的任务，并在整个研究过程中严格强调安全。为了实现全面的安全监督，我们整合了多种防御机制，包括即时监控、代理协作监控、工具使用监控和道德审查员组件。作为SafeScientist的补充，我们提出了\textBF{SciSafetyBench}，这是一个专门用于评估科学背景下人工智能安全性的新型基准，包括6个领域的240项高风险科学任务，以及30个专门设计的科学工具和120个工具相关的风险任务。大量实验表明，与传统的人工智能科学家框架相比，SafeScientist将安全性能显着提高了35%，而不会影响科学输出质量。此外，我们还严格验证了我们的安全管道针对各种对抗攻击方法的稳健性，进一步证实了我们集成方法的有效性。代码和数据可在https://github.com/ulab-uiuc/SafeScientist上获取。\textcolor{red}{警告：本文包含可能令人反感或有害的示例数据。}



## **2. SGD Jittering: A Training Strategy for Robust and Accurate Model-Based Architectures**

新元抖动：稳健且准确的基于模型的架构的训练策略 cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2410.14667v2) [paper-pdf](http://arxiv.org/pdf/2410.14667v2)

**Authors**: Peimeng Guan, Mark A. Davenport

**Abstract**: Inverse problems aim to reconstruct unseen data from corrupted or perturbed measurements. While most work focuses on improving reconstruction quality, generalization accuracy and robustness are equally important, especially for safety-critical applications. Model-based architectures (MBAs), such as loop unrolling methods, are considered more interpretable and achieve better reconstructions. Empirical evidence suggests that MBAs are more robust to perturbations than black-box solvers, but the accuracy-robustness tradeoff in MBAs remains underexplored. In this work, we propose a simple yet effective training scheme for MBAs, called SGD jittering, which injects noise iteration-wise during reconstruction. We theoretically demonstrate that SGD jittering not only generalizes better than the standard mean squared error training but is also more robust to average-case attacks. We validate SGD jittering using denoising toy examples, seismic deconvolution, and single-coil MRI reconstruction. Both SGD jittering and its SPGD extension yield cleaner reconstructions for out-of-distribution data and demonstrates enhanced robustness against adversarial attacks.

摘要: 逆问题的目的是从损坏或扰动的测量中重建不可见的数据。虽然大多数工作都集中在提高重建质量上，但泛化精度和鲁棒性同样重要，特别是对于安全关键型应用。基于模型的架构（MBA），如循环展开方法，被认为是更可解释的，并实现更好的重建。经验证据表明，工商管理硕士更强大的扰动比黑盒求解器，但在工商管理硕士的准确性和鲁棒性的权衡仍然未充分探讨。在这项工作中，我们提出了一种简单而有效的MBA训练方案，称为BCD抖动，它在重建期间以迭代方式注入噪音。我们从理论上证明，BCD抖动不仅比标准均方误差训练更好地概括，而且对平均情况攻击也更稳健。我们使用去噪玩具示例、地震去卷积和单线圈MRI重建来验证SGD抖动。Singapore抖动及其SPVD扩展都可以为分发外数据提供更清晰的重建，并表现出针对对抗性攻击的增强的鲁棒性。



## **3. TRAP: Targeted Redirecting of Agentic Preferences**

TRAP：有针对性地重新定向统计偏好 cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23518v1) [paper-pdf](http://arxiv.org/pdf/2505.23518v1)

**Authors**: Hangoo Kang, Jehyeok Yeon, Gagandeep Singh

**Abstract**: Autonomous agentic AI systems powered by vision-language models (VLMs) are rapidly advancing toward real-world deployment, yet their cross-modal reasoning capabilities introduce new attack surfaces for adversarial manipulation that exploit semantic reasoning across modalities. Existing adversarial attacks typically rely on visible pixel perturbations or require privileged model or environment access, making them impractical for stealthy, real-world exploitation. We introduce TRAP, a generative adversarial framework that manipulates the agent's decision-making using diffusion-based semantic injections. Our method combines negative prompt-based degradation with positive semantic optimization, guided by a Siamese semantic network and layout-aware spatial masking. Without requiring access to model internals, TRAP produces visually natural images yet induces consistent selection biases in agentic AI systems. We evaluate TRAP on the Microsoft Common Objects in Context (COCO) dataset, building multi-candidate decision scenarios. Across these scenarios, TRAP achieves a 100% attack success rate on leading models, including LLaVA-34B, Gemma3, and Mistral-3.1, significantly outperforming baselines such as SPSA, Bandit, and standard diffusion approaches. These results expose a critical vulnerability: Autonomous agents can be consistently misled through human-imperceptible cross-modal manipulations. These findings highlight the need for defense strategies beyond pixel-level robustness to address semantic vulnerabilities in cross-modal decision-making.

摘要: 由视觉语言模型（VLM）驱动的自主代理人工智能系统正在迅速向现实世界的部署发展，但它们的跨模态推理能力为利用跨模态语义推理的对抗性操纵引入了新的攻击面。现有的对抗性攻击通常依赖于可见像素扰动，或者需要特权模型或环境访问，这使得它们对于隐形的现实世界利用来说是不切实际的。我们介绍了TRAP，一个生成对抗框架，使用基于扩散的语义注入来操纵代理的决策。我们的方法结合了消极的基于语义的退化与积极的语义优化，指导下的暹罗语义网络和布局感知空间掩蔽。在不需要访问模型内部的情况下，TRAP会产生视觉上自然的图像，但在代理人工智能系统中会引起一致的选择偏差。我们在Microsoft上下文中的公共对象（COCO）数据集中评估TRAP，构建多候选决策场景。在这些场景中，TRAP在LLaVA-34 B、Gemma 3和Mistral-3.1等领先型号上实现了100%的攻击成功率，显着优于SPSA、Bandit和标准扩散方法等基线。这些结果暴露了一个关键的漏洞：自治代理可能会通过人类无法感知的跨模式操纵而持续误导。这些发现凸显了像素级稳健性之外的防御策略的必要性，以解决跨模式决策中的语义漏洞。



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



## **6. DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing**

德尔曼：通过模型编辑对大型语言模型越狱的动态防御 cs.CR

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2502.11647v2) [paper-pdf](http://arxiv.org/pdf/2502.11647v2)

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection.

摘要: 大型语言模型（LLM）广泛应用于决策制定，但其部署受到越狱攻击的威胁，即敌对用户操纵模型行为以绕过安全措施。现有的防御机制，例如安全微调和模型编辑，要么需要大量的参数修改，要么缺乏精确性，导致一般任务的性能下降，不适合部署后的安全对齐。为了应对这些挑战，我们提出了DELMAN（LLC动态编辑JAilbreak DefeNse），这是一种利用直接模型编辑来精确、动态地保护免受越狱攻击的新颖方法。德尔曼直接更新最少的相关参数集，以中和有害行为，同时保留模型的实用性。为了避免在良性上下文中触发安全响应，我们引入了KL分歧正规化，以确保更新后的模型在处理良性查询时与原始模型保持一致。实验结果表明，DELMAN在缓解越狱攻击的同时保持模型的实用性方面优于基线方法，并无缝适应新的攻击实例，为部署后模型保护提供了实用高效的解决方案。



## **7. Robustness-Congruent Adversarial Training for Secure Machine Learning Model Updates**

用于安全机器学习模型更新的稳健一致对抗训练 cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2402.17390v2) [paper-pdf](http://arxiv.org/pdf/2402.17390v2)

**Authors**: Daniele Angioni, Luca Demetrio, Maura Pintor, Luca Oneto, Davide Anguita, Battista Biggio, Fabio Roli

**Abstract**: Machine-learning models demand periodic updates to improve their average accuracy, exploiting novel architectures and additional data. However, a newly updated model may commit mistakes the previous model did not make. Such misclassifications are referred to as negative flips, experienced by users as a regression of performance. In this work, we show that this problem also affects robustness to adversarial examples, hindering the development of secure model update practices. In particular, when updating a model to improve its adversarial robustness, previously ineffective adversarial attacks on some inputs may become successful, causing a regression in the perceived security of the system. We propose a novel technique, named robustness-congruent adversarial training, to address this issue. It amounts to fine-tuning a model with adversarial training, while constraining it to retain higher robustness on the samples for which no adversarial example was found before the update. We show that our algorithm and, more generally, learning with non-regression constraints, provides a theoretically-grounded framework to train consistent estimators. Our experiments on robust models for computer vision confirm that both accuracy and robustness, even if improved after model update, can be affected by negative flips, and our robustness-congruent adversarial training can mitigate the problem, outperforming competing baseline methods.

摘要: 机器学习模型需要定期更新以提高其平均准确性，利用新颖的架构和额外的数据。然而，新更新的模型可能会犯以前模型没有犯的错误。这种错误分类被称为负面翻转，用户将其视为性能的倒退。在这项工作中，我们表明这个问题还会影响对抗性示例的稳健性，从而阻碍安全模型更新实践的开发。特别是，当更新模型以提高其对抗稳健性时，以前对某些输入无效的对抗攻击可能会成功，从而导致系统感知的安全性出现倒退。我们提出了一种新的技术，名为稳健性一致对抗训练，来解决这个问题。它相当于通过对抗训练来微调模型，同时限制它在更新之前没有发现对抗示例的样本上保持更高的鲁棒性。我们表明，我们的算法，以及更一般地说，具有非回归约束的学习，提供了一个基于理论的框架来训练一致的估计器。我们在计算机视觉鲁棒模型上的实验证实，即使在模型更新后得到改善，准确性和鲁棒性也会受到负面翻转的影响，而我们的鲁棒一致对抗训练可以缓解这个问题，优于竞争基线方法。



## **8. Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models**

基于大型语言模型语义理解能力的自适应越狱策略 cs.CL

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23404v1) [paper-pdf](http://arxiv.org/pdf/2505.23404v1)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin

**Abstract**: Adversarial attacks on Large Language Models (LLMs) via jailbreaking techniques-methods that circumvent their built-in safety and ethical constraints-have emerged as a critical challenge in AI security. These attacks compromise the reliability of LLMs by exploiting inherent weaknesses in their comprehension capabilities. This paper investigates the efficacy of jailbreaking strategies that are specifically adapted to the diverse levels of understanding exhibited by different LLMs. We propose the Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models, a novel framework that classifies LLMs into Type I and Type II categories according to their semantic comprehension abilities. For each category, we design tailored jailbreaking strategies aimed at leveraging their vulnerabilities to facilitate successful attacks. Extensive experiments conducted on multiple LLMs demonstrate that our adaptive strategy markedly improves the success rate of jailbreaking. Notably, our approach achieves an exceptional 98.9% success rate in jailbreaking GPT-4o(29 May 2025 release)

摘要: 通过越狱技术（规避其内置安全和道德约束的方法）对大型语言模型（LLM）进行的对抗攻击已成为人工智能安全领域的一个关键挑战。这些攻击通过利用LLM理解能力的固有弱点来损害LLM的可靠性。本文研究了专门适应不同法学硕士所表现出的不同理解水平的越狱策略的有效性。我们提出了基于大型语言模型语义理解能力的自适应越狱策略，这是一个新颖的框架，根据它们的语义理解能力将LLM分为类型I和类型II类别。对于每个类别，我们设计了量身定制的越狱策略，旨在利用其漏洞来促进成功的攻击。在多个LLM上进行的广泛实验表明，我们的自适应策略显着提高了越狱的成功率。值得注意的是，我们的方法在越狱GPT-4 o（2025年5月29日发布）中实现了98.9%的卓越成功率



## **9. Adversarial Semantic and Label Perturbation Attack for Pedestrian Attribute Recognition**

行人属性识别的对抗性语义和标签扰动攻击 cs.CV

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23313v1) [paper-pdf](http://arxiv.org/pdf/2505.23313v1)

**Authors**: Weizhe Kong, Xiao Wang, Ruichong Gao, Chenglong Li, Yu Zhang, Xing Yang, Yaowei Wang, Jin Tang

**Abstract**: Pedestrian Attribute Recognition (PAR) is an indispensable task in human-centered research and has made great progress in recent years with the development of deep neural networks. However, the potential vulnerability and anti-interference ability have still not been fully explored. To bridge this gap, this paper proposes the first adversarial attack and defense framework for pedestrian attribute recognition. Specifically, we exploit both global- and patch-level attacks on the pedestrian images, based on the pre-trained CLIP-based PAR framework. It first divides the input pedestrian image into non-overlapping patches and embeds them into feature embeddings using a projection layer. Meanwhile, the attribute set is expanded into sentences using prompts and embedded into attribute features using a pre-trained CLIP text encoder. A multi-modal Transformer is adopted to fuse the obtained vision and text tokens, and a feed-forward network is utilized for attribute recognition. Based on the aforementioned PAR framework, we adopt the adversarial semantic and label-perturbation to generate the adversarial noise, termed ASL-PAR. We also design a semantic offset defense strategy to suppress the influence of adversarial attacks. Extensive experiments conducted on both digital domains (i.e., PETA, PA100K, MSP60K, RAPv2) and physical domains fully validated the effectiveness of our proposed adversarial attack and defense strategies for the pedestrian attribute recognition. The source code of this paper will be released on https://github.com/Event-AHU/OpenPAR.

摘要: 行人属性识别（BAR）是以人为本的研究中不可或缺的任务，近年来随着深度神经网络的发展取得了长足的进展。但潜在的漏洞和抗干扰能力仍未得到充分挖掘。为了弥合这一差距，本文提出了第一个用于行人属性识别的对抗性攻击和防御框架。具体来说，我们基于预先训练的基于CLIP的VAR框架，利用对行人图像的全局和补丁级攻击。它首先将输入的行人图像划分为不重叠的补丁，并使用投影层将它们嵌入到特征嵌入中。与此同时，属性集使用提示扩展为句子，并使用预先训练的CLIP文本编码器嵌入到属性特征中。采用多模式Transformer将获得的视觉和文本标记融合，并利用前向网络进行属性识别。基于前面提到的VAR框架，我们采用对抗性语义和标签扰动来生成对抗性噪音，称为ASL-VAR。我们还设计了一个语义偏移防御策略来抑制对抗性攻击的影响。在两个数字域（即，PETA、PA 100 K、MSP 60 K、RAPv 2）和物理域的仿真实验，充分验证了本文提出的行人属性识别对抗性攻击和防御策略的有效性。本文的源代码将在https://github.com/Event-AHU/OpenPAR上发布。



## **10. Disrupting Vision-Language Model-Driven Navigation Services via Adversarial Object Fusion**

通过对抗对象融合扰乱视觉语言模型驱动的导航服务 cs.CR

Under review

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23266v1) [paper-pdf](http://arxiv.org/pdf/2505.23266v1)

**Authors**: Chunlong Xie, Jialing He, Shangwei Guo, Jiacheng Wang, Shudong Zhang, Tianwei Zhang, Tao Xiang

**Abstract**: We present Adversarial Object Fusion (AdvOF), a novel attack framework targeting vision-and-language navigation (VLN) agents in service-oriented environments by generating adversarial 3D objects. While foundational models like Large Language Models (LLMs) and Vision Language Models (VLMs) have enhanced service-oriented navigation systems through improved perception and decision-making, their integration introduces vulnerabilities in mission-critical service workflows. Existing adversarial attacks fail to address service computing contexts, where reliability and quality-of-service (QoS) are paramount. We utilize AdvOF to investigate and explore the impact of adversarial environments on the VLM-based perception module of VLN agents. In particular, AdvOF first precisely aggregates and aligns the victim object positions in both 2D and 3D space, defining and rendering adversarial objects. Then, we collaboratively optimize the adversarial object with regularization between the adversarial and victim object across physical properties and VLM perceptions. Through assigning importance weights to varying views, the optimization is processed stably and multi-viewedly by iterative fusions from local updates and justifications. Our extensive evaluations demonstrate AdvOF can effectively degrade agent performance under adversarial conditions while maintaining minimal interference with normal navigation tasks. This work advances the understanding of service security in VLM-powered navigation systems, providing computational foundations for robust service composition in physical-world deployments.

摘要: 我们提出了对抗性对象融合（AdvOF），这是一种新型攻击框架，通过生成对抗性3D对象，针对面向服务环境中的视觉和语言导航（VLN）代理。虽然大型语言模型（LLM）和视觉语言模型（VLM）等基础模型通过改进感知和决策增强了面向服务的导航系统，但它们的集成在关键任务服务工作流程中引入了漏洞。现有的对抗性攻击无法解决服务计算上下文，而服务计算上下文的可靠性和服务质量（Qos）至关重要。我们利用AdvOF来调查和探索对抗环境对VLN代理基于LM的感知模块的影响。特别是，AdvOF首先在2D和3D空间中精确地聚合和对齐受害对象的位置，定义和渲染对抗对象。然后，我们在物理属性和VLM感知之间通过对抗对象和受害者对象之间的正则化协作优化对抗对象。通过对不同视图赋予重要性权值，通过局部更新和调整的迭代融合，实现了稳定的多视图优化。我们的广泛评估表明，AdvOF可以有效地降低代理性能在对抗条件下，同时保持最小的干扰与正常的导航任务。这项工作促进了对基于VLM的导航系统中服务安全性的理解，为物理世界部署中的稳健服务组合提供了计算基础。



## **11. Are You Using Reliable Graph Prompts? Trojan Prompt Attacks on Graph Neural Networks**

您是否使用可靠的图表预算？图神经网络的特洛伊提示攻击 cs.LG

To be appeared in KDD 2025

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2410.13974v2) [paper-pdf](http://arxiv.org/pdf/2410.13974v2)

**Authors**: Minhua Lin, Zhiwei Zhang, Enyan Dai, Zongyu Wu, Yilong Wang, Xiang Zhang, Suhang Wang

**Abstract**: Graph Prompt Learning (GPL) has been introduced as a promising approach that uses prompts to adapt pre-trained GNN models to specific downstream tasks without requiring fine-tuning of the entire model. Despite the advantages of GPL, little attention has been given to its vulnerability to backdoor attacks, where an adversary can manipulate the model's behavior by embedding hidden triggers. Existing graph backdoor attacks rely on modifying model parameters during training, but this approach is impractical in GPL as GNN encoder parameters are frozen after pre-training. Moreover, downstream users may fine-tune their own task models on clean datasets, further complicating the attack. In this paper, we propose TGPA, a backdoor attack framework designed specifically for GPL. TGPA injects backdoors into graph prompts without modifying pre-trained GNN encoders and ensures high attack success rates and clean accuracy. To address the challenge of model fine-tuning by users, we introduce a finetuning-resistant poisoning approach that maintains the effectiveness of the backdoor even after downstream model adjustments. Extensive experiments on multiple datasets under various settings demonstrate the effectiveness of TGPA in compromising GPL models with fixed GNN encoders.

摘要: 图形提示学习（GPT）是一种有前途的方法，它使用提示将预训练的GNN模型适应特定的下游任务，而无需微调整个模型。尽管GPT有优势，但很少有人关注它对后门攻击的脆弱性，对手可以通过嵌入隐藏触发器来操纵模型的行为。现有的图后门攻击依赖于在训练期间修改模型参数，但这种方法在GPT中不切实际，因为GNN编码器参数在预训练后被冻结。此外，下游用户可能会在干净的数据集上微调自己的任务模型，从而使攻击变得更加复杂。在本文中，我们提出了TGMA，这是一个专门为GPT设计的后门攻击框架。TGMA在图形提示中注入后门，无需修改预训练的GNN编码器，并确保高攻击成功率和清晰的准确性。为了应对用户模型微调的挑战，我们引入了一种抗微调中毒方法，即使在下游模型调整之后也能保持后门的有效性。在各种设置下对多个数据集进行的广泛实验证明了TGMA在使用固定GNN编码器妥协GPT模型方面的有效性。



## **12. The Meeseeks Mesh: Spatially Consistent 3D Adversarial Objects for BEV Detector**

Meeseeks网格：BEV探测器的空间一致3D对抗对象 cs.CV

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.22499v2) [paper-pdf](http://arxiv.org/pdf/2505.22499v2)

**Authors**: Aixuan Li, Mochu Xiang, Jing Zhang, Yuchao Dai

**Abstract**: 3D object detection is a critical component in autonomous driving systems. It allows real-time recognition and detection of vehicles, pedestrians and obstacles under varying environmental conditions. Among existing methods, 3D object detection in the Bird's Eye View (BEV) has emerged as the mainstream framework. To guarantee a safe, robust and trustworthy 3D object detection, 3D adversarial attacks are investigated, where attacks are placed in 3D environments to evaluate the model performance, e.g. putting a film on a car, clothing a pedestrian. The vulnerability of 3D object detection models to 3D adversarial attacks serves as an important indicator to evaluate the robustness of the model against perturbations. To investigate this vulnerability, we generate non-invasive 3D adversarial objects tailored for real-world attack scenarios. Our method verifies the existence of universal adversarial objects that are spatially consistent across time and camera views. Specifically, we employ differentiable rendering techniques to accurately model the spatial relationship between adversarial objects and the target vehicle. Furthermore, we introduce an occlusion-aware module to enhance visual consistency and realism under different viewpoints. To maintain attack effectiveness across multiple frames, we design a BEV spatial feature-guided optimization strategy. Experimental results demonstrate that our approach can reliably suppress vehicle predictions from state-of-the-art 3D object detectors, serving as an important tool to test robustness of 3D object detection models before deployment. Moreover, the generated adversarial objects exhibit strong generalization capabilities, retaining its effectiveness at various positions and distances in the scene.

摘要: 3D物体检测是自动驾驶系统中的关键组成部分。它允许在不同环境条件下实时识别和检测车辆、行人和障碍物。在现有方法中，鸟瞰图（BEV）中的3D对象检测已成为主流框架。为了保证安全、稳健和值得信赖的3D对象检测，我们对3D对抗性攻击进行了研究，其中攻击被放置在3D环境中以评估模型性能，例如在汽车上贴上薄膜、给行人穿衣服。3D对象检测模型对3D对抗攻击的脆弱性是评估模型对扰动稳健性的重要指标。为了调查此漏洞，我们生成针对现实世界攻击场景量身定制的非侵入性3D对抗对象。我们的方法验证了存在跨时间和相机视图空间一致的普遍对抗对象。具体来说，我们采用可区分渲染技术来准确地建模对抗对象和目标车辆之间的空间关系。此外，我们还引入了遮挡感知模块，以增强不同视角下的视觉一致性和真实感。为了保持跨多帧的攻击有效性，我们设计了BEV空间特征引导的优化策略。实验结果表明，我们的方法可以可靠地抑制来自最先进3D对象检测器的车辆预测，作为在部署前测试3D对象检测模型稳健性的重要工具。此外，生成的对抗对象具有强大的概括能力，在场景中的不同位置和距离保持其有效性。



## **13. Fooling the Watchers: Breaking AIGC Detectors via Semantic Prompt Attacks**

愚弄观察者：通过语义提示攻击破解AIGC检测器 cs.CV

9 pages

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23192v1) [paper-pdf](http://arxiv.org/pdf/2505.23192v1)

**Authors**: Run Hao, Peng Ying

**Abstract**: The rise of text-to-image (T2I) models has enabled the synthesis of photorealistic human portraits, raising serious concerns about identity misuse and the robustness of AIGC detectors. In this work, we propose an automated adversarial prompt generation framework that leverages a grammar tree structure and a variant of the Monte Carlo tree search algorithm to systematically explore the semantic prompt space. Our method generates diverse, controllable prompts that consistently evade both open-source and commercial AIGC detectors. Extensive experiments across multiple T2I models validate its effectiveness, and the approach ranked first in a real-world adversarial AIGC detection competition. Beyond attack scenarios, our method can also be used to construct high-quality adversarial datasets, providing valuable resources for training and evaluating more robust AIGC detection and defense systems.

摘要: 文本到图像（T2I）模型的兴起使得合成逼真的人像成为可能，这引起了人们对身份滥用和AIGC检测器鲁棒性的严重关注。在这项工作中，我们提出了一个自动化的对抗提示生成框架，利用语法树结构和蒙特卡洛树搜索算法的变体，系统地探索语义提示空间。我们的方法产生了多样的，可控的提示，始终逃避开源和商业AIGC检测器。在多个T2I模型上进行的大量实验验证了其有效性，该方法在现实世界的对抗性AIGC检测竞赛中排名第一。除了攻击场景之外，我们的方法还可用于构建高质量的对抗数据集，为训练和评估更强大的AIGC检测和防御系统提供宝贵的资源。



## **14. Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context**

人类可读的对抗性预言：使用情境背景对LLM漏洞的调查 cs.CL

arXiv admin note: text overlap with arXiv:2407.14644

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2412.16359v3) [paper-pdf](http://arxiv.org/pdf/2412.16359v3)

**Authors**: Nilanjana Das, Edward Raff, Aman Chadha, Manas Gaur

**Abstract**: As the AI systems become deeply embedded in social media platforms, we've uncovered a concerning security vulnerability that goes beyond traditional adversarial attacks. It becomes important to assess the risks of LLMs before the general public use them on social media platforms to avoid any adverse impacts. Unlike obvious nonsensical text strings that safety systems can easily catch, our work reveals that human-readable situation-driven adversarial full-prompts that leverage situational context are effective but much harder to detect. We found that skilled attackers can exploit the vulnerabilities in open-source and proprietary LLMs to make a malicious user query safe for LLMs, resulting in generating a harmful response. This raises an important question about the vulnerabilities of LLMs. To measure the robustness against human-readable attacks, which now present a potent threat, our research makes three major contributions. First, we developed attacks that use movie scripts as situational contextual frameworks, creating natural-looking full-prompts that trick LLMs into generating harmful content. Second, we developed a method to transform gibberish adversarial text into readable, innocuous content that still exploits vulnerabilities when used within the full-prompts. Finally, we enhanced the AdvPrompter framework with p-nucleus sampling to generate diverse human-readable adversarial texts that significantly improve attack effectiveness against models like GPT-3.5-Turbo-0125 and Gemma-7b. Our findings show that these systems can be manipulated to operate beyond their intended ethical boundaries when presented with seemingly normal prompts that contain hidden adversarial elements. By identifying these vulnerabilities, we aim to drive the development of more robust safety mechanisms that can withstand sophisticated attacks in real-world applications.

摘要: 随着人工智能系统深入嵌入社交媒体平台，我们发现了一个超越传统对抗攻击的令人担忧的安全漏洞。在公众在社交媒体平台上使用LLM之前，评估LLM的风险变得重要，以避免任何不利影响。与安全系统可以轻松捕获的明显无意义的文本字符串不同，我们的工作表明，利用情景上下文的人类可读的情景驱动对抗性完整提示是有效的，但更难检测。我们发现，熟练的攻击者可以利用开源和专有LLM中的漏洞，使恶意用户查询对LLM安全，从而生成有害响应。这就提出了一个关于LLMs脆弱性的重要问题。为了衡量对人类可读攻击的鲁棒性，我们的研究做出了三个主要贡献。首先，我们开发了使用电影脚本作为情景上下文框架的攻击，创建看起来自然的完整提示，诱使LLM生成有害内容。第二，我们开发了一种方法，将胡言乱语的对抗性文本转换为可读的、无害的内容，当在完整提示中使用时，这些内容仍然会利用漏洞。最后，我们通过p核采样增强了Advancer框架，以生成各种人类可读的对抗文本，从而显着提高了针对GPT-3.5-Turbo-0125和Gemma-7 b等模型的攻击有效性。我们的研究结果表明，当出现包含隐藏对抗元素的看似正常的提示时，这些系统可能会被操纵以超出其预期的道德界限来运行。通过识别这些漏洞，我们的目标是推动开发更强大的安全机制，以抵御现实世界应用程序中的复杂攻击。



## **15. One Prompt to Verify Your Models: Black-Box Text-to-Image Models Verification via Non-Transferable Adversarial Attacks**

验证模型的一个提示：通过不可传输对抗性攻击进行黑匣子文本到图像模型验证 cs.CV

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2410.22725v4) [paper-pdf](http://arxiv.org/pdf/2410.22725v4)

**Authors**: Ji Guo, Wenbo Jiang, Rui Zhang, Guoming Lu, Hongwei Li

**Abstract**: Recently, various types of Text-to-Image (T2I) models have emerged (such as DALL-E and Stable Diffusion), and showing their advantages in different aspects. Therefore, some third-party service platforms collect different model interfaces and provide cheaper API services and more flexibility in T2I model selections. However, this also raises a new security concern: Are these third-party services truly offering the models they claim?   To answer this question, we first define the concept of T2I model verification, which aims to determine whether a black-box target model is identical to a given white-box reference T2I model. After that, we propose VerifyPrompt, which performs T2I model verification through a special designed verify prompt. Intuitionally, the verify prompt is an adversarial prompt for the target model without transferability for other models. It makes the target model generate a specific image while making other models produce entirely different images. Specifically, VerifyPrompt utilizes the Non-dominated Sorting Genetic Algorithm II (NSGA-II) to optimize the cosine similarity of a prompt's text encoding, generating verify prompts. Finally, by computing the CLIP-text similarity scores between the prompts the generated images, VerifyPrompt can determine whether the target model aligns with the reference model. Experimental results demonstrate that VerifyPrompt consistently achieves over 90\% accuracy across various T2I models, confirming its effectiveness in practical model platforms (such as Hugging Face).

摘要: 近年来，各种类型的文本到图像（T2 I）模型出现（例如DALL-E和稳定扩散），并在不同方面展示了它们的优势。因此，一些第三方服务平台收集不同的型号接口，提供更便宜的API服务和更灵活的T2 I型号选择。然而，这也引发了一个新的安全问题：这些第三方服务真的提供了他们声称的型号吗？   为了回答这个问题，我们首先定义了T2 I模型验证的概念，旨在确定黑匣子目标模型是否与给定的白盒参考T2 I模型相同。之后，我们提出了VerifyPrompt，通过特殊设计的验证提示来执行T2 I模型验证。直觉上，验证提示是目标模型的对抗提示，对于其他模型来说没有可移植性。它使目标模型生成特定的图像，同时使其他模型生成完全不同的图像。具体来说，VerifyPromise利用非支配排序遗传算法II（NSGA-II）来优化提示文本编码的cos相似度，从而生成验证提示。最后，通过计算生成的图像之间的CLIP文本相似度分数，VerifyPrompt可以确定目标模型是否与参考模型一致。实验结果表明，VerifyPrompt在各种T2 I模型中始终达到90%以上的准确率，证实了其在实际模型平台（例如Hugging Face）中的有效性。



## **16. Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates**

LLM可以欺骗CLIP吗？通过文本更新对预训练多模式表示的对抗性组合进行基准测试 cs.CL

ACL 2025 Main. Code is released at  https://vision.snu.ac.kr/projects/mac

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22943v1) [paper-pdf](http://arxiv.org/pdf/2505.22943v1)

**Authors**: Jaewoo Ahn, Heeseung Yun, Dayoon Ko, Gunhee Kim

**Abstract**: While pre-trained multimodal representations (e.g., CLIP) have shown impressive capabilities, they exhibit significant compositional vulnerabilities leading to counterintuitive judgments. We introduce Multimodal Adversarial Compositionality (MAC), a benchmark that leverages large language models (LLMs) to generate deceptive text samples to exploit these vulnerabilities across different modalities and evaluates them through both sample-wise attack success rate and group-wise entropy-based diversity. To improve zero-shot methods, we propose a self-training approach that leverages rejection-sampling fine-tuning with diversity-promoting filtering, which enhances both attack success rate and sample diversity. Using smaller language models like Llama-3.1-8B, our approach demonstrates superior performance in revealing compositional vulnerabilities across various multimodal representations, including images, videos, and audios.

摘要: 虽然预训练的多模式表示（例如，CLIP）表现出令人印象深刻的能力，它们表现出显着的合成漏洞，导致反直觉的判断。我们引入了多模式对抗组合（MAC），这是一个基准，利用大型语言模型（LLM）来生成欺骗性文本样本，以利用不同模式中的这些漏洞，并通过样本攻击成功率和基于分组的基于信息的多样性来评估它们。为了改进零射击方法，我们提出了一种自训练方法，该方法利用拒绝采样微调和促进多样性的过滤，从而增强了攻击成功率和样本多样性。使用Llama-3.1-8B等较小的语言模型，我们的方法在揭示各种多模式表示（包括图像、视频和音频）的合成漏洞方面表现出卓越的性能。



## **17. Efficient Preimage Approximation for Neural Network Certification**

神经网络认证的高效前像逼近 cs.LG

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22798v1) [paper-pdf](http://arxiv.org/pdf/2505.22798v1)

**Authors**: Anton Björklund, Mykola Zaitsev, Marta Kwiatkowska

**Abstract**: The growing reliance on artificial intelligence in safety- and security-critical applications demands effective neural network certification. A challenging real-world use case is certification against ``patch attacks'', where adversarial patches or lighting conditions obscure parts of images, for example traffic signs. One approach to certification, which also gives quantitative coverage estimates, utilizes preimages of neural networks, i.e., the set of inputs that lead to a specified output. However, these preimage approximation methods, including the state-of-the-art PREMAP algorithm, struggle with scalability. This paper presents novel algorithmic improvements to PREMAP involving tighter bounds, adaptive Monte Carlo sampling, and improved branching heuristics. We demonstrate efficiency improvements of at least an order of magnitude on reinforcement learning control benchmarks, and show that our method scales to convolutional neural networks that were previously infeasible. Our results demonstrate the potential of preimage approximation methodology for reliability and robustness certification.

摘要: 安全和安全关键应用中对人工智能的依赖日益增加，需要有效的神经网络认证。一个具有挑战性的现实世界用例是针对“补丁攻击”的认证，其中对抗性补丁或照明条件会模糊图像的部分，例如交通标志。一种认证方法还提供量化覆盖率估计，利用神经网络的预像，即导致指定输出的输入集。然而，这些前像逼近方法（包括最先进的PREMAP算法）在可扩展性方面遇到了困难。本文提出了对PREMAP的新颖算法改进，涉及更严格的边界、自适应蒙特卡罗采样和改进的分支启发法。我们在强化学习控制基准上证明了至少一个数量级的效率改进，并表明我们的方法可以扩展到以前不可行的卷积神经网络。我们的结果证明了前像逼近方法在可靠性和稳健性认证方面的潜力。



## **18. Adversarially Robust AI-Generated Image Detection for Free: An Information Theoretic Perspective**

对抗稳健的人工智能生成图像检测免费：信息论的角度 cs.CV

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22604v1) [paper-pdf](http://arxiv.org/pdf/2505.22604v1)

**Authors**: Ruixuan Zhang, He Wang, Zhengyu Zhao, Zhiqing Guo, Xun Yang, Yunfeng Diao, Meng Wang

**Abstract**: Rapid advances in Artificial Intelligence Generated Images (AIGI) have facilitated malicious use, such as forgery and misinformation. Therefore, numerous methods have been proposed to detect fake images. Although such detectors have been proven to be universally vulnerable to adversarial attacks, defenses in this field are scarce. In this paper, we first identify that adversarial training (AT), widely regarded as the most effective defense, suffers from performance collapse in AIGI detection. Through an information-theoretic lens, we further attribute the cause of collapse to feature entanglement, which disrupts the preservation of feature-label mutual information. Instead, standard detectors show clear feature separation. Motivated by this difference, we propose Training-free Robust Detection via Information-theoretic Measures (TRIM), the first training-free adversarial defense for AIGI detection. TRIM builds on standard detectors and quantifies feature shifts using prediction entropy and KL divergence. Extensive experiments across multiple datasets and attacks validate the superiority of our TRIM, e.g., outperforming the state-of-the-art defense by 33.88% (28.91%) on ProGAN (GenImage), while well maintaining original accuracy.

摘要: 人工智能生成图像（AIGI）的迅速发展助长了恶意使用，例如伪造和错误信息。因此，人们提出了多种方法来检测虚假图像。尽管此类探测器已被证明普遍容易受到对抗攻击，但该领域的防御措施却很少。在本文中，我们首先确定了对抗训练（AT），被广泛认为是最有效的防御，在AIGI检测中遭受性能崩溃。通过信息论的视角，我们进一步将崩溃的原因归因于特征纠缠，它破坏了特征标签互信息的保存。相反，标准检测器显示出清晰的特征分离。基于这种差异，我们提出了通过信息理论测量（TRIM）的免训练鲁棒检测，这是AIGI检测的第一个免训练对抗性防御。TRIM建立在标准检测器之上，并使用预测熵和KL方差量化特征移动。跨多个数据集和攻击的广泛实验验证了我们TRIM的优越性，例如，ProGAN（GenImage）上的最新防御能力比最先进的防御能力高出33.88%（28.91%），同时很好地保持了原始的准确性。



## **19. AdvAgent: Controllable Blackbox Red-teaming on Web Agents**

AdvAgent：基于Web Agent的可控黑盒红组 cs.CR

ICML 2025

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2410.17401v3) [paper-pdf](http://arxiv.org/pdf/2410.17401v3)

**Authors**: Chejian Xu, Mintong Kang, Jiawei Zhang, Zeyi Liao, Lingbo Mo, Mengqi Yuan, Huan Sun, Bo Li

**Abstract**: Foundation model-based agents are increasingly used to automate complex tasks, enhancing efficiency and productivity. However, their access to sensitive resources and autonomous decision-making also introduce significant security risks, where successful attacks could lead to severe consequences. To systematically uncover these vulnerabilities, we propose AdvAgent, a black-box red-teaming framework for attacking web agents. Unlike existing approaches, AdvAgent employs a reinforcement learning-based pipeline to train an adversarial prompter model that optimizes adversarial prompts using feedback from the black-box agent. With careful attack design, these prompts effectively exploit agent weaknesses while maintaining stealthiness and controllability. Extensive evaluations demonstrate that AdvAgent achieves high success rates against state-of-the-art GPT-4-based web agents across diverse web tasks. Furthermore, we find that existing prompt-based defenses provide only limited protection, leaving agents vulnerable to our framework. These findings highlight critical vulnerabilities in current web agents and emphasize the urgent need for stronger defense mechanisms. We release code at https://ai-secure.github.io/AdvAgent/.

摘要: 基于基础模型的代理越来越多地用于自动化复杂任务，提高效率和生产力。然而，他们对敏感资源的访问和自主决策也带来了重大的安全风险，成功的攻击可能会导致严重的后果。为了系统性地发现这些漏洞，我们提出了AdvAgent，这是一个用于攻击Web代理的黑匣子红团队框架。与现有方法不同，AdvAgent采用基于强化学习的管道来训练对抗性提示器模型，该模型使用黑匣子代理的反馈来优化对抗性提示。通过精心的攻击设计，这些提示可以有效地利用代理的弱点，同时保持隐蔽性和可控性。广泛的评估表明，AdvAgent在各种Web任务中针对最先进的基于GPT-4的Web代理取得了很高的成功率。此外，我们发现现有的基于预算的防御只能提供有限的保护，使代理容易受到我们的框架的影响。这些发现凸显了当前网络代理中的关键漏洞，并强调迫切需要更强大的防御机制。我们在https://ai-secure.github.io/AdvAgent/上发布代码。



## **20. Understanding Adversarial Training with Energy-based Models**

使用基于能量的模型理解对抗性培训 cs.LG

Under review for TPAMI

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22486v1) [paper-pdf](http://arxiv.org/pdf/2505.22486v1)

**Authors**: Mujtaba Hussain Mirza, Maria Rosaria Briglia, Filippo Bartolucci, Senad Beadini, Giuseppe Lisanti, Iacopo Masi

**Abstract**: We aim at using Energy-based Model (EBM) framework to better understand adversarial training (AT) in classifiers, and additionally to analyze the intrinsic generative capabilities of robust classifiers. By viewing standard classifiers through an energy lens, we begin by analyzing how the energies of adversarial examples, generated by various attacks, differ from those of the natural samples. The central focus of our work is to understand the critical phenomena of Catastrophic Overfitting (CO) and Robust Overfitting (RO) in AT from an energy perspective. We analyze the impact of existing AT approaches on the energy of samples during training and observe that the behavior of the ``delta energy' -- change in energy between original sample and its adversarial counterpart -- diverges significantly when CO or RO occurs. After a thorough analysis of these energy dynamics and their relationship with overfitting, we propose a novel regularizer, the Delta Energy Regularizer (DER), designed to smoothen the energy landscape during training. We demonstrate that DER is effective in mitigating both CO and RO across multiple benchmarks. We further show that robust classifiers, when being used as generative models, have limits in handling trade-off between image quality and variability. We propose an improved technique based on a local class-wise principal component analysis (PCA) and energy-based guidance for better class-specific initialization and adaptive stopping, enhancing sample diversity and generation quality. Considering that we do not explicitly train for generative modeling, we achieve a competitive Inception Score (IS) and Fr\'echet inception distance (FID) compared to hybrid discriminative-generative models.

摘要: 我们的目标是使用基于能量的模型（EBM）框架来更好地理解分类器中的对抗训练（AT），并分析稳健分类器的内在生成能力。通过能量镜头观察标准分类器，我们首先分析各种攻击产生的对抗性示例的能量与自然样本的能量有何不同。我们工作的中心重点是从能源角度了解AT中灾难性过度装配（CO）和鲁棒过度装配（RO）的关键现象。我们分析了现有AT方法在训练过程中对样本能量的影响，并观察到“delta能量”的行为-原始样本与其对抗性对应物之间的能量变化-在CO或RO发生时显着偏离。在深入分析了这些能量动态及其与过拟合的关系之后，我们提出了一种新的正则化器，Delta Energy Regularizer（DER），旨在平滑训练过程中的能量景观。我们证明了DER在多个基准上有效地减轻了CO和RO。我们进一步表明，强大的分类器，当被用作生成模型，在处理图像质量和可变性之间的权衡有限制。我们提出了一种基于局部类主成分分析（PCA）和基于能量的指导的改进技术，以实现更好的类特定初始化和自适应停止，增强样本多样性和生成质量。考虑到我们没有明确训练生成式建模，与混合区分生成模型相比，我们实现了有竞争力的初始评分（IS）和Frechet初始距离（DID）。



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



## **23. Accountable, Scalable and DoS-resilient Secure Vehicular Communication**

负责任、可扩展且具有Dos弹性的安全车辆通信 cs.CR

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.22162v1) [paper-pdf](http://arxiv.org/pdf/2505.22162v1)

**Authors**: Hongyu Jin, Panos Papadimitratos

**Abstract**: Paramount to vehicle safety, broadcasted Cooperative Awareness Messages (CAMs) and Decentralized Environmental Notification Messages (DENMs) are pseudonymously authenticated for security and privacy protection, with each node needing to have all incoming messages validated within an expiration deadline. This creates an asymmetry that can be easily exploited by external adversaries to launch a clogging Denial of Service (DoS) attack: each forged VC message forces all neighboring nodes to cryptographically validate it; at increasing rates, easy to generate forged messages gradually exhaust processing resources and severely degrade or deny timely validation of benign CAMs/DENMs. The result can be catastrophic when awareness of neighbor vehicle positions or critical reports are missed. We address this problem making the standardized VC pseudonymous authentication DoS-resilient. We propose efficient cryptographic constructs, which we term message verification facilitators, to prioritize processing resources for verification of potentially valid messages among bogus messages and verify multiple messages based on one signature verification. Any message acceptance is strictly based on public-key based message authentication/verification for accountability, i.e., non-repudiation is not sacrificed, unlike symmetric key based approaches. This further enables drastic misbehavior detection, also exploiting the newly introduced facilitators, based on probabilistic signature verification and cross-checking over multiple facilitators verifying the same message; while maintaining verification latency low even when under attack, trading off modest communication overhead. Our facilitators can also be used for efficient discovery and verification of DENM or any event-driven message, including misbehavior evidence used for our scheme.

摘要: 对于车辆安全来说，广播的合作感知消息（CAM）和分散环境通知消息（DENM）经过匿名身份验证，以实现安全和隐私保护，每个节点都需要在到期截止日期内验证所有输入消息。这造成了一种不对称性，外部对手很容易利用该不对称性发起阻塞性拒绝服务（DPS）攻击：每条伪造的VC消息迫使所有邻近节点对其进行加密验证;随着速度的增加，容易生成的伪造消息会逐渐耗尽处理资源，并严重降级或拒绝对良性CAM/DENI的及时验证。当忽视邻近车辆位置或关键报告时，结果可能是灾难性的。我们解决了这个问题，使标准化的VC假名身份验证具有Dos弹性。我们提出了高效的加密结构，我们将其称为消息验证促进者，以优先处理资源来验证虚假消息中的潜在有效消息，并基于一个签名验证来验证多个消息。任何消息接受都严格基于基于公钥的消息认证/验证，以实现问责制，即与基于对称密钥的方法不同，不可否认性不会被牺牲。这进一步实现了严重的不当行为检测，还利用了新引入的促进者，基于概率签名验证和对验证同一消息的多个促进者的交叉检查;同时即使在受到攻击时也保持较低的验证延迟，从而权衡了适度的通信费用。我们的促进者还可以用于高效发现和验证DELM或任何事件驱动的消息，包括用于我们计划的不当行为证据。



## **24. The Silent Saboteur: Imperceptible Adversarial Attacks against Black-Box Retrieval-Augmented Generation Systems**

沉默的破坏者：针对黑匣子检索增强生成系统的不可感知的对抗攻击 cs.IR

18 pages,accepted by ACL25 findings

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.18583v2) [paper-pdf](http://arxiv.org/pdf/2505.18583v2)

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Jianming Lv, Maarten de Rijke, Xueqi Cheng

**Abstract**: We explore adversarial attacks against retrieval-augmented generation (RAG) systems to identify their vulnerabilities. We focus on generating human-imperceptible adversarial examples and introduce a novel imperceptible retrieve-to-generate attack against RAG. This task aims to find imperceptible perturbations that retrieve a target document, originally excluded from the initial top-$k$ candidate set, in order to influence the final answer generation. To address this task, we propose ReGENT, a reinforcement learning-based framework that tracks interactions between the attacker and the target RAG and continuously refines attack strategies based on relevance-generation-naturalness rewards. Experiments on newly constructed factual and non-factual question-answering benchmarks demonstrate that ReGENT significantly outperforms existing attack methods in misleading RAG systems with small imperceptible text perturbations.

摘要: 我们探索针对检索增强生成（RAG）系统的对抗攻击以识别其漏洞。我们专注于生成人类不可感知的对抗示例，并引入一种针对RAG的新型不可感知的检索生成攻击。该任务的目的是找到不可感知的扰动，这些扰动检索最初被排除在初始顶级$k $候选集中的目标文档，以影响最终的答案生成。为了解决这一任务，我们提出了ReGENT，这是一个基于强化学习的框架，它跟踪攻击者和目标RAG之间的交互，并基于相关性生成自然性奖励不断完善攻击策略。对新构建的事实和非事实问答基准的实验表明，在具有微小难以察觉的文本扰动的误导性RAG系统中，ReGENT显着优于现有的攻击方法。



## **25. Understanding Model Ensemble in Transferable Adversarial Attack**

了解可转移对抗攻击中的模型集合 cs.LG

Accepted by ICML 2025

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2410.06851v3) [paper-pdf](http://arxiv.org/pdf/2410.06851v3)

**Authors**: Wei Yao, Zeliang Zhang, Huayi Tang, Yong Liu

**Abstract**: Model ensemble adversarial attack has become a powerful method for generating transferable adversarial examples that can target even unknown models, but its theoretical foundation remains underexplored. To address this gap, we provide early theoretical insights that serve as a roadmap for advancing model ensemble adversarial attack. We first define transferability error to measure the error in adversarial transferability, alongside concepts of diversity and empirical model ensemble Rademacher complexity. We then decompose the transferability error into vulnerability, diversity, and a constant, which rigidly explains the origin of transferability error in model ensemble attack: the vulnerability of an adversarial example to ensemble components, and the diversity of ensemble components. Furthermore, we apply the latest mathematical tools in information theory to bound the transferability error using complexity and generalization terms, contributing to three practical guidelines for reducing transferability error: (1) incorporating more surrogate models, (2) increasing their diversity, and (3) reducing their complexity in cases of overfitting. Finally, extensive experiments with 54 models validate our theoretical framework, representing a significant step forward in understanding transferable model ensemble adversarial attacks.

摘要: 模型集成对抗攻击已成为生成可转移对抗示例的一种强大方法，这些示例甚至可以针对未知模型，但其理论基础仍然没有充分研究。为了解决这一差距，我们提供了早期的理论见解，作为推进模型集成对抗攻击的路线图。我们首先定义可移植性误差来衡量对抗可移植性的误差，以及多样性和经验模型整体Rademacher复杂性的概念。然后，我们将可移植性错误分解为脆弱性、多样性和一个常数，这严格解释了模型集成攻击中可移植性错误的起源：对抗性示例对集成组件的脆弱性，以及集成组件的多样性。此外，我们应用信息论中的最新数学工具，使用复杂性和概括性术语来限制可移植性误差，为减少可移植性误差提供了三个实用指导方针：（1）合并更多的代理模型，（2）增加其多样性，（3）在过度适应的情况下降低其复杂性。最后，对54个模型的广泛实验验证了我们的理论框架，代表着在理解可转移模型集成对抗攻击方面向前迈出了重要一步。



## **26. Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs**

超越表面级模式：一个面向LLM的防御框架 cs.CR

16 pages, 12 figures, ACL 2025 findings

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2502.19041v2) [paper-pdf](http://arxiv.org/pdf/2502.19041v2)

**Authors**: Shiyu Xiang, Ansen Zhang, Yanfei Cao, Yang Fan, Ronghao Chen

**Abstract**: Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.

摘要: 尽管对齐大型语言模型（LLM）经过训练可以拒绝有害请求，但它们仍然容易受到越狱攻击。不幸的是，现有的方法通常专注于表面模式，而忽视了更深层次的攻击本质。因此，即使潜在的“攻击本质”保持不变，当攻击促使改变时，防御就会失败。为了解决这个问题，我们引入了EDDF，这是一个\textBF{E} sence-\textBF {D}riven \textBF{D}efense \textBF{F} ravoy Against LLM中的越狱攻击。EDDF是一种即插即用的输入过滤方法，分两个阶段运行：1）离线本质数据库构建，2）在线对抗性查询检测。EDDF背后的关键思想是从不同的已知攻击实例中提取“攻击本质”，并将其存储在离线载体数据库中。实验结果表明，EDDF通过将攻击成功率降低至少20%，显着优于现有方法，凸显了其对越狱攻击的卓越鲁棒性。



## **27. Seeing the Threat: Vulnerabilities in Vision-Language Models to Adversarial Attack**

看到威胁：视觉语言模型对对抗性攻击的脆弱性 cs.CL

Preprint

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.21967v1) [paper-pdf](http://arxiv.org/pdf/2505.21967v1)

**Authors**: Juan Ren, Mark Dras, Usman Naseem

**Abstract**: Large Vision-Language Models (LVLMs) have shown remarkable capabilities across a wide range of multimodal tasks. However, their integration of visual inputs introduces expanded attack surfaces, thereby exposing them to novel security vulnerabilities. In this work, we conduct a systematic representational analysis to uncover why conventional adversarial attacks can circumvent the safety mechanisms embedded in LVLMs. We further propose a novel two stage evaluation framework for adversarial attacks on LVLMs. The first stage differentiates among instruction non compliance, outright refusal, and successful adversarial exploitation. The second stage quantifies the degree to which the model's output fulfills the harmful intent of the adversarial prompt, while categorizing refusal behavior into direct refusals, soft refusals, and partial refusals that remain inadvertently helpful. Finally, we introduce a normative schema that defines idealized model behavior when confronted with harmful prompts, offering a principled target for safety alignment in multimodal systems.

摘要: 大型视觉语言模型（LVLM）在广泛的多模式任务中表现出了非凡的能力。然而，它们对视觉输入的集成引入了扩展的攻击面，从而使它们面临新型安全漏洞。在这项工作中，我们进行了系统的代表性分析，以揭示为什么传统的对抗性攻击可以规避LVLM中嵌入的安全机制。我们进一步提出了一种新颖的两阶段评估框架，用于对LVLM的对抗性攻击。第一阶段区分不遵守指示、彻底拒绝和成功的对抗性剥削。第二阶段量化模型的输出满足对抗提示有害意图的程度，同时将拒绝行为分为直接拒绝、软拒绝和无意中仍然有帮助的部分拒绝。最后，我们引入了一个规范模式，该模式定义了面对有害提示时的理想化模型行为，为多模式系统中的安全对齐提供了原则性目标。



## **28. Practical Adversarial Attacks on Stochastic Bandits via Fake Data Injection**

通过虚假数据注入对随机盗贼的实际对抗攻击 cs.LG

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.21938v1) [paper-pdf](http://arxiv.org/pdf/2505.21938v1)

**Authors**: Qirun Zeng, Eric He, Richard Hoffmann, Xuchuang Wang, Jinhang Zuo

**Abstract**: Adversarial attacks on stochastic bandits have traditionally relied on some unrealistic assumptions, such as per-round reward manipulation and unbounded perturbations, limiting their relevance to real-world systems. We propose a more practical threat model, Fake Data Injection, which reflects realistic adversarial constraints: the attacker can inject only a limited number of bounded fake feedback samples into the learner's history, simulating legitimate interactions. We design efficient attack strategies under this model, explicitly addressing both magnitude constraints (on reward values) and temporal constraints (on when and how often data can be injected). Our theoretical analysis shows that these attacks can mislead both Upper Confidence Bound (UCB) and Thompson Sampling algorithms into selecting a target arm in nearly all rounds while incurring only sublinear attack cost. Experiments on synthetic and real-world datasets validate the effectiveness of our strategies, revealing significant vulnerabilities in widely used stochastic bandit algorithms under practical adversarial scenarios.

摘要: 对随机强盗的对抗性攻击传统上依赖于一些不切实际的假设，例如每轮奖励操纵和无界扰动，限制了它们与现实世界系统的相关性。我们提出了一个更实用的威胁模型，假数据注入，它反映了现实的对抗性约束：攻击者只能将有限数量的有界假反馈样本注入到学习者的历史中，模拟合法的交互。我们在这个模型下设计了有效的攻击策略，明确地解决了幅度约束（奖励值）和时间约束（何时以及多久可以注入数据）。我们的理论分析表明，这些攻击可能会误导上置信界（UCB）和汤普森抽样算法在几乎所有回合中选择目标手臂，而仅产生次线性攻击成本。对合成和现实世界数据集的实验验证了我们策略的有效性，揭示了广泛使用的随机强盗算法在实际对抗场景下的显着漏洞。



## **29. RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments**

RedTeamCUA：混合Web操作系统环境中计算机使用代理的现实对抗测试 cs.CL

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.21936v1) [paper-pdf](http://arxiv.org/pdf/2505.21936v1)

**Authors**: Zeyi Liao, Jaylen Jones, Linxi Jiang, Eric Fosler-Lussier, Yu Su, Zhiqiang Lin, Huan Sun

**Abstract**: Computer-use agents (CUAs) promise to automate complex tasks across operating systems (OS) and the web, but remain vulnerable to indirect prompt injection. Current evaluations of this threat either lack support realistic but controlled environments or ignore hybrid web-OS attack scenarios involving both interfaces. To address this, we propose RedTeamCUA, an adversarial testing framework featuring a novel hybrid sandbox that integrates a VM-based OS environment with Docker-based web platforms. Our sandbox supports key features tailored for red teaming, such as flexible adversarial scenario configuration, and a setting that decouples adversarial evaluation from navigational limitations of CUAs by initializing tests directly at the point of an adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive benchmark with 864 examples that investigate realistic, hybrid web-OS attack scenarios and fundamental security vulnerabilities. Benchmarking current frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated, still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute adversarial tasks with an Attempt Rate as high as 92.5%, although failing to complete them due to capability limitations. Nevertheless, we observe concerning ASRs of up to 50% in realistic end-to-end settings, with the recently released frontier Claude 4 Opus | CUA showing an alarming ASR of 48%, demonstrating that indirect prompt injection presents tangible risks for even advanced CUAs despite their capabilities and safeguards. Overall, RedTeamCUA provides an essential framework for advancing realistic, controlled, and systematic analysis of CUA vulnerabilities, highlighting the urgent need for robust defenses to indirect prompt injection prior to real-world deployment.

摘要: 计算机使用代理（CUA）承诺在操作系统（OS）和网络上自动化复杂任务，但仍然容易受到间接提示注入的影响。当前对该威胁的评估要么缺乏对现实但受控的环境的支持，要么忽视了涉及两个接口的混合Web操作系统攻击场景。为了解决这个问题，我们提出了RedTeamCUA，这是一种对抗性测试框架，具有新型混合沙盒，该沙盒将基于虚拟机的操作系统环境与基于Docker的Web平台集成在一起。我们的沙箱支持为红色团队定制的关键功能，例如灵活的对抗场景配置，以及通过在对抗注入时直接初始化测试来将对抗评估与CUA的导航限制分开的设置。使用RedTeamCUA，我们开发RTC-Bench，这是一个包含864个示例的综合基准测试，可以调查现实的混合Web操作系统攻击场景和基本安全漏洞。对当前前沿CUA进行基准测试发现重大漏洞：Claude 3.7十四行诗|CUA的ASB为42.9%，而受评估的最安全的CUA Operator仍为7.6%。值得注意的是，CUA经常尝试执行尝试率高达92.5%的对抗任务，尽管由于能力限制而未能完成这些任务。尽管如此，我们观察到，在现实的端到端环境中，最近发布的前沿《Claude 4 Opus》中，ASB高达50%| CUA显示出惊人的48%的ASB，这表明间接即时注射即使是先进的CUA，也会带来切实的风险，尽管它们有能力和保障措施。总体而言，RedTeamCUA为推进对CUA漏洞的现实、受控和系统性分析提供了一个重要框架，凸显了在现实世界部署之前对间接提示注入的强大防御措施的迫切需要。



## **30. Rethinking Gradient-based Adversarial Attacks on Point Cloud Classification**

重新思考点云分类的基于一致性的对抗性攻击 cs.CV

**SubmitDate**: 2025-05-28    [abs](http://arxiv.org/abs/2505.21854v1) [paper-pdf](http://arxiv.org/pdf/2505.21854v1)

**Authors**: Jun Chen, Xinke Li, Mingyue Xu, Tianrui Li, Chongshou Li

**Abstract**: Gradient-based adversarial attacks have become a dominant approach for evaluating the robustness of point cloud classification models. However, existing methods often rely on uniform update rules that fail to consider the heterogeneous nature of point clouds, resulting in excessive and perceptible perturbations. In this paper, we rethink the design of gradient-based attacks by analyzing the limitations of conventional gradient update mechanisms and propose two new strategies to improve both attack effectiveness and imperceptibility. First, we introduce WAAttack, a novel framework that incorporates weighted gradients and an adaptive step-size strategy to account for the non-uniform contribution of points during optimization. This approach enables more targeted and subtle perturbations by dynamically adjusting updates according to the local structure and sensitivity of each point. Second, we propose SubAttack, a complementary strategy that decomposes the point cloud into subsets and focuses perturbation efforts on structurally critical regions. Together, these methods represent a principled rethinking of gradient-based adversarial attacks for 3D point cloud classification. Extensive experiments demonstrate that our approach outperforms state-of-the-art baselines in generating highly imperceptible adversarial examples. Code will be released upon paper acceptance.

摘要: 基于对象的对抗攻击已成为评估点云分类模型稳健性的主要方法。然而，现有的方法通常依赖于统一的更新规则，而无法考虑点云的异类性质，从而导致过度且可感知的扰动。本文通过分析传统梯度更新机制的局限性，重新思考基于梯度的攻击的设计，并提出两种新策略来提高攻击有效性和不可感知性。首先，我们介绍WAAttack，这是一个新颖的框架，它结合了加权梯度和自适应的步进策略，以考虑优化期间点的非均匀贡献。该方法通过根据每个点的局部结构和敏感性动态调整更新，实现更有针对性和微妙的扰动。其次，我们提出了SubAttack，这是一种补充策略，将点云分解为子集，并将扰动工作集中在结构关键区域。这些方法共同代表了对3D点云分类的基于梯度的对抗攻击的原则性重新思考。大量实验表明，我们的方法在生成高度难以察觉的对抗示例方面优于最先进的基线。代码将在纸质接受后发布。



## **31. What is Adversarial Training for Diffusion Models?**

什么是扩散模型的对抗训练？ cs.CV

40 pages

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21742v1) [paper-pdf](http://arxiv.org/pdf/2505.21742v1)

**Authors**: Briglia Maria Rosaria, Mujtaba Hussain Mirza, Giuseppe Lisanti, Iacopo Masi

**Abstract**: We answer the question in the title, showing that adversarial training (AT) for diffusion models (DMs) fundamentally differs from classifiers: while AT in classifiers enforces output invariance, AT in DMs requires equivariance to keep the diffusion process aligned with the data distribution. AT is a way to enforce smoothness in the diffusion flow, improving robustness to outliers and corrupted data. Unlike prior art, our method makes no assumptions about the noise model and integrates seamlessly into diffusion training by adding random noise, similar to randomized smoothing, or adversarial noise, akin to AT. This enables intrinsic capabilities such as handling noisy data, dealing with extreme variability such as outliers, preventing memorization, and improving robustness. We rigorously evaluate our approach with proof-of-concept datasets with known distributions in low- and high-dimensional space, thereby taking a perfect measure of errors; we further evaluate on standard benchmarks such as CIFAR-10, CelebA and LSUN Bedroom, showing strong performance under severe noise, data corruption, and iterative adversarial attacks.

摘要: 我们回答了标题中的问题，表明扩散模型（DM）的对抗训练（AT）与分类器有着根本不同：分类器中的对抗训练（AT）强制输出不变性，而DM中的对抗训练（AT）需要等变性以保持扩散过程与数据分布一致。AT是一种加强扩散流平滑性的方法，提高对异常值和损坏数据的稳健性。与现有技术不同，我们的方法不对噪音模型做出任何假设，并通过添加随机噪音（类似于随机平滑）或对抗噪音（类似于AT）无缝集成到扩散训练中。这实现了固有功能，例如处理有噪数据、处理异常值等极端变异性、防止记忆和提高稳健性。我们使用低维和高维空间中具有已知分布的概念验证数据集严格评估我们的方法，从而对错误进行完美的测量;我们进一步评估标准基准，如CIFAR-10，CelebA和LSUN Bedroom，在严重的噪声，数据损坏和迭代对抗攻击下表现出强大的性能。



## **32. Lazarus Group Targets Crypto-Wallets and Financial Data while employing new Tradecrafts**

Lazarus Group瞄准加密钱包和金融数据，同时采用新的Tradecrafts cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21725v1) [paper-pdf](http://arxiv.org/pdf/2505.21725v1)

**Authors**: Alessio Di Santo

**Abstract**: This report presents a comprehensive analysis of a malicious software sample, detailing its architecture, behavioral characteristics, and underlying intent. Through static and dynamic examination, the malware core functionalities, including persistence mechanisms, command-and-control communication, and data exfiltration routines, are identified and its supporting infrastructure is mapped. By correlating observed indicators of compromise with known techniques, tactics, and procedures, this analysis situates the sample within the broader context of contemporary threat campaigns and infers the capabilities and motivations of its likely threat actor.   Building on these findings, actionable threat intelligence is provided to support proactive defenses. Threat hunting teams receive precise detection hypotheses for uncovering latent adversarial presence, while monitoring systems can refine alert logic to detect anomalous activity in real time. Finally, the report discusses how this structured intelligence enhances predictive risk assessments, informs vulnerability prioritization, and strengthens organizational resilience against advanced persistent threats. By integrating detailed technical insights with strategic threat landscape mapping, this malware analysis report not only reconstructs past adversary actions but also establishes a robust foundation for anticipating and mitigating future attacks.

摘要: 本报告对恶意软件样本进行了全面分析，详细介绍了其架构、行为特征和潜在意图。通过静态和动态检查，识别恶意软件的核心功能，包括持久性机制、命令和控制通信以及数据溢出例程，并绘制其支持基础设施。通过将观察到的妥协指标与已知的技术、策略和程序关联起来，该分析将样本置于当代威胁活动的更广泛背景下，并推断其可能的威胁参与者的能力和动机。   在这些发现的基础上，提供可操作的威胁情报来支持主动防御。威胁搜寻团队接收精确的检测假设，以发现潜在的对抗存在，而监控系统可以完善警报逻辑以实时检测异常活动。最后，该报告讨论了这种结构化智能如何增强预测性风险评估、为漏洞优先排序提供信息以及加强组织应对高级持续威胁的弹性。通过将详细的技术见解与战略威胁格局映射集成，该恶意软件分析报告不仅重建了过去的对手行为，还为预测和减轻未来的攻击奠定了坚实的基础。



## **33. VideoMarkBench: Benchmarking Robustness of Video Watermarking**

VideoMarkBench：视频水印的鲁棒性基准 cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21620v1) [paper-pdf](http://arxiv.org/pdf/2505.21620v1)

**Authors**: Zhengyuan Jiang, Moyang Guo, Kecen Li, Yuepeng Hu, Yupu Wang, Zhicong Huang, Cheng Hong, Neil Zhenqiang Gong

**Abstract**: The rapid development of video generative models has led to a surge in highly realistic synthetic videos, raising ethical concerns related to disinformation and copyright infringement. Recently, video watermarking has been proposed as a mitigation strategy by embedding invisible marks into AI-generated videos to enable subsequent detection. However, the robustness of existing video watermarking methods against both common and adversarial perturbations remains underexplored. In this work, we introduce VideoMarkBench, the first systematic benchmark designed to evaluate the robustness of video watermarks under watermark removal and watermark forgery attacks. Our study encompasses a unified dataset generated by three state-of-the-art video generative models, across three video styles, incorporating four watermarking methods and seven aggregation strategies used during detection. We comprehensively evaluate 12 types of perturbations under white-box, black-box, and no-box threat models. Our findings reveal significant vulnerabilities in current watermarking approaches and highlight the urgent need for more robust solutions. Our code is available at https://github.com/zhengyuan-jiang/VideoMarkBench.

摘要: 视频生成模型的快速发展导致高度逼真的合成视频激增，引发了与虚假信息和版权侵权相关的道德担忧。最近，视频水印被提出作为一种缓解策略，通过将不可见标记嵌入人工智能生成的视频中以实现后续检测。然而，现有视频水印方法对常见和对抗扰动的鲁棒性仍然没有得到充分的研究。在这项工作中，我们介绍了VideoMarkBench，这是第一个旨在评估视频水印在水印去除和水印伪造攻击下的稳健性的系统基准。我们的研究涵盖了由三种最先进的视频生成模型生成的统一数据集，跨越三种视频风格，结合了检测期间使用的四种水印方法和七种聚合策略。我们在白盒、黑匣子和无箱威胁模型下全面评估了12种类型的扰动。我们的研究结果揭示了当前水印方法中的重大漏洞，并凸显了对更强大解决方案的迫切需求。我们的代码可在https://github.com/zhengyuan-jiang/VideoMarkBench上获取。



## **34. Preventing Adversarial AI Attacks Against Autonomous Situational Awareness: A Maritime Case Study**

防止针对自主情境感知的人工智能对抗攻击：海事案例研究 cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21609v1) [paper-pdf](http://arxiv.org/pdf/2505.21609v1)

**Authors**: Mathew J. Walter, Aaron Barrett, Kimberly Tam

**Abstract**: Adversarial artificial intelligence (AI) attacks pose a significant threat to autonomous transportation, such as maritime vessels, that rely on AI components. Malicious actors can exploit these systems to deceive and manipulate AI-driven operations. This paper addresses three critical research challenges associated with adversarial AI: the limited scope of traditional defences, inadequate security metrics, and the need to build resilience beyond model-level defences. To address these challenges, we propose building defences utilising multiple inputs and data fusion to create defensive components and an AI security metric as a novel approach toward developing more secure AI systems. We name this approach the Data Fusion Cyber Resilience (DFCR) method, and we evaluate it through real-world demonstrations and comprehensive quantitative analyses, comparing a system built with the DFCR method against single-input models and models utilising existing state-of-the-art defences. The findings show that the DFCR approach significantly enhances resilience against adversarial machine learning attacks in maritime autonomous system operations, achieving up to a 35\% reduction in loss for successful multi-pronged perturbation attacks, up to a 100\% reduction in loss for successful adversarial patch attacks and up to 100\% reduction in loss for successful spoofing attacks when using these more resilient systems. We demonstrate how DFCR and DFCR confidence scores can reduce adversarial AI contact confidence and improve decision-making by the system, even when typical adversarial defences have been compromised. Ultimately, this work contributes to the development of more secure and resilient AI-driven systems against adversarial attacks.

摘要: 对抗性人工智能（AI）攻击对依赖人工智能组件的自主交通（例如海上船只）构成了重大威胁。恶意行为者可以利用这些系统欺骗和操纵人工智能驱动的操作。本文解决了与对抗性人工智能相关的三个关键研究挑战：传统防御的范围有限、安全指标不足以及需要在模型级防御之外建立弹性。为了应对这些挑战，我们提议利用多个输入和数据融合来创建防御组件和人工智能安全指标来构建防御，作为开发更安全人工智能系统的新方法。我们将这种方法命名为数据融合网络韧性（DFCR）方法，并通过现实世界的演示和全面的定量分析对其进行评估，将使用DFCR方法构建的系统与单输入模型和利用现有最先进防御的模型进行比较。研究结果表明，DFCR方法显着增强了海上自治系统操作中针对对抗性机器学习攻击的弹性，成功的多管齐下干扰攻击的损失减少高达35%，成功的对抗性补丁攻击的损失减少高达100%，成功的欺骗攻击的损失减少高达100%。使用这些更具弹性的系统时，成功的欺骗攻击的损失减少高达100%。我们展示了DFCR和DFCR信心分数如何降低对抗性人工智能接触信心并改善系统的决策，即使典型的对抗性防御已受到损害。最终，这项工作有助于开发更安全、更有弹性的人工智能驱动系统来抵御对抗攻击。



## **35. AdInject: Real-World Black-Box Attacks on Web Agents via Advertising Delivery**

AdInjects：通过广告交付对Web代理进行现实世界的黑匣子攻击 cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21499v1) [paper-pdf](http://arxiv.org/pdf/2505.21499v1)

**Authors**: Haowei Wang, Junjie Wang, Xiaojun Jia, Rupeng Zhang, Mingyang Li, Zhe Liu, Yang Liu, Qing Wang

**Abstract**: Vision-Language Model (VLM) based Web Agents represent a significant step towards automating complex tasks by simulating human-like interaction with websites. However, their deployment in uncontrolled web environments introduces significant security vulnerabilities. Existing research on adversarial environmental injection attacks often relies on unrealistic assumptions, such as direct HTML manipulation, knowledge of user intent, or access to agent model parameters, limiting their practical applicability. In this paper, we propose AdInject, a novel and real-world black-box attack method that leverages the internet advertising delivery to inject malicious content into the Web Agent's environment. AdInject operates under a significantly more realistic threat model than prior work, assuming a black-box agent, static malicious content constraints, and no specific knowledge of user intent. AdInject includes strategies for designing malicious ad content aimed at misleading agents into clicking, and a VLM-based ad content optimization technique that infers potential user intents from the target website's context and integrates these intents into the ad content to make it appear more relevant or critical to the agent's task, thus enhancing attack effectiveness. Experimental evaluations demonstrate the effectiveness of AdInject, attack success rates exceeding 60% in most scenarios and approaching 100% in certain cases. This strongly demonstrates that prevalent advertising delivery constitutes a potent and real-world vector for environment injection attacks against Web Agents. This work highlights a critical vulnerability in Web Agent security arising from real-world environment manipulation channels, underscoring the urgent need for developing robust defense mechanisms against such threats. Our code is available at https://github.com/NicerWang/AdInject.

摘要: 基于视觉语言模型（VLM）的Web代理通过模拟与网站的类人交互来实现复杂任务自动化迈出了重要一步。然而，它们在不受控制的Web环境中的部署会带来严重的安全漏洞。现有的针对对抗性环境注入攻击的研究通常依赖于不切实际的假设，例如直接的HTML操作、用户意图的知识或对代理模型参数的访问，限制了它们的实际适用性。在本文中，我们提出了AdInib，这是一种新颖的现实世界黑匣子攻击方法，它利用互联网广告交付将恶意内容注入到Web代理的环境中。与之前的工作相比，AdInib在一个明显更现实的威胁模型下运行，假设黑匣子代理、静态恶意内容约束并且没有对用户意图的具体了解。AdInjects包括设计恶意广告内容的策略，旨在误导代理点击，以及一种基于LMA的广告内容优化技术，该技术从目标网站的上下文推断潜在的用户意图，并将这些意图集成到广告内容中，使其看起来与代理的任务更相关或至关重要，从而增强攻击有效性。实验评估证明了AdInib的有效性，在大多数情况下攻击成功率超过60%，在某些情况下攻击成功率接近100%。这有力地表明，流行的广告交付构成了针对Web代理的环境注入攻击的强大现实载体。这项工作强调了现实世界环境操纵渠道引起的Web Agent安全中的一个关键漏洞，强调了开发针对此类威胁的强大防御机制的迫切需要。我们的代码可在https://github.com/NicerWang/AdInject上获取。



## **36. Adversarial Attacks against Closed-Source MLLMs via Feature Optimal Alignment**

通过特征最佳对齐对闭源MLLM的对抗攻击 cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21494v1) [paper-pdf](http://arxiv.org/pdf/2505.21494v1)

**Authors**: Xiaojun Jia, Sensen Gao, Simeng Qin, Tianyu Pang, Chao Du, Yihao Huang, Xinfeng Li, Yiming Li, Bo Li, Yang Liu

**Abstract**: Multimodal large language models (MLLMs) remain vulnerable to transferable adversarial examples. While existing methods typically achieve targeted attacks by aligning global features-such as CLIP's [CLS] token-between adversarial and target samples, they often overlook the rich local information encoded in patch tokens. This leads to suboptimal alignment and limited transferability, particularly for closed-source models. To address this limitation, we propose a targeted transferable adversarial attack method based on feature optimal alignment, called FOA-Attack, to improve adversarial transfer capability. Specifically, at the global level, we introduce a global feature loss based on cosine similarity to align the coarse-grained features of adversarial samples with those of target samples. At the local level, given the rich local representations within Transformers, we leverage clustering techniques to extract compact local patterns to alleviate redundant local features. We then formulate local feature alignment between adversarial and target samples as an optimal transport (OT) problem and propose a local clustering optimal transport loss to refine fine-grained feature alignment. Additionally, we propose a dynamic ensemble model weighting strategy to adaptively balance the influence of multiple models during adversarial example generation, thereby further improving transferability. Extensive experiments across various models demonstrate the superiority of the proposed method, outperforming state-of-the-art methods, especially in transferring to closed-source MLLMs. The code is released at https://github.com/jiaxiaojunQAQ/FOA-Attack.

摘要: 多模式大型语言模型（MLLM）仍然容易受到可转移的对抗示例的影响。虽然现有方法通常通过在对抗样本和目标样本之间对齐全局特征（例如CLIP的[LIS]标记）来实现有针对性的攻击，但它们经常忽视补丁令牌中编码的丰富本地信息。这导致次优的对齐和有限的可移植性，特别是对于闭源模型。为了解决这一局限性，我们提出了一种基于特征最优对齐的有针对性的可转移对抗攻击方法，称为FOA-Attack，以提高对抗转移能力。具体来说，在全球层面，我们引入了基于cos相似性的全球特征损失，以将对抗样本的粗粒度特征与目标样本的粗粒度特征对齐。在局部层面，鉴于变形金刚中丰富的局部表示，我们利用集群技术来提取紧凑的局部模式，以减轻冗余的局部特征。然后，我们将对抗样本和目标样本之间的局部特征对齐公式化为最优传输（OT）问题，并提出局部集群最优传输损失来细化细粒度特征对齐。此外，我们还提出了一种动态集成模型加权策略，以自适应地平衡对抗性示例生成过程中多个模型的影响，从而进一步提高可移植性。跨各种模型的广泛实验证明了所提出方法的优越性，优于最先进的方法，特别是在转移到闭源MLLM方面。该代码发布于https://github.com/jiaxiaojunQAQ/FOA-Attack。



## **37. When Are Concepts Erased From Diffusion Models?**

概念何时从扩散模型中删除？ cs.LG

Project Page:  https://nyu-dice-lab.github.io/when-are-concepts-erased/

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.17013v3) [paper-pdf](http://arxiv.org/pdf/2505.17013v3)

**Authors**: Kevin Lu, Nicky Kriplani, Rohit Gandikota, Minh Pham, David Bau, Chinmay Hegde, Niv Cohen

**Abstract**: Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.

摘要: 概念擦除，即选择性地阻止模型生成特定概念的能力，引起了越来越多的兴趣，各种方法出现了来应对这一挑战。然而，目前尚不清楚这些方法如何彻底消除目标概念。我们首先提出了扩散模型中擦除机制的两个概念模型：（i）降低生成目标概念的可能性，（ii）干扰模型的内部引导机制。为了彻底评估某个概念是否已真正从模型中删除，我们引入了一套独立评估。我们的评估框架包括对抗性攻击、新颖的探测技术以及对模型替代世代的分析，以取代被删除的概念。我们的结果揭示了最大限度地减少副作用和保持对抗提示的鲁棒性之间的紧张关系。从广义上讲，我们的工作强调了对扩散模型中擦除进行全面评估的重要性。



## **38. On the Robustness of Adversarial Training Against Uncertainty Attacks**

论对抗训练对不确定性攻击的鲁棒性 cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2410.21952v2) [paper-pdf](http://arxiv.org/pdf/2410.21952v2)

**Authors**: Emanuele Ledda, Giovanni Scodeller, Daniele Angioni, Giorgio Piras, Antonio Emanuele Cinà, Giorgio Fumera, Battista Biggio, Fabio Roli

**Abstract**: In learning problems, the noise inherent to the task at hand hinders the possibility to infer without a certain degree of uncertainty. Quantifying this uncertainty, regardless of its wide use, assumes high relevance for security-sensitive applications. Within these scenarios, it becomes fundamental to guarantee good (i.e., trustworthy) uncertainty measures, which downstream modules can securely employ to drive the final decision-making process. However, an attacker may be interested in forcing the system to produce either (i) highly uncertain outputs jeopardizing the system's availability or (ii) low uncertainty estimates, making the system accept uncertain samples that would instead require a careful inspection (e.g., human intervention). Therefore, it becomes fundamental to understand how to obtain robust uncertainty estimates against these kinds of attacks. In this work, we reveal both empirically and theoretically that defending against adversarial examples, i.e., carefully perturbed samples that cause misclassification, additionally guarantees a more secure, trustworthy uncertainty estimate under common attack scenarios without the need for an ad-hoc defense strategy. To support our claims, we evaluate multiple adversarial-robust models from the publicly available benchmark RobustBench on the CIFAR-10 and ImageNet datasets.

摘要: 在学习问题中，手头任务固有的噪音阻碍了在没有一定不确定性的情况下进行推断的可能性。量化这种不确定性，无论其是否广泛使用，都假设与安全敏感应用程序高度相关。在这些场景中，保证良好（即，值得信赖的）不确定性措施，下游模块可以安全地使用这些措施来驱动最终的决策过程。然而，攻击者可能有兴趣强迫系统产生（i）高度不确定的输出，危及系统的可用性，或（ii）低不确定性估计，使系统接受不确定的样本，而这些样本将需要仔细检查（例如，人类干预）。因此，了解如何针对此类攻击获得稳健的不确定性估计变得至关重要。在这项工作中，我们从经验和理论上揭示了防御对抗性例子，即，经过精心扰动的样本会导致错误分类，还可以在常见攻击场景下保证更安全、更值得信赖的不确定性估计，而无需临时防御策略。为了支持我们的主张，我们在CIFAR-10和ImageNet数据集上评估了来自公开基准RobustBench的多个对抗稳健模型。



## **39. Attribute-Efficient PAC Learning of Sparse Halfspaces with Constant Malicious Noise Rate**

稀疏半空间中恒定恶意噪声率下的属性有效PAC学习 cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21430v1) [paper-pdf](http://arxiv.org/pdf/2505.21430v1)

**Authors**: Shiwei Zeng, Jie Shen

**Abstract**: Attribute-efficient learning of sparse halfspaces has been a fundamental problem in machine learning theory. In recent years, machine learning algorithms are faced with prevalent data corruptions or even adversarial attacks. It is of central interest to design efficient algorithms that are robust to noise corruptions. In this paper, we consider that there exists a constant amount of malicious noise in the data and the goal is to learn an underlying $s$-sparse halfspace $w^* \in \mathbb{R}^d$ with $\text{poly}(s,\log d)$ samples. Specifically, we follow a recent line of works and assume that the underlying distribution satisfies a certain concentration condition and a margin condition at the same time. Under such conditions, we show that attribute-efficiency can be achieved by simple variants to existing hinge loss minimization programs. Our key contribution includes: 1) an attribute-efficient PAC learning algorithm that works under constant malicious noise rate; 2) a new gradient analysis that carefully handles the sparsity constraint in hinge loss minimization.

摘要: 稀疏半空间的属性高效学习一直是机器学习理论中的一个基本问题。近年来，机器学习算法面临着普遍的数据损坏甚至对抗性攻击。设计对噪音破坏具有鲁棒性的高效算法是最重要的。在本文中，我们认为数据中存在恒定数量的恶意噪音，目标是学习底层的$s$-稀疏半空间$w #\in \mathbb{R}' d$，带有$\text{poly}（s，\log d）$ samples。具体来说，我们遵循最近的工作路线，并假设基本分布同时满足一定的集中度条件和边际条件。在这种条件下，我们表明属性效率可以通过现有铰链损失最小化程序的简单变体来实现。我们的主要贡献包括：1）一种在恒定恶意噪音率下工作的属性高效PAC学习算法; 2）一种新的梯度分析，仔细处理铰链损失最小化中的稀疏性约束。



## **40. A Framework for Adversarial Analysis of Decision Support Systems Prior to Deployment**

决策支持系统部署前的对抗性分析框架 cs.LG

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21414v1) [paper-pdf](http://arxiv.org/pdf/2505.21414v1)

**Authors**: Brett Bissey, Kyle Gatesman, Walker Dimon, Mohammad Alam, Luis Robaina, Joseph Weissman

**Abstract**: This paper introduces a comprehensive framework designed to analyze and secure decision-support systems trained with Deep Reinforcement Learning (DRL), prior to deployment, by providing insights into learned behavior patterns and vulnerabilities discovered through simulation. The introduced framework aids in the development of precisely timed and targeted observation perturbations, enabling researchers to assess adversarial attack outcomes within a strategic decision-making context. We validate our framework, visualize agent behavior, and evaluate adversarial outcomes within the context of a custom-built strategic game, CyberStrike. Utilizing the proposed framework, we introduce a method for systematically discovering and ranking the impact of attacks on various observation indices and time-steps, and we conduct experiments to evaluate the transferability of adversarial attacks across agent architectures and DRL training algorithms. The findings underscore the critical need for robust adversarial defense mechanisms to protect decision-making policies in high-stakes environments.

摘要: 本文介绍了一个全面的框架，旨在在部署之前分析和保护经过深度强化学习（DRL）训练的决策支持系统，通过提供对通过模拟发现的行为模式和漏洞的见解。引入的框架有助于开发精确定时和有针对性的观察扰动，使研究人员能够在战略决策背景下评估对抗性攻击的结果。我们在定制的战略游戏CyberStrike的背景下验证我们的框架、可视化代理行为并评估对抗结果。利用提出的框架，我们引入了一种系统性地发现和排名攻击对各种观察指标和时间步的影响的方法，并进行实验来评估对抗性攻击跨代理架构和DRL训练算法的可转移性。研究结果强调，迫切需要强大的对抗性防御机制来保护高风险环境中的决策政策。



## **41. Optimizing Robustness and Accuracy in Mixture of Experts: A Dual-Model Approach**

混合专家系统鲁棒性和准确性的双模型优化方法 cs.LG

15 pages, 7 figures, accepted by ICML 2025

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2502.06832v3) [paper-pdf](http://arxiv.org/pdf/2502.06832v3)

**Authors**: Xu Zhang, Kaidi Xu, Ziqing Hu, Ren Wang

**Abstract**: Mixture of Experts (MoE) have shown remarkable success in leveraging specialized expert networks for complex machine learning tasks. However, their susceptibility to adversarial attacks presents a critical challenge for deployment in robust applications. This paper addresses the critical question of how to incorporate robustness into MoEs while maintaining high natural accuracy. We begin by analyzing the vulnerability of MoE components, finding that expert networks are notably more susceptible to adversarial attacks than the router. Based on this insight, we propose a targeted robust training technique that integrates a novel loss function to enhance the adversarial robustness of MoE, requiring only the robustification of one additional expert without compromising training or inference efficiency. Building on this, we introduce a dual-model strategy that linearly combines a standard MoE model with our robustified MoE model using a smoothing parameter. This approach allows for flexible control over the robustness-accuracy trade-off. We further provide theoretical foundations by deriving certified robustness bounds for both the single MoE and the dual-model. To push the boundaries of robustness and accuracy, we propose a novel joint training strategy JTDMoE for the dual-model. This joint training enhances both robustness and accuracy beyond what is achievable with separate models. Experimental results on CIFAR-10 and TinyImageNet datasets using ResNet18 and Vision Transformer (ViT) architectures demonstrate the effectiveness of our proposed methods. The code is publicly available at https://github.com/TIML-Group/Robust-MoE-Dual-Model.

摘要: 混合专家（MoE）在利用专业专家网络执行复杂的机器学习任务方面取得了显着的成功。然而，它们对对抗攻击的敏感性给在强大应用程序中的部署带来了严峻的挑战。本文解决了如何将稳健性融入MoE同时保持高自然准确性的关键问题。我们首先分析MoE组件的漏洞，发现专家网络明显比路由器更容易受到对抗攻击。基于这一见解，我们提出了一种有针对性的鲁棒训练技术，该技术集成了一种新颖的损失函数来增强MoE的对抗鲁棒性，只需要一名额外专家的鲁棒性，而不会损害训练或推理效率。在此基础上，我们引入了一种双模型策略，该策略使用平滑参数将标准MoE模型与我们的鲁棒化MoE模型线性结合。这种方法允许灵活控制鲁棒性-准确性权衡。我们通过推导单MoE和双模型的经认证的鲁棒性界限来进一步提供理论基础。为了突破稳健性和准确性的界限，我们提出了一种新颖的双模型联合训练策略JTDMoE。这种联合训练增强了鲁棒性和准确性，超出了单独模型所能实现的能力。使用ResNet 18和Vision Transformer（ViT）架构在CIFAR-10和TinyImageNet数据集上的实验结果证明了我们提出的方法的有效性。该代码可在https://github.com/TIML-Group/Robust-MoE-Dual-Model上公开获取。



## **42. Boosting Adversarial Transferability via High-Frequency Augmentation and Hierarchical-Gradient Fusion**

通过高频增强和分层梯度融合提高对抗性可移植性 cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21181v1) [paper-pdf](http://arxiv.org/pdf/2505.21181v1)

**Authors**: Yayin Zheng, Chen Wan, Zihong Guo, Hailing Kuang, Xiaohai Lu

**Abstract**: Adversarial attacks have become a significant challenge in the security of machine learning models, particularly in the context of black-box defense strategies. Existing methods for enhancing adversarial transferability primarily focus on the spatial domain. This paper presents Frequency-Space Attack (FSA), a new adversarial attack framework that effectively integrates frequency-domain and spatial-domain transformations. FSA combines two key techniques: (1) High-Frequency Augmentation, which applies Fourier transform with frequency-selective amplification to diversify inputs and emphasize the critical role of high-frequency components in adversarial attacks, and (2) Hierarchical-Gradient Fusion, which merges multi-scale gradient decomposition and fusion to capture both global structures and fine-grained details, resulting in smoother perturbations. Our experiment demonstrates that FSA consistently outperforms state-of-the-art methods across various black-box models. Notably, our proposed FSA achieves an average attack success rate increase of 23.6% compared with BSR (CVPR 2024) on eight black-box defense models.

摘要: 对抗性攻击已经成为机器学习模型安全性的一个重大挑战，特别是在黑盒防御策略的背景下。用于增强对抗性可转移性的现有方法主要集中在空间域。本文提出了一种新的对抗性攻击框架--频率-空间攻击（FSA），它有效地集成了频域和空间域变换。FSA结合了两项关键技术：（1）高频增强，应用傅里叶变换和频率选择性放大来实现输入多样化，并强调高频分量在对抗攻击中的关键作用，（2）分层梯度融合，融合多尺度梯度分解和融合，以捕获全局结构和细粒度细节，从而产生更平滑的扰动。我们的实验表明，FSA在各种黑匣子模型中始终优于最先进的方法。值得注意的是，我们提出的FSA在八种黑匣子防御模型上的平均攻击成功率与SVR（CVPR 2024）相比提高了23.6%。



## **43. Unraveling Indirect In-Context Learning Using Influence Functions**

使用影响函数解开间接上下文学习 cs.LG

Under Review

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2501.01473v2) [paper-pdf](http://arxiv.org/pdf/2501.01473v2)

**Authors**: Hadi Askari, Shivanshu Gupta, Terry Tong, Fei Wang, Anshuman Chhabra, Muhao Chen

**Abstract**: In this work, we introduce a novel paradigm for generalized In-Context Learning (ICL), termed Indirect In-Context Learning. In Indirect ICL, we explore demonstration selection strategies tailored for two distinct real-world scenarios: Mixture of Tasks and Noisy ICL. We systematically evaluate the effectiveness of Influence Functions (IFs) as a selection tool for these settings, highlighting the potential of IFs to better capture the informativeness of examples within the demonstration pool. For the Mixture of Tasks setting, demonstrations are drawn from 28 diverse tasks, including MMLU, BigBench, StrategyQA, and CommonsenseQA. We demonstrate that combining BertScore-Recall (BSR) with an IF surrogate model can further improve performance, leading to average absolute accuracy gains of 0.37\% and 1.45\% for 3-shot and 5-shot setups when compared to traditional ICL metrics. In the Noisy ICL setting, we examine scenarios where demonstrations might be mislabeled or have adversarial noise. Our experiments show that reweighting traditional ICL selectors (BSR and Cosine Similarity) with IF-based selectors boosts accuracy by an average of 2.90\% for Cosine Similarity and 2.94\% for BSR on noisy GLUE benchmarks. For the adversarial sub-setting, we show the utility of using IFs for task-agnostic demonstration selection for backdoor attack mitigation. Showing a 32.89\% reduction in Attack Success Rate compared to task-aware methods. In sum, we propose a robust framework for demonstration selection that generalizes beyond traditional ICL, offering valuable insights into the role of IFs for Indirect ICL.

摘要: 在这项工作中，我们引入了一种新的广义内上下文学习（ICL）范式，称为间接内上下文学习。在间接ICL中，我们探索了为两种不同的现实世界场景量身定制的演示选择策略：任务混合和噪音ICL。我们系统地评估影响力函数（IF）作为这些环境的选择工具的有效性，强调了IF更好地捕捉演示池中示例信息量的潜力。对于“任务混合”设置，演示来自28个不同的任务，包括MMLU、BigBench、StrategyQA和CommonsenseQA。我们证明，将BertScore-Recall（SVR）与IF代理模型相结合可以进一步提高性能，与传统ICL指标相比，3次和5次设置的平均绝对准确性提高为0.37%和1.45%。在有噪音的ICL环境中，我们检查演示可能被错误标记或具有对抗性噪音的场景。我们的实验表明，用基于IF的选择器重新加权传统ICL选择器（BEP和Cosine相似性），在有噪的GLUE基准上，Cosine相似性的准确性平均提高了2.90%，而SVR的准确性平均提高了2.94%。对于对抗性子设置，我们展示了使用IF进行任务不可知演示选择以缓解后门攻击的实用性。与任务感知方法相比，攻击成功率降低了32.89%。总而言之，我们提出了一个强大的示范选择框架，该框架超越了传统ICL，为间接ICL中的国际单项指标的作用提供了有价值的见解。



## **44. TabAttackBench: A Benchmark for Adversarial Attacks on Tabular Data**

TabAttackBench：表格数据对抗性攻击的基准 cs.LG

63 pages, 22 figures, 6 tables

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21027v1) [paper-pdf](http://arxiv.org/pdf/2505.21027v1)

**Authors**: Zhipeng He, Chun Ouyang, Lijie Wen, Cong Liu, Catarina Moreira

**Abstract**: Adversarial attacks pose a significant threat to machine learning models by inducing incorrect predictions through imperceptible perturbations to input data. While these attacks have been extensively studied in unstructured data like images, their application to tabular data presents new challenges. These challenges arise from the inherent heterogeneity and complex feature interdependencies in tabular data, which differ significantly from those in image data. To address these differences, it is crucial to consider imperceptibility as a key criterion specific to tabular data. Most current research focuses primarily on achieving effective adversarial attacks, often overlooking the importance of maintaining imperceptibility. To address this gap, we propose a new benchmark for adversarial attacks on tabular data that evaluates both effectiveness and imperceptibility. In this study, we assess the effectiveness and imperceptibility of five adversarial attacks across four models using eleven tabular datasets, including both mixed and numerical-only datasets. Our analysis explores how these factors interact and influence the overall performance of the attacks. We also compare the results across different dataset types to understand the broader implications of these findings. The findings from this benchmark provide valuable insights for improving the design of adversarial attack algorithms, thereby advancing the field of adversarial machine learning on tabular data.

摘要: 对抗性攻击通过对输入数据的不可感知的扰动引发错误预测，对机器学习模型构成重大威胁。虽然这些攻击在图像等非结构化数据中得到了广泛研究，但它们对表格数据的应用带来了新的挑战。这些挑战源于表格数据中固有的同质性和复杂的特征相互依赖性，而表格数据与图像数据中的数据有显着不同。为了解决这些差异，将不可感知性视为表格数据特定的关键标准至关重要。当前的大多数研究主要关注于实现有效的对抗攻击，而往往忽视了保持不可感知性的重要性。为了解决这一差距，我们提出了一个针对表格数据的对抗攻击的新基准，该基准评估有效性和不可感知性。在这项研究中，我们使用11个表格数据集（包括混合数据集和纯数字数据集）评估了四种模型中五种对抗攻击的有效性和不可感知性。我们的分析探讨了这些因素如何相互作用并影响攻击的整体性能。我们还比较了不同数据集类型的结果，以了解这些发现的更广泛影响。该基准的研究结果为改进对抗性攻击算法的设计提供了宝贵的见解，从而推进表格数据上的对抗性机器学习领域。



## **45. Tradeoffs Between Alignment and Helpfulness in Language Models with Steering Methods**

具有引导方法的语言模型中的一致性和帮助性之间的权衡 cs.CL

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2401.16332v5) [paper-pdf](http://arxiv.org/pdf/2401.16332v5)

**Authors**: Yotam Wolf, Noam Wies, Dorin Shteyman, Binyamin Rothberg, Yoav Levine, Amnon Shashua

**Abstract**: Language model alignment has become an important component of AI safety, allowing safe interactions between humans and language models, by enhancing desired behaviors and inhibiting undesired ones. It is often done by tuning the model or inserting preset aligning prompts. Recently, representation engineering, a method which alters the model's behavior via changing its representations post-training, was shown to be effective in aligning LLMs (Zou et al., 2023a). Representation engineering yields gains in alignment oriented tasks such as resistance to adversarial attacks and reduction of social biases, but was also shown to cause a decrease in the ability of the model to perform basic tasks. In this paper we study the tradeoff between the increase in alignment and decrease in helpfulness of the model. We propose a theoretical framework which provides bounds for these two quantities, and demonstrate their relevance empirically. First, we find that under the conditions of our framework, alignment can be guaranteed with representation engineering, and at the same time that helpfulness is harmed in the process. Second, we show that helpfulness is harmed quadratically with the norm of the representation engineering vector, while the alignment increases linearly with it, indicating a regime in which it is efficient to use representation engineering. We validate our findings empirically, and chart the boundaries to the usefulness of representation engineering for alignment.

摘要: 语言模型对齐已成为人工智能安全的重要组成部分，通过增强期望的行为和抑制不期望的行为，允许人类与语言模型之间的安全交互。通常通过调整模型或插入预设对齐提示来完成。最近，表示工程（一种通过在训练后改变模型的表示来改变模型行为的方法）被证明在对齐LLM方面有效（Zou等人，2023 a）。表示工程在以对齐为导向的任务中产生了收益，例如抵抗对抗攻击和减少社会偏见，但也被证明会导致模型执行基本任务的能力下降。在本文中，我们研究了模型对齐度的增加和帮助性的减少之间的权衡。我们提出了一个理论框架，为这两个量提供了界限，并从经验上证明了它们的相关性。首先，我们发现在我们的框架条件下，可以通过表示工程来保证一致性，同时在这个过程中会损害帮助性。其次，我们表明，帮助性与表示工程载体的规范成二次关系受到损害，而对齐度则随其线性增加，这表明使用表示工程是有效的制度。我们从经验上验证了我们的发现，并绘制了对齐表示工程的有用性的界限。



## **46. NatADiff: Adversarial Boundary Guidance for Natural Adversarial Diffusion**

NatADiff：自然对抗扩散的对抗边界指南 cs.LG

10 pages, 3 figures, 2 tables

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20934v1) [paper-pdf](http://arxiv.org/pdf/2505.20934v1)

**Authors**: Max Collins, Jordan Vice, Tim French, Ajmal Mian

**Abstract**: Adversarial samples exploit irregularities in the manifold ``learned'' by deep learning models to cause misclassifications. The study of these adversarial samples provides insight into the features a model uses to classify inputs, which can be leveraged to improve robustness against future attacks. However, much of the existing literature focuses on constrained adversarial samples, which do not accurately reflect test-time errors encountered in real-world settings. To address this, we propose `NatADiff', an adversarial sampling scheme that leverages denoising diffusion to generate natural adversarial samples. Our approach is based on the observation that natural adversarial samples frequently contain structural elements from the adversarial class. Deep learning models can exploit these structural elements to shortcut the classification process, rather than learning to genuinely distinguish between classes. To leverage this behavior, we guide the diffusion trajectory towards the intersection of the true and adversarial classes, combining time-travel sampling with augmented classifier guidance to enhance attack transferability while preserving image fidelity. Our method achieves comparable attack success rates to current state-of-the-art techniques, while exhibiting significantly higher transferability across model architectures and better alignment with natural test-time errors as measured by FID. These results demonstrate that NatADiff produces adversarial samples that not only transfer more effectively across models, but more faithfully resemble naturally occurring test-time errors.

摘要: 对抗性样本利用深度学习模型“学习”的多个分支中的不规则性来导致错误分类。对这些对抗性样本的研究可以深入了解模型用于分类输入的特征，这些特征可以用于提高针对未来攻击的鲁棒性。然而，大部分现有文献都集中在受约束的对抗样本上，这些样本并不能准确反映现实世界环境中遇到的测试时错误。为了解决这个问题，我们提出了“NatADiff”，这是一种对抗性采样方案，利用去噪扩散来生成自然对抗性样本。我们的方法基于这样的观察：自然对抗样本经常包含对抗类别的结构元素。深度学习模型可以利用这些结构元素来缩短分类过程，而不是学习真正区分类别。为了利用这种行为，我们将扩散轨迹引导到真实类和对抗类的交叉点，将时间旅行采样与增强分类器引导相结合，以增强攻击的可转移性，同时保持图像保真度。我们的方法实现了与当前最先进技术相当的攻击成功率，同时在模型架构中表现出显着更高的可移植性，并与FID测量的自然测试时间误差更好地对齐。这些结果表明，NatADiff产生的对抗样本不仅可以更有效地在模型之间传递，而且更忠实地类似于自然发生的测试时错误。



## **47. ZeroPur: Succinct Training-Free Adversarial Purification**

ZeroPur：简洁的免培训对抗净化 cs.CV

17 pages, 7 figures, under review

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2406.03143v3) [paper-pdf](http://arxiv.org/pdf/2406.03143v3)

**Authors**: Erhu Liu, Zonglin Yang, Bo Liu, Bin Xiao, Xiuli Bi

**Abstract**: Adversarial purification is a kind of defense technique that can defend against various unseen adversarial attacks without modifying the victim classifier. Existing methods often depend on external generative models or cooperation between auxiliary functions and victim classifiers. However, retraining generative models, auxiliary functions, or victim classifiers relies on the domain of the fine-tuned dataset and is computation-consuming. In this work, we suppose that adversarial images are outliers of the natural image manifold, and the purification process can be considered as returning them to this manifold. Following this assumption, we present a simple adversarial purification method without further training to purify adversarial images, called ZeroPur. ZeroPur contains two steps: given an adversarial example, Guided Shift obtains the shifted embedding of the adversarial example by the guidance of its blurred counterparts; after that, Adaptive Projection constructs a directional vector by this shifted embedding to provide momentum, projecting adversarial images onto the manifold adaptively. ZeroPur is independent of external models and requires no retraining of victim classifiers or auxiliary functions, relying solely on victim classifiers themselves to achieve purification. Extensive experiments on three datasets (CIFAR-10, CIFAR-100, and ImageNet-1K) using various classifier architectures (ResNet, WideResNet) demonstrate that our method achieves state-of-the-art robust performance. The code will be publicly available.

摘要: 对抗净化是一种防御技术，可以在不修改受害者分类器的情况下防御各种看不见的对抗攻击。现有的方法通常依赖于外部生成模型或辅助功能和受害者分类器之间的合作。然而，重新训练生成模型、辅助函数或受害者分类器依赖于微调数据集的域，并且需要计算。在这项工作中，我们假设对抗图像是自然图像多维的离群值，净化过程可以被认为是将它们返回到该多维。遵循这一假设，我们提出了一种简单的对抗净化方法，无需进一步训练来净化对抗图像，称为ZeroPur。ZeroPur包含两个步骤：给定一个对抗性示例，Guided Change通过其模糊对应的引导获得对抗性示例的移动嵌入;之后，自适应投影通过这种移动嵌入来构建一个方向载体，以提供动量，将对抗性图像自适应地投影到多管上。ZeroPur独立于外部模型，不需要重新训练受害者分类器或辅助功能，仅依靠受害者分类器本身来实现净化。在三个数据集（CIFAR-10，CIFAR-100和ImageNet-1 K）上使用各种分类器架构（ResNet，WideResNet）进行的广泛实验表明，我们的方法实现了最先进的鲁棒性能。代码将公开。



## **48. Concealment of Intent: A Game-Theoretic Analysis**

意图的隐瞒：游戏理论分析 cs.CL

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20841v1) [paper-pdf](http://arxiv.org/pdf/2505.20841v1)

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.

摘要: 随着大型语言模型（LLM）的能力越来越强，对其安全部署的担忧也越来越大。尽管已经引入了对齐机制来阻止滥用，但它们仍然容易受到精心设计的对抗提示的影响。在这项工作中，我们提出了一种可扩展的攻击策略：意图隐藏对抗提示，通过技能的组合来隐藏恶意意图。我们开发了一个博弈论框架来建模此类攻击与应用提示和响应过滤的防御系统之间的相互作用。我们的分析确定了平衡点并揭示了攻击者的结构优势。为了应对这些威胁，我们提出并分析了一种针对意图隐藏攻击的防御机制。从经验上讲，我们验证了攻击对一系列恶意行为的多个现实世界LLM的有效性，展示了比现有对抗提示技术的明显优势。



## **49. MedSentry: Understanding and Mitigating Safety Risks in Medical LLM Multi-Agent Systems**

MedSentry：了解和缓解医学LLM多主体系统中的安全风险 cs.MA

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20824v1) [paper-pdf](http://arxiv.org/pdf/2505.20824v1)

**Authors**: Kai Chen, Taihang Zhen, Hewei Wang, Kailai Liu, Xinfeng Li, Jing Huo, Tianpei Yang, Jinfeng Xu, Wei Dong, Yang Gao

**Abstract**: As large language models (LLMs) are increasingly deployed in healthcare, ensuring their safety, particularly within collaborative multi-agent configurations, is paramount. In this paper we introduce MedSentry, a benchmark comprising 5 000 adversarial medical prompts spanning 25 threat categories with 100 subthemes. Coupled with this dataset, we develop an end-to-end attack-defense evaluation pipeline to systematically analyze how four representative multi-agent topologies (Layers, SharedPool, Centralized, and Decentralized) withstand attacks from 'dark-personality' agents. Our findings reveal critical differences in how these architectures handle information contamination and maintain robust decision-making, exposing their underlying vulnerability mechanisms. For instance, SharedPool's open information sharing makes it highly susceptible, whereas Decentralized architectures exhibit greater resilience thanks to inherent redundancy and isolation. To mitigate these risks, we propose a personality-scale detection and correction mechanism that identifies and rehabilitates malicious agents, restoring system safety to near-baseline levels. MedSentry thus furnishes both a rigorous evaluation framework and practical defense strategies that guide the design of safer LLM-based multi-agent systems in medical domains.

摘要: 随着大型语言模型（LLM）越来越多地部署在医疗保健中，确保其安全性，特别是在协作多代理配置中，至关重要。在本文中，我们介绍了MedSentry，这是一个基准，由5000个对抗性医疗提示组成，涵盖25个威胁类别和100个子主题。与此数据集相结合，我们开发了一个端到端的攻击防御评估管道，以系统地分析四种代表性的多智能体布局（Layers、SharedPool、Centralized和Decentralized）如何抵御来自“黑暗人格”智能体的攻击。我们的研究结果揭示了这些架构如何处理信息污染和维持稳健决策的关键差异，暴露了其潜在的脆弱性机制。例如，SharedPool的开放信息共享使其高度容易受到影响，而去中心化架构由于固有的冗余和隔离而表现出更大的弹性。为了减轻这些风险，我们提出了一种个性规模的检测和纠正机制，该机制可以识别和恢复恶意代理，将系统安全性恢复到接近基线的水平。因此，MedSentry提供了严格的评估框架和实用的防御策略，指导医学领域更安全的基于LLM的多智能体系统的设计。



## **50. TrojanStego: Your Language Model Can Secretly Be A Steganographic Privacy Leaking Agent**

TrojanStego：你的语言模型可以秘密地成为隐写隐私泄露代理 cs.CL

9 pages, 5 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20118v2) [paper-pdf](http://arxiv.org/pdf/2505.20118v2)

**Authors**: Dominik Meier, Jan Philip Wahle, Paul Röttger, Terry Ruas, Bela Gipp

**Abstract**: As large language models (LLMs) become integrated into sensitive workflows, concerns grow over their potential to leak confidential information. We propose TrojanStego, a novel threat model in which an adversary fine-tunes an LLM to embed sensitive context information into natural-looking outputs via linguistic steganography, without requiring explicit control over inference inputs. We introduce a taxonomy outlining risk factors for compromised LLMs, and use it to evaluate the risk profile of the threat. To implement TrojanStego, we propose a practical encoding scheme based on vocabulary partitioning learnable by LLMs via fine-tuning. Experimental results show that compromised models reliably transmit 32-bit secrets with 87% accuracy on held-out prompts, reaching over 97% accuracy using majority voting across three generations. Further, they maintain high utility, can evade human detection, and preserve coherence. These results highlight a new class of LLM data exfiltration attacks that are passive, covert, practical, and dangerous.

摘要: 随着大型语言模型（LLM）集成到敏感工作流程中，人们越来越担心它们泄露机密信息的可能性。我们提出了TrojanStego，这是一种新型威胁模型，其中对手微调LLM，通过语言隐写术将敏感的上下文信息嵌入到看起来自然的输出中，而不需要对推理输入进行显式控制。我们引入了一个分类法，概述了受影响的LLM的风险因素，并使用它来评估威胁的风险状况。为了实现TrojanStego，我们提出了一种基于词汇划分的实用编码方案，LLM可以通过微调学习。实验结果表明，受攻击的模型在发出的提示上以87%的准确率可靠地传输32位秘密，使用三代多数投票的准确率达到97%以上。此外，它们保持高实用性，可以逃避人类检测，并保持一致性。这些结果凸显了一类新型LLM数据泄露攻击，这些攻击是被动的、隐蔽的、实用的且危险的。



