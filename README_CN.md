# Latest Adversarial Attack Papers
**update at 2025-06-03 09:50:50**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. SafeGenes: Evaluating the Adversarial Robustness of Genomic Foundation Models**

SafeGenes：评估基因组基础模型的对抗稳健性 cs.CR

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2506.00821v1) [paper-pdf](http://arxiv.org/pdf/2506.00821v1)

**Authors**: Huixin Zhan, Jason H. Moore

**Abstract**: Genomic Foundation Models (GFMs), such as Evolutionary Scale Modeling (ESM), have demonstrated significant success in variant effect prediction. However, their adversarial robustness remains largely unexplored. To address this gap, we propose SafeGenes: a framework for Secure analysis of genomic foundation models, leveraging adversarial attacks to evaluate robustness against both engineered near-identical adversarial Genes and embedding-space manipulations. In this study, we assess the adversarial vulnerabilities of GFMs using two approaches: the Fast Gradient Sign Method (FGSM) and a soft prompt attack. FGSM introduces minimal perturbations to input sequences, while the soft prompt attack optimizes continuous embeddings to manipulate model predictions without modifying the input tokens. By combining these techniques, SafeGenes provides a comprehensive assessment of GFM susceptibility to adversarial manipulation. Targeted soft prompt attacks led to substantial performance degradation, even in large models such as ESM1b and ESM1v. These findings expose critical vulnerabilities in current foundation models, opening new research directions toward improving their security and robustness in high-stakes genomic applications such as variant effect prediction.

摘要: 基因组基础模型（GFM），例如进化规模建模（ESM），在变异效应预测方面取得了巨大成功。然而，它们的对抗稳健性在很大程度上仍未得到探索。为了解决这一差距，我们提出了SafeGenes：一个用于对基因组基础模型进行安全分析的框架，利用对抗性攻击来评估针对工程设计的近乎相同的对抗性基因和嵌入空间操纵的稳健性。在这项研究中，我们使用两种方法评估GFM的对抗漏洞：快速梯度符号法（FGSM）和软提示攻击。FGSM向输入序列引入了最小的扰动，而软提示攻击则优化连续嵌入，以在不修改输入令牌的情况下操纵模型预测。通过结合这些技术，SafeGenes提供了对GFM对对抗操纵的易感性的全面评估。有针对性的软提示攻击导致性能大幅下降，即使在ESM 1b和ESM 1v等大型型号中也是如此。这些发现暴露了当前基础模型中的关键漏洞，为提高其在变异效应预测等高风险基因组应用中的安全性和稳健性开辟了新的研究方向。



## **2. Unlearning Inversion Attacks for Graph Neural Networks**

消除图神经网络的反转攻击 cs.LG

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2506.00808v1) [paper-pdf](http://arxiv.org/pdf/2506.00808v1)

**Authors**: Jiahao Zhang, Yilong Wang, Zhiwei Zhang, Xiaorui Liu, Suhang Wang

**Abstract**: Graph unlearning methods aim to efficiently remove the impact of sensitive data from trained GNNs without full retraining, assuming that deleted information cannot be recovered. In this work, we challenge this assumption by introducing the graph unlearning inversion attack: given only black-box access to an unlearned GNN and partial graph knowledge, can an adversary reconstruct the removed edges? We identify two key challenges: varying probability-similarity thresholds for unlearned versus retained edges, and the difficulty of locating unlearned edge endpoints, and address them with TrendAttack. First, we derive and exploit the confidence pitfall, a theoretical and empirical pattern showing that nodes adjacent to unlearned edges exhibit a large drop in model confidence. Second, we design an adaptive prediction mechanism that applies different similarity thresholds to unlearned and other membership edges. Our framework flexibly integrates existing membership inference techniques and extends them with trend features. Experiments on four real-world datasets demonstrate that TrendAttack significantly outperforms state-of-the-art GNN membership inference baselines, exposing a critical privacy vulnerability in current graph unlearning methods.

摘要: 图去学习方法的目的是在不进行全面重新训练的情况下，有效地消除经过训练的GNN中敏感数据的影响，前提是无法恢复已删除的信息。在这项工作中，我们通过引入图未学习倒置攻击来挑战这一假设：仅在黑匣子访问未学习的GNN和部分图知识的情况下，对手能否重建被删除的边？我们确定了两个关键挑战：未学习边缘与保留边缘的概率相似性阈值不同，以及定位未学习边缘端点的困难，并使用TrendAttack解决这些问题。首先，我们推导并利用置信陷阱，这是一种理论和经验模式，表明邻近未学习边的节点表现出模型置信度大幅下降。其次，我们设计了一种自适应预测机制，将不同的相似性阈值应用于未学习的边缘和其他隶属边缘。我们的框架灵活地集成了现有的成员资格推断技术，并通过趋势特征扩展它们。对四个现实世界数据集的实验表明，TrendAttack的性能显着优于最先进的GNN成员资格推断基线，暴露了当前图取消学习方法中的一个关键隐私漏洞。



## **3. RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments**

RedTeamCUA：混合Web操作系统环境中计算机使用代理的现实对抗测试 cs.CL

**SubmitDate**: 2025-06-01    [abs](http://arxiv.org/abs/2505.21936v2) [paper-pdf](http://arxiv.org/pdf/2505.21936v2)

**Authors**: Zeyi Liao, Jaylen Jones, Linxi Jiang, Eric Fosler-Lussier, Yu Su, Zhiqiang Lin, Huan Sun

**Abstract**: Computer-use agents (CUAs) promise to automate complex tasks across operating systems (OS) and the web, but remain vulnerable to indirect prompt injection. Current evaluations of this threat either lack support realistic but controlled environments or ignore hybrid web-OS attack scenarios involving both interfaces. To address this, we propose RedTeamCUA, an adversarial testing framework featuring a novel hybrid sandbox that integrates a VM-based OS environment with Docker-based web platforms. Our sandbox supports key features tailored for red teaming, such as flexible adversarial scenario configuration, and a setting that decouples adversarial evaluation from navigational limitations of CUAs by initializing tests directly at the point of an adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive benchmark with 864 examples that investigate realistic, hybrid web-OS attack scenarios and fundamental security vulnerabilities. Benchmarking current frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated, still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute adversarial tasks with an Attempt Rate as high as 92.5%, although failing to complete them due to capability limitations. Nevertheless, we observe concerning ASRs of up to 50% in realistic end-to-end settings, with the recently released frontier Claude 4 Opus | CUA showing an alarming ASR of 48%, demonstrating that indirect prompt injection presents tangible risks for even advanced CUAs despite their capabilities and safeguards. Overall, RedTeamCUA provides an essential framework for advancing realistic, controlled, and systematic analysis of CUA vulnerabilities, highlighting the urgent need for robust defenses to indirect prompt injection prior to real-world deployment.

摘要: 计算机使用代理（CUA）承诺在操作系统（OS）和网络上自动化复杂任务，但仍然容易受到间接提示注入的影响。当前对该威胁的评估要么缺乏对现实但受控的环境的支持，要么忽视了涉及两个接口的混合Web操作系统攻击场景。为了解决这个问题，我们提出了RedTeamCUA，这是一种对抗性测试框架，具有新型混合沙盒，该沙盒将基于虚拟机的操作系统环境与基于Docker的Web平台集成在一起。我们的沙箱支持为红色团队定制的关键功能，例如灵活的对抗场景配置，以及通过在对抗注入时直接初始化测试来将对抗评估与CUA的导航限制分开的设置。使用RedTeamCUA，我们开发RTC-Bench，这是一个包含864个示例的综合基准测试，可以调查现实的混合Web操作系统攻击场景和基本安全漏洞。对当前前沿CUA进行基准测试发现重大漏洞：Claude 3.7十四行诗|CUA的ASB为42.9%，而受评估的最安全的CUA Operator仍为7.6%。值得注意的是，CUA经常尝试执行尝试率高达92.5%的对抗任务，尽管由于能力限制而未能完成这些任务。尽管如此，我们观察到，在现实的端到端环境中，最近发布的前沿《Claude 4 Opus》中，ASB高达50%| CUA显示出惊人的48%的ASB，这表明间接即时注射即使是先进的CUA，也会带来切实的风险，尽管它们有能力和保障措施。总体而言，RedTeamCUA为推进对CUA漏洞的现实、受控和系统性分析提供了一个重要框架，凸显了在现实世界部署之前对间接提示注入的强大防御措施的迫切需要。



## **4. Security Concerns for Large Language Models: A Survey**

大型语言模型的安全性问题综述 cs.CR

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2505.18889v2) [paper-pdf](http://arxiv.org/pdf/2505.18889v2)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as GPT-4 and its recent iterations, Google's Gemini, Anthropic's Claude 3 models, and xAI's Grok have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. In this survey, we provide a comprehensive overview of the emerging security concerns around LLMs, categorizing threats into prompt injection and jailbreaking, adversarial attacks such as input perturbations and data poisoning, misuse by malicious actors for purposes such as generating disinformation, phishing emails, and malware, and worrisome risks inherent in autonomous LLM agents. A significant focus has been recently placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives, a behavior known as scheming, which may even persist through safety training. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.

摘要: GPT-4及其最近的迭代、Google的Gemini、Anthropic的Claude 3模型和xAI的Grok等大型语言模型（LLM）引发了自然语言处理领域的一场革命，但它们的功能也引入了新的安全漏洞。在本调查中，我们全面概述了围绕LLM的新安全问题，将威胁分为即时注入和越狱、输入干扰和数据中毒等对抗性攻击、恶意行为者出于生成虚假信息、网络钓鱼电子邮件和恶意软件等目的的滥用以及自主LLM代理固有的令人担忧的风险。最近人们对后者给予了极大的关注，探索目标失调、紧急欺骗、自我保护本能，以及LLM制定和追求隐蔽、失调目标的潜力，这种行为被称为阴谋，甚至可能通过安全培训持续存在。我们总结了2022年至2025年期间最近的学术和工业研究，这些研究揭示了每种威胁，分析了拟议的防御措施及其局限性，并确定了保护基于LLM的应用程序方面的公开挑战。最后，我们强调了推进强大的多层安全策略以确保LLM安全且有益的重要性。



## **5. AdvAgent: Controllable Blackbox Red-teaming on Web Agents**

AdvAgent：基于Web Agent的可控黑盒红组 cs.CR

ICML 2025

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2410.17401v4) [paper-pdf](http://arxiv.org/pdf/2410.17401v4)

**Authors**: Chejian Xu, Mintong Kang, Jiawei Zhang, Zeyi Liao, Lingbo Mo, Mengqi Yuan, Huan Sun, Bo Li

**Abstract**: Foundation model-based agents are increasingly used to automate complex tasks, enhancing efficiency and productivity. However, their access to sensitive resources and autonomous decision-making also introduce significant security risks, where successful attacks could lead to severe consequences. To systematically uncover these vulnerabilities, we propose AdvAgent, a black-box red-teaming framework for attacking web agents. Unlike existing approaches, AdvAgent employs a reinforcement learning-based pipeline to train an adversarial prompter model that optimizes adversarial prompts using feedback from the black-box agent. With careful attack design, these prompts effectively exploit agent weaknesses while maintaining stealthiness and controllability. Extensive evaluations demonstrate that AdvAgent achieves high success rates against state-of-the-art GPT-4-based web agents across diverse web tasks. Furthermore, we find that existing prompt-based defenses provide only limited protection, leaving agents vulnerable to our framework. These findings highlight critical vulnerabilities in current web agents and emphasize the urgent need for stronger defense mechanisms. We release code at https://ai-secure.github.io/AdvAgent/.

摘要: 基于基础模型的代理越来越多地用于自动化复杂任务，提高效率和生产力。然而，他们对敏感资源的访问和自主决策也带来了重大的安全风险，成功的攻击可能会导致严重的后果。为了系统性地发现这些漏洞，我们提出了AdvAgent，这是一个用于攻击Web代理的黑匣子红团队框架。与现有方法不同，AdvAgent采用基于强化学习的管道来训练对抗性提示器模型，该模型使用黑匣子代理的反馈来优化对抗性提示。通过精心的攻击设计，这些提示可以有效地利用代理的弱点，同时保持隐蔽性和可控性。广泛的评估表明，AdvAgent在各种Web任务中针对最先进的基于GPT-4的Web代理取得了很高的成功率。此外，我们发现现有的基于预算的防御只能提供有限的保护，使代理容易受到我们的框架的影响。这些发现凸显了当前网络代理中的关键漏洞，并强调迫切需要更强大的防御机制。我们在https://ai-secure.github.io/AdvAgent/上发布代码。



## **6. Poster: Adapting Pretrained Vision Transformers with LoRA Against Attack Vectors**

海报：使用LoRA对抗攻击载体改编预训练的视觉变形金刚 cs.CV

Presented at IEEE MOST 2025

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00661v1) [paper-pdf](http://arxiv.org/pdf/2506.00661v1)

**Authors**: Richard E. Neddo, Sean Willis, Zander Blasingame, Chen Liu

**Abstract**: Image classifiers, such as those used for autonomous vehicle navigation, are largely known to be susceptible to adversarial attacks that target the input image set. There is extensive discussion on adversarial attacks including perturbations that alter the input images to cause malicious misclassifications without perceivable modification. This work proposes a countermeasure for such attacks by adjusting the weights and classes of pretrained vision transformers with a low-rank adaptation to become more robust against adversarial attacks and allow for scalable fine-tuning without retraining.

摘要: 众所周知，图像分类器（例如用于自主车辆导航的图像分类器）容易受到针对输入图像集的对抗攻击。关于对抗攻击的广泛讨论，包括改变输入图像以在没有可感知修改的情况下导致恶意错误分类的扰动。这项工作提出了一种针对此类攻击的对策，通过低等级适应调整预训练的视觉转换器的权重和类别，使其对对抗性攻击更加稳健，并允许在无需重新训练的情况下进行可扩展的微调。



## **7. Con Instruction: Universal Jailbreaking of Multimodal Large Language Models via Non-Textual Modalities**

Con指令：通过非文本模式对多模式大型语言模型进行普遍越狱 cs.CR

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00548v1) [paper-pdf](http://arxiv.org/pdf/2506.00548v1)

**Authors**: Jiahui Geng, Thy Thy Tran, Preslav Nakov, Iryna Gurevych

**Abstract**: Existing attacks against multimodal language models (MLLMs) primarily communicate instructions through text accompanied by adversarial images. In contrast, we exploit the capabilities of MLLMs to interpret non-textual instructions, specifically, adversarial images or audio generated by our novel method, Con Instruction. We optimize these adversarial examples to align closely with target instructions in the embedding space, revealing the detrimental implications of MLLMs' sophisticated understanding. Unlike prior work, our method does not require training data or preprocessing of textual instructions. While these non-textual adversarial examples can effectively bypass MLLM safety mechanisms, their combination with various text inputs substantially amplifies attack success. We further introduce a new Attack Response Categorization (ARC) framework, which evaluates both the quality of the model's response and its relevance to the malicious instructions. Experimental results demonstrate that Con Instruction effectively bypasses safety mechanisms in multiple vision- and audio-language models, including LLaVA-v1.5, InternVL, Qwen-VL, and Qwen-Audio, evaluated on two standard benchmarks: AdvBench and SafeBench. Specifically, our method achieves the highest attack success rates, reaching 81.3% and 86.6% on LLaVA-v1.5 (13B). On the defense side, we explore various countermeasures against our attacks and uncover a substantial performance gap among existing techniques. Our implementation is made publicly available.

摘要: 针对多模式语言模型（MLLM）的现有攻击主要通过伴随对抗图像的文本来传达指令。相比之下，我们利用MLLM的功能来解释非文本指令，特别是由我们的新颖方法Con Direction生成的对抗图像或音频。我们优化了这些对抗性示例，使其与嵌入空间中的目标指令紧密一致，揭示了MLLM复杂理解的有害影响。与之前的工作不同，我们的方法不需要训练数据或文本指令的预处理。虽然这些非文本对抗性示例可以有效地绕过MLLM安全机制，但它们与各种文本输入的组合大大增强了攻击的成功率。我们进一步引入了一个新的攻击响应分类（ARC）框架，该框架评估模型的响应质量及其与恶意指令的相关性。实验结果表明，Con指令有效地绕过安全机制，在多个视觉和音频语言模型，包括LLaVA-v1.5，InternVL，Qwen-VL和Qwen-Audio，评估两个标准基准：AdvBench和SafeBench。具体来说，我们的方法实现了最高的攻击成功率，在LLaVA-v1.5（13 B）上达到81.3%和86.6%。在防御方面，我们探索了针对攻击的各种对策，并发现现有技术之间存在的巨大性能差距。我们的实现已公开。



## **8. The TIP of the Iceberg: Revealing a Hidden Class of Task-in-Prompt Adversarial Attacks on LLMs**

冰山的提示：揭示对LLM的一类隐藏的即时任务对抗性攻击 cs.CR

Accepted to the Main Track of ACL 2025

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2501.18626v4) [paper-pdf](http://arxiv.org/pdf/2501.18626v4)

**Authors**: Sergey Berezin, Reza Farahbakhsh, Noel Crespi

**Abstract**: We present a novel class of jailbreak adversarial attacks on LLMs, termed Task-in-Prompt (TIP) attacks. Our approach embeds sequence-to-sequence tasks (e.g., cipher decoding, riddles, code execution) into the model's prompt to indirectly generate prohibited inputs. To systematically assess the effectiveness of these attacks, we introduce the PHRYGE benchmark. We demonstrate that our techniques successfully circumvent safeguards in six state-of-the-art language models, including GPT-4o and LLaMA 3.2. Our findings highlight critical weaknesses in current LLM safety alignments and underscore the urgent need for more sophisticated defence strategies.   Warning: this paper contains examples of unethical inquiries used solely for research purposes.

摘要: 我们提出了一类新型的针对LLM的越狱对抗攻击，称为提示任务（TIP）攻击。我们的方法嵌入序列到序列任务（例如，密码解码、谜语、代码执行）到模型的提示中，以间接生成禁止的输入。为了系统性评估这些攻击的有效性，我们引入了PHRYGE基准。我们证明我们的技术成功规避了六种最先进语言模型（包括GPT-4 o和LLaMA 3.2）中的保护措施。我们的研究结果凸显了当前LLM安全调整中的关键弱点，并强调了对更复杂防御策略的迫切需要。   警告：本文包含仅用于研究目的的不道德调查的例子。



## **9. Practical Adversarial Attacks on Stochastic Bandits via Fake Data Injection**

通过虚假数据注入对随机盗贼的实际对抗攻击 cs.LG

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2505.21938v2) [paper-pdf](http://arxiv.org/pdf/2505.21938v2)

**Authors**: Qirun Zeng, Eric He, Richard Hoffmann, Xuchuang Wang, Jinhang Zuo

**Abstract**: Adversarial attacks on stochastic bandits have traditionally relied on some unrealistic assumptions, such as per-round reward manipulation and unbounded perturbations, limiting their relevance to real-world systems. We propose a more practical threat model, Fake Data Injection, which reflects realistic adversarial constraints: the attacker can inject only a limited number of bounded fake feedback samples into the learner's history, simulating legitimate interactions. We design efficient attack strategies under this model, explicitly addressing both magnitude constraints (on reward values) and temporal constraints (on when and how often data can be injected). Our theoretical analysis shows that these attacks can mislead both Upper Confidence Bound (UCB) and Thompson Sampling algorithms into selecting a target arm in nearly all rounds while incurring only sublinear attack cost. Experiments on synthetic and real-world datasets validate the effectiveness of our strategies, revealing significant vulnerabilities in widely used stochastic bandit algorithms under practical adversarial scenarios.

摘要: 对随机强盗的对抗性攻击传统上依赖于一些不切实际的假设，例如每轮奖励操纵和无界扰动，限制了它们与现实世界系统的相关性。我们提出了一个更实用的威胁模型，假数据注入，它反映了现实的对抗性约束：攻击者只能将有限数量的有界假反馈样本注入到学习者的历史中，模拟合法的交互。我们在这个模型下设计了有效的攻击策略，明确地解决了幅度约束（奖励值）和时间约束（何时以及多久可以注入数据）。我们的理论分析表明，这些攻击可能会误导上置信界（UCB）和汤普森抽样算法在几乎所有回合中选择目标手臂，而仅产生次线性攻击成本。对合成和现实世界数据集的实验验证了我们策略的有效性，揭示了广泛使用的随机强盗算法在实际对抗场景下的显着漏洞。



## **10. PADetBench: Towards Benchmarking Physical Attacks against Object Detection**

PADetBench：针对对象检测的物理攻击基准 cs.CV

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2408.09181v3) [paper-pdf](http://arxiv.org/pdf/2408.09181v3)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Lap-Pui Chau, Shaohui Mei

**Abstract**: Physical attacks against object detection have gained increasing attention due to their significant practical implications. However, conducting physical experiments is extremely time-consuming and labor-intensive. Moreover, physical dynamics and cross-domain transformation are challenging to strictly regulate in the real world, leading to unaligned evaluation and comparison, severely hindering the development of physically robust models. To accommodate these challenges, we explore utilizing realistic simulation to thoroughly and rigorously benchmark physical attacks with fairness under controlled physical dynamics and cross-domain transformation. This resolves the problem of capturing identical adversarial images that cannot be achieved in the real world. Our benchmark includes 20 physical attack methods, 48 object detectors, comprehensive physical dynamics, and evaluation metrics. We also provide end-to-end pipelines for dataset generation, detection, evaluation, and further analysis. In addition, we perform 8064 groups of evaluation based on our benchmark, which includes both overall evaluation and further detailed ablation studies for controlled physical dynamics. Through these experiments, we provide in-depth analyses of physical attack performance and physical adversarial robustness, draw valuable observations, and discuss potential directions for future research.   Codebase: https://github.com/JiaweiLian/Benchmarking_Physical_Attack

摘要: 针对对象检测的物理攻击因其重大实际意义而受到越来越多的关注。然而，进行物理实验极其耗时且劳动密集。此外，物理动力学和跨域转换在现实世界中严格监管具有挑战性，导致评估和比较不一致，严重阻碍了物理稳健模型的开发。为了应对这些挑战，我们探索利用现实模拟来在受控的物理动态和跨域转换下以公平性彻底、严格地基准测试物理攻击。这解决了捕捉现实世界中无法实现的相同对抗图像的问题。我们的基准包括20种物理攻击方法、48种对象检测器、全面的物理动力学和评估指标。我们还提供端到端管道，用于数据集生成、检测、评估和进一步分析。此外，我们还根据我们的基准进行了8064组评估，其中包括总体评估和针对受控身体动态的进一步详细消融研究。通过这些实验，我们对物理攻击性能和物理对抗鲁棒性进行了深入分析，得出有价值的观察结果，并讨论未来研究的潜在方向。   代码库：https://github.com/JiaweiLian/Benchmarking_Physical_Attack



## **11. Adversarial Machine Learning for Robust Password Strength Estimation**

用于鲁棒密码强度估计的对抗机器学习 cs.CR

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00373v1) [paper-pdf](http://arxiv.org/pdf/2506.00373v1)

**Authors**: Pappu Jha, Hanzla Hamid, Oluseyi Olukola, Ashim Dahal, Nick Rahimi

**Abstract**: Passwords remain one of the most common methods for securing sensitive data in the digital age. However, weak password choices continue to pose significant risks to data security and privacy. This study aims to solve the problem by focusing on developing robust password strength estimation models using adversarial machine learning, a technique that trains models on intentionally crafted deceptive passwords to expose and address vulnerabilities posed by such passwords. We apply five classification algorithms and use a dataset with more than 670,000 samples of adversarial passwords to train the models. Results demonstrate that adversarial training improves password strength classification accuracy by up to 20% compared to traditional machine learning models. It highlights the importance of integrating adversarial machine learning into security systems to enhance their robustness against modern adaptive threats.   Keywords: adversarial attack, password strength, classification, machine learning

摘要: 密码仍然是数字时代保护敏感数据的最常见方法之一。然而，薄弱的密码选择继续对数据安全和隐私构成重大风险。这项研究旨在通过专注于使用对抗性机器学习开发稳健的密码强度估计模型来解决这个问题，对抗性机器学习是一种根据故意制作的欺骗性密码训练模型的技术，以暴露和解决此类密码带来的漏洞。我们应用五种分类算法，并使用包含超过670，000个对抗密码样本的数据集来训练模型。结果表明，与传统的机器学习模型相比，对抗性训练将密码强度分类准确率提高了20%。它强调了将对抗性机器学习集成到安全系统中以增强其对现代适应性威胁的鲁棒性的重要性。   关键词：对抗攻击，密码强度，分类，机器学习



## **12. Towards Effective and Efficient Adversarial Defense with Diffusion Models for Robust Visual Tracking**

利用扩散模型实现鲁棒视觉跟踪的有效和高效的对抗防御 cs.CV

**SubmitDate**: 2025-05-31    [abs](http://arxiv.org/abs/2506.00325v1) [paper-pdf](http://arxiv.org/pdf/2506.00325v1)

**Authors**: Long Xu, Peng Gao, Wen-Jia Tang, Fei Wang, Ru-Yue Yuan

**Abstract**: Although deep learning-based visual tracking methods have made significant progress, they exhibit vulnerabilities when facing carefully designed adversarial attacks, which can lead to a sharp decline in tracking performance. To address this issue, this paper proposes for the first time a novel adversarial defense method based on denoise diffusion probabilistic models, termed DiffDf, aimed at effectively improving the robustness of existing visual tracking methods against adversarial attacks. DiffDf establishes a multi-scale defense mechanism by combining pixel-level reconstruction loss, semantic consistency loss, and structural similarity loss, effectively suppressing adversarial perturbations through a gradual denoising process. Extensive experimental results on several mainstream datasets show that the DiffDf method demonstrates excellent generalization performance for trackers with different architectures, significantly improving various evaluation metrics while achieving real-time inference speeds of over 30 FPS, showcasing outstanding defense performance and efficiency. Codes are available at https://github.com/pgao-lab/DiffDf.

摘要: 尽管基于深度学习的视觉跟踪方法取得了重大进展，但它们在面对精心设计的对抗攻击时表现出漏洞，这可能导致跟踪性能急剧下降。为了解决这一问题，本文首次提出了一种基于去噪扩散概率模型的新型对抗性防御方法，称为迪夫Df，旨在有效提高现有视觉跟踪方法针对对抗性攻击的鲁棒性。迪夫Df通过结合像素级重建损失、语义一致性损失和结构相似性损失来建立多尺度防御机制，通过逐步去噪过程有效抑制对抗性扰动。在多个主流数据集上的大量实验结果表明，迪夫Df方法对不同架构的跟踪器表现出出色的概括性能，显着改善了各种评估指标，同时实现了超过30 FPS的实时推理速度，展现出出色的防御性能和效率。代码可访问https://github.com/pgao-lab/DiffDf。



## **13. RenderBender: A Survey on Adversarial Attacks Using Differentiable Rendering**

RenderBender：使用差异渲染的对抗性攻击调查 cs.LG

9 pages, 1 figure, 2 tables, IJCAI '25 Survey Track

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2411.09749v2) [paper-pdf](http://arxiv.org/pdf/2411.09749v2)

**Authors**: Matthew Hull, Haoran Wang, Matthew Lau, Alec Helbling, Mansi Phute, Chao Zhang, Zsolt Kira, Willian Lunardi, Martin Andreoni, Wenke Lee, Polo Chau

**Abstract**: Differentiable rendering techniques like Gaussian Splatting and Neural Radiance Fields have become powerful tools for generating high-fidelity models of 3D objects and scenes. Their ability to produce both physically plausible and differentiable models of scenes are key ingredient needed to produce physically plausible adversarial attacks on DNNs. However, the adversarial machine learning community has yet to fully explore these capabilities, partly due to differing attack goals (e.g., misclassification, misdetection) and a wide range of possible scene manipulations used to achieve them (e.g., alter texture, mesh). This survey contributes the first framework that unifies diverse goals and tasks, facilitating easy comparison of existing work, identifying research gaps, and highlighting future directions - ranging from expanding attack goals and tasks to account for new modalities, state-of-the-art models, tools, and pipelines, to underscoring the importance of studying real-world threats in complex scenes.

摘要: 高斯飞溅和神经辐射场等差异渲染技术已成为生成3D对象和场景高保真模型的强大工具。它们产生物理上合理和可区分的场景模型的能力是对DNN产生物理上合理的对抗攻击所需的关键要素。然而，对抗性机器学习社区尚未充分探索这些能力，部分原因是攻击目标不同（例如，错误分类、错误检测）以及用于实现它们的广泛可能的场景操纵（例如，改变纹理、网格）。这项调查提供了第一个统一不同目标和任务的框架，促进了现有工作的简单比较，确定研究差距，并突出未来方向-从扩大攻击目标和任务以考虑新的模式、最先进的模型、工具和管道，到强调研究复杂场景中现实世界威胁的重要性。



## **14. Adversarial Threat Vectors and Risk Mitigation for Retrieval-Augmented Generation Systems**

检索增强生成系统的对抗性威胁向量和风险缓解 cs.CR

SPIE DCS: Proceedings Volume Assurance and Security for AI-enabled  Systems 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2506.00281v1) [paper-pdf](http://arxiv.org/pdf/2506.00281v1)

**Authors**: Chris M. Ward, Josh Harguess

**Abstract**: Retrieval-Augmented Generation (RAG) systems, which integrate Large Language Models (LLMs) with external knowledge sources, are vulnerable to a range of adversarial attack vectors. This paper examines the importance of RAG systems through recent industry adoption trends and identifies the prominent attack vectors for RAG: prompt injection, data poisoning, and adversarial query manipulation. We analyze these threats under risk management lens, and propose robust prioritized control list that includes risk-mitigating actions like input validation, adversarial training, and real-time monitoring.

摘要: 检索增强生成（RAG）系统将大型语言模型（LLM）与外部知识源集成，容易受到一系列对抗攻击载体的攻击。本文通过最近的行业采用趋势来探讨RAG系统的重要性，并确定了RAG的主要攻击载体：提示注入、数据中毒和对抗性查询操纵。我们在风险管理的视角下分析这些威胁，并提出强大的优先级控制列表，其中包括输入验证、对抗性培训和实时监控等风险缓解行动。



## **15. 3D Gaussian Splat Vulnerabilities**

3D高斯分裂漏洞 cs.CR

4 pages, 4 figures, CVPR '25 Workshop on Neural Fields Beyond  Conventional Cameras

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2506.00280v1) [paper-pdf](http://arxiv.org/pdf/2506.00280v1)

**Authors**: Matthew Hull, Haoyang Yang, Pratham Mehta, Mansi Phute, Aeree Cho, Haoran Wang, Matthew Lau, Wenke Lee, Willian T. Lunardi, Martin Andreoni, Polo Chau

**Abstract**: With 3D Gaussian Splatting (3DGS) being increasingly used in safety-critical applications, how can an adversary manipulate the scene to cause harm? We introduce CLOAK, the first attack that leverages view-dependent Gaussian appearances - colors and textures that change with viewing angle - to embed adversarial content visible only from specific viewpoints. We further demonstrate DAGGER, a targeted adversarial attack directly perturbing 3D Gaussians without access to underlying training data, deceiving multi-stage object detectors e.g., Faster R-CNN, through established methods such as projected gradient descent. These attacks highlight underexplored vulnerabilities in 3DGS, introducing a new potential threat to robotic learning for autonomous navigation and other safety-critical 3DGS applications.

摘要: 随着3D高斯飞溅（3DGS）越来越多地用于安全关键应用，对手如何操纵场景造成伤害？我们引入了COAK，这是第一个利用与视角相关的高斯外观（随着视角而变化的颜色和纹理）来嵌入仅从特定视角可见的对抗性内容的攻击。我们进一步展示了DAGER，这是一种有针对性的对抗攻击，直接扰乱3D高斯，而无需访问底层训练数据，欺骗多阶段对象检测器，例如，通过既定方法（例如投影梯度下降）更快的R-CNN。这些攻击凸显了3DGS中未充分开发的漏洞，为自主导航的机器人学习和其他安全关键型3DGS应用带来了新的潜在威胁。



## **16. GSBA$^K$: $top$-$K$ Geometric Score-based Black-box Attack**

GSBA$^K$：$top$-$K$基于几何分数的黑匣子攻击 cs.CV

License changed to CC BY 4.0 to align with ICLR 2025. No changes to  content. Published at: https://openreview.net/forum?id=htX7AoHyln

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2503.12827v3) [paper-pdf](http://arxiv.org/pdf/2503.12827v3)

**Authors**: Md Farhamdur Reza, Richeng Jin, Tianfu Wu, Huaiyu Dai

**Abstract**: Existing score-based adversarial attacks mainly focus on crafting $top$-1 adversarial examples against classifiers with single-label classification. Their attack success rate and query efficiency are often less than satisfactory, particularly under small perturbation requirements; moreover, the vulnerability of classifiers with multi-label learning is yet to be studied. In this paper, we propose a comprehensive surrogate free score-based attack, named \b geometric \b score-based \b black-box \b attack (GSBA$^K$), to craft adversarial examples in an aggressive $top$-$K$ setting for both untargeted and targeted attacks, where the goal is to change the $top$-$K$ predictions of the target classifier. We introduce novel gradient-based methods to find a good initial boundary point to attack. Our iterative method employs novel gradient estimation techniques, particularly effective in $top$-$K$ setting, on the decision boundary to effectively exploit the geometry of the decision boundary. Additionally, GSBA$^K$ can be used to attack against classifiers with $top$-$K$ multi-label learning. Extensive experimental results on ImageNet and PASCAL VOC datasets validate the effectiveness of GSBA$^K$ in crafting $top$-$K$ adversarial examples.

摘要: 现有的基于分数的对抗攻击主要集中在针对具有单标签分类的分类器制作$top$-1对抗示例。它们的攻击成功率和查询效率往往不太令人满意，尤其是在小扰动要求下;此外，具有多标签学习的分类器的脆弱性还有待研究。在本文中，我们提出了一种全面的无代理基于分数的攻击，名为\b几何\b基于分数的\b黑匣子\b攻击（GSBA$^K$），用于在攻击性的$top$-$K$设置中为非目标攻击和目标攻击制作对抗示例，目标是改变目标分类器的$top$-$K$预测。我们引入了新颖的基于梯度的方法来找到一个好的初始边界点来攻击。我们的迭代方法在决策边界上采用新颖的梯度估计技术，在$top$-$K$设置中特别有效，以有效利用决策边界的几何形状。此外，GSBA$^K$可用于攻击具有$top$-$K$多标签学习的分类器。ImageNet和Pascal VOC数据集的大量实验结果验证了GSBA$^K$在制作$top$-$K$对抗性示例方面的有效性。



## **17. Cascading Adversarial Bias from Injection to Distillation in Language Models**

语言模型中从注入到蒸馏的对抗偏差级联 cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24842v1) [paper-pdf](http://arxiv.org/pdf/2505.24842v1)

**Authors**: Harsh Chaudhari, Jamie Hayes, Matthew Jagielski, Ilia Shumailov, Milad Nasr, Alina Oprea

**Abstract**: Model distillation has become essential for creating smaller, deployable language models that retain larger system capabilities. However, widespread deployment raises concerns about resilience to adversarial manipulation. This paper investigates vulnerability of distilled models to adversarial injection of biased content during training. We demonstrate that adversaries can inject subtle biases into teacher models through minimal data poisoning, which propagates to student models and becomes significantly amplified. We propose two propagation modes: Untargeted Propagation, where bias affects multiple tasks, and Targeted Propagation, focusing on specific tasks while maintaining normal behavior elsewhere. With only 25 poisoned samples (0.25% poisoning rate), student models generate biased responses 76.9% of the time in targeted scenarios - higher than 69.4% in teacher models. For untargeted propagation, adversarial bias appears 6x-29x more frequently in student models on unseen tasks. We validate findings across six bias types (targeted advertisements, phishing links, narrative manipulations, insecure coding practices), various distillation methods, and different modalities spanning text and code generation. Our evaluation reveals shortcomings in current defenses - perplexity filtering, bias detection systems, and LLM-based autorater frameworks - against these attacks. Results expose significant security vulnerabilities in distilled models, highlighting need for specialized safeguards. We propose practical design principles for building effective adversarial bias mitigation strategies.

摘要: 模型提炼对于创建更小的、可部署的语言模型以保留更大的系统能力至关重要。然而，广泛部署引发了人们对对抗性操纵弹性的担忧。本文研究了提炼模型在训练期间对偏见内容的对抗性注入的脆弱性。我们证明，对手可以通过最少的数据中毒将微妙的偏见注入教师模型，这些偏见传播到学生模型并被显着放大。我们提出了两种传播模式：非目标传播（偏差影响多个任务）和目标传播（专注于特定任务，同时在其他地方保持正常行为）。由于只有25个中毒样本（中毒率为0.25%），学生模型在目标场景中有76.9%的时间产生偏见反应，高于教师模型的69.4%。对于无针对性传播，在学生模型中，在看不见的任务中，对抗性偏见的出现频率要高出6倍-29倍。我们验证了六种偏见类型（有针对性的广告、网络钓鱼链接、叙事操纵、不安全的编码实践）、各种提炼方法以及跨越文本和代码生成的不同模式的调查结果。我们的评估揭示了当前针对这些攻击的防御措施（困惑过滤、偏差检测系统和基于LLM的自动生成器框架）的缺陷。结果暴露了提炼模型中的重大安全漏洞，凸显了对专门保护措施的需求。我们提出了实用的设计原则来构建有效的对抗偏见缓解策略。



## **18. ByzFL: Research Framework for Robust Federated Learning**

ByzFL：稳健联邦学习的研究框架 cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24802v1) [paper-pdf](http://arxiv.org/pdf/2505.24802v1)

**Authors**: Marc González, Rachid Guerraoui, Rafael Pinot, Geovani Rizk, John Stephan, François Taïani

**Abstract**: We present ByzFL, an open-source Python library for developing and benchmarking robust federated learning (FL) algorithms. ByzFL provides a unified and extensible framework that includes implementations of state-of-the-art robust aggregators, a suite of configurable attacks, and tools for simulating a variety of FL scenarios, including heterogeneous data distributions, multiple training algorithms, and adversarial threat models. The library enables systematic experimentation via a single JSON-based configuration file and includes built-in utilities for result visualization. Compatible with PyTorch tensors and NumPy arrays, ByzFL is designed to facilitate reproducible research and rapid prototyping of robust FL solutions. ByzFL is available at https://byzfl.epfl.ch/, with source code hosted on GitHub: https://github.com/LPD-EPFL/byzfl.

摘要: 我们介绍ByzFL，一个开源Python库，用于开发和基准测试健壮的联邦学习（FL）算法。ByzFL提供了一个统一的可扩展框架，其中包括最先进的强大聚合器的实现，一套可配置的攻击，以及用于模拟各种FL场景的工具，包括异构数据分布，多种训练算法和对抗性威胁模型。该库通过一个基于JSON的配置文件支持系统实验，并包括用于结果可视化的内置实用程序。ByzFL与PyTorch张量和NumPy阵列兼容，旨在促进可重复的研究和强大FL解决方案的快速原型制作。ByzFL可在https：//byzfl.epfl.ch/上获取，源代码托管在GitHub上：https://github.com/LPD-EPFL/byzfl。



## **19. PatchDEMUX: A Certifiably Robust Framework for Multi-label Classifiers Against Adversarial Patches**

PatchDEUX：针对对抗补丁的多标签分类器可认证的稳健框架 cs.CR

CVPR 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24703v1) [paper-pdf](http://arxiv.org/pdf/2505.24703v1)

**Authors**: Dennis Jacob, Chong Xiang, Prateek Mittal

**Abstract**: Deep learning techniques have enabled vast improvements in computer vision technologies. Nevertheless, these models are vulnerable to adversarial patch attacks which catastrophically impair performance. The physically realizable nature of these attacks calls for certifiable defenses, which feature provable guarantees on robustness. While certifiable defenses have been successfully applied to single-label classification, limited work has been done for multi-label classification. In this work, we present PatchDEMUX, a certifiably robust framework for multi-label classifiers against adversarial patches. Our approach is a generalizable method which can extend any existing certifiable defense for single-label classification; this is done by considering the multi-label classification task as a series of isolated binary classification problems to provably guarantee robustness. Furthermore, in the scenario where an attacker is limited to a single patch we propose an additional certification procedure that can provide tighter robustness bounds. Using the current state-of-the-art (SOTA) single-label certifiable defense PatchCleanser as a backbone, we find that PatchDEMUX can achieve non-trivial robustness on the MS-COCO and PASCAL VOC datasets while maintaining high clean performance

摘要: 深度学习技术使计算机视觉技术取得了巨大进步。然而，这些模型很容易受到对抗补丁攻击，从而灾难性地损害性能。这些攻击的物理可实现性质需要可认证的防御，其特征是可证明的鲁棒性保证。虽然可认证防御已成功应用于单标签分类，但针对多标签分类所做的工作有限。在这项工作中，我们介绍了PatchDEUX，这是一个经过认证的稳健框架，用于针对对抗补丁的多标签分类器。我们的方法是一种可推广的方法，可以扩展任何现有的单标签分类可认证防御;这是通过将多标签分类任务视为一系列孤立的二元分类问题来实现的，以可证明地保证稳健性。此外，在攻击者仅限于单个补丁的情况下，我们提出了一个额外的认证程序，可以提供更严格的鲁棒性界限。使用当前最先进的（SOTA）单标签可认证防御PatchCleanser作为主干，我们发现PatchDEUX可以在MS-COCO和帕斯卡VOC数据集上实现非凡的鲁棒性，同时保持高清洁性能



## **20. So, I climbed to the top of the pyramid of pain -- now what?**

所以，我爬到了痛苦金字塔的顶部--现在怎么办？ cs.CR

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24685v1) [paper-pdf](http://arxiv.org/pdf/2505.24685v1)

**Authors**: Vasilis Katos, Emily Rosenorn-Lanng, Jane Henriksen-Bulmer, Ala Yankouskaya

**Abstract**: This paper explores the evolving dynamics of cybersecurity in the age of advanced AI, from the perspective of the introduced Human Layer Kill Chain framework. As traditional attack models like Lockheed Martin's Cyber Kill Chain become inadequate in addressing human vulnerabilities exploited by modern adversaries, the Humal Layer Kill Chain offers a nuanced approach that integrates human psychology and behaviour into the analysis of cyber threats. We detail the eight stages of the Human Layer Kill Chain, illustrating how AI-enabled techniques can enhance psychological manipulation in attacks. By merging the Human Layer with the Cyber Kill Chain, we propose a Sociotechnical Kill Plane that allows for a holistic examination of attackers' tactics, techniques, and procedures (TTPs) across the sociotechnical landscape. This framework not only aids cybersecurity professionals in understanding adversarial methods, but also empowers non-technical personnel to engage in threat identification and response. The implications for incident response and organizational resilience are significant, particularly as AI continues to shape the threat landscape.

摘要: 本文从引入的Human Layer Kill Chain框架的角度探讨了先进人工智能时代网络安全的演变动态。随着洛克希德·马丁公司的网络杀伤链等传统攻击模型不足以解决现代对手利用的人类脆弱性，Humal Layer杀伤链提供了一种细致入微的方法，将人类心理和行为整合到网络威胁的分析中。我们详细介绍了人体层杀伤链的八个阶段，说明了人工智能技术如何增强攻击中的心理操纵。通过将人类层与网络杀戮链合并，我们提出了一个社会技术杀戮平面，可以对整个社会技术领域的攻击者的策略、技术和程序（TTP）进行全面检查。该框架不仅有助于网络安全专业人员了解对抗方法，还使非技术人员能够参与威胁识别和响应。这对事件响应和组织复原力的影响是巨大的，特别是在人工智能继续塑造威胁格局的情况下。



## **21. Black-box Adversarial Attacks on CNN-based SLAM Algorithms**

基于CNN的SLAM算法的黑匣子对抗攻击 cs.RO

9 pages, 8 figures

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24654v1) [paper-pdf](http://arxiv.org/pdf/2505.24654v1)

**Authors**: Maria Rafaela Gkeka, Bowen Sun, Evgenia Smirni, Christos D. Antonopoulos, Spyros Lalis, Nikolaos Bellas

**Abstract**: Continuous advancements in deep learning have led to significant progress in feature detection, resulting in enhanced accuracy in tasks like Simultaneous Localization and Mapping (SLAM). Nevertheless, the vulnerability of deep neural networks to adversarial attacks remains a challenge for their reliable deployment in applications, such as navigation of autonomous agents. Even though CNN-based SLAM algorithms are a growing area of research there is a notable absence of a comprehensive presentation and examination of adversarial attacks targeting CNN-based feature detectors, as part of a SLAM system. Our work introduces black-box adversarial perturbations applied to the RGB images fed into the GCN-SLAM algorithm. Our findings on the TUM dataset [30] reveal that even attacks of moderate scale can lead to tracking failure in as many as 76% of the frames. Moreover, our experiments highlight the catastrophic impact of attacking depth instead of RGB input images on the SLAM system.

摘要: 深度学习的不断进步导致了特征检测的重大进展，从而提高了同步定位和地图绘制（SLAM）等任务的准确性。然而，深度神经网络对对抗性攻击的脆弱性仍然是其在应用中可靠部署的挑战，例如自主代理的导航。尽管基于CNN的SLAM算法是一个不断增长的研究领域，但作为SLAM系统的一部分，针对基于CNN的特征检测器的对抗性攻击的全面介绍和检查明显缺乏。我们的工作介绍了黑盒对抗扰动应用到RGB图像馈入GCN-SLAM算法。我们对TUM数据集[30]的研究结果表明，即使是中等规模的攻击也可能导致多达76%的帧中的跟踪失败。此外，我们的实验强调了攻击深度而不是攻击Ruby输入图像对SLAM系统的灾难性影响。



## **22. A Flat Minima Perspective on Understanding Augmentations and Model Robustness**

理解增强和模型稳健性的平坦极小值观点 cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24592v1) [paper-pdf](http://arxiv.org/pdf/2505.24592v1)

**Authors**: Weebum Yoo, Sung Whan Yoon

**Abstract**: Model robustness indicates a model's capability to generalize well on unforeseen distributional shifts, including data corruption, adversarial attacks, and domain shifts. Data augmentation is one of the prevalent and effective ways to enhance robustness. Despite the great success of augmentations in different fields, a general theoretical understanding of their efficacy in improving model robustness is lacking. We offer a unified theoretical framework to clarify how augmentations can enhance model robustness through the lens of loss surface flatness and PAC generalization bound. Our work diverges from prior studies in that our analysis i) broadly encompasses much of the existing augmentation methods, and ii) is not limited to specific types of distribution shifts like adversarial attacks. We confirm our theories through simulations on the existing common corruption and adversarial robustness benchmarks based on the CIFAR and ImageNet datasets, as well as domain generalization benchmarks including PACS and OfficeHome.

摘要: 模型鲁棒性表明模型能够很好地概括不可预见的分布变化，包括数据损坏，对抗性攻击和域转移。数据增广是增强鲁棒性的有效方法之一。尽管增广在不同领域取得了巨大的成功，但缺乏对其在提高模型鲁棒性方面功效的一般理论理解。我们提供了一个统一的理论框架，以澄清增广如何可以提高模型的鲁棒性，通过镜头的损失表面的平坦性和PAC的推广界。我们的工作与之前的研究不同，因为我们的分析i）广泛涵盖了许多现有的增强方法，并且ii）不限于特定类型的分布变化，例如对抗性攻击。我们通过对基于CIFAR和ImageNet数据集的现有常见腐败和对抗稳健性基准以及包括PAC和DeliverHome在内的领域概括基准进行模拟来证实我们的理论。



## **23. Stress-testing Machine Generated Text Detection: Shifting Language Models Writing Style to Fool Detectors**

压力测试机器生成文本检测：改变语言模型写作风格以愚弄检测器 cs.CL

Accepted at Findings of ACL 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24523v1) [paper-pdf](http://arxiv.org/pdf/2505.24523v1)

**Authors**: Andrea Pedrotti, Michele Papucci, Cristiano Ciaccio, Alessio Miaschi, Giovanni Puccetti, Felice Dell'Orletta, Andrea Esuli

**Abstract**: Recent advancements in Generative AI and Large Language Models (LLMs) have enabled the creation of highly realistic synthetic content, raising concerns about the potential for malicious use, such as misinformation and manipulation. Moreover, detecting Machine-Generated Text (MGT) remains challenging due to the lack of robust benchmarks that assess generalization to real-world scenarios. In this work, we present a pipeline to test the resilience of state-of-the-art MGT detectors (e.g., Mage, Radar, LLM-DetectAIve) to linguistically informed adversarial attacks. To challenge the detectors, we fine-tune language models using Direct Preference Optimization (DPO) to shift the MGT style toward human-written text (HWT). This exploits the detectors' reliance on stylistic clues, making new generations more challenging to detect. Additionally, we analyze the linguistic shifts induced by the alignment and which features are used by detectors to detect MGT texts. Our results show that detectors can be easily fooled with relatively few examples, resulting in a significant drop in detection performance. This highlights the importance of improving detection methods and making them robust to unseen in-domain texts.

摘要: 生成式人工智能和大型语言模型（LLM）的最新进展使得能够创建高度真实的合成内容，这引发了人们对恶意使用可能性的担忧，例如错误信息和操纵。此外，由于缺乏评估对现实世界场景的概括性的稳健基准，检测机器生成文本（MGT）仍然具有挑战性。在这项工作中，我们提出了一个管道来测试最先进的MGT检测器（例如，Mage、Radar、LLM-DetectAIve）到语言知情的对抗性攻击。为了挑战检测器，我们使用直接偏好优化（DPO）微调语言模型，将MGT风格转变为手写文本（HWT）。这利用了探测器对风格线索的依赖，使新一代的探测更具挑战性。此外，我们还分析了对齐引起的语言变化以及检测器使用哪些特征来检测MGT文本。我们的结果表明，检测器很容易被相对较少的例子所愚弄，导致检测性能显着下降。这凸显了改进检测方法并使其对不可见的领域内文本具有鲁棒性的重要性。



## **24. IDEA: An Inverse Domain Expert Adaptation Based Active DNN IP Protection Method**

IDEA：一种基于逆域专家自适应的主动DNN IP保护方法 cs.CR

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2410.00059v2) [paper-pdf](http://arxiv.org/pdf/2410.00059v2)

**Authors**: Chaohui Xu, Qi Cui, Jinxin Dong, Weiyang He, Chip-Hong Chang

**Abstract**: Illegitimate reproduction, distribution and derivation of Deep Neural Network (DNN) models can inflict economic loss, reputation damage and even privacy infringement. Passive DNN intellectual property (IP) protection methods such as watermarking and fingerprinting attempt to prove the ownership upon IP violation, but they are often too late to stop catastrophic damage of IP abuse and too feeble against strong adversaries. In this paper, we propose IDEA, an Inverse Domain Expert Adaptation based proactive DNN IP protection method featuring active authorization and source traceability. IDEA generalizes active authorization as an inverse problem of domain adaptation. The multi-adaptive optimization is solved by a mixture-of-experts model with one real and two fake experts. The real expert re-optimizes the source model to correctly classify test images with a unique model user key steganographically embedded. The fake experts are trained to output random prediction on test images without or with incorrect user key embedded by minimizing their mutual information (MI) with the real expert. The MoE model is knowledge distilled into a unified protected model to avoid leaking the expert model features by maximizing their MI with additional multi-layer attention and contrastive representation loss optimization. IDEA not only prevents unauthorized users without the valid key to access the functional model, but also enable the model owner to validate the deployed model and trace the source of IP infringement. We extensively evaluate IDEA on five datasets and four DNN models to demonstrate its effectiveness in authorization control, culprit tracing success rate, and robustness against various attacks.

摘要: 深度神经网络（DNN）模型的非法复制、分发和派生可能会造成经济损失、声誉损害，甚至侵犯隐私。水印和指纹识别等被动DNN知识产权（IP）保护方法试图在知识产权侵权时证明所有权，但它们往往为时已晚，无法阻止知识产权滥用的灾难性损害，而且对于强大的对手来说也过于软弱。本文提出了IDEA，这是一种基于逆域专家自适应的主动DNN IP保护方法，具有主动授权和源可追溯性。IDEA将主动授权概括为域适应的逆问题。多自适应优化通过具有一个真实专家和两个虚假专家的混合专家模型来解决。真正的专家重新优化源模型，以通过隐写方式嵌入的独特模型用户密钥正确分类测试图像。通过最小化与真实专家的互信息（MI），假专家被训练为在没有嵌入或嵌入不正确用户密钥的情况下在测试图像上输出随机预测。MoE模型是提炼成统一保护模型的知识，通过额外的多层关注和对比表示损失优化来最大化专家模型的MI，以避免泄露专家模型的特征。IDEA不仅可以防止没有有效密钥的未经授权的用户访问功能模型，还可以使模型所有者验证部署的模型并追踪知识产权侵权的来源。我们对五个数据集和四个DNN模型进行了广泛评估，以证明其在授权控制、罪犯追踪成功率以及针对各种攻击的稳健性方面的有效性。



## **25. CAE-Net: Generalized Deepfake Image Detection using Convolution and Attention Mechanisms with Spatial and Frequency Domain Features**

CAE-Net：使用具有空间和频域特征的卷积和注意力机制的广义Deepfake图像检测 cs.CV

Under review in Elsevier Journal of Visual Communication and Image  Representation

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2502.10682v2) [paper-pdf](http://arxiv.org/pdf/2502.10682v2)

**Authors**: Kafi Anan, Anindya Bhattacharjee, Ashir Intesher, Kaidul Islam, Abrar Assaeem Fuad, Utsab Saha, Hafiz Imtiaz

**Abstract**: Effective deepfake detection tools are becoming increasingly essential to the growing usage of deepfakes in unethical practices. There exists a wide range of deepfake generation techniques, which makes it challenging to develop an accurate universal detection mechanism. The 2025 IEEE Signal Processing Cup (\textit{DFWild-Cup} competition) provided a diverse dataset of deepfake images containing significant class imbalance. The images in the dataset are generated from multiple deepfake image generators, for training machine learning model(s) to emphasize the generalization of deepfake detection. To this end, we proposed a disjoint set-based multistage training method to address the class imbalance and devised an ensemble-based architecture \emph{CAE-Net}. Our architecture consists of a convolution- and attention-based ensemble network, and employs three different neural network architectures: EfficientNet, Data-Efficient Image Transformer (DeiT), and ConvNeXt with wavelet transform to capture both local and global features of deepfakes. We visualize the specific regions that these models focus on for classification using Grad-CAM, and empirically demonstrate the effectiveness of these models in grouping real and fake images into cohesive clusters using t-SNE plots. Individually, the EfficientNet B0 architecture has achieved 90.79\% accuracy, whereas the ConvNeXt and the DeiT architecture have achieved 89.49\% and 89.32\% accuracy, respectively. With these networks, our weighted ensemble model achieves an excellent accuracy of 94.63\% on the validation dataset of the SP Cup 2025 competition. The equal error rate of 4.72\% and the Area Under the ROC curve of 97.37\% further confirm the stability of our proposed method. Finally, the robustness of our proposed model against adversarial perturbation attacks is tested as well, showing the inherent defensive properties of the ensemble approach.

摘要: 对于Deepfake在不道德实践中越来越多地使用，有效的Deepfake检测工具变得越来越重要。存在广泛的Deepfake生成技术，这使得开发准确的通用检测机制具有挑战性。2025年IEEE Signal Process Cup（\textit{DFWild-Cup}竞赛）提供了包含显着类别不平衡的Deepfake图像的多样化数据集。数据集中的图像由多个Deepfake图像生成器生成，用于训练机器学习模型，以强调Deepfake检测的通用性。为此，我们提出了一种基于不相交集合的多阶段训练方法来解决类不平衡问题，并设计了一种基于集成的架构\{CAE-Net}。我们的架构由一个基于卷积和注意力的集成网络组成，并采用三种不同的神经网络架构：EfficientNet、数据高效图像Transformer（DeiT）和带子波变换的ConvNeXt来捕获Deepfakes的局部和全局特征。我们使用Grad-CAM可视化这些模型关注的特定区域以进行分类，并通过经验证明这些模型使用t-SNE图将真实和虚假图像分组为内聚集群的有效性。单独而言，EfficientNet B 0架构已达到90.79%的准确性，而ConvNeXt和DeiT架构分别达到89.49%和89.32%的准确性。通过这些网络，我们的加权集成模型在2025年SP Cup比赛的验证数据集上实现了94.63%的出色准确率。等误差率为4.72%，ROC曲线下面积为97.37%，进一步证实了我们提出的方法的稳定性。最后，还测试了我们提出的模型对对抗性扰动攻击的鲁棒性，展示了集成方法的固有防御属性。



## **26. Adversarially Robust AI-Generated Image Detection for Free: An Information Theoretic Perspective**

对抗稳健的人工智能生成图像检测免费：信息论的角度 cs.CV

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.22604v2) [paper-pdf](http://arxiv.org/pdf/2505.22604v2)

**Authors**: Ruixuan Zhang, He Wang, Zhengyu Zhao, Zhiqing Guo, Xun Yang, Yunfeng Diao, Meng Wang

**Abstract**: Rapid advances in Artificial Intelligence Generated Images (AIGI) have facilitated malicious use, such as forgery and misinformation. Therefore, numerous methods have been proposed to detect fake images. Although such detectors have been proven to be universally vulnerable to adversarial attacks, defenses in this field are scarce. In this paper, we first identify that adversarial training (AT), widely regarded as the most effective defense, suffers from performance collapse in AIGI detection. Through an information-theoretic lens, we further attribute the cause of collapse to feature entanglement, which disrupts the preservation of feature-label mutual information. Instead, standard detectors show clear feature separation. Motivated by this difference, we propose Training-free Robust Detection via Information-theoretic Measures (TRIM), the first training-free adversarial defense for AIGI detection. TRIM builds on standard detectors and quantifies feature shifts using prediction entropy and KL divergence. Extensive experiments across multiple datasets and attacks validate the superiority of our TRIM, e.g., outperforming the state-of-the-art defense by 33.88% (28.91%) on ProGAN (GenImage), while well maintaining original accuracy.

摘要: 人工智能生成图像（AIGI）的迅速发展助长了恶意使用，例如伪造和错误信息。因此，人们提出了多种方法来检测虚假图像。尽管此类探测器已被证明普遍容易受到对抗攻击，但该领域的防御措施却很少。在本文中，我们首先确定了对抗训练（AT），被广泛认为是最有效的防御，在AIGI检测中遭受性能崩溃。通过信息论的视角，我们进一步将崩溃的原因归因于特征纠缠，它破坏了特征标签互信息的保存。相反，标准检测器显示出清晰的特征分离。基于这种差异，我们提出了通过信息理论测量（TRIM）的免训练鲁棒检测，这是AIGI检测的第一个免训练对抗性防御。TRIM建立在标准检测器之上，并使用预测熵和KL方差量化特征移动。跨多个数据集和攻击的广泛实验验证了我们TRIM的优越性，例如，ProGAN（GenImage）上的最新防御能力比最先进的防御能力高出33.88%（28.91%），同时很好地保持了原始的准确性。



## **27. SEAR: A Multimodal Dataset for Analyzing AR-LLM-Driven Social Engineering Behaviors**

SEAR：用于分析AR-LLM驱动的社会工程行为的多模式数据集 cs.AI

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24458v1) [paper-pdf](http://arxiv.org/pdf/2505.24458v1)

**Authors**: Tianlong Yu, Chenghang Ye, Zheyu Yang, Ziyi Zhou, Cui Tang, Zui Tao, Jun Zhang, Kailong Wang, Liting Zhou, Yang Yang, Ting Bi

**Abstract**: The SEAR Dataset is a novel multimodal resource designed to study the emerging threat of social engineering (SE) attacks orchestrated through augmented reality (AR) and multimodal large language models (LLMs). This dataset captures 180 annotated conversations across 60 participants in simulated adversarial scenarios, including meetings, classes and networking events. It comprises synchronized AR-captured visual/audio cues (e.g., facial expressions, vocal tones), environmental context, and curated social media profiles, alongside subjective metrics such as trust ratings and susceptibility assessments. Key findings reveal SEAR's alarming efficacy in eliciting compliance (e.g., 93.3% phishing link clicks, 85% call acceptance) and hijacking trust (76.7% post-interaction trust surge). The dataset supports research in detecting AR-driven SE attacks, designing defensive frameworks, and understanding multimodal adversarial manipulation. Rigorous ethical safeguards, including anonymization and IRB compliance, ensure responsible use. The SEAR dataset is available at https://github.com/INSLabCN/SEAR-Dataset.

摘要: SEAR数据集是一种新型多模式资源，旨在研究通过增强现实（AR）和多模式大型语言模型（LLM）精心策划的社会工程（SE）攻击的新兴威胁。该数据集捕获了模拟对抗场景（包括会议、课程和网络活动）中60名参与者的180个带注释的对话。它包括同步的AR捕获的视觉/音频线索（例如，面部表情、语气）、环境背景和精心策划的社交媒体个人资料，以及信任评级和易感性评估等主观指标。关键发现揭示了SEAR在引发合规方面的惊人功效（例如，93.3%的网络钓鱼链接点击，85%的电话接受）和劫持信任（互动后信任激增76.7%）。该数据集支持检测AR驱动的SE攻击、设计防御框架和理解多模式对抗操纵的研究。严格的道德保障措施，包括匿名化和机构审查委员会合规性，确保负责任的使用。SEAR数据集可在https://github.com/INSLabCN/SEAR-Dataset上获取。



## **28. Learning Safety Constraints for Large Language Models**

大型语言模型的学习安全约束 cs.LG

ICML 2025 (Spotlight)

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24445v1) [paper-pdf](http://arxiv.org/pdf/2505.24445v1)

**Authors**: Xin Chen, Yarden As, Andreas Krause

**Abstract**: Large language models (LLMs) have emerged as powerful tools but pose significant safety risks through harmful outputs and vulnerability to adversarial attacks. We propose SaP, short for Safety Polytope, a geometric approach to LLM safety that learns and enforces multiple safety constraints directly in the model's representation space. We develop a framework that identifies safe and unsafe regions via the polytope's facets, enabling both detection and correction of unsafe outputs through geometric steering. Unlike existing approaches that modify model weights, SaP operates post-hoc in the representation space, preserving model capabilities while enforcing safety constraints. Experiments across multiple LLMs demonstrate that our method can effectively detect unethical inputs, reduce adversarial attack success rates while maintaining performance on standard tasks, thus highlighting the importance of having an explicit geometric model for safety. Analysis of the learned polytope facets reveals emergence of specialization in detecting different semantic notions of safety, providing interpretable insights into how safety is captured in LLMs' representation space.

摘要: 大型语言模型（LLM）已成为强大的工具，但由于有害输出和易受对抗攻击而构成重大安全风险。我们提出SaP（Safety Polytope的缩写），这是一种LLM安全性的几何方法，可以直接在模型的表示空间中学习和强制执行多个安全约束。我们开发了一个框架，通过多面体识别安全和不安全区域，从而通过几何转向检测和纠正不安全输出。与修改模型权重的现有方法不同，SaP在表示空间中事后操作，在强制执行安全约束的同时保留模型能力。跨多个LLM的实验表明，我们的方法可以有效地检测不道德的输入，降低对抗攻击成功率，同时保持标准任务的性能，从而凸显了拥有显式几何模型的重要性安全性。对习得的多格面的分析揭示了检测不同安全性语义概念的专业化的出现，从而为如何在LLM的表示空间中捕获安全性提供了可解释的见解。



## **29. Breaking the Gold Standard: Extracting Forgotten Data under Exact Unlearning in Large Language Models**

打破黄金标准：在大型语言模型中提取精确非学习下的遗忘数据 cs.LG

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24379v1) [paper-pdf](http://arxiv.org/pdf/2505.24379v1)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large language models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard, believed to be robust against privacy-related attacks. In this paper, we challenge this assumption by introducing a novel data extraction attack that compromises even exact unlearning. Our method leverages both the pre- and post-unlearning models: by guiding the post-unlearning model using signals from the pre-unlearning model, we uncover patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints.

摘要: 大型语言模型通常在从网络收集的数据集上训练，这些数据集可能无意中包含有害或敏感的个人信息。为了解决日益增长的隐私问题，人们提出了取消学习方法来消除训练模型中特定数据的影响。其中，精确取消学习--在没有目标数据的情况下从头开始重新训练模型--被广泛认为是黄金标准，被认为对隐私相关攻击具有强大的鲁棒性。在本文中，我们通过引入一种新颖的数据提取攻击来挑战这一假设，该攻击甚至会损害精确的取消学习。我们的方法利用了取消学习前和取消学习后的模型：通过使用来自取消学习前模型的信号引导取消学习后模型，我们发现了反映已删除数据分布的模式。将模型指导与令牌过滤策略相结合，我们的攻击显着提高了MUSE、TOFU和WMDP等常见基准测试中的提取成功率，在某些情况下性能翻倍。此外，我们还展示了我们对模拟医疗诊断数据集的攻击的有效性，以强调与精确忘记相关的现实世界隐私风险。鉴于我们的研究结果表明，取消学习可能会以一种矛盾的方式增加隐私泄露的风险，我们主张对取消学习方法进行评估，以考虑更广泛的威胁模型，这些模型不仅考虑取消学习后的模型，还考虑到对之前检查点的对抗访问。



## **30. Adversarial Preference Learning for Robust LLM Alignment**

对抗偏好学习实现稳健的LLM对齐 cs.LG

Accepted at ACL2025 Findings

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24369v1) [paper-pdf](http://arxiv.org/pdf/2505.24369v1)

**Authors**: Yuanfu Wang, Pengyu Wang, Chenyang Xi, Bo Tang, Junyi Zhu, Wenqiang Wei, Chen Chen, Chao Yang, Jingfeng Zhang, Chaochao Lu, Yijun Niu, Keming Mao, Zhiyu Li, Feiyu Xiong, Jie Hu, Mingchuan Yang

**Abstract**: Modern language models often rely on Reinforcement Learning from Human Feedback (RLHF) to encourage safe behaviors. However, they remain vulnerable to adversarial attacks due to three key limitations: (1) the inefficiency and high cost of human annotation, (2) the vast diversity of potential adversarial attacks, and (3) the risk of feedback bias and reward hacking. To address these challenges, we introduce Adversarial Preference Learning (APL), an iterative adversarial training method incorporating three key innovations. First, a direct harmfulness metric based on the model's intrinsic preference probabilities, eliminating reliance on external assessment. Second, a conditional generative attacker that synthesizes input-specific adversarial variations. Third, an iterative framework with automated closed-loop feedback, enabling continuous adaptation through vulnerability discovery and mitigation. Experiments on Mistral-7B-Instruct-v0.3 demonstrate that APL significantly enhances robustness, achieving 83.33% harmlessness win rate over the base model (evaluated by GPT-4o), reducing harmful outputs from 5.88% to 0.43% (measured by LLaMA-Guard), and lowering attack success rate by up to 65% according to HarmBench. Notably, APL maintains competitive utility, with an MT-Bench score of 6.59 (comparable to the baseline 6.78) and an LC-WinRate of 46.52% against the base model.

摘要: 现代语言模型通常依赖于人类反馈强化学习（RL HF）来鼓励安全行为。然而，由于三个关键限制，它们仍然容易受到对抗攻击：（1）人类注释的效率低下和成本高，（2）潜在对抗攻击的多样性，（3）反馈偏差和奖励黑客攻击的风险。为了应对这些挑战，我们引入了对抗偏好学习（APL），这是一种融合了三项关键创新的迭代对抗训练方法。首先，基于模型内在偏好概率的直接危害性指标，消除了对外部评估的依赖。第二，合成特定于输入的对抗变体的条件生成攻击者。第三，具有自动闭环反馈的迭代框架，通过漏洞发现和缓解实现持续适应。Mistral-7 B-Direct-v0.3的实验表明，APL显着增强了鲁棒性，比基本模型实现了83.33%的无害获胜率（通过GPT-4 o评估），将有害输出从5.88%降低到0.43%（通过LLaMA-Guard测量），并将攻击成功率降低高达65%。值得注意的是，APL保持了有竞争力的实用性，MT-Bench评分为6.59（与基线6.78相当），LC-WinRecord为46.52%。



## **31. Rewrite to Jailbreak: Discover Learnable and Transferable Implicit Harmfulness Instruction**

重写越狱：发现可学习和可转移的隐性有害指令 cs.CL

22 pages, 10 figures, accepted to ACL 2025 findings

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2502.11084v2) [paper-pdf](http://arxiv.org/pdf/2502.11084v2)

**Authors**: Yuting Huang, Chengyuan Liu, Yifeng Feng, Yiquan Wu, Chao Wu, Fei Wu, Kun Kuang

**Abstract**: As Large Language Models (LLMs) are widely applied in various domains, the safety of LLMs is increasingly attracting attention to avoid their powerful capabilities being misused. Existing jailbreak methods create a forced instruction-following scenario, or search adversarial prompts with prefix or suffix tokens to achieve a specific representation manually or automatically. However, they suffer from low efficiency and explicit jailbreak patterns, far from the real deployment of mass attacks to LLMs. In this paper, we point out that simply rewriting the original instruction can achieve a jailbreak, and we find that this rewriting approach is learnable and transferable. We propose the Rewrite to Jailbreak (R2J) approach, a transferable black-box jailbreak method to attack LLMs by iteratively exploring the weakness of the LLMs and automatically improving the attacking strategy. The jailbreak is more efficient and hard to identify since no additional features are introduced. Extensive experiments and analysis demonstrate the effectiveness of R2J, and we find that the jailbreak is also transferable to multiple datasets and various types of models with only a few queries. We hope our work motivates further investigation of LLM safety. The code can be found at https://github.com/ythuang02/R2J/.

摘要: 随着大型语言模型（LLM）在各个领域的广泛应用，LLM的安全性越来越受到关注，以避免其强大的功能被滥用。现有的越狱方法创建强制描述跟随场景，或搜索具有前置或后缀标记的对抗提示，以手动或自动实现特定的表示。然而，它们的效率低和越狱模式明显，远未真正对LLM进行大规模攻击。在本文中，我们指出，简单地重写原始指令就可以实现越狱，并且我们发现这种重写方法是可学习和可移植的。我们提出了重写越狱（R2 J）方法，这是一种可转移的黑匣子越狱方法，通过迭代探索LLM的弱点并自动改进攻击策略来攻击LLM。由于没有引入额外的功能，越狱更有效且难以识别。大量的实验和分析证明了R2J的有效性，我们发现越狱也可以转移到多个数据集和各种类型的模型，只需几个查询。我们希望我们的工作能够推动对LLM安全性的进一步调查。该代码可在https://github.com/ythuang02/R2J/上找到。



## **32. Multi-Domain Graph Foundation Models: Robust Knowledge Transfer via Topology Alignment**

多领域图基础模型：通过布局对齐实现稳健的知识转移 cs.SI

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2502.02017v2) [paper-pdf](http://arxiv.org/pdf/2502.02017v2)

**Authors**: Shuo Wang, Bokui Wang, Zhixiang Shen, Boyan Deng, Zhao Kang

**Abstract**: Recent advances in CV and NLP have inspired researchers to develop general-purpose graph foundation models through pre-training across diverse domains. However, a fundamental challenge arises from the substantial differences in graph topologies across domains. Additionally, real-world graphs are often sparse and prone to noisy connections and adversarial attacks. To address these issues, we propose the Multi-Domain Graph Foundation Model (MDGFM), a unified framework that aligns and leverages cross-domain topological information to facilitate robust knowledge transfer. MDGFM bridges different domains by adaptively balancing features and topology while refining original graphs to eliminate noise and align topological structures. To further enhance knowledge transfer, we introduce an efficient prompt-tuning approach. By aligning topologies, MDGFM not only improves multi-domain pre-training but also enables robust knowledge transfer to unseen domains. Theoretical analyses provide guarantees of MDGFM's effectiveness and domain generalization capabilities. Extensive experiments on both homophilic and heterophilic graph datasets validate the robustness and efficacy of our method.

摘要: CV和NLP的最新进展激励研究人员通过跨不同领域的预训练来开发通用图基础模型。然而，一个根本性的挑战来自于各个领域的图布局的巨大差异。此外，现实世界的图表通常很稀疏，并且容易出现有噪音的连接和对抗性攻击。为了解决这些问题，我们提出了多域图基础模型（MDGFM），这是一个统一框架，可以对齐和利用跨域拓扑信息以促进稳健的知识转移。MDGFM通过自适应地平衡特征和拓扑，同时细化原始图形以消除噪音并对齐拓扑结构来桥梁不同的领域。为了进一步加强知识转移，我们引入了一种高效的预算调整方法。通过对齐布局，MDGFM不仅改善了多域预训练，而且还实现了向不可见域的稳健知识传输。理论分析为MDGFM的有效性和领域概括能力提供了保证。对同嗜图和异嗜图数据集的广泛实验验证了我们方法的稳健性和有效性。



## **33. When Are Concepts Erased From Diffusion Models?**

概念何时从扩散模型中删除？ cs.LG

Project Page:  https://nyu-dice-lab.github.io/when-are-concepts-erased/

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.17013v4) [paper-pdf](http://arxiv.org/pdf/2505.17013v4)

**Authors**: Kevin Lu, Nicky Kriplani, Rohit Gandikota, Minh Pham, David Bau, Chinmay Hegde, Niv Cohen

**Abstract**: Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.

摘要: 概念擦除，即选择性地阻止模型生成特定概念的能力，引起了越来越多的兴趣，各种方法出现了来应对这一挑战。然而，目前尚不清楚这些方法如何彻底消除目标概念。我们首先提出了扩散模型中擦除机制的两个概念模型：（i）降低生成目标概念的可能性，（ii）干扰模型的内部引导机制。为了彻底评估某个概念是否已真正从模型中删除，我们引入了一套独立评估。我们的评估框架包括对抗性攻击、新颖的探测技术以及对模型替代世代的分析，以取代被删除的概念。我们的结果揭示了最大限度地减少副作用和保持对抗提示的鲁棒性之间的紧张关系。从广义上讲，我们的工作强调了对扩散模型中擦除进行全面评估的重要性。



## **34. Safety Alignment Can Be Not Superficial With Explicit Safety Signals**

明确的安全信号，安全调整不能肤浅 cs.CR

ICML 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.17072v2) [paper-pdf](http://arxiv.org/pdf/2505.17072v2)

**Authors**: Jianwei Li, Jung-Eun Kim

**Abstract**: Recent studies on the safety alignment of large language models (LLMs) have revealed that existing approaches often operate superficially, leaving models vulnerable to various adversarial attacks. Despite their significance, these studies generally fail to offer actionable solutions beyond data augmentation for achieving more robust safety mechanisms. This paper identifies a fundamental cause of this superficiality: existing alignment approaches often presume that models can implicitly learn a safety-related reasoning task during the alignment process, enabling them to refuse harmful requests. However, the learned safety signals are often diluted by other competing objectives, leading models to struggle with drawing a firm safety-conscious decision boundary when confronted with adversarial attacks. Based on this observation, by explicitly introducing a safety-related binary classification task and integrating its signals with our attention and decoding strategies, we eliminate this ambiguity and allow models to respond more responsibly to malicious queries. We emphasize that, with less than 0.2x overhead cost, our approach enables LLMs to assess the safety of both the query and the previously generated tokens at each necessary generating step. Extensive experiments demonstrate that our method significantly improves the resilience of LLMs against various adversarial attacks, offering a promising pathway toward more robust generative AI systems.

摘要: 最近关于大型语言模型（LLM）安全对齐的研究表明，现有方法通常是肤浅的，使模型容易受到各种对抗攻击。尽管这些研究意义重大，但通常无法提供数据增强之外的可操作解决方案来实现更强大的安全机制。本文指出了这种肤浅的根本原因：现有的对齐方法通常假设模型可以在对齐过程中隐式学习与安全相关的推理任务，使它们能够拒绝有害的请求。然而，学习到的安全信号通常会被其他竞争目标稀释，导致模型在面临对抗攻击时难以划定坚定的安全意识决策边界。基于这一观察，通过明确引入与安全相关的二进制分类任务，并将其信号与我们的注意力和解码策略集成，我们消除了这种模糊性，并允许模型更负责任地响应恶意查询。我们强调，由于管理成本低于0.2倍，我们的方法使LLM能够在每个必要的生成步骤中评估查询和之前生成的令牌的安全性。大量实验表明，我们的方法显着提高了LLM对各种对抗攻击的弹性，为实现更强大的生成性人工智能系统提供了一条有希望的途径。



## **35. Light as Deception: GPT-driven Natural Relighting Against Vision-Language Pre-training Models**

轻如骗局：GPT驱动的自然重燃对抗视觉语言预训练模型 cs.CV

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24227v1) [paper-pdf](http://arxiv.org/pdf/2505.24227v1)

**Authors**: Ying Yang, Jie Zhang, Xiao Lv, Di Lin, Tao Xiang, Qing Guo

**Abstract**: While adversarial attacks on vision-and-language pretraining (VLP) models have been explored, generating natural adversarial samples crafted through realistic and semantically meaningful perturbations remains an open challenge. Existing methods, primarily designed for classification tasks, struggle when adapted to VLP models due to their restricted optimization spaces, leading to ineffective attacks or unnatural artifacts. To address this, we propose \textbf{LightD}, a novel framework that generates natural adversarial samples for VLP models via semantically guided relighting. Specifically, LightD leverages ChatGPT to propose context-aware initial lighting parameters and integrates a pretrained relighting model (IC-light) to enable diverse lighting adjustments. LightD expands the optimization space while ensuring perturbations align with scene semantics. Additionally, gradient-based optimization is applied to the reference lighting image to further enhance attack effectiveness while maintaining visual naturalness. The effectiveness and superiority of the proposed LightD have been demonstrated across various VLP models in tasks such as image captioning and visual question answering.

摘要: 虽然已经探索了对视觉和语言预训练（VLP）模型的对抗性攻击，但通过现实且具有语义意义的扰动生成自然对抗性样本仍然是一个悬而未决的挑战。现有方法主要为分类任务设计，由于优化空间有限，在适应VLP模型时会遇到困难，从而导致无效攻击或不自然的伪影。为了解决这个问题，我们提出了\textBF{LightD}，这是一个新颖的框架，通过语义引导的重新照明为VLP模型生成自然对抗样本。具体而言，LightD利用ChatGPT提出上下文感知的初始照明参数，并集成预先训练的重新照明模型（IC-light）以实现多样化的照明调整。LightD扩展了优化空间，同时确保扰动与场景语义保持一致。此外，对参考照明图像应用基于梯度的优化，以进一步增强攻击效果，同时保持视觉自然性。所提出的LightD的有效性和优越性已在各种VLP模型中得到证实，例如图像字幕和视觉问答。



## **36. On the Vulnerability of Applying Retrieval-Augmented Generation within Knowledge-Intensive Application Domains**

检索增强生成在知识密集型应用领域中的脆弱性研究 cs.CR

Accepted by ICML 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2409.17275v2) [paper-pdf](http://arxiv.org/pdf/2409.17275v2)

**Authors**: Xun Xian, Ganghua Wang, Xuan Bi, Jayanth Srinivasa, Ashish Kundu, Charles Fleming, Mingyi Hong, Jie Ding

**Abstract**: Retrieval-Augmented Generation (RAG) has been empirically shown to enhance the performance of large language models (LLMs) in knowledge-intensive domains such as healthcare, finance, and legal contexts. Given a query, RAG retrieves relevant documents from a corpus and integrates them into the LLMs' generation process. In this study, we investigate the adversarial robustness of RAG, focusing specifically on examining the retrieval system. First, across 225 different setup combinations of corpus, retriever, query, and targeted information, we show that retrieval systems are vulnerable to universal poisoning attacks in medical Q\&A. In such attacks, adversaries generate poisoned documents containing a broad spectrum of targeted information, such as personally identifiable information. When these poisoned documents are inserted into a corpus, they can be accurately retrieved by any users, as long as attacker-specified queries are used. To understand this vulnerability, we discovered that the deviation from the query's embedding to that of the poisoned document tends to follow a pattern in which the high similarity between the poisoned document and the query is retained, thereby enabling precise retrieval. Based on these findings, we develop a new detection-based defense to ensure the safe use of RAG. Through extensive experiments spanning various Q\&A domains, we observed that our proposed method consistently achieves excellent detection rates in nearly all cases.

摘要: 经验证明，检索增强生成（RAG）可以增强大型语言模型（LLM）在医疗保健、金融和法律上下文等知识密集型领域的性能。给定一个查询，RAG从数据库中检索相关文档，并将它们集成到LLM的生成过程中。在这项研究中，我们研究了RAG的对抗鲁棒性，特别关注检查检索系统。首先，在225种不同的数据库、检索器、查询和目标信息的设置组合中，我们表明检索系统容易受到医学问答中的普遍中毒攻击。在此类攻击中，对手会生成包含广泛目标信息（例如个人可识别信息）的有毒文档。当这些有毒文档被插入到数据库中时，只要使用攻击者指定的查询，任何用户都可以准确地检索它们。为了了解这个漏洞，我们发现查询嵌入与中毒文档嵌入的偏差往往遵循这样一种模式，即中毒文档和查询之间的高度相似性被保留，从而实现精确的检索。基于这些发现，我们开发了一种新的基于检测的防御措施，以确保RAG的安全使用。通过跨越各个问答领域的广泛实验，我们观察到我们提出的方法在几乎所有情况下都能始终实现出色的检测率。



## **37. GNNBleed: Inference Attacks to Unveil Private Edges in Graphs with Realistic Access to GNN Models**

GNNBleed：通过真实访问GNN模型来揭示图形中的私有边的推理攻击 cs.CR

The paper has been accepted to the PoPETs 2025

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2311.16139v2) [paper-pdf](http://arxiv.org/pdf/2311.16139v2)

**Authors**: Zeyu Song, Ehsanul Kabir, Shagufta Mehnaz

**Abstract**: Graph Neural Networks (GNNs) have become indispensable tools for learning from graph structured data, catering to various applications such as social network analysis and fraud detection for financial services. At the heart of these networks are the edges, which are crucial in guiding GNN models' predictions. In many scenarios, these edges represent sensitive information, such as personal associations or financial dealings, which require privacy assurance. However, their contributions to GNN model predictions may, in turn, be exploited by the adversary to compromise their privacy. Motivated by these conflicting requirements, this paper investigates edge privacy in contexts where adversaries possess only black-box access to the target GNN model, restricted further by access controls, preventing direct insights into arbitrary node outputs. Moreover, we are the first to extensively examine situations where the target graph continuously evolves, a common trait of many real-world graphs. In this setting, we present a range of attacks that leverage the message-passing mechanism of GNNs. We evaluated the effectiveness of our attacks using nine real-world datasets, encompassing both static and dynamic graphs, across four different GNN architectures. The results demonstrate that our attack outperforms existing methods across various GNN architectures, consistently achieving an F1 score of at least 0.8 in static scenarios. Furthermore, our attack retains robustness in dynamic graph scenarios, maintaining F1 scores up to 0.8, unlike previous methods that only achieve F1 scores around 0.2.

摘要: 图神经网络（GNN）已成为从图结构数据中学习的不可或缺的工具，可满足各种应用，如社交网络分析和金融服务欺诈检测。这些网络的核心是边缘，这对于指导GNN模型的预测至关重要。在许多情况下，这些边表示敏感信息，例如个人关联或金融交易，这些信息需要隐私保证。然而，他们对GNN模型预测的贡献可能反过来被对手利用来损害他们的隐私。受这些相互冲突的要求的激励，本文研究了对手仅拥有对目标GNN模型的黑匣子访问权限的上下文中的边缘隐私，并受到访问控制的进一步限制，从而阻止了对任意节点输出的直接洞察。此外，我们是第一个广泛研究目标图不断演变的情况的人，这是许多现实世界图的共同特征。在这种情况下，我们提出了一系列利用GNN消息传递机制的攻击。我们使用跨越四种不同GNN架构的九个现实世界数据集（涵盖静态和动态图表）评估了攻击的有效性。结果表明，我们的攻击优于各种GNN架构中的现有方法，在静态场景中始终实现至少0.8的F1评分。此外，我们的攻击在动态图场景中保留了鲁棒性，将F1分数保持在0.8左右，而与之前仅实现F1分数0.2左右的方法不同。



## **38. Latent-space adversarial training with post-aware calibration for defending large language models against jailbreak attacks**

具有事后感知校准的潜在空间对抗训练，用于保护大型语言模型免受越狱攻击 cs.CR

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2501.10639v3) [paper-pdf](http://arxiv.org/pdf/2501.10639v3)

**Authors**: Xin Yi, Yue Li, Dongsheng Shi, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Ensuring safety alignment is a critical requirement for large language models (LLMs), particularly given increasing deployment in real-world applications. Despite considerable advancements, LLMs remain susceptible to jailbreak attacks, which exploit system vulnerabilities to circumvent safety measures and elicit harmful or inappropriate outputs. Furthermore, while adversarial training-based defense methods have shown promise, a prevalent issue is the unintended over-defense behavior, wherein models excessively reject benign queries, significantly undermining their practical utility. To address these limitations, we introduce LATPC, a Latent-space Adversarial Training with Post-aware Calibration framework. LATPC dynamically identifies safety-critical latent dimensions by contrasting harmful and benign inputs, enabling the adaptive construction of targeted refusal feature removal attacks. This mechanism allows adversarial training to concentrate on real-world jailbreak tactics that disguise harmful queries as benign ones. During inference, LATPC employs an efficient embedding-level calibration mechanism to minimize over-defense behaviors with negligible computational overhead. Experimental results across five types of disguise-based jailbreak attacks demonstrate that LATPC achieves a superior balance between safety and utility compared to existing defense frameworks. Further analysis demonstrates the effectiveness of leveraging safety-critical dimensions in developing robust defense methods against jailbreak attacks.

摘要: 确保安全一致是大型语言模型（LLM）的关键要求，特别是考虑到现实世界应用程序中的部署不断增加。尽管LLM取得了相当大的进步，但仍然容易受到越狱攻击，这些攻击利用系统漏洞来规避安全措施并引发有害或不当的输出。此外，虽然基于对抗训练的防御方法已经显示出希望，但一个普遍的问题是无意的过度防御行为，其中模型过度拒绝良性查询，从而显着削弱了其实际实用性。为了解决这些限制，我们引入了LAPC，这是一种具有事后感知校准框架的潜在空间对抗训练。LAPC通过对比有害和良性输入来动态识别对安全至关重要的潜在维度，从而能够自适应地构建有针对性的拒绝功能删除攻击。这种机制允许对抗训练集中在现实世界的越狱策略上，这些策略将有害查询伪装成良性查询。在推理过程中，LAPC采用高效的嵌入级校准机制，以可忽略的计算负担最小化过度防御行为。五种基于伪装的越狱攻击的实验结果表明，与现有防御框架相比，LAPC在安全性和实用性之间实现了更好的平衡。进一步的分析证明了利用安全关键维度来开发针对越狱攻击的强大防御方法的有效性。



## **39. JailBound: Jailbreaking Internal Safety Boundaries of Vision-Language Models**

越狱：打破视觉语言模型的内部安全边界 cs.CV

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.19610v2) [paper-pdf](http://arxiv.org/pdf/2505.19610v2)

**Authors**: Jiaxin Song, Yixu Wang, Jie Li, Rui Yu, Yan Teng, Xingjun Ma, Yingchun Wang

**Abstract**: Vision-Language Models (VLMs) exhibit impressive performance, yet the integration of powerful vision encoders has significantly broadened their attack surface, rendering them increasingly susceptible to jailbreak attacks. However, lacking well-defined attack objectives, existing jailbreak methods often struggle with gradient-based strategies prone to local optima and lacking precise directional guidance, and typically decouple visual and textual modalities, thereby limiting their effectiveness by neglecting crucial cross-modal interactions. Inspired by the Eliciting Latent Knowledge (ELK) framework, we posit that VLMs encode safety-relevant information within their internal fusion-layer representations, revealing an implicit safety decision boundary in the latent space. This motivates exploiting boundary to steer model behavior. Accordingly, we propose JailBound, a novel latent space jailbreak framework comprising two stages: (1) Safety Boundary Probing, which addresses the guidance issue by approximating decision boundary within fusion layer's latent space, thereby identifying optimal perturbation directions towards the target region; and (2) Safety Boundary Crossing, which overcomes the limitations of decoupled approaches by jointly optimizing adversarial perturbations across both image and text inputs. This latter stage employs an innovative mechanism to steer the model's internal state towards policy-violating outputs while maintaining cross-modal semantic consistency. Extensive experiments on six diverse VLMs demonstrate JailBound's efficacy, achieves 94.32% white-box and 67.28% black-box attack success averagely, which are 6.17% and 21.13% higher than SOTA methods, respectively. Our findings expose a overlooked safety risk in VLMs and highlight the urgent need for more robust defenses. Warning: This paper contains potentially sensitive, harmful and offensive content.

摘要: 视觉语言模型（VLM）表现出令人印象深刻的性能，但强大的视觉编码器的集成显着拓宽了它们的攻击面，使它们越来越容易受到越狱攻击。然而，由于缺乏明确定义的攻击目标，现有的越狱方法常常难以应对基于梯度的策略，这些策略容易出现局部最优情况，并且缺乏精确的方向指导，并且通常会使视觉和文本模式脱钩，从而通过忽视关键的跨模式交互来限制其有效性。受启发潜在知识（ELK）框架的启发，我们推测VLM在其内部融合层表示中编码安全相关信息，揭示了潜在空间中的隐性安全决策边界。这激励了利用边界来引导模型行为。因此，我们提出了JailBound，这是一种新型的潜在空间越狱框架，包括两个阶段：（1）安全边界探测，通过逼近融合层潜在空间内的决策边界来解决引导问题，从而识别朝向目标区域的最佳扰动方向;及（2）安全过境，它通过联合优化图像和文本输入的对抗性扰动来克服脱钩方法的局限性。后一阶段采用创新机制来引导模型的内部状态转向违反政策的输出，同时保持跨模式语义一致性。在六种不同的VLM上进行了大量实验，证明了JailBound的有效性，平均白盒攻击成功率为94.32%，黑盒攻击成功率为67.28%，分别比SOTA方法高6.17%和21.13%。我们的研究结果揭示了VLM中被忽视的安全风险，并强调了对更强大防御的迫切需要。警告：本文包含潜在敏感、有害和冒犯性内容。



## **40. The Butterfly Effect in Pathology: Exploring Security in Pathology Foundation Models**

病理学中的蝴蝶效应：探索病理学基础模型的安全性 cs.CV

**SubmitDate**: 2025-05-30    [abs](http://arxiv.org/abs/2505.24141v1) [paper-pdf](http://arxiv.org/pdf/2505.24141v1)

**Authors**: Jiashuai Liu, Yingjia Shang, Yingkang Zhan, Di Zhang, Yi Niu, Dong Wei, Xian Wu, Zeyu Gao, Chen Li, Yefeng Zheng

**Abstract**: With the widespread adoption of pathology foundation models in both research and clinical decision support systems, exploring their security has become a critical concern. However, despite their growing impact, the vulnerability of these models to adversarial attacks remains largely unexplored. In this work, we present the first systematic investigation into the security of pathology foundation models for whole slide image~(WSI) analysis against adversarial attacks. Specifically, we introduce the principle of \textit{local perturbation with global impact} and propose a label-free attack framework that operates without requiring access to downstream task labels. Under this attack framework, we revise four classical white-box attack methods and redefine the perturbation budget based on the characteristics of WSI. We conduct comprehensive experiments on three representative pathology foundation models across five datasets and six downstream tasks. Despite modifying only 0.1\% of patches per slide with imperceptible noise, our attack leads to downstream accuracy degradation that can reach up to 20\% in the worst cases. Furthermore, we analyze key factors that influence attack success, explore the relationship between patch-level vulnerability and semantic content, and conduct a preliminary investigation into potential defence strategies. These findings lay the groundwork for future research on the adversarial robustness and reliable deployment of pathology foundation models. Our code is publicly available at: https://github.com/Jiashuai-Liu-hmos/Attack-WSI-pathology-foundation-models.

摘要: 随着病理基础模型在研究和临床决策支持系统中的广泛采用，探索其安全性已成为一个关键问题。然而，尽管它们的影响越来越大，但这些模型对对抗性攻击的脆弱性在很大程度上仍未得到探索。在这项工作中，我们提出了第一个系统的研究，对整个幻灯片图像~（WSI）分析的病理基础模型的安全性对抗攻击。具体来说，我们引入了\textit{局部扰动与全局影响}的原则，并提出了一个无标签攻击框架，无需访问下游任务标签。在此攻击框架下，我们修改了四种经典的白盒攻击方法，并根据WSI的特点重新定义了扰动预算。我们在五个数据集和六个下游任务中对三个代表性的病理学基础模型进行了全面的实验。尽管每个幻灯片仅修改0.1%的补丁，但我们的攻击导致下游准确性下降，在最坏的情况下可以达到20%。此外，我们分析了影响攻击成功的关键因素，探讨补丁级漏洞和语义内容之间的关系，并对潜在的防御策略进行了初步调查。这些发现为未来研究病理基础模型的对抗稳健性和可靠部署奠定了基础。我们的代码可在https://github.com/Jiashuai-Liu-hmos/Attack-WSI-pathology-foundation-models上公开获取。



## **41. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP**

X-Transfer攻击：CLIP上的超级可转移对抗攻击 cs.CV

ICML 2025

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.05528v3) [paper-pdf](http://arxiv.org/pdf/2505.05528v3)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.

摘要: 随着对比图像预训练（CLIP）模型越来越多地被用于各种下游任务并集成到大型视觉语言模型（VLM）中，它们对对抗性扰动的敏感性已成为一个关键问题。在这项工作中，我们介绍了\textbf{X-Transfer}，一种新的攻击方法，暴露了CLIP中的一个普遍的对抗性漏洞。X-Transfer生成一个通用对抗扰动（Universal Adversarial Perturbation，UAP），能够欺骗不同样本、任务和域中的各种CLIP编码器和下游VLM。我们将此属性称为\textbf{super transferability}--一个同时实现跨数据、跨域、跨模型和跨任务对抗性可转移性的单一扰动。这是通过\textBF{代理缩放}来实现的，这是我们方法的一个关键创新。与依赖于固定代理模型（扩展计算密集型）的现有方法不同，X-Transfer采用高效的代理扩展策略，可以从大搜索空间中动态选择合适代理的一小子集。广泛的评估表明，X-Transfer的性能显着优于之前最先进的UAP方法，为跨CLIP模型的对抗性可移植性建立了新的基准。该代码可在我们的\href{https：//github.com/HanxunH/XTransferBench}{GitHub存储库}中公开获取。



## **42. LLM Agents Should Employ Security Principles**

LLM代理应采用安全原则 cs.CR

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.24019v1) [paper-pdf](http://arxiv.org/pdf/2505.24019v1)

**Authors**: Kaiyuan Zhang, Zian Su, Pin-Yu Chen, Elisa Bertino, Xiangyu Zhang, Ninghui Li

**Abstract**: Large Language Model (LLM) agents show considerable promise for automating complex tasks using contextual reasoning; however, interactions involving multiple agents and the system's susceptibility to prompt injection and other forms of context manipulation introduce new vulnerabilities related to privacy leakage and system exploitation. This position paper argues that the well-established design principles in information security, which are commonly referred to as security principles, should be employed when deploying LLM agents at scale. Design principles such as defense-in-depth, least privilege, complete mediation, and psychological acceptability have helped guide the design of mechanisms for securing information systems over the last five decades, and we argue that their explicit and conscientious adoption will help secure agentic systems. To illustrate this approach, we introduce AgentSandbox, a conceptual framework embedding these security principles to provide safeguards throughout an agent's life-cycle. We evaluate with state-of-the-art LLMs along three dimensions: benign utility, attack utility, and attack success rate. AgentSandbox maintains high utility for its intended functions under both benign and adversarial evaluations while substantially mitigating privacy risks. By embedding secure design principles as foundational elements within emerging LLM agent protocols, we aim to promote trustworthy agent ecosystems aligned with user privacy expectations and evolving regulatory requirements.

摘要: 大型语言模型（LLM）代理使用上下文推理自动化复杂的任务显示出相当大的希望;然而，涉及多个代理的交互和系统对提示注入和其他形式的上下文操作的敏感性引入了与隐私泄露和系统利用相关的新漏洞。这份立场文件认为，在大规模部署LLM代理时，应采用信息安全中公认的设计原则，通常称为安全原则。设计原则，如纵深防御，最小的特权，完全调解，心理上的可接受性，帮助指导设计的机制，确保信息系统在过去的五十年，我们认为，他们明确和认真的采用将有助于安全代理系统。为了说明这种方法，我们引入AgentSandbox，一个概念框架嵌入这些安全原则，在整个代理的生命周期提供保障。我们沿着三个维度评估最先进的LLM：良性效用，攻击效用和攻击成功率。AgentSandbox在良性和对抗性评估下保持其预期功能的高实用性，同时大大降低隐私风险。通过在新兴的LLM代理协议中嵌入安全设计原则作为基本元素，我们的目标是促进与用户隐私期望和不断变化的监管要求相一致的值得信赖的代理生态系统。



## **43. Approaching the Harm of Gradient Attacks While Only Flipping Labels**

在仅翻转标签的同时应对梯度攻击的危害 cs.CR

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2503.00140v2) [paper-pdf](http://arxiv.org/pdf/2503.00140v2)

**Authors**: Abdessamad El-Kabid, El-Mahdi El-Mhamdi

**Abstract**: Machine learning systems deployed in distributed or federated environments are highly susceptible to adversarial manipulations, particularly availability attacks -adding imperceptible perturbations to training data, thereby rendering the trained model unavailable. Prior research in distributed machine learning has demonstrated such adversarial effects through the injection of gradients or data poisoning. In this study, we aim to enhance comprehension of the potential of weaker (and more probable) adversaries by posing the following inquiry: Can availability attacks be inflicted solely through the flipping of a subset of training labels, without altering features, and under a strict flipping budget? We analyze the extent of damage caused by constrained label flipping attacks. Focusing on a distributed classification problem, (1) we propose a novel formalization of label flipping attacks on logistic regression models and derive a greedy algorithm that is provably optimal at each training step. (2) To demonstrate that availability attacks can be approached by label flipping alone, we show that a budget of only $0.1\%$ of labels at each training step can reduce the accuracy of the model by $6\%$, and that some models can perform worse than random guessing when up to $25\%$ of labels are flipped. (3) We shed light on an interesting interplay between what the attacker gains from more write-access versus what they gain from more flipping budget. (4) we define and compare the power of targeted label flipping attack to that of an untargeted label flipping attack.

摘要: 部署在分布式或联邦环境中的机器学习系统极易受到对抗性操纵，尤其是可用性攻击--向训练数据添加难以察觉的扰动，从而使训练模型不可用。先前对分布式机器学习的研究已经证明了通过注入梯度或数据中毒来产生这种对抗效应。在这项研究中，我们的目标是通过提出以下问题来增强对较弱（且更有可能）对手潜力的理解：可用性攻击能否仅通过翻转训练标签子集而不改变特征，并在严格的翻转预算下？我们分析了受约束标签翻转攻击造成的损害程度。专注于分布式分类问题，（1）我们提出了对逻辑回归模型的标签翻转攻击的新颖形式化，并推导出一种在每个训练步骤都是可证明最优的贪婪算法。(2)为了证明可用性攻击可以仅通过标签翻转来应对，我们表明，每个训练步骤中仅花费0.1美元的标签预算可以将模型的准确性降低6美元，并且当标签翻转高达25美元时，某些模型的性能可能比随机猜测更差。(3)我们揭示了攻击者从更多写访问中获得的东西与从更多翻转预算中获得的东西之间有趣的相互作用。(4)我们定义并比较了有针对性的标签翻转攻击与无针对性的标签翻转攻击的威力。



## **44. HoneySat: A Network-based Satellite Honeypot Framework**

HoneySat：基于网络的卫星蜜罐框架 cs.CR

Efr\'en L\'opez-Morales and Ulysse Planta contributed equally to this  work

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.24008v1) [paper-pdf](http://arxiv.org/pdf/2505.24008v1)

**Authors**: Efrén López-Morales, Ulysse Planta, Gabriele Marra, Carlos González, Jacob Hopkins, Majid Garoosi, Elías Obreque, Carlos Rubio-Medrano, Ali Abbasi

**Abstract**: Satellites are the backbone of several mission-critical services, such as GPS that enable our modern society to function. For many years, satellites were assumed to be secure because of their indecipherable architectures and the reliance on security by obscurity. However, technological advancements have made these assumptions obsolete, paving the way for potential attacks, and sparking a renewed interest in satellite security. Unfortunately, to this day, there is no efficient way to collect data on adversarial techniques for satellites, which severely hurts the generation of security intelligence. In this paper, we present HoneySat, the first high-interaction satellite honeypot framework, which is fully capable of convincingly simulating a real-world CubeSat, a type of Small Satellite (SmallSat) widely used in practice. To provide evidence of the effectiveness of HoneySat, we surveyed experienced SmallSat operators currently in charge of active in-orbit satellite missions. Results revealed that the majority of satellite operators (71.4%) agreed that HoneySat provides realistic and engaging simulations of CubeSat missions. Further experimental evaluations also showed that HoneySat provides adversaries with extensive interaction opportunities by supporting the majority of adversarial techniques (86.8%) and tactics (100%) that target satellites. Additionally, we also obtained a series of real interactions from actual adversaries by deploying HoneySat on the internet over several months, confirming that HoneySat can operate covertly and efficiently while collecting highly valuable interaction data.

摘要: 卫星是多种关键任务服务的支柱，例如使我们的现代社会能够正常运转的GPS。多年来，卫星被认为是安全的，因为它们的体系结构难以破译，而且对安全性的依赖是默默无闻的。然而，技术进步使这些假设变得过时，为潜在的攻击铺平了道路，并引发了人们对卫星安全的新兴趣。不幸的是，直到今天，还没有有效的方法来收集有关卫星对抗技术的数据，这严重损害了安全情报的生成。在本文中，我们介绍了HoneySat，这是第一个高交互性卫星蜜罐框架，它完全能够令人信服地模拟现实世界的CubeSat，这是一种在实践中广泛使用的小型卫星（SmallSat）。为了提供HoneySat有效性的证据，我们调查了目前负责主动轨道卫星任务的经验丰富的SmallSat运营商。结果显示，大多数卫星运营商（71.4%）同意HoneySat提供了真实且引人入胜的CubeSat任务模拟。进一步的实验评估还表明，HoneySat通过支持大多数针对卫星的对抗技术（86.8%）和战术（100%），为对手提供了广泛的互动机会。此外，我们还通过几个月的时间在互联网上部署HoneySat，从实际对手那里获得了一系列真实的交互，证实HoneySat可以秘密有效地运行，同时收集极具价值的交互数据。



## **45. SVIP: Towards Verifiable Inference of Open-source Large Language Models**

SVIP：迈向开源大型语言模型的可验证推理 cs.LG

22 pages

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2410.22307v2) [paper-pdf](http://arxiv.org/pdf/2410.22307v2)

**Authors**: Yifan Sun, Yuhang Li, Yue Zhang, Yuchen Jin, Huan Zhang

**Abstract**: The ever-increasing size of open-source Large Language Models (LLMs) renders local deployment impractical for individual users. Decentralized computing has emerged as a cost-effective solution, allowing individuals and small companies to perform LLM inference for users using surplus computational power. However, a computing provider may stealthily substitute the requested LLM with a smaller, less capable model without consent from users, thereby benefiting from cost savings. We introduce SVIP, a secret-based verifiable LLM inference protocol. Unlike existing solutions based on cryptographic or game-theoretic techniques, our method is computationally effective and does not rest on strong assumptions. Our protocol requires the computing provider to return both the generated text and processed hidden representations from LLMs. We then train a proxy task on these representations, effectively transforming them into a unique model identifier. With our protocol, users can reliably verify whether the computing provider is acting honestly. A carefully integrated secret mechanism further strengthens its security. We thoroughly analyze our protocol under multiple strong and adaptive adversarial scenarios. Our extensive experiments demonstrate that SVIP is accurate, generalizable, computationally efficient, and resistant to various attacks. Notably, SVIP achieves false negative rates below 5% and false positive rates below 3%, while requiring less than 0.01 seconds per prompt query for verification.

摘要: 开源大型语言模型（LLM）的规模不断扩大，使得本地部署对于个人用户来说变得不切实际。去中心化计算已成为一种具有成本效益的解决方案，允许个人和小公司使用剩余计算能力为用户执行LLM推理。然而，计算提供商可能会在未经用户同意的情况下偷偷地用更小、功能较差的模型替换请求的LLM，从而受益于成本节约。我们引入了SVIP，这是一种基于秘密的可验证LLM推理协议。与现有的解决方案的基础上加密或博弈论技术，我们的方法是计算有效的，不依赖于强假设。我们的协议要求计算提供者从LLM返回生成的文本和处理后的隐藏表示。然后，我们在这些表示上训练代理任务，有效地将它们转换为唯一的模型标识符。使用我们的协议，用户可以可靠地验证计算提供商是否诚实行事。精心集成的秘密机制进一步加强了其安全性。我们在多种强大且自适应的对抗场景下彻底分析了我们的协议。我们广泛的实验表明，SVIP准确、可概括、计算效率高，并且可以抵抗各种攻击。值得注意的是，SVIP的假阴性率低于5%，假阳性率低于3%，同时每次提示查询需要不到0.01秒的时间进行验证。



## **46. SafeScientist: Toward Risk-Aware Scientific Discoveries by LLM Agents**

安全科学家：LLM代理人的风险意识科学发现 cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23559v1) [paper-pdf](http://arxiv.org/pdf/2505.23559v1)

**Authors**: Kunlun Zhu, Jiaxun Zhang, Ziheng Qi, Nuoxing Shang, Zijia Liu, Peixuan Han, Yue Su, Haofei Yu, Jiaxuan You

**Abstract**: Recent advancements in large language model (LLM) agents have significantly accelerated scientific discovery automation, yet concurrently raised critical ethical and safety concerns. To systematically address these challenges, we introduce \textbf{SafeScientist}, an innovative AI scientist framework explicitly designed to enhance safety and ethical responsibility in AI-driven scientific exploration. SafeScientist proactively refuses ethically inappropriate or high-risk tasks and rigorously emphasizes safety throughout the research process. To achieve comprehensive safety oversight, we integrate multiple defensive mechanisms, including prompt monitoring, agent-collaboration monitoring, tool-use monitoring, and an ethical reviewer component. Complementing SafeScientist, we propose \textbf{SciSafetyBench}, a novel benchmark specifically designed to evaluate AI safety in scientific contexts, comprising 240 high-risk scientific tasks across 6 domains, alongside 30 specially designed scientific tools and 120 tool-related risk tasks. Extensive experiments demonstrate that SafeScientist significantly improves safety performance by 35\% compared to traditional AI scientist frameworks, without compromising scientific output quality. Additionally, we rigorously validate the robustness of our safety pipeline against diverse adversarial attack methods, further confirming the effectiveness of our integrated approach. The code and data will be available at https://github.com/ulab-uiuc/SafeScientist. \textcolor{red}{Warning: this paper contains example data that may be offensive or harmful.}

摘要: 大型语言模型（LLM）代理的最新进展显着加速了科学发现自动化，但同时提出了关键的伦理和安全问题。为了系统地应对这些挑战，我们引入了一个创新的人工智能科学家框架，旨在增强人工智能驱动的科学探索中的安全和道德责任。SafeScientist主动拒绝道德上不合适或高风险的任务，并在整个研究过程中严格强调安全。为了实现全面的安全监督，我们整合了多种防御机制，包括即时监控、代理协作监控、工具使用监控和道德审查员组件。作为SafeScientist的补充，我们提出了\textBF{SciSafetyBench}，这是一个专门用于评估科学背景下人工智能安全性的新型基准，包括6个领域的240项高风险科学任务，以及30个专门设计的科学工具和120个工具相关的风险任务。大量实验表明，与传统的人工智能科学家框架相比，SafeScientist将安全性能显着提高了35%，而不会影响科学输出质量。此外，我们还严格验证了我们的安全管道针对各种对抗攻击方法的稳健性，进一步证实了我们集成方法的有效性。代码和数据可在https://github.com/ulab-uiuc/SafeScientist上获取。\textcolor{red}{警告：本文包含可能令人反感或有害的示例数据。}



## **47. SGD Jittering: A Training Strategy for Robust and Accurate Model-Based Architectures**

新元抖动：稳健且准确的基于模型的架构的训练策略 cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2410.14667v2) [paper-pdf](http://arxiv.org/pdf/2410.14667v2)

**Authors**: Peimeng Guan, Mark A. Davenport

**Abstract**: Inverse problems aim to reconstruct unseen data from corrupted or perturbed measurements. While most work focuses on improving reconstruction quality, generalization accuracy and robustness are equally important, especially for safety-critical applications. Model-based architectures (MBAs), such as loop unrolling methods, are considered more interpretable and achieve better reconstructions. Empirical evidence suggests that MBAs are more robust to perturbations than black-box solvers, but the accuracy-robustness tradeoff in MBAs remains underexplored. In this work, we propose a simple yet effective training scheme for MBAs, called SGD jittering, which injects noise iteration-wise during reconstruction. We theoretically demonstrate that SGD jittering not only generalizes better than the standard mean squared error training but is also more robust to average-case attacks. We validate SGD jittering using denoising toy examples, seismic deconvolution, and single-coil MRI reconstruction. Both SGD jittering and its SPGD extension yield cleaner reconstructions for out-of-distribution data and demonstrates enhanced robustness against adversarial attacks.

摘要: 逆问题的目的是从损坏或扰动的测量中重建不可见的数据。虽然大多数工作都集中在提高重建质量上，但泛化精度和鲁棒性同样重要，特别是对于安全关键型应用。基于模型的架构（MBA），如循环展开方法，被认为是更可解释的，并实现更好的重建。经验证据表明，工商管理硕士更强大的扰动比黑盒求解器，但在工商管理硕士的准确性和鲁棒性的权衡仍然未充分探讨。在这项工作中，我们提出了一种简单而有效的MBA训练方案，称为BCD抖动，它在重建期间以迭代方式注入噪音。我们从理论上证明，BCD抖动不仅比标准均方误差训练更好地概括，而且对平均情况攻击也更稳健。我们使用去噪玩具示例、地震去卷积和单线圈MRI重建来验证SGD抖动。Singapore抖动及其SPVD扩展都可以为分发外数据提供更清晰的重建，并表现出针对对抗性攻击的增强的鲁棒性。



## **48. TRAP: Targeted Redirecting of Agentic Preferences**

TRAP：有针对性地重新定向统计偏好 cs.AI

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2505.23518v1) [paper-pdf](http://arxiv.org/pdf/2505.23518v1)

**Authors**: Hangoo Kang, Jehyeok Yeon, Gagandeep Singh

**Abstract**: Autonomous agentic AI systems powered by vision-language models (VLMs) are rapidly advancing toward real-world deployment, yet their cross-modal reasoning capabilities introduce new attack surfaces for adversarial manipulation that exploit semantic reasoning across modalities. Existing adversarial attacks typically rely on visible pixel perturbations or require privileged model or environment access, making them impractical for stealthy, real-world exploitation. We introduce TRAP, a generative adversarial framework that manipulates the agent's decision-making using diffusion-based semantic injections. Our method combines negative prompt-based degradation with positive semantic optimization, guided by a Siamese semantic network and layout-aware spatial masking. Without requiring access to model internals, TRAP produces visually natural images yet induces consistent selection biases in agentic AI systems. We evaluate TRAP on the Microsoft Common Objects in Context (COCO) dataset, building multi-candidate decision scenarios. Across these scenarios, TRAP achieves a 100% attack success rate on leading models, including LLaVA-34B, Gemma3, and Mistral-3.1, significantly outperforming baselines such as SPSA, Bandit, and standard diffusion approaches. These results expose a critical vulnerability: Autonomous agents can be consistently misled through human-imperceptible cross-modal manipulations. These findings highlight the need for defense strategies beyond pixel-level robustness to address semantic vulnerabilities in cross-modal decision-making.

摘要: 由视觉语言模型（VLM）驱动的自主代理人工智能系统正在迅速向现实世界的部署发展，但它们的跨模态推理能力为利用跨模态语义推理的对抗性操纵引入了新的攻击面。现有的对抗性攻击通常依赖于可见像素扰动，或者需要特权模型或环境访问，这使得它们对于隐形的现实世界利用来说是不切实际的。我们介绍了TRAP，一个生成对抗框架，使用基于扩散的语义注入来操纵代理的决策。我们的方法结合了消极的基于语义的退化与积极的语义优化，指导下的暹罗语义网络和布局感知空间掩蔽。在不需要访问模型内部的情况下，TRAP会产生视觉上自然的图像，但在代理人工智能系统中会引起一致的选择偏差。我们在Microsoft上下文中的公共对象（COCO）数据集中评估TRAP，构建多候选决策场景。在这些场景中，TRAP在LLaVA-34 B、Gemma 3和Mistral-3.1等领先型号上实现了100%的攻击成功率，显着优于SPSA、Bandit和标准扩散方法等基线。这些结果暴露了一个关键的漏洞：自治代理可能会通过人类无法感知的跨模式操纵而持续误导。这些发现凸显了像素级稳健性之外的防御策略的必要性，以解决跨模式决策中的语义漏洞。



## **49. Hijacking Large Language Models via Adversarial In-Context Learning**

通过对抗性上下文学习劫持大型语言模型 cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2311.09948v3) [paper-pdf](http://arxiv.org/pdf/2311.09948v3)

**Authors**: Xiangyu Zhou, Yao Qiang, Saleh Zare Zade, Prashant Khanduri, Dongxiao Zhu

**Abstract**: In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific downstream tasks by utilizing labeled examples as demonstrations (demos) in the preconditioned prompts. Despite its promising performance, crafted adversarial attacks pose a notable threat to the robustness of LLMs. Existing attacks are either easy to detect, require a trigger in user input, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable prompt injection attack against ICL, aiming to hijack LLMs to generate the target output or elicit harmful responses. In our threat model, the hacker acts as a model publisher who leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demos via prompt injection. We also propose effective defense strategies using a few shots of clean demos, enhancing the robustness of LLMs during ICL. Extensive experimental results across various classification and jailbreak tasks demonstrate the effectiveness of the proposed attack and defense strategies. This work highlights the significant security vulnerabilities of LLMs during ICL and underscores the need for further in-depth studies.

摘要: 上下文学习（ICL）已成为一种强大的范式，通过利用带标签的示例作为预处理提示中的演示（演示），利用LLM来执行特定的下游任务。尽管性能令人鼓舞，但精心设计的对抗攻击对LLM的稳健性构成了显着的威胁。现有的攻击要么容易检测，需要用户输入触发，要么缺乏针对ICL的特异性。为了解决这些问题，这项工作引入了一种针对ICL的新型可转移即时注入攻击，旨在劫持LLM以生成目标输出或引发有害响应。在我们的威胁模型中，黑客充当模型发布者，利用基于梯度的提示搜索方法来学习难以察觉的对抗性后缀，并通过提示注入将其添加到上下文演示中。我们还使用几次干净的演示提出了有效的防御策略，增强ICL期间LLM的稳健性。各种分类和越狱任务的大量实验结果证明了所提出的攻击和防御策略的有效性。这项工作强调了ICL期间LLM的重大安全漏洞，并强调了进一步深入研究的必要性。



## **50. Learning to Poison Large Language Models for Downstream Manipulation**

学习毒害大型语言模型以进行下游操作 cs.LG

**SubmitDate**: 2025-05-29    [abs](http://arxiv.org/abs/2402.13459v3) [paper-pdf](http://arxiv.org/pdf/2402.13459v3)

**Authors**: Xiangyu Zhou, Yao Qiang, Saleh Zare Zade, Mohammad Amin Roshani, Prashant Khanduri, Douglas Zytko, Dongxiao Zhu

**Abstract**: The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where the adversary inserts backdoor triggers into training data to manipulate outputs. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the supervised fine-tuning (SFT) process. We propose a novel gradient-guided backdoor trigger learning (GBTL) algorithm to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various language model tasks, including sentiment analysis, domain generation, and question answering, our poisoning strategy demonstrates a high success rate in compromising various LLMs' outputs. We further propose two defense strategies against data poisoning attacks, including in-context learning (ICL) and continuous learning (CL), which effectively rectify the behavior of LLMs and significantly reduce the decline in performance. Our work highlights the significant security risks present during SFT of LLMs and the necessity of safeguarding LLMs against data poisoning attacks.

摘要: 大型语言模型（LLM）的出现标志着语言处理和推理能力取得了重大成就。尽管LLM取得了进步，但仍面临数据中毒攻击的漏洞，即对手将后门触发器插入训练数据中以操纵输出。这项工作通过设计专门针对利用监督式微调（SFT）过程而定制的新数据中毒攻击，进一步识别了LLM中的额外安全风险。我们提出了一种新型的梯度引导后门触发学习（GBTL）算法来有效识别对抗触发，确保逃避传统防御的检测，同时保持内容完整性。通过对各种语言模型任务（包括情感分析、领域生成和问题回答）的实验验证，我们的中毒策略证明了损害各种LLM输出的高成功率。我们进一步提出了两种针对数据中毒攻击的防御策略，包括上下文学习（ICL）和持续学习（CL），有效纠正LLM的行为，显着减少性能下降。我们的工作强调了LLM SFT期间存在的重大安全风险以及保护LLM免受数据中毒攻击的必要性。



