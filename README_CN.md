# Latest Adversarial Attack Papers
**update at 2025-05-20 10:32:49**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants**

XOXO：针对人工智能编码助理的隐形跨源上下文中毒攻击 cs.CR

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2503.14281v2) [paper-pdf](http://arxiv.org/pdf/2503.14281v2)

**Authors**: Adam Štorek, Mukur Gupta, Noopur Bhatt, Aditya Gupta, Janie Kim, Prashast Srivastava, Suman Jana

**Abstract**: AI coding assistants are widely used for tasks like code generation. These tools now require large and complex contexts, automatically sourced from various origins$\unicode{x2014}$across files, projects, and contributors$\unicode{x2014}$forming part of the prompt fed to underlying LLMs. This automatic context-gathering introduces new vulnerabilities, allowing attackers to subtly poison input to compromise the assistant's outputs, potentially generating vulnerable code or introducing critical errors. We propose a novel attack, Cross-Origin Context Poisoning (XOXO), that is challenging to detect as it relies on adversarial code modifications that are semantically equivalent. Traditional program analysis techniques struggle to identify these perturbations since the semantics of the code remains correct, making it appear legitimate. This allows attackers to manipulate coding assistants into producing incorrect outputs, while shifting the blame to the victim developer. We introduce a novel, task-agnostic, black-box attack algorithm GCGS that systematically searches the transformation space using a Cayley Graph, achieving a 75.72% attack success rate on average across five tasks and eleven models, including GPT 4.1 and Claude 3.5 Sonnet v2 used by popular AI coding assistants. Furthermore, defenses like adversarial fine-tuning are ineffective against our attack, underscoring the need for new security measures in LLM-powered coding tools.

摘要: 人工智能编码助手广泛用于代码生成等任务。这些工具现在需要大型而复杂的上下文，自动从各种来源$\unicode {x2014}$跨文件、项目和贡献者$\unicode {x2014}$获取，形成了向底层LLM提供提示的一部分。这种自动上下文收集引入了新的漏洞，允许攻击者巧妙地毒害输入以损害助手的输出，从而可能生成易受攻击的代码或引入严重错误。我们提出了一种新型攻击，即跨源上下文中毒（XOXO），检测起来很有挑战性，因为它依赖于语义等效的对抗性代码修改。传统的程序分析技术很难识别这些扰动，因为代码的语义保持正确，使其看起来合法。这使得攻击者能够操纵编码助手产生错误的输出，同时将责任归咎于受害开发人员。我们引入了一种新颖的、任务不可知的黑匣子攻击算法GCGS，该算法使用凯莱图系统性地搜索转换空间，在五个任务和十一个模型（包括GPT 4.1和Claude 3.5 Sonnet v2）上平均实现75.72%的攻击成功率流行AI编码助手使用。此外，对抗性微调等防御措施对我们的攻击无效，这凸显了LLM支持的编码工具中需要新的安全措施。



## **2. Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks**

调查LLM作为法官架构对预算注入攻击的脆弱性 cs.CL

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.13348v1) [paper-pdf](http://arxiv.org/pdf/2505.13348v1)

**Authors**: Narek Maloyan, Bislan Ashinov, Dmitry Namiot

**Abstract**: Large Language Models (LLMs) are increasingly employed as evaluators (LLM-as-a-Judge) for assessing the quality of machine-generated text. This paradigm offers scalability and cost-effectiveness compared to human annotation. However, the reliability and security of such systems, particularly their robustness against adversarial manipulations, remain critical concerns. This paper investigates the vulnerability of LLM-as-a-Judge architectures to prompt-injection attacks, where malicious inputs are designed to compromise the judge's decision-making process. We formalize two primary attack strategies: Comparative Undermining Attack (CUA), which directly targets the final decision output, and Justification Manipulation Attack (JMA), which aims to alter the model's generated reasoning. Using the Greedy Coordinate Gradient (GCG) optimization method, we craft adversarial suffixes appended to one of the responses being compared. Experiments conducted on the MT-Bench Human Judgments dataset with open-source instruction-tuned LLMs (Qwen2.5-3B-Instruct and Falcon3-3B-Instruct) demonstrate significant susceptibility. The CUA achieves an Attack Success Rate (ASR) exceeding 30\%, while JMA also shows notable effectiveness. These findings highlight substantial vulnerabilities in current LLM-as-a-Judge systems, underscoring the need for robust defense mechanisms and further research into adversarial evaluation and trustworthiness in LLM-based assessment frameworks.

摘要: 大型语言模型（LLM）越来越多地被用作评估器（LLM as-a-Judge）来评估机器生成文本的质量。与人类注释相比，该范式提供了可扩展性和成本效益。然而，此类系统的可靠性和安全性，特别是它们对对抗性操纵的鲁棒性，仍然是关键问题。本文研究了LLM as-a-Judge架构对预算注入攻击的脆弱性，其中恶意输入旨在损害法官的决策过程。我们正式化了两种主要的攻击策略：比较挖掘攻击（CUA），直接针对最终决策输出，和合理化操纵攻击（JMA），旨在改变模型生成的推理。使用贪婪坐标梯度（GCG）优化方法，我们制作附加到正在比较的一个响应上的对抗后缀。在MT-Bench Human Judgments数据集上使用开源描述调整的LLM（Qwen 2.5 - 3B-Direct和Falcon 3 - 3B-Direct）进行的实验证明了显着的易感性。CUA的攻击成功率（ASB）超过30%，而JMA也表现出显着的有效性。这些发现凸显了当前法学硕士作为法官系统中的重大漏洞，强调了强大的防御机制以及对基于法学硕士的评估框架中的对抗性评估和可信度进行进一步研究的必要性。



## **3. FlowPure: Continuous Normalizing Flows for Adversarial Purification**

FlowPure：对抗性净化的连续标准化流程 cs.LG

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.13280v1) [paper-pdf](http://arxiv.org/pdf/2505.13280v1)

**Authors**: Elias Collaert, Abel Rodríguez, Sander Joos, Lieven Desmet, Vera Rimmer

**Abstract**: Despite significant advancements in the area, adversarial robustness remains a critical challenge in systems employing machine learning models. The removal of adversarial perturbations at inference time, known as adversarial purification, has emerged as a promising defense strategy. To achieve this, state-of-the-art methods leverage diffusion models that inject Gaussian noise during a forward process to dilute adversarial perturbations, followed by a denoising step to restore clean samples before classification. In this work, we propose FlowPure, a novel purification method based on Continuous Normalizing Flows (CNFs) trained with Conditional Flow Matching (CFM) to learn mappings from adversarial examples to their clean counterparts. Unlike prior diffusion-based approaches that rely on fixed noise processes, FlowPure can leverage specific attack knowledge to improve robustness under known threats, while also supporting a more general stochastic variant trained on Gaussian perturbations for settings where such knowledge is unavailable. Experiments on CIFAR-10 and CIFAR-100 demonstrate that our method outperforms state-of-the-art purification-based defenses in preprocessor-blind and white-box scenarios, and can do so while fully preserving benign accuracy in the former. Moreover, our results show that not only is FlowPure a highly effective purifier but it also holds a strong potential for adversarial detection, identifying preprocessor-blind PGD samples with near-perfect accuracy.

摘要: 尽管该领域取得了重大进展，但对抗鲁棒性仍然是采用机器学习模型的系统的一个关键挑战。在推理时消除对抗性扰动，称为对抗性净化，已成为一种有前途的防御策略。为了实现这一目标，最先进的方法利用扩散模型，该模型在正向过程中注入高斯噪音以稀释对抗性扰动，然后进行去噪步骤以在分类之前恢复干净的样本。在这项工作中，我们提出了FlowPure，这是一种新型净化方法，基于用条件流匹配（CGM）训练的连续正规化流（CNF），以学习从对抗性示例到干净对应的映射。与依赖于固定噪声过程的先前基于扩散的方法不同，FlowPure可以利用特定的攻击知识来提高已知威胁下的鲁棒性，同时还支持在高斯扰动上训练的更一般的随机变量，用于此类知识不可用的设置。在CIFAR-10和CIFAR-100上的实验表明，我们的方法在预处理器盲和白盒场景中的性能优于最先进的基于纯化的防御，并且可以在完全保留良性准确性的同时做到这一点。此外，我们的研究结果表明，FlowPure不仅是一种高效的净化器，而且在对抗性检测方面也具有很强的潜力，能够以近乎完美的准确度识别预处理器盲PGD样本。



## **4. Constrained Adversarial Learning for Automated Software Testing: a literature review**

用于自动化软件测试的约束对抗学习：文献综述 cs.SE

36 pages, 4 tables, 2 figures, Discover Applied Sciences journal

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2303.07546v3) [paper-pdf](http://arxiv.org/pdf/2303.07546v3)

**Authors**: João Vitorino, Tiago Dias, Tiago Fonseca, Eva Maia, Isabel Praça

**Abstract**: It is imperative to safeguard computer applications and information systems against the growing number of cyber-attacks. Automated software testing tools can be developed to quickly analyze many lines of code and detect vulnerabilities by generating function-specific testing data. This process draws similarities to the constrained adversarial examples generated by adversarial machine learning methods, so there could be significant benefits to the integration of these methods in testing tools to identify possible attack vectors. Therefore, this literature review is focused on the current state-of-the-art of constrained data generation approaches applied for adversarial learning and software testing, aiming to guide researchers and developers to enhance their software testing tools with adversarial testing methods and improve the resilience and robustness of their information systems. The found approaches were systematized, and the advantages and limitations of those specific for white-box, grey-box, and black-box testing were analyzed, identifying research gaps and opportunities to automate the testing tools with data generated by adversarial attacks.

摘要: 保护计算机应用程序和信息系统免受日益增多的网络攻击至关重要。可以开发自动化软件测试工具来快速分析多行代码并通过生成特定于功能的测试数据来检测漏洞。该过程与对抗性机器学习方法生成的受约束对抗示例具有相似之处，因此将这些方法集成到测试工具中以识别可能的攻击载体可能会带来显着的好处。因此，本次文献综述重点关注当前应用于对抗性学习和软件测试的约束数据生成方法的最新发展水平，旨在指导研究人员和开发人员通过对抗性测试方法增强其软件测试工具，并提高其信息系统的弹性和稳健性。对所发现的方法进行了系统化，并分析了白盒、灰盒和黑盒测试方法的优点和局限性，确定了研究差距和机会，以利用对抗性攻击生成的数据自动化测试工具。



## **5. Test-time Adversarial Defense with Opposite Adversarial Path and High Attack Time Cost**

具有相反对抗路径和高攻击时间成本的测试时对抗防御 cs.LG

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2410.16805v2) [paper-pdf](http://arxiv.org/pdf/2410.16805v2)

**Authors**: Cheng-Han Yeh, Kuanchun Yu, Chun-Shien Lu

**Abstract**: Deep learning models are known to be vulnerable to adversarial attacks by injecting sophisticated designed perturbations to input data. Training-time defenses still exhibit a significant performance gap between natural accuracy and robust accuracy. In this paper, we investigate a new test-time adversarial defense method via diffusion-based recovery along opposite adversarial paths (OAPs). We present a purifier that can be plugged into a pre-trained model to resist adversarial attacks. Different from prior arts, the key idea is excessive denoising or purification by integrating the opposite adversarial direction with reverse diffusion to push the input image further toward the opposite adversarial direction. For the first time, we also exemplify the pitfall of conducting AutoAttack (Rand) for diffusion-based defense methods. Through the lens of time complexity, we examine the trade-off between the effectiveness of adaptive attack and its computation complexity against our defense. Experimental evaluation along with time cost analysis verifies the effectiveness of the proposed method.

摘要: 众所周知，深度学习模型通过向输入数据注入复杂的设计扰动而容易受到对抗攻击。训练时防御在自然准确性和稳健准确性之间仍然表现出显着的性能差距。本文研究了一种新的测试时对抗防御方法，通过沿着相反对抗路径（OAP）的基于扩散的恢复。我们提供了一种净化器，它可以插入预先训练的模型中以抵抗对抗攻击。与现有技术不同，其核心思想是通过将相反的对抗方向与反向扩散相结合来进行过度降噪或净化，以将输入图像进一步推向相反的对抗方向。我们还首次指出了针对基于扩散的防御方法进行AutoAttack（Rand）的陷阱。通过时间复杂性的视角，我们研究了自适应攻击的有效性与其针对我们防御的计算复杂性之间的权衡。实验评估和时间成本分析验证了该方法的有效性。



## **6. Countermeasure against Detector Blinding Attack with Secret Key Leakage Estimation**

利用密钥泄露估计对抗检测器致盲攻击的对策 quant-ph

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12974v1) [paper-pdf](http://arxiv.org/pdf/2505.12974v1)

**Authors**: Dmitry M. Melkonian, Daniil S. Bulavkin, Kirill E. Bugai, Kirill A. Balygin, Dmitriy A. Dvoretskiy

**Abstract**: We present a countermeasure against the detector blinding attack (DBA) utilizing statistical analysis of error and double-click events accumulated during a quantum key distribution session under randomized modulation of single-photon avalanche diode (SPAD) detection efficiencies via gate voltage manipulation. Building upon prior work demonstrating the ineffectiveness of this countermeasure against continuous-wave (CW) DBA, we extend the analysis to evaluate its performance against pulsed DBA. Our findings reveal an approximately 25 dB increase in the trigger pulse energies difference between high and low gate voltage applied under pulsed DBA conditions compared to CW DBA. This heightened difference enables a re-evaluation of the feasibility of utilizing SPAD detection probability variations as a countermeasure and makes it possible to estimate the fraction of bits compromised by an adversary during pulsed DBA.

摘要: 我们利用对通过门电压操纵对单量子雪崩二极管（SPAD）检测效率进行随机调制的量子密钥分发会话期间积累的错误和双击事件的统计分析，提出了一种针对检测器致盲攻击（DBA）的对策。在之前的工作证明了这种对策对连续波（CW）DBA无效的基础上，我们扩展了分析以评估其对脉冲DBA的性能。我们的研究结果表明，与CW DBA相比，在脉冲DBA条件下施加的高和低门电压之间的触发脉冲能量差增加了约25分贝。这种增大的差异使得可以重新评估利用SPAD检测概率变化作为对策的可行性，并使得可以估计在脉冲DBA期间被对手损害的比特比例。



## **7. DICTION:DynamIC robusT whIte bOx watermarkiNg scheme for deep neural networks**

DICTION：动态robusT whIte bOx水印用于深度神经网络的方案 cs.CR

24 pages, 4 figures, PrePrint

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2210.15745v2) [paper-pdf](http://arxiv.org/pdf/2210.15745v2)

**Authors**: Reda Bellafqira, Gouenou Coatrieux

**Abstract**: Deep neural network (DNN) watermarking is a suitable method for protecting the ownership of deep learning (DL) models. It secretly embeds an identifier (watermark) within the model, which can be retrieved by the owner to prove ownership. In this paper, we first provide a unified framework for white box DNN watermarking schemes. It includes current state-of-the-art methods outlining their theoretical inter-connections. Next, we introduce DICTION, a new white-box Dynamic Robust watermarking scheme, we derived from this framework. Its main originality stands on a generative adversarial network (GAN) strategy where the watermark extraction function is a DNN trained as a GAN discriminator taking the target model to watermark as a GAN generator with a latent space as the input of the GAN trigger set. DICTION can be seen as a generalization of DeepSigns which, to the best of our knowledge, is the only other Dynamic white-box watermarking scheme from the literature. Experiments conducted on the same model test set as Deepsigns demonstrate that our scheme achieves much better performance. Especially, with DICTION, one can increase the watermark capacity while preserving the target model accuracy at best and simultaneously ensuring strong watermark robustness against a wide range of watermark removal and detection attacks.

摘要: 深度神经网络（DNN）水印是一种保护深度学习（DL）模型所有权的合适方法。它秘密地在模型中嵌入一个标识符（水印），所有者可以检索该标识符以证明所有权。本文首先为白盒DNN水印方案提供了一个统一的框架。它包括目前最先进的方法，概述了它们的理论相互联系。接下来，我们介绍了DICTION，这是一种新的白盒动态鲁棒水印方案，我们从这个框架中衍生出来。它的主要独创性基于生成对抗网络（GAN）策略，其中水印提取功能是一个DNN，作为GAN预设，将目标模型进行水印作为GAN生成器，其中潜在空间作为GAN触发器集的输入。DICTION可以被视为DeepSigns的概括，据我们所知，DeepSigns是文献中唯一的其他动态白盒水印方案。在与Deepsards相同的模型测试集上进行的实验表明，我们的方案实现了更好的性能。特别是，与DICTION，可以增加水印容量，同时保持目标模型的准确性最好，同时确保强大的水印鲁棒性对广泛的水印删除和检测攻击。



## **8. Language Models That Walk the Talk: A Framework for Formal Fairness Certificates**

直言不讳的语言模型：正式公平证书的框架 cs.AI

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12767v1) [paper-pdf](http://arxiv.org/pdf/2505.12767v1)

**Authors**: Danqing Chen, Tobias Ladner, Ahmed Rayen Mhadhbi, Matthias Althoff

**Abstract**: As large language models become integral to high-stakes applications, ensuring their robustness and fairness is critical. Despite their success, large language models remain vulnerable to adversarial attacks, where small perturbations, such as synonym substitutions, can alter model predictions, posing risks in fairness-critical areas, such as gender bias mitigation, and safety-critical areas, such as toxicity detection. While formal verification has been explored for neural networks, its application to large language models remains limited. This work presents a holistic verification framework to certify the robustness of transformer-based language models, with a focus on ensuring gender fairness and consistent outputs across different gender-related terms. Furthermore, we extend this methodology to toxicity detection, offering formal guarantees that adversarially manipulated toxic inputs are consistently detected and appropriately censored, thereby ensuring the reliability of moderation systems. By formalizing robustness within the embedding space, this work strengthens the reliability of language models in ethical AI deployment and content moderation.

摘要: 随着大型语言模型成为高风险应用程序的组成部分，确保其稳健性和公平性至关重要。尽管取得了成功，大型语言模型仍然容易受到对抗攻击，其中同义词替换等小扰动可能会改变模型预测，从而在性别偏见缓解等公平关键领域和安全关键领域带来风险，例如毒性检测。虽然已经探索了神经网络的形式验证，但其在大型语言模型中的应用仍然有限。这项工作提出了一个整体验证框架，以验证基于转换器的语言模型的稳健性，重点是确保性别公平性和不同性别相关术语的一致输出。此外，我们将这种方法扩展到毒性检测，提供正式保证，以一致地检测和适当审查敌对操纵的有毒输入，从而确保审核系统的可靠性。通过形式化嵌入空间内的鲁棒性，这项工作增强了语言模型在道德人工智能部署和内容审核中的可靠性。



## **9. BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models**

BackdoorLLM：大型语言模型后门攻击和防御的综合基准 cs.AI

22 pages

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2408.12798v2) [paper-pdf](http://arxiv.org/pdf/2408.12798v2)

**Authors**: Yige Li, Hanxun Huang, Yunhan Zhao, Xingjun Ma, Jun Sun

**Abstract**: Generative large language models (LLMs) have achieved state-of-the-art results on a wide range of tasks, yet they remain susceptible to backdoor attacks: carefully crafted triggers in the input can manipulate the model to produce adversary-specified outputs. While prior research has predominantly focused on backdoor risks in vision and classification settings, the vulnerability of LLMs in open-ended text generation remains underexplored. To fill this gap, we introduce BackdoorLLM (Our BackdoorLLM benchmark was awarded First Prize in the SafetyBench competition, https://www.mlsafety.org/safebench/winners, organized by the Center for AI Safety, https://safe.ai/.), the first comprehensive benchmark for systematically evaluating backdoor threats in text-generation LLMs. BackdoorLLM provides: (i) a unified repository of benchmarks with a standardized training and evaluation pipeline; (ii) a diverse suite of attack modalities, including data poisoning, weight poisoning, hidden-state manipulation, and chain-of-thought hijacking; (iii) over 200 experiments spanning 8 distinct attack strategies, 7 real-world scenarios, and 6 model architectures; (iv) key insights into the factors that govern backdoor effectiveness and failure modes in LLMs; and (v) a defense toolkit encompassing 7 representative mitigation techniques. Our code and datasets are available at https://github.com/bboylyg/BackdoorLLM. We will continuously incorporate emerging attack and defense methodologies to support the research in advancing the safety and reliability of LLMs.

摘要: 生成式大型语言模型（LLM）在广泛的任务上取得了最先进的结果，但它们仍然容易受到后门攻击：输入中精心制作的触发器可以操纵模型以产生对手指定的输出。虽然先前的研究主要集中在视觉和分类设置中的后门风险，但LLM在开放式文本生成中的脆弱性仍然没有得到充分的研究。为了填补这一空白，我们引入了BackdoorLLM（我们的BackdoorLLM基准测试在SafetyBench竞赛中获得一等奖，https：//www.mlsafety.org/safebench/winners，由AI安全中心组织，https：//safe.ai/.），系统评估文本生成LLM中后门威胁的第一个全面基准。BackdoorLLM提供：（i）一个统一的基准库，具有标准化的培训和评估管道;（ii）一套多样化的攻击模式，包括数据中毒，权重中毒，隐藏状态操纵和思想链劫持;（iii）超过200个实验，涵盖8种不同的攻击策略，7种真实场景和6种模型架构;（iv）对制约LLM后门有效性和故障模式的因素的关键见解;以及（v）包含7种代表性缓解技术的防御工具包。我们的代码和数据集可在https://github.com/bboylyg/BackdoorLLM上获取。我们将不断结合新兴的攻击和防御方法，以支持研究，提高LLM的安全性和可靠性。



## **10. Bullying the Machine: How Personas Increase LLM Vulnerability**

欺凌机器：角色扮演如何增加LLM漏洞 cs.AI

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12692v1) [paper-pdf](http://arxiv.org/pdf/2505.12692v1)

**Authors**: Ziwei Xu, Udit Sanghi, Mohan Kankanhalli

**Abstract**: Large Language Models (LLMs) are increasingly deployed in interactions where they are prompted to adopt personas. This paper investigates whether such persona conditioning affects model safety under bullying, an adversarial manipulation that applies psychological pressures in order to force the victim to comply to the attacker. We introduce a simulation framework in which an attacker LLM engages a victim LLM using psychologically grounded bullying tactics, while the victim adopts personas aligned with the Big Five personality traits. Experiments using multiple open-source LLMs and a wide range of adversarial goals reveal that certain persona configurations -- such as weakened agreeableness or conscientiousness -- significantly increase victim's susceptibility to unsafe outputs. Bullying tactics involving emotional or sarcastic manipulation, such as gaslighting and ridicule, are particularly effective. These findings suggest that persona-driven interaction introduces a novel vector for safety risks in LLMs and highlight the need for persona-aware safety evaluation and alignment strategies.

摘要: 大型语言模型（LLM）越来越多地部署在交互中，它们会被提示采用角色。本文研究了这种角色条件反射是否会影响欺凌下的模型安全性，欺凌是一种对抗性操纵，施加心理压力以迫使受害者服从攻击者。我们引入了一个模拟框架，其中攻击者LLM使用基于心理的欺凌策略与受害者LLM互动，而受害者则采用与五大人格特征一致的角色。使用多个开源LLM和广泛的对抗目标的实验表明，某些角色配置（例如减弱的宜人性或危险性）会显着增加受害者对不安全输出的易感性。涉及情感或讽刺操纵的欺凌策略，例如煤气灯和嘲笑，尤其有效。这些发现表明，个性驱动的交互为LLM中的安全风险引入了一种新的载体，并强调了对个性意识的安全评估和协调策略的必要性。



## **11. RoVo: Robust Voice Protection Against Unauthorized Speech Synthesis with Embedding-Level Perturbations**

RoVo：针对未经授权的语音合成的强大语音保护，具有嵌入级扰动 cs.LG

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12686v1) [paper-pdf](http://arxiv.org/pdf/2505.12686v1)

**Authors**: Seungmin Kim, Sohee Park, Donghyun Kim, Jisu Lee, Daeseon Choi

**Abstract**: With the advancement of AI-based speech synthesis technologies such as Deep Voice, there is an increasing risk of voice spoofing attacks, including voice phishing and fake news, through unauthorized use of others' voices. Existing defenses that inject adversarial perturbations directly into audio signals have limited effectiveness, as these perturbations can easily be neutralized by speech enhancement methods. To overcome this limitation, we propose RoVo (Robust Voice), a novel proactive defense technique that injects adversarial perturbations into high-dimensional embedding vectors of audio signals, reconstructing them into protected speech. This approach effectively defends against speech synthesis attacks and also provides strong resistance to speech enhancement models, which represent a secondary attack threat.   In extensive experiments, RoVo increased the Defense Success Rate (DSR) by over 70% compared to unprotected speech, across four state-of-the-art speech synthesis models. Specifically, RoVo achieved a DSR of 99.5% on a commercial speaker-verification API, effectively neutralizing speech synthesis attack. Moreover, RoVo's perturbations remained robust even under strong speech enhancement conditions, outperforming traditional methods. A user study confirmed that RoVo preserves both naturalness and usability of protected speech, highlighting its effectiveness in complex and evolving threat scenarios.

摘要: 随着Deep Voice等基于人工智能的语音合成技术的进步，通过未经授权使用他人语音进行语音欺骗攻击的风险越来越大，包括语音网络钓鱼和假新闻。将对抗性扰动直接注入音频信号的现有防御措施的有效性有限，因为这些扰动很容易被语音增强方法抵消。为了克服这一限制，我们提出了RoVo（鲁棒语音），这是一种新型的主动防御技术，它将对抗性扰动注入音频信号的多维嵌入载体中，将它们重建为受保护的语音。这种方法有效地防御语音合成攻击，并且还对代表二次攻击威胁的语音增强模型提供了强大的抵抗力。   在广泛的实验中，与无保护语音相比，RoVo在四种最先进的语音合成模型中将防御成功率（SVR）提高了70%以上。具体来说，RoVo在商业说话者验证API上实现了99.5%的SWR，有效地中和了语音合成攻击。此外，即使在强语音增强条件下，RoVo的扰动也保持稳健，优于传统方法。一项用户研究证实，RoVo保留了受保护语音的自然性和可用性，凸显了其在复杂且不断变化的威胁场景中的有效性。



## **12. On the Mechanisms of Adversarial Data Augmentation for Robust and Adaptive Transfer Learning**

鲁棒自适应迁移学习的对抗数据增强机制 cs.LG

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12681v1) [paper-pdf](http://arxiv.org/pdf/2505.12681v1)

**Authors**: Hana Satou, Alan Mitkiy

**Abstract**: Transfer learning across domains with distribution shift remains a fundamental challenge in building robust and adaptable machine learning systems. While adversarial perturbations are traditionally viewed as threats that expose model vulnerabilities, recent studies suggest that they can also serve as constructive tools for data augmentation. In this work, we systematically investigate the role of adversarial data augmentation (ADA) in enhancing both robustness and adaptivity in transfer learning settings. We analyze how adversarial examples, when used strategically during training, improve domain generalization by enriching decision boundaries and reducing overfitting to source-domain-specific features. We further propose a unified framework that integrates ADA with consistency regularization and domain-invariant representation learning. Extensive experiments across multiple benchmark datasets -- including VisDA, DomainNet, and Office-Home -- demonstrate that our method consistently improves target-domain performance under both unsupervised and few-shot domain adaptation settings. Our results highlight a constructive perspective of adversarial learning, transforming perturbation from a destructive attack into a regularizing force for cross-domain transferability.

摘要: 具有分布转移的跨领域迁移学习仍然是构建强大且适应性强的机器学习系统的根本挑战。虽然对抗性扰动传统上被视为暴露模型漏洞的威胁，但最近的研究表明，它们也可以作为数据增强的建设性工具。在这项工作中，我们系统地研究了对抗数据增强（ADA）在增强迁移学习环境中的稳健性和适应性方面的作用。我们分析了当在训练期间战略性地使用对抗性示例时，如何通过丰富决策边界和减少对源域特定特征的过度匹配来提高领域概括性。我们进一步提出了一个统一的框架，将ADA与一致性正规化和域不变表示学习集成在一起。跨多个基准数据集（包括VisDA、DomainNet和Deliver-Home）的广泛实验表明，我们的方法在无监督和少镜头域适应设置下都能持续提高目标域性能。我们的结果强调了对抗学习的建设性观点，将干扰从破坏性攻击转化为跨域可移植性的正规化力量。



## **13. Spiking Neural Network: a low power solution for physical layer authentication**

Spiking神经网络：物理层身份验证的低功耗解决方案 cs.LG

11 pages, 7 figures and 2 pages

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12647v1) [paper-pdf](http://arxiv.org/pdf/2505.12647v1)

**Authors**: Jung Hoon Lee, Sujith Vijayan

**Abstract**: Deep learning (DL) is a powerful tool that can solve complex problems, and thus, it seems natural to assume that DL can be used to enhance the security of wireless communication. However, deploying DL models to edge devices in wireless networks is challenging, as they require significant amounts of computing and power resources. Notably, Spiking Neural Networks (SNNs) are known to be efficient in terms of power consumption, meaning they can be an alternative platform for DL models for edge devices. In this study, we ask if SNNs can be used in physical layer authentication. Our evaluation suggests that SNNs can learn unique physical properties (i.e., `fingerprints') of RF transmitters and use them to identify individual devices. Furthermore, we find that SNNs are also vulnerable to adversarial attacks and that an autoencoder can be used clean out adversarial perturbations to harden SNNs against them.

摘要: 深度学习（DL）是一种可以解决复杂问题的强大工具，因此，我们很自然地认为DL可以用于增强无线通信的安全性。然而，将DL模型部署到无线网络中的边缘设备具有挑战性，因为它们需要大量的计算和电力资源。值得注意的是，众所周知，Spiking神经网络（SNN）在功耗方面效率很高，这意味着它们可以成为边缘设备DL模型的替代平台。在这项研究中，我们询问SNN是否可以用于物理层认证。我们的评估表明SNN可以学习独特的物理性质（即，“指纹”）并使用它们来识别各个设备。此外，我们发现SNN也容易受到对抗性攻击，并且可以使用自动编码器清除对抗性干扰，以增强SNN对抗它们的能力。



## **14. Adversarial Attacks on Data Attribution**

对数据归因的对抗性攻击 cs.LG

Accepted at the 13th International Conference on Learning  Representations (ICLR 2025)

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2409.05657v4) [paper-pdf](http://arxiv.org/pdf/2409.05657v4)

**Authors**: Xinhe Wang, Pingbang Hu, Junwei Deng, Jiaqi W. Ma

**Abstract**: Data attribution aims to quantify the contribution of individual training data points to the outputs of an AI model, which has been used to measure the value of training data and compensate data providers. Given the impact on financial decisions and compensation mechanisms, a critical question arises concerning the adversarial robustness of data attribution methods. However, there has been little to no systematic research addressing this issue. In this work, we aim to bridge this gap by detailing a threat model with clear assumptions about the adversary's goal and capabilities and proposing principled adversarial attack methods on data attribution. We present two methods, Shadow Attack and Outlier Attack, which generate manipulated datasets to inflate the compensation adversarially. The Shadow Attack leverages knowledge about the data distribution in the AI applications, and derives adversarial perturbations through "shadow training", a technique commonly used in membership inference attacks. In contrast, the Outlier Attack does not assume any knowledge about the data distribution and relies solely on black-box queries to the target model's predictions. It exploits an inductive bias present in many data attribution methods - outlier data points are more likely to be influential - and employs adversarial examples to generate manipulated datasets. Empirically, in image classification and text generation tasks, the Shadow Attack can inflate the data-attribution-based compensation by at least 200%, while the Outlier Attack achieves compensation inflation ranging from 185% to as much as 643%. Our implementation is ready at https://github.com/TRAIS-Lab/adversarial-attack-data-attribution.

摘要: 数据归因旨在量化单个训练数据点对人工智能模型输出的贡献，人工智能模型已用于衡量训练数据的价值并补偿数据提供商。鉴于对财务决策和薪酬机制的影响，出现了一个关于数据归因方法的对抗稳健性的关键问题。然而，几乎没有针对这个问题的系统性研究。在这项工作中，我们的目标是通过详细描述威胁模型来弥合这一差距，其中对对手的目标和能力做出明确假设，并提出关于数据属性的有原则的对抗攻击方法。我们提出了两种方法，影子攻击和离群点攻击，它们生成操纵数据集以不利地夸大补偿。影子攻击利用有关人工智能应用程序中数据分布的知识，并通过“影子训练”（一种常用于成员资格推理攻击的技术）来推导对抗性扰动。相比之下，离群点攻击不假设有关数据分布的任何知识，而是仅依赖于对目标模型预测的黑匣子查询。它利用了许多数据归因方法中存在的归纳偏差--离群数据点更有可能具有影响力--并利用对抗性示例来生成操纵的数据集。从经验上看，在图像分类和文本生成任务中，影子攻击可以将基于数据属性的补偿膨胀至少200%，而异常值攻击则实现了从185%到高达643%的补偿膨胀。我们的实施已在https://github.com/TRAIS-Lab/adversarial-attack-data-attribution上准备好。



## **15. PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation**

PoisonArena：揭露检索增强一代中的竞争中毒攻击 cs.IR

29 pages

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12574v1) [paper-pdf](http://arxiv.org/pdf/2505.12574v1)

**Authors**: Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong, Liang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems, widely used to improve the factual grounding of large language models (LLMs), are increasingly vulnerable to poisoning attacks, where adversaries inject manipulated content into the retriever's corpus. While prior research has predominantly focused on single-attacker settings, real-world scenarios often involve multiple, competing attackers with conflicting objectives. In this work, we introduce PoisonArena, the first benchmark to systematically study and evaluate competing poisoning attacks in RAG. We formalize the multi-attacker threat model, where attackers vie to control the answer to the same query using mutually exclusive misinformation. PoisonArena leverages the Bradley-Terry model to quantify each method's competitive effectiveness in such adversarial environments. Through extensive experiments on the Natural Questions and MS MARCO datasets, we demonstrate that many attack strategies successful in isolation fail under competitive pressure. Our findings highlight the limitations of conventional evaluation metrics like Attack Success Rate (ASR) and F1 score and underscore the need for competitive evaluation to assess real-world attack robustness. PoisonArena provides a standardized framework to benchmark and develop future attack and defense strategies under more realistic, multi-adversary conditions. Project page: https://github.com/yxf203/PoisonArena.

摘要: 检索增强生成（RAG）系统，广泛用于改善大型语言模型（LLM）的事实基础，越来越容易受到中毒攻击，其中对手将操纵的内容注入检索器的语料库。虽然以前的研究主要集中在单个攻击者的设置，但现实世界的场景往往涉及多个相互竞争的攻击者，这些攻击者的目标相互冲突。在这项工作中，我们介绍PoisonArena，第一个基准系统地研究和评估竞争中毒攻击在RAG。我们形式化的多攻击者威胁模型，攻击者争夺控制答案相同的查询使用互斥的错误信息。PoisonArena利用Bradley-Terry模型来量化每种方法在此类对抗环境中的竞争有效性。通过对Natural Questions和MS MARCO数据集的广泛实验，我们证明了许多孤立成功的攻击策略在竞争压力下失败。我们的研究结果强调了攻击成功率（SVR）和F1评分等传统评估指标的局限性，并强调了竞争性评估来评估现实世界攻击稳健性的必要性。PoisonArena提供了一个标准化的框架，可以在更现实的多对手条件下基准和开发未来的攻击和防御策略。项目页面：https://github.com/yxf203/PoisonArena。



## **16. A Survey of Attacks on Large Language Models**

大型语言模型攻击调查 cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12567v1) [paper-pdf](http://arxiv.org/pdf/2505.12567v1)

**Authors**: Wenrui Xu, Keshab K. Parhi

**Abstract**: Large language models (LLMs) and LLM-based agents have been widely deployed in a wide range of applications in the real world, including healthcare diagnostics, financial analysis, customer support, robotics, and autonomous driving, expanding their powerful capability of understanding, reasoning, and generating natural languages. However, the wide deployment of LLM-based applications exposes critical security and reliability risks, such as the potential for malicious misuse, privacy leakage, and service disruption that weaken user trust and undermine societal safety. This paper provides a systematic overview of the details of adversarial attacks targeting both LLMs and LLM-based agents. These attacks are organized into three phases in LLMs: Training-Phase Attacks, Inference-Phase Attacks, and Availability & Integrity Attacks. For each phase, we analyze the details of representative and recently introduced attack methods along with their corresponding defenses. We hope our survey will provide a good tutorial and a comprehensive understanding of LLM security, especially for attacks on LLMs. We desire to raise attention to the risks inherent in widely deployed LLM-based applications and highlight the urgent need for robust mitigation strategies for evolving threats.

摘要: 大型语言模型（LLM）和基于LLM的代理已被广泛部署在现实世界的广泛应用中，包括医疗诊断，财务分析，客户支持，机器人和自动驾驶，扩展了其强大的理解，推理和生成自然语言的能力。然而，基于LLM的应用程序的广泛部署暴露了关键的安全性和可靠性风险，例如恶意滥用、隐私泄露和服务中断的可能性，这些都会削弱用户信任并破坏社会安全。本文系统地概述了针对LLM和基于LLM的代理的对抗性攻击的细节。这些攻击在LLM中分为三个阶段：训练阶段攻击、推理阶段攻击和可用性和完整性攻击。对于每个阶段，我们都会分析代表性和最近引入的攻击方法及其相应防御的详细信息。我们希望我们的调查能够提供一个很好的教程和对LLM安全性的全面了解，尤其是对于LLM的攻击。我们希望引起人们对广泛部署的基于LLM的应用程序固有风险的关注，并强调迫切需要针对不断变化的威胁制定强有力的缓解策略。



## **17. An In-kernel Forensics Engine for Investigating Evasive Attacks**

用于调查规避攻击的内核内取证引擎 cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.06498v2) [paper-pdf](http://arxiv.org/pdf/2505.06498v2)

**Authors**: Javad Zandi, Lalchandra Rampersaud, Amin Kharraz

**Abstract**: Over the years, adversarial attempts against critical services have become more effective and sophisticated in launching low-profile attacks. This trend has always been concerning. However, an even more alarming trend is the increasing difficulty of collecting relevant evidence about these attacks and the involved threat actors in the early stages before significant damage is done. This issue puts defenders at a significant disadvantage, as it becomes exceedingly difficult to understand the attack details and formulate an appropriate response. Developing robust forensics tools to collect evidence about modern threats has never been easy. One main challenge is to provide a robust trade-off between achieving sufficient visibility while leaving minimal detectable artifacts. This paper will introduce LASE, an open-source Low-Artifact Forensics Engine to perform threat analysis and forensics in Windows operating system. LASE augments current analysis tools by providing detailed, system-wide monitoring capabilities while minimizing detectable artifacts. We designed multiple deployment scenarios, showing LASE's potential in evidence gathering and threat reasoning in a real-world setting. By making LASE and its execution trace data available to the broader research community, this work encourages further exploration in the field by reducing the engineering costs for threat analysis and building a longitudinal behavioral analysis catalog for diverse security domains.

摘要: 多年来，针对关键服务的对抗尝试在发起低调攻击方面变得更加有效和复杂。这种趋势一直令人担忧。然而，一个更令人震惊的趋势是，在造成重大损害之前的早期阶段收集有关这些攻击和相关威胁行为者的相关证据的难度越来越大。这个问题使防御者处于明显的劣势，因为了解攻击细节并制定适当的应对措施变得极其困难。开发强大的取证工具来收集有关现代威胁的证据从来都不是一件容易的事。一个主要挑战是在实现足够的可见性同时留下最少的可检测伪影之间提供稳健的权衡。本文将介绍LASE，这是一个开源的低功耗取证引擎，用于在Windows操作系统中执行威胁分析和取证。LASE通过提供详细的系统范围监控功能，同时最大限度地减少可检测的伪影来增强当前的分析工具。我们设计了多种部署场景，展示了LASE在现实环境中证据收集和威胁推理方面的潜力。通过向更广泛的研究界提供LASE及其执行跟踪数据，这项工作通过降低威胁分析的工程成本并为不同安全领域构建纵向行为分析目录来鼓励该领域的进一步探索。



## **18. IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems**

针对基于LLM的多代理系统的IP泄露攻击 cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12442v1) [paper-pdf](http://arxiv.org/pdf/2505.12442v1)

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.

摘要: 大型语言模型（LLM）的快速发展导致了通过协作执行复杂任务的多智能体系统（MAS）的出现。然而，MAS的复杂性质，包括其架构和代理交互，引发了有关知识产权（IP）保护的严重担忧。本文介绍MASLEAK，这是一种新型攻击框架，旨在从MAS应用程序中提取敏感信息。MASLEAK针对的是实用的黑匣子设置，其中对手不了解MAS架构或代理配置。对手只能通过其公共API与MAS交互，提交攻击查询$q$并观察最终代理的输出。受计算机蠕虫传播和感染脆弱网络主机的方式的启发，MASLEAK精心设计了对抗性查询$q$，以引发、传播和保留每个MAS代理的响应，这些响应揭示了全套专有组件，包括代理数量、系统布局、系统提示、任务指令和工具使用。我们构建了包含810个应用程序的第一个MAS应用程序合成数据集，并根据现实世界的MAS应用程序（包括Coze和CrewAI）评估MASLEAK。MASLEAK在提取MAS IP方面实现了高准确性，系统提示和任务指令的平均攻击成功率为87%，大多数情况下系统架构的平均攻击成功率为92%。最后，我们讨论了我们发现的影响和潜在的防御措施。



## **19. CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement**

捕获：上下文感知提示注入测试和稳健性增强 cs.CL

Accepted in ACL LLMSec Workshop 2025

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12368v1) [paper-pdf](http://arxiv.org/pdf/2505.12368v1)

**Authors**: Gauri Kholkar, Ratinder Ahuja

**Abstract**: Prompt injection remains a major security risk for large language models. However, the efficacy of existing guardrail models in context-aware settings remains underexplored, as they often rely on static attack benchmarks. Additionally, they have over-defense tendencies. We introduce CAPTURE, a novel context-aware benchmark assessing both attack detection and over-defense tendencies with minimal in-domain examples. Our experiments reveal that current prompt injection guardrail models suffer from high false negatives in adversarial cases and excessive false positives in benign scenarios, highlighting critical limitations.

摘要: 提示注入仍然是大型语言模型的主要安全风险。然而，现有护栏模型在上下文感知环境中的功效仍然没有得到充分的探索，因为它们通常依赖于静态攻击基准。此外，他们还有过度防御的倾向。我们引入了CAPTURE，这是一种新型的上下文感知基准，通过最少的领域内示例来评估攻击检测和过度防御倾向。我们的实验表明，当前的即时注射护栏模型在对抗性情况下存在高假阴性，在良性情况下存在过多假阳性，凸显了严重的局限性。



## **20. The Tower of Babel Revisited: Multilingual Jailbreak Prompts on Closed-Source Large Language Models**

重访巴别塔：多语言越狱依赖闭源大型语言模型 cs.CL

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12287v1) [paper-pdf](http://arxiv.org/pdf/2505.12287v1)

**Authors**: Linghan Huang, Haolin Jin, Zhaoge Bi, Pengyue Yang, Peizhou Zhao, Taozhao Chen, Xiongfei Wu, Lei Ma, Huaming Chen

**Abstract**: Large language models (LLMs) have seen widespread applications across various domains, yet remain vulnerable to adversarial prompt injections. While most existing research on jailbreak attacks and hallucination phenomena has focused primarily on open-source models, we investigate the frontier of closed-source LLMs under multilingual attack scenarios. We present a first-of-its-kind integrated adversarial framework that leverages diverse attack techniques to systematically evaluate frontier proprietary solutions, including GPT-4o, DeepSeek-R1, Gemini-1.5-Pro, and Qwen-Max. Our evaluation spans six categories of security contents in both English and Chinese, generating 38,400 responses across 32 types of jailbreak attacks. Attack success rate (ASR) is utilized as the quantitative metric to assess performance from three dimensions: prompt design, model architecture, and language environment. Our findings suggest that Qwen-Max is the most vulnerable, while GPT-4o shows the strongest defense. Notably, prompts in Chinese consistently yield higher ASRs than their English counterparts, and our novel Two-Sides attack technique proves to be the most effective across all models. This work highlights a dire need for language-aware alignment and robust cross-lingual defenses in LLMs, and we hope it will inspire researchers, developers, and policymakers toward more robust and inclusive AI systems.

摘要: 大型语言模型（LLM）已经在各个领域得到了广泛的应用，但仍然容易受到对抗性提示注入的影响。虽然大多数关于越狱攻击和幻觉现象的现有研究主要集中在开源模型上，但我们研究了多语言攻击场景下闭源LLM的前沿。我们提出了一个首创的集成对抗框架，该框架利用各种攻击技术来系统地评估前沿专有解决方案，包括GPT-4 o，DeepSeek-R1，Gemini-1.5-Pro和Qwen-Max。我们的评估涵盖了中英文六类安全内容，针对32种越狱攻击生成了38，400个响应。攻击成功率（ASB）被用作量化指标，从三个维度评估性能：提示设计、模型架构和语言环境。我们的研究结果表明，Qwen-Max最脆弱，而GPT-4 o表现出最强的防御能力。值得注意的是，中文提示始终比英语提示产生更高的ASB，而且我们新颖的双面攻击技术被证明是所有模型中最有效的。这项工作凸显了LLM中对语言感知一致和强大的跨语言防御的迫切需求，我们希望它能够激励研究人员、开发人员和政策制定者开发更强大、更具包容性的人工智能系统。



## **21. `Do as I say not as I do': A Semi-Automated Approach for Jailbreak Prompt Attack against Multimodal LLMs**

“照我说的做，而不是照我做的做”：针对多模式LLM的越狱提示攻击的半自动方法 cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2502.00735v3) [paper-pdf](http://arxiv.org/pdf/2502.00735v3)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen, Kim-Kwang Raymond Choo

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the state-of-the-art multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. Further, to better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flanking Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios.

摘要: 大型语言模型（LLM）由于处理文本、音频、图像和视频等多种类型输入数据的能力不断增强，已在各个领域得到广泛应用。虽然LLM在理解和生成不同场景的上下文方面表现出出色的性能，但它们很容易受到基于预算的攻击（主要通过文本输入）。本文中，我们介绍了针对多模式LLM的第一个基于语音的越狱攻击，称为侧翼攻击，它可以同时处理针对多模式LLM的不同类型的输入。我们的工作受到单语语音驱动的大型语言模型的最新进展的推动，这些模型在LLM的传统基于文本的漏洞之外引入了新的攻击表面。为了调查这些风险，我们研究了最先进的多模式LLM，它们可以通过音频输入等不同类型的输入访问，重点关注对抗性提示如何绕过其防御机制。我们提出了一种新颖的策略，其中不允许的提示两侧是良性的、叙述驱动的提示。它集成在侧翼攻击中，该攻击试图将交互上下文人性化并通过虚构的环境执行攻击。此外，为了更好地评估攻击性能，我们提出了一个用于策略违规检测的半自动自我评估框架。我们证明了侧翼攻击能够操纵最先进的LLM生成不对齐和禁止的输出，在七种禁止的情况下，平均攻击成功率从0.67到0.93不等。



## **22. Self-Destructive Language Model**

自毁语言模型 cs.LG

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12186v1) [paper-pdf](http://arxiv.org/pdf/2505.12186v1)

**Authors**: Yuhui Wang, Rongyi Zhu, Ting Wang

**Abstract**: Harmful fine-tuning attacks pose a major threat to the security of large language models (LLMs), allowing adversaries to compromise safety guardrails with minimal harmful data. While existing defenses attempt to reinforce LLM alignment, they fail to address models' inherent "trainability" on harmful data, leaving them vulnerable to stronger attacks with increased learning rates or larger harmful datasets. To overcome this critical limitation, we introduce SEAM, a novel alignment-enhancing defense that transforms LLMs into self-destructive models with intrinsic resilience to misalignment attempts. Specifically, these models retain their capabilities for legitimate tasks while exhibiting substantial performance degradation when fine-tuned on harmful data. The protection is achieved through a novel loss function that couples the optimization trajectories of benign and harmful data, enhanced with adversarial gradient ascent to amplify the self-destructive effect. To enable practical training, we develop an efficient Hessian-free gradient estimate with theoretical error bounds. Extensive evaluation across LLMs and datasets demonstrates that SEAM creates a no-win situation for adversaries: the self-destructive models achieve state-of-the-art robustness against low-intensity attacks and undergo catastrophic performance collapse under high-intensity attacks, rendering them effectively unusable. (warning: this paper contains potentially harmful content generated by LLMs.)

摘要: 有害的微调攻击对大型语言模型（LLM）的安全性构成了重大威胁，使攻击者能够以最小的有害数据破坏安全护栏。虽然现有的防御措施试图加强LLM对齐，但它们未能解决模型在有害数据上固有的“可训练性”，使它们容易受到学习率提高或更大的有害数据集的更强攻击。为了克服这一关键限制，我们引入了SEAM，这是一种新的增强防御机制，它将LLM转换为具有内在弹性的自毁模型，以应对未对准尝试。具体来说，这些模型保留了合法任务的能力，同时在对有害数据进行微调时表现出显著的性能下降。这种保护是通过一种新的损失函数来实现的，该函数将良性和有害数据的优化轨迹结合起来，通过对抗性梯度上升来增强，以放大自毁效应。为了实现实际训练，我们开发了一个有效的Hessian自由梯度估计与理论误差界。对LLM和数据集的广泛评估表明，SEAM为对手创造了一个没有胜利的局面：自毁模型对低强度攻击具有最先进的鲁棒性，并在高强度攻击下经历灾难性的性能崩溃，使它们实际上无法使用。（警告：本文包含由LLM生成的潜在有害内容。



## **23. EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective**

EVALOOP：从自我一致性的角度评估LLM编程稳健性 cs.SE

19 pages, 11 figures

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12185v1) [paper-pdf](http://arxiv.org/pdf/2505.12185v1)

**Authors**: Sen Fang, Weiyuan Ding, Bowen Xu

**Abstract**: Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.

摘要: 评估大型语言模型（LLM）的编程能力对于它们在软件工程中的有效使用至关重要。然而，当前的评估主要衡量静态基准上生成的代码的准确性，忽视了编程任务期间模型稳健性的关键方面。虽然对抗性攻击提供了有关模型稳健性的见解，但它们的有效性有限，并且评估可能会受到限制。当前用于稳健性评估的对抗攻击方法会产生不一致的结果，难以在不同的LLM之间提供统一的评估。我们引入EVALOOP，这是一种新型评估框架，从自一致性的角度评估稳健性，即利用流行软件工程任务中固有的自然二重性，例如，代码生成和代码摘要。EVALOOP启动独立反馈循环：LLM生成输出（例如，代码）来自输入（例如，自然语言规范），然后使用生成的输出作为输入来产生新的输出（例如，将该代码总结为新规范）。EVALOOP重复该过程以评估每个循环中EVALOOP的有效性。这种循环策略本质上评估稳健性，而不依赖任何外部攻击设置，提供了一个统一的指标来评估LLM在编程中的稳健性。我们评估了16个著名的LLM（例如，GPT-4.1，O 4-mini）在EVALOOP上发现EVALOOP通常会在十个循环内导致pass@1性能绝对下降5.01%-19.31%。有趣的是，稳健性并不总是与初始性能一致（即，一次性查询）;例如，GPT-3.5-Turbo尽管初始代码生成优于DeepSeek-V2，但在重复评估循环中表现出较低的鲁棒性。



## **24. ImF: Implicit Fingerprint for Large Language Models**

ImF：大型语言模型的隐式指纹 cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2503.21805v2) [paper-pdf](http://arxiv.org/pdf/2503.21805v2)

**Authors**: Wu jiaxuan, Peng Wanli, Fu hang, Xue Yiming, Wen juan

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making protecting intellectual property (IP) for LLMs crucial. Recently, embedding fingerprints into LLMs has emerged as a prevalent method for establishing model ownership. However, existing fingerprinting techniques typically embed identifiable patterns with weak semantic coherence, resulting in fingerprints that significantly differ from the natural question-answering (QA) behavior inherent to LLMs. This discrepancy undermines the stealthiness of the embedded fingerprints and makes them vulnerable to adversarial attacks. In this paper, we first demonstrate the critical vulnerability of existing fingerprint embedding methods by introducing a novel adversarial attack named Generation Revision Intervention (GRI) attack. GRI attack exploits the semantic fragility of current fingerprinting methods, effectively erasing fingerprints by disrupting their weakly correlated semantic structures. Our empirical evaluation highlights that traditional fingerprinting approaches are significantly compromised by the GRI attack, revealing severe limitations in their robustness under realistic adversarial conditions. To advance the state-of-the-art in model fingerprinting, we propose a novel model fingerprint paradigm called Implicit Fingerprints (ImF). ImF leverages steganography techniques to subtly embed ownership information within natural texts, subsequently using Chain-of-Thought (CoT) prompting to construct semantically coherent and contextually natural QA pairs. This design ensures that fingerprints seamlessly integrate with the standard model behavior, remaining indistinguishable from regular outputs and substantially reducing the risk of accidental triggering and targeted removal. We conduct a comprehensive evaluation of ImF on 15 diverse LLMs, spanning different architectures and varying scales.

摘要: 训练大型语言模型（LLM）是资源密集型且昂贵的，因此保护LLM的知识产权（IP）至关重要。最近，将指纹嵌入LLM已成为建立模型所有权的流行方法。然而，现有的指纹识别技术通常嵌入具有弱语义一致性的可识别模式，导致指纹与LLM固有的自然问答（QA）行为显着不同。这种差异削弱了嵌入指纹的隐蔽性，并使它们容易受到对抗攻击。在本文中，我们首先证明了现有的指纹嵌入方法的关键漏洞，通过引入一种新的对抗性攻击，称为生成修订干预（GRI）攻击。GRI攻击利用了当前指纹识别方法的语义脆弱性，通过破坏指纹的弱相关语义结构来有效地擦除指纹。我们的经验评估强调，传统的指纹识别方法受到GRI攻击的严重影响，在现实的对抗条件下，它们的鲁棒性存在严重的局限性。为了推进国家的最先进的模型指纹，我们提出了一种新的模型指纹范例称为隐式指纹（ImF）。ImF利用隐写技术将所有权信息巧妙地嵌入自然文本中，随后使用思想链（CoT）提示构建语义连贯且上下文自然的QA对。这种设计确保指纹与标准模型行为无缝集成，与常规输出保持无区别，并大幅降低意外触发和有针对性删除的风险。我们对15个不同的LLM进行了ImF的全面评估，涵盖不同的架构和不同的规模。



## **25. FABLE: A Localized, Targeted Adversarial Attack on Weather Forecasting Models**

寓言：对天气预报模型的局部化、有针对性的对抗攻击 cs.LG

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.12167v1) [paper-pdf](http://arxiv.org/pdf/2505.12167v1)

**Authors**: Yue Deng, Asadullah Hill Galib, Xin Lan, Pang-Ning Tan, Lifeng Luo

**Abstract**: Deep learning-based weather forecasting models have recently demonstrated significant performance improvements over gold-standard physics-based simulation tools. However, these models are vulnerable to adversarial attacks, which raises concerns about their trustworthiness. In this paper, we first investigate the feasibility of applying existing adversarial attack methods to weather forecasting models. We argue that a successful attack should (1) not modify significantly its original inputs, (2) be faithful, i.e., achieve the desired forecast at targeted locations with minimal changes to non-targeted locations, and (3) be geospatio-temporally realistic. However, balancing these criteria is a challenge as existing methods are not designed to preserve the geospatio-temporal dependencies of the original samples. To address this challenge, we propose a novel framework called FABLE (Forecast Alteration By Localized targeted advErsarial attack), which employs a 3D discrete wavelet decomposition to extract the varying components of the geospatio-temporal data. By regulating the magnitude of adversarial perturbations across different components, FABLE can generate adversarial inputs that maintain geospatio-temporal coherence while remaining faithful and closely aligned with the original inputs. Experimental results on multiple real-world datasets demonstrate the effectiveness of our framework over baseline methods across various metrics.

摘要: 基于深度学习的天气预报模型最近表现出比金标准的基于物理的模拟工具显着的性能改进。然而，这些模型很容易受到对抗攻击，这引发了对其可信度的担忧。在本文中，我们首先研究将现有的对抗攻击方法应用于天气预报模型的可行性。我们认为，成功的攻击应该（1）不显着修改其原始输入，（2）忠实，即，在目标位置实现所需的预测，对非目标位置的改变最小，以及（3）具有地理时空现实性。然而，平衡这些标准是一个挑战，因为现有方法的设计并不是为了保留原始样本的地理时空依赖性而设计的。为了应对这一挑战，我们提出了一种名为FABLE（通过本地化有针对性的广告攻击预测Alteration）的新型框架，该框架采用3D离散子波分解来提取地理时空数据的不同成分。通过调节不同组件之间对抗性扰动的幅度，FABLE可以生成对抗性输入，这些输入保持地理时空一致性，同时保持忠实并与原始输入密切一致。多个现实世界数据集的实验结果证明了我们的框架相对于各种指标的基线方法的有效性。



## **26. Black-box Adversaries from Latent Space: Unnoticeable Attacks on Human Pose and Shape Estimation**

来自潜伏空间的黑匣子对手：对人体姿势和形状估计的不明显攻击 cs.CV

17 pages, 6 figures

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.12009v1) [paper-pdf](http://arxiv.org/pdf/2505.12009v1)

**Authors**: Zhiying Li, Guanggang Geng, Yeying Jin, Zhizhi Guo, Bruce Gu, Jidong Huo, Zhaoxin Fan, Wenjun Wu

**Abstract**: Expressive human pose and shape (EHPS) estimation is vital for digital human generation, particularly in live-streaming applications. However, most existing EHPS models focus primarily on minimizing estimation errors, with limited attention on potential security vulnerabilities. Current adversarial attacks on EHPS models often require white-box access (e.g., model details or gradients) or generate visually conspicuous perturbations, limiting their practicality and ability to expose real-world security threats. To address these limitations, we propose a novel Unnoticeable Black-Box Attack (UBA) against EHPS models. UBA leverages the latent-space representations of natural images to generate an optimal adversarial noise pattern and iteratively refine its attack potency along an optimized direction in digital space. Crucially, this process relies solely on querying the model's output, requiring no internal knowledge of the EHPS architecture, while guiding the noise optimization toward greater stealth and effectiveness. Extensive experiments and visual analyses demonstrate the superiority of UBA. Notably, UBA increases the pose estimation errors of EHPS models by 17.27%-58.21% on average, revealing critical vulnerabilities. These findings underscore the urgent need to address and mitigate security risks associated with digital human generation systems.

摘要: 表现性人体姿势和形状（EHPS）估计对于数字人类生成至关重要，特别是在直播应用中。然而，大多数现有的EHPS模型主要关注于最大限度地减少估计误差，而对潜在的安全漏洞的关注有限。当前对EHPS模型的对抗攻击通常需要白盒访问（例如，模型细节或梯度）或生成视觉上明显的扰动，限制了其实用性和暴露现实世界安全威胁的能力。为了解决这些限制，我们提出了一种针对EHPS模型的新型不明显黑匣子攻击（UBA）。UBA利用自然图像的潜空间表示来生成最佳对抗性噪音模式，并沿着数字空间中的优化方向迭代地细化其攻击能力。至关重要的是，该过程仅依赖于查询模型的输出，不需要EHPS架构的内部知识，同时引导噪音优化实现更大的隐蔽性和有效性。大量的实验和视觉分析证明了UBA的优越性。值得注意的是，UBA将EHPS模型的位姿估计误差平均增加了17.27%-58.21%，揭示了关键漏洞。这些发现凸显了解决和减轻与数字人类生成系统相关的安全风险的迫切需要。



## **27. Adversarial Attacks on Both Face Recognition and Face Anti-spoofing Models**

对人脸识别和人脸反欺骗模型的对抗攻击 cs.CV

Proceedings of the 34th International Joint Conference on Artificial  Intelligence (IJCAI 2025)

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2405.16940v2) [paper-pdf](http://arxiv.org/pdf/2405.16940v2)

**Authors**: Fengfan Zhou, Qianyu Zhou, Hefei Ling, Xuequan Lu

**Abstract**: Adversarial attacks on Face Recognition (FR) systems have demonstrated significant effectiveness against standalone FR models. However, their practicality diminishes in complete FR systems that incorporate Face Anti-Spoofing (FAS) models, as these models can detect and mitigate a substantial number of adversarial examples. To address this critical yet under-explored challenge, we introduce a novel attack setting that targets both FR and FAS models simultaneously, thereby enhancing the practicability of adversarial attacks on integrated FR systems. Specifically, we propose a new attack method, termed Reference-free Multi-level Alignment (RMA), designed to improve the capacity of black-box attacks on both FR and FAS models. The RMA framework is built upon three key components. Firstly, we propose an Adaptive Gradient Maintenance module to address the imbalances in gradient contributions between FR and FAS models. Secondly, we develop a Reference-free Intermediate Biasing module to improve the transferability of adversarial examples against FAS models. In addition, we introduce a Multi-level Feature Alignment module to reduce feature discrepancies at various levels of representation. Extensive experiments showcase the superiority of our proposed attack method to state-of-the-art adversarial attacks.

摘要: 对面部识别（FR）系统的对抗攻击已经证明了对独立的FR模型的显着有效性。然而，它们的实用性在包含面部反欺骗（FAA）模型的完整FR系统中会减弱，因为这些模型可以检测和减轻大量对抗示例。为了解决这一关键但尚未充分探索的挑战，我们引入了一种同时针对FR和FAA模型的新型攻击设置，从而增强了对集成FR系统进行对抗攻击的实用性。具体来说，我们提出了一种新的攻击方法，称为无引用多层对齐（LMA），旨在提高对FR和FAA模型的黑匣子攻击能力。LMA框架建立在三个关键组件之上。首先，我们提出了一个自适应梯度维护模块来解决FR和FAA模型之间梯度贡献的不平衡问题。其次，我们开发了一个无参考中间偏置模块，以提高对抗性示例针对FAA模型的可移植性。此外，我们还引入了多级别特征对齐模块，以减少各个表示级别的特征差异。大量的实验展示了我们提出的攻击方法相对于最先进的对抗攻击的优越性。



## **28. Understanding and Enhancing the Transferability of Jailbreaking Attacks**

了解并增强越狱攻击的可转移性 cs.LG

Accepted by ICLR 2025

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2502.03052v2) [paper-pdf](http://arxiv.org/pdf/2502.03052v2)

**Authors**: Runqi Lin, Bo Han, Fengwang Li, Tongling Liu

**Abstract**: Jailbreaking attacks can effectively manipulate open-source large language models (LLMs) to produce harmful responses. However, these attacks exhibit limited transferability, failing to disrupt proprietary LLMs consistently. To reliably identify vulnerabilities in proprietary LLMs, this work investigates the transferability of jailbreaking attacks by analysing their impact on the model's intent perception. By incorporating adversarial sequences, these attacks can redirect the source LLM's focus away from malicious-intent tokens in the original input, thereby obstructing the model's intent recognition and eliciting harmful responses. Nevertheless, these adversarial sequences fail to mislead the target LLM's intent perception, allowing the target LLM to refocus on malicious-intent tokens and abstain from responding. Our analysis further reveals the inherent distributional dependency within the generated adversarial sequences, whose effectiveness stems from overfitting the source LLM's parameters, resulting in limited transferability to target LLMs. To this end, we propose the Perceived-importance Flatten (PiF) method, which uniformly disperses the model's focus across neutral-intent tokens in the original input, thus obscuring malicious-intent tokens without relying on overfitted adversarial sequences. Extensive experiments demonstrate that PiF provides an effective and efficient red-teaming evaluation for proprietary LLMs.

摘要: 越狱攻击可以有效地操纵开源大型语言模型（LLM）以产生有害响应。然而，这些攻击的可转让性有限，无法一致破坏专有LLM。为了可靠地识别专有LLM中的漏洞，这项工作通过分析越狱攻击对模型意图感知的影响来研究越狱攻击的可转移性。通过结合对抗序列，这些攻击可以将源LLM的焦点从原始输入中的恶意意图标记重新定向，从而阻碍模型的意图识别并引发有害响应。然而，这些对抗序列未能误导目标LLM的意图感知，从而允许目标LLM重新关注恶意意图代币并放弃回应。我们的分析进一步揭示了生成的对抗序列内固有的分布依赖性，其有效性源于过度匹配源LLM的参数，导致目标LLM的可移植性有限。为此，我们提出了感知重要性拉平（PiF）方法，该方法将模型的焦点均匀分散到原始输入中的中立意图标记上，从而在不依赖过度匹配的对抗序列的情况下模糊恶意意图标记。大量实验表明，PiF为专有LLM提供了有效且高效的红色团队评估。



## **29. Robust Deep Reinforcement Learning against Adversarial Behavior Manipulation**

对抗对抗行为操纵的稳健深度强化学习 cs.LG

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2406.03862v2) [paper-pdf](http://arxiv.org/pdf/2406.03862v2)

**Authors**: Shojiro Yamabe, Kazuto Fukuchi, Jun Sakuma

**Abstract**: This study investigates behavior-targeted attacks on reinforcement learning and their countermeasures. Behavior-targeted attacks aim to manipulate the victim's behavior as desired by the adversary through adversarial interventions in state observations. Existing behavior-targeted attacks have some limitations, such as requiring white-box access to the victim's policy. To address this, we propose a novel attack method using imitation learning from adversarial demonstrations, which works under limited access to the victim's policy and is environment-agnostic. In addition, our theoretical analysis proves that the policy's sensitivity to state changes impacts defense performance, particularly in the early stages of the trajectory. Based on this insight, we propose time-discounted regularization, which enhances robustness against attacks while maintaining task performance. To the best of our knowledge, this is the first defense strategy specifically designed for behavior-targeted attacks.

摘要: 本研究调查了针对强化学习的行为攻击及其对策。以行为为目标的攻击旨在通过状态观察中的对抗干预，按照对手的意愿操纵受害者的行为。现有的针对行为的攻击有一些局限性，例如需要白盒访问受害者的策略。为了解决这个问题，我们提出了一种新颖的攻击方法，使用对抗性演示中的模仿学习，该方法在有限的访问受害者政策的情况下工作，并且是环境不可知的。此外，我们的理论分析证明，政策对状态变化的敏感性会影响国防绩效，特别是在轨迹的早期阶段。基于这一见解，我们提出了时间折扣正规化，这在保持任务性能的同时增强了针对攻击的鲁棒性。据我们所知，这是第一个专门为针对行为的攻击而设计的防御策略。



## **30. Adversarial Robustness for Unified Multi-Modal Encoders via Efficient Calibration**

通过高效校准实现统一多模式编码器的对抗鲁棒性 cs.CV

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.11895v1) [paper-pdf](http://arxiv.org/pdf/2505.11895v1)

**Authors**: Chih-Ting Liao, Bin Ren, Guofeng Mei, Xu Zheng

**Abstract**: Recent unified multi-modal encoders align a wide range of modalities into a shared representation space, enabling diverse cross-modal tasks. Despite their impressive capabilities, the robustness of these models under adversarial perturbations remains underexplored, which is a critical concern for safety-sensitive applications. In this work, we present the first comprehensive study of adversarial vulnerability in unified multi-modal encoders. We find that even mild adversarial perturbations lead to substantial performance drops across all modalities. Non-visual inputs, such as audio and point clouds, are especially fragile, while visual inputs like images and videos also degrade significantly. To address this, we propose an efficient adversarial calibration framework that improves robustness across modalities without modifying pretrained encoders or semantic centers, ensuring compatibility with existing foundation models. Our method introduces modality-specific projection heads trained solely on adversarial examples, while keeping the backbone and embeddings frozen. We explore three training objectives: fixed-center cross-entropy, clean-to-adversarial L2 alignment, and clean-adversarial InfoNCE, and we introduce a regularization strategy to ensure modality-consistent alignment under attack. Experiments on six modalities and three Bind-style models show that our method improves adversarial robustness by up to 47.3 percent at epsilon = 4/255, while preserving or even improving clean zero-shot and retrieval performance with less than 1 percent trainable parameters.

摘要: 最近的统一多模式编码器将广泛的模式对齐到共享的表示空间中，从而实现多样化的跨模式任务。尽管这些模型的能力令人印象深刻，但在对抗性扰动下的稳健性仍然没有得到充分的研究，这是安全敏感应用程序的一个关键问题。在这项工作中，我们首次对统一多模式编码器中的对抗脆弱性进行了全面研究。我们发现，即使是轻微的对抗扰动，也会导致所有模式的性能大幅下降。音频和点云等非视觉输入尤其脆弱，而图像和视频等视觉输入也会显着退化。为了解决这个问题，我们提出了一种高效的对抗性校准框架，该框架在无需修改预训练的编码器或语义中心的情况下提高了各个模式的鲁棒性，确保与现有基础模型的兼容性。我们的方法引入了仅在对抗性示例上训练的特定模式投影头，同时保持主干和嵌入冻结。我们探索了三个训练目标：固定中心交叉熵、干净对抗L2对齐和干净对抗InfoNSO，并引入了一种正规化策略来确保在攻击下的模式一致对齐。对六种模式和三种绑定风格模型的实验表明，我们的方法在RST = 4/255时将对抗鲁棒性提高了高达47.3%，同时保留甚至改进了干净的零射击和检索性能，可训练参数少于1%。



## **31. SynFuzz: Leveraging Fuzzing of Netlist to Detect Synthesis Bugs**

SynFuzz：利用网表的模糊化来检测合成错误 cs.CR

15 pages, 10 figures, 5 tables

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2504.18812v2) [paper-pdf](http://arxiv.org/pdf/2504.18812v2)

**Authors**: Raghul Saravanan, Sudipta Paria, Aritra Dasgupta, Venkat Nitin Patnala, Swarup Bhunia, Sai Manoj P D

**Abstract**: In the evolving landscape of integrated circuit (IC) design, the increasing complexity of modern processors and intellectual property (IP) cores has introduced new challenges in ensuring design correctness and security. The recent advancements in hardware fuzzing techniques have shown their efficacy in detecting hardware bugs and vulnerabilities at the RTL abstraction level of hardware. However, they suffer from several limitations, including an inability to address vulnerabilities introduced during synthesis and gate-level transformations. These methods often fail to detect issues arising from library adversaries, where compromised or malicious library components can introduce backdoors or unintended behaviors into the design. In this paper, we present a novel hardware fuzzer, SynFuzz, designed to overcome the limitations of existing hardware fuzzing frameworks. SynFuzz focuses on fuzzing hardware at the gate-level netlist to identify synthesis bugs and vulnerabilities that arise during the transition from RTL to the gate-level. We analyze the intrinsic hardware behaviors using coverage metrics specifically tailored for the gate-level. Furthermore, SynFuzz implements differential fuzzing to uncover bugs associated with EDA libraries. We evaluated SynFuzz on popular open-source processors and IP designs, successfully identifying 7 new synthesis bugs. Additionally, by exploiting the optimization settings of EDA tools, we performed a compromised library mapping attack (CLiMA), creating a malicious version of hardware designs that remains undetectable by traditional verification methods. We also demonstrate how SynFuzz overcomes the limitations of the industry-standard formal verification tool, Cadence Conformal, providing a more robust and comprehensive approach to hardware verification.

摘要: 在集成电路（IC）设计不断发展的格局中，现代处理器和知识产权（IP）核的复杂性日益增加，为确保设计正确性和安全性带来了新的挑战。硬件模糊技术的最新进展表明，它们在硬件RTL抽象级别检测硬件错误和漏洞方面的功效。然而，它们存在一些限制，包括无法解决合成和门级转换期间引入的漏洞。这些方法通常无法检测到库对手引起的问题，其中受损害或恶意的库组件可能会在设计中引入后门或意外行为。在本文中，我们提出了一种新型的硬件模糊器SynFuzz，旨在克服现有硬件模糊框架的局限性。SynFuzz专注于在门级网表上模糊硬件，以识别从RTL过渡到门级过程中出现的合成错误和漏洞。我们使用专门为门户级定制的覆盖指标来分析固有的硬件行为。此外，SynFuzz还实现了差异模糊化来发现与EDA库相关的错误。我们在流行的开源处理器和IP设计上评估了SynFuzz，成功识别出7个新的合成错误。此外，通过利用EDA工具的优化设置，我们执行了受损库映射攻击（CLiMA），创建了传统验证方法仍然无法检测到的硬件设计的恶意版本。我们还展示了SynFuzz如何克服行业标准形式验证工具Cadence Conformal的局限性，为硬件验证提供更稳健、更全面的方法。



## **32. Adversarial Attacks of Vision Tasks in the Past 10 Years: A Survey**

过去10年视觉任务的对抗性攻击：一项调查 cs.CV

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2410.23687v2) [paper-pdf](http://arxiv.org/pdf/2410.23687v2)

**Authors**: Chiyu Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, Zhe Liu

**Abstract**: With the advent of Large Vision-Language Models (LVLMs), new attack vectors, such as cognitive bias, prompt injection, and jailbreaking, have emerged. Understanding these attacks promotes system robustness improvement and neural networks demystification. However, existing surveys often target attack taxonomy and lack in-depth analysis like 1) unified insights into adversariality, transferability, and generalization; 2) detailed evaluations framework; 3) motivation-driven attack categorizations; and 4) an integrated perspective on both traditional and LVLM attacks. This article addresses these gaps by offering a thorough summary of traditional and LVLM adversarial attacks, emphasizing their connections and distinctions, and providing actionable insights for future research.

摘要: 随着大型视觉语言模型（LVLM）的出现，出现了新的攻击向量，如认知偏差，即时注入和越狱。了解这些攻击有助于提高系统的鲁棒性和神经网络的神秘性。然而，现有的调查通常针对攻击分类，缺乏深入的分析，如1）对对抗性，可转移性和泛化的统一见解; 2）详细的评估框架; 3）动机驱动的攻击分类;以及4）对传统和LVLM攻击的综合观点。本文通过全面总结传统和LVLM对抗性攻击来解决这些差距，强调它们的联系和区别，并为未来的研究提供可操作的见解。



## **33. Kick Bad Guys Out! Conditionally Activated Anomaly Detection in Federated Learning with Zero-Knowledge Proof Verification**

把坏人踢出去！具有零知识证明验证的联邦学习中的一致激活异常检测 cs.CR

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2310.04055v5) [paper-pdf](http://arxiv.org/pdf/2310.04055v5)

**Authors**: Shanshan Han, Wenxuan Wu, Baturalp Buyukates, Weizhao Jin, Qifan Zhang, Yuhang Yao, Salman Avestimehr, Chaoyang He

**Abstract**: Federated Learning (FL) systems are vulnerable to adversarial attacks, such as model poisoning and backdoor attacks. However, existing defense mechanisms often fall short in real-world settings due to key limitations: they may rely on impractical assumptions, introduce distortions by modifying aggregation functions, or degrade model performance even in benign scenarios. To address these issues, we propose a novel anomaly detection method designed specifically for practical FL scenarios. Our approach employs a two-stage, conditionally activated detection mechanism: cross-round check first detects whether suspicious activity has occurred, and, if warranted, a cross-client check filters out malicious participants. This mechanism preserves utility while avoiding unrealistic assumptions. Moreover, to ensure the transparency and integrity of the defense mechanism, we incorporate zero-knowledge proofs, enabling clients to verify the detection without relying solely on the server's goodwill. To the best of our knowledge, this is the first method to bridge the gap between theoretical advances in FL security and the demands of real-world deployment. Extensive experiments across diverse tasks and real-world edge devices demonstrate the effectiveness of our method over state-of-the-art defenses.

摘要: 联邦学习（FL）系统容易受到对抗攻击，例如模型中毒和后门攻击。然而，由于关键限制，现有的防御机制在现实世界环境中往往表现不佳：它们可能依赖于不切实际的假设，通过修改聚合函数引入失真，或者即使在良性情况下也会降低模型性能。为了解决这些问题，我们提出了一种专门针对实际FL场景设计的新型异常检测方法。我们的方法采用两阶段、有条件激活的检测机制：跨轮检查首先检测是否发生可疑活动，如果有必要，跨客户端检查过滤出恶意参与者。该机制保留了实用性，同时避免了不切实际的假设。此外，为了确保防御机制的透明度和完整性，我们引入了零知识证明，使客户能够验证检测结果，而无需仅仅依赖服务器的善意。据我们所知，这是弥合FL安全理论进步与现实世界部署需求之间差距的第一种方法。跨各种任务和现实世界边缘设备的广泛实验证明了我们的方法相对于最先进防御的有效性。



## **34. DiffuseDef: Improved Robustness to Adversarial Attacks via Iterative Denoising**

diffuseDef：通过迭代去噪提高对抗性攻击的鲁棒性 cs.CL

Accepted to ACL 2025

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2407.00248v2) [paper-pdf](http://arxiv.org/pdf/2407.00248v2)

**Authors**: Zhenhao Li, Huichi Zhou, Marek Rei, Lucia Specia

**Abstract**: Pretrained language models have significantly advanced performance across various natural language processing tasks. However, adversarial attacks continue to pose a critical challenge to systems built using these models, as they can be exploited with carefully crafted adversarial texts. Inspired by the ability of diffusion models to predict and reduce noise in computer vision, we propose a novel and flexible adversarial defense method for language classification tasks, DiffuseDef, which incorporates a diffusion layer as a denoiser between the encoder and the classifier. The diffusion layer is trained on top of the existing classifier, ensuring seamless integration with any model in a plug-and-play manner. During inference, the adversarial hidden state is first combined with sampled noise, then denoised iteratively and finally ensembled to produce a robust text representation. By integrating adversarial training, denoising, and ensembling techniques, we show that DiffuseDef improves over existing adversarial defense methods and achieves state-of-the-art performance against common black-box and white-box adversarial attacks.

摘要: 预训练的语言模型在各种自然语言处理任务中具有显著的性能提升。然而，对抗性攻击继续对使用这些模型构建的系统构成严峻挑战，因为它们可以通过精心制作的对抗性文本来利用。受扩散模型在计算机视觉中预测和降低噪声的能力的启发，我们提出了一种用于语言分类任务的新颖灵活的对抗性防御方法DiffuseDef，它在编码器和分类器之间引入了扩散层作为去噪器。扩散层在现有分类器的基础上进行训练，确保以即插即用的方式与任何模型无缝集成。在推理过程中，对抗性隐藏状态首先与采样噪音相结合，然后迭代去噪，最后集成以产生稳健的文本表示。通过集成对抗性训练、去噪和集成技术，我们证明了DistuseDef比现有的对抗性防御方法进行了改进，并针对常见的黑匣子和白盒对抗性攻击实现了最先进的性能。



## **35. Securing Visually-Aware Recommender Systems: An Adversarial Image Reconstruction and Detection Framework**

保护视觉感知推荐系统：对抗性图像重建和检测框架 cs.CV

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2306.07992v2) [paper-pdf](http://arxiv.org/pdf/2306.07992v2)

**Authors**: Minglei Yin, Bin Liu, Neil Zhenqiang Gong, Xin Li

**Abstract**: With rich visual data, such as images, becoming readily associated with items, visually-aware recommendation systems (VARS) have been widely used in different applications. Recent studies have shown that VARS are vulnerable to item-image adversarial attacks, which add human-imperceptible perturbations to the clean images associated with those items. Attacks on VARS pose new security challenges to a wide range of applications such as e-Commerce and social networks where VARS are widely used. How to secure VARS from such adversarial attacks becomes a critical problem. Currently, there is still a lack of systematic study on how to design secure defense strategies against visual attacks on VARS. In this paper, we attempt to fill this gap by proposing an adversarial image reconstruction and detection framework to secure VARS. Our proposed method can simultaneously (1) secure VARS from adversarial attacks characterized by local perturbations by image reconstruction based on global vision transformers; and (2) accurately detect adversarial examples using a novel contrastive learning approach. Meanwhile, our framework is designed to be used as both a filter and a detector so that they can be jointly trained to improve the flexibility of our defense strategy to a variety of attacks and VARS models. We have conducted extensive experimental studies with two popular attack methods (FGSM and PGD). Our experimental results on two real-world datasets show that our defense strategy against visual attacks is effective and outperforms existing methods on different attacks. Moreover, our method can detect adversarial examples with high accuracy.

摘要: 随着诸如图像的丰富视觉数据变得容易与项目相关联，视觉感知推荐系统（VARS）已经被广泛用于不同的应用中。最近的研究表明，VARS很容易受到项目图像对抗性攻击的影响，这会给与这些项目相关的干净图像增加人类无法察觉的扰动。对VAR的攻击给广泛使用VAR的电子商务和社交网络等广泛应用带来了新的安全挑战。如何保护VAR免受此类对抗攻击成为一个关键问题。目前，关于如何设计针对VAR视觉攻击的安全防御策略仍然缺乏系统的研究。在本文中，我们试图通过提出一种对抗性图像重建和检测框架来保护VAR来填补这一空白。我们提出的方法可以同时（1）通过基于全局视觉变换器的图像重建来保护VAR免受以局部扰动为特征的对抗性攻击;和（2）使用新型的对比学习方法准确检测对抗性示例。与此同时，我们的框架旨在同时用作过滤器和检测器，以便它们可以进行联合训练，以提高我们防御策略对各种攻击和VAR模型的灵活性。我们使用两种流行的攻击方法（FGSM和PVD）进行了广泛的实验研究。我们在两个现实世界数据集上的实验结果表明，我们针对视觉攻击的防御策略是有效的，并且在不同攻击上优于现有方法。此外，我们的方法可以高准确性地检测对抗示例。



## **36. Co-Evolutionary Defence of Active Directory Attack Graphs via GNN-Approximated Dynamic Programming**

通过GNN逼近动态规划实现Active目录攻击图的协同进化防御 cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11710v1) [paper-pdf](http://arxiv.org/pdf/2505.11710v1)

**Authors**: Diksha Goel, Hussain Ahmad, Kristen Moore, Mingyu Guo

**Abstract**: Modern enterprise networks increasingly rely on Active Directory (AD) for identity and access management. However, this centralization exposes a single point of failure, allowing adversaries to compromise high-value assets. Existing AD defense approaches often assume static attacker behavior, but real-world adversaries adapt dynamically, rendering such methods brittle. To address this, we model attacker-defender interactions in AD as a Stackelberg game between an adaptive attacker and a proactive defender. We propose a co-evolutionary defense framework that combines Graph Neural Network Approximated Dynamic Programming (GNNDP) to model attacker strategies, with Evolutionary Diversity Optimization (EDO) to generate resilient blocking strategies. To ensure scalability, we introduce a Fixed-Parameter Tractable (FPT) graph reduction method that reduces complexity while preserving strategic structure. Our framework jointly refines attacker and defender policies to improve generalization and prevent premature convergence. Experiments on synthetic AD graphs show near-optimal results (within 0.1 percent of optimality on r500) and improved performance on larger graphs (r1000 and r2000), demonstrating the framework's scalability and effectiveness.

摘要: 现代企业网络越来越依赖活动目录（AD）进行身份和访问管理。然而，这种集中化暴露了单点失败，允许对手损害高价值资产。现有的AD防御方法通常假设静态攻击者行为，但现实世界的对手会动态适应，从而使此类方法变得脆弱。为了解决这个问题，我们将AD中的攻击者与防御者的交互建模为适应性攻击者和主动防御者之间的Stackelberg游戏。我们提出了一个协同进化防御框架，该框架将图形神经网络逼近动态规划（GNNDP）结合起来对攻击者策略进行建模，并将进化多样性优化（EDO）结合起来生成弹性拦截策略。为了确保可扩展性，我们引入了一种固定参数可跟踪（FPT）图约简方法，该方法可以降低复杂性，同时保留战略结构。我们的框架共同完善了攻击者和防御者政策，以提高普遍性并防止过早收敛。合成AD图上的实验显示出接近最佳的结果（在r500上的最佳性的0.1%内），并且在更大的图（r1000和r2000）上的性能得到了改进，证明了该框架的可扩展性和有效性。



## **37. Unveiling the Black Box: A Multi-Layer Framework for Explaining Reinforcement Learning-Based Cyber Agents**

揭开黑盒子：一个解释基于强化学习的网络代理的多层框架 cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11708v1) [paper-pdf](http://arxiv.org/pdf/2505.11708v1)

**Authors**: Diksha Goel, Kristen Moore, Jeff Wang, Minjune Kim, Thanh Thi Nguyen

**Abstract**: Reinforcement Learning (RL) agents are increasingly used to simulate sophisticated cyberattacks, but their decision-making processes remain opaque, hindering trust, debugging, and defensive preparedness. In high-stakes cybersecurity contexts, explainability is essential for understanding how adversarial strategies are formed and evolve over time. In this paper, we propose a unified, multi-layer explainability framework for RL-based attacker agents that reveals both strategic (MDP-level) and tactical (policy-level) reasoning. At the MDP level, we model cyberattacks as a Partially Observable Markov Decision Processes (POMDPs) to expose exploration-exploitation dynamics and phase-aware behavioural shifts. At the policy level, we analyse the temporal evolution of Q-values and use Prioritised Experience Replay (PER) to surface critical learning transitions and evolving action preferences. Evaluated across CyberBattleSim environments of increasing complexity, our framework offers interpretable insights into agent behaviour at scale. Unlike previous explainable RL methods, which are often post-hoc, domain-specific, or limited in depth, our approach is both agent- and environment-agnostic, supporting use cases ranging from red-team simulation to RL policy debugging. By transforming black-box learning into actionable behavioural intelligence, our framework enables both defenders and developers to better anticipate, analyse, and respond to autonomous cyber threats.

摘要: 强化学习（RL）代理越来越多地用于模拟复杂的网络攻击，但它们的决策过程仍然不透明，阻碍了信任、调试和防御准备。在高风险的网络安全背景下，可解释性对于了解对抗策略如何形成和随着时间的推移而演变至关重要。在本文中，我们为基于RL的攻击者代理提出了一个统一的多层解释性框架，该框架揭示了战略（MPP级）和战术（策略级）推理。在MDP层面，我们将网络攻击建模为部分可观察的马尔科夫决策过程（POMDPs），以揭示探索利用动态和阶段感知行为转变。在政策层面，我们分析Q值的时间演变，并使用优先体验回放（PER）来揭示关键的学习转变和不断变化的行动偏好。我们的框架在复杂性日益增加的CyberBattleSim环境中进行评估，提供了对大规模代理行为的可解释见解。与之前的可解释RL方法（通常是事后处理的、特定于领域的或深度有限）不同，我们的方法既不依赖于代理，又不依赖于环境，支持从红队模拟到RL策略调试等各种用例。通过将黑匣子学习转化为可操作的行为智能，我们的框架使防御者和开发人员能够更好地预测、分析和应对自主网络威胁。



## **38. On the Sharp Input-Output Analysis of Nonlinear Systems under Adversarial Attacks**

对抗攻击下非线性系统的尖锐输入输出分析 math.OC

28 pages, 2 figures

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11688v1) [paper-pdf](http://arxiv.org/pdf/2505.11688v1)

**Authors**: Jihun Kim, Yuchen Fang, Javad Lavaei

**Abstract**: This paper is concerned with learning the input-output mapping of general nonlinear dynamical systems. While the existing literature focuses on Gaussian inputs and benign disturbances, we significantly broaden the scope of admissible control inputs and allow correlated, nonzero-mean, adversarial disturbances. With our reformulation as a linear combination of basis functions, we prove that the $l_1$-norm estimator overcomes the challenges as long as the probability that the system is under adversarial attack at a given time is smaller than a certain threshold. We provide an estimation error bound that decays with the input memory length and prove its optimality by constructing a problem instance that suffers from the same bound under adversarial attacks. Our work provides a sharp input-output analysis for a generic nonlinear and partially observed system under significantly generalized assumptions compared to existing works.

摘要: 本文研究一般非线性动力系统的输入输出映射的学习。虽然现有文献关注高斯输入和良性干扰，但我们显着扩大了可接受控制输入的范围，并允许相关的、非零均值的、对抗性干扰。通过将我们的公式重新定义为基函数的线性组合，我们证明只要系统在给定时间受到对抗攻击的概率小于某个阈值，$l_1$-norm估计器就能克服挑战。我们提供了一个随着输入内存长度而衰减的估计误差界，并通过构造一个在对抗性攻击下遭受相同界的问题实例来证明其最优性。与现有作品相比，我们的工作在显着概括的假设下为一般非线性和部分观察系统提供了尖锐的输入输出分析。



## **39. Benchmarking Unsupervised Online IDS for Masquerade Attacks in CAN**

在CAN中对无监督在线IDS进行伪装攻击的基准测试 cs.CR

17 pages, 10 figures, 4 tables

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2406.13778v2) [paper-pdf](http://arxiv.org/pdf/2406.13778v2)

**Authors**: Pablo Moriano, Steven C. Hespeler, Mingyan Li, Robert A. Bridges

**Abstract**: Vehicular controller area networks (CANs) are susceptible to masquerade attacks by malicious adversaries. In masquerade attacks, adversaries silence a targeted ID and then send malicious frames with forged content at the expected timing of benign frames. As masquerade attacks could seriously harm vehicle functionality and are the stealthiest attacks to detect in CAN, recent work has devoted attention to compare frameworks for detecting masquerade attacks in CAN. However, most existing works report offline evaluations using CAN logs already collected using simulations that do not comply with the domain's real-time constraints. Here we contribute to advance the state of the art by introducing a benchmark study of four different non-deep learning (DL)-based unsupervised online intrusion detection systems (IDS) for masquerade attacks in CAN. Our approach differs from existing benchmarks in that we analyze the effect of controlling streaming data conditions in a sliding window setting. In doing so, we use realistic masquerade attacks being replayed from the ROAD dataset. We show that although benchmarked IDS are not effective at detecting every attack type, the method that relies on detecting changes in the hierarchical structure of clusters of time series produces the best results at the expense of higher computational overhead. We discuss limitations, open challenges, and how the benchmarked methods can be used for practical unsupervised online CAN IDS for masquerade attacks.

摘要: 车辆控制器区域网络（CAN）容易受到恶意对手的伪装攻击。在伪装攻击中，对手会压制目标ID，然后在良性帧的预期时间发送包含伪造内容的恶意帧。由于化装攻击可能会严重损害车辆功能，并且是CAN中最隐蔽的攻击，因此最近的工作重点关注比较用于检测CAN中化装攻击的框架。然而，大多数现有作品使用已经通过不符合域实时限制的模拟收集的CAN日志来报告离线评估。在这里，我们通过引入针对CAN中伪装攻击的四个不同的基于非深度学习（DL）的无监督在线入侵检测系统（IDS）的基准研究，为推进最新技术水平做出贡献。我们的方法与现有基准的不同之处在于，我们分析了在滑动窗口设置中控制流数据条件的影响。在此过程中，我们使用从ROAD数据集中重播的真实化装攻击。我们表明，尽管基准IDS不能有效检测每种攻击类型，但依赖于检测时间序列集群分层结构变化的方法可以产生最好的结果，但代价是更高的计算负担。我们讨论了局限性、开放挑战以及如何将基准方法用于实际的无监督在线CAN IDS以进行伪装攻击。



## **40. To Think or Not to Think: Exploring the Unthinking Vulnerability in Large Reasoning Models**

思考或不思考：探索大型推理模型中不思考的脆弱性 cs.CL

39 pages, 13 tables, 14 figures

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2502.12202v2) [paper-pdf](http://arxiv.org/pdf/2502.12202v2)

**Authors**: Zihao Zhu, Hongbao Zhang, Ruotong Wang, Ke Xu, Siwei Lyu, Baoyuan Wu

**Abstract**: Large Reasoning Models (LRMs) are designed to solve complex tasks by generating explicit reasoning traces before producing final answers. However, we reveal a critical vulnerability in LRMs -- termed Unthinking Vulnerability -- wherein the thinking process can be bypassed by manipulating special delimiter tokens. It is empirically demonstrated to be widespread across mainstream LRMs, posing both a significant risk and potential utility, depending on how it is exploited. In this paper, we systematically investigate this vulnerability from both malicious and beneficial perspectives. On the malicious side, we introduce Breaking of Thought (BoT), a novel attack that enables adversaries to bypass the thinking process of LRMs, thereby compromising their reliability and availability. We present two variants of BoT: a training-based version that injects backdoor during the fine-tuning stage, and a training-free version based on adversarial attack during the inference stage. As a potential defense, we propose thinking recovery alignment to partially mitigate the vulnerability. On the beneficial side, we introduce Monitoring of Thought (MoT), a plug-and-play framework that allows model owners to enhance efficiency and safety. It is implemented by leveraging the same vulnerability to dynamically terminate redundant or risky reasoning through external monitoring. Extensive experiments show that BoT poses a significant threat to reasoning reliability, while MoT provides a practical solution for preventing overthinking and jailbreaking. Our findings expose an inherent flaw in current LRM architectures and underscore the need for more robust reasoning systems in the future.

摘要: 大型推理模型（LRM）旨在通过在生成最终答案之前生成显式推理痕迹来解决复杂任务。然而，我们揭示了LRM中的一个关键漏洞--称为“无思考漏洞”--其中思维过程可以通过操纵特殊的Inbox令牌来绕过。经验证明，它在主流LRM中广泛存在，既构成重大风险，又构成潜在效用，具体取决于它的利用方式。在本文中，我们从恶意和有益的角度系统地调查了该漏洞。在恶意方面，我们引入了突破思想（BoT），这是一种新型攻击，使对手能够绕过LRM的思维过程，从而损害其可靠性和可用性。我们提出了BoT的两个变体：一个是在微调阶段注入后门的基于训练的版本，另一个是在推理阶段基于对抗攻击的免训练版本。作为一种潜在的防御措施，我们建议考虑恢复对齐来部分缓解漏洞。从有利的方面来说，我们引入了思维监控（MoT），这是一种即插即用框架，允许模型所有者提高效率和安全性。它是通过利用相同的漏洞通过外部监控动态终止冗余或有风险的推理来实现的。大量实验表明，BoT对推理可靠性构成了重大威胁，而MoT则为防止过度思考和越狱提供了实用的解决方案。我们的研究结果暴露了当前LRM架构中的固有缺陷，并强调了未来对更强大推理系统的需求。



## **41. LLMs unlock new paths to monetizing exploits**

LLM开辟了利用货币化的新途径 cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11449v1) [paper-pdf](http://arxiv.org/pdf/2505.11449v1)

**Authors**: Nicholas Carlini, Milad Nasr, Edoardo Debenedetti, Barry Wang, Christopher A. Choquette-Choo, Daphne Ippolito, Florian Tramèr, Matthew Jagielski

**Abstract**: We argue that Large language models (LLMs) will soon alter the economics of cyberattacks. Instead of attacking the most commonly used software and monetizing exploits by targeting the lowest common denominator among victims, LLMs enable adversaries to launch tailored attacks on a user-by-user basis. On the exploitation front, instead of human attackers manually searching for one difficult-to-identify bug in a product with millions of users, LLMs can find thousands of easy-to-identify bugs in products with thousands of users. And on the monetization front, instead of generic ransomware that always performs the same attack (encrypt all your data and request payment to decrypt), an LLM-driven ransomware attack could tailor the ransom demand based on the particular content of each exploited device.   We show that these two attacks (and several others) are imminently practical using state-of-the-art LLMs. For example, we show that without any human intervention, an LLM finds highly sensitive personal information in the Enron email dataset (e.g., an executive having an affair with another employee) that could be used for blackmail. While some of our attacks are still too expensive to scale widely today, the incentives to implement these attacks will only increase as LLMs get cheaper. Thus, we argue that LLMs create a need for new defense-in-depth approaches.

摘要: 我们认为大型语言模型（LLM）很快就会改变网络攻击的经济学。LLM不是攻击最常用的软件并通过针对受害者中最低的共同点来利用漏洞获利，而是使对手能够针对每个用户发起量身定制的攻击。在剥削方面，LLM可以在拥有数千名用户的产品中找到数千个易于识别的错误，而不是人类攻击者手动搜索拥有数百万用户的产品中的数千个易于识别的错误。在货币化方面，LLM驱动的勒索软件攻击不是总是执行相同攻击（加密所有数据并请求付费解密）的通用勒索软件，而是可以根据每个被利用设备的特定内容定制赎金需求。   我们表明，使用最先进的LLM，这两种攻击（以及其他几种攻击）迫在眉睫。例如，我们表明，在没有任何人为干预的情况下，LLM可以在安然电子邮件数据集中找到高度敏感的个人信息（例如，高管与另一名员工有外遇）可能用于勒索。虽然今天我们的一些攻击仍然过于昂贵，无法广泛扩展，但随着LLM变得更便宜，实施这些攻击的动机只会增加。因此，我们认为LLM需要新的深度防御方法。



## **42. CARES: Comprehensive Evaluation of Safety and Adversarial Robustness in Medical LLMs**

CARES：医学LLM安全性和对抗稳健性的综合评估 cs.CL

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11413v1) [paper-pdf](http://arxiv.org/pdf/2505.11413v1)

**Authors**: Sijia Chen, Xiaomin Li, Mengxue Zhang, Eric Hanchen Jiang, Qingcheng Zeng, Chen-Hsiang Yu

**Abstract**: Large language models (LLMs) are increasingly deployed in medical contexts, raising critical concerns about safety, alignment, and susceptibility to adversarial manipulation. While prior benchmarks assess model refusal capabilities for harmful prompts, they often lack clinical specificity, graded harmfulness levels, and coverage of jailbreak-style attacks. We introduce CARES (Clinical Adversarial Robustness and Evaluation of Safety), a benchmark for evaluating LLM safety in healthcare. CARES includes over 18,000 prompts spanning eight medical safety principles, four harm levels, and four prompting styles: direct, indirect, obfuscated, and role-play, to simulate both malicious and benign use cases. We propose a three-way response evaluation protocol (Accept, Caution, Refuse) and a fine-grained Safety Score metric to assess model behavior. Our analysis reveals that many state-of-the-art LLMs remain vulnerable to jailbreaks that subtly rephrase harmful prompts, while also over-refusing safe but atypically phrased queries. Finally, we propose a mitigation strategy using a lightweight classifier to detect jailbreak attempts and steer models toward safer behavior via reminder-based conditioning. CARES provides a rigorous framework for testing and improving medical LLM safety under adversarial and ambiguous conditions.

摘要: 大型语言模型（LLM）越来越多地被部署在医疗环境中，引起了人们对安全性、对齐性和对抗性操纵敏感性的严重关注。虽然先前的基准评估模型拒绝有害提示的能力，但它们通常缺乏临床特异性，分级的危害级别和越狱式攻击的覆盖范围。我们介绍了CARES（临床对抗性鲁棒性和安全性评估），这是评估LLM在医疗保健中安全性的基准。CARES包含超过18，000个提示，涵盖八个医疗安全原则，四个危害级别和四种提示风格：直接，间接，模糊和角色扮演，以模拟恶意和良性用例。我们提出了一个三向响应评估协议（接受、谨慎、拒绝）和细粒度的安全评分指标来评估模型行为。我们的分析表明，许多最先进的LLM仍然容易受到越狱的影响，这些越狱巧妙地重新表达有害提示，同时也过度拒绝安全但措辞合理的查询。最后，我们提出了一种缓解策略，使用轻量级分类器来检测越狱尝试，并通过基于条件反射来引导模型转向更安全的行为。CARES提供了一个严格的框架，用于在对抗和模糊的条件下测试和改善医学LLM安全性。



## **43. Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities**

基于有限样本浓度不等式的LLM文本检测零次统计检验 stat.ML

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2501.02406v4) [paper-pdf](http://arxiv.org/pdf/2501.02406v4)

**Authors**: Tara Radvand, Mojtaba Abdolmaleki, Mohamed Mostagir, Ambuj Tewari

**Abstract**: Verifying the provenance of content is crucial to the function of many organizations, e.g., educational institutions, social media platforms, firms, etc. This problem is becoming increasingly challenging as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions utilize in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within the institution. In this paper, we answer the following question: Given a piece of text, can we identify whether it was produced by a particular LLM or not? We model LLM-generated text as a sequential stochastic process with complete dependence on history. We then design zero-shot statistical tests to (i) distinguish between text generated by two different known sets of LLMs $A$ (non-sanctioned) and $B$ (in-house), and (ii) identify whether text was generated by a known LLM or generated by any unknown model, e.g., a human or some other language generation process. We prove that the type I and type II errors of our test decrease exponentially with the length of the text. For that, we show that if $B$ generates the text, then except with an exponentially small probability in string length, the log-perplexity of the string under $A$ converges to the average cross-entropy of $B$ and $A$. We then present experiments using LLMs with white-box access to support our theoretical results and empirically examine the robustness of our results to black-box settings and adversarial attacks. In the black-box setting, our method achieves an average TPR of 82.5\% at a fixed FPR of 5\%. Under adversarial perturbations, our minimum TPR is 48.6\% at the same FPR threshold. Both results outperform all non-commercial baselines. See https://github.com/TaraRadvand74/llm-text-detection for code, data, and an online demo of the project.

摘要: 验证内容的出处对于许多组织的功能至关重要，例如，教育机构、社交媒体平台、公司等。随着大型语言模型（LLM）生成的文本与人类生成的内容几乎无法区分，这个问题变得越来越具有挑战性。此外，许多机构利用内部LLM，并希望确保外部未经批准的LLM不会在机构内制作内容。在本文中，我们回答了以下问题：给定一段文本，我们能否识别它是否是由特定的LLM生成的？我们将LLM生成的文本建模为一个完全依赖于历史的顺序随机过程。然后，我们设计零镜头统计测试，以（i）区分由两组不同的已知LLM $A$（未经批准）和$B$（内部）生成的文本，以及（ii）识别文本是由已知LLM生成还是由任何未知模型生成，例如人类或某种其他语言生成过程。我们证明，我们测试的I型和II型错误随着文本长度的增加而呈指数级减少。为此，我们表明，如果$B$生成文本，那么除了字符串长度的概率呈指数级小外，$A$下的字符串的log困惑度会收敛到$B$和$A$的平均交叉熵。然后，我们使用具有白盒访问权限的LLM进行了实验，以支持我们的理论结果，并从经验上检查我们的结果对黑匣子设置和对抗攻击的鲁棒性。在黑匣子设置中，我们的方法在固定FPR为5%时实现了82.5%的平均TPA。在对抗性扰动下，在相同FPR阈值下，我们的最低TPA为48.6%。这两个结果都优于所有非商业基线。请参阅https://github.com/TaraRadvand74/llm-text-detection了解该项目的代码、数据和在线演示。



## **44. GenoArmory: A Unified Evaluation Framework for Adversarial Attacks on Genomic Foundation Models**

GenoArmory：对抗性攻击基因组基础模型的统一评估框架 cs.LG

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.10983v1) [paper-pdf](http://arxiv.org/pdf/2505.10983v1)

**Authors**: Haozheng Luo, Chenghao Qiu, Yimin Wang, Shang Wu, Jiahao Yu, Han Liu, Binghui Wang, Yan Chen

**Abstract**: We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GenoArmory. Unlike existing GFM benchmarks, GenoArmory offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Additionally, we introduce GenoAdv, a new adversarial sample dataset designed to improve GFM safety. Empirically, classification models exhibit greater robustness to adversarial perturbations compared to generative models, highlighting the impact of task type on model vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features.

摘要: 我们为基因组基础模型（GFM）提出了第一个统一的对抗攻击基准，名为GenoArmory。与现有的GFM基准不同，GenoArmory提供了第一个全面的评估框架来系统地评估GFM对对抗攻击的脆弱性。在方法上，我们评估了五个国家的最先进的GFM使用四个广泛采用的攻击算法和三个防御策略的对抗鲁棒性。重要的是，我们的基准提供了一个可访问的和全面的框架，以分析GFM漏洞的模型架构，量化方案和训练数据集。此外，我们还引入了GenoAdv，这是一个新的对抗性样本数据集，旨在提高GFM安全性。从经验上看，与生成模型相比，分类模型对对抗性扰动表现出更大的鲁棒性，凸显了任务类型对模型脆弱性的影响。此外，对抗性攻击经常针对具有生物学意义的基因组区域，这表明这些模型有效地捕获了有意义的序列特征。



## **45. On the Security Risks of ML-based Malware Detection Systems: A Survey**

基于ML的恶意软件检测系统安全风险研究综述 cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.10903v1) [paper-pdf](http://arxiv.org/pdf/2505.10903v1)

**Authors**: Ping He, Yuhao Mao, Changjiang Li, Lorenzo Cavallaro, Ting Wang, Shouling Ji

**Abstract**: Malware presents a persistent threat to user privacy and data integrity. To combat this, machine learning-based (ML-based) malware detection (MD) systems have been developed. However, these systems have increasingly been attacked in recent years, undermining their effectiveness in practice. While the security risks associated with ML-based MD systems have garnered considerable attention, the majority of prior works is limited to adversarial malware examples, lacking a comprehensive analysis of practical security risks. This paper addresses this gap by utilizing the CIA principles to define the scope of security risks. We then deconstruct ML-based MD systems into distinct operational stages, thus developing a stage-based taxonomy. Utilizing this taxonomy, we summarize the technical progress and discuss the gaps in the attack and defense proposals related to the ML-based MD systems within each stage. Subsequently, we conduct two case studies, using both inter-stage and intra-stage analyses according to the stage-based taxonomy to provide new empirical insights. Based on these analyses and insights, we suggest potential future directions from both inter-stage and intra-stage perspectives.

摘要: 恶意软件对用户隐私和数据完整性构成持续威胁。为了解决这个问题，开发了基于机器学习（基于ML）的恶意软件检测（MD）系统。然而，近年来，这些系统越来越受到攻击，削弱了它们在实践中的有效性。虽然与基于ML的MD系统相关的安全风险引起了相当大的关注，但大多数先前的作品仅限于对抗性恶意软件示例，缺乏对实际安全风险的全面分析。本文通过利用中央情报局的原则来定义安全风险的范围来解决这一差距。然后，我们将基于ML的MD系统解构为不同的操作阶段，从而开发出基于阶段的分类法。利用该分类法，我们总结了技术进展，并讨论了每个阶段与基于ML的MD系统相关的攻击和防御提案中的差距。随后，我们进行了两个案例研究，根据基于阶段的分类法使用阶段间和阶段内分析，以提供新的经验见解。基于这些分析和见解，我们从阶段间和阶段内的角度提出潜在的未来方向。



## **46. LARGO: Latent Adversarial Reflection through Gradient Optimization for Jailbreaking LLMs**

LARGO：通过越狱LLM的梯度优化实现潜在的对抗反射 cs.LG

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.10838v1) [paper-pdf](http://arxiv.org/pdf/2505.10838v1)

**Authors**: Ran Li, Hao Wang, Chengzhi Mao

**Abstract**: Efficient red-teaming method to uncover vulnerabilities in Large Language Models (LLMs) is crucial. While recent attacks often use LLMs as optimizers, the discrete language space make gradient-based methods struggle. We introduce LARGO (Latent Adversarial Reflection through Gradient Optimization), a novel latent self-reflection attack that reasserts the power of gradient-based optimization for generating fluent jailbreaking prompts. By operating within the LLM's continuous latent space, LARGO first optimizes an adversarial latent vector and then recursively call the same LLM to decode the latent into natural language. This methodology yields a fast, effective, and transferable attack that produces fluent and stealthy prompts. On standard benchmarks like AdvBench and JailbreakBench, LARGO surpasses leading jailbreaking techniques, including AutoDAN, by 44 points in attack success rate. Our findings demonstrate a potent alternative to agentic LLM prompting, highlighting the efficacy of interpreting and attacking LLM internals through gradient optimization.

摘要: 发现大型语言模型（LLM）中漏洞的高效红色团队方法至关重要。虽然最近的攻击经常使用LLM作为优化器，但离散语言空间使得基于梯度的方法变得困难。我们引入了LARGO（通过梯度优化的潜在对抗反射），这是一种新型的潜在自我反射攻击，它重申了基于梯度的优化用于生成流畅的越狱提示的力量。通过在LLM的连续潜在空间内操作，LARGO首先优化对抗性潜在载体，然后循环调用相同的LLM将潜在载体解码为自然语言。这种方法可以产生快速、有效且可转移的攻击，从而产生流畅且隐蔽的提示。在AdvBench和JailbreakBench等标准基准上，LARGO的攻击成功率比AutoDAN等领先越狱技术高出44分。我们的研究结果证明了代理LLM提示的有效替代方案，强调了通过梯度优化解释和攻击LLM内部内容的有效性。



## **47. Mitigating Many-Shot Jailbreaking**

缓解多次越狱 cs.LG

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2504.09604v3) [paper-pdf](http://arxiv.org/pdf/2504.09604v3)

**Authors**: Christopher M. Ackerman, Nina Panickssery

**Abstract**: Many-shot jailbreaking (MSJ) is an adversarial technique that exploits the long context windows of modern LLMs to circumvent model safety training by including in the prompt many examples of a "fake" assistant responding inappropriately before the final request. With enough examples, the model's in-context learning abilities override its safety training, and it responds as if it were the "fake" assistant. In this work, we probe the effectiveness of different fine-tuning and input sanitization approaches on mitigating MSJ attacks, alone and in combination. We find incremental mitigation effectiveness for each, and show that the combined techniques significantly reduce the effectiveness of MSJ attacks, while retaining model performance in benign in-context learning and conversational tasks. We suggest that our approach could meaningfully ameliorate this vulnerability if incorporated into model safety post-training.

摘要: 多镜头越狱（MSJ）是一种对抗性技术，它利用现代LLM的长上下文窗口来规避模型安全培训，方法是在提示中包含许多“假”助理在最终请求之前做出不当反应的示例。有了足够多的例子，该模型的上下文学习能力就会凌驾于其安全培训之上，并且它的反应就好像它是“假”助手一样。在这项工作中，我们探讨了不同的微调和输入清理方法单独和组合在减轻MSJ攻击方面的有效性。我们发现每种技术的增量缓解效果，并表明组合技术显着降低了MSJ攻击的有效性，同时保留了良性上下文学习和对话任务中的模型性能。我们认为，如果将我们的方法纳入模型安全培训后，可以有意义地改善这种脆弱性。



## **48. SecReEvalBench: A Multi-turned Security Resilience Evaluation Benchmark for Large Language Models**

SecReEvalBench：大型语言模型的多角度安全弹性评估基准 cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.07584v2) [paper-pdf](http://arxiv.org/pdf/2505.07584v2)

**Authors**: Huining Cui, Wei Liu

**Abstract**: The increasing deployment of large language models in security-sensitive domains necessitates rigorous evaluation of their resilience against adversarial prompt-based attacks. While previous benchmarks have focused on security evaluations with limited and predefined attack domains, such as cybersecurity attacks, they often lack a comprehensive assessment of intent-driven adversarial prompts and the consideration of real-life scenario-based multi-turn attacks. To address this gap, we present SecReEvalBench, the Security Resilience Evaluation Benchmark, which defines four novel metrics: Prompt Attack Resilience Score, Prompt Attack Refusal Logic Score, Chain-Based Attack Resilience Score and Chain-Based Attack Rejection Time Score. Moreover, SecReEvalBench employs six questioning sequences for model assessment: one-off attack, successive attack, successive reverse attack, alternative attack, sequential ascending attack with escalating threat levels and sequential descending attack with diminishing threat levels. In addition, we introduce a dataset customized for the benchmark, which incorporates both neutral and malicious prompts, categorised across seven security domains and sixteen attack techniques. In applying this benchmark, we systematically evaluate five state-of-the-art open-weighted large language models, Llama 3.1, Gemma 2, Mistral v0.3, DeepSeek-R1 and Qwen 3. Our findings offer critical insights into the strengths and weaknesses of modern large language models in defending against evolving adversarial threats. The SecReEvalBench dataset is publicly available at https://kaggle.com/datasets/5a7ee22cf9dab6c93b55a73f630f6c9b42e936351b0ae98fbae6ddaca7fe248d, which provides a groundwork for advancing research in large language model security.

摘要: 大型语言模型在安全敏感领域的部署越来越多，需要严格评估它们对抗基于预算的敌对攻击的弹性。虽然之前的基准侧重于有限且预定义的攻击域（例如网络安全攻击）的安全评估，但它们通常缺乏对意图驱动的对抗提示的全面评估以及对现实生活中基于情景的多回合攻击的考虑。为了解决这一差距，我们提出了SecReEvalBench，安全韧性评估基准，它定义了四个新颖的指标：即时攻击韧性分数、即时攻击拒绝逻辑分数、基于链的攻击韧性分数和基于链的攻击拒绝时间分数。此外，SecReEvalBench采用六个提问序列进行模型评估：一次性攻击、连续攻击、连续反向攻击、替代攻击、威胁级别不断上升的顺序上升攻击和威胁级别不断下降的顺序下降攻击。此外，我们还引入了一个为基准定制的数据集，其中包含中性和恶意提示，分为七个安全域和十六种攻击技术。在应用该基准时，我们系统地评估了五个最先进的开放加权大型语言模型：Llama 3.1、Gemma 2、Mistral v0.3、DeepSeek-R1和Qwen 3。我们的研究结果为现代大型语言模型在防御不断变化的对抗威胁方面的优势和弱点提供了重要的见解。SecReEvalBench数据集可在https：//kaggle.com/guardets/5a7ee22CF9dab6c93b55a73f630f6c9 b42 e936351 b 0ae 98 fbae 6ddaca 7 fe 248 d上公开，为推进大型语言模型安全性研究提供了基础。



## **49. Model-Targeted Data Poisoning Attacks against ITS Applications with Provable Convergence**

可证明收敛的面向模型的ITS应用数据中毒攻击 math.OC

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.03966v2) [paper-pdf](http://arxiv.org/pdf/2505.03966v2)

**Authors**: Xin Wang, Feilong Wang, Yuan Hong, R. Tyrrell Rockafellar, Xuegang, Ban

**Abstract**: The growing reliance of intelligent systems on data makes the systems vulnerable to data poisoning attacks. Such attacks could compromise machine learning or deep learning models by disrupting the input data. Previous studies on data poisoning attacks are subject to specific assumptions, and limited attention is given to learning models with general (equality and inequality) constraints or lacking differentiability. Such learning models are common in practice, especially in Intelligent Transportation Systems (ITS) that involve physical or domain knowledge as specific model constraints. Motivated by ITS applications, this paper formulates a model-target data poisoning attack as a bi-level optimization problem with a constrained lower-level problem, aiming to induce the model solution toward a target solution specified by the adversary by modifying the training data incrementally. As the gradient-based methods fail to solve this optimization problem, we propose to study the Lipschitz continuity property of the model solution, enabling us to calculate the semi-derivative, a one-sided directional derivative, of the solution over data. We leverage semi-derivative descent to solve the bi-level optimization problem, and establish the convergence conditions of the method to any attainable target model. The model and solution method are illustrated with a simulation of a poisoning attack on the lane change detection using SVM.

摘要: 智能系统对数据的日益依赖使得系统容易受到数据中毒攻击。这种攻击可能会破坏输入数据，从而危及机器学习或深度学习模型。以往关于数据中毒攻击的研究都受到特定假设的限制，对具有一般（等式和不等式）约束或缺乏可微性的学习模型的关注有限。这种学习模型在实践中很常见，特别是在涉及物理或领域知识作为特定模型约束的智能交通系统（ITS）中。受ITS应用的启发，本文将模型-目标数据中毒攻击描述为具有约束较低层问题的双层优化问题，旨在通过增量修改训练数据将模型解引导到对手指定的目标解。由于基于梯度的方法无法解决这个优化问题，我们建议研究模型解的Lipschitz连续性，使我们能够计算解对数据的半导（单边方向导）。我们利用半导下降来解决双层优化问题，并建立该方法对任何可达到的目标模型的收敛条件。通过对使用支持者对车道变更检测的中毒攻击进行模拟，说明了该模型和解决方法。



## **50. Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines**

大型语言模型搜索引擎的对抗性攻击动态 cs.CL

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2501.00745v2) [paper-pdf](http://arxiv.org/pdf/2501.00745v2)

**Authors**: Xiyang Hu

**Abstract**: The increasing integration of Large Language Model (LLM) based search engines has transformed the landscape of information retrieval. However, these systems are vulnerable to adversarial attacks, especially ranking manipulation attacks, where attackers craft webpage content to manipulate the LLM's ranking and promote specific content, gaining an unfair advantage over competitors. In this paper, we study the dynamics of ranking manipulation attacks. We frame this problem as an Infinitely Repeated Prisoners' Dilemma, where multiple players strategically decide whether to cooperate or attack. We analyze the conditions under which cooperation can be sustained, identifying key factors such as attack costs, discount rates, attack success rates, and trigger strategies that influence player behavior. We identify tipping points in the system dynamics, demonstrating that cooperation is more likely to be sustained when players are forward-looking. However, from a defense perspective, we find that simply reducing attack success probabilities can, paradoxically, incentivize attacks under certain conditions. Furthermore, defensive measures to cap the upper bound of attack success rates may prove futile in some scenarios. These insights highlight the complexity of securing LLM-based systems. Our work provides a theoretical foundation and practical insights for understanding and mitigating their vulnerabilities, while emphasizing the importance of adaptive security strategies and thoughtful ecosystem design.

摘要: 基于大语言模型（LLM）的搜索引擎的日益集成已经改变了信息检索的格局。然而，这些系统很容易受到对抗攻击，尤其是排名操纵攻击，攻击者精心制作网页内容来操纵LLM的排名并推广特定内容，从而获得相对于竞争对手的不公平优势。本文中，我们研究了排名操纵攻击的动力学。我们将这个问题定义为“无限重复的囚徒困境”，其中多个参与者战略性地决定是合作还是攻击。我们分析了持续合作的条件，确定了攻击成本、折扣率、攻击成功率以及影响玩家行为的触发策略等关键因素。我们确定了系统动态中的临界点，证明当参与者具有前瞻性时，合作更有可能持续。然而，从防御的角度来看，我们发现，矛盾的是，简单地降低攻击成功概率就可以在某些条件下激励攻击。此外，在某些情况下，限制攻击成功率上限的防御措施可能是徒劳的。这些见解突出了保护基于LLM的系统的复杂性。我们的工作为理解和减轻其脆弱性提供了理论基础和实践见解，同时强调了自适应安全策略和周到的生态系统设计的重要性。



