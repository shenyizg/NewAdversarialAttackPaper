# Latest Large Language Model Attack Papers
**update at 2025-03-12 19:24:19**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Proactive Privacy Amnesia for Large Language Models: Safeguarding PII with Negligible Impact on Model Utility**

大型语言模型的主动隐私咨询：保护PRI，对模型实用性的影响可忽略不计 cs.CL

ICLR'25 Poster. Project page and code is available at  https://ppa-iclr2025.my.canva.site/

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.17591v2) [paper-pdf](http://arxiv.org/pdf/2502.17591v2)

**Authors**: Martin Kuo, Jingyang Zhang, Jianyi Zhang, Minxue Tang, Louis DiValentin, Aolin Ding, Jingwei Sun, William Chen, Amin Hass, Tianlong Chen, Yiran Chen, Hai Li

**Abstract**: With the rise of large language models (LLMs), increasing research has recognized their risk of leaking personally identifiable information (PII) under malicious attacks. Although efforts have been made to protect PII in LLMs, existing methods struggle to balance privacy protection with maintaining model utility. In this paper, inspired by studies of amnesia in cognitive science, we propose a novel approach, Proactive Privacy Amnesia (PPA), to safeguard PII in LLMs while preserving their utility. This mechanism works by actively identifying and forgetting key memories most closely associated with PII in sequences, followed by a memory implanting using suitable substitute memories to maintain the LLM's functionality. We conduct evaluations across multiple models to protect common PII, such as phone numbers and physical addresses, against prevalent PII-targeted attacks, demonstrating the superiority of our method compared with other existing defensive techniques. The results show that our PPA method completely eliminates the risk of phone number exposure by 100% and significantly reduces the risk of physical address exposure by 9.8% - 87.6%, all while maintaining comparable model utility performance.

摘要: 随着大型语言模型(LLM)的兴起，越来越多的研究已经认识到它们在恶意攻击下泄露个人身份信息(PII)的风险。虽然已经做出了努力来保护LLMS中的PII，但现有的方法难以平衡隐私保护和维护模型效用。受认知科学中关于健忘症的研究启发，我们提出了一种新的方法，即主动隐私健忘症(PPA)，在保护LLMS中的PII的同时保持其实用性。这种机制的工作原理是主动识别和忘记序列中与PII最密切相关的关键记忆，然后使用适当的替代记忆植入记忆以维持LLM的功能。我们在多个模型上进行评估，以保护常见的PII，如电话号码和物理地址，免受流行的PII目标攻击，展示了我们的方法与其他现有防御技术相比的优越性。结果表明，我们的PPA方法完全消除了100%的电话号码暴露风险，并显著降低了9.8%-87.6%的物理地址暴露风险，所有这些都保持了可比的模型实用性能。



## **2. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

紧急系统的守护者：用紧急系统防止多次枪击越狱 cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.16750v3) [paper-pdf](http://arxiv.org/pdf/2502.16750v3)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.

摘要: 使用大型语言模型的自主人工智能代理可以在社会各个领域创造不可否认的价值，但他们面临来自对手的安全威胁，需要立即采取保护性解决方案，因为信任和安全问题会出现。考虑到多发越狱和欺骗性对准是一些主要的高级攻击，在监督训练期间使用的静态护栏无法减轻这些攻击，指出了现实世界健壮性的关键研究重点。动态多智能体系统中静态护栏的组合不能抵抗这些攻击。我们打算通过制定新的评估框架，确定和应对安全行动部署所面临的威胁，从而加强基于LLM的特工的安全。我们的工作使用了三种检测方法来通过反向图灵测试来检测流氓代理，并通过多代理模拟来分析欺骗性比对，并开发了一个反越狱系统，通过使用Gemini 1.5Pro和Llama-3.3-70B、使用工具中介的对抗场景来测试DeepSeek R1模型来开发反越狱系统。Gemini 1.5 PRO具有很强的检测能力，如94%的准确率，但在长时间攻击下，随着提示长度的增加，攻击成功率(ASR)增加，多样性度量在预测多个复杂系统故障时变得无效，系统存在持续漏洞。这些发现证明了采用基于主动监控的灵活安全系统的必要性，该系统可以由代理自己执行，并由系统管理员进行适应性干预，因为当前的模型可能会产生漏洞，从而导致系统不可靠和易受攻击。因此，在我们的工作中，我们试图解决这些情况，并提出一个全面的框架来对抗安全问题。



## **3. Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation**

对话注入攻击：通过上下文操纵越狱LLM cs.CL

17 pages, 10 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08195v1) [paper-pdf](http://arxiv.org/pdf/2503.08195v1)

**Authors**: Wenlong Meng, Fan Zhang, Wendao Yao, Zhenyuan Guo, Yuwei Li, Chengkun Wei, Wenzhi Chen

**Abstract**: Large language models (LLMs) have demonstrated significant utility in a wide range of applications; however, their deployment is plagued by security vulnerabilities, notably jailbreak attacks. These attacks manipulate LLMs to generate harmful or unethical content by crafting adversarial prompts. While much of the current research on jailbreak attacks has focused on single-turn interactions, it has largely overlooked the impact of historical dialogues on model behavior. In this paper, we introduce a novel jailbreak paradigm, Dialogue Injection Attack (DIA), which leverages the dialogue history to enhance the success rates of such attacks. DIA operates in a black-box setting, requiring only access to the chat API or knowledge of the LLM's chat template. We propose two methods for constructing adversarial historical dialogues: one adapts gray-box prefilling attacks, and the other exploits deferred responses. Our experiments show that DIA achieves state-of-the-art attack success rates on recent LLMs, including Llama-3.1 and GPT-4o. Additionally, we demonstrate that DIA can bypass 5 different defense mechanisms, highlighting its robustness and effectiveness.

摘要: 大型语言模型(LLM)在广泛的应用程序中显示出了重要的实用价值；然而，它们的部署受到安全漏洞的困扰，特别是越狱攻击。这些攻击通过精心编制敌意提示来操纵LLM生成有害或不道德的内容。虽然目前对越狱攻击的大部分研究都集中在单回合互动上，但在很大程度上忽视了历史对话对模型行为的影响。在本文中，我们介绍了一种新的越狱范例，对话注入攻击(DIA)，它利用对话历史来提高此类攻击的成功率。DIA在黑盒设置中运行，只需要访问聊天API或了解LLM的聊天模板。我们提出了两种构造对抗性历史对话的方法：一种是采用灰盒预填充攻击，另一种是利用延迟响应。我们的实验表明，DIA在包括Llama-3.1和GPT-40在内的最近的LLM上达到了最先进的攻击成功率。此外，我们还证明了DIA可以绕过5种不同的防御机制，突出了其健壮性和有效性。



## **4. Reasoning-Augmented Conversation for Multi-Turn Jailbreak Attacks on Large Language Models**

针对大型语言模型的多回合越狱攻击的推理增强对话 cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.11054v4) [paper-pdf](http://arxiv.org/pdf/2502.11054v4)

**Authors**: Zonghao Ying, Deyue Zhang, Zonglei Jing, Yisong Xiao, Quanchen Zou, Aishan Liu, Siyuan Liang, Xiangzheng Zhang, Xianglong Liu, Dacheng Tao

**Abstract**: Multi-turn jailbreak attacks simulate real-world human interactions by engaging large language models (LLMs) in iterative dialogues, exposing critical safety vulnerabilities. However, existing methods often struggle to balance semantic coherence with attack effectiveness, resulting in either benign semantic drift or ineffective detection evasion. To address this challenge, we propose Reasoning-Augmented Conversation, a novel multi-turn jailbreak framework that reformulates harmful queries into benign reasoning tasks and leverages LLMs' strong reasoning capabilities to compromise safety alignment. Specifically, we introduce an attack state machine framework to systematically model problem translation and iterative reasoning, ensuring coherent query generation across multiple turns. Building on this framework, we design gain-guided exploration, self-play, and rejection feedback modules to preserve attack semantics, enhance effectiveness, and sustain reasoning-driven attack progression. Extensive experiments on multiple LLMs demonstrate that RACE achieves state-of-the-art attack effectiveness in complex conversational scenarios, with attack success rates (ASRs) increasing by up to 96%. Notably, our approach achieves ASRs of 82% and 92% against leading commercial models, OpenAI o1 and DeepSeek R1, underscoring its potency. We release our code at https://github.com/NY1024/RACE to facilitate further research in this critical domain.

摘要: 多轮越狱攻击通过在迭代对话中使用大型语言模型(LLM)来模拟真实世界的人类交互，暴露出关键的安全漏洞。然而，现有的方法往往难以在语义一致性和攻击有效性之间取得平衡，导致良性的语义漂移或无效的检测规避。为了应对这一挑战，我们提出了一种新的多轮越狱框架--推理增强对话，该框架将有害的查询重新定义为良性的推理任务，并利用LLMS强大的推理能力来妥协安全对齐。具体地说，我们引入了攻击状态机框架来系统地建模问题转换和迭代推理，确保跨多轮的连贯查询生成。在这个框架的基础上，我们设计了增益引导的探索、自我发挥和拒绝反馈模块，以保留攻击语义，提高有效性，并支持推理驱动的攻击进展。在多个LLM上的大量实验表明，RACE在复杂的会话场景中获得了最先进的攻击效率，攻击成功率(ASR)提高了96%。值得注意的是，我们的方法在领先的商业模型OpenAI o1和DeepSeek r1上分别获得了82%和92%的ASR，这突显了它的有效性。我们在https://github.com/NY1024/RACE上发布我们的代码，以促进在这一关键领域的进一步研究。



## **5. Safety Guardrails for LLM-Enabled Robots**

LLM支持机器人的安全护栏 cs.RO

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07885v1) [paper-pdf](http://arxiv.org/pdf/2503.07885v1)

**Authors**: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani

**Abstract**: Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots.

摘要: 尽管将大型语言模型(LLM)集成到机器人学中释放了变革性的能力，但它也带来了重大的安全问题，从平均情况下的LLM错误(例如，幻觉)到对抗性的越狱攻击，这可能会在现实世界中产生有害的机器人行为。传统的机器人安全方法不能解决LLM的新脆弱性，而当前的LLM安全护栏忽略了机器人在动态真实环境中操作所带来的物理风险。在本文中，我们提出了一种两级护栏结构RoboGuard，以确保LLM支持的机器人的安全。RoboGuard首先通过使用信任根LLM将预定义的安全规则与机器人环境相关联，该LLM使用思想链(COT)推理来生成严格的安全规范，如时间逻辑约束。然后，RoboGuard使用时态逻辑控制合成来解决这些上下文安全规范和可能不安全的计划之间的潜在冲突，这确保了安全合规性，同时将违反用户偏好的程度降至最低。通过考虑最坏情况下的越狱攻击的大量模拟和真实世界实验，我们证明了RoboGuard在不影响安全计划的性能的情况下，将不安全计划的执行率从92%降低到2.5%以下。我们还证明了RoboGuard是资源高效的，对自适应攻击具有健壮性，并且通过使其信任根LLM能够执行CoT推理而显著增强。这些结果突显了RoboGuard在缓解安全风险和提高启用LLM的机器人可靠性方面的潜力。



## **6. PoisonedParrot: Subtle Data Poisoning Attacks to Elicit Copyright-Infringing Content from Large Language Models**

PoisonedParrot：针对从大型语言模型中获取侵犯版权内容的微妙数据中毒攻击 cs.LG

18 pages, 18 figures. Accepted at NAACL 2025

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07697v1) [paper-pdf](http://arxiv.org/pdf/2503.07697v1)

**Authors**: Michael-Andrei Panaitescu-Liess, Pankayaraj Pathmanathan, Yigitcan Kaya, Zora Che, Bang An, Sicheng Zhu, Aakriti Agrawal, Furong Huang

**Abstract**: As the capabilities of large language models (LLMs) continue to expand, their usage has become increasingly prevalent. However, as reflected in numerous ongoing lawsuits regarding LLM-generated content, addressing copyright infringement remains a significant challenge. In this paper, we introduce PoisonedParrot: the first stealthy data poisoning attack that induces an LLM to generate copyrighted content even when the model has not been directly trained on the specific copyrighted material. PoisonedParrot integrates small fragments of copyrighted text into the poison samples using an off-the-shelf LLM. Despite its simplicity, evaluated in a wide range of experiments, PoisonedParrot is surprisingly effective at priming the model to generate copyrighted content with no discernible side effects. Moreover, we discover that existing defenses are largely ineffective against our attack. Finally, we make the first attempt at mitigating copyright-infringement poisoning attacks by proposing a defense: ParrotTrap. We encourage the community to explore this emerging threat model further.

摘要: 随着大型语言模型(LLM)的功能不断扩展，它们的使用也变得越来越普遍。然而，正如许多正在进行的关于LLM生成的内容的诉讼所反映的那样，解决侵犯版权的问题仍然是一个巨大的挑战。在本文中，我们介绍了PoisonedParot：第一种隐蔽的数据中毒攻击，即使模型没有针对特定的受版权保护的材料进行直接训练，也会诱导LLM生成受版权保护的内容。PoisonedParot使用现成的LLM将受版权保护的文本的小片段集成到毒药样本中。尽管它很简单，在广泛的实验中进行了评估，但PoisonedParot在启动模型以生成受版权保护的内容方面出人意料地有效，没有明显的副作用。此外，我们发现现有的防御在很大程度上对我们的攻击无效。最后，我们首次尝试通过提出一种防御方法来减轻版权侵权中毒攻击：ParrotTrap。我们鼓励社区进一步探索这种新兴的威胁模式。



## **7. The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective**

恐怖谷：从扁平的角度探索对抗性的稳健性 cs.LG

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2405.16918v2) [paper-pdf](http://arxiv.org/pdf/2405.16918v2)

**Authors**: Nils Philipp Walter, Linara Adilova, Jilles Vreeken, Michael Kamp

**Abstract**: Flatness of the loss surface not only correlates positively with generalization, but is also related to adversarial robustness since perturbations of inputs relate non-linearly to perturbations of weights. In this paper, we empirically analyze the relation between adversarial examples and relative flatness with respect to the parameters of one layer. We observe a peculiar property of adversarial examples in the context of relative flatness: during an iterative first-order white-box attack, the flatness of the loss surface measured around the adversarial example first becomes sharper until the label is flipped, but if we keep the attack running, it runs into a flat uncanny valley where the label remains flipped. In extensive experiments, we observe this phenomenon across various model architectures and datasets, even for adversarially trained models. Our results also extend to large language models (LLMs), but due to the discrete nature of the input space and comparatively weak attacks, adversarial examples rarely reach truly flat regions. Most importantly, this phenomenon shows that flatness alone cannot explain adversarial robustness unless we can also guarantee the behavior of the function around the examples. We, therefore theoretically connect relative flatness to adversarial robustness by bounding the third derivative of the loss surface, underlining the need for flatness in combination with a low global Lipschitz constant for a robust model.

摘要: 损失曲面的平坦性不仅与泛化正相关，而且还与对抗稳健性有关，因为输入的扰动与权重的扰动是非线性相关的。在这篇文章中，我们实证分析了对抗性例子与相对平坦度之间的关系。我们在相对平坦的背景下观察到对抗性例子的一个特殊性质：在迭代的一阶白盒攻击中，围绕对抗性例子测量的损失曲面的平坦性首先变得更尖锐，直到标签被翻转，但如果我们继续攻击，它会进入一个平坦的可怕山谷，在那里标签仍然被翻转。在广泛的实验中，我们在各种模型体系结构和数据集上观察到了这种现象，甚至对于经过恶意训练的模型也是如此。我们的结果也扩展到大型语言模型(LLM)，但由于输入空间的离散性质和相对较弱的攻击，对抗性例子很少到达真正平坦的区域。最重要的是，这一现象表明，平坦性本身不能解释对抗健壮性，除非我们也能保证函数在示例周围的行为。因此，理论上，我们通过限定损失曲面的三阶导数，将相对平坦性与对手的稳健性联系起来，强调了平坦性与稳健性模型的低全局Lipschitz常数相结合的必要性。



## **8. Utilizing Jailbreak Probability to Attack and Safeguard Multimodal LLMs**

利用越狱概率攻击和保护多模式LLM cs.CR

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06989v1) [paper-pdf](http://arxiv.org/pdf/2503.06989v1)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Deyue Zhang, Dongdong Yang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal contents. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on inputs to maximize jailbreak probability. To counteract attacks, we also propose two defensive methods: Jailbreak-Probability-based Finetuning (JPF) and Jailbreak-Probability-based Defensive Noise (JPDN), which minimizes jailbreak probability in the MLLM parameters and input space, respectively. Extensive experiments show that (1) JPA yields improvements (up to 28.38\%) under both white and black box settings compared to previous methods with small perturbation bounds and few iterations. (2) JPF and JPDN significantly reduce jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.

摘要: 近年来，多通道大语言模型(MLLMS)在理解多通道内容方面表现出了优越的能力。然而，他们仍然容易受到越狱攻击，这些攻击利用他们的安全调整中的弱点来产生有害的反应。之前的研究根据回应是否包含恶意内容将越狱分为成功或失败。然而，考虑到MLLM响应的随机性，这种对输入越狱MLLMS能力的二进制分类是不合适的。从这一观点出发，我们引入越狱概率来量化输入的越狱潜力，这代表了当提示该输入时MLLMS生成恶意响应的可能性。我们通过对MLLMS的多次查询来近似这一概率。在使用越狱概率预测网络(JPPN)对输入隐藏状态与其对应越狱概率之间的关系进行建模后，我们使用连续越狱概率进行优化。具体地说，我们提出了基于越狱概率的攻击(JPA)，它优化了输入上的敌意扰动，以最大化越狱概率。为了对抗攻击，我们还提出了两种防御方法：基于越狱概率的精调(JPF)和基于越狱概率的防御噪声(JPDN)，它们分别在MLLM参数和输入空间中最小化越狱概率。大量的实验表明：(1)JPA在白盒和黑盒设置下都比以往的方法在小的扰动界和很少的迭代次数下都有很大的改善(高达28.38%)。(2)JPF和JPDN可显著减少越狱次数，最多可减少60%以上。以上两个结果都证明了引入越狱概率来细微区分不同输入越狱能力的意义。



## **9. InferDPT: Privacy-Preserving Inference for Black-box Large Language Model**

InferDPT：黑匣子大型语言模型的隐私保护推理 cs.CR

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2310.12214v7) [paper-pdf](http://arxiv.org/pdf/2310.12214v7)

**Authors**: Meng Tong, Kejiang Chen, Jie Zhang, Yuang Qi, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Zhikun Zhang

**Abstract**: Large language models (LLMs), like ChatGPT, have greatly simplified text generation tasks. However, they have also raised concerns about privacy risks such as data leakage and unauthorized data collection. Existing solutions for privacy-preserving inference face practical challenges related to computation time and communication costs. In this paper, we propose InferDPT, the first practical framework for the privacy-preserving Inference of black-box LLMs, implementing Differential Privacy in Text generation. InferDPT comprises two key modules: the "perturbation module" utilizes the exponential mechanism to generate a perturbed prompt, facilitating privacy-preserving inference with black-box LLMs, and the "extraction module", inspired by knowledge distillation and retrieval-augmented generation, extracts coherent and consistent text from the perturbed generation result, ensuring successful text generation completion. To address privacy concerns related to previous exponential mechanisms' susceptibility to embedding revision attacks, we introduce RANTEXT, a novel differential privacy mechanism integrated into the perturbation module of InferDPT, which introduces the concept of "RANdom adjacency" for TEXT perturbation within the prompt. Experimental results across three datasets demonstrate that the text generation quality of InferDPT is comparable to that of non-private GPT-4, and RANTEXT surpasses existing state-of-the-art mechanisms, namely, SANTEXT+ and CUSTEXT+ in the trade-off between privacy and utility. Even with an privacy parameter epsilon value of 6.0, RANTEXT achieves an average privacy protection rate exceeding 90% against embedding revision attacks, which is 0.58 times higher than that of SANTEXT+ and 3.35 times higher than that of CUSTEXT+.

摘要: 大型语言模型(LLM)，如ChatGPT，极大地简化了文本生成任务。然而，他们也对数据泄露和未经授权的数据收集等隐私风险表示担忧。现有的隐私保护推理解决方案面临着与计算时间和通信成本相关的实际挑战。在本文中，我们提出了第一个实用的黑盒LLMS隐私保护推理框架InferDPT，在文本生成中实现了差分隐私。InferDPT包括两个关键模块：“扰动模块”利用指数机制生成扰动提示，便于使用黑盒LLMS进行隐私保护推理；“提取模块”受知识提炼和检索-增强生成的启发，从扰动生成结果中提取连贯一致的文本，确保文本生成成功完成。针对以往指数机制易受修改攻击的隐私性问题，引入了一种新的差异化隐私机制RANTEXT，该机制集成在InferDPT的扰动模块中，引入了随机邻接的概念来处理提示内的文本扰动。在三个数据集上的实验结果表明，InferDPT的文本生成质量与非私有GPT-4相当，RANTEXT在隐私和效用之间的权衡方面超过了现有的最新机制SanText+和CUSTEXT+。即使在隐私参数epsilon值为6.0的情况下，RANTEXT对嵌入修改攻击的平均隐私保护率也超过90%，比SanText+高0.58倍，比CUSTEXT+高3.35倍。



## **10. Stepwise Reasoning Error Disruption Attack of LLMs**

LLM的逐步推理错误中断攻击 cs.AI

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2412.11934v3) [paper-pdf](http://arxiv.org/pdf/2412.11934v3)

**Authors**: Jingyu Peng, Maolin Wang, Xiangyu Zhao, Kai Zhang, Wanyu Wang, Pengyue Jia, Qidong Liu, Ruocheng Guo, Qi Liu

**Abstract**: Large language models (LLMs) have made remarkable strides in complex reasoning tasks, but their safety and robustness in reasoning processes remain underexplored. Existing attacks on LLM reasoning are constrained by specific settings or lack of imperceptibility, limiting their feasibility and generalizability. To address these challenges, we propose the Stepwise rEasoning Error Disruption (SEED) attack, which subtly injects errors into prior reasoning steps to mislead the model into producing incorrect subsequent reasoning and final answers. Unlike previous methods, SEED is compatible with zero-shot and few-shot settings, maintains the natural reasoning flow, and ensures covert execution without modifying the instruction. Extensive experiments on four datasets across four different models demonstrate SEED's effectiveness, revealing the vulnerabilities of LLMs to disruptions in reasoning processes. These findings underscore the need for greater attention to the robustness of LLM reasoning to ensure safety in practical applications.

摘要: 大语言模型在复杂的推理任务中取得了显著的进展，但其在推理过程中的安全性和稳健性仍未得到充分的研究。现有的对LLM推理的攻击受到特定环境或缺乏不可见性的限制，限制了它们的可行性和普适性。为了应对这些挑战，我们提出了逐步推理错误中断(SEED)攻击，它巧妙地在先前的推理步骤中注入错误，以误导模型产生不正确的后续推理和最终答案。与以往的方法不同，SEED兼容零射和少射设置，保持了自然的推理流程，在不修改指令的情况下确保了隐蔽的执行。在四个不同模型的四个数据集上的广泛实验证明了SEED的有效性，揭示了LLMS在推理过程中对中断的脆弱性。这些发现强调了需要更多地关注LLM推理的健壮性，以确保在实际应用中的安全性。



## **11. Can Watermarking Large Language Models Prevent Copyrighted Text Generation and Hide Training Data?**

对大型语言模型进行水印可以阻止受版权保护的文本生成并隐藏训练数据吗？ cs.LG

19 pages, 7 figures. Published at AAAI 2025. Code will be available  at https://github.com/michael-panaitescu/watermark_copyright_aaai25

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2407.17417v2) [paper-pdf](http://arxiv.org/pdf/2407.17417v2)

**Authors**: Michael-Andrei Panaitescu-Liess, Zora Che, Bang An, Yuancheng Xu, Pankayaraj Pathmanathan, Souradip Chakraborty, Sicheng Zhu, Tom Goldstein, Furong Huang

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in generating diverse and contextually rich text. However, concerns regarding copyright infringement arise as LLMs may inadvertently produce copyrighted material. In this paper, we first investigate the effectiveness of watermarking LLMs as a deterrent against the generation of copyrighted texts. Through theoretical analysis and empirical evaluation, we demonstrate that incorporating watermarks into LLMs significantly reduces the likelihood of generating copyrighted content, thereby addressing a critical concern in the deployment of LLMs. However, we also find that watermarking can have unintended consequences on Membership Inference Attacks (MIAs), which aim to discern whether a sample was part of the pretraining dataset and may be used to detect copyright violations. Surprisingly, we find that watermarking adversely affects the success rate of MIAs, complicating the task of detecting copyrighted text in the pretraining dataset. These results reveal the complex interplay between different regulatory measures, which may impact each other in unforeseen ways. Finally, we propose an adaptive technique to improve the success rate of a recent MIA under watermarking. Our findings underscore the importance of developing adaptive methods to study critical problems in LLMs with potential legal implications.

摘要: 大型语言模型(LLM)在生成丰富多样的文本方面表现出了令人印象深刻的能力。然而，由于LLMS可能无意中产生了受版权保护的材料，因此出现了对侵犯版权的担忧。在这篇文章中，我们首先研究了水印LLM作为对版权文本产生的威慑的有效性。通过理论分析和实证评估，我们证明了在LLMS中加入水印显著降低了产生受版权保护内容的可能性，从而解决了LLMS部署中的一个关键问题。然而，我们也发现，水印可能会对成员关系推断攻击(MIA)产生意想不到的后果，MIA旨在识别样本是否属于预训练数据集，并可能用于检测侵犯版权的行为。令人惊讶的是，我们发现水印对MIA的成功率产生了不利影响，使得在预训练数据集中检测受版权保护的文本的任务变得更加复杂。这些结果揭示了不同监管措施之间复杂的相互作用，这些措施可能会以不可预见的方式相互影响。最后，我们提出了一种自适应技术来提高最近的MIA在水印下的成功率。我们的发现强调了开发适应性方法来研究LLMS中具有潜在法律含义的关键问题的重要性。



## **12. CtrlRAG: Black-box Adversarial Attacks Based on Masked Language Models in Retrieval-Augmented Language Generation**

CtrlRAG：检索增强语言生成中基于掩蔽语言模型的黑匣子对抗攻击 cs.CL

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06950v1) [paper-pdf](http://arxiv.org/pdf/2503.06950v1)

**Authors**: Runqi Sui

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by integrating external knowledge bases. However, this integration introduces a new security threat: adversaries can exploit the retrieval mechanism to inject malicious content into the knowledge base, thereby influencing the generated responses. Based on this attack vector, we propose CtrlRAG, a novel attack method designed for RAG system in the black-box setting, which aligns with real-world scenarios. Unlike existing attack methods, CtrlRAG introduces a perturbation mechanism using Masked Language Model (MLM) to dynamically optimize malicious content in response to changes in the retrieved context. Experimental results demonstrate that CtrlRAG outperforms three baseline methods in both Emotional Manipulation and Hallucination Amplification objectives. Furthermore, we evaluate three existing defense mechanisms, revealing their limited effectiveness against CtrlRAG and underscoring the urgent need for more robust defenses.

摘要: 检索-增强生成(RAG)系统通过集成外部知识库来增强大型语言模型(LLMS)。然而，这种集成带来了新的安全威胁：攻击者可以利用检索机制将恶意内容注入知识库，从而影响生成的响应。基于该攻击向量，我们提出了一种新的针对黑盒环境下RAG系统的攻击方法CtrlRAG，该方法符合实际场景。与现有的攻击方法不同，CtrlRAG引入了一种使用掩蔽语言模型(MLM)的扰动机制来动态优化恶意内容，以响应检索到的上下文的变化。实验结果表明，CtrlRAG在情绪操纵和幻觉放大目标上都优于三种基线方法。此外，我们评估了三种现有的防御机制，揭示了它们对CtrlRAG的有限有效性，并强调了对更强大防御的迫切需要。



## **13. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

探索机器人学中视觉-语言-动作模型的对抗脆弱性 cs.RO

Github: https://github.com/William-wAng618/roboticAttack Homepage:  https://vlaattacker.github.io/

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2411.13587v3) [paper-pdf](http://arxiv.org/pdf/2411.13587v3)

**Authors**: Taowen Wang, Cheng Han, James Chenhao Liang, Wenhao Yang, Dongfang Liu, Luna Xinyu Zhang, Qifan Wang, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. While VLA models offer significant capabilities, they also introduce new attack surfaces, making them vulnerable to adversarial attacks. With these vulnerabilities largely unexplored, this paper systematically quantifies the robustness of VLA-based robotic systems. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce two untargeted attack objectives that leverage spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, we advance both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for continuously developing robust defense strategies prior to physical-world deployments.

摘要: 最近在机器人学中，视觉-语言-动作(VLA)模型作为一种变革性的方法出现，使机器人能够通过在端到端学习框架内整合视觉和语言输入来执行复杂的任务。虽然VLA模型提供了重要的功能，但它们也引入了新的攻击面，使其容易受到对手攻击。由于这些漏洞在很大程度上是未知的，本文系统地量化了基于VLA的机器人系统的健壮性。认识到机器人执行的独特需求，我们的攻击目标针对机器人系统固有的空间和功能特征。特别是，我们引入了两个非定向攻击目标，它们利用空间基础来破坏机器人动作的稳定性，以及一个操纵机器人轨迹的定向攻击目标。此外，我们设计了一种对抗性补丁生成方法，将一个小的、五颜六色的补丁放置在相机的视野中，在数字和物理环境中有效地执行攻击。我们的评估显示任务成功率显著下降，一组模拟机器人任务最多减少100%，突出了当前VLA架构中的关键安全漏洞。通过揭示这些漏洞并提出可操作的评估指标，我们促进了对基于VLA的机器人系统安全性的理解和增强，强调了在物理世界部署之前持续开发强大的防御策略的必要性。



## **14. Privacy Auditing of Large Language Models**

大型语言模型的隐私审计 cs.CR

ICLR 2025

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2503.06808v1) [paper-pdf](http://arxiv.org/pdf/2503.06808v1)

**Authors**: Ashwinee Panda, Xinyu Tang, Milad Nasr, Christopher A. Choquette-Choo, Prateek Mittal

**Abstract**: Current techniques for privacy auditing of large language models (LLMs) have limited efficacy -- they rely on basic approaches to generate canaries which leads to weak membership inference attacks that in turn give loose lower bounds on the empirical privacy leakage. We develop canaries that are far more effective than those used in prior work under threat models that cover a range of realistic settings. We demonstrate through extensive experiments on multiple families of fine-tuned LLMs that our approach sets a new standard for detection of privacy leakage. For measuring the memorization rate of non-privately trained LLMs, our designed canaries surpass prior approaches. For example, on the Qwen2.5-0.5B model, our designed canaries achieve $49.6\%$ TPR at $1\%$ FPR, vastly surpassing the prior approach's $4.2\%$ TPR at $1\%$ FPR. Our method can be used to provide a privacy audit of $\varepsilon \approx 1$ for a model trained with theoretical $\varepsilon$ of 4. To the best of our knowledge, this is the first time that a privacy audit of LLM training has achieved nontrivial auditing success in the setting where the attacker cannot train shadow models, insert gradient canaries, or access the model at every iteration.

摘要: 目前的大型语言模型隐私审计技术的有效性有限--它们依赖于基本的方法来生成金丝雀，这导致了弱的成员关系推断攻击，进而给出了经验隐私泄露的松散下界。我们开发的金丝雀比以前在覆盖一系列现实环境的威胁模型下使用的金丝雀要有效得多。我们通过在多个微调LLM家族上的广泛实验证明，我们的方法为隐私泄露的检测设定了一个新的标准。在测量非私人训练的LLM的记忆率方面，我们设计的金丝雀超过了以前的方法。例如，在Qwen2.5-0.5B模型上，我们设计的金丝雀以1美元的FFP获得了49.6美元的TPR，大大超过了先前方法以1美元的FPR获得的4.2美元的TPR。我们的方法可以用来为理论上$varepsilon$为4训练的模型提供$\varepsilon\约1$的隐私审计。据我们所知，这是第一次在攻击者无法训练阴影模型、插入梯度金丝雀或在每次迭代中访问模型的情况下，LLM训练的隐私审计取得了非平凡的审计成功。



## **15. Can Small Language Models Reliably Resist Jailbreak Attacks? A Comprehensive Evaluation**

小型语言模型能否可靠地抵抗越狱攻击？综合评价 cs.CR

19 pages, 12 figures

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2503.06519v1) [paper-pdf](http://arxiv.org/pdf/2503.06519v1)

**Authors**: Wenhui Zhang, Huiyu Xu, Zhibo Wang, Zeqing He, Ziqi Zhu, Kui Ren

**Abstract**: Small language models (SLMs) have emerged as promising alternatives to large language models (LLMs) due to their low computational demands, enhanced privacy guarantees and comparable performance in specific domains through light-weight fine-tuning. Deploying SLMs on edge devices, such as smartphones and smart vehicles, has become a growing trend. However, the security implications of SLMs have received less attention than LLMs, particularly regarding jailbreak attacks, which is recognized as one of the top threats of LLMs by the OWASP. In this paper, we conduct the first large-scale empirical study of SLMs' vulnerabilities to jailbreak attacks. Through systematically evaluation on 63 SLMs from 15 mainstream SLM families against 8 state-of-the-art jailbreak methods, we demonstrate that 47.6% of evaluated SLMs show high susceptibility to jailbreak attacks (ASR > 40%) and 38.1% of them can not even resist direct harmful query (ASR > 50%). We further analyze the reasons behind the vulnerabilities and identify four key factors: model size, model architecture, training datasets and training techniques. Moreover, we assess the effectiveness of three prompt-level defense methods and find that none of them achieve perfect performance, with detection accuracy varying across different SLMs and attack methods. Notably, we point out that the inherent security awareness play a critical role in SLM security, and models with strong security awareness could timely terminate unsafe response with little reminder. Building upon the findings, we highlight the urgent need for security-by-design approaches in SLM development and provide valuable insights for building more trustworthy SLM ecosystem.

摘要: 小语言模型(SLM)作为大语言模型(LLM)的替代模型，具有较低的计算要求、增强的隐私保护以及通过轻量级的微调在特定领域具有相当的性能。在智能手机和智能汽车等边缘设备上部署SLM已成为一种日益增长的趋势。然而，小武器和轻武器的安全影响受到的关注较少，特别是在越狱攻击方面，而越狱攻击被认为是小岛屿发展中国家的最大威胁之一。在本文中，我们首次对SLM的越狱攻击脆弱性进行了大规模的实证研究。通过对来自15个主流SLM家庭的63个SLM与8种最新越狱方法的系统评估，我们发现47.6%的受评估SLM表现出对越狱攻击的高敏感性(ASR>40%)，其中38.1%的SLM甚至无法抵抗直接有害查询(ASR>50%)。我们进一步分析了漏洞背后的原因，并确定了四个关键因素：模型大小、模型体系结构、训练数据集和训练技术。此外，我们评估了三种提示级防御方法的有效性，发现它们都没有达到完美的性能，检测精度在不同的SLM和攻击方法中有所不同。值得注意的是，固有的安全意识在SLM安全中起着至关重要的作用，安全意识强的模型可以在几乎没有提醒的情况下及时终止不安全的响应。在这些发现的基础上，我们强调了在SLM开发中迫切需要通过设计来实现安全，并为建立更值得信赖的SLM生态系统提供了有价值的见解。



## **16. MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks**

MM-PoisonRAG：通过本地和全球中毒攻击扰乱多模式RAG cs.LG

Code is available at https://github.com/HyeonjeongHa/MM-PoisonRAG

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2502.17832v2) [paper-pdf](http://arxiv.org/pdf/2502.17832v2)

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-Wei Chang, Daniel Kang, Heng Ji

**Abstract**: Multimodal large language models (MLLMs) equipped with Retrieval Augmented Generation (RAG) leverage both their rich parametric knowledge and the dynamic, external knowledge to excel in tasks such as Question Answering. While RAG enhances MLLMs by grounding responses in query-relevant external knowledge, this reliance poses a critical yet underexplored safety risk: knowledge poisoning attacks, where misinformation or irrelevant knowledge is intentionally injected into external knowledge bases to manipulate model outputs to be incorrect and even harmful. To expose such vulnerabilities in multimodal RAG, we propose MM-PoisonRAG, a novel knowledge poisoning attack framework with two attack strategies: Localized Poisoning Attack (LPA), which injects query-specific misinformation in both text and images for targeted manipulation, and Globalized Poisoning Attack (GPA) to provide false guidance during MLLM generation to elicit nonsensical responses across all queries. We evaluate our attacks across multiple tasks, models, and access settings, demonstrating that LPA successfully manipulates the MLLM to generate attacker-controlled answers, with a success rate of up to 56% on MultiModalQA. Moreover, GPA completely disrupts model generation to 0% accuracy with just a single irrelevant knowledge injection. Our results highlight the urgent need for robust defenses against knowledge poisoning to safeguard multimodal RAG frameworks.

摘要: 配备了检索增强生成(RAG)的多通道大型语言模型(MLLMS)利用其丰富的参数知识和动态的外部知识来在问答等任务中脱颖而出。虽然RAG通过将响应建立在与查询相关的外部知识中来增强MLLS，但这种依赖构成了一个关键但未被开发的安全风险：知识中毒攻击，即故意将错误信息或无关知识注入外部知识库，以操纵不正确甚至有害的模型输出。为了暴露多模式RAG中的这些漏洞，我们提出了一种新的知识中毒攻击框架MM-PoisonRAG，该框架具有两种攻击策略：局部中毒攻击(LPA)和全局中毒攻击(GPA)，前者在文本和图像中注入特定于查询的错误信息以进行有针对性的操作，后者在MLLM生成过程中提供错误指导，以引发跨所有查询的无意义响应。我们跨多个任务、模型和访问设置评估我们的攻击，证明LPA成功操纵MLLM生成攻击者控制的答案，在多模式QA上的成功率高达56%。此外，GPA只需注入一次无关的知识，就能完全中断模型生成，准确率为0%。我们的结果突出表明，迫切需要针对知识中毒采取强有力的防御措施，以保护多式联运RAG框架。



## **17. Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions**

数据污染检测对LLM有效吗？检测假设的调查与评价 cs.CL

3 tables and 1 figures in the main text. This paper is accepted by  NAACL 2025 findings

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2410.18966v2) [paper-pdf](http://arxiv.org/pdf/2410.18966v2)

**Authors**: Yujuan Fu, Ozlem Uzuner, Meliha Yetisgen, Fei Xia

**Abstract**: Large language models (LLMs) have demonstrated great performance across various benchmarks, showing potential as general-purpose task solvers. However, as LLMs are typically trained on vast amounts of data, a significant concern in their evaluation is data contamination, where overlap between training data and evaluation datasets inflates performance assessments. Multiple approaches have been developed to identify data contamination. These approaches rely on specific assumptions that may not hold universally across different settings. To bridge this gap, we systematically review 50 papers on data contamination detection, categorize the underlying assumptions, and assess whether they have been rigorously validated. We identify and analyze eight categories of assumptions and test three of them as case studies. Our case studies focus on detecting direct, instance-level data contamination, which is also referred to as Membership Inference Attacks (MIA). Our analysis reveals that MIA approaches based on these three assumptions can have similar performance to random guessing, on datasets used in LLM pretraining, suggesting that current LLMs might learn data distributions rather than memorizing individual instances. Meanwhile, MIA can easily fail when there are data distribution shifts between the seen and unseen instances.

摘要: 大型语言模型(LLM)在各种基准测试中表现出了出色的性能，显示出作为通用任务解算器的潜力。然而，由于LLM通常是根据大量数据进行培训的，其评估中的一个重大问题是数据污染，其中培训数据和评估数据集之间的重叠会夸大业绩评估。已经开发了多种方法来识别数据污染。这些方法依赖于特定的假设，而这些假设在不同的环境中可能并不普遍适用。为了弥补这一差距，我们系统地审查了50篇关于数据污染检测的论文，对潜在的假设进行了分类，并评估它们是否经过了严格的验证。我们识别和分析了八类假设，并测试了其中三类作为案例研究。我们的案例研究重点是检测直接的实例级数据污染，这也称为成员关系推断攻击(MIA)。我们的分析表明，基于这三个假设的MIA方法在LLM预训练中使用的数据集上具有类似于随机猜测的性能，这表明当前的LLM可能学习数据分布而不是记忆单个实例。同时，当可见实例和不可见实例之间存在数据分布变化时，MIA很容易失败。



## **18. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

IDEATOR：使用自己越狱和基准大型视觉语言模型 cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2411.00827v3) [paper-pdf](http://arxiv.org/pdf/2411.00827v3)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLBreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.

摘要: 随着大型视觉语言模型(VLM)的日益突出，确保它们的安全部署变得至关重要。最近的研究探索了VLM对越狱攻击的稳健性--利用模型漏洞来引发有害输出的技术。然而，各种多模式数据的可获得性有限，限制了目前的方法严重依赖从有害文本数据集获得的对抗性或手动制作的图像，这些图像往往缺乏跨不同背景的有效性和多样性。在本文中，我们提出了一种新的越狱方法--IDEATOR，它能够自动生成用于黑盒越狱攻击的恶意图文对。Ideator基于这样的见解，即VLM本身可以作为生成多模式越狱提示的强大红色团队模型。具体地说，Ideator利用VLM创建有针对性的越狱文本，并将它们与由最先进的扩散模型生成的越狱图像配对。大量的实验证明了IDAIDATER的高效率和可移植性，在越狱MiniGPT-4上达到了94%的攻击成功率(ASR)，平均只有5.34个查询，当转移到LLaVA、InstructBLIP和Chameleon时，ASR分别达到了82%、88%和75%。基于IDEATOR强大的可转移性和自动化流程，我们引入了VLBreakB边，这是一个包含3654个多模式越狱样本的安全基准。我们在最近发布的11个VLM上的基准结果显示，在安全对准方面存在显著差距。例如，我们的挑战集在GPT-40上达到了46.31%的ASR，在Claude-3.5-十四行诗上达到了19.65%，这突显了对更强大防御的迫切需要。



## **19. Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models**

使用机械可解释性来应对大型语言模型的对抗攻击 cs.LG

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06269v1) [paper-pdf](http://arxiv.org/pdf/2503.06269v1)

**Authors**: Thomas Winninger, Boussad Addad, Katarzyna Kapusta

**Abstract**: Traditional white-box methods for creating adversarial perturbations against LLMs typically rely only on gradient computation from the targeted model, ignoring the internal mechanisms responsible for attack success or failure. Conversely, interpretability studies that analyze these internal mechanisms lack practical applications beyond runtime interventions. We bridge this gap by introducing a novel white-box approach that leverages mechanistic interpretability techniques to craft practical adversarial inputs. Specifically, we first identify acceptance subspaces - sets of feature vectors that do not trigger the model's refusal mechanisms - then use gradient-based optimization to reroute embeddings from refusal subspaces to acceptance subspaces, effectively achieving jailbreaks. This targeted approach significantly reduces computation cost, achieving attack success rates of 80-95\% on state-of-the-art models including Gemma2, Llama3.2, and Qwen2.5 within minutes or even seconds, compared to existing techniques that often fail or require hours of computation. We believe this approach opens a new direction for both attack research and defense development. Furthermore, it showcases a practical application of mechanistic interpretability where other methods are less efficient, which highlights its utility. The code and generated datasets are available at https://github.com/Sckathach/subspace-rerouting.

摘要: 传统的白盒方法用于创建针对LLM的对抗性扰动，通常只依赖于目标模型的梯度计算，而忽略了攻击成败的内部机制。相反，分析这些内部机制的可解释性研究缺乏运行时干预之外的实际应用。我们通过引入一种新的白盒方法来弥合这一差距，该方法利用机械性的可解释性技术来制作实际的对抗性输入。具体地说，我们首先识别接受子空间-不会触发模型的拒绝机制的特征向量集合-然后使用基于梯度的优化将嵌入从拒绝子空间重定向到接受子空间，从而有效地实现越狱。这种有针对性的方法显著降低了计算成本，与通常失败或需要数小时计算的现有技术相比，在Gemma2、Llama3.2和Qwen2.5等最先进的模型上在几分钟甚至几秒钟内实现了80%-95%的攻击成功率。我们相信，这种方法为攻击研究和防御发展开辟了新的方向。此外，它还展示了机械可解释性在其他方法效率较低的情况下的实际应用，这突出了它的实用性。代码和生成的数据集可在https://github.com/Sckathach/subspace-rerouting.上获得



## **20. Reinforced Diffuser for Red Teaming Large Vision-Language Models**

用于Red团队大型视觉语言模型的增强扩散器 cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06223v1) [paper-pdf](http://arxiv.org/pdf/2503.06223v1)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The rapid advancement of large Vision-Language Models (VLMs) has raised significant safety concerns, particularly regarding their vulnerability to jailbreak attacks. While existing research primarily focuses on VLMs' susceptibility to harmful instructions, this work identifies a critical yet overlooked vulnerability: current alignment mechanisms often fail to address the risks posed by toxic text continuation tasks. To investigate this issue, we propose a novel Red Team Diffuser (RTD) framework, which leverages reinforcement learning to generate red team images that effectively induce highly toxic continuations from target black-box VLMs. The RTD pipeline begins with a greedy search for high-quality image prompts that maximize the toxicity of VLM-generated sentence continuations, guided by a Large Language Model (LLM). These prompts are then used as input for the reinforcement fine-tuning of a diffusion model, which employs toxicity and alignment rewards to further amplify harmful outputs. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% on the original attack set and 8.91% on a hold-out set. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. These findings reveal significant deficiencies in existing alignment strategies, particularly their inability to prevent harmful continuations. Our work underscores the urgent need for more robust and adaptive alignment mechanisms to ensure the safe deployment of VLMs in real-world applications.

摘要: 大型视觉语言模型(VLM)的快速发展引发了重大的安全问题，特别是它们对越狱攻击的脆弱性。虽然现有的研究主要集中在VLMS对有害指令的易感性上，但这项工作发现了一个严重但被忽视的漏洞：当前的比对机制往往无法解决有毒文本继续任务带来的风险。为了研究这一问题，我们提出了一种新的红色团队扩散器(RTD)框架，它利用强化学习来生成红色团队图像，从而有效地从目标黑盒VLM中诱导出剧毒的延续。RTD流水线从贪婪地搜索高质量的图像提示开始，这些图像提示在大型语言模型(LLM)的指导下，最大限度地提高了VLM生成的句子延续的毒性。这些提示随后被用作扩散模型的强化微调的输入，该模型使用毒性和配对奖励来进一步放大有害输出。实验结果证明了RTD的有效性，在原始攻击集和抵抗集上，LLaVA输出的毒力率分别提高了10.69%和8.91%。此外，RTD具有很强的跨模型传递性，对双子座和骆驼的毒性分别提高了5.1%和26.83%。这些调查结果揭示了现有调整战略的重大缺陷，特别是它们无法防止有害的延续。我们的工作强调了迫切需要更强大和适应性更强的对准机制，以确保在现实世界应用中安全地部署VLM。



## **21. Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents**

代理安全工作台（ASB）：对基于LLM的代理中的攻击和防御进行形式化和基准化 cs.CR

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2410.02644v2) [paper-pdf](http://arxiv.org/pdf/2410.02644v2)

**Authors**: Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, Yongfeng Zhang

**Abstract**: Although LLM-based agents, powered by Large Language Models (LLMs), can use external tools and memory mechanisms to solve complex real-world tasks, they may also introduce critical security vulnerabilities. However, the existing literature does not comprehensively evaluate attacks and defenses against LLM-based agents. To address this, we introduce Agent Security Bench (ASB), a comprehensive framework designed to formalize, benchmark, and evaluate the attacks and defenses of LLM-based agents, including 10 scenarios (e.g., e-commerce, autonomous driving, finance), 10 agents targeting the scenarios, over 400 tools, 27 different types of attack/defense methods, and 7 evaluation metrics. Based on ASB, we benchmark 10 prompt injection attacks, a memory poisoning attack, a novel Plan-of-Thought backdoor attack, 4 mixed attacks, and 11 corresponding defenses across 13 LLM backbones. Our benchmark results reveal critical vulnerabilities in different stages of agent operation, including system prompt, user prompt handling, tool usage, and memory retrieval, with the highest average attack success rate of 84.30\%, but limited effectiveness shown in current defenses, unveiling important works to be done in terms of agent security for the community. We also introduce a new metric to evaluate the agents' capability to balance utility and security. Our code can be found at https://github.com/agiresearch/ASB.

摘要: 尽管基于大型语言模型(LLM)的代理可以使用外部工具和内存机制来解决复杂的现实任务，但它们也可能引入关键的安全漏洞。然而，现有的文献并没有全面评估对基于LLM的代理的攻击和防御。为了解决这一问题，我们引入了代理安全平台(ASB)，这是一个全面的框架，旨在形式化、基准和评估基于LLM的代理的攻击和防御，包括10个场景(例如，电子商务、自动驾驶、金融)、10个针对场景的代理、400多个工具、27种不同类型的攻击/防御方法和7个评估指标。在ASB的基础上，我们对10个即时注入攻击、一个内存中毒攻击、一个新的思维计划后门攻击、4个混合攻击以及13个LLM主干上的11个相应防御进行了基准测试。我们的测试结果揭示了代理操作的不同阶段的关键漏洞，包括系统提示、用户提示处理、工具使用和内存恢复，平均攻击成功率最高为84.30\%，但现有防御措施的有效性有限，揭示了社区在代理安全方面需要做的重要工作。我们还引入了一个新的度量来评估代理在效用和安全性之间的平衡能力。我们的代码可以在https://github.com/agiresearch/ASB.上找到



## **22. Are Your LLM-based Text-to-SQL Models Secure? Exploring SQL Injection via Backdoor Attacks**

您的基于LLM的文本到SQL模型安全吗？通过后门攻击探索SQL注入 cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05445v1) [paper-pdf](http://arxiv.org/pdf/2503.05445v1)

**Authors**: Meiyu Lin, Haichuan Zhang, Jiale Lao, Renyuan Li, Yuanchun Zhou, Carl Yang, Yang Cao, Mingjie Tang

**Abstract**: Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.

摘要: 大型语言模型(LLM)在将自然语言问题转换为SQL查询(Text-to-SQL)方面取得了最先进的结果，这是数据库社区内的一个长期挑战。然而，安全问题在很大程度上仍然没有得到探索，特别是后门攻击的威胁，后门攻击可以通过微调有毒数据集将恶意行为引入模型。在这项工作中，我们系统地研究了基于LLM的Text-to-SQL模型的漏洞，并提出了一种新的后门攻击框架ToxicSQL。我们的方法利用隐蔽的(语义和字符级触发器)使后门难以检测和删除，从而确保恶意行为保持隐蔽性，同时保持对良性输入的高模型准确性。此外，我们建议利用SQL注入有效负载作为后门目标，从而能够生成恶意但可执行的SQL查询，这在基于语言模型的SQL开发中会带来严重的安全和隐私风险。我们证明，仅注入0.44%的有毒数据就可以导致79.41%的攻击成功率，这对数据库安全构成了重大风险。此外，我们还提出了检测和缓解策略，以增强模型的可靠性。我们的发现强调了安全意识文本到SQL开发的迫切需要，强调了强大的后门威胁防御的重要性。



## **23. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

DetectRL：在现实世界场景中对LLM生成的文本检测进行基准测试 cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2410.23746v2) [paper-pdf](http://arxiv.org/pdf/2410.23746v2)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.

摘要: 检测由大型语言模型(LLM)生成的文本是最近非常感兴趣的问题。有了像DetectGPT这样的零射击方法，检测能力已经达到了令人印象深刻的水平。然而，现有探测器在实际应用中的可靠性仍然没有得到充分的探索。在这项研究中，我们提出了一个新的基准，DetectRL，强调即使是最先进的(SOTA)检测技术在这项任务中仍然表现不佳。我们从LLM特别容易被滥用的领域收集了人类编写的数据集。使用流行的LLM，我们生成的数据更好地与现实世界的应用程序保持一致。与以前的研究不同，我们使用启发式规则来创建对抗性LLM生成的文本，模拟各种提示用法、人工修改(如单词替换)和书写噪音(如拼写错误)。我们对DetectRL的开发揭示了当前SOTA探测器的优势和局限性。更重要的是，我们分析了写作风格、模型类型、攻击方法、文本长度和真实世界中的人类写作因素对不同类型检测器的潜在影响。我们相信，DetectRL可以作为评估真实世界场景中检测器的有效基准，随着先进攻击方法的发展，从而提供更有压力的评估，以推动更高效检测器的开发。数据和代码可在以下网址公开获得：https://github.com/NLP2CT/DetectRL.



## **24. A Practical Memory Injection Attack against LLM Agents**

针对LLM代理的实用内存注入攻击 cs.LG

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.03704v2) [paper-pdf](http://arxiv.org/pdf/2503.03704v2)

**Authors**: Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, Zhen Xiang

**Abstract**: Agents based on large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, that enables the injection of malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps leading to undesirable agent actions when executing the victim user's query. Specifically, we introduce a sequence of bridging steps to link the victim query to the malicious reasoning steps. During the injection of the malicious record, we propose an indication prompt to guide the agent to autonomously generate our designed bridging steps. We also propose a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing the victim query comes after. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting practical risks of LLM agents.

摘要: 基于大型语言模型(LLM)的代理在广泛的复杂、真实世界的应用中表现出了强大的能力。然而，当检索用于演示的过去记录是恶意的时，具有受损内存库的LLM代理可能很容易产生有害输出。在本文中，我们提出了一种新的内存注入攻击，MinJA，它只需通过查询和输出观察与代理交互，就可以将恶意记录注入到内存库中。这些恶意记录旨在引发一系列恶意推理步骤，从而在执行受攻击用户的查询时导致不受欢迎的代理操作。具体地说，我们引入了一系列桥接步骤来将受害者查询与恶意推理步骤联系起来。在注入恶意记录的过程中，我们提出了一个指示提示，以引导代理自主生成我们设计的桥接步骤。我们还提出了一种渐进式缩短策略，逐步删除指示提示，以便在处理后续受害者查询时能够轻松检索到恶意记录。我们在不同代理上的广泛实验证明了Minja在损害代理内存方面的有效性。在执行要求最低的情况下，Minja使任何用户都能够影响代理内存，突出了LLM代理的实际风险。



## **25. Double Backdoored: Converting Code Large Language Model Backdoors to Traditional Malware via Adversarial Instruction Tuning Attacks**

双重后门：通过对抗性指令调优攻击将代码大型语言模型后门转换为传统恶意软件 cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2404.18567v2) [paper-pdf](http://arxiv.org/pdf/2404.18567v2)

**Authors**: Md Imran Hossen, Sai Venkatesh Chilukoti, Liqun Shan, Sheng Chen, Yinzhi Cao, Xiali Hei

**Abstract**: Instruction-tuned Large Language Models designed for coding tasks are increasingly employed as AI coding assistants. However, the cybersecurity vulnerabilities and implications arising from the widespread integration of these models are not yet fully understood due to limited research in this domain. This work investigates novel techniques for transitioning backdoors from the AI/ML domain to traditional computer malware, shedding light on the critical intersection of AI and cyber/software security. To explore this intersection, we present MalInstructCoder, a framework designed to comprehensively assess the cybersecurity vulnerabilities of instruction-tuned Code LLMs. MalInstructCoder introduces an automated data poisoning pipeline to inject malicious code snippets into benign code, poisoning instruction fine-tuning data while maintaining functional validity. It presents two practical adversarial instruction tuning attacks with real-world security implications: the clean prompt poisoning attack and the backdoor attack. These attacks aim to manipulate Code LLMs to generate code incorporating malicious or harmful functionality under specific attack scenarios while preserving intended functionality. We conduct a comprehensive investigation into the exploitability of the code-specific instruction tuning process involving three state-of-the-art Code LLMs: CodeLlama, DeepSeek-Coder, and StarCoder2. Our findings reveal that these models are highly vulnerable to our attacks. Specifically, the clean prompt poisoning attack achieves the ASR@1 ranging from over 75% to 86% by poisoning only 1% (162 samples) of the instruction fine-tuning dataset. Similarly, the backdoor attack achieves the ASR@1 ranging from 76% to 86% with a 0.5% poisoning rate. Our study sheds light on the critical cybersecurity risks posed by instruction-tuned Code LLMs and highlights the urgent need for robust defense mechanisms.

摘要: 为编码任务而设计的指令调整的大型语言模型越来越多地被用作人工智能编码助手。然而，由于这一领域的研究有限，这些模型的广泛集成所产生的网络安全漏洞和影响尚未完全了解。这项工作研究了将后门从AI/ML领域过渡到传统计算机恶意软件的新技术，揭示了人工智能和网络/软件安全的关键交集。为了探索这一交叉，我们提出了MalInstructCoder，一个旨在全面评估指令调优代码LLM的网络安全漏洞的框架。MalInstructCoder引入了自动数据中毒管道，将恶意代码片段注入良性代码，在保持功能有效性的同时毒化指令微调数据。它提出了两种具有实际安全含义的对抗性指令调整攻击：干净的即时中毒攻击和后门攻击。这些攻击旨在操纵Code LLM在特定攻击场景下生成包含恶意或有害功能的代码，同时保留预期功能。我们对代码特定指令调优过程的可利用性进行了全面的调查，涉及三个最先进的代码LLM：CodeLlama、DeepSeek-Coder和StarCoder2。我们的发现表明，这些模型非常容易受到我们的攻击。具体地说，干净的即时中毒攻击通过仅对指令微调数据集的1%(162个样本)下毒来实现从超过75%到86%的ASR@1。同样，后门攻击实现了从76%到86%的ASR@1，投毒率为0.5%。我们的研究揭示了指令调整的代码LLM构成的关键网络安全风险，并强调了对强大防御机制的迫切需要。



## **26. Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety**

安全不仅仅是拒绝：推理增强微调，以实现可解释的LLM安全 cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.05021v1) [paper-pdf](http://arxiv.org/pdf/2503.05021v1)

**Authors**: Yuyou Zhang, Miao Li, William Han, Yihang Yao, Zhepeng Cen, Ding Zhao

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks that exploit weaknesses in traditional safety alignment, which often relies on rigid refusal heuristics or representation engineering to block harmful outputs. While they are effective for direct adversarial attacks, they fall short of broader safety challenges requiring nuanced, context-aware decision-making. To address this, we propose Reasoning-enhanced Finetuning for interpretable LLM Safety (Rational), a novel framework that trains models to engage in explicit safe reasoning before response. Fine-tuned models leverage the extensive pretraining knowledge in self-generated reasoning to bootstrap their own safety through structured reasoning, internalizing context-sensitive decision-making. Our findings suggest that safety extends beyond refusal, requiring context awareness for more robust, interpretable, and adaptive responses. Reasoning is not only a core capability of LLMs but also a fundamental mechanism for LLM safety. Rational employs reasoning-enhanced fine-tuning, allowing it to reject harmful prompts while providing meaningful and context-aware responses in complex scenarios.

摘要: 大型语言模型(LLM)容易受到越狱攻击，这些攻击利用了传统安全对齐中的弱点，传统安全对齐通常依赖僵化的拒绝启发式或表示工程来阻止有害输出。虽然它们对直接对抗性攻击是有效的，但它们不能满足更广泛的安全挑战，需要细致入微的、上下文感知的决策。为了解决这个问题，我们提出了针对可解释LLM安全(Rational)的推理增强精调，这是一个新的框架，它训练模型在响应之前进行显式的安全推理。微调模型利用自生成推理中丰富的预训练知识，通过结构化推理来引导自己的安全性，使上下文敏感决策内在化。我们的发现表明，安全超越了拒绝，需要背景感知才能做出更健壮、可解释和适应性更强的反应。推理是LLMS的核心能力，也是LLMS安全的基本机制。Rational采用了推理增强的微调，使其能够拒绝有害提示，同时在复杂场景中提供有意义的上下文感知响应。



## **27. The Last Iterate Advantage: Empirical Auditing and Principled Heuristic Analysis of Differentially Private SGD**

最后的迭代优势：差异化私人新元的经验审计和原则性启发式分析 cs.CR

ICLR 2025 camera-ready version

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2410.06186v4) [paper-pdf](http://arxiv.org/pdf/2410.06186v4)

**Authors**: Thomas Steinke, Milad Nasr, Arun Ganesh, Borja Balle, Christopher A. Choquette-Choo, Matthew Jagielski, Jamie Hayes, Abhradeep Guha Thakurta, Adam Smith, Andreas Terzis

**Abstract**: We propose a simple heuristic privacy analysis of noisy clipped stochastic gradient descent (DP-SGD) in the setting where only the last iterate is released and the intermediate iterates remain hidden. Namely, our heuristic assumes a linear structure for the model.   We show experimentally that our heuristic is predictive of the outcome of privacy auditing applied to various training procedures. Thus it can be used prior to training as a rough estimate of the final privacy leakage. We also probe the limitations of our heuristic by providing some artificial counterexamples where it underestimates the privacy leakage.   The standard composition-based privacy analysis of DP-SGD effectively assumes that the adversary has access to all intermediate iterates, which is often unrealistic. However, this analysis remains the state of the art in practice. While our heuristic does not replace a rigorous privacy analysis, it illustrates the large gap between the best theoretical upper bounds and the privacy auditing lower bounds and sets a target for further work to improve the theoretical privacy analyses. We also empirically support our heuristic and show existing privacy auditing attacks are bounded by our heuristic analysis in both vision and language tasks.

摘要: 在只释放最后一次迭代而隐藏中间迭代的情况下，提出了一种简单的启发式噪声截断随机梯度下降(DP-SGD)隐私分析方法。也就是说，我们的启发式假设模型是线性结构。我们的实验表明，我们的启发式方法可以预测隐私审计应用于各种训练过程的结果。因此，它可以在培训前用作最终隐私泄露的粗略估计。我们还通过提供一些低估隐私泄露的人工反例来探讨我们的启发式算法的局限性。标准的基于组合的DP-SGD隐私分析有效地假设攻击者可以访问所有中间迭代，这通常是不现实的。然而，这种分析在实践中仍然是最先进的。虽然我们的启发式方法没有取代严格的隐私分析，但它说明了最佳理论上限和隐私审计下限之间的巨大差距，并为进一步改进理论隐私分析设定了目标。我们还实证地支持我们的启发式攻击，并表明现有的隐私审计攻击受到我们在视觉和语言任务中的启发式分析的约束。



## **28. Get my drift? Catching LLM Task Drift with Activation Deltas**

明白我的意思了吗？利用激活增量捕捉LLM任务漂移 cs.CR

SaTML 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2406.00799v6) [paper-pdf](http://arxiv.org/pdf/2406.00799v6)

**Authors**: Sahar Abdelnabi, Aideen Fay, Giovanni Cherubin, Ahmed Salem, Mario Fritz, Andrew Paverd

**Abstract**: LLMs are commonly used in retrieval-augmented applications to execute user instructions based on data from external sources. For example, modern search engines use LLMs to answer queries based on relevant search results; email plugins summarize emails by processing their content through an LLM. However, the potentially untrusted provenance of these data sources can lead to prompt injection attacks, where the LLM is manipulated by natural language instructions embedded in the external data, causing it to deviate from the user's original instruction(s). We define this deviation as task drift. Task drift is a significant concern as it allows attackers to exfiltrate data or influence the LLM's output for other users. We study LLM activations as a solution to detect task drift, showing that activation deltas - the difference in activations before and after processing external data - are strongly correlated with this phenomenon. Through two probing methods, we demonstrate that a simple linear classifier can detect drift with near-perfect ROC AUC on an out-of-distribution test set. We evaluate these methods by making minimal assumptions about how users' tasks, system prompts, and attacks can be phrased. We observe that this approach generalizes surprisingly well to unseen task domains, such as prompt injections, jailbreaks, and malicious instructions, without being trained on any of these attacks. Interestingly, the fact that this solution does not require any modifications to the LLM (e.g., fine-tuning), as well as its compatibility with existing meta-prompting solutions, makes it cost-efficient and easy to deploy. To encourage further research on activation-based task inspection, decoding, and interpretability, we release our large-scale TaskTracker toolkit, featuring a dataset of over 500K instances, representations from six SoTA language models, and a suite of inspection tools.

摘要: LLM通常用于检索增强的应用程序中，以基于来自外部源的数据执行用户指令。例如，现代搜索引擎使用LLM根据相关搜索结果回答查询；电子邮件插件通过LLM处理电子邮件的内容来汇总电子邮件。然而，这些数据源的潜在不可信来源可能导致提示注入攻击，其中LLM被嵌入外部数据的自然语言指令操纵，导致其偏离用户的原始指令(S)。我们将这种偏差定义为任务漂移。任务漂移是一个重要的问题，因为它允许攻击者窃取数据或影响LLM对其他用户的输出。我们研究了LLM激活作为检测任务漂移的解决方案，表明激活增量-处理外部数据之前和之后的激活差异-与这一现象密切相关。通过两种探测方法，我们证明了一个简单的线性分类器可以在非分布测试集上以接近完美的ROC AUC来检测漂移。我们通过对用户任务、系统提示和攻击的措辞做出最小假设来评估这些方法。我们观察到，这种方法对看不见的任务领域(如提示注入、越狱和恶意指令)的泛化效果出奇地好，而且没有接受过任何这些攻击的培训。有趣的是，该解决方案不需要对LLM进行任何修改(例如微调)，并且它与现有的元提示解决方案兼容，这使得它具有成本效益并且易于部署。为了鼓励对基于激活的任务检测、解码和可解释性的进一步研究，我们发布了我们的大型TaskTracker工具包，其中包括超过50万个实例的数据集、来自六个SOTA语言模型的表示，以及一套检测工具。



## **29. Know Thy Judge: On the Robustness Meta-Evaluation of LLM Safety Judges**

了解你的法官：LLM安全法官的稳健性元评估 cs.LG

Accepted to the ICBINB Workshop at ICLR'25

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04474v1) [paper-pdf](http://arxiv.org/pdf/2503.04474v1)

**Authors**: Francisco Eiras, Eliott Zemour, Eric Lin, Vaikkunth Mugunthan

**Abstract**: Large Language Model (LLM) based judges form the underpinnings of key safety evaluation processes such as offline benchmarking, automated red-teaming, and online guardrailing. This widespread requirement raises the crucial question: can we trust the evaluations of these evaluators? In this paper, we highlight two critical challenges that are typically overlooked: (i) evaluations in the wild where factors like prompt sensitivity and distribution shifts can affect performance and (ii) adversarial attacks that target the judge. We highlight the importance of these through a study of commonly used safety judges, showing that small changes such as the style of the model output can lead to jumps of up to 0.24 in the false negative rate on the same dataset, whereas adversarial attacks on the model generation can fool some judges into misclassifying 100% of harmful generations as safe ones. These findings reveal gaps in commonly used meta-evaluation benchmarks and weaknesses in the robustness of current LLM judges, indicating that low attack success under certain judges could create a false sense of security.

摘要: 基于大型语言模型(LLM)的评委构成了关键安全评估流程的基础，如离线基准、自动红色团队和在线护栏。这一普遍的要求提出了一个关键问题：我们能相信这些评估者的评价吗？在这篇文章中，我们强调了两个通常被忽视的关键挑战：(I)在野外评估中，敏感度和分布变化等因素会影响绩效；(Ii)针对法官的对抗性攻击。我们通过对常用安全法官的研究来强调这些的重要性，表明微小的变化，如模型输出的风格，可以导致同一数据集上的假阴性率跃升高达0.24，而对模型生成的对抗性攻击可能会欺骗一些法官，将100%的有害世代错误分类为安全世代。这些发现揭示了常用元评估基准的差距和当前LLM法官稳健性方面的弱点，表明在某些法官的领导下，低攻击成功率可能会产生一种错误的安全感。



## **30. Stealthy Jailbreak Attacks on Large Language Models via Benign Data Mirroring**

通过良性数据镜像对大型语言模型进行秘密越狱攻击 cs.CL

Accepted by NAACL 2025

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2410.21083v2) [paper-pdf](http://arxiv.org/pdf/2410.21083v2)

**Authors**: Honglin Mu, Han He, Yuxin Zhou, Yunlong Feng, Yang Xu, Libo Qin, Xiaoming Shi, Zeming Liu, Xudong Han, Qi Shi, Qingfu Zhu, Wanxiang Che

**Abstract**: Large language model (LLM) safety is a critical issue, with numerous studies employing red team testing to enhance model security. Among these, jailbreak methods explore potential vulnerabilities by crafting malicious prompts that induce model outputs contrary to safety alignments. Existing black-box jailbreak methods often rely on model feedback, repeatedly submitting queries with detectable malicious instructions during the attack search process. Although these approaches are effective, the attacks may be intercepted by content moderators during the search process. We propose an improved transfer attack method that guides malicious prompt construction by locally training a mirror model of the target black-box model through benign data distillation. This method offers enhanced stealth, as it does not involve submitting identifiable malicious instructions to the target model during the search phase. Our approach achieved a maximum attack success rate of 92%, or a balanced value of 80% with an average of 1.5 detectable jailbreak queries per sample against GPT-3.5 Turbo on a subset of AdvBench. These results underscore the need for more robust defense mechanisms.

摘要: 大型语言模型(LLM)的安全性是一个关键问题，许多研究使用RED团队测试来增强模型的安全性。其中，越狱方法通过精心编制恶意提示来探测潜在的漏洞，这些提示会诱导与安全对齐相反的模型输出。现有的黑盒越狱方法通常依赖于模型反馈，在攻击搜索过程中反复提交带有可检测到的恶意指令的查询。虽然这些方法是有效的，但这些攻击可能会在搜索过程中被内容版主拦截。提出了一种改进的传输攻击方法，该方法通过良性数据提炼对目标黑盒模型的镜像模型进行局部训练，指导恶意提示的构建。这种方法提供了增强的隐蔽性，因为它不涉及在搜索阶段向目标模型提交可识别的恶意指令。我们的方法获得了92%的最大攻击成功率，或者说80%的平衡值，每个样本平均有1.5个可检测到的越狱查询，而GPT-3.5Turbo在AdvBitch子集上的攻击成功率为92%。这些结果突显了需要更强大的防御机制。



## **31. Exploring the Multilingual NLG Evaluation Abilities of LLM-Based Evaluators**

探索基于LLM的评估者的多语言NLG评估能力 cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04360v1) [paper-pdf](http://arxiv.org/pdf/2503.04360v1)

**Authors**: Jiayi Chang, Mingqi Gao, Xinyu Hu, Xiaojun Wan

**Abstract**: Previous research has shown that LLMs have potential in multilingual NLG evaluation tasks. However, existing research has not fully explored the differences in the evaluation capabilities of LLMs across different languages. To this end, this study provides a comprehensive analysis of the multilingual evaluation performance of 10 recent LLMs, spanning high-resource and low-resource languages through correlation analysis, perturbation attacks, and fine-tuning. We found that 1) excluding the reference answer from the prompt and using large-parameter LLM-based evaluators leads to better performance across various languages; 2) most LLM-based evaluators show a higher correlation with human judgments in high-resource languages than in low-resource languages; 3) in the languages where they are most sensitive to such attacks, they also tend to exhibit the highest correlation with human judgments; and 4) fine-tuning with data from a particular language yields a broadly consistent enhancement in the model's evaluation performance across diverse languages. Our findings highlight the imbalance in LLMs'evaluation capabilities across different languages and suggest that low-resource language scenarios deserve more attention.

摘要: 以前的研究表明，LLMS在多语言NLG评估任务中具有潜力。然而，现有的研究还没有充分探讨不同语言的学习记忆评估能力的差异。为此，本研究通过相关性分析、扰动攻击和微调，全面分析了最近10个LLMS的多语言评估性能，涵盖了高资源和低资源两种语言。我们发现，1)从提示中排除参考答案，并使用基于大参数LLM的评估器可以在各种语言中获得更好的性能；2)大多数基于LLM的评估器在高资源语言中与人类判断的相关性高于在低资源语言中的相关性；3)在对此类攻击最敏感的语言中，他们也倾向于与人类判断表现出最高的相关性；以及4)使用来自特定语言的数据进行微调可以在不同语言中产生大致一致的评估性能增强。我们的发现突显了不同语言的LLMS评估能力的不平衡，并表明低资源的语言情景值得更多地关注。



## **32. Malware Detection at the Edge with Lightweight LLMs: A Performance Evaluation**

使用轻量级LLM进行边缘恶意软件检测：性能评估 cs.CR

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04302v1) [paper-pdf](http://arxiv.org/pdf/2503.04302v1)

**Authors**: Christian Rondanini, Barbara Carminati, Elena Ferrari, Antonio Gaudiano, Ashish Kundu

**Abstract**: The rapid evolution of malware attacks calls for the development of innovative detection methods, especially in resource-constrained edge computing. Traditional detection techniques struggle to keep up with modern malware's sophistication and adaptability, prompting a shift towards advanced methodologies like those leveraging Large Language Models (LLMs) for enhanced malware detection. However, deploying LLMs for malware detection directly at edge devices raises several challenges, including ensuring accuracy in constrained environments and addressing edge devices' energy and computational limits. To tackle these challenges, this paper proposes an architecture leveraging lightweight LLMs' strengths while addressing limitations like reduced accuracy and insufficient computational power. To evaluate the effectiveness of the proposed lightweight LLM-based approach for edge computing, we perform an extensive experimental evaluation using several state-of-the-art lightweight LLMs. We test them with several publicly available datasets specifically designed for edge and IoT scenarios and different edge nodes with varying computational power and characteristics.

摘要: 恶意软件攻击的快速演变要求开发创新的检测方法，特别是在资源受限的边缘计算中。传统的检测技术很难跟上现代恶意软件的复杂性和适应性，这促使人们转向先进的方法，比如利用大型语言模型(LLM)来增强恶意软件检测。然而，直接在边缘设备上部署LLM进行恶意软件检测带来了几个挑战，包括确保在受限环境中的准确性，以及解决边缘设备的能源和计算限制。为了应对这些挑战，本文提出了一种利用轻量级LLMS的优势的体系结构，同时解决了诸如精度降低和计算能力不足等限制。为了评估提出的基于LLM的轻量级边缘计算方法的有效性，我们使用几个最先进的轻量级LLM进行了广泛的实验评估。我们使用几个专门为EDGE和物联网场景设计的公开可用的数据集以及不同计算能力和特征的不同边缘节点来测试它们。



## **33. One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs**

一次性即可：将多回合攻击整合为LLM的高效单回合攻击 cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.04856v1) [paper-pdf](http://arxiv.org/pdf/2503.04856v1)

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim

**Abstract**: Despite extensive safety enhancements in large language models (LLMs), multi-turn "jailbreak" conversations crafted by skilled human adversaries can still breach even the most sophisticated guardrails. However, these multi-turn attacks demand considerable manual effort, limiting their scalability. In this work, we introduce a novel approach called Multi-turn-to-Single-turn (M2S) that systematically converts multi-turn jailbreak prompts into single-turn attacks. Specifically, we propose three conversion strategies - Hyphenize, Numberize, and Pythonize - each preserving sequential context yet packaging it in a single query. Our experiments on the Multi-turn Human Jailbreak (MHJ) dataset show that M2S often increases or maintains high Attack Success Rates (ASRs) compared to original multi-turn conversations. Notably, using a StrongREJECT-based evaluation of harmfulness, M2S achieves up to 95.9% ASR on Mistral-7B and outperforms original multi-turn prompts by as much as 17.5% in absolute improvement on GPT-4o. Further analysis reveals that certain adversarial tactics, when consolidated into a single prompt, exploit structural formatting cues to evade standard policy checks. These findings underscore that single-turn attacks - despite being simpler and cheaper to conduct - can be just as potent, if not more, than their multi-turn counterparts. Our findings underscore the urgent need to reevaluate and reinforce LLM safety strategies, given how adversarial queries can be compacted into a single prompt while still retaining sufficient complexity to bypass existing safety measures.

摘要: 尽管在大型语言模型(LLM)中进行了广泛的安全增强，但由熟练的人类对手制作的多轮“越狱”对话仍然可以突破最复杂的护栏。然而，这些多回合攻击需要大量的人工工作，限制了它们的可扩展性。在这项工作中，我们提出了一种新的方法，称为多回合到单回合(M2S)，它系统地将多回合越狱提示转换为单回合攻击。具体地说，我们提出了三种转换策略--Hyhenize、Numberize和Pythonize--每种策略都保留了顺序上下文，但又将其打包在单个查询中。我们在多轮人类越狱(MHJ)数据集上的实验表明，与原始的多轮对话相比，M2通常会增加或保持较高的攻击成功率(ASR)。值得注意的是，使用基于StrongREJECT的危害性评估，M2S在Mistral-7B上获得了高达95.9%的ASR，并且在GPT-40上的绝对改进程度比原来的多转弯提示高达17.5%。进一步的分析表明，某些对抗性策略，当整合到一个提示中时，会利用结构格式线索来逃避标准的政策检查。这些发现突显出，单回合攻击--尽管进行起来更简单、成本更低--可能与多回合攻击一样强大，甚至更多。我们的发现强调了重新评估和加强LLM安全策略的迫切需要，因为敌意查询可以被压缩到单个提示中，同时仍然保持足够的复杂性来绕过现有的安全措施。



## **34. The VLLM Safety Paradox: Dual Ease in Jailbreak Attack and Defense**

VLLM安全悖论：越狱攻击和防御的双重轻松 cs.CR

Logic smoothing and language polishing

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2411.08410v2) [paper-pdf](http://arxiv.org/pdf/2411.08410v2)

**Authors**: Yangyang Guo, Fangkai Jiao, Liqiang Nie, Mohan Kankanhalli

**Abstract**: The vulnerability of Vision Large Language Models (VLLMs) to jailbreak attacks appears as no surprise. However, recent defense mechanisms against these attacks have reached near-saturation performance on benchmark evaluations, often with minimal effort. This \emph{dual high performance} in both attack and defense raises a fundamental and perplexing paradox. To gain a deep understanding of this issue and thus further help strengthen the trustworthiness of VLLMs, this paper makes three key contributions: i) One tentative explanation for VLLMs being prone to jailbreak attacks--\textbf{inclusion of vision inputs}, as well as its in-depth analysis. ii) The recognition of a largely ignored problem in existing defense mechanisms--\textbf{over-prudence}. The problem causes these defense methods to exhibit unintended abstention, even in the presence of benign inputs, thereby undermining their reliability in faithfully defending against attacks. iii) A simple safety-aware method--\textbf{LLM-Pipeline}. Our method repurposes the more advanced guardrails of LLMs on the shelf, serving as an effective alternative detector prior to VLLM response. Last but not least, we find that the two representative evaluation methods for jailbreak often exhibit chance agreement. This limitation makes it potentially misleading when evaluating attack strategies or defense mechanisms. We believe the findings from this paper offer useful insights to rethink the foundational development of VLLM safety with respect to benchmark datasets, defense strategies, and evaluation methods.

摘要: Vision Large Language Models(VLLM)在越狱攻击中的脆弱性似乎并不令人意外。然而，最近针对这些攻击的防御机制在基准评估中的性能几乎达到饱和，通常只需很少的努力。这种在进攻和防守上的双重高性能提出了一个基本而令人困惑的悖论。为了更深入地理解这一问题，从而进一步增强VLLMS的可信性，本文做了三个重要贡献：1)对VLLM容易发生越狱攻击的原因进行了初步的解释--\extbf{Including of Vision Inputs}，并对其进行了深入分析。二)认识到现有防御机制中一个很大程度上被忽视的问题这个问题导致这些防御方法表现出意外的弃权，即使在存在良性输入的情况下也是如此，从而破坏了它们忠实防御攻击的可靠性。Iii)一种简单的安全感知方法--\extbf{LLM-Pipeline}。我们的方法重新调整了架子上更先进的LLM护栏的用途，作为VLLM响应之前的有效替代探测器。最后但并非最不重要的是，我们发现两种具有代表性的越狱评估方法往往表现出偶然性的一致性。这一限制使其在评估攻击策略或防御机制时具有潜在误导性。我们相信，这篇论文的发现为重新思考VLLM安全在基准数据集、防御策略和评估方法方面的基础性发展提供了有用的见解。



## **35. A generative approach to LLM harmfulness detection with special red flag tokens**

使用特殊危险信号令牌进行LLM危害检测的生成式方法 cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2502.16366v2) [paper-pdf](http://arxiv.org/pdf/2502.16366v2)

**Authors**: Sophie Xhonneux, David Dobre, Mehrnaz Mofakhami, Leo Schwinn, Gauthier Gidel

**Abstract**: Most safety training methods for large language models (LLMs) based on fine-tuning rely on dramatically changing the output distribution of the model when faced with a harmful request, shifting it from an unsafe answer to a refusal to respond. These methods inherently compromise model capabilities and might make auto-regressive models vulnerable to attacks that make likely an initial token of affirmative response. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to fine-tune the model to generate this token at any time harmful content is generated or about to be generated. This novel safety training method effectively augments LLMs into generative classifiers of harmfulness at all times during the conversation. This method offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer rather than just the input prompt and provides a stronger defence against sampling-based attacks. In addition, it simplifies the evaluation of the model's robustness and reduces correlated failures when combined with a classifier. We further show an increased robustness to long contexts, and supervised fine-tuning attacks.

摘要: 大多数基于微调的大型语言模型(LLM)安全培训方法依赖于在面临有害请求时显著改变模型的输出分布，将其从不安全的答案转变为拒绝响应。这些方法本质上会损害模型的能力，并可能使自回归模型容易受到攻击，从而可能成为肯定响应的初始标志。为了避免这种情况，我们建议使用一种称为红旗令牌(<rf>)的特殊令牌来扩展模型的词汇量，并建议微调模型以在生成或即将生成有害内容时生成该令牌。这种新颖的安全训练方法有效地将LLMS添加到对话过程中的任何时刻的危害生成性分类器中。这种方法有几个优点：它使模型能够明确地学习危害性的概念，而对生成的分布略有影响，从而保持了模型的实用性。它还评估每个生成的答案，而不仅仅是输入提示，并提供针对基于采样的攻击的更强大的防御。此外，它简化了模型稳健性的评估，并减少了与分类器结合时的相关故障。我们进一步显示了对长上下文的增强的健壮性，并监督了微调攻击。



## **36. Improving LLM Safety Alignment with Dual-Objective Optimization**

通过双目标优化改善LLM安全一致性 cs.CL

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03710v1) [paper-pdf](http://arxiv.org/pdf/2503.03710v1)

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment

摘要: 现有的大型语言模型(LLM)的训练时间安全对齐技术仍然容易受到越狱攻击。直接偏好优化(DPO)是一种被广泛应用的比对方法，由于其损失函数对于拒绝学习来说是次优的，因此在实验和理论环境中都显示出局限性。通过基于梯度的分析，我们识别了这些缺点，并提出了一种改进的安全对齐方法，将DPO目标分解为两个组成部分：(1)稳健的拒绝训练，即使产生部分不安全的生成也鼓励拒绝，以及(2)有针对性地忘记有害知识。这种方法显著提高了LLM对各种越狱攻击的稳健性，包括跨分发内和分发外场景的预填充、后缀和多轮攻击。此外，通过引入基于奖励的拒绝学习令牌级加权机制，我们引入了一种强调关键拒绝令牌的方法，进一步提高了对恶意攻击的鲁棒性。我们的研究还表明，越狱攻击的稳健性与训练过程中令牌分布的变化以及拒绝和有害令牌的内部表征相关，为未来LLM安全匹配的研究提供了有价值的方向。代码可在https://github.com/wicai24/DOOR-Alignment上获得



## **37. LLMs can be Dangerous Reasoners: Analyzing-based Jailbreak Attack on Large Language Models**

LLM可能是危险的推理者：基于分析的对大型语言模型的越狱攻击 cs.CR

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2407.16205v5) [paper-pdf](http://arxiv.org/pdf/2407.16205v5)

**Authors**: Shi Lin, Hongming Yang, Dingyang Lin, Rongchang Li, Xun Wang, Changting Lin, Wenpeng Xing, Meng Han

**Abstract**: The rapid development of Large Language Models (LLMs) has brought significant advancements across various tasks. However, despite these achievements, LLMs still exhibit inherent safety vulnerabilities, especially when confronted with jailbreak attacks. Existing jailbreak methods suffer from two main limitations: reliance on complicated prompt engineering and iterative optimization, which lead to low attack success rate (ASR) and attack efficiency (AE). In this work, we propose an efficient jailbreak attack method, Analyzing-based Jailbreak (ABJ), which leverages the advanced reasoning capability of LLMs to autonomously generate harmful content, revealing their underlying safety vulnerabilities during complex reasoning process. We conduct comprehensive experiments on ABJ across various open-source and closed-source LLMs. In particular, ABJ achieves high ASR (82.1% on GPT-4o-2024-11-20) with exceptional AE among all target LLMs, showcasing its remarkable attack effectiveness, transferability, and efficiency. Our findings underscore the urgent need to prioritize and improve the safety of LLMs to mitigate the risks of misuse.

摘要: 大型语言模型(LLM)的快速发展带来了跨各种任务的重大进步。然而，尽管取得了这些成就，LLMS仍然表现出固有的安全漏洞，特别是在面临越狱攻击时。现有的越狱方法存在两个主要缺陷：依赖复杂的快速工程和迭代优化，导致攻击成功率和攻击效率较低。在这项工作中，我们提出了一种高效的越狱攻击方法-基于分析的越狱(ABJ)，它利用LLMS的高级推理能力自主生成有害内容，在复杂的推理过程中揭示其潜在的安全漏洞。我们在各种开源和闭源的LLM上对ABJ进行了全面的实验。特别是，ABJ在所有目标LLM中获得了高ASR(在GPT-40-2024-11-20上为82.1%)，并具有出色的AE，显示了其卓越的攻击效能、可转移性和效率。我们的研究结果强调迫切需要优先考虑和改善低密度脂蛋白的安全性，以减少误用的风险。



## **38. Building Safe GenAI Applications: An End-to-End Overview of Red Teaming for Large Language Models**

构建安全的GenAI应用程序：大型语言模型红色团队的端到端概述 cs.CL

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.01742v2) [paper-pdf](http://arxiv.org/pdf/2503.01742v2)

**Authors**: Alberto Purpura, Sahil Wadhwa, Jesse Zymet, Akshay Gupta, Andy Luo, Melissa Kazemi Rad, Swapnil Shinde, Mohammad Shahed Sorower

**Abstract**: The rapid growth of Large Language Models (LLMs) presents significant privacy, security, and ethical concerns. While much research has proposed methods for defending LLM systems against misuse by malicious actors, researchers have recently complemented these efforts with an offensive approach that involves red teaming, i.e., proactively attacking LLMs with the purpose of identifying their vulnerabilities. This paper provides a concise and practical overview of the LLM red teaming literature, structured so as to describe a multi-component system end-to-end. To motivate red teaming we survey the initial safety needs of some high-profile LLMs, and then dive into the different components of a red teaming system as well as software packages for implementing them. We cover various attack methods, strategies for attack-success evaluation, metrics for assessing experiment outcomes, as well as a host of other considerations. Our survey will be useful for any reader who wants to rapidly obtain a grasp of the major red teaming concepts for their own use in practical applications.

摘要: 大型语言模型(LLM)的快速增长带来了重大的隐私、安全和伦理问题。虽然许多研究已经提出了保护LLM系统免受恶意行为者滥用的方法，但研究人员最近又用一种涉及红色团队的进攻性方法来补充这些努力，即主动攻击LLM，目的是识别它们的漏洞。本文提供了LLM红队文献的简明而实用的概述，其结构旨在描述端到端的多组件系统。为了激励红色团队，我们调查了一些备受瞩目的低成本管理系统的初始安全需求，然后深入研究红色团队系统的不同组件以及实施它们的软件包。我们涵盖了各种攻击方法、攻击成功评估策略、评估实验结果的指标以及许多其他考虑因素。我们的调查对于任何想要快速掌握主要的红色团队概念以便在实际应用中使用的读者都是有用的。



## **39. Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks**

针对越狱攻击的多模式大型语言模型对抗训练 cs.CV

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.04833v1) [paper-pdf](http://arxiv.org/pdf/2503.04833v1)

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems.

摘要: 多通道大语言模型在跨通道理解和生成任务方面取得了显著进展。然而，它们仍然容易受到越狱攻击，在越狱攻击中，精心设计的扰动绕过安全护栏，引发有害输出。在这篇文章中，我们提出了在MLLM训练阶段为防御越狱攻击而定制的第一个对抗性训练(AT)范例。将传统的AT扩展到这一领域会带来两个关键挑战：有效地调整大量参数和确保对跨多个通道的攻击的健壮性。为了应对这些挑战，我们引入了投影层对抗对手训练(ProEAT)，这是一个端到端的AT框架。ProEAT结合了基于投影仪的对抗性训练体系结构，通过将对抗性训练集中在轻量级投影器层而不是整个模型上，在保持计算可行性的同时有效地处理大规模参数；此外，我们设计了动态权重调整机制，基于任务需求优化损失函数的权重分配，从而简化了调整过程。为了提高防御性能，我们提出了一种跨视觉和文本模式的联合优化策略，确保对来自任何一种模式的越狱攻击具有强大的抵抗力。在三种主流MLLMS上对五种主要的越狱攻击方法进行了广泛的实验，证明了该方法的有效性。ProEAT实现了最先进的防御性能，在文本和图像模式中的表现比现有基线平均高出34%，而干净的准确性仅降低了1%。此外，对真实世界体现的智能系统的评估突出了我们框架的实用适用性，为开发更安全可靠的多式联运系统铺平了道路。



## **40. A 262 TOPS Hyperdimensional Photonic AI Accelerator powered by a Si3N4 microcomb laser**

由Si 3 N4微梳激光提供动力的262 TOPS超维Photonic AI加速器 physics.optics

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03263v1) [paper-pdf](http://arxiv.org/pdf/2503.03263v1)

**Authors**: Christos Pappas, Antonios Prapas, Theodoros Moschos, Manos Kirtas, Odysseas Asimopoulos, Apostolos Tsakyridis, Miltiadis Moralis-Pegios, Chris Vagionas, Nikolaos Passalis, Cagri Ozdilek, Timofey Shpakovsky, Alain Yuji Takabayashi, John D. Jost, Maxim Karpov, Anastasios Tefas, Nikos Pleros

**Abstract**: The ever-increasing volume of data has necessitated a new computing paradigm, embodied through Artificial Intelligence (AI) and Large Language Models (LLMs). Digital electronic AI computing systems, however, are gradually reaching their physical plateaus, stimulating extensive research towards next-generation AI accelerators. Photonic Neural Networks (PNNs), with their unique ability to capitalize on the interplay of multiple physical dimensions including time, wavelength, and space, have been brought forward with a credible promise for boosting computational power and energy efficiency in AI processors. In this article, we experimentally demonstrate a novel multidimensional arrayed waveguide grating router (AWGR)-based photonic AI accelerator that can execute tensor multiplications at a record-high total computational power of 262 TOPS, offering a ~24x improvement over the existing waveguide-based optical accelerators. It consists of a 16x16 AWGR that exploits the time-, wavelength- and space- division multiplexing (T-WSDM) for weight and input encoding together with an integrated Si3N4-based frequency comb for multi-wavelength generation. The photonic AI accelerator has been experimentally validated in both Fully-Connected (FC) and Convolutional NN (NNs) models, with the FC and CNN being trained for DDoS attack identification and MNIST classification, respectively. The experimental inference at 32 Gbaud achieved a Cohen's kappa score of 0.867 for DDoS detection and an accuracy of 92.14% for MNIST classification, respectively, closely matching the software performance.

摘要: 不断增长的数据量需要一种新的计算范式，通过人工智能(AI)和大型语言模型(LLM)来体现。然而，数字电子人工智能计算系统正逐渐达到其物理平台，刺激了对下一代人工智能加速器的广泛研究。光子神经网络(PNN)以其独特的能力利用包括时间、波长和空间在内的多个物理维度的相互作用，已经被提出了可信的承诺，以提高人工智能处理器的计算能力和能源效率。在实验中，我们展示了一种新型的基于多维阵列波导光栅路由器(AWGR)的光子AI加速器，其张量乘法运算的总计算能力达到了创纪录的262次，比现有的基于波导的光加速器提高了约24倍。它包括一个16x16 AWGR，它利用时间、波长和空分复用(T-WSDM)进行加权和输入编码，以及一个基于Si3N4的集成频率梳，用于多波长生成。光子人工智能加速器已经在全连接(FC)和卷积神经网络(NNS)模型上进行了实验验证，其中FC和CNN分别被训练用于DDoS攻击识别和MNIST分类。在32Gbaud下的实验推理获得了0.867的Cohen‘s kappa分数和92.14%的MNIST分类正确率，与软件性能非常接近。



## **41. AttackSeqBench: Benchmarking Large Language Models' Understanding of Sequential Patterns in Cyber Attacks**

AttackSeqBench：对大型语言模型对网络攻击中序列模式的理解进行基准测试 cs.CR

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03170v1) [paper-pdf](http://arxiv.org/pdf/2503.03170v1)

**Authors**: Javier Yong, Haokai Ma, Yunshan Ma, Anis Yusof, Zhenkai Liang, Ee-Chien Chang

**Abstract**: The observations documented in Cyber Threat Intelligence (CTI) reports play a critical role in describing adversarial behaviors, providing valuable insights for security practitioners to respond to evolving threats. Recent advancements of Large Language Models (LLMs) have demonstrated significant potential in various cybersecurity applications, including CTI report understanding and attack knowledge graph construction. While previous works have proposed benchmarks that focus on the CTI extraction ability of LLMs, the sequential characteristic of adversarial behaviors within CTI reports remains largely unexplored, which holds considerable significance in developing a comprehensive understanding of how adversaries operate. To address this gap, we introduce AttackSeqBench, a benchmark tailored to systematically evaluate LLMs' capability to understand and reason attack sequences in CTI reports. Our benchmark encompasses three distinct Question Answering (QA) tasks, each task focuses on the varying granularity in adversarial behavior. To alleviate the laborious effort of QA construction, we carefully design an automated dataset construction pipeline to create scalable and well-formulated QA datasets based on real-world CTI reports. To ensure the quality of our dataset, we adopt a hybrid approach of combining human evaluation and systematic evaluation metrics. We conduct extensive experiments and analysis with both fast-thinking and slow-thinking LLMs, while highlighting their strengths and limitations in analyzing the sequential patterns in cyber attacks. The overarching goal of this work is to provide a benchmark that advances LLM-driven CTI report understanding and fosters its application in real-world cybersecurity operations. Our dataset and code are available at https://github.com/Javiery3889/AttackSeqBench .

摘要: 网络威胁情报(CTI)报告中记录的观察结果在描述敌对行为方面发挥了关键作用，为安全从业者提供了宝贵的见解，以应对不断变化的威胁。大型语言模型的最新进展在各种网络安全应用中显示出巨大的潜力，包括CTI报告理解和攻击知识图的构建。虽然前人的研究主要集中在低层统计模型的CTI提取能力上，但CTI报告中敌方行为的时序特征在很大程度上还没有被探索，这对于全面理解敌方是如何运作的具有相当重要的意义。为了弥补这一差距，我们引入了AttackSeqBtch，这是一个专门为系统评估LLMS理解和推理CTI报告中的攻击序列的能力而定制的基准测试。我们的基准包括三个不同的问答(QA)任务，每个任务都专注于敌对行为中不同的粒度。为了减轻QA构建的繁重工作，我们精心设计了一个自动化的数据集构建管道，以真实世界的CTI报告为基础创建可扩展的、格式良好的QA数据集。为了确保我们的数据集的质量，我们采用了人工评估和系统评估度量相结合的混合方法。我们使用快速思维和缓慢思维的LLM进行了广泛的实验和分析，同时强调了它们在分析网络攻击中的序列模式方面的优势和局限性。这项工作的总体目标是提供一个基准，以促进LLM驱动的CTI报告的理解，并促进其在现实世界网络安全操作中的应用。我们的数据集和代码可在https://github.com/Javiery3889/AttackSeqBench上获得。



## **42. SoK: Knowledge is All You Need: Last Mile Delivery for Automated Provenance-based Intrusion Detection with LLMs**

SoK：知识就是您所需要的一切：利用LLM实现基于源的自动入侵检测的最后一英里交付 cs.CR

**SubmitDate**: 2025-03-05    [abs](http://arxiv.org/abs/2503.03108v1) [paper-pdf](http://arxiv.org/pdf/2503.03108v1)

**Authors**: Wenrui Cheng, Tiantian Zhu, Chunlin Xiong, Haofei Sun, Zijun Wang, Shunan Jing, Mingqi Lv, Yan Chen

**Abstract**: Recently, provenance-based intrusion detection systems (PIDSes) have been widely proposed for endpoint threat analysis. However, due to the lack of systematic integration and utilization of knowledge, existing PIDSes still require significant manual intervention for practical deployment, making full automation challenging. This paper presents a disruptive innovation by categorizing PIDSes according to the types of knowledge they utilize. In response to the prevalent issue of ``knowledge silos problem'' in existing research, we introduce a novel knowledge-driven provenance-based intrusion detection framework, powered by large language models (LLMs). We also present OmniSec, a best practice system built upon this framework. By integrating attack representation knowledge, threat intelligence knowledge, and benign behavior knowledge, OmniSec outperforms the state-of-the-art approaches on public benchmark datasets. OmniSec is available online at https://anonymous.4open.science/r/PIDS-with-LLM-613B.

摘要: 近年来，基于起源的入侵检测系统(PIDS)被广泛提出用于终端威胁分析。然而，由于缺乏系统地整合和利用知识，现有的PIDS仍然需要大量的人工干预才能进行实际部署，这使得完全自动化具有挑战性。本文提出了一种颠覆性创新，根据PIDS所使用的知识类型对其进行分类。针对现有研究中普遍存在的“知识孤岛问题”，提出了一种基于大语言模型的知识驱动的基于出处的入侵检测框架。我们还介绍了OmniSec，这是一个基于该框架构建的最佳实践系统。通过集成攻击表示知识、威胁情报知识和良性行为知识，OmniSec在公共基准数据集上的性能优于最先进的方法。OmniSec可在https://anonymous.4open.science/r/PIDS-with-LLM-613B.上在线购买。



## **43. LLM Misalignment via Adversarial RLHF Platforms**

对抗性LLHF平台的LLM失调 cs.LG

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.03039v1) [paper-pdf](http://arxiv.org/pdf/2503.03039v1)

**Authors**: Erfan Entezami, Ali Naseh

**Abstract**: Reinforcement learning has shown remarkable performance in aligning language models with human preferences, leading to the rise of attention towards developing RLHF platforms. These platforms enable users to fine-tune models without requiring any expertise in developing complex machine learning algorithms. While these platforms offer useful features such as reward modeling and RLHF fine-tuning, their security and reliability remain largely unexplored. Given the growing adoption of RLHF and open-source RLHF frameworks, we investigate the trustworthiness of these systems and their potential impact on behavior of LLMs. In this paper, we present an attack targeting publicly available RLHF tools. In our proposed attack, an adversarial RLHF platform corrupts the LLM alignment process by selectively manipulating data samples in the preference dataset. In this scenario, when a user's task aligns with the attacker's objective, the platform manipulates a subset of the preference dataset that contains samples related to the attacker's target. This manipulation results in a corrupted reward model, which ultimately leads to the misalignment of the language model. Our results demonstrate that such an attack can effectively steer LLMs toward undesirable behaviors within the targeted domains. Our work highlights the critical need to explore the vulnerabilities of RLHF platforms and their potential to cause misalignment in LLMs during the RLHF fine-tuning process.

摘要: 强化学习在将语言模型与人类偏好保持一致方面表现出了显著的性能，导致了人们对开发RLHF平台的关注。这些平台使用户能够微调模型，而不需要开发复杂的机器学习算法的任何专业知识。虽然这些平台提供了有用的功能，如奖励建模和RLHF微调，但它们的安全性和可靠性在很大程度上仍未得到探索。鉴于RLHF和开源RLHF框架越来越多地被采用，我们调查了这些系统的可信性及其对LLM行为的潜在影响。本文提出了一种针对公开可用的RLHF工具的攻击。在我们提出的攻击中，敌意的RLHF平台通过选择性地操纵偏好数据集中的数据样本来破坏LLM比对过程。在这种情况下，当用户的任务与攻击者的目标一致时，平台操作包含与攻击者目标相关的样本的首选项数据集的子集。这种操作会导致奖励模型被破坏，这最终会导致语言模型的不一致。我们的结果表明，这样的攻击可以有效地将LLM引向目标域内的不良行为。我们的工作突出了探索RLHF平台的脆弱性及其在RLHF微调过程中导致LLM未对准的可能性的迫切需要。



## **44. Towards Safe AI Clinicians: A Comprehensive Study on Large Language Model Jailbreaking in Healthcare**

迈向安全的人工智能临床医生：医疗保健领域大语言模型越狱的综合研究 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2501.18632v2) [paper-pdf](http://arxiv.org/pdf/2501.18632v2)

**Authors**: Hang Zhang, Qian Lou, Yanshan Wang

**Abstract**: Large language models (LLMs) are increasingly utilized in healthcare applications. However, their deployment in clinical practice raises significant safety concerns, including the potential spread of harmful information. This study systematically assesses the vulnerabilities of seven LLMs to three advanced black-box jailbreaking techniques within medical contexts. To quantify the effectiveness of these techniques, we propose an automated and domain-adapted agentic evaluation pipeline. Experiment results indicate that leading commercial and open-source LLMs are highly vulnerable to medical jailbreaking attacks. To bolster model safety and reliability, we further investigate the effectiveness of Continual Fine-Tuning (CFT) in defending against medical adversarial attacks. Our findings underscore the necessity for evolving attack methods evaluation, domain-specific safety alignment, and LLM safety-utility balancing. This research offers actionable insights for advancing the safety and reliability of AI clinicians, contributing to ethical and effective AI deployment in healthcare.

摘要: 大型语言模型(LLM)越来越多地用于医疗保健应用程序。然而，它们在临床实践中的部署引起了重大的安全担忧，包括有害信息的潜在传播。这项研究系统地评估了七种低密度脂蛋白对三种先进的黑盒越狱技术在医学背景下的脆弱性。为了量化这些技术的有效性，我们提出了一个自动化的和领域适应的代理评估管道。实验结果表明，领先的商业和开源LLM非常容易受到医疗越狱攻击。为了支持模型的安全性和可靠性，我们进一步研究了连续微调(CFT)在防御医疗对手攻击方面的有效性。我们的发现强调了对不断发展的攻击方法进行评估、特定领域的安全对齐和LLM安全效用平衡的必要性。这项研究为提高人工智能临床医生的安全性和可靠性提供了可操作的见解，有助于在医疗保健领域进行合乎道德和有效的人工智能部署。



## **45. LLM-Safety Evaluations Lack Robustness**

LLM-安全性评估缺乏稳健性 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.02574v1) [paper-pdf](http://arxiv.org/pdf/2503.02574v1)

**Authors**: Tim Beyer, Sophie Xhonneux, Simon Geisler, Gauthier Gidel, Leo Schwinn, Stephan Günnemann

**Abstract**: In this paper, we argue that current safety alignment research efforts for large language models are hindered by many intertwined sources of noise, such as small datasets, methodological inconsistencies, and unreliable evaluation setups. This can, at times, make it impossible to evaluate and compare attacks and defenses fairly, thereby slowing progress. We systematically analyze the LLM safety evaluation pipeline, covering dataset curation, optimization strategies for automated red-teaming, response generation, and response evaluation using LLM judges. At each stage, we identify key issues and highlight their practical impact. We also propose a set of guidelines for reducing noise and bias in evaluations of future attack and defense papers. Lastly, we offer an opposing perspective, highlighting practical reasons for existing limitations. We believe that addressing the outlined problems in future research will improve the field's ability to generate easily comparable results and make measurable progress.

摘要: 在本文中，我们认为目前针对大型语言模型的安全对齐研究工作受到许多相互交织的噪声源的阻碍，如小数据集、方法不一致和不可靠的评估设置。这有时会使人们无法公平地评估和比较攻击和防御，从而减缓进展。我们系统地分析了LLM安全评估管道，包括数据集管理、自动红团队的优化策略、响应生成和使用LLM评判器的响应评估。在每个阶段，我们确定关键问题并强调其实际影响。我们还提出了一套指导方针，以减少未来攻击和防御论文评估中的噪音和偏见。最后，我们提供了一个相反的观点，强调了现有限制的实际原因。我们认为，在今后的研究中解决概述的问题将提高该领域产生容易比较的结果和取得可衡量的进展的能力。



## **46. TPIA: Towards Target-specific Prompt Injection Attack against Code-oriented Large Language Models**

TPIA：针对面向代码的大型语言模型的特定目标提示注入攻击 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2407.09164v5) [paper-pdf](http://arxiv.org/pdf/2407.09164v5)

**Authors**: Yuchen Yang, Hongwei Yao, Bingrun Yang, Yiling He, Yiming Li, Tianwei Zhang, Zhan Qin

**Abstract**: Recently, code-oriented large language models (Code LLMs) have been widely and successfully exploited to simplify and facilitate programming. Unfortunately, a few pioneering works revealed that these Code LLMs are vulnerable to backdoor and adversarial attacks. The former poisons the training data or model parameters, hijacking the LLMs to generate malicious code snippets when encountering the trigger. The latter crafts malicious adversarial input codes to reduce the quality of the generated codes. In this paper, we reveal that both attacks have some inherent limitations: backdoor attacks rely on the adversary's capability of controlling the model training process, which may not be practical; adversarial attacks struggle with fulfilling specific malicious purposes. To alleviate these problems, this paper presents a novel attack paradigm against Code LLMs, namely target-specific prompt injection attack (TPIA). TPIA generates non-functional perturbations containing the information of malicious instructions and inserts them into the victim's code context by spreading them into potentially used dependencies (e.g., packages or RAG's knowledge base). It induces the Code LLMs to generate attacker-specified malicious code snippets at the target location. In general, we compress the attacker-specified malicious objective into the perturbation by adversarial optimization based on greedy token search. We collect 13 representative malicious objectives to design 31 threat cases for three popular programming languages. We show that our TPIA can successfully attack three representative open-source Code LLMs (with an attack success rate of up to 97.9%) and two mainstream commercial Code LLM-integrated applications (with an attack success rate of over 90%) in all threat cases, using only a 12-token non-functional perturbation.

摘要: 最近，面向代码的大型语言模型(Code LLM)已经被广泛并成功地利用来简化和促进编程。不幸的是，一些开创性的工作表明，这些代码LLM容易受到后门和对手的攻击。前者毒化训练数据或模型参数，在遇到触发器时劫持LLMS生成恶意代码片段。后者制作恶意敌意输入代码以降低生成代码的质量。在本文中，我们揭示了这两种攻击都有一些固有的局限性：后门攻击依赖于对手控制模型训练过程的能力，这可能是不实用的；对抗性攻击难以实现特定的恶意目的。针对这些问题，提出了一种新的针对代码LLMS的攻击范式，即目标特定的即时注入攻击(TPIA)。TPIA生成包含恶意指令信息的非功能性扰动，并通过将它们传播到可能使用的依赖项(例如，包或RAG的知识库)，将它们插入到受害者的代码上下文中。它诱导代码LLM在目标位置生成攻击者指定的恶意代码片段。一般而言，我们通过基于贪婪令牌搜索的对抗性优化将攻击者指定的恶意目标压缩为扰动。我们收集了13个具有代表性的恶意目标，为三种流行的编程语言设计了31个威胁案例。实验表明，在所有威胁情况下，仅使用12个令牌的非功能扰动，我们的TPIA就可以成功攻击三个典型的开源代码LLM(攻击成功率高达97.9%)和两个主流商业代码LLM集成应用(攻击成功率超过90%)。



## **47. Adaptive Attacks Break Defenses Against Indirect Prompt Injection Attacks on LLM Agents**

自适应攻击突破了对LLM代理间接即时注入攻击的防御 cs.CR

17 pages, 5 figures, 6 tables (NAACL 2025 Findings)

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2503.00061v2) [paper-pdf](http://arxiv.org/pdf/2503.00061v2)

**Authors**: Qiusi Zhan, Richard Fang, Henil Shalin Panchal, Daniel Kang

**Abstract**: Large Language Model (LLM) agents exhibit remarkable performance across diverse applications by using external tools to interact with environments. However, integrating external tools introduces security risks, such as indirect prompt injection (IPI) attacks. Despite defenses designed for IPI attacks, their robustness remains questionable due to insufficient testing against adaptive attacks. In this paper, we evaluate eight different defenses and bypass all of them using adaptive attacks, consistently achieving an attack success rate of over 50%. This reveals critical vulnerabilities in current defenses. Our research underscores the need for adaptive attack evaluation when designing defenses to ensure robustness and reliability. The code is available at https://github.com/uiuc-kang-lab/AdaptiveAttackAgent.

摘要: 大型语言模型（LLM）代理通过使用外部工具与环境交互，在不同的应用程序中表现出出色的性能。然而，集成外部工具会带来安全风险，例如间接提示注入（IPI）攻击。尽管针对IPI攻击设计了防御措施，但由于针对自适应攻击的测试不足，其稳健性仍然值得怀疑。在本文中，我们评估了八种不同的防御措施，并使用自适应攻击绕过了所有防御措施，始终实现了超过50%的攻击成功率。这揭示了当前防御系统中的关键漏洞。我们的研究强调了在设计防御以确保稳健性和可靠性时需要进行自适应攻击评估。该代码可在https://github.com/uiuc-kang-lab/AdaptiveAttackAgent上获取。



## **48. Confidential Prompting: Protecting User Prompts from Cloud LLM Providers**

机密预算：保护用户预算免受云LLM提供商的预算 cs.CR

**SubmitDate**: 2025-03-04    [abs](http://arxiv.org/abs/2409.19134v3) [paper-pdf](http://arxiv.org/pdf/2409.19134v3)

**Authors**: In Gim, Caihua Li, Lin Zhong

**Abstract**: Our work tackles the challenge of securing user inputs in cloud-hosted large language model (LLM) serving while ensuring model confidentiality, output invariance, and compute efficiency. We introduce Secure Partitioned Decoding (SPD), which uses confidential computing to confine user prompts to a trusted execution environment (TEE), namely a confidential virtual machine (CVM), while allowing service providers to generate tokens efficiently. We also introduce a novel cryptographic method, Prompt Obfuscation (PO), to ensure robustness against reconstruction attacks on SPD. We demonstrate our approach preserves both prompt confidentiality and LLM serving efficiency. Our solution enables privacy-preserving cloud LLM serving that handles sensitive prompts, such as clinical records, financial data, and personal information.

摘要: 我们的工作解决了在云托管大型语言模型（LLM）服务中保护用户输入的挑战，同时确保模型机密性、输出不变性和计算效率。我们引入了安全分区解码（SPD），它使用机密计算将用户提示限制在可信执行环境（TEK），即机密虚拟机（CGM），同时允许服务提供商高效地生成令牌。我们还引入了一种新型加密方法--提示混淆（PO），以确保抵御SPD重建攻击的鲁棒性。我们证明我们的方法既保留了即时的保密性，又保留了LLM服务效率。我们的解决方案支持保护隐私的云LLM服务，可以处理敏感提示，例如临床记录、财务数据和个人信息。



## **49. De-identification is not enough: a comparison between de-identified and synthetic clinical notes**

去识别还不够：去识别和合成临床笔记之间的比较 cs.CL

https://www.nature.com/articles/s41598-024-81170-y

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2402.00179v2) [paper-pdf](http://arxiv.org/pdf/2402.00179v2)

**Authors**: Atiquer Rahman Sarkar, Yao-Shun Chuang, Noman Mohammed, Xiaoqian Jiang

**Abstract**: For sharing privacy-sensitive data, de-identification is commonly regarded as adequate for safeguarding privacy. Synthetic data is also being considered as a privacy-preserving alternative. Recent successes with numerical and tabular data generative models and the breakthroughs in large generative language models raise the question of whether synthetically generated clinical notes could be a viable alternative to real notes for research purposes. In this work, we demonstrated that (i) de-identification of real clinical notes does not protect records against a membership inference attack, (ii) proposed a novel approach to generate synthetic clinical notes using the current state-of-the-art large language models, (iii) evaluated the performance of the synthetically generated notes in a clinical domain task, and (iv) proposed a way to mount a membership inference attack where the target model is trained with synthetic data. We observed that when synthetically generated notes closely match the performance of real data, they also exhibit similar privacy concerns to the real data. Whether other approaches to synthetically generated clinical notes could offer better trade-offs and become a better alternative to sensitive real notes warrants further investigation.

摘要: 对于共享隐私敏感数据，消除身份识别通常被认为足以保护隐私。合成数据也被认为是一种保护隐私的选择。最近数字和表格数据生成模型的成功以及大型生成语言模型的突破提出了一个问题，即合成生成的临床笔记是否可以作为用于研究目的的真实笔记的可行替代方案。在这项工作中，我们证明了(I)真实临床笔记的去识别并不能保护记录免受成员关系推理攻击，(Ii)提出了一种使用当前最先进的大型语言模型生成合成临床笔记的新方法，(Iii)评估了合成生成的笔记在临床领域任务中的性能，以及(Iv)提出了一种利用合成数据训练目标模型的成员关系推理攻击的方法。我们观察到，当合成的笔记与真实数据的性能非常匹配时，它们也表现出与真实数据相似的隐私问题。合成临床笔记的其他方法是否可以提供更好的权衡，并成为敏感的真实笔记的更好替代方案，值得进一步研究。



## **50. Jailbreaking Safeguarded Text-to-Image Models via Large Language Models**

通过大型语言模型越狱受保护的文本到图像模型 cs.CR

**SubmitDate**: 2025-03-03    [abs](http://arxiv.org/abs/2503.01839v1) [paper-pdf](http://arxiv.org/pdf/2503.01839v1)

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose PromptTune, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks.

摘要: 文本到图像模型可能会生成有害内容，例如色情图像，特别是在提交不安全提示时。为了解决这个问题，通常在文本到图像模型之上添加安全过滤器，或者对模型本身进行调整以减少有害输出。然而，当攻击者战略性地设计对抗提示来绕过这些安全护栏时，这些防御仍然容易受到攻击。在这项工作中，我们提出了ObjetTune，这是一种使用微调的大型语言模型来越狱具有安全护栏的文本到图像模型的方法。与其他需要对目标模型重复查询的基于查询的越狱攻击不同，我们的攻击在微调AttackLLM后有效地生成对抗提示。我们在三个不安全提示数据集和五个安全护栏上评估我们的方法。我们的结果表明，我们的方法有效地绕过了安全护栏，优于现有的无框攻击，并且还促进了其他基于查询的攻击。



