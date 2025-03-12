# Latest Adversarial Attack Papers
**update at 2025-03-12 19:26:14**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Birds look like cars: Adversarial analysis of intrinsically interpretable deep learning**

鸟看起来像汽车：本质上可解释的深度学习的对抗分析 cs.LG

Preprint

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08636v1) [paper-pdf](http://arxiv.org/pdf/2503.08636v1)

**Authors**: Hubert Baniecki, Przemyslaw Biecek

**Abstract**: A common belief is that intrinsically interpretable deep learning models ensure a correct, intuitive understanding of their behavior and offer greater robustness against accidental errors or intentional manipulation. However, these beliefs have not been comprehensively verified, and growing evidence casts doubt on them. In this paper, we highlight the risks related to overreliance and susceptibility to adversarial manipulation of these so-called "intrinsically (aka inherently) interpretable" models by design. We introduce two strategies for adversarial analysis with prototype manipulation and backdoor attacks against prototype-based networks, and discuss how concept bottleneck models defend against these attacks. Fooling the model's reasoning by exploiting its use of latent prototypes manifests the inherent uninterpretability of deep neural networks, leading to a false sense of security reinforced by a visual confirmation bias. The reported limitations of prototype-based networks put their trustworthiness and applicability into question, motivating further work on the robustness and alignment of (deep) interpretable models.

摘要: 人们普遍认为，本质上可解释的深度学习模型确保了对其行为的正确、直观的理解，并提供了针对意外错误或故意操作的更强的健壮性。然而，这些信念还没有得到全面证实，越来越多的证据让人对它们产生了怀疑。在这篇文章中，我们强调了与过度依赖和易受敌意操纵有关的风险，这些模型被设计为“本质上(也就是内在)可解释的”模型。我们介绍了利用原型操纵和后门攻击对基于原型的网络进行敌意分析的两种策略，并讨论了概念瓶颈模型如何防御这些攻击。通过利用潜在原型来愚弄模型的推理，表明了深层神经网络固有的不可解释性，导致了一种错误的安全感，并被视觉确认偏差所强化。已报道的基于原型的网络的局限性使它们的可信性和适用性受到质疑，促使进一步研究(深度)可解释模型的健壮性和一致性。



## **2. Beyond Optimal Fault Tolerance**

超越最佳故障容忍度 cs.DC

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2501.06044v4) [paper-pdf](http://arxiv.org/pdf/2501.06044v4)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal "recoverable fault-tolerance" achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic "recovery procedure" that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.

摘要: 任何协议可实现的最佳容错已在广泛的设置中得到了表征。例如，对于在部分同步设置中操作的状态机复制(SMR)协议，当且仅当$\Alpha+2\Beta\leq 1$时，可以同时保证针对$\Alpha$受限的对手(即，控制少于$\Alpha$部分参与者的对手)的一致性和针对$\beta$受限的攻击者的活性。本文刻画了当标准一致性要求被放宽以允许有限数量的一致性违规时，SMR协议在多大程度上可能获得比最优更好的容错保证。我们证明了如果没有额外的时间假设，绑定回滚是不可能的，并研究了当攻击时间附近的消息延迟由参数$\Delta^*$(该参数可以任意大于在部分同步模型中限制GST后消息延迟的参数$\Delta$)限定时，容忍一致性违规并从一致性违规中恢复的协议。这里，协议的容错性可以是$r$的非常数函数，我们证明了对于每个$r$，协议的最优“可恢复容错性”的上下界是匹配的。例如，对于在部分同步设置中保证对1/3有界攻击者的活跃性的协议，5/9有界的攻击者总是可以导致一次一致性违反而不是两次，而2/3有界的攻击者总是可以引起两次一致性违反而不是三次一致性违反。我们的积极结果是通过通用的“恢复程序”实现的，该程序可以嫁接到任何负责任的SMR协议上，并在违规后恢复一致性，同时仅回滚在前$2\Delta^*$时间步长中完成的事务。



## **3. Low-Cost Privacy-Preserving Decentralized Learning**

低成本保护隐私的分散学习 cs.LG

24 pages, accepted at Pets 2025

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2403.11795v3) [paper-pdf](http://arxiv.org/pdf/2403.11795v3)

**Authors**: Sayan Biswas, Davide Frey, Romaric Gaudel, Anne-Marie Kermarrec, Dimitri Lerévérend, Rafael Pires, Rishi Sharma, François Taïani

**Abstract**: Decentralized learning (DL) is an emerging paradigm of collaborative machine learning that enables nodes in a network to train models collectively without sharing their raw data or relying on a central server. This paper introduces Zip-DL, a privacy-aware DL algorithm that leverages correlated noise to achieve robust privacy against local adversaries while ensuring efficient convergence at low communication costs. By progressively neutralizing the noise added during distributed averaging, Zip-DL combines strong privacy guarantees with high model accuracy. Its design requires only one communication round per gradient descent iteration, significantly reducing communication overhead compared to competitors. We establish theoretical bounds on both convergence speed and privacy guarantees. Moreover, extensive experiments demonstrating Zip-DL's practical applicability make it outperform state-of-the-art methods in the accuracy vs. vulnerability trade-off. Specifically, Zip-DL (i) reduces membership-inference attack success rates by up to 35% compared to baseline DL, (ii) decreases attack efficacy by up to 13% compared to competitors offering similar utility, and (iii) achieves up to 59% higher accuracy to completely nullify a basic attack scenario, compared to a state-of-the-art privacy-preserving approach under the same threat model. These results position Zip-DL as a practical and efficient solution for privacy-preserving decentralized learning in real-world applications.

摘要: 分散学习是一种新兴的协作机器学习范式，它使网络中的节点能够在不共享原始数据或依赖中央服务器的情况下集体训练模型。本文介绍了ZIP-DL算法，这是一种隐私感知的DL算法，它利用相关噪声来实现对本地攻击者的稳健隐私保护，同时确保以较低的通信成本实现高效的收敛。通过逐步中和分布式平均过程中添加的噪声，Zip-DL将强大的隐私保证与高模型精度相结合。它的设计在每次梯度下降迭代中只需要一轮通信，与竞争对手相比显著减少了通信开销。我们建立了收敛速度和隐私保障的理论界限。此外，大量实验证明了Zip-DL的实用适用性，使其在准确性与脆弱性之间的权衡上优于最先进的方法。具体地说，与基准DL相比，Zip-DL(I)将成员资格推理攻击成功率降低了高达35%，(Ii)与提供类似实用工具的竞争对手相比，攻击效率降低了高达13%，(Iii)与相同威胁模型下最先进的隐私保护方法相比，可实现高达59%的准确率，以完全消除基本攻击场景。这些结果将Zip-DL定位为现实世界应用程序中保护隐私的分散学习的实用而有效的解决方案。



## **4. Adv-CPG: A Customized Portrait Generation Framework with Facial Adversarial Attacks**

Adv-CPG：具有面部对抗攻击的定制肖像生成框架 cs.CV

Accepted by CVPR-25

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08269v1) [paper-pdf](http://arxiv.org/pdf/2503.08269v1)

**Authors**: Junying Wang, Hongyuan Zhang, Yuan Yuan

**Abstract**: Recent Customized Portrait Generation (CPG) methods, taking a facial image and a textual prompt as inputs, have attracted substantial attention. Although these methods generate high-fidelity portraits, they fail to prevent the generated portraits from being tracked and misused by malicious face recognition systems. To address this, this paper proposes a Customized Portrait Generation framework with facial Adversarial attacks (Adv-CPG). Specifically, to achieve facial privacy protection, we devise a lightweight local ID encryptor and an encryption enhancer. They implement progressive double-layer encryption protection by directly injecting the target identity and adding additional identity guidance, respectively. Furthermore, to accomplish fine-grained and personalized portrait generation, we develop a multi-modal image customizer capable of generating controlled fine-grained facial features. To the best of our knowledge, Adv-CPG is the first study that introduces facial adversarial attacks into CPG. Extensive experiments demonstrate the superiority of Adv-CPG, e.g., the average attack success rate of the proposed Adv-CPG is 28.1% and 2.86% higher compared to the SOTA noise-based attack methods and unconstrained attack methods, respectively.

摘要: 最近的定制肖像生成(CPG)方法以面部图像和文本提示为输入，引起了广泛的关注。虽然这些方法会生成高保真的肖像，但它们无法防止生成的肖像被恶意的人脸识别系统跟踪和滥用。针对这一问题，提出了一种基于人脸对抗攻击的个性化肖像生成框架(ADV-CPG)。具体地说，为了实现面部隐私保护，我们设计了一个轻量级的本地ID加密器和一个加密增强器。它们分别通过直接注入目标身份和添加额外的身份指导来实施渐进式双层加密保护。此外，为了实现细粒度和个性化的肖像生成，我们开发了一个能够生成受控细粒度人脸特征的多模式图像定制器。据我们所知，ADV-CPG是第一个将面部对抗性攻击引入CPG的研究。实验结果表明，ADV-CPG的平均攻击成功率比基于噪声的SOTA攻击方法和无约束攻击方法分别高出28.1%和2.86%。



## **5. A Grey-box Text Attack Framework using Explainable AI**

使用可解释人工智能的灰箱文本攻击框架 cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08226v1) [paper-pdf](http://arxiv.org/pdf/2503.08226v1)

**Authors**: Esther Chiramal, Kelvin Soh Boon Kai

**Abstract**: Explainable AI is a strong strategy implemented to understand complex black-box model predictions in a human interpretable language. It provides the evidence required to execute the use of trustworthy and reliable AI systems. On the other hand, however, it also opens the door to locating possible vulnerabilities in an AI model. Traditional adversarial text attack uses word substitution, data augmentation techniques and gradient-based attacks on powerful pre-trained Bidirectional Encoder Representations from Transformers (BERT) variants to generate adversarial sentences. These attacks are generally whitebox in nature and not practical as they can be easily detected by humans E.g. Changing the word from "Poor" to "Rich". We proposed a simple yet effective Grey-box cum Black-box approach that does not require the knowledge of the model while using a set of surrogate Transformer/BERT models to perform the attack using Explainable AI techniques. As Transformers are the current state-of-the-art models for almost all Natural Language Processing (NLP) tasks, an attack generated from BERT1 is transferable to BERT2. This transferability is made possible due to the attention mechanism in the transformer that allows the model to capture long-range dependencies in a sequence. Using the power of BERT generalisation via attention, we attempt to exploit how transformers learn by attacking a few surrogate transformer variants which are all based on a different architecture. We demonstrate that this approach is highly effective to generate semantically good sentences by changing as little as one word that is not detectable by humans while still fooling other BERT models.

摘要: 可解释人工智能是一种强大的策略，用于理解人类可解释语言中的复杂黑盒模型预测。它提供了使用值得信赖和可靠的人工智能系统所需的证据。然而，另一方面，它也为定位人工智能模型中可能的漏洞打开了大门。传统的对抗性文本攻击使用单词替换、数据增强技术和基于梯度的攻击，对来自Transformers(BERT)变体的强大的预训练双向编码器表示进行攻击，以生成对抗性句子。这些攻击通常是白盒性质的，并不实用，因为它们很容易被人类发现，例如，将单词从“穷”改为“富”。我们提出了一种简单而有效的灰盒和黑盒方法，该方法不需要模型知识，同时使用一组代理Transformer/BERT模型来执行攻击，使用可解释的人工智能技术。由于Transformers是几乎所有自然语言处理(NLP)任务的当前最先进的模型，从BERT1生成的攻击可以转移到BERT2。这种可转移性之所以成为可能，是因为转换器中的注意力机制允许模型捕获序列中的远程依赖关系。使用通过注意的BERT泛化的力量，我们试图通过攻击几个都基于不同体系结构的代理变压器变体来探索变压器是如何学习的。我们证明了这种方法是非常有效的，可以通过更改一个人类无法检测的单词来生成语义良好的句子，同时仍然可以愚弄其他BERT模型。



## **6. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

紧急系统的守护者：用紧急系统防止多次枪击越狱 cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.16750v3) [paper-pdf](http://arxiv.org/pdf/2502.16750v3)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.

摘要: 使用大型语言模型的自主人工智能代理可以在社会各个领域创造不可否认的价值，但他们面临来自对手的安全威胁，需要立即采取保护性解决方案，因为信任和安全问题会出现。考虑到多发越狱和欺骗性对准是一些主要的高级攻击，在监督训练期间使用的静态护栏无法减轻这些攻击，指出了现实世界健壮性的关键研究重点。动态多智能体系统中静态护栏的组合不能抵抗这些攻击。我们打算通过制定新的评估框架，确定和应对安全行动部署所面临的威胁，从而加强基于LLM的特工的安全。我们的工作使用了三种检测方法来通过反向图灵测试来检测流氓代理，并通过多代理模拟来分析欺骗性比对，并开发了一个反越狱系统，通过使用Gemini 1.5Pro和Llama-3.3-70B、使用工具中介的对抗场景来测试DeepSeek R1模型来开发反越狱系统。Gemini 1.5 PRO具有很强的检测能力，如94%的准确率，但在长时间攻击下，随着提示长度的增加，攻击成功率(ASR)增加，多样性度量在预测多个复杂系统故障时变得无效，系统存在持续漏洞。这些发现证明了采用基于主动监控的灵活安全系统的必要性，该系统可以由代理自己执行，并由系统管理员进行适应性干预，因为当前的模型可能会产生漏洞，从而导致系统不可靠和易受攻击。因此，在我们的工作中，我们试图解决这些情况，并提出一个全面的框架来对抗安全问题。



## **7. Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation**

对话注入攻击：通过上下文操纵越狱LLM cs.CL

17 pages, 10 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08195v1) [paper-pdf](http://arxiv.org/pdf/2503.08195v1)

**Authors**: Wenlong Meng, Fan Zhang, Wendao Yao, Zhenyuan Guo, Yuwei Li, Chengkun Wei, Wenzhi Chen

**Abstract**: Large language models (LLMs) have demonstrated significant utility in a wide range of applications; however, their deployment is plagued by security vulnerabilities, notably jailbreak attacks. These attacks manipulate LLMs to generate harmful or unethical content by crafting adversarial prompts. While much of the current research on jailbreak attacks has focused on single-turn interactions, it has largely overlooked the impact of historical dialogues on model behavior. In this paper, we introduce a novel jailbreak paradigm, Dialogue Injection Attack (DIA), which leverages the dialogue history to enhance the success rates of such attacks. DIA operates in a black-box setting, requiring only access to the chat API or knowledge of the LLM's chat template. We propose two methods for constructing adversarial historical dialogues: one adapts gray-box prefilling attacks, and the other exploits deferred responses. Our experiments show that DIA achieves state-of-the-art attack success rates on recent LLMs, including Llama-3.1 and GPT-4o. Additionally, we demonstrate that DIA can bypass 5 different defense mechanisms, highlighting its robustness and effectiveness.

摘要: 大型语言模型(LLM)在广泛的应用程序中显示出了重要的实用价值；然而，它们的部署受到安全漏洞的困扰，特别是越狱攻击。这些攻击通过精心编制敌意提示来操纵LLM生成有害或不道德的内容。虽然目前对越狱攻击的大部分研究都集中在单回合互动上，但在很大程度上忽视了历史对话对模型行为的影响。在本文中，我们介绍了一种新的越狱范例，对话注入攻击(DIA)，它利用对话历史来提高此类攻击的成功率。DIA在黑盒设置中运行，只需要访问聊天API或了解LLM的聊天模板。我们提出了两种构造对抗性历史对话的方法：一种是采用灰盒预填充攻击，另一种是利用延迟响应。我们的实验表明，DIA在包括Llama-3.1和GPT-40在内的最近的LLM上达到了最先进的攻击成功率。此外，我们还证明了DIA可以绕过5种不同的防御机制，突出了其健壮性和有效性。



## **8. MAGIC: Mastering Physical Adversarial Generation in Context through Collaborative LLM Agents**

MAGIC：通过协作LLM代理掌握上下文中的物理对抗生成 cs.CV

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.08014v2) [paper-pdf](http://arxiv.org/pdf/2412.08014v2)

**Authors**: Yun Xing, Nhat Chung, Jie Zhang, Yue Cao, Ivor Tsang, Yang Liu, Lei Ma, Qing Guo

**Abstract**: Physical adversarial attacks in driving scenarios can expose critical vulnerabilities in visual perception models. However, developing such attacks remains challenging due to diverse real-world environments and the requirement for maintaining visual naturality. Building upon this challenge, we reformulate physical adversarial attacks as a one-shot patch generation problem. Our approach generates adversarial patches through a deep generative model that considers the specific scene context, enabling direct physical deployment in matching environments. The primary challenge lies in simultaneously achieving two objectives: generating adversarial patches that effectively mislead object detection systems while determining contextually appropriate deployment within the scene. We propose MAGIC (Mastering Physical Adversarial Generation In Context), a novel framework powered by multi-modal LLM agents to address these challenges. MAGIC automatically understands scene context and generates adversarial patch through the synergistic interaction of language and vision capabilities. In particular, MAGIC orchestrates three specialized LLM agents: The adv-patch generation agent (GAgent) masters the creation of deceptive patches through strategic prompt engineering for text-to-image models. The adv-patch deployment agent (DAgent) ensures contextual coherence by determining optimal deployment strategies based on scene understanding. The self-examination agent (EAgent) completes this trilogy by providing critical oversight and iterative refinement of both processes. We validate our method on both digital and physical levels, i.e., nuImage and manually captured real-world scenes, where both statistical and visual results prove that our MAGIC is powerful and effective for attacking widely applied object detection systems, i.e., YOLO and DETR series.

摘要: 驾驶场景中的物理对抗性攻击可以暴露视觉感知模型中的关键漏洞。然而，由于现实世界环境的多样性和保持视觉自然性的要求，开发此类攻击仍然具有挑战性。在这一挑战的基础上，我们将物理对抗性攻击重新定义为一次性补丁生成问题。我们的方法通过深度生成模型生成对抗性补丁，该模型考虑了特定的场景上下文，支持在匹配环境中直接物理部署。主要挑战在于同时实现两个目标：生成有效误导目标检测系统的对抗性补丁，同时确定场景中的上下文适当部署。我们提出了MAGIC(掌握上下文中的物理对手生成)，这是一个由多模式LLM代理支持的新框架来应对这些挑战。Magic自动理解场景背景，并通过语言和视觉能力的协同交互生成对抗性补丁。特别是，Magic协调了三个专门的LLM代理：adv-patch生成代理(Gagent)通过针对文本到图像模型的战略提示工程掌握了欺骗性补丁的创建。Adv-patch部署代理通过基于场景理解来确定最佳部署策略，从而确保上下文的一致性。自我检查代理(EAgent)通过提供对两个过程的关键监督和迭代细化来完成这三部曲。我们在数字和物理两个层面上验证了我们的方法，即NuImage和手动捕获的真实场景，其中统计和视觉结果都证明了我们的魔力对于攻击广泛应用的目标检测系统(如YOLO和DETR系列)是强大和有效的。



## **9. MIGA: Mutual Information-Guided Attack on Denoising Models for Semantic Manipulation**

MIGA：对语义操纵去噪模型的相互信息引导攻击 cs.CV

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.06966v2) [paper-pdf](http://arxiv.org/pdf/2503.06966v2)

**Authors**: Guanghao Li, Mingzhi Chen, Hao Yu, Shuting Dong, Wenhao Jiang, Ming Tang, Chun Yuan

**Abstract**: Deep learning-based denoising models have been widely employed in vision tasks, functioning as filters to eliminate noise while retaining crucial semantic information. Additionally, they play a vital role in defending against adversarial perturbations that threaten downstream tasks. However, these models can be intrinsically susceptible to adversarial attacks due to their dependence on specific noise assumptions. Existing attacks on denoising models mainly aim at deteriorating visual clarity while neglecting semantic manipulation, rendering them either easily detectable or limited in effectiveness. In this paper, we propose Mutual Information-Guided Attack (MIGA), the first method designed to directly attack deep denoising models by strategically disrupting their ability to preserve semantic content via adversarial perturbations. By minimizing the mutual information between the original and denoised images, a measure of semantic similarity. MIGA forces the denoiser to produce perceptually clean yet semantically altered outputs. While these images appear visually plausible, they encode systematically distorted semantics, revealing a fundamental vulnerability in denoising models. These distortions persist in denoised outputs and can be quantitatively assessed through downstream task performance. We propose new evaluation metrics and systematically assess MIGA on four denoising models across five datasets, demonstrating its consistent effectiveness in disrupting semantic fidelity. Our findings suggest that denoising models are not always robust and can introduce security risks in real-world applications.

摘要: 基于深度学习的去噪模型被广泛应用于视觉任务中，在保留关键语义信息的同时起到滤除噪声的作用。此外，它们在防御威胁下游任务的对抗性干扰方面发挥着至关重要的作用。然而，由于这些模型依赖于特定的噪声假设，因此可能本质上容易受到对抗性攻击。现有的对去噪模型的攻击主要是针对视觉清晰度的下降，而忽略了语义操作，使得它们要么容易被检测到，要么效果有限。在本文中，我们提出了互信息制导攻击(MIGA)，这是第一种设计用于直接攻击深度去噪模型的方法，该方法通过对抗性扰动来战略性地破坏深度去噪模型保持语义内容的能力。通过最小化原始图像和去噪图像之间的互信息，来衡量语义相似性。MIGA强制去噪器产生感知上干净但语义改变的输出。虽然这些图像在视觉上看起来是可信的，但它们编码的是系统性扭曲的语义，揭示了去噪模型中的一个根本漏洞。这些失真持续存在于去噪输出中，并可通过下游任务执行情况进行量化评估。我们提出了新的评价指标，并在五个数据集的四个去噪模型上对MIGA进行了系统的评估，证明了它在破坏语义保真度方面的一致有效性。我们的发现表明，去噪模型并不总是健壮的，可能会在现实世界的应用中引入安全风险。



## **10. Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks**

通过更强的个人攻击实现百万规模的对抗稳健性评估 cs.LG

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2411.15210v4) [paper-pdf](http://arxiv.org/pdf/2411.15210v4)

**Authors**: Yong Xie, Weijie Zheng, Hanxun Huang, Guangnan Ye, Xingjun Ma

**Abstract**: As deep learning models are increasingly deployed in safety-critical applications, evaluating their vulnerabilities to adversarial perturbations is essential for ensuring their reliability and trustworthiness. Over the past decade, a large number of white-box adversarial robustness evaluation methods (i.e., attacks) have been proposed, ranging from single-step to multi-step methods and from individual to ensemble methods. Despite these advances, challenges remain in conducting meaningful and comprehensive robustness evaluations, particularly when it comes to large-scale testing and ensuring evaluations reflect real-world adversarial risks. In this work, we focus on image classification models and propose a novel individual attack method, Probability Margin Attack (PMA), which defines the adversarial margin in the probability space rather than the logits space. We analyze the relationship between PMA and existing cross-entropy or logits-margin-based attacks, and show that PMA can outperform the current state-of-the-art individual methods. Building on PMA, we propose two types of ensemble attacks that balance effectiveness and efficiency. Furthermore, we create a million-scale dataset, CC1M, derived from the existing CC3M dataset, and use it to conduct the first million-scale white-box adversarial robustness evaluation of adversarially-trained ImageNet models. Our findings provide valuable insights into the robustness gaps between individual versus ensemble attacks and small-scale versus million-scale evaluations.

摘要: 随着深度学习模型越来越多地被部署在安全关键应用中，评估它们对敌意扰动的脆弱性对于确保它们的可靠性和可信性至关重要。在过去的十年里，已经提出了大量的白盒对抗健壮性评估方法(即攻击)，从单步方法到多步方法，从个体方法到集成方法。尽管取得了这些进展，但在进行有意义和全面的稳健性评估方面仍然存在挑战，特别是在进行大规模测试和确保评估反映现实世界的对抗性风险方面。本文重点研究了图像分类模型，提出了一种新的个体攻击方法--概率边缘攻击(PMA)，它在概率空间而不是Logits空间中定义了敌方边缘。我们分析了PMA与现有的基于交叉熵或Logits差值的攻击之间的关系，并证明了PMA的性能优于目前最先进的个别方法。在PMA的基础上，我们提出了两种平衡有效性和效率的集成攻击。此外，我们从现有的CC3M数据集中创建了一个百万尺度的数据集CC1M，并使用它对经过对手训练的ImageNet模型进行了第一次百万尺度的白盒对抗健壮性评估。我们的发现对个体攻击与整体攻击以及小规模评估与百万规模评估之间的稳健性差距提供了有价值的见解。



## **11. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

对抗性文本的人在循环生成：以藏传文字为例 cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.12478v2) [paper-pdf](http://arxiv.org/pdf/2412.12478v2)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models perform excellently on various tasks, but even SOTA LLMs are susceptible to textual adversarial attacks. Adversarial texts play crucial roles in multiple subfields of NLP. However, current research has the following issues. (1) Most textual adversarial attack methods target rich-resourced languages. How do we generate adversarial texts for less-studied languages? (2) Most textual adversarial attack methods are prone to generating invalid or ambiguous adversarial texts. How do we construct high-quality adversarial robustness benchmarks? (3) New language models may be immune to part of previously generated adversarial texts. How do we update adversarial robustness benchmarks? To address the above issues, we introduce HITL-GAT, a system based on a general approach to human-in-the-loop generation of adversarial texts. HITL-GAT contains four stages in one pipeline: victim model construction, adversarial example generation, high-quality benchmark construction, and adversarial robustness evaluation. Additionally, we utilize HITL-GAT to make a case study on Tibetan script which can be a reference for the adversarial research of other less-studied languages.

摘要: 基于DNN的语言模型在各种任务中表现出色，但即使是Sota LLM也容易受到文本攻击。对抗性语篇在自然语言处理的多个子领域发挥着至关重要的作用。然而，目前的研究存在以下问题。(1)大多数文本对抗性攻击方法针对的是资源丰富的语言。如何为较少研究的语言生成对抗性文本？(2)大多数文本对抗性攻击方法容易产生无效或歧义的对抗性文本。我们如何构建高质量的对抗性健壮性基准？(3)新的语言模型可能对先前生成的部分对抗性文本免疫。我们如何更新对手健壮性基准？为了解决上述问题，我们引入了HITL-GAT，这是一个基于人在环中生成对抗性文本的通用方法的系统。HITL-GAT在一条流水线上包括四个阶段：受害者模型构建、对手实例生成、高质量基准构建和对手健壮性评估。此外，我们还利用HITL-GAT对藏文进行了实例研究，对其他研究较少的语言的对抗性研究具有一定的借鉴意义。



## **12. Safety Guardrails for LLM-Enabled Robots**

LLM支持机器人的安全护栏 cs.RO

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07885v1) [paper-pdf](http://arxiv.org/pdf/2503.07885v1)

**Authors**: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani

**Abstract**: Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots.

摘要: 尽管将大型语言模型(LLM)集成到机器人学中释放了变革性的能力，但它也带来了重大的安全问题，从平均情况下的LLM错误(例如，幻觉)到对抗性的越狱攻击，这可能会在现实世界中产生有害的机器人行为。传统的机器人安全方法不能解决LLM的新脆弱性，而当前的LLM安全护栏忽略了机器人在动态真实环境中操作所带来的物理风险。在本文中，我们提出了一种两级护栏结构RoboGuard，以确保LLM支持的机器人的安全。RoboGuard首先通过使用信任根LLM将预定义的安全规则与机器人环境相关联，该LLM使用思想链(COT)推理来生成严格的安全规范，如时间逻辑约束。然后，RoboGuard使用时态逻辑控制合成来解决这些上下文安全规范和可能不安全的计划之间的潜在冲突，这确保了安全合规性，同时将违反用户偏好的程度降至最低。通过考虑最坏情况下的越狱攻击的大量模拟和真实世界实验，我们证明了RoboGuard在不影响安全计划的性能的情况下，将不安全计划的执行率从92%降低到2.5%以下。我们还证明了RoboGuard是资源高效的，对自适应攻击具有健壮性，并且通过使其信任根LLM能够执行CoT推理而显著增强。这些结果突显了RoboGuard在缓解安全风险和提高启用LLM的机器人可靠性方面的潜力。



## **13. ReLATE: Resilient Learner Selection for Multivariate Time-Series Classification Against Adversarial Attacks**

ReLATE：针对对抗性攻击的多元时间序列分类的弹性学习者选择 cs.LG

Accepted by the AAAI-25 Workshop on Artificial Intelligence for Time  Series Analysis (AI4TS)

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07882v1) [paper-pdf](http://arxiv.org/pdf/2503.07882v1)

**Authors**: Cagla Ipek Kocal, Onat Gungor, Aaron Tartz, Tajana Rosing, Baris Aksanli

**Abstract**: Minimizing computational overhead in time-series classification, particularly in deep learning models, presents a significant challenge. This challenge is further compounded by adversarial attacks, emphasizing the need for resilient methods that ensure robust performance and efficient model selection. We introduce ReLATE, a framework that identifies robust learners based on dataset similarity, reduces computational overhead, and enhances resilience. ReLATE maintains multiple deep learning models in well-known adversarial attack scenarios, capturing model performance. ReLATE identifies the most analogous dataset to a given target using a similarity metric, then applies the optimal model from the most similar dataset. ReLATE reduces computational overhead by an average of 81.2%, enhancing adversarial resilience and streamlining robust model selection, all without sacrificing performance, within 4.2% of Oracle.

摘要: 最大限度地减少时间序列分类中的计算负担，特别是深度学习模型中的计算负担，提出了一个重大挑战。对抗性攻击进一步加剧了这一挑战，强调了对确保稳健性能和高效模型选择的弹性方法的需要。我们引入了ReLATE，这是一个基于数据集相似性识别稳健学习器、减少计算负担并增强弹性的框架。ReLATE在众所周知的对抗性攻击场景中维护多个深度学习模型，捕捉模型性能。ReLATE使用相似性指标识别与给定目标最相似的数据集，然后应用来自最相似数据集的最佳模型。ReLATE平均降低了81.2%的计算负担，增强了对抗弹性并简化了稳健的模型选择，而所有这些都不会牺牲性能，仅为Oracle的4.2%。



## **14. On the Byzantine Fault Tolerance of signSGD with Majority Vote**

论多数票签名新加坡元的拜占庭式过失容忍 cs.LG

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2502.19170v2) [paper-pdf](http://arxiv.org/pdf/2502.19170v2)

**Authors**: Emanuele Mengoli, Luzius Moll, Virgilio Strozzi, El-Mahdi El-Mhamdi

**Abstract**: In distributed learning, sign-based compression algorithms such as signSGD with majority vote provide a lightweight alternative to SGD with an additional advantage: fault tolerance (almost) for free. However, for signSGD with majority vote, this fault tolerance has been shown to cover only the case of weaker adversaries, i.e., ones that are not omniscient or cannot collude to base their attack on common knowledge and strategy. In this work, we close this gap and provide new insights into how signSGD with majority vote can be resilient against omniscient and colluding adversaries, which craft an attack after communicating with other adversaries, thus having better information to perform the most damaging attack based on a common optimal strategy. Our core contribution is in providing a proof that begins by defining the omniscience framework and the strongest possible damage against signSGD with majority vote without imposing any restrictions on the attacker. Thanks to the filtering effect of the sign-based method, we upper-bound the space of attacks to the optimal strategy for maximizing damage by an attacker. Hence, we derive an explicit probabilistic bound in terms of incorrect aggregation without resorting to unknown constants, providing a convergence bound on signSGD with majority vote in the presence of Byzantine attackers, along with a precise convergence rate. Our findings are supported by experiments on the MNIST dataset in a distributed learning environment with adversaries of varying strength.

摘要: 在分布式学习中，基于符号的压缩算法，如多数投票的signSGD，提供了一种轻量级的SGD替代方案，具有额外的优势：几乎是免费的容错。然而，对于拥有多数票的signSGD来说，这种容错已经被证明只涵盖较弱的对手的情况，即那些不是无所不知的或不能串通以基于常识和策略的攻击的情况。在这项工作中，我们缩小了这一差距，并提供了新的见解，即拥有多数选票的signSGD如何具有抵御无所不知和串通的对手的能力，这些对手在与其他对手沟通后策划攻击，从而拥有更好的信息，基于共同的最优策略执行最具破坏性的攻击。我们的核心贡献是提供了一种证明，首先定义了无所不知的框架，并以多数票对signSGD造成了最强的破坏，而不对攻击者施加任何限制。由于基于符号的方法的过滤效果，我们将攻击空间上界到最优策略，以最大化攻击者的损害。因此，我们在不求助于未知常量的情况下，得到了不正确聚集的显式概率界，在拜占庭攻击者存在的情况下，提供了带多数投票的signSGD的收敛界，并提供了精确的收敛速度。我们的发现得到了在MNIST数据集上的实验支持，该实验在分布式学习环境中具有不同强度的对手。



## **15. Runtime Detection of Adversarial Attacks in AI Accelerators Using Performance Counters**

使用性能计数器检测人工智能加速器中的对抗攻击 cs.CR

7 pages, 8 figures

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07568v1) [paper-pdf](http://arxiv.org/pdf/2503.07568v1)

**Authors**: Habibur Rahaman, Atri Chatterjee, Swarup Bhunia

**Abstract**: Rapid adoption of AI technologies raises several major security concerns, including the risks of adversarial perturbations, which threaten the confidentiality and integrity of AI applications. Protecting AI hardware from misuse and diverse security threats is a challenging task. To address this challenge, we propose SAMURAI, a novel framework for safeguarding against malicious usage of AI hardware and its resilience to attacks. SAMURAI introduces an AI Performance Counter (APC) for tracking dynamic behavior of an AI model coupled with an on-chip Machine Learning (ML) analysis engine, known as TANTO (Trained Anomaly Inspection Through Trace Observation). APC records the runtime profile of the low-level hardware events of different AI operations. Subsequently, the summary information recorded by the APC is processed by TANTO to efficiently identify potential security breaches and ensure secure, responsible use of AI. SAMURAI enables real-time detection of security threats and misuse without relying on traditional software-based solutions that require model integration. Experimental results demonstrate that SAMURAI achieves up to 97% accuracy in detecting adversarial attacks with moderate overhead on various AI models, significantly outperforming conventional software-based approaches. It enhances security and regulatory compliance, providing a comprehensive solution for safeguarding AI against emergent threats.

摘要: 人工智能技术的快速采用带来了几个主要的安全问题，包括对抗性扰动的风险，这威胁到人工智能应用程序的机密性和完整性。保护人工智能硬件免受滥用和多样化的安全威胁是一项具有挑战性的任务。为了应对这一挑战，我们提出了武士，这是一个新的框架，用于防范恶意使用人工智能硬件及其对攻击的弹性。Samurai引入了AI性能计数器(APC)来跟踪AI模型的动态行为，并结合了芯片上的机器学习(ML)分析引擎，称为Tanto(通过跟踪观察进行训练的异常检测)。APC记录不同AI操作的底层硬件事件的运行时配置文件。随后，APC记录的摘要信息由Tanto处理，以有效地识别潜在的安全漏洞，并确保安全、负责任地使用人工智能。Samurai能够实时检测安全威胁和滥用，而不需要依赖需要模型集成的传统基于软件的解决方案。实验结果表明，武士在各种人工智能模型上以适度的开销检测敌意攻击的准确率高达97%，显著优于传统的基于软件的方法。它增强了安全性和监管合规性，为保护人工智能免受紧急威胁提供了全面的解决方案。



## **16. Transform-Dependent Adversarial Attacks**

依赖转换的对抗攻击 cs.CV

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2406.08443v2) [paper-pdf](http://arxiv.org/pdf/2406.08443v2)

**Authors**: Yaoteng Tan, Zikui Cai, M. Salman Asif

**Abstract**: Deep networks are highly vulnerable to adversarial attacks, yet conventional attack methods utilize static adversarial perturbations that induce fixed mispredictions. In this work, we exploit an overlooked property of adversarial perturbations--their dependence on image transforms--and introduce transform-dependent adversarial attacks. Unlike traditional attacks, our perturbations exhibit metamorphic properties, enabling diverse adversarial effects as a function of transformation parameters. We demonstrate that this transform-dependent vulnerability exists across different architectures (e.g., CNN and transformer), vision tasks (e.g., image classification and object detection), and a wide range of image transforms. Additionally, we show that transform-dependent perturbations can serve as a defense mechanism, preventing sensitive information disclosure when image enhancement transforms pose a risk of revealing private content. Through analysis in blackbox and defended model settings, we show that transform-dependent perturbations achieve high targeted attack success rates, outperforming state-of-the-art transfer attacks by 17-31% in blackbox scenarios. Our work introduces novel, controllable paradigm for adversarial attack deployment, revealing a previously overlooked vulnerability in deep networks.

摘要: 深层网络很容易受到敌意攻击，然而传统的攻击方法利用静态的对抗性扰动，导致固定的错误预测。在这项工作中，我们利用了对抗性扰动的一个被忽视的属性--它们对图像变换的依赖--并引入了依赖于变换的对抗性攻击。与传统的攻击不同，我们的扰动表现出变形的性质，使不同的对抗效果作为变换参数的函数。我们证明了这种依赖于变换的漏洞存在于不同的架构(例如，CNN和Transformer)、视觉任务(例如，图像分类和目标检测)以及广泛的图像变换中。此外，我们还证明了依赖于变换的扰动可以作为一种防御机制，当图像增强变换带来泄露隐私内容的风险时，可以防止敏感信息泄露。通过在黑盒和防御模型设置下的分析，我们证明了依赖于变换的扰动获得了高的目标攻击成功率，在黑盒场景中比最先进的传输攻击高出17%-31%。我们的工作为对抗性攻击部署引入了新的、可控的范例，揭示了以前在深层网络中被忽视的漏洞。



## **17. Learning to Localize Leakage of Cryptographic Sensitive Variables**

学习本地化加密敏感变量的泄漏 cs.LG

52 pages, 30 figures. Our code can be found at  https://github.com/jimgammell/learning_to_localize_leakage

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07464v1) [paper-pdf](http://arxiv.org/pdf/2503.07464v1)

**Authors**: Jimmy Gammell, Anand Raghunathan, Abolfazl Hashemi, Kaushik Roy

**Abstract**: While cryptographic algorithms such as the ubiquitous Advanced Encryption Standard (AES) are secure, *physical implementations* of these algorithms in hardware inevitably 'leak' sensitive data such as cryptographic keys. A particularly insidious form of leakage arises from the fact that hardware consumes power and emits radiation in a manner that is statistically associated with the data it processes and the instructions it executes. Supervised deep learning has emerged as a state-of-the-art tool for carrying out *side-channel attacks*, which exploit this leakage by learning to map power/radiation measurements throughout encryption to the sensitive data operated on during that encryption. In this work we develop a principled deep learning framework for determining the relative leakage due to measurements recorded at different points in time, in order to inform *defense* against such attacks. This information is invaluable to cryptographic hardware designers for understanding *why* their hardware leaks and how they can mitigate it (e.g. by indicating the particular sections of code or electronic components which are responsible). Our framework is based on an adversarial game between a family of classifiers trained to estimate the conditional distributions of sensitive data given subsets of measurements, and a budget-constrained noise distribution which probabilistically erases individual measurements to maximize the loss of these classifiers. We demonstrate our method's efficacy and ability to overcome limitations of prior work through extensive experimental comparison with 8 baseline methods using 3 evaluation metrics and 6 publicly-available power/EM trace datasets from AES, ECC and RSA implementations. We provide an open-source PyTorch implementation of these experiments.

摘要: 虽然加密算法(如无处不在的高级加密标准(AES))是安全的，但这些算法在硬件中的*物理实现*不可避免地‘泄露’敏感数据，如密钥。一种特别隐蔽的泄漏形式源于这样一个事实，即硬件消耗功率并发出辐射，其方式与其处理的数据和执行的指令在统计上相关联。有监督的深度学习已经成为一种执行*侧通道攻击*的最先进工具，它通过学习将整个加密过程中的功率/辐射测量映射到加密过程中操作的敏感数据来利用这种泄漏。在这项工作中，我们开发了一个原则性的深度学习框架，用于确定由于在不同时间点记录的测量而导致的相对泄漏，以便为防御此类攻击提供信息。这些信息对于密码硬件设计者来说是非常有价值的，他们可以理解他们的硬件泄漏以及他们如何减轻泄漏(例如，通过指示负责的特定代码段或电子组件)。我们的框架基于一组分类器和预算受限噪声分布之间的对抗性博弈，前者被训练为估计给定测量子集的敏感数据的条件分布，后者概率地擦除单个测量以最大化这些分类器的损失。我们通过使用3个评估指标和6个来自AES、ECC和RSA实现的公开可用的POWER/EM跟踪数据集与8种基线方法进行广泛的实验比较，证明了我们的方法的有效性和克服先前工作的局限性的能力。我们提供了这些实验的开放源码的PyTorch实现。



## **18. The Uncanny Valley: Exploring Adversarial Robustness from a Flatness Perspective**

恐怖谷：从扁平的角度探索对抗性的稳健性 cs.LG

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2405.16918v2) [paper-pdf](http://arxiv.org/pdf/2405.16918v2)

**Authors**: Nils Philipp Walter, Linara Adilova, Jilles Vreeken, Michael Kamp

**Abstract**: Flatness of the loss surface not only correlates positively with generalization, but is also related to adversarial robustness since perturbations of inputs relate non-linearly to perturbations of weights. In this paper, we empirically analyze the relation between adversarial examples and relative flatness with respect to the parameters of one layer. We observe a peculiar property of adversarial examples in the context of relative flatness: during an iterative first-order white-box attack, the flatness of the loss surface measured around the adversarial example first becomes sharper until the label is flipped, but if we keep the attack running, it runs into a flat uncanny valley where the label remains flipped. In extensive experiments, we observe this phenomenon across various model architectures and datasets, even for adversarially trained models. Our results also extend to large language models (LLMs), but due to the discrete nature of the input space and comparatively weak attacks, adversarial examples rarely reach truly flat regions. Most importantly, this phenomenon shows that flatness alone cannot explain adversarial robustness unless we can also guarantee the behavior of the function around the examples. We, therefore theoretically connect relative flatness to adversarial robustness by bounding the third derivative of the loss surface, underlining the need for flatness in combination with a low global Lipschitz constant for a robust model.

摘要: 损失曲面的平坦性不仅与泛化正相关，而且还与对抗稳健性有关，因为输入的扰动与权重的扰动是非线性相关的。在这篇文章中，我们实证分析了对抗性例子与相对平坦度之间的关系。我们在相对平坦的背景下观察到对抗性例子的一个特殊性质：在迭代的一阶白盒攻击中，围绕对抗性例子测量的损失曲面的平坦性首先变得更尖锐，直到标签被翻转，但如果我们继续攻击，它会进入一个平坦的可怕山谷，在那里标签仍然被翻转。在广泛的实验中，我们在各种模型体系结构和数据集上观察到了这种现象，甚至对于经过恶意训练的模型也是如此。我们的结果也扩展到大型语言模型(LLM)，但由于输入空间的离散性质和相对较弱的攻击，对抗性例子很少到达真正平坦的区域。最重要的是，这一现象表明，平坦性本身不能解释对抗健壮性，除非我们也能保证函数在示例周围的行为。因此，理论上，我们通过限定损失曲面的三阶导数，将相对平坦性与对手的稳健性联系起来，强调了平坦性与稳健性模型的低全局Lipschitz常数相结合的必要性。



## **19. MIBench: A Comprehensive Framework for Benchmarking Model Inversion Attack and Defense**

MIBunch：基准模型倒置攻击和防御的综合框架 cs.CV

20 pages

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2410.05159v3) [paper-pdf](http://arxiv.org/pdf/2410.05159v3)

**Authors**: Yixiang Qiu, Hongyao Yu, Hao Fang, Tianqu Zhuang, Wenbo Yu, Bin Chen, Xuan Wang, Shu-Tao Xia, Ke Xu

**Abstract**: Model Inversion (MI) attacks aim at leveraging the output information of target models to reconstruct privacy-sensitive training data, raising critical concerns regarding the privacy vulnerabilities of Deep Neural Networks (DNNs). Unfortunately, in tandem with the rapid evolution of MI attacks, the absence of a comprehensive benchmark with standardized metrics and reproducible implementations has emerged as a formidable challenge. This deficiency has hindered objective comparison of methodological advancements and reliable assessment of defense efficacy. To address this critical gap, we build the first practical benchmark named MIBench for systematic evaluation of model inversion attacks and defenses. This benchmark bases on an extensible and reproducible modular-based toolbox which currently integrates a total of 19 state-of-the-art attack and defense methods and encompasses 9 standardized evaluation protocols. Capitalizing on this foundation, we conduct extensive evaluation from multiple perspectives to holistically compare and analyze various methods across different scenarios, such as the impact of target resolution, model predictive power, defense performance and adversarial robustness.

摘要: 模型反转(MI)攻击旨在利用目标模型的输出信息来重建隐私敏感的训练数据，这引发了人们对深度神经网络(DNN)隐私漏洞的严重担忧。不幸的是，随着MI攻击的快速发展，缺乏具有标准化指标和可重复实现的全面基准已成为一个艰巨的挑战。这一缺陷阻碍了对方法进步的客观比较和对防御效能的可靠评估。为了解决这一关键差距，我们构建了第一个名为MIBch的实用基准，用于对模型反转攻击和防御进行系统评估。该基准基于一个可扩展和可重现的模块化工具箱，该工具箱目前集成了19种最先进的攻击和防御方法，并包含9个标准化的评估协议。在此基础上，我们从多个角度进行了广泛的评估，对各种方法在不同场景下的目标分辨率、模型预测力、防御性能和对抗健壮性的影响进行了全面的比较和分析。



## **20. State Frequency Estimation for Anomaly Detection**

异常检测的状态频率估计 cs.LG

12 pages

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2412.03442v2) [paper-pdf](http://arxiv.org/pdf/2412.03442v2)

**Authors**: Clinton Cao, Agathe Blaise, Annibale Panichella, Sicco Verwer

**Abstract**: Many works have studied the efficacy of state machines for detecting anomalies within NetFlows. These works typically learn a model from unlabeled data and compute anomaly scores for arbitrary traces based on their likelihood of occurrence or how well they fit within the model. However, these methods do not dynamically adapt their scores based on the traces seen at test time. This becomes a problem when an adversary produces seemingly common traces in their attack, causing the model to miss the detection by assigning low anomaly scores. We propose SEQUENT, a new unsupervised approach that uses the state visit frequency of a state machine to adapt its scoring dynamically for anomaly detection. SEQUENT subsequently uses the scores to generate root causes for anomalies. These allow the grouping of alarms and simplify the analysis of anomalies. We evaluate SEQUENT's effectiveness in detecting network anomalies on three publicly available NetFlow datasets and compare its performance against various existing unsupervised anomaly detection methods. Our evaluation shows promising results for using the state visit frequency of a state machine to detect network anomalies.

摘要: 许多工作已经研究了状态机对检测NetFlow中异常的有效性。这些工作通常从未标记的数据中学习模型，并根据它们发生的可能性或它们在模型中的匹配程度来计算任意踪迹的异常分数。然而，这些方法不会根据测试时看到的痕迹动态调整它们的分数。当对手在他们的攻击中产生看似常见的痕迹，导致模型通过分配较低的异常分数而错过检测时，这就成了一个问题。我们提出了一种新的无监督方法Sequent，它利用状态机的状态访问频率来动态调整其评分以进行异常检测。Sequent随后使用这些分数来生成异常的根本原因。这些功能允许对警报进行分组，并简化异常分析。我们评估了Sequent在三个公开可用的NetFlow数据集上检测网络异常的有效性，并将其性能与现有的各种非监督异常检测方法进行了比较。我们的评估结果表明，使用状态机的状态访问频率来检测网络异常是有希望的结果。



## **21. Machine Against the RAG: Jamming Retrieval-Augmented Generation with Blocker Documents**

机器对抗RAG：用阻止器文档干扰检索增强生成 cs.CR

To appear in USENIX Security Symposium 2025

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2406.05870v4) [paper-pdf](http://arxiv.org/pdf/2406.05870v4)

**Authors**: Avital Shafran, Roei Schuster, Vitaly Shmatikov

**Abstract**: Retrieval-augmented generation (RAG) systems respond to queries by retrieving relevant documents from a knowledge database and applying an LLM to the retrieved documents. We demonstrate that RAG systems that operate on databases with untrusted content are vulnerable to denial-of-service attacks we call jamming. An adversary can add a single ``blocker'' document to the database that will be retrieved in response to a specific query and result in the RAG system not answering this query, ostensibly because it lacks relevant information or because the answer is unsafe.   We describe and measure the efficacy of several methods for generating blocker documents, including a new method based on black-box optimization. Our method (1) does not rely on instruction injection, (2) does not require the adversary to know the embedding or LLM used by the target RAG system, and (3) does not employ an auxiliary LLM.   We evaluate jamming attacks on several embeddings and LLMs and demonstrate that the existing safety metrics for LLMs do not capture their vulnerability to jamming. We then discuss defenses against blocker documents.

摘要: 检索-增强生成(RAG)系统通过从知识数据库中检索相关文档并将LLM应用于所检索的文档来响应查询。我们演示了在包含不可信内容的数据库上运行的RAG系统容易受到我们称为干扰的拒绝服务攻击。攻击者可以在数据库中添加一个“拦截器”文档，该文档将响应于特定查询而被检索，并导致RAG系统不回答该查询，表面上是因为它缺乏相关信息或因为答案不安全。我们描述并测试了几种生成拦截器文档的方法的有效性，其中包括一种基于黑盒优化的新方法。我们的方法(1)不依赖于指令注入，(2)不要求攻击者知道目标RAG系统使用的嵌入或LLM，(3)不使用辅助LLM。我们评估了几种嵌入和LLM上的干扰攻击，并证明了现有的LLM安全度量没有捕捉到它们对干扰的脆弱性。然后我们讨论针对拦截器文档的防御。



## **22. PGD-Imp: Rethinking and Unleashing Potential of Classic PGD with Dual Strategies for Imperceptible Adversarial Attacks**

PGD-Imp：通过双重策略重新思考和释放经典PVD的潜力，以应对难以感知的对抗攻击 cs.LG

Accepted by IEEE ICASSP 2025. Please cite this paper using the  following format: J. Li, Z. Yu, Z. He, Z. Wang, X. Kang, "PGD-Imp: Rethinking  and Unleashing Potential of Classic PGD with Dual Strategies for  Imperceptible Adversarial Attacks," in proc. of International Conference on  Acoustics, Speech, and Signal Processing 2025 (ICASSP 2025), Hyderabad,  India, 2025-4-06 to 2025-04-11

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2412.11168v3) [paper-pdf](http://arxiv.org/pdf/2412.11168v3)

**Authors**: Jin Li, Zitong Yu, Ziqiang He, Z. Jane Wang, Xiangui Kang

**Abstract**: Imperceptible adversarial attacks have recently attracted increasing research interests. Existing methods typically incorporate external modules or loss terms other than a simple $l_p$-norm into the attack process to achieve imperceptibility, while we argue that such additional designs may not be necessary. In this paper, we rethink the essence of imperceptible attacks and propose two simple yet effective strategies to unleash the potential of PGD, the common and classical attack, for imperceptibility from an optimization perspective. Specifically, the Dynamic Step Size is introduced to find the optimal solution with minimal attack cost towards the decision boundary of the attacked model, and the Adaptive Early Stop strategy is adopted to reduce the redundant strength of adversarial perturbations to the minimum level. The proposed PGD-Imperceptible (PGD-Imp) attack achieves state-of-the-art results in imperceptible adversarial attacks for both untargeted and targeted scenarios. When performing untargeted attacks against ResNet-50, PGD-Imp attains 100$\%$ (+0.3$\%$) ASR, 0.89 (-1.76) $l_2$ distance, and 52.93 (+9.2) PSNR with 57s (-371s) running time, significantly outperforming existing methods.

摘要: 潜伏的敌意攻击最近吸引了越来越多的研究兴趣。现有的方法通常在攻击过程中加入外部模块或损失项，而不是简单的$L_p$-范数来实现不可感知性，而我们认为这样的额外设计可能不是必要的。本文从优化的角度重新思考了不可察觉攻击的本质，并提出了两种简单而有效的策略来释放PGD攻击--普通攻击和经典攻击--的不可感知性。具体地，引入动态步长在攻击模型的决策边界附近寻找攻击代价最小的最优解，并采用自适应提前停止策略将敌方扰动的冗余强度降至最小。建议的PGD-Imp(PGD-Imp)攻击在非目标场景和目标场景中都实现了最先进的不可感知对手攻击。在对ResNet-50进行非定向攻击时，PGD-Imp在57s(-371s)的运行时间内获得了100$(+0.3$)ASR，0.89(-1.76)$L_2$距离和52.93(+9.2)PSNR，显著优于现有方法。



## **23. Breaking the Limits of Quantization-Aware Defenses: QADT-R for Robustness Against Patch-Based Adversarial Attacks in QNNs**

打破量化感知防御的局限：QADT-R针对QNN中基于补丁的对抗攻击的鲁棒性 cs.CV

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07058v1) [paper-pdf](http://arxiv.org/pdf/2503.07058v1)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Quantized Neural Networks (QNNs) have emerged as a promising solution for reducing model size and computational costs, making them well-suited for deployment in edge and resource-constrained environments. While quantization is known to disrupt gradient propagation and enhance robustness against pixel-level adversarial attacks, its effectiveness against patch-based adversarial attacks remains largely unexplored. In this work, we demonstrate that adversarial patches remain highly transferable across quantized models, achieving over 70\% attack success rates (ASR) even at extreme bit-width reductions (e.g., 2-bit). This challenges the common assumption that quantization inherently mitigates adversarial threats. To address this, we propose Quantization-Aware Defense Training with Randomization (QADT-R), a novel defense strategy that integrates Adaptive Quantization-Aware Patch Generation (A-QAPA), Dynamic Bit-Width Training (DBWT), and Gradient-Inconsistent Regularization (GIR) to enhance resilience against highly transferable patch-based attacks. A-QAPA generates adversarial patches within quantized models, ensuring robustness across different bit-widths. DBWT introduces bit-width cycling during training to prevent overfitting to a specific quantization setting, while GIR injects controlled gradient perturbations to disrupt adversarial optimization. Extensive evaluations on CIFAR-10 and ImageNet show that QADT-R reduces ASR by up to 25\% compared to prior defenses such as PBAT and DWQ. Our findings further reveal that PBAT-trained models, while effective against seen patch configurations, fail to generalize to unseen patches due to quantization shift. Additionally, our empirical analysis of gradient alignment, spatial sensitivity, and patch visibility provides insights into the mechanisms that contribute to the high transferability of patch-based attacks in QNNs.

摘要: 量化神经网络(QNN)作为一种有前途的解决方案，可以减少模型的规模和计算成本，使其非常适合在边缘和资源受限的环境中部署。虽然众所周知，量化可以中断梯度传播并增强对像素级攻击的稳健性，但它对抗基于补丁的攻击的有效性在很大程度上仍未被探索。在这项工作中，我们证明了敌意补丁在量化模型之间仍然具有高度的可传输性，即使在极端的比特宽度减少(例如，2比特)时，攻击成功率(ASR)也达到了70%以上。这挑战了量化天生就能缓解对抗性威胁的普遍假设。针对这一问题，我们提出了基于随机化的量化感知防御训练(QADT-R)，这是一种融合了自适应量化感知补丁生成(A-QAPA)、动态位宽训练(DBWT)和梯度不一致正则化(GIR)的防御策略，以增强对高度可传输的基于补丁攻击的抗攻击能力。A-QAPA在量化模型中生成对抗性补丁，确保了不同位宽的健壮性。DBWT在训练期间引入位宽循环，以防止过度适应特定的量化设置，而GIR注入受控的梯度扰动，以中断对抗性优化。对CIFAR-10和ImageNet的广泛评估表明，QADT-R与之前的防御措施(如PBAT和DWQ)相比，ASR降低了25%。我们的发现进一步表明，PBAT训练的模型，虽然对看得见的补丁配置有效，但由于量化漂移而不能推广到看不见的补丁。此外，我们对梯度对齐、空间敏感度和补丁可见性的经验分析有助于深入了解基于补丁的攻击在QNN中的高可传递性的机制。



## **24. Utilizing Jailbreak Probability to Attack and Safeguard Multimodal LLMs**

利用越狱概率攻击和保护多模式LLM cs.CR

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06989v1) [paper-pdf](http://arxiv.org/pdf/2503.06989v1)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Deyue Zhang, Dongdong Yang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal contents. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on inputs to maximize jailbreak probability. To counteract attacks, we also propose two defensive methods: Jailbreak-Probability-based Finetuning (JPF) and Jailbreak-Probability-based Defensive Noise (JPDN), which minimizes jailbreak probability in the MLLM parameters and input space, respectively. Extensive experiments show that (1) JPA yields improvements (up to 28.38\%) under both white and black box settings compared to previous methods with small perturbation bounds and few iterations. (2) JPF and JPDN significantly reduce jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.

摘要: 近年来，多通道大语言模型(MLLMS)在理解多通道内容方面表现出了优越的能力。然而，他们仍然容易受到越狱攻击，这些攻击利用他们的安全调整中的弱点来产生有害的反应。之前的研究根据回应是否包含恶意内容将越狱分为成功或失败。然而，考虑到MLLM响应的随机性，这种对输入越狱MLLMS能力的二进制分类是不合适的。从这一观点出发，我们引入越狱概率来量化输入的越狱潜力，这代表了当提示该输入时MLLMS生成恶意响应的可能性。我们通过对MLLMS的多次查询来近似这一概率。在使用越狱概率预测网络(JPPN)对输入隐藏状态与其对应越狱概率之间的关系进行建模后，我们使用连续越狱概率进行优化。具体地说，我们提出了基于越狱概率的攻击(JPA)，它优化了输入上的敌意扰动，以最大化越狱概率。为了对抗攻击，我们还提出了两种防御方法：基于越狱概率的精调(JPF)和基于越狱概率的防御噪声(JPDN)，它们分别在MLLM参数和输入空间中最小化越狱概率。大量的实验表明：(1)JPA在白盒和黑盒设置下都比以往的方法在小的扰动界和很少的迭代次数下都有很大的改善(高达28.38%)。(2)JPF和JPDN可显著减少越狱次数，最多可减少60%以上。以上两个结果都证明了引入越狱概率来细微区分不同输入越狱能力的意义。



## **25. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

人工智能生成图像检测中的漏洞：对抗性攻击的挑战 cs.CV

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2407.20836v3) [paper-pdf](http://arxiv.org/pdf/2407.20836v3)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario. The code will be shared upon acceptance.

摘要: 最近在图像合成方面的进步，特别是随着GaN和扩散模型的出现，放大了公众对虚假信息传播的担忧。为了解决这些问题，人们已经提出了许多人工智能生成的图像(AIGI)检测器，并在识别虚假图像方面取得了良好的性能。然而，对于AIGI检测器的对抗健壮性，目前还缺乏系统的了解。本文研究了白盒和黑盒环境下最新的AIGI检测器抵抗敌意攻击的脆弱性，这是迄今为止很少被研究的。为此，我们提出了一种攻击AIGI探测器的新方法。首先，受真伪图像在频域存在明显差异的启发，在频域下加入扰动，使图像偏离其原有的频率分布。其次，我们探索了代理模型的完全后验分布，以进一步缩小不同AIGI检测器之间的差距，例如在CNN和VITS之间传输敌意示例。这是通过引入一种新颖的后训练贝叶斯策略来实现的，该策略将单个代理转变为贝叶斯策略，能够使用一个预先训练的代理来模拟不同的受害者模型，而不需要重新训练。我们将我们的方法命名为基于频率的后训练贝叶斯攻击，或称FPBA。通过FPBA，我们证明了敌意攻击是对AIGI检测器的真正威胁，因为FPBA可以跨模型、生成器、防御方法提供成功的黑盒攻击，甚至可以逃避交叉生成器检测，这是现实世界中的一个关键检测场景。代码将在接受后共享。



## **26. CtrlRAG: Black-box Adversarial Attacks Based on Masked Language Models in Retrieval-Augmented Language Generation**

CtrlRAG：检索增强语言生成中基于掩蔽语言模型的黑匣子对抗攻击 cs.CL

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06950v1) [paper-pdf](http://arxiv.org/pdf/2503.06950v1)

**Authors**: Runqi Sui

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by integrating external knowledge bases. However, this integration introduces a new security threat: adversaries can exploit the retrieval mechanism to inject malicious content into the knowledge base, thereby influencing the generated responses. Based on this attack vector, we propose CtrlRAG, a novel attack method designed for RAG system in the black-box setting, which aligns with real-world scenarios. Unlike existing attack methods, CtrlRAG introduces a perturbation mechanism using Masked Language Model (MLM) to dynamically optimize malicious content in response to changes in the retrieved context. Experimental results demonstrate that CtrlRAG outperforms three baseline methods in both Emotional Manipulation and Hallucination Amplification objectives. Furthermore, we evaluate three existing defense mechanisms, revealing their limited effectiveness against CtrlRAG and underscoring the urgent need for more robust defenses.

摘要: 检索-增强生成(RAG)系统通过集成外部知识库来增强大型语言模型(LLMS)。然而，这种集成带来了新的安全威胁：攻击者可以利用检索机制将恶意内容注入知识库，从而影响生成的响应。基于该攻击向量，我们提出了一种新的针对黑盒环境下RAG系统的攻击方法CtrlRAG，该方法符合实际场景。与现有的攻击方法不同，CtrlRAG引入了一种使用掩蔽语言模型(MLM)的扰动机制来动态优化恶意内容，以响应检索到的上下文的变化。实验结果表明，CtrlRAG在情绪操纵和幻觉放大目标上都优于三种基线方法。此外，我们评估了三种现有的防御机制，揭示了它们对CtrlRAG的有限有效性，并强调了对更强大防御的迫切需要。



## **27. When Lighting Deceives: Exposing Vision-Language Models' Illumination Vulnerability Through Illumination Transformation Attack**

当灯光欺骗时：通过照明转换攻击暴露视觉语言模型的照明脆弱性 cs.CV

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.06903v1) [paper-pdf](http://arxiv.org/pdf/2503.06903v1)

**Authors**: Hanqing Liu, Shouwei Ruan, Yao Huang, Shiji Zhao, Xingxing Wei

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable success in various tasks, yet their robustness to real-world illumination variations remains largely unexplored. To bridge this gap, we propose \textbf{I}llumination \textbf{T}ransformation \textbf{A}ttack (\textbf{ITA}), the first framework to systematically assess VLMs' robustness against illumination changes. However, there still exist two key challenges: (1) how to model global illumination with fine-grained control to achieve diverse lighting conditions and (2) how to ensure adversarial effectiveness while maintaining naturalness. To address the first challenge, we innovatively decompose global illumination into multiple parameterized point light sources based on the illumination rendering equation. This design enables us to model more diverse lighting variations that previous methods could not capture. Then, by integrating these parameterized lighting variations with physics-based lighting reconstruction techniques, we could precisely render such light interactions in the original scenes, finally meeting the goal of fine-grained lighting control. For the second challenge, by controlling illumination through the lighting reconstrution model's latent space rather than direct pixel manipulation, we inherently preserve physical lighting priors. Furthermore, to prevent potential reconstruction artifacts, we design additional perceptual constraints for maintaining visual consistency with original images and diversity constraints for avoiding light source convergence.   Extensive experiments demonstrate that our ITA could significantly reduce the performance of advanced VLMs, e.g., LLaVA-1.6, while possessing competitive naturalness, exposing VLMS' critical illuminiation vulnerabilities.

摘要: 视觉语言模型已经在各种任务中取得了显著的成功，但它们对真实世界光照变化的稳健性在很大程度上还没有被探索。为了弥补这一差距，我们提出了第一个系统地评估VLMS对光照变化的稳健性的框架然而，仍然存在两个关键挑战：(1)如何通过细粒度控制对全局光照进行建模，以实现不同的光照条件；(2)如何在保持自然度的同时确保对抗效果。为了解决第一个挑战，我们创新性地基于光照渲染方程将全局光照分解为多个参数化点光源。这种设计使我们能够对以前的方法无法捕捉到的更多样化的照明变化进行建模。然后，通过将这些参数化的光照变化与基于物理的光照重建技术相结合，我们可以在原始场景中精确地渲染这种光照交互，最终达到细粒度光照控制的目标。对于第二个挑战，通过光照重建模型的潜在空间而不是直接的像素操作来控制光照，我们本质上保持了物理光照的先验。此外，为了防止潜在的重建伪影，我们设计了额外的感知约束来保持与原始图像的视觉一致性，并设计了多样性约束来避免光源收敛。广泛的实验表明，我们的ITA可以显著降低高级VLMS(例如LLaVA-1.6)的性能，同时具有竞争性的自然性，暴露了VLMS的关键照明漏洞。



## **28. Exploring the Adversarial Vulnerabilities of Vision-Language-Action Models in Robotics**

探索机器人学中视觉-语言-动作模型的对抗脆弱性 cs.RO

Github: https://github.com/William-wAng618/roboticAttack Homepage:  https://vlaattacker.github.io/

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2411.13587v3) [paper-pdf](http://arxiv.org/pdf/2411.13587v3)

**Authors**: Taowen Wang, Cheng Han, James Chenhao Liang, Wenhao Yang, Dongfang Liu, Luna Xinyu Zhang, Qifan Wang, Jiebo Luo, Ruixiang Tang

**Abstract**: Recently in robotics, Vision-Language-Action (VLA) models have emerged as a transformative approach, enabling robots to execute complex tasks by integrating visual and linguistic inputs within an end-to-end learning framework. While VLA models offer significant capabilities, they also introduce new attack surfaces, making them vulnerable to adversarial attacks. With these vulnerabilities largely unexplored, this paper systematically quantifies the robustness of VLA-based robotic systems. Recognizing the unique demands of robotic execution, our attack objectives target the inherent spatial and functional characteristics of robotic systems. In particular, we introduce two untargeted attack objectives that leverage spatial foundations to destabilize robotic actions, and a targeted attack objective that manipulates the robotic trajectory. Additionally, we design an adversarial patch generation approach that places a small, colorful patch within the camera's view, effectively executing the attack in both digital and physical environments. Our evaluation reveals a marked degradation in task success rates, with up to a 100\% reduction across a suite of simulated robotic tasks, highlighting critical security gaps in current VLA architectures. By unveiling these vulnerabilities and proposing actionable evaluation metrics, we advance both the understanding and enhancement of safety for VLA-based robotic systems, underscoring the necessity for continuously developing robust defense strategies prior to physical-world deployments.

摘要: 最近在机器人学中，视觉-语言-动作(VLA)模型作为一种变革性的方法出现，使机器人能够通过在端到端学习框架内整合视觉和语言输入来执行复杂的任务。虽然VLA模型提供了重要的功能，但它们也引入了新的攻击面，使其容易受到对手攻击。由于这些漏洞在很大程度上是未知的，本文系统地量化了基于VLA的机器人系统的健壮性。认识到机器人执行的独特需求，我们的攻击目标针对机器人系统固有的空间和功能特征。特别是，我们引入了两个非定向攻击目标，它们利用空间基础来破坏机器人动作的稳定性，以及一个操纵机器人轨迹的定向攻击目标。此外，我们设计了一种对抗性补丁生成方法，将一个小的、五颜六色的补丁放置在相机的视野中，在数字和物理环境中有效地执行攻击。我们的评估显示任务成功率显著下降，一组模拟机器人任务最多减少100%，突出了当前VLA架构中的关键安全漏洞。通过揭示这些漏洞并提出可操作的评估指标，我们促进了对基于VLA的机器人系统安全性的理解和增强，强调了在物理世界部署之前持续开发强大的防御策略的必要性。



## **29. Quantum Chernoff divergence in advantage distillation for quantum key distribution and device-independent quantum key distribution**

量子密钥分发和设备无关量子密钥分发的优势提炼中的量子勒夫分歧 quant-ph

Close to published version

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2212.06975v3) [paper-pdf](http://arxiv.org/pdf/2212.06975v3)

**Authors**: Mikka Stasiuk, Norbert Lütkenhaus, Ernest Y. -Z. Tan

**Abstract**: Device-independent quantum key distribution (DIQKD) aims to mitigate adversarial exploitation of imperfections in quantum devices, by providing an approach for secret key distillation with modest security assumptions. Advantage distillation, a two-way communication procedure in error correction, has proven effective in raising noise tolerances in both device-dependent and device-independent QKD. Previously, device-independent security proofs against IID collective attacks were developed for an advantage distillation protocol known as the repetition-code protocol, based on security conditions involving the fidelity between some states in the protocol. However, there exists a gap between the sufficient and necessary security conditions, which hinders the calculation of tight noise-tolerance bounds based on the fidelity. We close this gap by presenting an alternative proof structure that replaces the fidelity with the quantum Chernoff divergence, a distinguishability measure that arises in symmetric hypothesis testing. Working in the IID collective attacks model, we derive matching sufficient and necessary conditions for the repetition-code protocol to be secure (up to a natural conjecture regarding the latter case) in terms of the quantum Chernoff divergence, hence indicating that this serves as the relevant quantity of interest for this protocol. Furthermore, using this security condition we obtain some improvements over previous results on the noise tolerance thresholds for DIQKD. Our results provide insight into a fundamental question in quantum information theory regarding the circumstances under which DIQKD is possible.

摘要: 独立于设备的量子密钥分发(DIQKD)旨在通过提供一种在适度的安全假设下提取密钥的方法来减少对量子设备中缺陷的恶意利用。优势蒸馏，一种纠错的双向通信过程，已被证明在提高设备相关和设备无关的量子密钥分发中的噪声容限方面都是有效的。以前，针对一种称为重码协议的优势提取协议，基于涉及协议中某些状态之间的保真度的安全条件，开发了针对IID集体攻击的独立于设备的安全证明。然而，在充分和必要的安全条件之间存在差距，这阻碍了基于保真度的紧噪声容限的计算。我们通过提出另一种证明结构来缩小这一差距，该结构用量子切尔诺夫发散取代了保真度，量子切尔诺夫发散是对称假设检验中出现的一种可区分性衡量标准。在IID集体攻击模型下，根据量子Chernoff散度，我们得到了重码协议安全的充要条件(直到关于后者的一个自然猜想)，从而表明这是该协议的相关关注量。此外，利用这一安全条件，我们在DIQKD的噪声容忍门限上得到了一些改进。我们的结果为量子信息论中的一个基本问题提供了洞察力，这个问题涉及到DIQKD可能的情况。



## **30. AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection**

AnywhereDoor：对象检测的多目标后门攻击 cs.CR

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2503.06529v1) [paper-pdf](http://arxiv.org/pdf/2503.06529v1)

**Authors**: Jialin Lu, Junjie Shan, Ziqi Zhao, Ka-Ho Chow

**Abstract**: As object detection becomes integral to many safety-critical applications, understanding its vulnerabilities is essential. Backdoor attacks, in particular, pose a serious threat by implanting hidden triggers in victim models, which adversaries can later exploit to induce malicious behaviors during inference. However, current understanding is limited to single-target attacks, where adversaries must define a fixed malicious behavior (target) before training, making inference-time adaptability impossible. Given the large output space of object detection (including object existence prediction, bounding box estimation, and classification), the feasibility of flexible, inference-time model control remains unexplored. This paper introduces AnywhereDoor, a multi-target backdoor attack for object detection. Once implanted, AnywhereDoor allows adversaries to make objects disappear, fabricate new ones, or mislabel them, either across all object classes or specific ones, offering an unprecedented degree of control. This flexibility is enabled by three key innovations: (i) objective disentanglement to scale the number of supported targets; (ii) trigger mosaicking to ensure robustness even against region-based detectors; and (iii) strategic batching to address object-level data imbalances that hinder manipulation. Extensive experiments demonstrate that AnywhereDoor grants attackers a high degree of control, improving attack success rates by 26% compared to adaptations of existing methods for such flexible control.

摘要: 随着对象检测成为许多安全关键型应用程序不可或缺的一部分，了解其漏洞至关重要。尤其是后门攻击，通过在受害者模型中植入隐藏的触发器，构成了严重的威胁，攻击者稍后可以利用这些触发器在推理过程中诱导恶意行为。然而，目前的理解仅限于单目标攻击，即攻击者必须在训练前定义一个固定的恶意行为(目标)，这使得推理时间适应性变得不可能。考虑到目标检测(包括目标存在预测、包围盒估计和分类)的大输出空间，灵活的推理时间模型控制的可行性仍未被探索。本文介绍了Anywhere Door，一种用于目标检测的多目标后门攻击。一旦被植入，Anywhere Door允许对手在所有对象类或特定对象类中让对象消失、捏造新对象或错误标记对象，提供前所未有的控制程度。这种灵活性是由三项关键创新实现的：(I)客观解缠，以扩大受支持目标的数量；(Ii)触发马赛克，以确保即使针对基于区域的探测器也具有稳健性；以及(Iii)战略批处理，以解决阻碍操纵的对象级数据失衡。广泛的实验表明，Anywhere Door给予攻击者高度的控制，与采用现有方法进行这种灵活的控制相比，攻击成功率提高了26%。



## **31. Visual Privacy Auditing with Diffusion Models**

使用扩散模型的视觉隐私审计 cs.LG

Published in Transactions on Machine Learning Research (TMLR)

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2403.07588v2) [paper-pdf](http://arxiv.org/pdf/2403.07588v2)

**Authors**: Kristian Schwethelm, Johannes Kaiser, Moritz Knolle, Sarah Lockfisch, Daniel Rueckert, Alexander Ziller

**Abstract**: Data reconstruction attacks on machine learning models pose a substantial threat to privacy, potentially leaking sensitive information. Although defending against such attacks using differential privacy (DP) provides theoretical guarantees, determining appropriate DP parameters remains challenging. Current formal guarantees on the success of data reconstruction suffer from overly stringent assumptions regarding adversary knowledge about the target data, particularly in the image domain, raising questions about their real-world applicability. In this work, we empirically investigate this discrepancy by introducing a reconstruction attack based on diffusion models (DMs) that only assumes adversary access to real-world image priors and specifically targets the DP defense. We find that (1) real-world data priors significantly influence reconstruction success, (2) current reconstruction bounds do not model the risk posed by data priors well, and (3) DMs can serve as heuristic auditing tools for visualizing privacy leakage.

摘要: 对机器学习模型的数据重建攻击对隐私构成了实质性威胁，可能会泄露敏感信息。尽管使用差分隐私(DP)防御此类攻击提供了理论上的保证，但确定适当的DP参数仍然具有挑战性。目前对数据重建成功的正式保证受到关于目标数据的敌对知识的过于严格的假设，特别是在图像领域，这引发了对其在现实世界中的适用性的质疑。在这项工作中，我们通过引入一种基于扩散模型(DM)的重建攻击来实证研究这种差异，该攻击只假设对手可以访问真实世界的图像先验，并专门针对DP防御。我们发现(1)真实世界的数据先验显著影响重建成功，(2)当前的重建边界没有很好地模拟数据先验带来的风险，(3)数据挖掘可以作为启发式审计工具来可视化隐私泄露。



## **32. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

一个扰动就足够了：关于针对视觉语言预训练模型生成普遍对抗性扰动 cs.CV

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2406.05491v3) [paper-pdf](http://arxiv.org/pdf/2406.05491v3)

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Hao Wu, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models have exhibited unprecedented capability in many applications by taking full advantage of the multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these methods are generally instance-specific and require generating perturbations for each input sample. In this paper, we reveal that VLP models are also vulnerable to the instance-agnostic universal adversarial perturbation (UAP). Specifically, we design a novel Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC) to achieve the attack. In light that the pivotal multimodal alignment is achieved through the advanced contrastive learning technique, we devise to turn this powerful weapon against themselves, i.e., employ a malicious version of contrastive learning to train the C-PGC based on our carefully crafted positive and negative image-text pairs for essentially destroying the alignment relationship learned by VLP models. Besides, C-PGC fully utilizes the characteristics of Vision-and-Language (V+L) scenarios by incorporating both unimodal and cross-modal information as effective guidance. Extensive experiments show that C-PGC successfully forces adversarial samples to move away from their original area in the VLP model's feature space, thus essentially enhancing attacks across various victim models and V+L tasks. The GitHub repository is available at https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks.

摘要: 视觉语言预训练(VLP)模型充分利用了多通道对齐的优势，在许多应用中表现出了前所未有的能力。然而，之前的研究表明，它们很容易受到恶意制作的对手样本的攻击。尽管最近取得了成功，但这些方法通常是特定于实例的，需要为每个输入样本生成扰动。在这篇文章中，我们揭示了VLP模型也容易受到实例不可知的通用对抗扰动(UAP)的影响。具体地说，我们设计了一种新的具有交叉模式条件的对比训练扰动生成器(C-PGC)来实现攻击。鉴于关键的多通道对齐是通过先进的对比学习技术实现的，我们打算将这一强大的武器转化为针对自己的强大武器，即使用恶意版本的对比学习来训练基于我们精心设计的正和负图文对的C-PGC，以从根本上破坏VLP模型学习的对齐关系。此外，C-PGC充分利用了视觉与语言(V+L)情景的特点，融合了单峰和跨通道信息作为有效的指导。大量实验表明，C-PGC成功地迫使敌方样本在VLP模型的特征空间中离开其原始区域，从而本质上增强了对各种受害者模型和V+L任务的攻击。GitHub存储库可在https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks.上获得



## **33. Long-tailed Adversarial Training with Self-Distillation**

自我蒸馏的长尾对抗训练 cs.CV

ICLR 2025

**SubmitDate**: 2025-03-09    [abs](http://arxiv.org/abs/2503.06461v1) [paper-pdf](http://arxiv.org/pdf/2503.06461v1)

**Authors**: Seungju Cho, Hongsin Lee, Changick Kim

**Abstract**: Adversarial training significantly enhances adversarial robustness, yet superior performance is predominantly achieved on balanced datasets.   Addressing adversarial robustness in the context of unbalanced or long-tailed distributions is considerably more challenging, mainly due to the scarcity of tail data instances.   Previous research on adversarial robustness within long-tailed distributions has primarily focused on combining traditional long-tailed natural training with existing adversarial robustness methods.   In this study, we provide an in-depth analysis for the challenge that adversarial training struggles to achieve high performance on tail classes in long-tailed distributions.   Furthermore, we propose a simple yet effective solution to advance adversarial robustness on long-tailed distributions through a novel self-distillation technique.   Specifically, this approach leverages a balanced self-teacher model, which is trained using a balanced dataset sampled from the original long-tailed dataset. Our extensive experiments demonstrate state-of-the-art performance in both clean and robust accuracy for long-tailed adversarial robustness, with significant improvements in tail class performance on various datasets. We improve the accuracy against PGD attacks for tail classes by 20.3, 7.1, and 3.8 percentage points on CIFAR-10, CIFAR-100, and Tiny-ImageNet, respectively, while achieving the highest robust accuracy.

摘要: 对抗性训练显著增强了对抗性的稳健性，但优异的性能主要是在平衡的数据集上实现的。在不平衡或长尾分布的背景下解决对手健壮性的挑战要大得多，这主要是由于尾部数据实例的稀缺。以往关于长尾分布下的对抗稳健性的研究主要集中在将传统的长尾自然训练与现有的对抗稳健性方法相结合。在这项研究中，我们对对抗性训练在长尾分布的尾类上取得高成绩所面临的挑战进行了深入的分析。此外，我们提出了一种简单而有效的解决方案，通过一种新的自蒸馏技术来提高长尾分布上的敌意稳健性。具体地说，这种方法利用了平衡的自学模型，该模型是使用从原始长尾数据集中采样的平衡数据集进行训练的。我们广泛的实验表明，在长尾对抗健壮性方面，我们在清洁和稳健的准确性方面都表现出了最先进的性能，在各种数据集上的尾类性能都有了显著的改善。我们在CIFAR-10、CIFAR-100和Tiny-ImageNet上分别将尾类对PGD攻击的准确率提高了20.3、7.1和3.8个百分点，同时获得了最高的稳健准确率。



## **34. Adversarial Robustness of Discriminative Self-Supervised Learning in Vision**

视觉中辨别性自我监督学习的对抗鲁棒性 cs.CV

53 pages

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06361v1) [paper-pdf](http://arxiv.org/pdf/2503.06361v1)

**Authors**: Ömer Veysel Çağatan, Ömer Faruk Tal, M. Emre Gürsoy

**Abstract**: Self-supervised learning (SSL) has advanced significantly in visual representation learning, yet comprehensive evaluations of its adversarial robustness remain limited. In this study, we evaluate the adversarial robustness of seven discriminative self-supervised models and one supervised model across diverse tasks, including ImageNet classification, transfer learning, segmentation, and detection. Our findings suggest that discriminative SSL models generally exhibit better robustness to adversarial attacks compared to their supervised counterpart on ImageNet, with this advantage extending to transfer learning when using linear evaluation. However, when fine-tuning is applied, the robustness gap between SSL and supervised models narrows considerably. Similarly, this robustness advantage diminishes in segmentation and detection tasks. We also investigate how various factors might influence adversarial robustness, including architectural choices, training duration, data augmentations, and batch sizes. Our analysis contributes to the ongoing exploration of adversarial robustness in visual self-supervised representation systems.

摘要: 自监督学习(SSL)在视觉表征学习方面取得了显著进展，但对其对抗鲁棒性的综合评价仍然有限。在这项研究中，我们评估了七个判别性自我监督模型和一个监督模型在不同任务中的对抗健壮性，包括ImageNet分类、迁移学习、分割和检测。我们的研究结果表明，与ImageNet上的监督模型相比，鉴别性SSL模型对敌意攻击表现出了更好的稳健性，当使用线性评估时，这一优势扩展到迁移学习。然而，当应用微调时，SSL和监督模型之间的稳健性差距显著缩小。同样，这种稳健性优势在分割和检测任务中会减弱。我们还调查了各种因素如何影响对手的健壮性，包括架构选择、训练持续时间、数据扩充和批次大小。我们的分析有助于继续探索视觉自监督表示系统中的对抗性稳健性。



## **35. Reproducing HotFlip for Corpus Poisoning Attacks in Dense Retrieval**

在密集检索中重现HotFlip以应对Corpus中毒攻击 cs.IR

This paper has been accepted for oral presentation in the  reproducibility track at ECIR 2025

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2501.04802v2) [paper-pdf](http://arxiv.org/pdf/2501.04802v2)

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Evangelos Kanoulas

**Abstract**: HotFlip is a topical gradient-based word substitution method for attacking language models. Recently, this method has been further applied to attack retrieval systems by generating malicious passages that are injected into a corpus, i.e., corpus poisoning. However, HotFlip is known to be computationally inefficient, with the majority of time being spent on gradient accumulation for each query-passage pair during the adversarial token generation phase, making it impossible to generate an adequate number of adversarial passages in a reasonable amount of time. Moreover, the attack method itself assumes access to a set of user queries, a strong assumption that does not correspond to how real-world adversarial attacks are usually performed. In this paper, we first significantly boost the efficiency of HotFlip, reducing the adversarial generation process from 4 hours per document to only 15 minutes, using the same hardware. We further contribute experiments and analysis on two additional tasks: (1) transfer-based black-box attacks, and (2) query-agnostic attacks. Whenever possible, we provide comparisons between the original method and our improved version. Our experiments demonstrate that HotFlip can effectively attack a variety of dense retrievers, with an observed trend that its attack performance diminishes against more advanced and recent methods. Interestingly, we observe that while HotFlip performs poorly in a black-box setting, indicating limited capacity for generalization, in query-agnostic scenarios its performance is correlated to the volume of injected adversarial passages.

摘要: HotFlip是一种基于主题梯度的单词替换方法，用于攻击语言模型。最近，这种方法被进一步应用于通过生成注入到语料库中的恶意段落来攻击检索系统，即语料库中毒。然而，HotFlip的计算效率很低，在对抗性令牌生成阶段，大部分时间花费在每个查询通道对的梯度累加上，使得在合理的时间内生成足够数量的对抗性通道是不可能的。此外，攻击方法本身假设可以访问一组用户查询，这是一个强烈的假设，与现实世界中通常如何执行对抗性攻击并不相符。在本文中，我们首先显著提高了HotFlip的效率，在使用相同硬件的情况下，将敌意生成过程从每个文档4小时减少到仅15分钟。我们进一步对两个额外的任务进行了实验和分析：(1)基于传输的黑盒攻击；(2)查询无关攻击。只要有可能，我们就会提供原始方法和改进后的方法之间的比较。我们的实验表明，HotFlip可以有效地攻击各种密集检索犬，并观察到其攻击性能与更先进和最新的方法相比有所下降的趋势。有趣的是，我们观察到，虽然HotFlip在黑盒环境中表现不佳，表明泛化能力有限，但在查询不可知的场景中，它的性能与注入的敌意段落的数量相关。



## **36. IDEATOR: Jailbreaking and Benchmarking Large Vision-Language Models Using Themselves**

IDEATOR：使用自己越狱和基准大型视觉语言模型 cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2411.00827v3) [paper-pdf](http://arxiv.org/pdf/2411.00827v3)

**Authors**: Ruofan Wang, Juncheng Li, Yixu Wang, Bo Wang, Xiaosen Wang, Yan Teng, Yingchun Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: As large Vision-Language Models (VLMs) gain prominence, ensuring their safe deployment has become critical. Recent studies have explored VLM robustness against jailbreak attacks-techniques that exploit model vulnerabilities to elicit harmful outputs. However, the limited availability of diverse multimodal data has constrained current approaches to rely heavily on adversarial or manually crafted images derived from harmful text datasets, which often lack effectiveness and diversity across different contexts. In this paper, we propose IDEATOR, a novel jailbreak method that autonomously generates malicious image-text pairs for black-box jailbreak attacks. IDEATOR is grounded in the insight that VLMs themselves could serve as powerful red team models for generating multimodal jailbreak prompts. Specifically, IDEATOR leverages a VLM to create targeted jailbreak texts and pairs them with jailbreak images generated by a state-of-the-art diffusion model. Extensive experiments demonstrate IDEATOR's high effectiveness and transferability, achieving a 94% attack success rate (ASR) in jailbreaking MiniGPT-4 with an average of only 5.34 queries, and high ASRs of 82%, 88%, and 75% when transferred to LLaVA, InstructBLIP, and Chameleon, respectively. Building on IDEATOR's strong transferability and automated process, we introduce the VLBreakBench, a safety benchmark comprising 3,654 multimodal jailbreak samples. Our benchmark results on 11 recently released VLMs reveal significant gaps in safety alignment. For instance, our challenge set achieves ASRs of 46.31% on GPT-4o and 19.65% on Claude-3.5-Sonnet, underscoring the urgent need for stronger defenses.

摘要: 随着大型视觉语言模型(VLM)的日益突出，确保它们的安全部署变得至关重要。最近的研究探索了VLM对越狱攻击的稳健性--利用模型漏洞来引发有害输出的技术。然而，各种多模式数据的可获得性有限，限制了目前的方法严重依赖从有害文本数据集获得的对抗性或手动制作的图像，这些图像往往缺乏跨不同背景的有效性和多样性。在本文中，我们提出了一种新的越狱方法--IDEATOR，它能够自动生成用于黑盒越狱攻击的恶意图文对。Ideator基于这样的见解，即VLM本身可以作为生成多模式越狱提示的强大红色团队模型。具体地说，Ideator利用VLM创建有针对性的越狱文本，并将它们与由最先进的扩散模型生成的越狱图像配对。大量的实验证明了IDAIDATER的高效率和可移植性，在越狱MiniGPT-4上达到了94%的攻击成功率(ASR)，平均只有5.34个查询，当转移到LLaVA、InstructBLIP和Chameleon时，ASR分别达到了82%、88%和75%。基于IDEATOR强大的可转移性和自动化流程，我们引入了VLBreakB边，这是一个包含3654个多模式越狱样本的安全基准。我们在最近发布的11个VLM上的基准结果显示，在安全对准方面存在显著差距。例如，我们的挑战集在GPT-40上达到了46.31%的ASR，在Claude-3.5-十四行诗上达到了19.65%，这突显了对更强大防御的迫切需要。



## **37. Exploring Adversarial Transferability between Kolmogorov-arnold Networks**

探索Kolmogorov-Arnold网络之间的对抗可移植性 cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06276v1) [paper-pdf](http://arxiv.org/pdf/2503.06276v1)

**Authors**: Songping Wang, Xinquan Yue, Yueming Lyu, Caifeng Shan

**Abstract**: Kolmogorov-Arnold Networks (KANs) have emerged as a transformative model paradigm, significantly impacting various fields. However, their adversarial robustness remains less underexplored, especially across different KAN architectures. To explore this critical safety issue, we conduct an analysis and find that due to overfitting to the specific basis functions of KANs, they possess poor adversarial transferability among different KANs. To tackle this challenge, we propose AdvKAN, the first transfer attack method for KANs. AdvKAN integrates two key components: 1) a Breakthrough-Defense Surrogate Model (BDSM), which employs a breakthrough-defense training strategy to mitigate overfitting to the specific structures of KANs. 2) a Global-Local Interaction (GLI) technique, which promotes sufficient interaction between adversarial gradients of hierarchical levels, further smoothing out loss surfaces of KANs. Both of them work together to enhance the strength of transfer attack among different KANs. Extensive experimental results on various KANs and datasets demonstrate the effectiveness of AdvKAN, which possesses notably superior attack capabilities and deeply reveals the vulnerabilities of KANs. Code will be released upon acceptance.

摘要: Kolmogorov-Arnold Networks(KANS)已成为一种变革性的模型范式，对各个领域产生了重大影响。然而，它们的对抗性健壮性仍然没有得到很好的开发，特别是在不同的KAN架构上。为了探讨这一关键的安全问题，我们进行了分析，发现由于对KANS的特定基函数过度拟合，它们在不同KANS之间具有较差的对抗性可转移性。为了应对这一挑战，我们提出了第一种针对KANS的传输攻击方法--AdvKAN。AdvKAN集成了两个关键组件：1)突破-防御代理模型(BDSM)，该模型采用突破-防御训练策略来缓解对KANS特定结构的过度适应。2)全局-局部交互(Global-Local Interaction，GLI)技术，它促进了层级之间的对抗性梯度之间的充分交互，进一步平滑了KANS的损失曲面。两者共同努力，增强了不同阵容之间的转移进攻实力。在不同的KANS和数据集上的大量实验结果证明了AdvKAN的有效性，它具有明显优越的攻击能力，并深刻揭示了KANS的脆弱性。代码将在接受后发布。



## **38. Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models**

使用机械可解释性来应对大型语言模型的对抗攻击 cs.LG

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06269v1) [paper-pdf](http://arxiv.org/pdf/2503.06269v1)

**Authors**: Thomas Winninger, Boussad Addad, Katarzyna Kapusta

**Abstract**: Traditional white-box methods for creating adversarial perturbations against LLMs typically rely only on gradient computation from the targeted model, ignoring the internal mechanisms responsible for attack success or failure. Conversely, interpretability studies that analyze these internal mechanisms lack practical applications beyond runtime interventions. We bridge this gap by introducing a novel white-box approach that leverages mechanistic interpretability techniques to craft practical adversarial inputs. Specifically, we first identify acceptance subspaces - sets of feature vectors that do not trigger the model's refusal mechanisms - then use gradient-based optimization to reroute embeddings from refusal subspaces to acceptance subspaces, effectively achieving jailbreaks. This targeted approach significantly reduces computation cost, achieving attack success rates of 80-95\% on state-of-the-art models including Gemma2, Llama3.2, and Qwen2.5 within minutes or even seconds, compared to existing techniques that often fail or require hours of computation. We believe this approach opens a new direction for both attack research and defense development. Furthermore, it showcases a practical application of mechanistic interpretability where other methods are less efficient, which highlights its utility. The code and generated datasets are available at https://github.com/Sckathach/subspace-rerouting.

摘要: 传统的白盒方法用于创建针对LLM的对抗性扰动，通常只依赖于目标模型的梯度计算，而忽略了攻击成败的内部机制。相反，分析这些内部机制的可解释性研究缺乏运行时干预之外的实际应用。我们通过引入一种新的白盒方法来弥合这一差距，该方法利用机械性的可解释性技术来制作实际的对抗性输入。具体地说，我们首先识别接受子空间-不会触发模型的拒绝机制的特征向量集合-然后使用基于梯度的优化将嵌入从拒绝子空间重定向到接受子空间，从而有效地实现越狱。这种有针对性的方法显著降低了计算成本，与通常失败或需要数小时计算的现有技术相比，在Gemma2、Llama3.2和Qwen2.5等最先进的模型上在几分钟甚至几秒钟内实现了80%-95%的攻击成功率。我们相信，这种方法为攻击研究和防御发展开辟了新的方向。此外，它还展示了机械可解释性在其他方法效率较低的情况下的实际应用，这突出了它的实用性。代码和生成的数据集可在https://github.com/Sckathach/subspace-rerouting.上获得



## **39. MUNBa: Machine Unlearning via Nash Bargaining**

MUNba：通过纳什讨价还价的机器学习 cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2411.15537v2) [paper-pdf](http://arxiv.org/pdf/2411.15537v2)

**Authors**: Jing Wu, Mehrtash Harandi

**Abstract**: Machine Unlearning (MU) aims to selectively erase harmful behaviors from models while retaining the overall utility of the model. As a multi-task learning problem, MU involves balancing objectives related to forgetting specific concepts/data and preserving general performance. A naive integration of these forgetting and preserving objectives can lead to gradient conflicts and dominance, impeding MU algorithms from reaching optimal solutions. To address the gradient conflict and dominance issue, we reformulate MU as a two-player cooperative game, where the two players, namely, the forgetting player and the preservation player, contribute via their gradient proposals to maximize their overall gain and balance their contributions. To this end, inspired by the Nash bargaining theory, we derive a closed-form solution to guide the model toward the Pareto stationary point. Our formulation of MU guarantees an equilibrium solution, where any deviation from the final state would lead to a reduction in the overall objectives for both players, ensuring optimality in each objective. We evaluate our algorithm's effectiveness on a diverse set of tasks across image classification and image generation. Extensive experiments with ResNet, vision-language model CLIP, and text-to-image diffusion models demonstrate that our method outperforms state-of-the-art MU algorithms, achieving a better trade-off between forgetting and preserving. Our results also highlight improvements in forgetting precision, preservation of generalization, and robustness against adversarial attacks.

摘要: 机器遗忘旨在选择性地消除模型中的有害行为，同时保持模型的整体效用。作为一个多任务学习问题，MU涉及到平衡与忘记特定概念/数据相关的目标和保持总体性能。这些遗忘和保留目标的天真集成可能会导致梯度冲突和优势，阻碍MU算法获得最优解。为了解决梯度冲突和优势问题，我们将MU重新描述为一个两人合作博弈，其中两个参与者，即遗忘者和保留者，通过他们的梯度方案做出贡献，以最大化他们的整体收益，平衡他们的贡献。为此，在纳什讨价还价理论的启发下，我们推导出了一个闭合形式的解，将模型引向帕累托稳定点。我们的MU公式保证了一个均衡的解决方案，其中任何与最终状态的偏离都将导致两个球员的总体目标的减少，确保每个目标的最优化。我们评估了我们的算法在不同的任务集上的有效性，包括图像分类和图像生成。在ResNet、视觉语言模型CLIP和文本到图像扩散模型上的大量实验表明，该方法的性能优于最先进的MU算法，在遗忘和保存之间实现了更好的权衡。我们的结果还强调了在忘记精确度、保持泛化和对对手攻击的健壮性方面的改进。



## **40. Boosting the Local Invariance for Better Adversarial Transferability**

增强局部不变性以获得更好的对抗可移植性 cs.CV

**SubmitDate**: 2025-03-08    [abs](http://arxiv.org/abs/2503.06140v1) [paper-pdf](http://arxiv.org/pdf/2503.06140v1)

**Authors**: Bohan Liu, Xiaosen Wang

**Abstract**: Transfer-based attacks pose a significant threat to real-world applications by directly targeting victim models with adversarial examples generated on surrogate models. While numerous approaches have been proposed to enhance adversarial transferability, existing works often overlook the intrinsic relationship between adversarial perturbations and input images. In this work, we find that adversarial perturbation often exhibits poor translation invariance for a given clean image and model, which is attributed to local invariance. Through empirical analysis, we demonstrate that there is a positive correlation between the local invariance of adversarial perturbations w.r.t. the input image and their transferability across different models. Based on this finding, we propose a general adversarial transferability boosting technique called Local Invariance Boosting approach (LI-Boost). Extensive experiments on the standard ImageNet dataset demonstrate that LI-Boost could significantly boost various types of transfer-based attacks (e.g., gradient-based, input transformation-based, model-related, advanced objective function, ensemble, etc.) on CNNs, ViTs, and defense mechanisms. Our approach presents a promising direction for future research in improving adversarial transferability across different models.

摘要: 基于传输的攻击直接以受害者模型为目标，在代理模型上生成对抗性示例，从而对现实世界的应用程序构成重大威胁。虽然已经提出了许多方法来增强对抗性转移，但现有的工作往往忽略了对抗性扰动与输入图像之间的内在联系。在这项工作中，我们发现对于给定的干净图像和模型，对抗性扰动通常表现出较差的平移不变性，这归因于局部不变性。通过实证分析，我们证明了对抗性扰动的局部不变性与w.r.t.输入图像及其跨不同模型的可转移性。基于这一发现，我们提出了一种通用的对抗性可转移性增强技术，称为局部不变增强方法(LI-Boost)。在标准ImageNet数据集上的大量实验表明，LI-Boost能够显著增强各种类型的基于传输的攻击(例如，基于梯度、基于输入变换、与模型相关、高级目标函数、集成等)。关于CNN、VITS和防御机制。我们的方法为未来在提高不同模型之间的对抗性转移方面的研究提供了一个很有前途的方向。



## **41. Continual Adversarial Defense**

持续对抗防御 cs.CV

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2312.09481v5) [paper-pdf](http://arxiv.org/pdf/2312.09481v5)

**Authors**: Qian Wang, Hefei Ling, Yingwei Li, Qihao Liu, Ruoxi Jia, Ning Yu

**Abstract**: In response to the rapidly evolving nature of adversarial attacks against visual classifiers on a monthly basis, numerous defenses have been proposed to generalize against as many known attacks as possible. However, designing a defense method that generalizes to all types of attacks is not realistic because the environment in which defense systems operate is dynamic and comprises various unique attacks that emerge as time goes on. A well-matched approach to the dynamic environment lies in a defense system that continuously collects adversarial data online to quickly improve itself. Therefore, we put forward a practical defense deployment against a challenging threat model and propose, for the first time, the Continual Adversarial Defense (CAD) framework that adapts to attack sequences under four principles: (1)~continual adaptation to new attacks without catastrophic forgetting, (2)~few-shot adaptation, (3)~memory-efficient adaptation, and (4)~high accuracy on both clean and adversarial data. We explore and integrate cutting-edge continual learning, few-shot learning, and ensemble learning techniques to qualify the principles. Extensive experiments validate the effectiveness of our approach against multiple stages of modern adversarial attacks and demonstrate significant improvements over numerous baseline methods. In particular, CAD is capable of quickly adapting with minimal budget and a low cost of defense failure while maintaining good performance against previous attacks. Our research sheds light on a brand-new paradigm for continual defense adaptation against dynamic and evolving attacks.

摘要: 为了应对每月针对视觉分类器的对抗性攻击迅速演变的性质，提出了许多防御措施，以概括尽可能多的已知攻击。然而，设计一种概括所有类型攻击的防御方法是不现实的，因为防御系统运行的环境是动态的，包括随着时间的推移而出现的各种独特的攻击。一种与动态环境相匹配的方法在于一个防御系统，该系统不断在线收集敌对数据，以快速改进自己。因此，我们提出了一种针对具有挑战性的威胁模型的实用防御部署，并首次提出了适应攻击序列的持续对抗防御(CAD)框架，该框架遵循四个原则：(1)持续适应新的攻击而不会灾难性地忘记；(2)少射击适应；(3)内存高效适应；(4)对干净和敌对数据的高准确度。我们探索并集成了尖端的持续学习、少机会学习和集成学习技术来验证这些原则。广泛的实验验证了我们的方法对现代对抗性攻击的多个阶段的有效性，并证明了与许多基线方法相比有显著的改进。特别是，CAD能够以最小的预算和较低的防御失败成本快速适应，同时保持对先前攻击的良好性能。我们的研究揭示了一种针对动态和不断变化的攻击进行持续防御适应的全新范式。



## **42. Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis**

注意差距：通过查询更新分析检测正在形成的黑匣子对抗攻击 cs.CR

13 pages

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.02986v2) [paper-pdf](http://arxiv.org/pdf/2503.02986v2)

**Authors**: Jeonghwan Park, Niall McLaughlin, Ihsen Alouani

**Abstract**: Adversarial attacks remain a significant threat that can jeopardize the integrity of Machine Learning (ML) models. In particular, query-based black-box attacks can generate malicious noise without having access to the victim model's architecture, making them practical in real-world contexts. The community has proposed several defenses against adversarial attacks, only to be broken by more advanced and adaptive attack strategies. In this paper, we propose a framework that detects if an adversarial noise instance is being generated. Unlike existing stateful defenses that detect adversarial noise generation by monitoring the input space, our approach learns adversarial patterns in the input update similarity space. In fact, we propose to observe a new metric called Delta Similarity (DS), which we show it captures more efficiently the adversarial behavior. We evaluate our approach against 8 state-of-the-art attacks, including adaptive attacks, where the adversary is aware of the defense and tries to evade detection. We find that our approach is significantly more robust than existing defenses both in terms of specificity and sensitivity.

摘要: 对抗性攻击仍然是一个严重的威胁，可能会危及机器学习(ML)模型的完整性。特别是，基于查询的黑盒攻击可以在不访问受害者模型的体系结构的情况下生成恶意噪声，使它们在真实世界的上下文中具有实用性。社区已经提出了几种针对对抗性攻击的防御措施，但都被更先进和适应性更强的攻击策略打破了。在这篇文章中，我们提出了一个框架，它检测是否正在生成对抗性噪声实例。与现有的通过监测输入空间来检测对抗性噪声产生的状态防御方法不同，我们的方法在输入更新相似性空间中学习对抗性模式。事实上，我们提出了一种新的度量，称为Delta相似度(DS)，我们表明它更有效地捕获了对手的行为。我们评估了我们的方法针对8种最先进的攻击，包括自适应攻击，在这些攻击中，对手知道防御并试图逃避检测。我们发现，我们的方法在特异性和敏感性方面都明显比现有的防御方法更稳健。



## **43. Benchmarking Vision Language Model Unlearning via Fictitious Facial Identity Dataset**

通过虚构面部身份数据集对视觉语言模型取消学习进行基准测试 cs.CV

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2411.03554v3) [paper-pdf](http://arxiv.org/pdf/2411.03554v3)

**Authors**: Yingzi Ma, Jiongxiao Wang, Fei Wang, Siyuan Ma, Jiazhao Li, Jinsheng Pan, Xiujun Li, Furong Huang, Lichao Sun, Bo Li, Yejin Choi, Muhao Chen, Chaowei Xiao

**Abstract**: Machine unlearning has emerged as an effective strategy for forgetting specific information in the training data. However, with the increasing integration of visual data, privacy concerns in Vision Language Models (VLMs) remain underexplored. To address this, we introduce Facial Identity Unlearning Benchmark (FIUBench), a novel VLM unlearning benchmark designed to robustly evaluate the effectiveness of unlearning algorithms under the Right to be Forgotten setting. Specifically, we formulate the VLM unlearning task via constructing the Fictitious Facial Identity VQA dataset and apply a two-stage evaluation pipeline that is designed to precisely control the sources of information and their exposure levels. In terms of evaluation, since VLM supports various forms of ways to ask questions with the same semantic meaning, we also provide robust evaluation metrics including membership inference attacks and carefully designed adversarial privacy attacks to evaluate the performance of algorithms. Through the evaluation of four baseline VLM unlearning algorithms within FIUBench, we find that all methods remain limited in their unlearning performance, with significant trade-offs between model utility and forget quality. Furthermore, our findings also highlight the importance of privacy attacks for robust evaluations. We hope FIUBench will drive progress in developing more effective VLM unlearning algorithms.

摘要: 机器遗忘已经成为一种遗忘训练数据中特定信息的有效策略。然而，随着视觉数据的日益集成，视觉语言模型(VLM)中的隐私问题仍然没有得到充分的研究。为了解决这个问题，我们引入了面部身份遗忘基准(FIUB边)，这是一个新的VLM遗忘基准，设计用于在被遗忘的权利设置下稳健地评估遗忘算法的有效性。具体地说，我们通过构建虚拟面部身份VQA数据集来制定VLM遗忘任务，并应用旨在精确控制信息源及其暴露水平的两阶段评估管道。在评估方面，由于VLM支持多种形式的具有相同语义的问题，我们还提供了健壮的评估指标，包括成员关系推理攻击和精心设计的对抗性隐私攻击来评估算法的性能。通过对FIUBuch四种基线VLM遗忘算法的评估，我们发现所有方法的遗忘性能都是有限的，模型效用和遗忘质量之间存在着显著的权衡。此外，我们的发现还强调了隐私攻击对于稳健评估的重要性。我们希望FIUB边将推动在开发更有效的VLM遗忘算法方面取得进展。



## **44. Toward Robust Non-Transferable Learning: A Survey and Benchmark**

迈向稳健的不可转移学习：调查和基准 cs.LG

Code is available at https://github.com/tmllab/NTLBench

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2502.13593v2) [paper-pdf](http://arxiv.org/pdf/2502.13593v2)

**Authors**: Ziming Hong, Yongli Xiang, Tongliang Liu

**Abstract**: Over the past decades, researchers have primarily focused on improving the generalization abilities of models, with limited attention given to regulating such generalization. However, the ability of models to generalize to unintended data (e.g., harmful or unauthorized data) can be exploited by malicious adversaries in unforeseen ways, potentially resulting in violations of model ethics. Non-transferable learning (NTL), a task aimed at reshaping the generalization abilities of deep learning models, was proposed to address these challenges. While numerous methods have been proposed in this field, a comprehensive review of existing progress and a thorough analysis of current limitations remain lacking. In this paper, we bridge this gap by presenting the first comprehensive survey on NTL and introducing NTLBench, the first benchmark to evaluate NTL performance and robustness within a unified framework. Specifically, we first introduce the task settings, general framework, and criteria of NTL, followed by a summary of NTL approaches. Furthermore, we emphasize the often-overlooked issue of robustness against various attacks that can destroy the non-transferable mechanism established by NTL. Experiments conducted via NTLBench verify the limitations of existing NTL methods in robustness. Finally, we discuss the practical applications of NTL, along with its future directions and associated challenges.

摘要: 在过去的几十年里，研究人员主要专注于提高模型的泛化能力，而对规范这种泛化的关注很少。然而，模型概括为非预期数据(例如，有害或未经授权的数据)的能力可能会被恶意攻击者以不可预见的方式利用，可能导致违反模型道德。为应对这些挑战，提出了一项旨在重塑深度学习模型泛化能力的任务--不可迁移学习(NTL)。虽然在这一领域提出了许多方法，但仍然缺乏对现有进展的全面审查和对当前限制的透彻分析。在本文中，我们通过介绍第一次关于NTL的全面调查并引入NTLBitch来弥合这一差距，NTLBuchch是第一个在统一框架内评估NTL性能和健壮性的基准。具体地说，我们首先介绍了网络学习的任务设置、总体框架和标准，然后对网络学习的方法进行了概述。此外，我们强调了经常被忽视的问题，即对各种攻击的健壮性，这些攻击可以破坏NTL建立的不可转移机制。通过NTLBitch进行的实验验证了现有NTL方法在稳健性方面的局限性。最后，我们讨论了NTL的实际应用，以及它的未来方向和相关的挑战。



## **45. Robust Intrusion Detection System with Explainable Artificial Intelligence**

具有可解释人工智能的稳健入侵检测系统 cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05303v1) [paper-pdf](http://arxiv.org/pdf/2503.05303v1)

**Authors**: Betül Güvenç Paltun, Ramin Fuladi, Rim El Malki

**Abstract**: Machine learning (ML) models serve as powerful tools for threat detection and mitigation; however, they also introduce potential new risks. Adversarial input can exploit these models through standard interfaces, thus creating new attack pathways that threaten critical network operations. As ML advancements progress, adversarial strategies become more advanced, and conventional defenses such as adversarial training are costly in computational terms and often fail to provide real-time detection. These methods typically require a balance between robustness and model performance, which presents challenges for applications that demand instant response. To further investigate this vulnerability, we suggest a novel strategy for detecting and mitigating adversarial attacks using eXplainable Artificial Intelligence (XAI). This approach is evaluated in real time within intrusion detection systems (IDS), leading to the development of a zero-touch mitigation strategy. Additionally, we explore various scenarios in the Radio Resource Control (RRC) layer within the Open Radio Access Network (O-RAN) framework, emphasizing the critical need for enhanced mitigation techniques to strengthen IDS defenses against advanced threats and implement a zero-touch mitigation solution. Extensive testing across different scenarios in the RRC layer of the O-RAN infrastructure validates the ability of the framework to detect and counteract integrated RRC-layer attacks when paired with adversarial strategies, emphasizing the essential need for robust defensive mechanisms to strengthen IDS against complex threats.

摘要: 机器学习(ML)模型是检测和缓解威胁的强大工具，但它们也带来了潜在的新风险。敌意输入可以通过标准接口利用这些模型，从而创建威胁关键网络操作的新攻击路径。随着ML的进步，对抗性策略变得更加先进，而传统的防御方法，如对抗性训练，在计算方面代价高昂，而且往往无法提供实时检测。这些方法通常需要在稳健性和模型性能之间取得平衡，这给需要即时响应的应用程序带来了挑战。为了进一步研究这一漏洞，我们提出了一种使用可解释人工智能(XAI)检测和缓解对手攻击的新策略。这种方法在入侵检测系统(IDS)中进行实时评估，从而开发出一种零接触缓解策略。此外，我们还探讨了开放式无线接入网络(O-RAN)框架内无线资源控制(RRC)层的各种场景，强调了增强缓解技术的迫切需要，以加强入侵检测系统对高级威胁的防御并实施零接触缓解解决方案。在O-RAN基础设施的RRC层的不同场景中进行的广泛测试验证了该框架在与对抗性策略配合使用时检测和对抗集成RRC层攻击的能力，强调了加强入侵检测系统对抗复杂威胁的强大防御机制的基本必要性。



## **46. Jailbreaking is (Mostly) Simpler Than You Think**

越狱（大多数）比你想象的要简单 cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05264v1) [paper-pdf](http://arxiv.org/pdf/2503.05264v1)

**Authors**: Mark Russinovich, Ahmed Salem

**Abstract**: We introduce the Context Compliance Attack (CCA), a novel, optimization-free method for bypassing AI safety mechanisms. Unlike current approaches -- which rely on complex prompt engineering and computationally intensive optimization -- CCA exploits a fundamental architectural vulnerability inherent in many deployed AI systems. By subtly manipulating conversation history, CCA convinces the model to comply with a fabricated dialogue context, thereby triggering restricted behavior. Our evaluation across a diverse set of open-source and proprietary models demonstrates that this simple attack can circumvent state-of-the-art safety protocols. We discuss the implications of these findings and propose practical mitigation strategies to fortify AI systems against such elementary yet effective adversarial tactics.

摘要: 我们引入了上下文合规攻击（PCA），这是一种新颖的、无需优化的方法，用于绕过人工智能安全机制。与当前的方法（依赖于复杂的即时工程和计算密集型优化）不同，CAA利用了许多已部署的人工智能系统固有的基本架构漏洞。通过巧妙地操纵对话历史，CAA说服模型遵守捏造的对话上下文，从而触发受限制的行为。我们对各种开源和专有模型的评估表明，这种简单的攻击可以规避最先进的安全协议。我们讨论了这些发现的影响，并提出了实用的缓解策略，以加强人工智能系统对抗这种基本但有效的对抗策略。



## **47. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

DetectRL：在现实世界场景中对LLM生成的文本检测进行基准测试 cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2410.23746v2) [paper-pdf](http://arxiv.org/pdf/2410.23746v2)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.

摘要: 检测由大型语言模型(LLM)生成的文本是最近非常感兴趣的问题。有了像DetectGPT这样的零射击方法，检测能力已经达到了令人印象深刻的水平。然而，现有探测器在实际应用中的可靠性仍然没有得到充分的探索。在这项研究中，我们提出了一个新的基准，DetectRL，强调即使是最先进的(SOTA)检测技术在这项任务中仍然表现不佳。我们从LLM特别容易被滥用的领域收集了人类编写的数据集。使用流行的LLM，我们生成的数据更好地与现实世界的应用程序保持一致。与以前的研究不同，我们使用启发式规则来创建对抗性LLM生成的文本，模拟各种提示用法、人工修改(如单词替换)和书写噪音(如拼写错误)。我们对DetectRL的开发揭示了当前SOTA探测器的优势和局限性。更重要的是，我们分析了写作风格、模型类型、攻击方法、文本长度和真实世界中的人类写作因素对不同类型检测器的潜在影响。我们相信，DetectRL可以作为评估真实世界场景中检测器的有效基准，随着先进攻击方法的发展，从而提供更有压力的评估，以推动更高效检测器的开发。数据和代码可在以下网址公开获得：https://github.com/NLP2CT/DetectRL.



## **48. Safety-Critical Traffic Simulation with Adversarial Transfer of Driving Intentions**

具有驾驶意图对抗性转移的安全关键交通模拟 cs.RO

Accepted by ICRA 2025

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2503.05180v1) [paper-pdf](http://arxiv.org/pdf/2503.05180v1)

**Authors**: Zherui Huang, Xing Gao, Guanjie Zheng, Licheng Wen, Xuemeng Yang, Xiao Sun

**Abstract**: Traffic simulation, complementing real-world data with a long-tail distribution, allows for effective evaluation and enhancement of the ability of autonomous vehicles to handle accident-prone scenarios. Simulating such safety-critical scenarios is nontrivial, however, from log data that are typically regular scenarios, especially in consideration of dynamic adversarial interactions between the future motions of autonomous vehicles and surrounding traffic participants. To address it, this paper proposes an innovative and efficient strategy, termed IntSim, that explicitly decouples the driving intentions of surrounding actors from their motion planning for realistic and efficient safety-critical simulation. We formulate the adversarial transfer of driving intention as an optimization problem, facilitating extensive exploration of diverse attack behaviors and efficient solution convergence. Simultaneously, intention-conditioned motion planning benefits from powerful deep models and large-scale real-world data, permitting the simulation of realistic motion behaviors for actors. Specially, through adapting driving intentions based on environments, IntSim facilitates the flexible realization of dynamic adversarial interactions with autonomous vehicles. Finally, extensive open-loop and closed-loop experiments on real-world datasets, including nuScenes and Waymo, demonstrate that the proposed IntSim achieves state-of-the-art performance in simulating realistic safety-critical scenarios and further improves planners in handling such scenarios.

摘要: 交通模拟用长尾分布补充了真实世界的数据，允许有效地评估和增强自动驾驶车辆处理事故多发场景的能力。然而，从通常是常规场景的日志数据中模拟这样的安全关键场景并不容易，特别是考虑到自动驾驶车辆未来的运动与周围交通参与者之间的动态对抗性交互。为了解决这一问题，本文提出了一种创新而高效的策略，称为IntSim，该策略明确地将周围参与者的驾驶意图与他们的运动规划解耦，以实现逼真和高效的安全关键模拟。我们将驾驶意图的对抗性转移描述为一个优化问题，便于广泛探索不同的攻击行为和高效的解收敛。同时，意图约束运动规划得益于强大的深层模型和大规模的真实世界数据，允许模拟演员的真实运动行为。特别是，通过基于环境自适应驾驶意图，IntSim有助于灵活实现与自动驾驶车辆的动态对抗性交互。最后，在包括nuScenes和Waymo在内的真实数据集上进行了大量的开环和闭环实验，结果表明，IntSim在模拟现实安全关键场景方面达到了最先进的性能，并进一步提高了规划者处理此类场景的能力。



## **49. Double Backdoored: Converting Code Large Language Model Backdoors to Traditional Malware via Adversarial Instruction Tuning Attacks**

双重后门：通过对抗性指令调优攻击将代码大型语言模型后门转换为传统恶意软件 cs.CR

**SubmitDate**: 2025-03-07    [abs](http://arxiv.org/abs/2404.18567v2) [paper-pdf](http://arxiv.org/pdf/2404.18567v2)

**Authors**: Md Imran Hossen, Sai Venkatesh Chilukoti, Liqun Shan, Sheng Chen, Yinzhi Cao, Xiali Hei

**Abstract**: Instruction-tuned Large Language Models designed for coding tasks are increasingly employed as AI coding assistants. However, the cybersecurity vulnerabilities and implications arising from the widespread integration of these models are not yet fully understood due to limited research in this domain. This work investigates novel techniques for transitioning backdoors from the AI/ML domain to traditional computer malware, shedding light on the critical intersection of AI and cyber/software security. To explore this intersection, we present MalInstructCoder, a framework designed to comprehensively assess the cybersecurity vulnerabilities of instruction-tuned Code LLMs. MalInstructCoder introduces an automated data poisoning pipeline to inject malicious code snippets into benign code, poisoning instruction fine-tuning data while maintaining functional validity. It presents two practical adversarial instruction tuning attacks with real-world security implications: the clean prompt poisoning attack and the backdoor attack. These attacks aim to manipulate Code LLMs to generate code incorporating malicious or harmful functionality under specific attack scenarios while preserving intended functionality. We conduct a comprehensive investigation into the exploitability of the code-specific instruction tuning process involving three state-of-the-art Code LLMs: CodeLlama, DeepSeek-Coder, and StarCoder2. Our findings reveal that these models are highly vulnerable to our attacks. Specifically, the clean prompt poisoning attack achieves the ASR@1 ranging from over 75% to 86% by poisoning only 1% (162 samples) of the instruction fine-tuning dataset. Similarly, the backdoor attack achieves the ASR@1 ranging from 76% to 86% with a 0.5% poisoning rate. Our study sheds light on the critical cybersecurity risks posed by instruction-tuned Code LLMs and highlights the urgent need for robust defense mechanisms.

摘要: 为编码任务而设计的指令调整的大型语言模型越来越多地被用作人工智能编码助手。然而，由于这一领域的研究有限，这些模型的广泛集成所产生的网络安全漏洞和影响尚未完全了解。这项工作研究了将后门从AI/ML领域过渡到传统计算机恶意软件的新技术，揭示了人工智能和网络/软件安全的关键交集。为了探索这一交叉，我们提出了MalInstructCoder，一个旨在全面评估指令调优代码LLM的网络安全漏洞的框架。MalInstructCoder引入了自动数据中毒管道，将恶意代码片段注入良性代码，在保持功能有效性的同时毒化指令微调数据。它提出了两种具有实际安全含义的对抗性指令调整攻击：干净的即时中毒攻击和后门攻击。这些攻击旨在操纵Code LLM在特定攻击场景下生成包含恶意或有害功能的代码，同时保留预期功能。我们对代码特定指令调优过程的可利用性进行了全面的调查，涉及三个最先进的代码LLM：CodeLlama、DeepSeek-Coder和StarCoder2。我们的发现表明，这些模型非常容易受到我们的攻击。具体地说，干净的即时中毒攻击通过仅对指令微调数据集的1%(162个样本)下毒来实现从超过75%到86%的ASR@1。同样，后门攻击实现了从76%到86%的ASR@1，投毒率为0.5%。我们的研究揭示了指令调整的代码LLM构成的关键网络安全风险，并强调了对强大防御机制的迫切需要。



## **50. Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety**

安全不仅仅是拒绝：推理增强微调，以实现可解释的LLM安全 cs.CL

**SubmitDate**: 2025-03-06    [abs](http://arxiv.org/abs/2503.05021v1) [paper-pdf](http://arxiv.org/pdf/2503.05021v1)

**Authors**: Yuyou Zhang, Miao Li, William Han, Yihang Yao, Zhepeng Cen, Ding Zhao

**Abstract**: Large Language Models (LLMs) are vulnerable to jailbreak attacks that exploit weaknesses in traditional safety alignment, which often relies on rigid refusal heuristics or representation engineering to block harmful outputs. While they are effective for direct adversarial attacks, they fall short of broader safety challenges requiring nuanced, context-aware decision-making. To address this, we propose Reasoning-enhanced Finetuning for interpretable LLM Safety (Rational), a novel framework that trains models to engage in explicit safe reasoning before response. Fine-tuned models leverage the extensive pretraining knowledge in self-generated reasoning to bootstrap their own safety through structured reasoning, internalizing context-sensitive decision-making. Our findings suggest that safety extends beyond refusal, requiring context awareness for more robust, interpretable, and adaptive responses. Reasoning is not only a core capability of LLMs but also a fundamental mechanism for LLM safety. Rational employs reasoning-enhanced fine-tuning, allowing it to reject harmful prompts while providing meaningful and context-aware responses in complex scenarios.

摘要: 大型语言模型(LLM)容易受到越狱攻击，这些攻击利用了传统安全对齐中的弱点，传统安全对齐通常依赖僵化的拒绝启发式或表示工程来阻止有害输出。虽然它们对直接对抗性攻击是有效的，但它们不能满足更广泛的安全挑战，需要细致入微的、上下文感知的决策。为了解决这个问题，我们提出了针对可解释LLM安全(Rational)的推理增强精调，这是一个新的框架，它训练模型在响应之前进行显式的安全推理。微调模型利用自生成推理中丰富的预训练知识，通过结构化推理来引导自己的安全性，使上下文敏感决策内在化。我们的发现表明，安全超越了拒绝，需要背景感知才能做出更健壮、可解释和适应性更强的反应。推理是LLMS的核心能力，也是LLMS安全的基本机制。Rational采用了推理增强的微调，使其能够拒绝有害提示，同时在复杂场景中提供有意义的上下文感知响应。



