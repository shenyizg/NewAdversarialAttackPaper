# Latest Adversarial Attack Papers
**update at 2025-03-31 15:47:47**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. CoRPA: Adversarial Image Generation for Chest X-rays Using Concept Vector Perturbations and Generative Models**

CoRPA：使用概念载体扰动和生成模型的胸部X射线对抗图像生成 eess.IV

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2502.05214v2) [paper-pdf](http://arxiv.org/pdf/2502.05214v2)

**Authors**: Amy Rafferty, Rishi Ramaesh, Ajitha Rajan

**Abstract**: Deep learning models for medical image classification tasks are becoming widely implemented in AI-assisted diagnostic tools, aiming to enhance diagnostic accuracy, reduce clinician workloads, and improve patient outcomes. However, their vulnerability to adversarial attacks poses significant risks to patient safety. Current attack methodologies use general techniques such as model querying or pixel value perturbations to generate adversarial examples designed to fool a model. These approaches may not adequately address the unique characteristics of clinical errors stemming from missed or incorrectly identified clinical features. We propose the Concept-based Report Perturbation Attack (CoRPA), a clinically-focused black-box adversarial attack framework tailored to the medical imaging domain. CoRPA leverages clinical concepts to generate adversarial radiological reports and images that closely mirror realistic clinical misdiagnosis scenarios. We demonstrate the utility of CoRPA using the MIMIC-CXR-JPG dataset of chest X-rays and radiological reports. Our evaluation reveals that deep learning models exhibiting strong resilience to conventional adversarial attacks are significantly less robust when subjected to CoRPA's clinically-focused perturbations. This underscores the importance of addressing domain-specific vulnerabilities in medical AI systems. By introducing a specialized adversarial attack framework, this study provides a foundation for developing robust, real-world-ready AI models in healthcare, ensuring their safe and reliable deployment in high-stakes clinical environments.

摘要: 医学图像分类任务的深度学习模型正被广泛应用于人工智能辅助诊断工具中，旨在提高诊断准确性，减少临床医生的工作量，并改善患者的预后。然而，它们在对抗性攻击下的脆弱性对患者的安全构成了重大风险。当前的攻击方法使用诸如模型查询或像素值扰动等一般技术来生成旨在愚弄模型的对抗性示例。这些方法可能不能充分解决因遗漏或错误识别临床特征而产生的临床错误的独特特征。我们提出了一种基于概念的报告扰动攻击(CORPA)，这是一种针对医学成像领域定制的针对临床的黑盒对抗攻击框架。CORPA利用临床概念生成对抗性放射学报告和图像，密切反映现实的临床误诊场景。我们使用胸片和放射学报告的MIMIC-CXR-JPG数据集演示了CORPA的实用性。我们的评估表明，深度学习模型对传统的对抗性攻击表现出很强的弹性，但当受到Corpa的临床重点扰动时，其健壮性明显较差。这突显了解决医疗人工智能系统中特定领域漏洞的重要性。通过引入专门的对抗性攻击框架，本研究为在医疗保健领域开发强大的、现实世界就绪的人工智能模型提供了基础，确保它们在高风险的临床环境中安全可靠地部署。



## **2. Leveraging Expert Input for Robust and Explainable AI-Assisted Lung Cancer Detection in Chest X-rays**

利用专家输入在胸部X光检查中进行稳健且可解释的人工智能辅助肺癌检测 cs.LG

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2403.19444v2) [paper-pdf](http://arxiv.org/pdf/2403.19444v2)

**Authors**: Amy Rafferty, Rishi Ramaesh, Ajitha Rajan

**Abstract**: Deep learning models show significant potential for advancing AI-assisted medical diagnostics, particularly in detecting lung cancer through medical image modalities such as chest X-rays. However, the black-box nature of these models poses challenges to their interpretability and trustworthiness, limiting their adoption in clinical practice. This study examines both the interpretability and robustness of a high-performing lung cancer detection model based on InceptionV3, utilizing a public dataset of chest X-rays and radiological reports. We evaluate the clinical utility of multiple explainable AI (XAI) techniques, including both post-hoc and ante-hoc approaches, and find that existing methods often fail to provide clinically relevant explanations, displaying inconsistencies and divergence from expert radiologist assessments. To address these limitations, we collaborated with a radiologist to define diagnosis-specific clinical concepts and developed ClinicXAI, an expert-driven approach leveraging the concept bottleneck methodology. ClinicXAI generated clinically meaningful explanations which closely aligned with the practical requirements of clinicians while maintaining high diagnostic accuracy. We also assess the robustness of ClinicXAI in comparison to the original InceptionV3 model by subjecting both to a series of widely utilized adversarial attacks. Our analysis demonstrates that ClinicXAI exhibits significantly greater resilience to adversarial perturbations. These findings underscore the importance of incorporating domain expertise into the design of interpretable and robust AI systems for medical diagnostics, paving the way for more trustworthy and effective AI solutions in healthcare.

摘要: 深度学习模型在推进人工智能辅助医疗诊断方面显示出巨大潜力，特别是在通过胸部X光等医学图像模式检测肺癌方面。然而，这些模型的黑匣子性质对其可解释性和可信性提出了挑战，限制了它们在临床实践中的采用。这项研究利用胸部X光片和放射学报告的公共数据集，检查了基于InceptionV3的高性能肺癌检测模型的可解释性和稳健性。我们评估了多种可解释人工智能（XAI）技术（包括事后和临时方法）的临床实用性，并发现现有方法常常无法提供临床相关的解释，从而显示出与放射科专家评估的不一致和分歧。为了解决这些局限性，我们与放射科医生合作定义了诊断特定的临床概念，并开发了ClinicXAI，这是一种利用概念瓶颈方法的专家驱动方法。ClinicXAI产生了具有临床意义的解释，这些解释与临床医生的实际要求密切一致，同时保持了高诊断准确性。我们还通过对ClinicXAI进行一系列广泛使用的对抗攻击，与原始InceptionV3模型相比，评估了ClinicXAI的稳健性。我们的分析表明，ClinicXAI对对抗性扰动表现出明显更强的弹性。这些发现强调了将领域专业知识纳入可解释和强大的医疗诊断AI系统设计的重要性，为医疗保健领域更值得信赖和有效的AI解决方案铺平了道路。



## **3. Privacy-Preserving Secure Neighbor Discovery for Wireless Networks**

无线网络的隐私保护安全邻居发现 cs.CR

10 pages, 6 figures. Author's version; accepted and presented at the  IEEE 23rd International Conference on Trust, Security and Privacy in  Computing and Communications (TrustCom) 2024

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2503.22232v1) [paper-pdf](http://arxiv.org/pdf/2503.22232v1)

**Authors**: Ahmed Mohamed Hussain, Panos Papadimitratos

**Abstract**: Traditional Neighbor Discovery (ND) and Secure Neighbor Discovery (SND) are key elements for network functionality. SND is a hard problem, satisfying not only typical security properties (authentication, integrity) but also verification of direct communication, which involves distance estimation based on time measurements and device coordinates. Defeating relay attacks, also known as "wormholes", leading to stealthy Byzantine links and significant degradation of communication and adversarial control, is key in many wireless networked systems. However, SND is not concerned with privacy; it necessitates revealing the identity and location of the device(s) participating in the protocol execution. This can be a deterrent for deployment, especially involving user-held devices in the emerging Internet of Things (IoT) enabled smart environments. To address this challenge, we present a novel Privacy-Preserving Secure Neighbor Discovery (PP-SND) protocol, enabling devices to perform SND without revealing their actual identities and locations, effectively decoupling discovery from the exposure of sensitive information. We use Homomorphic Encryption (HE) for computing device distances without revealing their actual coordinates, as well as employing a pseudonymous device authentication to hide identities while preserving communication integrity. PP-SND provides SND [1] along with pseudonymity, confidentiality, and unlinkability. Our presentation here is not specific to one wireless technology, and we assess the performance of the protocols (cryptographic overhead) on a Raspberry Pi 4 and provide a security and privacy analysis.

摘要: 传统邻居发现（ND）和安全邻居发现（SND）是网络功能的关键要素。SND是一个很难的问题，不仅满足典型的安全属性（认证，完整性），但也验证直接通信，其中涉及基于时间测量和设备坐标的距离估计。击败中继攻击，也被称为“虫洞”，导致隐形拜占庭链路和通信和对抗控制的显着退化，是许多无线网络系统中的关键。然而，SND不涉及隐私;它需要揭示参与协议执行的设备的身份和位置。这可能会阻碍部署，特别是涉及新兴的物联网（IoT）智能环境中的用户持有设备。为了应对这一挑战，我们提出了一种新型的隐私保护安全邻居发现（PP-SND）协议，使设备能够在不透露其实际身份和位置的情况下执行SND，从而有效地将发现与敏感信息的暴露脱钩。我们在不透露其实际坐标的情况下使用Homomorphic加密（HE）来计算设备距离，并采用假名设备身份验证来隐藏身份，同时保持通信完整性。PP-SND提供SND [1]以及同义性、机密性和不可链接性。我们这里的演示并不特定于一种无线技术，我们评估了Raspberry Pi 4上协议的性能（加密负载），并提供安全和隐私分析。



## **4. Data-Free Universal Attack by Exploiting the Intrinsic Vulnerability of Deep Models**

利用深度模型的固有漏洞进行无数据通用攻击 cs.LG

Accepted in AAAI 2025

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2503.22205v1) [paper-pdf](http://arxiv.org/pdf/2503.22205v1)

**Authors**: YangTian Yan, Jinyu Tian

**Abstract**: Deep neural networks (DNNs) are susceptible to Universal Adversarial Perturbations (UAPs), which are instance agnostic perturbations that can deceive a target model across a wide range of samples. Unlike instance-specific adversarial examples, UAPs present a greater challenge as they must generalize across different samples and models. Generating UAPs typically requires access to numerous examples, which is a strong assumption in real-world tasks. In this paper, we propose a novel data-free method called Intrinsic UAP (IntriUAP), by exploiting the intrinsic vulnerabilities of deep models. We analyze a series of popular deep models composed of linear and nonlinear layers with a Lipschitz constant of 1, revealing that the vulnerability of these models is predominantly influenced by their linear components. Based on this observation, we leverage the ill-conditioned nature of the linear components by aligning the UAP with the right singular vectors corresponding to the maximum singular value of each linear layer. Remarkably, our method achieves highly competitive performance in attacking popular image classification deep models without using any image samples. We also evaluate the black-box attack performance of our method, showing that it matches the state-of-the-art baseline for data-free methods on models that conform to our theoretical framework. Beyond the data-free assumption, IntriUAP also operates under a weaker assumption, where the adversary only can access a few of the victim model's layers. Experiments demonstrate that the attack success rate decreases by only 4% when the adversary has access to just 50% of the linear layers in the victim model.

摘要: 深度神经网络(DNN)容易受到通用对抗性扰动(UAP)的影响，UAP是一种实例不可知的扰动，可以在广泛的样本范围内欺骗目标模型。与特定于实例的对抗性例子不同，UAP提出了更大的挑战，因为它们必须在不同的样本和模型中推广。生成UAP通常需要访问大量示例，这在现实世界的任务中是一个强有力的假设。本文利用深层模型的固有脆弱性，提出了一种新的无数据方法--本征UAP(Intrative UAP)。我们分析了一系列流行的由线性层和非线性层组成的Lipschitz常数为1的深层模型，揭示了这些模型的脆弱性主要受其线性分量的影响。基于这一观察，我们通过将UAP与对应于每个线性层的最大奇异值的右奇异向量对齐来利用线性分量的病态性质。值得注意的是，我们的方法在不使用任何图像样本的情况下，在攻击流行的图像分类深度模型方面取得了极具竞争力的性能。我们还评估了我们的方法的黑盒攻击性能，表明它符合符合我们的理论框架的模型上无数据方法的最新基线。除了无数据假设之外，IntriUAP还在一个较弱的假设下运行，即对手只能访问受害者模型的几个层。实验表明，当攻击者只访问受害者模型中50%的线性层时，攻击成功率仅下降4%。



## **5. AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models**

AnyAttack：走向对视觉语言模型的大规模自我监督对抗攻击 cs.LG

CVPR 2025

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2410.05346v3) [paper-pdf](http://arxiv.org/pdf/2410.05346v3)

**Authors**: Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Yunhao Chen, Jitao Sang, Dit-Yan Yeung

**Abstract**: Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks. Traditional targeted adversarial attacks require specific targets and labels, limiting their real-world impact.We present AnyAttack, a self-supervised framework that transcends the limitations of conventional attacks through a novel foundation model approach. By pre-training on the massive LAION-400M dataset without label supervision, AnyAttack achieves unprecedented flexibility - enabling any image to be transformed into an attack vector targeting any desired output across different VLMs.This approach fundamentally changes the threat landscape, making adversarial capabilities accessible at an unprecedented scale. Our extensive validation across five open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) demonstrates AnyAttack's effectiveness across diverse multimodal tasks. Most concerning, AnyAttack seamlessly transfers to commercial systems including Google Gemini, Claude Sonnet, Microsoft Copilot and OpenAI GPT, revealing a systemic vulnerability requiring immediate attention.

摘要: 由于其多通道能力，视觉语言模型(VLM)在现实世界场景中发现了许多有影响力的应用。然而，最近的研究表明，VLM很容易受到基于图像的对抗性攻击。传统的定向攻击需要特定的目标和标签，从而限制了它们在现实世界中的影响，我们提出了一个自我监督框架AnyAttack，它通过一种新的基础模型方法超越了传统攻击的限制。通过在没有标签监管的海量LAION-400M数据集上进行预训练，AnyAttack实现了前所未有的灵活性-使任何图像都能够转换为针对不同VLM中任何所需输出的攻击矢量。这种方法从根本上改变了威胁格局，使敌方能力能够以前所未有的规模获得。我们对五个开源VLM(CLIP、BLIP、BLIP2、InstructBLIP和MiniGPT-4)的广泛验证证明了AnyAttack在各种多模式任务中的有效性。最令人担忧的是，AnyAttack无缝传输到包括Google Gemini、Claude Sonnet、Microsoft Copilot和OpenAI GPT在内的商业系统，暴露出一个需要立即关注的系统性漏洞。



## **6. Foot-In-The-Door: A Multi-turn Jailbreak for LLMs**

一脚踏进门：LLC的多次越狱 cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2502.19820v3) [paper-pdf](http://arxiv.org/pdf/2502.19820v3)

**Authors**: Zixuan Weng, Xiaolong Jin, Jinyuan Jia, Xiangyu Zhang

**Abstract**: Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.

摘要: 随着大型语言模型越来越多地融入现实世界的应用程序中，确保人工智能的安全至关重要。一个关键的挑战是越狱，敌意提示绕过内置的保护措施，导致有害的不允许输出。受心理学进门原则的启发，我们引入了FITD，这是一种新颖的多转弯越狱方法，它利用了这样一种现象，即较小的初始承诺降低了对更重大或更不道德的违法行为的抵抗力。我们的方法通过中间桥提示逐步升级用户查询的恶意意图，并使模型本身的响应保持一致，以诱导有毒响应。在两个越狱基准上的广泛实验结果表明，FITD在七个广泛使用的模型上实现了94%的平均攻击成功率，性能优于现有的最先进方法。此外，我们还提供了对LLM自我腐败的深入分析，强调了当前调整策略中的漏洞，并强调了多轮交互中固有的风险。代码可在https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.上获得



## **7. Learning to Lie: Reinforcement Learning Attacks Damage Human-AI Teams and Teams of LLMs**

学会撒谎：强化学习攻击损害人类人工智能团队和LLM团队 cs.HC

17 pages, 9 figures, accepted to ICLR 2025 Workshop on Human-AI  Coevolution

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21983v1) [paper-pdf](http://arxiv.org/pdf/2503.21983v1)

**Authors**: Abed Kareem Musaffar, Anand Gokhale, Sirui Zeng, Rasta Tadayon, Xifeng Yan, Ambuj Singh, Francesco Bullo

**Abstract**: As artificial intelligence (AI) assistants become more widely adopted in safety-critical domains, it becomes important to develop safeguards against potential failures or adversarial attacks. A key prerequisite to developing these safeguards is understanding the ability of these AI assistants to mislead human teammates. We investigate this attack problem within the context of an intellective strategy game where a team of three humans and one AI assistant collaborate to answer a series of trivia questions. Unbeknownst to the humans, the AI assistant is adversarial. Leveraging techniques from Model-Based Reinforcement Learning (MBRL), the AI assistant learns a model of the humans' trust evolution and uses that model to manipulate the group decision-making process to harm the team. We evaluate two models -- one inspired by literature and the other data-driven -- and find that both can effectively harm the human team. Moreover, we find that in this setting our data-driven model is capable of accurately predicting how human agents appraise their teammates given limited information on prior interactions. Finally, we compare the performance of state-of-the-art LLM models to human agents on our influence allocation task to evaluate whether the LLMs allocate influence similarly to humans or if they are more robust to our attack. These results enhance our understanding of decision-making dynamics in small human-AI teams and lay the foundation for defense strategies.

摘要: 随着人工智能（AI）助手在安全关键领域越来越广泛地采用，开发针对潜在故障或对抗攻击的防护措施变得重要。开发这些保护措施的一个关键先决条件是了解这些人工智能助手误导人类队友的能力。我们在一个推理策略游戏的背景下调查了这个攻击问题，其中三名人类和一名人工智能助理组成的团队合作回答一系列琐碎问题。人类不知道的是，人工智能助手是敌对的。利用基于模型的强化学习（MBRL）的技术，人工智能助手学习人类信任演变的模型，并使用该模型来操纵群体决策过程以伤害团队。我们评估了两个模型--一个受文献启发，另一个受数据驱动--并发现两者都可以有效地伤害人类团队。此外，我们发现，在这种情况下，我们的数据驱动模型能够准确预测人类代理如何在先前互动的有限信息的情况下评估其队友。最后，我们将最先进的LLM模型与人类代理在影响力分配任务中的性能进行了比较，以评估LLM是否以类似于人类的方式分配影响力，或者它们是否对我们的攻击更稳健。这些结果增强了我们对小型人工智能团队决策动态的理解，并为防御策略奠定了基础。



## **8. Harnessing Chain-of-Thought Metadata for Task Routing and Adversarial Prompt Detection**

利用思想链元数据进行任务路由和对抗提示检测 cs.CL

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21464v1) [paper-pdf](http://arxiv.org/pdf/2503.21464v1)

**Authors**: Ryan Marinelli, Josef Pichlmeier, Tamas Bisztray

**Abstract**: In this work, we propose a metric called Number of Thoughts (NofT) to determine the difficulty of tasks pre-prompting and support Large Language Models (LLMs) in production contexts. By setting thresholds based on the number of thoughts, this metric can discern the difficulty of prompts and support more effective prompt routing. A 2% decrease in latency is achieved when routing prompts from the MathInstruct dataset through quantized, distilled versions of Deepseek with 1.7 billion, 7 billion, and 14 billion parameters. Moreover, this metric can be used to detect adversarial prompts used in prompt injection attacks with high efficacy. The Number of Thoughts can inform a classifier that achieves 95% accuracy in adversarial prompt detection. Our experiments ad datasets used are available on our GitHub page: https://github.com/rymarinelli/Number_Of_Thoughts/tree/main.

摘要: 在这项工作中，我们提出了一种名为“思考数量”（NofT）的指标，以确定预提示任务的难度并支持生产环境中的大型语言模型（LLM）。通过根据想法数量设置阈值，该指标可以辨别提示的难度并支持更有效的提示路由。当通过具有17亿、70亿和140亿参数的量化、提炼版本的Deepseek从MathDirect数据集中路由提示时，延迟可降低2%。此外，该指标可用于高效检测提示注射攻击中使用的对抗提示。思维数量可以通知分类器，在对抗性提示检测中达到95%的准确率。我们使用的实验和数据集可以在我们的GitHub页面上找到：https://github.com/rymarinelli/Number_Of_Thoughts/tree/main。



## **9. Tricking Retrievers with Influential Tokens: An Efficient Black-Box Corpus Poisoning Attack**

用有影响力的代币欺骗猎犬：一种有效的黑匣子库中毒攻击 cs.LG

Accepted to NAACL 2025 Main Track

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21315v1) [paper-pdf](http://arxiv.org/pdf/2503.21315v1)

**Authors**: Cheng Wang, Yiwei Wang, Yujun Cai, Bryan Hooi

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models by incorporating external knowledge, addressing issues like outdated internal knowledge and hallucination. However, their reliance on external knowledge bases makes them vulnerable to corpus poisoning attacks, where adversarial passages can be injected to manipulate retrieval results. Existing methods for crafting such passages, such as random token replacement or training inversion models, are often slow and computationally expensive, requiring either access to retriever's gradients or large computational resources. To address these limitations, we propose Dynamic Importance-Guided Genetic Algorithm (DIGA), an efficient black-box method that leverages two key properties of retrievers: insensitivity to token order and bias towards influential tokens. By focusing on these characteristics, DIGA dynamically adjusts its genetic operations to generate effective adversarial passages with significantly reduced time and memory usage. Our experimental evaluation shows that DIGA achieves superior efficiency and scalability compared to existing methods, while maintaining comparable or better attack success rates across multiple datasets.

摘要: 检索增强生成（RAG）系统通过整合外部知识来增强大型语言模型，解决过时的内部知识和幻觉等问题。然而，它们对外部知识库的依赖使它们容易受到文集中毒攻击，其中可以注入对抗性段落来操纵检索结果。用于制作此类段落的现有方法，例如随机令牌替换或训练倒置模型，通常速度缓慢且计算昂贵，需要访问检索器的梯度或大量计算资源。为了解决这些限制，我们提出了动态重要引导遗传算法（DIGA），这是一种有效的黑匣子方法，利用检索器的两个关键属性：对代币顺序的不敏感和对有影响力代币的偏见。通过关注这些特征，DIGA动态地调整其遗传操作，以生成有效的对抗通道，并显着减少时间和内存使用。我们的实验评估表明，与现有方法相比，DIGA实现了更高的效率和可扩展性，同时在多个数据集上保持了可比或更好的攻击成功率。



## **10. Clean Image May be Dangerous: Data Poisoning Attacks Against Deep Hashing**

干净的图像可能是危险的：针对深度哈希的数据中毒攻击 cs.CV

Accepted by TMM

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21236v1) [paper-pdf](http://arxiv.org/pdf/2503.21236v1)

**Authors**: Shuai Li, Jie Zhang, Yuang Qi, Kejiang Chen, Tianwei Zhang, Weiming Zhang, Nenghai Yu

**Abstract**: Large-scale image retrieval using deep hashing has become increasingly popular due to the exponential growth of image data and the remarkable feature extraction capabilities of deep neural networks (DNNs). However, deep hashing methods are vulnerable to malicious attacks, including adversarial and backdoor attacks. It is worth noting that these attacks typically involve altering the query images, which is not a practical concern in real-world scenarios. In this paper, we point out that even clean query images can be dangerous, inducing malicious target retrieval results, like undesired or illegal images. To the best of our knowledge, we are the first to study data \textbf{p}oisoning \textbf{a}ttacks against \textbf{d}eep \textbf{hash}ing \textbf{(\textit{PADHASH})}. Specifically, we first train a surrogate model to simulate the behavior of the target deep hashing model. Then, a strict gradient matching strategy is proposed to generate the poisoned images. Extensive experiments on different models, datasets, hash methods, and hash code lengths demonstrate the effectiveness and generality of our attack method.

摘要: 由于图像数据的指数级增长和深度神经网络（DNN）出色的特征提取能力，使用深度哈希的大规模图像检索变得越来越受欢迎。然而，深度哈希方法很容易受到恶意攻击，包括对抗性和后门攻击。值得注意的是，这些攻击通常涉及更改查询图像，这在现实世界场景中不是一个实际问题。在本文中，我们指出，即使是干净的查询图像也可能是危险的，会引发恶意目标检索结果，例如不需要或非法的图像。据我们所知，我们是第一个研究数据\textBF{p}oisoning \textBF{a} tacks针对\textBF{d}eep \textBF{hash}ing \textBF{（\texttit {PADHASH}）}。具体来说，我们首先训练一个代理模型来模拟目标深度哈希模型的行为。然后，提出了严格的梯度匹配策略来生成中毒图像。对不同模型、数据集、哈希方法和哈希码长度的广泛实验证明了我们攻击方法的有效性和通用性。



## **11. Adversarial Wear and Tear: Exploiting Natural Damage for Generating Physical-World Adversarial Examples**

对抗性磨损和撕裂：利用自然损害产生物理世界对抗性例子 cs.CV

11 pages, 9 figures

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21164v1) [paper-pdf](http://arxiv.org/pdf/2503.21164v1)

**Authors**: Samra Irshad, Seungkyu Lee, Nassir Navab, Hong Joo Lee, Seong Tae Kim

**Abstract**: The presence of adversarial examples in the physical world poses significant challenges to the deployment of Deep Neural Networks in safety-critical applications such as autonomous driving. Most existing methods for crafting physical-world adversarial examples are ad-hoc, relying on temporary modifications like shadows, laser beams, or stickers that are tailored to specific scenarios. In this paper, we introduce a new class of physical-world adversarial examples, AdvWT, which draws inspiration from the naturally occurring phenomenon of `wear and tear', an inherent property of physical objects. Unlike manually crafted perturbations, `wear and tear' emerges organically over time due to environmental degradation, as seen in the gradual deterioration of outdoor signboards. To achieve this, AdvWT follows a two-step approach. First, a GAN-based, unsupervised image-to-image translation network is employed to model these naturally occurring damages, particularly in the context of outdoor signboards. The translation network encodes the characteristics of damaged signs into a latent `damage style code'. In the second step, we introduce adversarial perturbations into the style code, strategically optimizing its transformation process. This manipulation subtly alters the damage style representation, guiding the network to generate adversarial images where the appearance of damages remains perceptually realistic, while simultaneously ensuring their effectiveness in misleading neural networks. Through comprehensive experiments on two traffic sign datasets, we show that AdvWT effectively misleads DNNs in both digital and physical domains. AdvWT achieves an effective attack success rate, greater robustness, and a more natural appearance compared to existing physical-world adversarial examples. Additionally, integrating AdvWT into training enhances a model's generalizability to real-world damaged signs.

摘要: 物理世界中对抗性示例的存在对在自动驾驶等安全关键型应用中部署深度神经网络提出了重大挑战。大多数现有的制作物理世界对抗示例的方法都是临时的，依赖于临时修改，如阴影，激光束或针对特定场景定制的贴纸。在本文中，我们引入了一类新的物理世界对抗性示例AdvWT，它从自然发生的“磨损”现象（物理对象的固有属性）中汲取灵感。与手工制作的扰动不同，由于环境退化，随着时间的推移，“磨损”会自然出现，如户外招牌的逐渐退化。为了实现这一目标，AdvWT遵循两步方法。首先，采用基于GAN的无监督图像到图像翻译网络来对这些自然发生的损害进行建模，特别是在户外招牌的情况下。翻译网络将损坏标志的特征编码为潜在的“损坏风格代码”。在第二步中，我们将对抗性扰动引入风格代码，战略性地优化其转换过程。这种操纵微妙地改变了损害风格的表示，引导网络生成对抗图像，其中损害的外观在感知上保持真实，同时确保其有效地误导神经网络。通过对两个交通标志数据集的全面实验，我们表明AdvWT在数字和物理领域有效误导DNN。与现有的物理世界对抗示例相比，AdvWT实现了有效的攻击成功率、更强的鲁棒性和更自然的外观。此外，将AdvWT集成到训练中可以增强模型对现实世界受损迹象的概括性。



## **12. Robust Federated Learning Against Poisoning Attacks: A GAN-Based Defense Framework**

针对中毒攻击的稳健联邦学习：基于GAN的防御框架 cs.CR

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20884v1) [paper-pdf](http://arxiv.org/pdf/2503.20884v1)

**Authors**: Usama Zafar, André Teixeira, Salman Toor

**Abstract**: Federated Learning (FL) enables collaborative model training across decentralized devices without sharing raw data, but it remains vulnerable to poisoning attacks that compromise model integrity. Existing defenses often rely on external datasets or predefined heuristics (e.g. number of malicious clients), limiting their effectiveness and scalability. To address these limitations, we propose a privacy-preserving defense framework that leverages a Conditional Generative Adversarial Network (cGAN) to generate synthetic data at the server for authenticating client updates, eliminating the need for external datasets. Our framework is scalable, adaptive, and seamlessly integrates into FL workflows. Extensive experiments on benchmark datasets demonstrate its robust performance against a variety of poisoning attacks, achieving high True Positive Rate (TPR) and True Negative Rate (TNR) of malicious and benign clients, respectively, while maintaining model accuracy. The proposed framework offers a practical and effective solution for securing federated learning systems.

摘要: 联邦学习（FL）支持跨分散设备的协作模型训练，而无需共享原始数据，但它仍然容易受到损害模型完整性的中毒攻击。现有的防御通常依赖于外部数据集或预定义的入侵（例如恶意客户端的数量），限制了它们的有效性和可扩展性。为了解决这些限制，我们提出了一个保护隐私的防御框架，该框架利用条件生成对抗网络（cGAN）在服务器上生成合成数据，以验证客户端更新，从而消除了对外部数据集的需求。我们的框架是可扩展的，自适应的，并无缝集成到FL工作流程。对基准数据集的广泛实验证明了其针对各种中毒攻击的稳健性能，分别实现恶意和良性客户端的高真阳性率（TPA）和真阴性率（TNR），同时保持模型准确性。拟议的框架为保护联邦学习系统提供了一个实用而有效的解决方案。



## **13. The mathematics of adversarial attacks in AI -- Why deep learning is unstable despite the existence of stable neural networks**

人工智能中对抗性攻击的数学--为什么尽管存在稳定的神经网络，深度学习却不稳定 cs.LG

31 pages, 1 figure. Revised to make minor changes to notation and  references

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2109.06098v2) [paper-pdf](http://arxiv.org/pdf/2109.06098v2)

**Authors**: Alexander Bastounis, Anders C Hansen, Verner Vlačić

**Abstract**: The unprecedented success of deep learning (DL) makes it unchallenged when it comes to classification problems. However, it is well established that the current DL methodology produces universally unstable neural networks (NNs). The instability problem has caused an enormous research effort -- with a vast literature on so-called adversarial attacks -- yet there has been no solution to the problem. Our paper addresses why there has been no solution to the problem, as we prove the following mathematical paradox: any training procedure based on training neural networks for classification problems with a fixed architecture will yield neural networks that are either inaccurate or unstable (if accurate) -- despite the provable existence of both accurate and stable neural networks for the same classification problems. The key is that the stable and accurate neural networks must have variable dimensions depending on the input, in particular, variable dimensions is a necessary condition for stability.   Our result points towards the paradox that accurate and stable neural networks exist, however, modern algorithms do not compute them. This yields the question: if the existence of neural networks with desirable properties can be proven, can one also find algorithms that compute them? There are cases in mathematics where provable existence implies computability, but will this be the case for neural networks? The contrary is true, as we demonstrate how neural networks can provably exist as approximate minimisers to standard optimisation problems with standard cost functions, however, no randomised algorithm can compute them with probability better than 1/2.

摘要: 深度学习的空前成功使其在分类问题上无人能及。然而，众所周知，当前的DL方法产生了普遍不稳定的神经网络(NNS)。不稳定问题已经引起了巨大的研究努力--有大量关于所谓的对抗性攻击的文献--但这个问题还没有解决方案。我们的论文解决了为什么这个问题没有解决方案，因为我们证明了以下数学悖论：任何基于对具有固定体系结构的分类问题的神经网络进行训练的训练过程都会产生不准确或不稳定(如果准确)的神经网络--尽管对于相同的分类问题存在准确和稳定的神经网络。关键是稳定和准确的神经网络必须具有依赖于输入的可变维度，特别是可变维度是稳定的必要条件。我们的结果指出了这样一个悖论，即准确和稳定的神经网络是存在的，然而，现代算法并不计算它们。这就产生了一个问题：如果可以证明具有理想特性的神经网络的存在，人们还能找到计算它们的算法吗？在数学中，有可证明的存在意味着可计算的情况，但神经网络会是这样吗？相反，当我们展示神经网络如何证明可以作为具有标准成本函数的标准优化问题的近似最小化存在时，然而，没有任何随机算法可以以比1/2更好的概率计算它们。



## **14. Intelligent Code Embedding Framework for High-Precision Ransomware Detection via Multimodal Execution Path Analysis**

通过多模式执行路径分析实现高精度勒索软件检测的智能代码嵌入框架 cs.CR

arXiv admin note: This paper has been withdrawn by arXiv due to  disputed and unverifiable authorship

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2501.15836v2) [paper-pdf](http://arxiv.org/pdf/2501.15836v2)

**Authors**: Levi Gareth, Maximilian Fairbrother, Peregrine Blackwood, Lucasta Underhill, Benedict Ruthermore

**Abstract**: Modern threat landscapes continue to evolve with increasing sophistication, challenging traditional detection methodologies and necessitating innovative solutions capable of addressing complex adversarial tactics. A novel framework was developed to identify ransomware activity through multimodal execution path analysis, integrating high-dimensional embeddings and dynamic heuristic derivation mechanisms to capture behavioral patterns across diverse attack variants. The approach demonstrated high adaptability, effectively mitigating obfuscation strategies and polymorphic characteristics often employed by ransomware families to evade detection. Comprehensive experimental evaluations revealed significant advancements in precision, recall, and accuracy metrics compared to baseline techniques, particularly under conditions of variable encryption speeds and obfuscated execution flows. The framework achieved scalable and computationally efficient performance, ensuring robust applicability across a range of system configurations, from resource-constrained environments to high-performance infrastructures. Notable findings included reduced false positive rates and enhanced detection latency, even for ransomware families employing sophisticated encryption mechanisms. The modular design allowed seamless integration of additional modalities, enabling extensibility and future-proofing against emerging threat vectors. Quantitative analyses further highlighted the system's energy efficiency, emphasizing its practicality for deployment in environments with stringent operational constraints. The results underline the importance of integrating advanced computational techniques and dynamic adaptability to safeguard digital ecosystems from increasingly complex threats.

摘要: 现代威胁格局继续演变，日益复杂，挑战传统的检测方法，并需要能够解决复杂的对抗策略的创新解决方案。开发了一个新颖的框架，通过多模式执行路径分析来识别勒索软件活动，集成了多维嵌入和动态启发式派生机制，以捕获不同攻击变体之间的行为模式。该方法表现出高度的适应性，有效地减轻了勒索软件家族经常使用的混淆策略和多态特征。全面的实验评估显示，与基线技术相比，准确性、召回率和准确性指标有了显着进步，特别是在加密速度可变和执行流模糊的情况下。该框架实现了可扩展和计算效率高的性能，确保了一系列系统配置（从资源受限的环境到高性能基础设施）的稳健适用性。值得注意的发现包括降低假阳性率和增强检测延迟，即使对于采用复杂加密机制的勒索软件家族也是如此。模块化设计允许无缝集成其他模式，实现可扩展性和面向未来的针对新出现的威胁载体的能力。定量分析进一步强调了该系统的能源效率，强调了其在具有严格运营限制的环境中部署的实用性。结果强调了集成先进计算技术和动态适应性以保护数字生态系统免受日益复杂的威胁的重要性。



## **15. A Survey of Secure Semantic Communications**

安全语义通信综述 cs.CR

160 pages, 27 figures

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2501.00842v2) [paper-pdf](http://arxiv.org/pdf/2501.00842v2)

**Authors**: Rui Meng, Song Gao, Dayu Fan, Haixiao Gao, Yining Wang, Xiaodong Xu, Bizhu Wang, Suyu Lv, Zhidi Zhang, Mengying Sun, Shujun Han, Chen Dong, Xiaofeng Tao, Ping Zhang

**Abstract**: Semantic communication (SemCom) is regarded as a promising and revolutionary technology in 6G, aiming to transcend the constraints of ``Shannon's trap" by filtering out redundant information and extracting the core of effective data. Compared to traditional communication paradigms, SemCom offers several notable advantages, such as reducing the burden on data transmission, enhancing network management efficiency, and optimizing resource allocation. Numerous researchers have extensively explored SemCom from various perspectives, including network architecture, theoretical analysis, potential technologies, and future applications. However, as SemCom continues to evolve, a multitude of security and privacy concerns have arisen, posing threats to the confidentiality, integrity, and availability of SemCom systems. This paper presents a comprehensive survey of the technologies that can be utilized to secure SemCom. Firstly, we elaborate on the entire life cycle of SemCom, which includes the model training, model transfer, and semantic information transmission phases. Then, we identify the security and privacy issues that emerge during these three stages. Furthermore, we summarize the techniques available to mitigate these security and privacy threats, including data cleaning, robust learning, defensive strategies against backdoor attacks, adversarial training, differential privacy, cryptography, blockchain technology, model compression, and physical-layer security. Lastly, this paper outlines future research directions to guide researchers in related fields.

摘要: 语义通信(SemCom)被认为是6G中一种很有前途的革命性技术，旨在通过过滤冗余信息和提取有效数据的核心来超越香农陷阱的限制。与传统的通信模式相比，SemCom具有一些显著的优势，如减轻数据传输负担，提高网络管理效率，优化资源配置。许多研究人员从不同的角度对SemCom进行了广泛的探索，包括网络体系结构、理论分析、潜在技术和未来应用。然而，随着SemCom的不断发展，出现了大量的安全和隐私问题，对SemCom系统的机密性、完整性和可用性构成了威胁。本文对可用于确保SemCom安全的技术进行了全面的综述。首先，详细阐述了SemCom的整个生命周期，包括模型训练、模型迁移、语义信息传递等阶段。然后，我们确定在这三个阶段中出现的安全和隐私问题。此外，我们还总结了可用于缓解这些安全和隐私威胁的技术，包括数据清理、稳健学习、针对后门攻击的防御策略、对抗性训练、差异隐私、密码学、区块链技术、模型压缩和物理层安全。最后，本文概述了未来的研究方向，以指导相关领域的研究人员。



## **16. $β$-GNN: A Robust Ensemble Approach Against Graph Structure Perturbation**

$β$-GNN：一种针对图结构扰动的鲁棒集成方法 cs.LG

This is the author's version of the paper accepted at EuroMLSys 2025

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20630v1) [paper-pdf](http://arxiv.org/pdf/2503.20630v1)

**Authors**: Haci Ismail Aslan, Philipp Wiesner, Ping Xiong, Odej Kao

**Abstract**: Graph Neural Networks (GNNs) are playing an increasingly important role in the efficient operation and security of computing systems, with applications in workload scheduling, anomaly detection, and resource management. However, their vulnerability to network perturbations poses a significant challenge. We propose $\beta$-GNN, a model enhancing GNN robustness without sacrificing clean data performance. $\beta$-GNN uses a weighted ensemble, combining any GNN with a multi-layer perceptron. A learned dynamic weight, $\beta$, modulates the GNN's contribution. This $\beta$ not only weights GNN influence but also indicates data perturbation levels, enabling proactive mitigation. Experimental results on diverse datasets show $\beta$-GNN's superior adversarial accuracy and attack severity quantification. Crucially, $\beta$-GNN avoids perturbation assumptions, preserving clean data structure and performance.

摘要: 图形神经网络（GNN）在计算系统的高效操作和安全性方面发挥着越来越重要的作用，在工作负载调度、异常检测和资源管理方面应用。然而，它们对网络扰动的脆弱性构成了重大挑战。我们提出了$\Beta$-GNN，这是一个在不牺牲干净数据性能的情况下增强GNN稳健性的模型。$\Beta$-GNN使用加权集合，将任何GNN与多层感知器相结合。习得的动态权重$\Beta$调节GNN的贡献。此$\Beta$不仅加权GNN影响，还指示数据扰动水平，从而实现主动缓解。不同数据集的实验结果表明$\Beta$-GNN具有卓越的对抗准确性和攻击严重性量化。至关重要的是，$\Beta$-GNN避免了扰动假设，保留了干净的数据结构和性能。



## **17. Robust Deep Reinforcement Learning in Robotics via Adaptive Gradient-Masked Adversarial Attacks**

通过自适应者掩蔽对抗攻击在机器人技术中进行鲁棒的深度强化学习 cs.LG

9 pages, 6 figures

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20844v1) [paper-pdf](http://arxiv.org/pdf/2503.20844v1)

**Authors**: Zongyuan Zhang, Tianyang Duan, Zheng Lin, Dong Huang, Zihan Fang, Zekai Sun, Ling Xiong, Hongbin Liang, Heming Cui, Yong Cui, Yue Gao

**Abstract**: Deep reinforcement learning (DRL) has emerged as a promising approach for robotic control, but its realworld deployment remains challenging due to its vulnerability to environmental perturbations. Existing white-box adversarial attack methods, adapted from supervised learning, fail to effectively target DRL agents as they overlook temporal dynamics and indiscriminately perturb all state dimensions, limiting their impact on long-term rewards. To address these challenges, we propose the Adaptive Gradient-Masked Reinforcement (AGMR) Attack, a white-box attack method that combines DRL with a gradient-based soft masking mechanism to dynamically identify critical state dimensions and optimize adversarial policies. AGMR selectively allocates perturbations to the most impactful state features and incorporates a dynamic adjustment mechanism to balance exploration and exploitation during training. Extensive experiments demonstrate that AGMR outperforms state-of-the-art adversarial attack methods in degrading the performance of the victim agent and enhances the victim agent's robustness through adversarial defense mechanisms.

摘要: 深度强化学习（DRL）已成为机器人控制的一种有前途的方法，但由于其易受环境扰动的影响，其在现实世界中的部署仍然具有挑战性。现有的白盒对抗性攻击方法，从监督学习，未能有效地针对DRL代理，因为它们忽略了时间动态和不加区别地扰动所有状态维度，限制了它们对长期奖励的影响。为了解决这些挑战，我们提出了自适应屏蔽强化（AGMR）攻击，这是一种白盒攻击方法，将DRL与基于梯度的软屏蔽机制相结合，以动态识别关键状态维度并优化对抗策略。AGMR有选择地将扰动分配给最有影响力的状态特征，并采用动态调整机制来平衡训练过程中的探索和利用。大量实验表明，AGMR在降低受害者代理的性能方面优于最先进的对抗攻击方法，并通过对抗防御机制增强受害者代理的鲁棒性。



## **18. State-Aware Perturbation Optimization for Robust Deep Reinforcement Learning**

用于稳健深度强化学习的状态感知扰动优化 cs.LG

15 pages, 11 figures

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20613v1) [paper-pdf](http://arxiv.org/pdf/2503.20613v1)

**Authors**: Zongyuan Zhang, Tianyang Duan, Zheng Lin, Dong Huang, Zihan Fang, Zekai Sun, Ling Xiong, Hongbin Liang, Heming Cui, Yong Cui

**Abstract**: Recently, deep reinforcement learning (DRL) has emerged as a promising approach for robotic control. However, the deployment of DRL in real-world robots is hindered by its sensitivity to environmental perturbations. While existing whitebox adversarial attacks rely on local gradient information and apply uniform perturbations across all states to evaluate DRL robustness, they fail to account for temporal dynamics and state-specific vulnerabilities. To combat the above challenge, we first conduct a theoretical analysis of white-box attacks in DRL by establishing the adversarial victim-dynamics Markov decision process (AVD-MDP), to derive the necessary and sufficient conditions for a successful attack. Based on this, we propose a selective state-aware reinforcement adversarial attack method, named STAR, to optimize perturbation stealthiness and state visitation dispersion. STAR first employs a soft mask-based state-targeting mechanism to minimize redundant perturbations, enhancing stealthiness and attack effectiveness. Then, it incorporates an information-theoretic optimization objective to maximize mutual information between perturbations, environmental states, and victim actions, ensuring a dispersed state-visitation distribution that steers the victim agent into vulnerable states for maximum return reduction. Extensive experiments demonstrate that STAR outperforms state-of-the-art benchmarks.

摘要: 最近，深度强化学习（DRL）已经成为机器人控制的一种很有前途的方法。然而，DRL在现实世界的机器人部署是阻碍其对环境扰动的敏感性。虽然现有的白盒对抗性攻击依赖于局部梯度信息，并在所有状态之间应用均匀扰动来评估DRL的鲁棒性，但它们无法考虑时间动态和特定于状态的漏洞。为了应对上述挑战，我们首先通过建立对抗性受害者动态马尔可夫决策过程（AVD-MDP）对DRL中的白盒攻击进行理论分析，以获得成功攻击的充分必要条件。在此基础上，我们提出了一种选择性的状态感知强化对抗攻击方法STAR，以优化扰动隐秘性和状态访问分散度。STAR首先采用基于软屏蔽的状态瞄准机制来最大限度地减少冗余扰动，增强隐身性和攻击有效性。然后，它结合了一个信息论优化目标，以最大化扰动、环境状态和受害者行为之间的互信息，确保分散的状态访问分布，将受害者代理引导到脆弱状态以获得最大回报减少。大量实验表明，STAR的性能优于最先进的基准。



## **19. Feature Statistics with Uncertainty Help Adversarial Robustness**

具有不确定性的特征统计有助于对抗稳健性 cs.LG

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20583v1) [paper-pdf](http://arxiv.org/pdf/2503.20583v1)

**Authors**: Ran Wang, Xinlei Zhou, Rihao Li, Meng Hu, Wenhui Wu, Yuheng Jia

**Abstract**: Despite the remarkable success of deep neural networks (DNNs), the security threat of adversarial attacks poses a significant challenge to the reliability of DNNs. By introducing randomness into different parts of DNNs, stochastic methods can enable the model to learn some uncertainty, thereby improving model robustness efficiently. In this paper, we theoretically discover a universal phenomenon that adversarial attacks will shift the distributions of feature statistics. Motivated by this theoretical finding, we propose a robustness enhancement module called Feature Statistics with Uncertainty (FSU). It resamples channel-wise feature means and standard deviations of examples from multivariate Gaussian distributions, which helps to reconstruct the attacked examples and calibrate the shifted distributions. The calibration recovers some domain characteristics of the data for classification, thereby mitigating the influence of perturbations and weakening the ability of attacks to deceive models. The proposed FSU module has universal applicability in training, attacking, predicting and fine-tuning, demonstrating impressive robustness enhancement ability at trivial additional time cost. For example, against powerful optimization-based CW attacks, by incorporating FSU into attacking and predicting phases, it endows many collapsed state-of-the-art models with 50%-80% robust accuracy on CIFAR10, CIFAR100 and SVHN.

摘要: 尽管深度神经网络(DNN)取得了显著的成功，但敌意攻击带来的安全威胁对DNN的可靠性构成了巨大的挑战。通过将随机性引入到DNN的不同部分，随机方法可以使模型学习一些不确定性，从而有效地提高模型的稳健性。在本文中，我们从理论上发现了一个普遍现象，即对抗性攻击会改变特征统计量的分布。受这一理论发现的启发，我们提出了一种称为不确定性特征统计(FSU)的稳健性增强模块。它从多元高斯分布中重采样样本的通道特征均值和标准差，从而帮助重建被攻击的样本和校准移位分布。校准恢复了数据的某些领域特征用于分类，从而减轻了扰动的影响，削弱了攻击欺骗模型的能力。所提出的FSU模型在训练、攻击、预测和微调方面具有普遍的适用性，在很小的额外时间代价下表现出令人印象深刻的健壮性增强能力。例如，针对强大的基于优化的CW攻击，通过将FSU引入攻击和预测阶段，它赋予许多崩溃的最先进模型在CIFAR10、CIFAR100和SVHN上50%-80%的稳健准确率。



## **20. Aligning Visual Contrastive learning models via Preference Optimization**

通过偏好优化调整视觉对比学习模型 cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2411.08923v3) [paper-pdf](http://arxiv.org/pdf/2411.08923v3)

**Authors**: Amirabbas Afzali, Borna Khodabandeh, Ali Rasekh, Mahyar JafariNodeh, Sepehr kazemi, Simon Gottschalk

**Abstract**: Contrastive learning models have demonstrated impressive abilities to capture semantic similarities by aligning representations in the embedding space. However, their performance can be limited by the quality of the training data and its inherent biases. While Preference Optimization (PO) methods such as Reinforcement Learning from Human Feedback (RLHF) and Direct Preference Optimization (DPO) have been applied to align generative models with human preferences, their use in contrastive learning has yet to be explored. This paper introduces a novel method for training contrastive learning models using different PO methods to break down complex concepts. Our method systematically aligns model behavior with desired preferences, enhancing performance on the targeted task. In particular, we focus on enhancing model robustness against typographic attacks and inductive biases, commonly seen in contrastive vision-language models like CLIP. Our experiments demonstrate that models trained using PO outperform standard contrastive learning techniques while retaining their ability to handle adversarial challenges and maintain accuracy on other downstream tasks. This makes our method well-suited for tasks requiring fairness, robustness, and alignment with specific preferences. We evaluate our method for tackling typographic attacks on images and explore its ability to disentangle gender concepts and mitigate gender bias, showcasing the versatility of our approach.

摘要: 对比学习模型已经显示出令人印象深刻的能力，通过在嵌入空间中对齐表征来捕捉语义相似性。然而，它们的表现可能会受到训练数据质量及其固有偏差的限制。虽然偏好优化(PO)方法，如人类反馈强化学习(RLHF)和直接偏好优化(DPO)已被应用于使生成模型与人类偏好保持一致，但它们在对比学习中的应用还有待探索。本文介绍了一种训练对比学习模型的新方法，该方法使用不同的PO方法来分解复杂的概念。我们的方法系统地使模型行为与期望的偏好保持一致，从而提高目标任务的性能。特别是，我们专注于增强模型对排版攻击和归纳偏见的健壮性，这在对比视觉语言模型中很常见，如CLIP。我们的实验表明，使用PO训练的模型优于标准的对比学习技术，同时保持了它们处理对抗性挑战和在其他下游任务上保持准确性的能力。这使得我们的方法非常适合需要公平性、健壮性和与特定偏好一致的任务。我们评估了我们处理图像排版攻击的方法，并探索了它理清性别概念和减轻性别偏见的能力，展示了我们方法的多功能性。



## **21. Lipschitz Constant Meets Condition Number: Learning Robust and Compact Deep Neural Networks**

Lipschitz常数满足条件数：学习稳健且紧凑的深度神经网络 cs.LG

13 pages, 6 figures

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20454v1) [paper-pdf](http://arxiv.org/pdf/2503.20454v1)

**Authors**: Yangqi Feng, Shing-Ho J. Lin, Baoyuan Gao, Xian Wei

**Abstract**: Recent research has revealed that high compression of Deep Neural Networks (DNNs), e.g., massive pruning of the weight matrix of a DNN, leads to a severe drop in accuracy and susceptibility to adversarial attacks. Integration of network pruning into an adversarial training framework has been proposed to promote adversarial robustness. It has been observed that a highly pruned weight matrix tends to be ill-conditioned, i.e., increasing the condition number of the weight matrix. This phenomenon aggravates the vulnerability of a DNN to input noise. Although a highly pruned weight matrix is considered to be able to lower the upper bound of the local Lipschitz constant to tolerate large distortion, the ill-conditionedness of such a weight matrix results in a non-robust DNN model. To overcome this challenge, this work develops novel joint constraints to adjust the weight distribution of networks, namely, the Transformed Sparse Constraint joint with Condition Number Constraint (TSCNC), which copes with smoothing distribution and differentiable constraint functions to reduce condition number and thus avoid the ill-conditionedness of weight matrices. Furthermore, our theoretical analyses unveil the relevance between the condition number and the local Lipschitz constant of the weight matrix, namely, the sharply increasing condition number becomes the dominant factor that restricts the robustness of over-sparsified models. Extensive experiments are conducted on several public datasets, and the results show that the proposed constraints significantly improve the robustness of a DNN with high pruning rates.

摘要: 最近的研究表明，深度神经网络(DNN)的高度压缩，例如对DNN的权重矩阵进行大量剪枝，导致准确率和对对手攻击的敏感性严重下降。已提出将网络修剪整合到对抗性训练框架中以提高对抗性健壮性。已经观察到，高度剪枝的权重矩阵往往是病态的，即增加了权重矩阵的条件数。这种现象加剧了DNN对输入噪声的脆弱性。虽然高度剪枝的权重矩阵被认为能够降低局部Lipschitz常数的上界以容忍大的失真，但这样的权重矩阵的病态导致了非稳健的DNN模型。为了克服这一挑战，本文提出了一种新的联合约束来调整网络的权值分布，即变换稀疏约束与条件数约束(TSCNC)，它通过平滑分布和可微约束函数来减少条件数，从而避免了权值矩阵的病态。此外，我们的理论分析揭示了条件数与权重矩阵的局部Lipschitz常数之间的相关性，即急剧增加的条件数成为限制过稀疏模型稳健性的主要因素。在几个公共数据集上进行了大量的实验，结果表明，所提出的约束显著提高了具有高剪枝率的DNN的健壮性。



## **22. UnReference: analysis of the effect of spoofing on RTK reference stations for connected rovers**

UnReference：分析欺骗对已连接漫游者的TEK参考站的影响 cs.CR

To appear the the 2025 IEEE/ION Position, Navigation and Localization  Symposium

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20364v1) [paper-pdf](http://arxiv.org/pdf/2503.20364v1)

**Authors**: Marco Spanghero, Panos Papadimitratos

**Abstract**: Global Navigation Satellite Systems (GNSS) provide standalone precise navigation for a wide gamut of applications. Nevertheless, applications or systems such as unmanned vehicles (aerial or ground vehicles and surface vessels) generally require a much higher level of accuracy than those provided by standalone receivers. The most effective and economical way of achieving centimeter-level accuracy is to rely on corrections provided by fixed \emph{reference station} receivers to improve the satellite ranging measurements. Differential GNSS (DGNSS) and Real Time Kinematics (RTK) provide centimeter-level accuracy by distributing online correction streams to connected nearby mobile receivers typically termed \emph{rovers}. However, due to their static nature, reference stations are prime targets for GNSS attacks, both simplistic jamming and advanced spoofing, with different levels of adversarial control and complexity. Jamming the reference station would deny corrections and thus accuracy to the rovers. Spoofing the reference station would force it to distribute misleading corrections. As a result, all connected rovers using those corrections will be equally influenced by the adversary independently of their actual trajectory. We evaluate a battery of tests generated with an RF simulator to test the robustness of a common DGNSS/RTK processing library and receivers. We test both jamming and synchronized spoofing to demonstrate that adversarial action on the rover using reference spoofing is both effective and convenient from an adversarial perspective. Additionally, we discuss possible strategies based on existing countermeasures (self-validation of the PNT solution and monitoring of own clock drift) that the rover and the reference station can adopt to avoid using or distributing bogus corrections.

摘要: 全球导航卫星系统（GNSS）为广泛的应用提供独立的精确导航。然而，无人驾驶车辆（空中或地面车辆和水面船只）等应用或系统通常需要比独立接收器提供的准确度高得多的准确度。实现厘米级准确度的最有效和最经济的方法是依靠固定\{参考站}接收器提供的修正来改进卫星距离测量。差异GNSS（DGNSS）和实时运动学（TEK）通过将在线修正流分发到连接的附近移动接收器（通常称为\{rovers}）来提供厘米级的准确度。然而，由于其静态性质，参考站是全球导航卫星系统攻击的主要目标，包括简单化干扰和高级欺骗，具有不同水平的对抗控制和复杂性。干扰参考站将无法进行纠正，从而无法保证漫游车的准确性。欺骗参考站将迫使它发布误导性的更正。因此，所有使用这些修正的连接漫游者将平等地受到对手的影响，而与其实际轨迹无关。我们评估使用RF模拟器生成的一系列测试，以测试常见DGNSS/TEK处理库和接收器的稳健性。我们测试了干扰和同步欺骗，以证明从对抗的角度来看，使用参考欺骗对火星车进行对抗动作既有效又方便。此外，我们还讨论了基于现有对策（PNT解决方案的自我验证和监控自己的时钟漂移）的可能策略，漫游者和参考站可以采用这些策略来避免使用或分发虚假修正。



## **23. Enabling Heterogeneous Adversarial Transferability via Feature Permutation Attacks**

通过特征排列攻击实现异类对抗可移植性 cs.CV

PAKDD 2025. Main Track

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20310v1) [paper-pdf](http://arxiv.org/pdf/2503.20310v1)

**Authors**: Tao Wu, Tie Luo

**Abstract**: Adversarial attacks in black-box settings are highly practical, with transfer-based attacks being the most effective at generating adversarial examples (AEs) that transfer from surrogate models to unseen target models. However, their performance significantly degrades when transferring across heterogeneous architectures -- such as CNNs, MLPs, and Vision Transformers (ViTs) -- due to fundamental architectural differences. To address this, we propose Feature Permutation Attack (FPA), a zero-FLOP, parameter-free method that enhances adversarial transferability across diverse architectures. FPA introduces a novel feature permutation (FP) operation, which rearranges pixel values in selected feature maps to simulate long-range dependencies, effectively making CNNs behave more like ViTs and MLPs. This enhances feature diversity and improves transferability both across heterogeneous architectures and within homogeneous CNNs. Extensive evaluations on 14 state-of-the-art architectures show that FPA achieves maximum absolute gains in attack success rates of 7.68% on CNNs, 14.57% on ViTs, and 14.48% on MLPs, outperforming existing black-box attacks. Additionally, FPA is highly generalizable and can seamlessly integrate with other transfer-based attacks to further boost their performance. Our findings establish FPA as a robust, efficient, and computationally lightweight strategy for enhancing adversarial transferability across heterogeneous architectures.

摘要: 黑匣子环境中的对抗性攻击非常实用，基于转移的攻击在生成从代理模型转移到不可见目标模型的对抗性示例（AE）方面最有效。然而，由于基本的架构差异，当跨异类架构（例如CNN、MLP和Vision Transformers（ViT））传输时，它们的性能会显着下降。为了解决这个问题，我们提出了特征排列攻击（PFA），这是一种零FLOP、无参数方法，可以增强不同架构之间的对抗性可转移性。FTA引入了一种新颖的特征排列（FP）操作，该操作重新排列选定特征地图中的像素值以模拟长期依赖性，有效地使CNN的行为更像ViT和MLP。这增强了特征多样性，并提高了跨异构架构和同构CNN内的可移植性。对14种最先进的架构进行的广泛评估表明，FPA在CNN上的攻击成功率为7.68%，在ViT上为14.57%，在MLP上为14.48%，优于现有的黑盒攻击。此外，FPA具有高度的通用性，可以与其他基于传输的攻击无缝集成，以进一步提高其性能。我们的研究结果建立FPA作为一个强大的，高效的，计算轻量级的策略，以提高对抗跨异构体系结构的可移植性。



## **24. Are We There Yet? Unraveling the State-of-the-Art Graph Network Intrusion Detection Systems**

我们到了吗？图网络入侵检测系统的研究现状 cs.CR

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20281v1) [paper-pdf](http://arxiv.org/pdf/2503.20281v1)

**Authors**: Chenglong Wang, Pujia Zheng, Jiaping Gui, Cunqing Hua, Wajih Ul Hassan

**Abstract**: Network Intrusion Detection Systems (NIDS) are vital for ensuring enterprise security. Recently, Graph-based NIDS (GIDS) have attracted considerable attention because of their capability to effectively capture the complex relationships within the graph structures of data communications. Despite their promise, the reproducibility and replicability of these GIDS remain largely unexplored, posing challenges for developing reliable and robust detection systems. This study bridges this gap by designing a systematic approach to evaluate state-of-the-art GIDS, which includes critically assessing, extending, and clarifying the findings of these systems. We further assess the robustness of GIDS under adversarial attacks. Evaluations were conducted on three public datasets as well as a newly collected large-scale enterprise dataset. Our findings reveal significant performance discrepancies, highlighting challenges related to dataset scale, model inputs, and implementation settings. We demonstrate difficulties in reproducing and replicating results, particularly concerning false positive rates and robustness against adversarial attacks. This work provides valuable insights and recommendations for future research, emphasizing the importance of rigorous reproduction and replication studies in developing robust and generalizable GIDS solutions.

摘要: 网络入侵检测系统（NIDS）对于确保企业安全至关重要。近年来，基于图的网络入侵检测系统（GIDS）因其能够有效地捕捉数据通信的图结构中的复杂关系而引起了相当大的关注。尽管它们有希望，但这些GIDS的可重复性和可复制性在很大程度上仍然没有被探索，这给开发可靠和强大的检测系统带来了挑战。这项研究通过设计一种系统性的方法来评估最先进的GIDS，其中包括批判性地评估、扩展和澄清这些系统的调查结果，从而弥合了这一差距。我们进一步评估了GIDS在对抗攻击下的稳健性。对三个公共数据集以及新收集的大型企业数据集进行了评估。我们的研究结果揭示了显着的性能差异，凸显了与数据集规模、模型输入和实施设置相关的挑战。我们证明了复制和复制结果的困难，特别是在假阳性率和对抗性攻击的稳健性方面。这项工作为未来的研究提供了宝贵的见解和建议，强调了严格的复制和复制研究在开发稳健且可推广的GIDS解决方案中的重要性。



## **25. Hi-ALPS -- An Experimental Robustness Quantification of Six LiDAR-based Object Detection Systems for Autonomous Driving**

Hi-ALPS --六种基于LiDART的自动驾驶目标检测系统的实验鲁棒性量化 cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.17168v2) [paper-pdf](http://arxiv.org/pdf/2503.17168v2)

**Authors**: Alexandra Arzberger, Ramin Tavakoli Kolagari

**Abstract**: Light Detection and Ranging (LiDAR) is an essential sensor technology for autonomous driving as it can capture high-resolution 3D data. As 3D object detection systems (OD) can interpret such point cloud data, they play a key role in the driving decisions of autonomous vehicles. Consequently, such 3D OD must be robust against all types of perturbations and must therefore be extensively tested. One approach is the use of adversarial examples, which are small, sometimes sophisticated perturbations in the input data that change, i.e., falsify, the prediction of the OD. These perturbations are carefully designed based on the weaknesses of the OD. The robustness of the OD cannot be quantified with adversarial examples in general, because if the OD is vulnerable to a given attack, it is unclear whether this is due to the robustness of the OD or whether the attack algorithm produces particularly strong adversarial examples. The contribution of this work is Hi-ALPS -- Hierarchical Adversarial-example-based LiDAR Perturbation Level System, where higher robustness of the OD is required to withstand the perturbations as the perturbation levels increase. In doing so, the Hi-ALPS levels successively implement a heuristic followed by established adversarial example approaches. In a series of comprehensive experiments using Hi-ALPS, we quantify the robustness of six state-of-the-art 3D OD under different types of perturbations. The results of the experiments show that none of the OD is robust against all Hi-ALPS levels; an important factor for the ranking is that human observers can still correctly recognize the perturbed objects, as the respective perturbations are small. To increase the robustness of the OD, we discuss the applicability of state-of-the-art countermeasures. In addition, we derive further suggestions for countermeasures based on our experimental results.

摘要: 光探测和测距（LiDAR）是自动驾驶的重要传感器技术，因为它可以捕获高分辨率的3D数据。由于3D物体检测系统（OD）可以解释这些点云数据，因此它们在自动驾驶汽车的驾驶决策中发挥着关键作用。因此，这种3D OD必须对所有类型的扰动具有鲁棒性，因此必须进行广泛的测试。一种方法是使用对抗性示例，这些示例是输入数据中发生变化的小的，有时是复杂的扰动，即，”“假的，假的，假的。这些扰动是根据OD的弱点精心设计的。OD的鲁棒性通常不能用对抗性示例来量化，因为如果OD容易受到给定攻击，则不清楚这是由于OD的鲁棒性还是攻击算法产生特别强的对抗性示例。这项工作的贡献是Hi-ALPS -层次化的不利的例子为基础的激光雷达扰动水平系统，其中更高的鲁棒性的OD需要承受的扰动，随着扰动水平的增加。在这样做时，Hi-ALPS级别依次实现启发式，然后是建立的对抗性示例方法。在一系列使用Hi-ALPS的综合实验中，我们量化了六种最先进的3D OD在不同类型扰动下的鲁棒性。实验结果表明，没有一个OD对所有Hi-ALPS水平都是鲁棒的;排名的一个重要因素是，人类观察者仍然可以正确地识别扰动对象，因为相应的扰动很小。为了增加OD的鲁棒性，我们讨论了最先进的对策的适用性。此外，我们还根据实验结果得出了进一步的对策建议。



## **26. How Secure is Forgetting? Linking Machine Unlearning to Machine Learning Attacks**

忘记有多安全？将机器取消学习与机器学习攻击联系起来 cs.CR

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20257v1) [paper-pdf](http://arxiv.org/pdf/2503.20257v1)

**Authors**: Muhammed Shafi K. P., Serena Nicolazzo, Antonino Nocera, Vinod P

**Abstract**: As Machine Learning (ML) evolves, the complexity and sophistication of security threats against this paradigm continue to grow as well, threatening data privacy and model integrity. In response, Machine Unlearning (MU) is a recent technology that aims to remove the influence of specific data from a trained model, enabling compliance with privacy regulations and user requests. This can be done for privacy compliance (e.g., GDPR's right to be forgotten) or model refinement. However, the intersection between classical threats in ML and MU remains largely unexplored. In this Systematization of Knowledge (SoK), we provide a structured analysis of security threats in ML and their implications for MU. We analyze four major attack classes, namely, Backdoor Attacks, Membership Inference Attacks (MIA), Adversarial Attacks, and Inversion Attacks, we investigate their impact on MU and propose a novel classification based on how they are usually used in this context. Finally, we identify open challenges, including ethical considerations, and explore promising future research directions, paving the way for future research in secure and privacy-preserving Machine Unlearning.

摘要: 随着机器学习（ML）的发展，针对这种范式的安全威胁的复杂性和复杂性也在不断增长，威胁着数据隐私和模型完整性。作为回应，机器非学习（MU）是一种最新的技术，旨在从训练模型中消除特定数据的影响，从而遵守隐私法规和用户请求。这可以为了隐私合规性（例如，GDPR的被遗忘权）或模型细化。然而，ML和MU中经典威胁之间的交叉点在很大程度上仍未被探索。在此知识系统化（SoK）中，我们对ML中的安全威胁及其对MU的影响进行了结构化分析。我们分析了四种主要的攻击类别，即后门攻击、会员推断攻击（MIA）、对抗性攻击和倒置攻击，我们调查了它们对MU的影响，并根据它们通常在这种情况下的使用方式提出了一种新颖的分类。最后，我们确定了开放的挑战，包括道德考虑，并探索有前途的未来研究方向，为未来在安全和保护隐私的机器非学习方面的研究铺平道路。



## **27. Defending against Backdoor Attack on Deep Neural Networks**

防御深度神经网络的后门攻击 cs.CR

This workshop manuscript is not a publication and will not be  published anywhere

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2002.12162v3) [paper-pdf](http://arxiv.org/pdf/2002.12162v3)

**Authors**: Hao Cheng, Kaidi Xu, Sijia Liu, Pin-Yu Chen, Pu Zhao, Xue Lin

**Abstract**: Although deep neural networks (DNNs) have achieved a great success in various computer vision tasks, it is recently found that they are vulnerable to adversarial attacks. In this paper, we focus on the so-called \textit{backdoor attack}, which injects a backdoor trigger to a small portion of training data (also known as data poisoning) such that the trained DNN induces misclassification while facing examples with this trigger. To be specific, we carefully study the effect of both real and synthetic backdoor attacks on the internal response of vanilla and backdoored DNNs through the lens of Gard-CAM. Moreover, we show that the backdoor attack induces a significant bias in neuron activation in terms of the $\ell_\infty$ norm of an activation map compared to its $\ell_1$ and $\ell_2$ norm. Spurred by our results, we propose the \textit{$\ell_\infty$-based neuron pruning} to remove the backdoor from the backdoored DNN. Experiments show that our method could effectively decrease the attack success rate, and also hold a high classification accuracy for clean images.

摘要: 尽管深度神经网络（DNN）在各种计算机视觉任务中取得了巨大成功，但最近发现它们很容易受到对抗攻击。在本文中，我们重点关注所谓的\texttit {后门攻击}，它向一小部分训练数据注入后门触发器（也称为数据中毒），使得训练后的DNN在面对具有该触发器的示例时会引发错误分类。具体来说，我们通过Gard-CAM的镜头仔细研究了真实和合成的后门攻击对香草和后门DNN内部反应的影响。此外，我们表明，后门攻击在激活地图的$\ell_\infty$规范方面引起了神经元激活的显着偏差，而其$\ell_1 $和$\ell_2 $规范。受我们的研究结果的启发，我们提出了\textit{$\ell_\infty$-based neuron pruning}来从后门DNN中移除后门。实验结果表明，该方法能有效降低攻击成功率，对干净图像也能保持较高的分类准确率。



## **28. Persistence of Backdoor-based Watermarks for Neural Networks: A Comprehensive Evaluation**

神经网络背景水印持久性的综合评价 cs.LG

Preprint. Under Review

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2501.02704v2) [paper-pdf](http://arxiv.org/pdf/2501.02704v2)

**Authors**: Anh Tu Ngo, Chuan Song Heng, Nandish Chattopadhyay, Anupam Chattopadhyay

**Abstract**: Deep Neural Networks (DNNs) have gained considerable traction in recent years due to the unparalleled results they gathered. However, the cost behind training such sophisticated models is resource intensive, resulting in many to consider DNNs to be intellectual property (IP) to model owners. In this era of cloud computing, high-performance DNNs are often deployed all over the internet so that people can access them publicly. As such, DNN watermarking schemes, especially backdoor-based watermarks, have been actively developed in recent years to preserve proprietary rights. Nonetheless, there lies much uncertainty on the robustness of existing backdoor watermark schemes, towards both adversarial attacks and unintended means such as fine-tuning neural network models. One reason for this is that no complete guarantee of robustness can be assured in the context of backdoor-based watermark. In this paper, we extensively evaluate the persistence of recent backdoor-based watermarks within neural networks in the scenario of fine-tuning, we propose/develop a novel data-driven idea to restore watermark after fine-tuning without exposing the trigger set. Our empirical results show that by solely introducing training data after fine-tuning, the watermark can be restored if model parameters do not shift dramatically during fine-tuning. Depending on the types of trigger samples used, trigger accuracy can be reinstated to up to 100%. Our study further explores how the restoration process works using loss landscape visualization, as well as the idea of introducing training data in fine-tuning stage to alleviate watermark vanishing.

摘要: 近年来，深度神经网络（DNN）由于其收集的无与伦比的结果而获得了相当大的吸引力。然而，训练这种复杂模型的成本是资源密集型的，导致许多人认为DNN是模型所有者的知识产权（IP）。在这个云计算时代，高性能DNN通常部署在整个互联网上，以便人们可以公开访问它们。因此，DNN水印方案，特别是基于后门的水印，近年来已经被积极开发以保护专有权利。尽管如此，现有后门水印方案对于对抗性攻击和微调神经网络模型等非预期手段的稳健性仍存在很大的不确定性。原因之一是，在基于后门的水印的背景下，无法完全保证稳健性。在本文中，我们广泛评估了神经网络中最近基于后门的水印在微调场景中的持久性，我们提出/开发了一种新颖的数据驱动思想，用于在微调后恢复水印，而不暴露触发集。我们的经验结果表明，通过在微调后仅引入训练数据，如果模型参数在微调期间没有发生显着变化，则可以恢复水印。根据所使用的触发样本类型，触发准确度可以恢复至高达100%。我们的研究进一步探索了恢复过程如何使用损失景观可视化，以及在微调阶段引入训练数据以减轻水印消失的想法。



## **29. Joint Task Offloading and User Scheduling in 5G MEC under Jamming Attacks**

干扰攻击下的5G MEC联合任务卸载和用户调度 cs.CR

6 pages, 5 figures, Accepted to IEEE International Conference in  Communications (ICC) 2025

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2501.13227v2) [paper-pdf](http://arxiv.org/pdf/2501.13227v2)

**Authors**: Mohammadreza Amini, Burak Kantarci, Claude D'Amours, Melike Erol-Kantarci

**Abstract**: In this paper, we propose a novel joint task offloading and user scheduling (JTO-US) framework for 5G mobile edge computing (MEC) systems under security threats from jamming attacks. The goal is to minimize the delay and the ratio of dropped tasks, taking into account both communication and computation delays. The system model includes a 5G network equipped with MEC servers and an adversarial on-off jammer that disrupts communication. The proposed framework optimally schedules tasks and users to minimize the impact of jamming while ensuring that high-priority tasks are processed efficiently. Genetic algorithm (GA) is used to solve the optimization problem, and the results are compared with benchmark methods such as GA without considering jamming effect, Shortest Job First (SJF), and Shortest Deadline First (SDF). The simulation results demonstrate that the proposed JTO-US framework achieves the lowest drop ratio and effectively manages priority tasks, outperforming existing methods. Particularly, when the jamming probability is 0.8, the proposed framework mitigates the jammer's impact by reducing the drop ratio to 63%, compared to 89% achieved by the next best method.

摘要: 针对5G移动边缘计算(MEC)系统面临干扰攻击的安全威胁，提出了一种新的联合任务卸载和用户调度(JTO-US)框架。目标是最小化延迟和丢弃任务的比率，同时考虑通信和计算延迟。该系统模型包括一个配备MEC服务器的5G网络和一个会中断通信的对抗性开关干扰器。该框架对任务和用户进行优化调度，在保证高优先级任务得到有效处理的同时，将干扰的影响降至最低。采用遗传算法(GA)对优化问题进行求解，并与不考虑干扰影响的GA、最短作业优先(SJF)、最短截止时间优先(SDF)等基准算法进行了比较。仿真结果表明，JTO-US框架实现了最低的丢包率，有效地管理了优先级任务，优于已有的方法。特别是，当干扰概率为0.8时，所提出的框架通过将丢包率降低到63%来减轻干扰的影响，而次优方法的丢包率为89%。



## **30. Exploring Adversarial Threat Models in Cyber Physical Battery Systems**

探索网络物理电池系统中的对抗威胁模型 eess.SY

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2401.13801v2) [paper-pdf](http://arxiv.org/pdf/2401.13801v2)

**Authors**: Shanthan Kumar Padisala, Shashank Dhananjay Vyas, Satadru Dey

**Abstract**: Technological advancements like the Internet of Things (IoT) have facilitated data exchange across various platforms. This data exchange across various platforms has transformed the traditional battery system into a cyber physical system. Such connectivity makes modern cyber physical battery systems vulnerable to cyber threats where a cyber attacker can manipulate sensing and actuation signals to bring the battery system into an unsafe operating condition. Hence, it is essential to build resilience in modern cyber physical battery systems (CPBS) under cyber attacks. The first step of building such resilience is to analyze potential adversarial behavior, that is, how the adversaries can inject attacks into the battery systems. However, it has been found that in this under-explored area of battery cyber physical security, such an adversarial threat model has not been studied in a systematic manner. In this study, we address this gap and explore adversarial attack generation policies based on optimal control framework. The framework is developed by performing theoretical analysis, which is subsequently supported by evaluation with experimental data generated from a commercial battery cell.

摘要: 物联网（IoT）等技术进步促进了各种平台之间的数据交换。这种跨各种平台的数据交换已将传统的电池系统转变为网络物理系统。这种连接性使现代网络物理电池系统容易受到网络威胁的影响，网络攻击者可以操纵传感和驱动信号，使电池系统进入不安全的操作状态。因此，在网络攻击下建立现代网络物理电池系统（CPBS）的弹性至关重要。建立这种弹性的第一步是分析潜在的对抗行为，即对手如何将攻击注入电池系统。然而，人们发现，在这个尚未充分探索的电池网络物理安全领域，这种对抗性威胁模型尚未得到系统的研究。在这项研究中，我们解决了这一差距，并探索了基于最佳控制框架的对抗性攻击生成策略。该框架是通过执行理论分析来开发的，随后通过使用商用电池单元生成的实验数据的评估来支持该框架。



## **31. Bitstream Collisions in Neural Image Compression via Adversarial Perturbations**

基于对抗扰动的神经图像压缩中的比特流冲突 cs.CR

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19817v1) [paper-pdf](http://arxiv.org/pdf/2503.19817v1)

**Authors**: Jordan Madden, Lhamo Dorje, Xiaohua Li

**Abstract**: Neural image compression (NIC) has emerged as a promising alternative to classical compression techniques, offering improved compression ratios. Despite its progress towards standardization and practical deployment, there has been minimal exploration into it's robustness and security. This study reveals an unexpected vulnerability in NIC - bitstream collisions - where semantically different images produce identical compressed bitstreams. Utilizing a novel whitebox adversarial attack algorithm, this paper demonstrates that adding carefully crafted perturbations to semantically different images can cause their compressed bitstreams to collide exactly. The collision vulnerability poses a threat to the practical usability of NIC, particularly in security-critical applications. The cause of the collision is analyzed, and a simple yet effective mitigation method is presented.

摘要: 神经图像压缩（NIC）已成为一个有前途的替代经典压缩技术，提供更高的压缩比。尽管它在标准化和实际部署方面取得了进展，但对其健壮性和安全性的探索却很少。这项研究揭示了一个意想不到的漏洞在NIC -比特流冲突-语义不同的图像产生相同的压缩比特流。利用一种新的白盒对抗攻击算法，本文证明了向语义不同的图像添加精心制作的扰动可以导致它们的压缩比特流准确地碰撞。碰撞漏洞对NIC的实际可用性构成威胁，特别是在安全关键应用程序中。分析了碰撞原因，并提出了一种简单有效的缓解方法。



## **32. SITA: Structurally Imperceptible and Transferable Adversarial Attacks for Stylized Image Generation**

SITA：用于风格化图像生成的结构上不可感知且可转移的对抗攻击 cs.CV

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19791v1) [paper-pdf](http://arxiv.org/pdf/2503.19791v1)

**Authors**: Jingdan Kang, Haoxin Yang, Yan Cai, Huaidong Zhang, Xuemiao Xu, Yong Du, Shengfeng He

**Abstract**: Image generation technology has brought significant advancements across various fields but has also raised concerns about data misuse and potential rights infringements, particularly with respect to creating visual artworks. Current methods aimed at safeguarding artworks often employ adversarial attacks. However, these methods face challenges such as poor transferability, high computational costs, and the introduction of noticeable noise, which compromises the aesthetic quality of the original artwork. To address these limitations, we propose a Structurally Imperceptible and Transferable Adversarial (SITA) attacks. SITA leverages a CLIP-based destylization loss, which decouples and disrupts the robust style representation of the image. This disruption hinders style extraction during stylized image generation, thereby impairing the overall stylization process. Importantly, SITA eliminates the need for a surrogate diffusion model, leading to significantly reduced computational overhead. The method's robust style feature disruption ensures high transferability across diverse models. Moreover, SITA introduces perturbations by embedding noise within the imperceptible structural details of the image. This approach effectively protects against style extraction without compromising the visual quality of the artwork. Extensive experiments demonstrate that SITA offers superior protection for artworks against unauthorized use in stylized generation. It significantly outperforms existing methods in terms of transferability, computational efficiency, and noise imperceptibility. Code is available at https://github.com/A-raniy-day/SITA.

摘要: 图像生成技术在各个领域带来了重大进步，但也引起了人们对数据滥用和潜在权利侵犯的担忧，特别是在创作视觉艺术作品方面。目前旨在保护艺术品的方法往往采用对抗性攻击。然而，这些方法面临着可移植性差、计算成本高以及引入明显的噪声等挑战，这损害了原始艺术品的美学质量。为了解决这些局限性，我们提出了一种结构上不可感知和可转移的对抗性攻击(SITA)。SITA利用了基于剪辑的去风格化损失，这分离并破坏了图像的健壮样式表示。这种干扰阻碍了风格化图像生成期间的样式提取，从而损害了整个风格化过程。重要的是，SITA消除了对代理扩散模型的需要，从而显著减少了计算开销。该方法稳健的风格特征破坏确保了在不同模型之间的高度可转移性。此外，SITA通过在图像的不可察觉的结构细节中嵌入噪声来引入扰动。这种方法有效地防止了样式提取，而不会影响图稿的视觉质量。广泛的实验表明，SITA为艺术品提供了卓越的保护，防止在风格化生成中未经授权使用。它在可转移性、计算效率和噪声不可见性方面明显优于现有方法。代码可在https://github.com/A-raniy-day/SITA.上找到



## **33. Lifting Linear Sketches: Optimal Bounds and Adversarial Robustness**

提升线性草图：最佳边界和对抗稳健性 cs.DS

To appear in STOC 2025

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19629v1) [paper-pdf](http://arxiv.org/pdf/2503.19629v1)

**Authors**: Elena Gribelyuk, Honghao Lin, David P. Woodruff, Huacheng Yu, Samson Zhou

**Abstract**: We introduce a novel technique for ``lifting'' dimension lower bounds for linear sketches in the real-valued setting to dimension lower bounds for linear sketches with polynomially-bounded integer entries when the input is a polynomially-bounded integer vector. Using this technique, we obtain the first optimal sketching lower bounds for discrete inputs in a data stream, for classical problems such as approximating the frequency moments, estimating the operator norm, and compressed sensing. Additionally, we lift the adaptive attack of Hardt and Woodruff (STOC, 2013) for breaking any real-valued linear sketch via a sequence of real-valued queries, and show how to obtain an attack on any integer-valued linear sketch using integer-valued queries. This shows that there is no linear sketch in a data stream with insertions and deletions that is adversarially robust for approximating any $L_p$ norm of the input, resolving a central open question for adversarially robust streaming algorithms. To do so, we introduce a new pre-processing technique of independent interest which, given an integer-valued linear sketch, increases the dimension of the sketch by only a constant factor in order to make the orthogonal lattice to its row span smooth. This pre-processing then enables us to leverage results in lattice theory on discrete Gaussian distributions and reason that efficient discrete sketches imply efficient continuous sketches. Our work resolves open questions from the Banff '14 and '17 workshops on Communication Complexity and Applications, as well as the STOC '21 and FOCS '23 workshops on adaptivity and robustness.

摘要: 当输入是多项式有界的整数向量时，我们引入了一种新的技术来提升实值环境下的线性草图的维度下界，以此来标注具有多项式有界的整数项的线性草图的维度下界。利用这一技术，我们得到了数据流中离散输入的第一个最优草图下界，用于近似频率矩、估计算子范数和压缩传感等经典问题。此外，我们解除了Hardt和Woodruff(STOEC，2013)通过实值查询序列破坏任何实值线性草图的自适应攻击，并展示了如何使用整值查询来获得对任何整值线性草图的攻击。这表明在具有插入和删除的数据流中不存在对于近似输入的任何$L_p$范数具有相反健壮性的线性草图，从而解决了相反健壮性流算法的一个中心未决问题。为此，我们引入了一种新的独立感兴趣的预处理技术，该技术在给定整数值的线性草图的情况下，仅将草图的维度增加一个恒定因子，以使正交格的行跨度光滑。然后，这种预处理使我们能够利用离散高斯分布的格子理论中的结果，并推断高效的离散草图意味着高效的连续草图。我们的工作解决了Banff‘14和’17关于通信复杂性和应用的研讨会，以及STEC‘21和FOCS’23关于适应性和稳健性的研讨会的未决问题。



## **34. Semantic Entanglement-Based Ransomware Detection via Probabilistic Latent Encryption Mapping**

通过概率潜在加密映射的基于语义纠缠的勒索软件检测 cs.CR

arXiv admin note: This paper has been withdrawn by arXiv due to  disputed and unverifiable authorship

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2502.02730v2) [paper-pdf](http://arxiv.org/pdf/2502.02730v2)

**Authors**: Mohammad Eisa, Quentin Yardley, Rafael Witherspoon, Harriet Pendlebury, Clement Rutherford

**Abstract**: Encryption-based attacks have introduced significant challenges for detection mechanisms that rely on predefined signatures, heuristic indicators, or static rule-based classifications. Probabilistic Latent Encryption Mapping presents an alternative detection framework that models ransomware-induced encryption behaviors through statistical representations of entropy deviations and probabilistic dependencies in execution traces. Unlike conventional approaches that depend on explicit bytecode analysis or predefined cryptographic function call monitoring, probabilistic inference techniques classify encryption anomalies based on their underlying statistical characteristics, ensuring greater adaptability to polymorphic attack strategies. Evaluations demonstrate that entropy-driven classification reduces false positive rates while maintaining high detection accuracy across diverse ransomware families and encryption methodologies. Experimental results further highlight the framework's ability to differentiate between benign encryption workflows and adversarial cryptographic manipulations, ensuring that classification performance remains effective across cloud-based and localized execution environments. Benchmark comparisons illustrate that probabilistic modeling exhibits advantages over heuristic and machine learning-based detection approaches, particularly in handling previously unseen encryption techniques and adversarial obfuscation strategies. Computational efficiency analysis confirms that detection latency remains within operational feasibility constraints, reinforcing the viability of probabilistic encryption classification for real-time security infrastructures. The ability to systematically infer encryption-induced deviations without requiring static attack signatures strengthens detection robustness against adversarial evasion techniques.

摘要: 基于加密的攻击给依赖于预定义签名、启发式指示符或静态基于规则的分类的检测机制带来了重大挑战。概率潜伏加密映射提供了一种替代检测框架，该框架通过执行痕迹中的执行偏差和概率依赖性的统计表示来对勒索软件引发的加密行为进行建模。与依赖于显式字节码分析或预定义加密函数调用监控的传统方法不同，概率推理技术根据加密异常的底层统计特征对加密异常进行分类，确保对多态攻击策略的更大适应性。评估表明，信息量驱动的分类可以降低假阳性率，同时在不同勒索软件系列和加密方法中保持高检测准确性。实验结果进一步凸显了该框架区分良性加密工作流程和对抗性加密操作的能力，确保分类性能在基于云的和本地化的执行环境中保持有效。基准比较表明，概率建模比启发式和基于机器学习的检测方法具有优势，特别是在处理以前未见过的加密技术和对抗性混淆策略方面。计算效率分析证实，检测延迟仍在操作可行性约束范围内，增强了实时安全基础设施概率加密分类的可行性。无需静态攻击签名即可系统地推断识别引起的偏差的能力增强了针对对抗性规避技术的检测鲁棒性。



## **35. Nanopass Back-Translation of Call-Return Trees for Mechanized Secure Compilation Proofs**

用于机械化安全编译证明的调用返回树的Nanopass反向翻译 cs.PL

ITP'25 submission

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19609v1) [paper-pdf](http://arxiv.org/pdf/2503.19609v1)

**Authors**: Jérémy Thibault, Joseph Lenormand, Catalin Hritcu

**Abstract**: Researchers aim to build secure compilation chains enforcing that if there is no attack a source context can mount against a source program then there is also no attack an adversarial target context can mount against the compiled program. Proving that these compilation chains are secure is, however, challenging, and involves a non-trivial back-translation step: for any attack a target context mounts against the compiled program one has to exhibit a source context mounting the same attack against the source program. We describe a novel back-translation technique, which results in simpler proofs that can be more easily mechanized in a proof assistant. Given a finite set of finite trace prefixes, capturing the interaction recorded during an attack between a target context and the compiled program, we build a call-return tree that we back-translate into a source context producing the same trace prefixes. We use state in the generated source context to record the current location in the call-return tree. The back-translation is done in several small steps, each adding to the tree new information describing how the location should change depending on how the context regains control. To prove this back-translation correct we give semantics to every intermediate call-return tree language, using ghost state to store information and explicitly enforce execution invariants. We prove several small forward simulations, basically seeing the back-translation as a verified nanopass compiler. Thanks to this modular structure, we are able to mechanize this complex back-translation and its correctness proof in the Rocq prover without too much effort.

摘要: 研究人员的目标是构建安全的编译链，强制执行如果源上下文不会对源程序发动攻击，那么敌意目标上下文也不会对编译后的程序发动攻击。然而，证明这些编译链是安全的是具有挑战性的，并且涉及到不平凡的回译步骤：对于任何针对编译程序的目标上下文挂载的攻击，必须展示对源程序挂载相同攻击的源上下文。我们描述了一种新的反向翻译技术，它导致了更简单的证明，可以在证明助手中更容易地机械化。给定一组有限的跟踪前缀，捕获攻击期间目标上下文和编译程序之间记录的交互，我们构建调用返回树，并将其反向转换为产生相同跟踪前缀的源上下文。我们在生成的源上下文中使用状态来记录调用返回树中的当前位置。反向转换分几个小步骤完成，每个步骤都向树中添加新的信息，描述位置应该如何改变，这取决于上下文如何重新控制。为了证明这种回译的正确性，我们给出了每一种中间调用-返回树语言的语义，使用重影状态来存储信息，并显式地强制执行不变量。我们证明了几个小的正向模拟，基本上将反向翻译视为经过验证的Nanopass编译器。由于这种模块化结构，我们能够机械化这种复杂的反向翻译及其在Rocq证明器中的正确性证明，而不需要太多的努力。



## **36. Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?**

LLM的安全培训是否适用于语义相关的自然知识？ cs.CL

Accepted in ICLR 2025

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2412.03235v2) [paper-pdf](http://arxiv.org/pdf/2412.03235v2)

**Authors**: Sravanti Addepalli, Yerram Varun, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.

摘要: 众所周知，大型语言模型（LLM）很容易受到精心设计的对抗攻击或越狱，从而导致生成令人反感的内容，尽管它们使用安全微调方法与人类偏好保持一致。虽然输入令牌空间的大维度使得不可避免地找到可以越狱这些模型的对抗性提示，但我们的目标是评估安全性微调的LLM对于与有毒种子提示在语义上相关的自然提示是否安全，这些提示会在对齐后引发安全响应。我们惊讶地发现，像GPT-4这样的流行对齐LLM可能会使用天真的提示而受到损害，这些提示甚至不是为了越狱模型的目标而精心设计的。此外，我们经验表明，给定一个从未对齐的模型中引发有毒反应的种子提示，人们可以系统地生成几个语义相关的自然提示，这些自然提示可以越狱对齐的LLM。为此，我们提出了一种响应引导问题增强（ReG-QA）方法来评估安全性一致的LLM对自然提示的概括，该方法首先使用未对齐的LLM（Q到A）在给定种子问题的情况下生成几个有毒答案，并进一步利用LLM生成可能产生这些答案（A到Q）的问题。有趣的是，我们发现，GPT-4 o等安全微调的LLM很容易从不安全内容中产生自然越狱问题（无需否认），因此可以用于后一步（A到Q）。我们获得的攻击成功率与JailbreakBench排行榜上领先的对抗性攻击方法相当/更好，同时对Smooth-LLM和Synonym Substitution等防御措施显着更加稳定，这些防御措施对排行榜上现有的所有攻击都有效。



## **37. Towards LLM Unlearning Resilient to Relearning Attacks: A Sharpness-Aware Minimization Perspective and Beyond**

迈向LLM摆脱学习对重新学习攻击的弹性：敏锐意识的最小化视角及超越 cs.LG

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2502.05374v3) [paper-pdf](http://arxiv.org/pdf/2502.05374v3)

**Authors**: Chongyu Fan, Jinghan Jia, Yihua Zhang, Anil Ramakrishna, Mingyi Hong, Sijia Liu

**Abstract**: The LLM unlearning technique has recently been introduced to comply with data regulations and address the safety and ethical concerns of LLMs by removing the undesired data-model influence. However, state-of-the-art unlearning methods face a critical vulnerability: they are susceptible to ``relearning'' the removed information from a small number of forget data points, known as relearning attacks. In this paper, we systematically investigate how to make unlearned models robust against such attacks. For the first time, we establish a connection between robust unlearning and sharpness-aware minimization (SAM) through a unified robust optimization framework, in an analogy to adversarial training designed to defend against adversarial attacks. Our analysis for SAM reveals that smoothness optimization plays a pivotal role in mitigating relearning attacks. Thus, we further explore diverse smoothing strategies to enhance unlearning robustness. Extensive experiments on benchmark datasets, including WMDP and MUSE, demonstrate that SAM and other smoothness optimization approaches consistently improve the resistance of LLM unlearning to relearning attacks. Notably, smoothness-enhanced unlearning also helps defend against (input-level) jailbreaking attacks, broadening our proposal's impact in robustifying LLM unlearning. Codes are available at https://github.com/OPTML-Group/Unlearn-Smooth.

摘要: 最近引入了LLM解除学习技术，以遵守数据法规，并通过消除不希望看到的数据模型影响来解决LLM的安全和伦理问题。然而，最先进的遗忘方法面临着一个严重的漏洞：它们容易受到从少数忘记数据点移除的信息的“重新学习”，称为重新学习攻击。在本文中，我们系统地研究了如何使未学习模型对此类攻击具有健壮性。第一次，我们通过一个统一的稳健优化框架在稳健遗忘和敏锐度感知最小化(SAM)之间建立了联系，类似于旨在防御对手攻击的对抗性训练。我们对SAM的分析表明，平滑优化在减轻再学习攻击方面起着关键作用。因此，我们进一步探索不同的平滑策略来增强遗忘的稳健性。在WMDP和MUSE等基准数据集上的大量实验表明，SAM和其他平滑优化方法一致地提高了LLM遗忘对重新学习攻击的抵抗力。值得注意的是，流畅性增强的遗忘也有助于防御(输入级)越狱攻击，扩大了我们的提议在强化LLM遗忘方面的影响。有关代码，请访问https://github.com/OPTML-Group/Unlearn-Smooth.



## **38. Boosting the Transferability of Audio Adversarial Examples with Acoustic Representation Optimization**

通过声学表示优化提高音频对抗示例的可移植性 cs.SD

Accepted to ICME 2025

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19591v1) [paper-pdf](http://arxiv.org/pdf/2503.19591v1)

**Authors**: Weifei Jin, Junjie Su, Hejia Wang, Yulin Ye, Jie Hao

**Abstract**: With the widespread application of automatic speech recognition (ASR) systems, their vulnerability to adversarial attacks has been extensively studied. However, most existing adversarial examples are generated on specific individual models, resulting in a lack of transferability. In real-world scenarios, attackers often cannot access detailed information about the target model, making query-based attacks unfeasible. To address this challenge, we propose a technique called Acoustic Representation Optimization that aligns adversarial perturbations with low-level acoustic characteristics derived from speech representation models. Rather than relying on model-specific, higher-layer abstractions, our approach leverages fundamental acoustic representations that remain consistent across diverse ASR architectures. By enforcing an acoustic representation loss to guide perturbations toward these robust, lower-level representations, we enhance the cross-model transferability of adversarial examples without degrading audio quality. Our method is plug-and-play and can be integrated with any existing attack methods. We evaluate our approach on three modern ASR models, and the experimental results demonstrate that our method significantly improves the transferability of adversarial examples generated by previous methods while preserving the audio quality.

摘要: 随着自动语音识别(ASR)系统的广泛应用，人们对其对抗攻击的脆弱性进行了广泛的研究。然而，现有的大多数对抗性例子都是在特定的个体模型上生成的，导致缺乏可转移性。在现实世界的场景中，攻击者通常无法访问有关目标模型的详细信息，这使得基于查询的攻击不可行。为了应对这一挑战，我们提出了一种称为声学表示优化的技术，该技术将对抗性扰动与来自语音表示模型的低级声学特征对齐。我们的方法不依赖于特定于模型的高层抽象，而是利用在不同ASR体系结构中保持一致的基本声学表示。通过强制实施声学表示损失来引导扰动朝向这些健壮的、较低级别的表示，我们在不降低音频质量的情况下增强了对抗性例子的跨模型可转移性。我们的方法是即插即用的，可以与任何现有的攻击方法集成。我们在三个现代ASR模型上对我们的方法进行了评估，实验结果表明，我们的方法在保持音频质量的同时，显著地提高了由先前方法生成的对抗性样本的可转移性。



## **39. Towards Imperceptible Adversarial Attacks for Time Series Classification with Local Perturbations and Frequency Analysis**

基于局部扰动和频率分析的时间序列分类的不可感知对抗攻击 cs.CR

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19519v1) [paper-pdf](http://arxiv.org/pdf/2503.19519v1)

**Authors**: Wenwei Gu, Renyi Zhong, Jianping Zhang, Michael R. Lyu

**Abstract**: Adversarial attacks in time series classification (TSC) models have recently gained attention due to their potential to compromise model robustness. Imperceptibility is crucial, as adversarial examples detected by the human vision system (HVS) can render attacks ineffective. Many existing methods fail to produce high-quality imperceptible examples, often generating perturbations with more perceptible low-frequency components, like square waves, and global perturbations that reduce stealthiness. This paper aims to improve the imperceptibility of adversarial attacks on TSC models by addressing frequency components and time series locality. We propose the Shapelet-based Frequency-domain Attack (SFAttack), which uses local perturbations focused on time series shapelets to enhance discriminative information and stealthiness. Additionally, we introduce a low-frequency constraint to confine perturbations to high-frequency components, enhancing imperceptibility.

摘要: 时间序列分类（TSC）模型中的对抗性攻击最近受到关注，因为它们可能会损害模型的鲁棒性。不可感知性至关重要，因为人类视觉系统（HVS）检测到的对抗性示例可能会使攻击无效。许多现有的方法无法产生高质量的不可感知的示例，通常会产生具有更可感知的低频分量（如方波）的扰动，以及降低隐身性的全局扰动。本文旨在通过解决频率分量和时间序列局部性来提高TSC模型上对抗性攻击的不可感知性。我们提出了基于Shapelet的频域攻击（SFAttack），它使用专注于时间序列形状的局部扰动来增强区分信息和隐蔽性。此外，我们引入了一个低频约束，将扰动限制在高频分量，增强了不可感知性。



## **40. Using Anomaly Detection to Detect Poisoning Attacks in Federated Learning Applications**

使用异常检测检测联邦学习应用程序中的中毒攻击 cs.LG

We will updated this article soon

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2207.08486v4) [paper-pdf](http://arxiv.org/pdf/2207.08486v4)

**Authors**: Ali Raza, Shujun Li, Kim-Phuc Tran, Ludovic Koehl, Kim Duc Tran

**Abstract**: Adversarial attacks such as poisoning attacks have attracted the attention of many machine learning researchers. Traditionally, poisoning attacks attempt to inject adversarial training data in order to manipulate the trained model. In federated learning (FL), data poisoning attacks can be generalized to model poisoning attacks, which cannot be detected by simpler methods due to the lack of access to local training data by the detector. State-of-the-art poisoning attack detection methods for FL have various weaknesses, e.g., the number of attackers has to be known or not high enough, working with i.i.d. data only, and high computational complexity. To overcome above weaknesses, we propose a novel framework for detecting poisoning attacks in FL, which employs a reference model based on a public dataset and an auditor model to detect malicious updates. We implemented a detector based on the proposed framework and using a one-class support vector machine (OC-SVM), which reaches the lowest possible computational complexity O(K) where K is the number of clients. We evaluated our detector's performance against state-of-the-art (SOTA) poisoning attacks for two typical applications of FL: electrocardiograph (ECG) classification and human activity recognition (HAR). Our experimental results validated the performance of our detector over other SOTA detection methods.

摘要: 中毒攻击等对抗性攻击引起了许多机器学习研究人员的关注。传统上，中毒攻击试图注入对抗性的训练数据，以操纵训练的模型。在联邦学习中，数据中毒攻击可以被概括为模型中毒攻击，但由于检测器无法访问本地训练数据，因此无法用更简单的方法检测到中毒攻击。目前针对FL的中毒攻击检测方法有很多缺点，例如，攻击者的数量必须已知或不够高，与I.I.D.配合使用。仅限数据，且计算复杂性高。为了克服上述缺陷，我们提出了一种新的FL中毒攻击检测框架，该框架使用基于公共数据集的参考模型和审计者模型来检测恶意更新。我们基于提出的框架实现了一个检测器，并使用了单类支持向量机(OC-SVM)，它达到了最低的计算复杂度O(K)，其中K是客户端的数量。我们针对FL的两个典型应用：心电图分类和人类活动识别(HAR)，评估了我们的检测器对最先进的(SOTA)中毒攻击的性能。我们的实验结果验证了我们的检测器相对于其他SOTA检测方法的性能。



## **41. NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary**

NoPain：通过最佳传输奇异边界进行无箱点云攻击 cs.CV

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.00063v3) [paper-pdf](http://arxiv.org/pdf/2503.00063v3)

**Authors**: Zezeng Li, Xiaoyu Du, Na Lei, Liming Chen, Weimin Wang

**Abstract**: Adversarial attacks exploit the vulnerability of deep models against adversarial samples. Existing point cloud attackers are tailored to specific models, iteratively optimizing perturbations based on gradients in either a white-box or black-box setting. Despite their promising attack performance, they often struggle to produce transferable adversarial samples due to overfitting the specific parameters of surrogate models. To overcome this issue, we shift our focus to the data distribution itself and introduce a novel approach named NoPain, which employs optimal transport (OT) to identify the inherent singular boundaries of the data manifold for cross-network point cloud attacks. Specifically, we first calculate the OT mapping from noise to the target feature space, then identify singular boundaries by locating non-differentiable positions. Finally, we sample along singular boundaries to generate adversarial point clouds. Once the singular boundaries are determined, NoPain can efficiently produce adversarial samples without the need of iterative updates or guidance from the surrogate classifiers. Extensive experiments demonstrate that the proposed end-to-end method outperforms baseline approaches in terms of both transferability and efficiency, while also maintaining notable advantages even against defense strategies. Code and model are available at https://github.com/cognaclee/nopain

摘要: 对抗性攻击利用深度模型针对对抗性样本的脆弱性。现有的点云攻击者是为特定模型量身定做的，基于白盒或黑盒设置中的渐变迭代优化扰动。尽管它们的攻击性能很有希望，但由于代理模型的特定参数过高，它们经常难以产生可转移的对抗性样本。为了解决这一问题，我们将重点转移到数据分布本身，并引入了一种名为NoPain的新方法，该方法使用最优传输(OT)来识别跨网络点云攻击数据流形的固有奇异边界。具体地，我们首先计算噪声到目标特征空间的OT映射，然后通过定位不可微位置来识别奇异边界。最后，我们沿着奇异边界进行采样以生成对抗性点云。一旦确定了奇异边界，NoPain就可以有效地产生对抗性样本，而不需要迭代更新或来自代理分类器的指导。大量的实验表明，端到端方法在可转移性和效率方面都优于基线方法，同时在与防御策略相比也保持了显著的优势。代码和型号可在https://github.com/cognaclee/nopain上找到



## **42. Improving Transferable Targeted Attacks with Feature Tuning Mixup**

通过功能调整Mixup改进可转移有针对性的攻击 cs.CV

CVPR 2025

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2411.15553v2) [paper-pdf](http://arxiv.org/pdf/2411.15553v2)

**Authors**: Kaisheng Liang, Xuelong Dai, Yanjie Li, Dong Wang, Bin Xiao

**Abstract**: Deep neural networks (DNNs) exhibit vulnerability to adversarial examples that can transfer across different DNN models. A particularly challenging problem is developing transferable targeted attacks that can mislead DNN models into predicting specific target classes. While various methods have been proposed to enhance attack transferability, they often incur substantial computational costs while yielding limited improvements. Recent clean feature mixup methods use random clean features to perturb the feature space but lack optimization for disrupting adversarial examples, overlooking the advantages of attack-specific perturbations. In this paper, we propose Feature Tuning Mixup (FTM), a novel method that enhances targeted attack transferability by combining both random and optimized noises in the feature space. FTM introduces learnable feature perturbations and employs an efficient stochastic update strategy for optimization. These learnable perturbations facilitate the generation of more robust adversarial examples with improved transferability. We further demonstrate that attack performance can be enhanced through an ensemble of multiple FTM-perturbed surrogate models. Extensive experiments on the ImageNet-compatible dataset across various DNN models demonstrate that our method achieves significant improvements over state-of-the-art methods while maintaining low computational cost.

摘要: 深度神经网络（DNN）对可以在不同DNN模型之间传输的对抗性示例表现出脆弱性。一个特别具有挑战性的问题是开发可转移的有针对性的攻击，这些攻击可能会误导DNN模型预测特定的目标类别。虽然已经提出了各种方法来增强攻击的可转移性，但它们通常会产生大量的计算成本，同时产生有限的改进。最近的清洁特征混合方法使用随机清洁特征来扰动特征空间，但缺乏对破坏对抗性示例的优化，忽视了攻击特定扰动的优势。在本文中，我们提出了特征调整混合（FTM），一种新的方法，提高了有针对性的攻击转移结合随机和优化的噪声在特征空间。FTM引入了可学习的特征扰动，并采用了一种有效的随机更新策略进行优化。这些可学习的扰动有助于生成更强大的对抗性示例，并提高了可移植性。我们进一步证明了攻击性能可以通过多个FTM扰动代理模型的集成来增强。对各种DNN模型的ImageNet兼容数据集进行的大量实验表明，我们的方法比最先进的方法实现了显着改进，同时保持了较低的计算成本。



## **43. ImF: Implicit Fingerprint for Large Language Models**

ImF：大型语言模型的隐式指纹 cs.CL

16 pages, 7 figures

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.21805v1) [paper-pdf](http://arxiv.org/pdf/2503.21805v1)

**Authors**: Wu jiaxuan, Peng Wanli, Fu hang, Xue Yiming, Wen juan

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making intellectual property (IP) protection essential. Most existing model fingerprint methods inject fingerprints into LLMs to protect model ownership. These methods create fingerprint pairs with weak semantic correlations, lacking the contextual coherence and semantic relatedness founded in normal question-answer (QA) pairs in LLMs. In this paper, we propose a Generation Revision Intervention (GRI) attack that can effectively exploit this flaw to erase fingerprints, highlighting the need for more secure model fingerprint methods. Thus, we propose a novel injected fingerprint paradigm called Implicit Fingerprints (ImF). ImF constructs fingerprint pairs with strong semantic correlations, disguising them as natural QA pairs within LLMs. This ensures the fingerprints are consistent with normal model behavior, making them indistinguishable and robust against detection and removal. Our experiment on multiple LLMs demonstrates that ImF retains high verification success rates under adversarial conditions, offering a reliable solution for protecting LLM ownership.

摘要: 训练大型语言模型（LLM）是资源密集型且昂贵的，因此知识产权（IP）保护至关重要。大多数现有的模型指纹方法将指纹注入LLM以保护模型所有权。这些方法创建了语义相关性较弱的指纹对，缺乏LLM中正常问答（QA）对中建立的上下文一致性和语义相关性。在本文中，我们提出了一代修订干预（GRI）攻击，可以有效地利用该缺陷来擦除指纹，强调了对更安全模型指纹方法的需求。因此，我们提出了一种新型的注入指纹范式，称为隐式指纹（ImF）。ImF构建具有强语义相关性的指纹对，将它们伪装成LLM内的自然QA对。这确保了指纹与正常模型行为一致，使其不可区分，并且对于检测和删除具有鲁棒性。我们对多个LLM的实验表明，ImF在对抗条件下保持了很高的验证成功率，为保护LLM所有权提供了可靠的解决方案。



## **44. Stop Walking in Circles! Bailing Out Early in Projected Gradient Descent**

别再绕圈子了！在预计的梯度下降中提前退出 cs.CV

To appear in the 2025 IEEE/CVF Conference on Computer Vision and  Pattern Recognition (CVPR)

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19347v1) [paper-pdf](http://arxiv.org/pdf/2503.19347v1)

**Authors**: Philip Doldo, Derek Everett, Amol Khanna, Andre T Nguyen, Edward Raff

**Abstract**: Projected Gradient Descent (PGD) under the $L_\infty$ ball has become one of the defacto methods used in adversarial robustness evaluation for computer vision (CV) due to its reliability and efficacy, making a strong and easy-to-implement iterative baseline. However, PGD is computationally demanding to apply, especially when using thousands of iterations is the current best-practice recommendation to generate an adversarial example for a single image. In this work, we introduce a simple novel method for early termination of PGD based on cycle detection by exploiting the geometry of how PGD is implemented in practice and show that it can produce large speedup factors while providing the \emph{exact} same estimate of model robustness as standard PGD. This method substantially speeds up PGD without sacrificing any attack strength, enabling evaluations of robustness that were previously computationally intractable.

摘要: $L_\infty$ ball下的投影梯度下降（PVD）由于其可靠性和有效性，已成为计算机视觉（CV）对抗鲁棒性评估中使用的事实上的方法之一，并构成了强大且易于实施的迭代基线。然而，PGP的应用在计算上要求很高，尤其是当使用数千次迭代是当前最佳实践建议来为单个图像生成对抗性示例时。在这项工作中，我们引入了一种基于循环检测提前终止PVD的简单新颖方法，通过利用实践中如何实施PVD的几何结构，并表明它可以产生很大的加速因子，同时提供与标准PVD相同的模型稳健性估计。该方法在不牺牲任何攻击强度的情况下大幅加快了PVD，从而能够评估以前在计算上难以处理的稳健性。



## **45. Efficient Adversarial Detection Frameworks for Vehicle-to-Microgrid Services in Edge Computing**

边缘计算中车辆到微电网服务的高效对抗性检测框架 cs.CR

6 pages, 3 figures, Accepted to 2025 IEEE International Conference on  Communications (ICC) Workshops

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19318v1) [paper-pdf](http://arxiv.org/pdf/2503.19318v1)

**Authors**: Ahmed Omara, Burak Kantarci

**Abstract**: As Artificial Intelligence (AI) becomes increasingly integrated into microgrid control systems, the risk of malicious actors exploiting vulnerabilities in Machine Learning (ML) algorithms to disrupt power generation and distribution grows. Detection models to identify adversarial attacks need to meet the constraints of edge environments, where computational power and memory are often limited. To address this issue, we propose a novel strategy that optimizes detection models for Vehicle-to-Microgrid (V2M) edge environments without compromising performance against inference and evasion attacks. Our approach integrates model design and compression into a unified process and results in a highly compact detection model that maintains high accuracy. We evaluated our method against four benchmark evasion attacks-Fast Gradient Sign Method (FGSM), Basic Iterative Method (BIM), Carlini & Wagner method (C&W) and Conditional Generative Adversarial Network (CGAN) method-and two knowledge-based attacks, white-box and gray-box. Our optimized model reduces memory usage from 20MB to 1.3MB, inference time from 3.2 seconds to 0.9 seconds, and GPU utilization from 5% to 2.68%.

摘要: 随着人工智能(AI)越来越多地集成到微电网控制系统中，恶意行为者利用机器学习(ML)算法中的漏洞来扰乱发电和配电的风险越来越大。识别敌意攻击的检测模型需要满足边缘环境的约束，在边缘环境中，计算能力和内存往往是有限的。为了解决这个问题，我们提出了一种新的策略，该策略优化了车辆到微电网(V2M)边缘环境的检测模型，而不会对推理和规避攻击的性能造成影响。我们的方法将模型设计和压缩集成到一个统一的过程中，结果是一个高度紧凑的检测模型，保持了高精度。我们对快速梯度符号法(FGSM)、基本迭代法(BIM)、Carlini&Wagner法(C&W)和条件生成对抗网络(CGAN)等四种基准规避攻击和两种基于知识的白盒和灰盒攻击进行了测试。我们的优化模型将内存使用量从20MB减少到1.3MB，推理时间从3.2秒减少到0.9秒，GPU使用率从5%减少到2.68%。



## **46. Robustness of Proof of Team Sprint (PoTS) Against Attacks: A Simulation-Based Analysis**

团队冲刺证明（PoTS）对攻击的稳健性：基于模拟的分析 cs.DC

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19293v1) [paper-pdf](http://arxiv.org/pdf/2503.19293v1)

**Authors**: Naoki Yonezawa

**Abstract**: This study evaluates the robustness of Proof of Team Sprint (PoTS) against adversarial attacks through simulations, focusing on the attacker win rate and computational efficiency under varying team sizes (\( N \)) and attacker ratios (\( \alpha \)). Our results demonstrate that PoTS effectively reduces an attacker's ability to dominate the consensus process. For instance, when \( \alpha = 0.5 \), the attacker win rate decreases from 50.7\% at \( N = 1 \) to below 0.4\% at \( N = 8 \), effectively neutralizing adversarial influence. Similarly, at \( \alpha = 0.8 \), the attacker win rate drops from 80.47\% at \( N = 1 \) to only 2.79\% at \( N = 16 \). In addition to its strong security properties, PoTS maintains high computational efficiency. We introduce the concept of Normalized Computation Efficiency (NCE) to quantify this efficiency gain, showing that PoTS significantly improves resource utilization as team size increases. The results indicate that as \( N \) grows, PoTS not only enhances security but also achieves better computational efficiency due to the averaging effects of execution time variations. These findings highlight PoTS as a promising alternative to traditional consensus mechanisms, offering both robust security and efficient resource utilization. By leveraging team-based block generation and randomized participant reassignment, PoTS provides a scalable and resilient approach to decentralized consensus.

摘要: 本研究通过模拟评估团队冲刺证明（PoTS）对对抗性攻击的稳健性，重点关注不同团队规模（\（N \））和攻击者比例（\（\Alpha \））下的攻击者获胜率和计算效率。我们的结果表明，PoTS有效地降低了攻击者主导共识过程的能力。例如，当\（\Alpha = 0.5 \）时，攻击者获胜率从\（N = 1 \）的50.7%下降到\（N = 8 \）的0.4%以下，有效地中和了对抗影响。同样，在\（\Alpha = 0.8 \）时，攻击者获胜率从\（N = 1 \）时的80.47%下降到\（N = 16 \）时的仅2.79%。PoTS除了强大的安全属性外，还保持了高计算效率。我们引入了规范化计算效率（NCO）的概念来量化这种效率收益，表明随着团队规模的增加，PoTS显着提高了资源利用率。结果表明，随着\（N \）的增长，PoTS不仅增强了安全性，而且由于执行时间变化的平均效应，还实现了更好的计算效率。这些发现凸显了PoTS是传统共识机制的一个有希望的替代方案，既提供强大的安全性又有效的资源利用率。通过利用基于团队的区块生成和随机参与者重新分配，PoTS为去中心化共识提供了一种可扩展和弹性的方法。



## **47. Activation Functions Considered Harmful: Recovering Neural Network Weights through Controlled Channels**

被认为有害的激活功能：通过受控通道恢复神经网络权重 cs.CR

17 pages, 5 figures

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.19142v1) [paper-pdf](http://arxiv.org/pdf/2503.19142v1)

**Authors**: Jesse Spielman, David Oswald, Mark Ryan, Jo Van Bulck

**Abstract**: With high-stakes machine learning applications increasingly moving to untrusted end-user or cloud environments, safeguarding pre-trained model parameters becomes essential for protecting intellectual property and user privacy. Recent advancements in hardware-isolated enclaves, notably Intel SGX, hold the promise to secure the internal state of machine learning applications even against compromised operating systems. However, we show that privileged software adversaries can exploit input-dependent memory access patterns in common neural network activation functions to extract secret weights and biases from an SGX enclave.   Our attack leverages the SGX-Step framework to obtain a noise-free, instruction-granular page-access trace. In a case study of an 11-input regression network using the Tensorflow Microlite library, we demonstrate complete recovery of all first-layer weights and biases, as well as partial recovery of parameters from deeper layers under specific conditions. Our novel attack technique requires only 20 queries per input per weight to obtain all first-layer weights and biases with an average absolute error of less than 1%, improving over prior model stealing attacks.   Additionally, a broader ecosystem analysis reveals the widespread use of activation functions with input-dependent memory access patterns in popular machine learning frameworks (either directly or via underlying math libraries). Our findings highlight the limitations of deploying confidential models in SGX enclaves and emphasise the need for stricter side-channel validation of machine learning implementations, akin to the vetting efforts applied to secure cryptographic libraries.

摘要: 随着高风险的机器学习应用程序越来越多地转向不受信任的最终用户或云环境，保护预先训练的模型参数对于保护知识产权和用户隐私变得至关重要。硬件隔离飞地（尤其是英特尔SGX）的最新进展有望保护机器学习应用程序的内部状态，即使是针对受攻击的操作系统。然而，我们表明，特权软件对手可以利用常见神经网络激活函数中的输入相关存储器访问模式，从SGX飞地中提取秘密权重和偏差。   我们的攻击利用SGX-Step框架来获得无噪音、描述粒度的页面访问跟踪。在使用TensorFlow MicroLite库的11个输入回归网络的案例研究中，我们展示了在特定条件下所有第一层权重和偏差的完全恢复，以及从较深层部分恢复参数。我们的新颖攻击技术仅需要每个输入和权重20个查询，即可获得所有第一层权重和偏差，平均绝对误差小于1%，比之前的模型窃取攻击有所改进。   此外，更广泛的生态系统分析揭示了流行的机器学习框架中广泛使用具有输入相关内存访问模式的激活函数（直接或通过底层数学库）。我们的研究结果强调了在SGX飞地中部署机密模型的局限性，并强调需要对机器学习实施进行更严格的侧通道验证，类似于应用于安全加密库的审查工作。



## **48. Masks and Mimicry: Strategic Obfuscation and Impersonation Attacks on Authorship Verification**

面具和模仿：对作者身份验证的战略混淆和模仿攻击 cs.CL

Accepted at NLP4DH Workshop @ NAACL 2025

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.19099v1) [paper-pdf](http://arxiv.org/pdf/2503.19099v1)

**Authors**: Kenneth Alperin, Rohan Leekha, Adaku Uchendu, Trang Nguyen, Srilakshmi Medarametla, Carlos Levya Capote, Seth Aycock, Charlie Dagli

**Abstract**: The increasing use of Artificial Intelligence (AI) technologies, such as Large Language Models (LLMs) has led to nontrivial improvements in various tasks, including accurate authorship identification of documents. However, while LLMs improve such defense techniques, they also simultaneously provide a vehicle for malicious actors to launch new attack vectors. To combat this security risk, we evaluate the adversarial robustness of authorship models (specifically an authorship verification model) to potent LLM-based attacks. These attacks include untargeted methods - \textit{authorship obfuscation} and targeted methods - \textit{authorship impersonation}. For both attacks, the objective is to mask or mimic the writing style of an author while preserving the original texts' semantics, respectively. Thus, we perturb an accurate authorship verification model, and achieve maximum attack success rates of 92\% and 78\% for both obfuscation and impersonation attacks, respectively.

摘要: 人工智能（AI）技术（例如大型语言模型（LLM））的越来越多的使用导致了各种任务的重要改进，包括文档的准确作者身份识别。然而，在LLM改进此类防御技术的同时，它们也同时为恶意行为者提供了发起新攻击载体的工具。为了应对这种安全风险，我们评估了作者身份模型（特别是作者身份验证模型）对强大的基于LLM的攻击的对抗稳健性。这些攻击包括非目标方法- \textit{authorship obfuscation}和目标方法- \textit{authorship imperation}。对于这两种攻击，目标是分别掩盖或模仿作者的写作风格，同时保留原始文本的语义。因此，我们扰乱了准确的作者身份验证模型，并分别实现了模糊和模仿攻击的最大攻击成功率92%和78%。



## **49. Quantum Byzantine Multiple Access Channels**

量子拜占庭多址接入信道 cs.IT

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2502.12047v2) [paper-pdf](http://arxiv.org/pdf/2502.12047v2)

**Authors**: Minglai Cai, Christian Deppe

**Abstract**: In communication theory, attacks like eavesdropping or jamming are typically assumed to occur at the channel level, while communication parties are expected to follow established protocols. But what happens if one of the parties turns malicious? In this work, we investigate a compelling scenario: a multiple-access channel with two transmitters and one receiver, where one transmitter deviates from the protocol and acts dishonestly. To address this challenge, we introduce the Byzantine multiple-access classical-quantum channel and derive an achievable communication rate for this adversarial setting.

摘要: 在通信理论中，窃听或干扰等攻击通常被假设发生在通道级别，而通信方预计会遵循既定的协议。但如果其中一方变得恶意会发生什么？在这项工作中，我们研究了一个引人注目的场景：具有两个发射机和一个接收机的多址通道，其中一个发射机偏离了协议并行为不诚实。为了应对这一挑战，我们引入了拜占庭式多址经典量子通道，并推导出针对这种对抗环境的可实现的通信速率。



## **50. MF-CLIP: Leveraging CLIP as Surrogate Models for No-box Adversarial Attacks**

MF-CLIP：利用CLIP作为无框对抗攻击的代理模型 cs.LG

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2307.06608v3) [paper-pdf](http://arxiv.org/pdf/2307.06608v3)

**Authors**: Jiaming Zhang, Lingyu Qiu, Qi Yi, Yige Li, Jitao Sang, Changsheng Xu, Dit-Yan Yeung

**Abstract**: The vulnerability of Deep Neural Networks (DNNs) to adversarial attacks poses a significant challenge to their deployment in safety-critical applications. While extensive research has addressed various attack scenarios, the no-box attack setting where adversaries have no prior knowledge, including access to training data of the target model, remains relatively underexplored despite its practical relevance. This work presents a systematic investigation into leveraging large-scale Vision-Language Models (VLMs), particularly CLIP, as surrogate models for executing no-box attacks. Our theoretical and empirical analyses reveal a key limitation in the execution of no-box attacks stemming from insufficient discriminative capabilities for direct application of vanilla CLIP as a surrogate model. To address this limitation, we propose MF-CLIP: a novel framework that enhances CLIP's effectiveness as a surrogate model through margin-aware feature space optimization. Comprehensive evaluations across diverse architectures and datasets demonstrate that MF-CLIP substantially advances the state-of-the-art in no-box attacks, surpassing existing baselines by 15.23% on standard models and achieving a 9.52% improvement on adversarially trained models. Our code will be made publicly available to facilitate reproducibility and future research in this direction.

摘要: 深度神经网络(DNN)对敌意攻击的脆弱性对其在安全关键应用中的部署构成了巨大的挑战。虽然广泛的研究已经解决了各种攻击情景，但对手事先不知道的无盒子攻击环境，包括获得目标模型的训练数据，尽管具有实际意义，但仍然相对探索不足。本文对大规模视觉语言模型，特别是CLIP，作为执行非盒子攻击的代理模型进行了系统的研究。我们的理论和经验分析揭示了执行非盒子攻击的一个关键限制，这是因为对于直接应用Vanilla CLIP作为代理模型的区分能力不足。为了解决这一局限性，我们提出了MF-CLIP：一种新颖的框架，通过边缘感知的特征空间优化来增强CLIP作为代理模型的有效性。跨不同架构和数据集的综合评估表明，MF-CLIP在非盒子攻击方面大幅提升了最先进的水平，在标准模型上超过了现有基准15.23%，在对抗训练的模型上实现了9.52%的改进。我们的代码将公开提供，以促进可重复性和未来在这一方向的研究。



