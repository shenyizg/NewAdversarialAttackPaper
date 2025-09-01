# Latest Adversarial Attack Papers
**update at 2025-09-01 10:13:26**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Detecting Stealthy Data Poisoning Attacks in AI Code Generators**

检测人工智能代码生成器中的隐形数据中毒攻击 cs.CR

Accepted to the 3rd IEEE International Workshop on Reliable and  Secure AI for Software Engineering (ReSAISE, 2025), co-located with ISSRE  2025

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21636v1) [paper-pdf](http://arxiv.org/pdf/2508.21636v1)

**Authors**: Cristina Improta

**Abstract**: Deep learning (DL) models for natural language-to-code generation have become integral to modern software development pipelines. However, their heavy reliance on large amounts of data, often collected from unsanitized online sources, exposes them to data poisoning attacks, where adversaries inject malicious samples to subtly bias model behavior. Recent targeted attacks silently replace secure code with semantically equivalent but vulnerable implementations without relying on explicit triggers to launch the attack, making it especially hard for detection methods to distinguish clean from poisoned samples. We present a systematic study on the effectiveness of existing poisoning detection methods under this stealthy threat model. Specifically, we perform targeted poisoning on three DL models (CodeBERT, CodeT5+, AST-T5), and evaluate spectral signatures analysis, activation clustering, and static analysis as defenses. Our results show that all methods struggle to detect triggerless poisoning, with representation-based approaches failing to isolate poisoned samples and static analysis suffering false positives and false negatives, highlighting the need for more robust, trigger-independent defenses for AI-assisted code generation.

摘要: 用于自然语言到代码生成的深度学习（DL）模型已成为现代软件开发管道的组成部分。然而，它们严重依赖大量数据，这些数据通常是从未经清理的在线来源收集的，这使它们面临数据中毒攻击，对手会注入恶意样本以微妙地偏向模型行为。最近的有针对性的攻击以语义等效但脆弱的实现悄然取代安全代码，而不依赖显式触发器来发起攻击，这使得检测方法特别难以区分干净样本和有毒样本。我们对这种隐形威胁模型下现有中毒检测方法的有效性进行了系统研究。具体来说，我们对三种DL模型（CodeBRT、CodeT 5+、AST-T5）执行有针对性的中毒，并评估光谱特征分析、激活集群和静态分析作为防御措施。我们的结果表明，所有方法都很难检测无指示器中毒，基于表示的方法无法隔离中毒样本，静态分析会出现假阳性和假阴性，这凸显了人工智能辅助代码生成需要更强大、独立于指示器的防御。



## **2. Adversarial Patch Attack for Ship Detection via Localized Augmentation**

通过局部增强进行船舶检测的对抗补丁攻击 cs.CV

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21472v1) [paper-pdf](http://arxiv.org/pdf/2508.21472v1)

**Authors**: Chun Liu, Panpan Ding, Zheng Zheng, Hailong Wang, Bingqian Zhu, Tao Xu, Zhigang Han, Jiayao Wang

**Abstract**: Current ship detection techniques based on remote sensing imagery primarily rely on the object detection capabilities of deep neural networks (DNNs). However, DNNs are vulnerable to adversarial patch attacks, which can lead to misclassification by the detection model or complete evasion of the targets. Numerous studies have demonstrated that data transformation-based methods can improve the transferability of adversarial examples. However, excessive augmentation of image backgrounds or irrelevant regions may introduce unnecessary interference, resulting in false detections of the object detection model. These errors are not caused by the adversarial patches themselves but rather by the over-augmentation of background and non-target areas. This paper proposes a localized augmentation method that applies augmentation only to the target regions, avoiding any influence on non-target areas. By reducing background interference, this approach enables the loss function to focus more directly on the impact of the adversarial patch on the detection model, thereby improving the attack success rate. Experiments conducted on the HRSC2016 dataset demonstrate that the proposed method effectively increases the success rate of adversarial patch attacks and enhances their transferability.

摘要: 当前基于遥感图像的船舶检测技术主要依赖于深度神经网络（DNN）的物体检测能力。然而，DNN很容易受到对抗补丁攻击，这可能会导致检测模型的错误分类或完全逃避目标。许多研究表明，基于数据转换的方法可以提高对抗性示例的可移植性。然而，图像背景或不相关区域的过度增强可能会引入不必要的干扰，导致对象检测模型的错误检测。这些错误不是由对抗补丁本身引起的，而是由背景和非目标区域的过度扩大引起的。本文提出了一种局部增强方法，仅对目标区域进行增强，避免对非目标区域产生任何影响。通过减少背景干扰，这种方法使损失函数能够更直接地关注对抗补丁对检测模型的影响，从而提高攻击成功率。在HRSC 2016数据集上进行的实验表明，该方法有效提高了对抗性补丁攻击的成功率，增强了其可移植性。



## **3. Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review**

发布到Perish：对LLM辅助同行评审的即时注入攻击 cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.20863v2) [paper-pdf](http://arxiv.org/pdf/2508.20863v2)

**Authors**: Matteo Gioele Collu, Umberto Salviati, Roberto Confalonieri, Mauro Conti, Giovanni Apruzzese

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.

摘要: 大型语言模型（LLM）越来越多地融入到科学同行评审过程中，这引发了有关其可靠性和操纵弹性的新问题。在这项工作中，我们调查了隐藏的提示注入攻击的可能性，即作者在论文的PDF中嵌入对抗性文本以影响LLM生成的评论。我们首先正式化三种不同的威胁模型，这些模型设想攻击者具有不同的动机--并非所有这些都暗示着恶意意图。对于每个威胁模型，我们设计了对抗性提示，这些提示对人类读者来说是不可见的，但可以将LLM的输出引导到作者想要的结果。通过对领域学者的用户研究，我们得出了四个代表性的审查提示，用于从法学硕士那里获得同行审查。然后，我们评估对抗提示在（i）不同的审查提示、（ii）不同的基于LLM的商业系统和（iii）不同的同行评审论文中的稳健性。我们的结果表明，对抗性提示可以可靠地误导LLM，有时会对“诚实但懒惰”的评论者产生不利影响。最后，我们提出并根据经验评估了在自动内容检查下降低对抗性提示可检测性的方法。



## **4. Time Tells All: Deanonymization of Blockchain RPC Users with Zero Transaction Fee (Extended Version)**

时间证明一切：以零交易费实现区块链PCC用户去匿名化（扩展版本） cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21440v1) [paper-pdf](http://arxiv.org/pdf/2508.21440v1)

**Authors**: Shan Wang, Ming Yang, Yu Liu, Yue Zhang, Shuaiqing Zhang, Zhen Ling, Jiannong Cao, Xinwen Fu

**Abstract**: Remote Procedure Call (RPC) services have become a primary gateway for users to access public blockchains. While they offer significant convenience, RPC services also introduce critical privacy challenges that remain insufficiently examined. Existing deanonymization attacks either do not apply to blockchain RPC users or incur costs like transaction fees assuming an active network eavesdropper. In this paper, we propose a novel deanonymization attack that can link an IP address of a RPC user to this user's blockchain pseudonym. Our analysis reveals a temporal correlation between the timestamps of transaction confirmations recorded on the public ledger and those of TCP packets sent by the victim when querying transaction status. We assume a strong passive adversary with access to network infrastructure, capable of monitoring traffic at network border routers or Internet exchange points. By monitoring network traffic and analyzing public ledgers, the attacker can link the IP address of the TCP packet to the pseudonym of the transaction initiator by exploiting the temporal correlation. This deanonymization attack incurs zero transaction fee. We mathematically model and analyze the attack method, perform large-scale measurements of blockchain ledgers, and conduct real-world attacks to validate the attack. Our attack achieves a high success rate of over 95% against normal RPC users on various blockchain networks, including Ethereum, Bitcoin and Solana.

摘要: 远程过程调用（PRC）服务已成为用户访问公共区块链的主要门户。虽然它们提供了巨大的便利，但它们也带来了关键的隐私挑战，这些挑战仍然没有得到充分的审查。现有的去匿名攻击要么不适用于区块链PRC用户，要么假设有活跃的网络窃听者会产生交易费等成本。在本文中，我们提出了一种新型的去匿名攻击，可以将PRC用户的IP地址链接到该用户的区块链假名。我们的分析揭示了公共分类帐上记录的交易确认的时间戳与受害者在查询交易状态时发送的MAC数据包的时间戳之间存在时间相关性。我们假设一个强大的被动对手，可以访问网络基础设施，能够监控网络边界路由器或互联网交换点的流量。通过监控网络流量和分析公共分类帐，攻击者可以利用时间相关性将Tcp数据包的IP地址链接到交易发起者的假名。这种去匿名化攻击的交易费用为零。我们对攻击方法进行数学建模和分析，对区块链账本进行大规模测量，并进行现实世界的攻击以验证攻击。我们的攻击针对各种区块链网络（包括以太坊、比特币和Solana）上的普通PRC用户，成功率超过95%。



## **5. On the Adversarial Robustness of Spiking Neural Networks Trained by Local Learning**

局部学习训练的尖峰神经网络的对抗鲁棒性 cs.LG

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2504.08897v2) [paper-pdf](http://arxiv.org/pdf/2504.08897v2)

**Authors**: Jiaqi Lin, Abhronil Sengupta

**Abstract**: Recent research has shown the vulnerability of Spiking Neural Networks (SNNs) under adversarial examples that are nearly indistinguishable from clean data in the context of frame-based and event-based information. The majority of these studies are constrained in generating adversarial examples using Backpropagation Through Time (BPTT), a gradient-based method which lacks biological plausibility. In contrast, local learning methods, which relax many of BPTT's constraints, remain under-explored in the context of adversarial attacks. To address this problem, we examine adversarial robustness in SNNs through the framework of four types of training algorithms. We provide an in-depth analysis of the ineffectiveness of gradient-based adversarial attacks to generate adversarial instances in this scenario. To overcome these limitations, we introduce a hybrid adversarial attack paradigm that leverages the transferability of adversarial instances. The proposed hybrid approach demonstrates superior performance, outperforming existing adversarial attack methods. Furthermore, the generalizability of the method is assessed under multi-step adversarial attacks, adversarial attacks in black-box FGSM scenarios, and within the non-spiking domain.

摘要: 最近的研究表明，尖峰神经网络（SNN）在对抗性示例下的脆弱性，这些示例与基于帧和基于事件的信息背景下的干净数据几乎无法区分。这些研究中的大多数在使用时间反向传播（BPTT）生成对抗性示例方面受到限制，BPTT是一种基于梯度的方法，缺乏生物学相似性。相比之下，本地学习方法放松了BPTT的许多限制，但在对抗性攻击的背景下仍然没有得到充分的探索。为了解决这个问题，我们通过四种类型的训练算法的框架来检查SNN中的对抗鲁棒性。我们深入分析了基于梯度的对抗攻击在这种情况下生成对抗实例的无效性。为了克服这些限制，我们引入了一种混合对抗攻击范式，该范式利用对抗实例的可移植性。提出的混合方法表现出卓越的性能，优于现有的对抗攻击方法。此外，在多步对抗性攻击、黑匣子FGSM场景中的对抗性攻击和非尖峰域中评估了该方法的通用性。



## **6. The WASM Cloak: Evaluating Browser Fingerprinting Defenses Under WebAssembly based Obfuscation**

WASM斗篷：在基于WebAssembly的混淆下评估浏览器指纹防御 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21219v1) [paper-pdf](http://arxiv.org/pdf/2508.21219v1)

**Authors**: A H M Nazmus Sakib, Mahsin Bin Akram, Joseph Spracklen, Sahan Kalutarage, Raveen Wijewickrama, Igor Bilogrevic, Murtuza Jadliwala

**Abstract**: Browser fingerprinting defenses have historically focused on detecting JavaScript(JS)-based tracking techniques. However, the widespread adoption of WebAssembly (WASM) introduces a potential blind spot, as adversaries can convert JS to WASM's low-level binary format to obfuscate malicious logic. This paper presents the first systematic evaluation of how such WASM-based obfuscation impacts the robustness of modern fingerprinting defenses. We develop an automated pipeline that translates real-world JS fingerprinting scripts into functional WASM-obfuscated variants and test them against two classes of defenses: state-of-the-art detectors in research literature and commercial, in-browser tools. Our findings reveal a notable divergence: detectors proposed in the research literature that rely on feature-based analysis of source code show moderate vulnerability, stemming from outdated datasets or a lack of WASM compatibility. In contrast, defenses such as browser extensions and native browser features remained completely effective, as their API-level interception is agnostic to the script's underlying implementation. These results highlight a gap between academic and practical defense strategies and offer insights into strengthening detection approaches against WASM-based obfuscation, while also revealing opportunities for more evasive techniques in future attacks.

摘要: 浏览器指纹识别防御历来专注于检测基于JavaScript（JS）的跟踪技术。然而，WebAssembly（WASM）的广泛采用引入了一个潜在的盲点，因为对手可以将JS转换为WASM的低级二进制格式来混淆恶意逻辑。本文首次对这种基于WASM的模糊处理如何影响现代指纹防御的稳健性进行了系统评估。我们开发了一个自动化管道，将现实世界的JS指纹识别脚本翻译为功能性WASM模糊变体，并针对两类防御措施对其进行测试：研究文献中最先进的检测器和商业浏览器内工具。我们的研究结果揭示了一个显着的分歧：研究文献中提出的依赖于源代码基于特征的分析的检测器显示出中度漏洞，源于过时的数据集或缺乏WASM兼容性。相比之下，浏览器扩展和原生浏览器功能等防御仍然完全有效，因为它们的API级拦截与脚本的底层实现无关。这些结果凸显了学术和实际防御策略之间的差距，并为加强针对基于WASM的模糊的检测方法提供了见解，同时也揭示了未来攻击中使用更具规避性技术的机会。



## **7. First-Place Solution to NeurIPS 2024 Invisible Watermark Removal Challenge**

NeurIPS 2024隐形水印去除挑战赛的一流解决方案 cs.CV

Winning solution to the NeurIPS 2024 Erasing the Invisible challenge

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21072v1) [paper-pdf](http://arxiv.org/pdf/2508.21072v1)

**Authors**: Fahad Shamshad, Tameem Bakr, Yahia Shaaban, Noor Hussein, Karthik Nandakumar, Nils Lukas

**Abstract**: Content watermarking is an important tool for the authentication and copyright protection of digital media. However, it is unclear whether existing watermarks are robust against adversarial attacks. We present the winning solution to the NeurIPS 2024 Erasing the Invisible challenge, which stress-tests watermark robustness under varying degrees of adversary knowledge. The challenge consisted of two tracks: a black-box and beige-box track, depending on whether the adversary knows which watermarking method was used by the provider. For the beige-box track, we leverage an adaptive VAE-based evasion attack, with a test-time optimization and color-contrast restoration in CIELAB space to preserve the image's quality. For the black-box track, we first cluster images based on their artifacts in the spatial or frequency-domain. Then, we apply image-to-image diffusion models with controlled noise injection and semantic priors from ChatGPT-generated captions to each cluster with optimized parameter settings. Empirical evaluations demonstrate that our method successfully achieves near-perfect watermark removal (95.7%) with negligible impact on the residual image's quality. We hope that our attacks inspire the development of more robust image watermarking methods.

摘要: 内容水印是数字媒体认证和版权保护的重要工具。然而，目前尚不清楚现有的水印是否能够抵御对抗攻击。我们向NeurIPS 2024 Erase the Invisible挑战展示了获胜的解决方案，该解决方案在不同程度的对手知识下对水印稳健性进行压力测试。挑战由两个轨道组成：黑匣子和米色盒轨道，具体取决于对手是否知道提供商使用了哪种水印方法。对于米色盒轨道，我们利用基于VAR的自适应规避攻击，并在CIELAB空间中进行测试时优化和颜色对比度恢复，以保持图像的质量。对于黑匣子轨道，我们首先根据图像在空间或频域中的伪影对图像进行分组。然后，我们将具有受控噪音注入和ChatGPT生成的字幕的语义先验的图像到图像扩散模型应用到每个集群，并优化参数设置。经验评估表明，我们的方法成功实现了近乎完美的水印去除（95.7%），而对残留图像质量的影响可以忽略不计。我们希望我们的攻击能够激发更鲁棒的图像水印方法的开发。



## **8. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

EmantSleuth：通过语义意图不变性检测提示注入 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20890v1) [paper-pdf](http://arxiv.org/pdf/2508.20890v1)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.

摘要: 大型语言模型（LLM）越来越多地集成到现实世界的应用程序中，从虚拟助手到自治代理。然而，它们的灵活性也引入了新的攻击向量，特别是提示注入（PI），其中攻击者通过精心制作的输入操纵模型行为。随着攻击者不断地使用释义、混淆甚至多任务注入策略，现有的基准不再足以捕获所有新兴威胁。   为了解决这一差距，我们构建了一个新的基准，系统地扩展了以前的努力。我们的基准涵盖了两种广泛使用的现有基准，同时引入了新的操纵技术和多任务场景，从而提供了更全面的评估设置。我们发现，现有的防御虽然在原始基准上有效，但在我们的基准下表现出明显的弱点，这凸显了对更强大解决方案的需求。我们的关键见解是，虽然攻击形式可能会有所不同，但对手的意图（注入未经授权的任务）保持不变。在这一观察的基础上，我们提出了EmittSleuth，这是一个面向语义的防御框架，它通过对任务级意图而不是表面特征进行推理来检测提示注入。在最先进的基准测试中进行评估后，AktSleuth始终优于现有的防御，同时保持相当的运行时间和成本效率。这些结果表明，基于意图的语义推理提供了一个强大的，有效的，和可推广的策略，以抵御不断发展的即时注入威胁的LLM。



## **9. FusionCounting: Robust visible-infrared image fusion guided by crowd counting via multi-task learning**

FusionCounting：通过多任务学习由人群计数引导的鲁棒可见光-红外图像融合 cs.CV

11 pages, 9 figures

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20817v1) [paper-pdf](http://arxiv.org/pdf/2508.20817v1)

**Authors**: He Li, Xinyu Liu, Weihang Kong, Xingchen Zhang

**Abstract**: Most visible and infrared image fusion (VIF) methods focus primarily on optimizing fused image quality. Recent studies have begun incorporating downstream tasks, such as semantic segmentation and object detection, to provide semantic guidance for VIF. However, semantic segmentation requires extensive annotations, while object detection, despite reducing annotation efforts compared with segmentation, faces challenges in highly crowded scenes due to overlapping bounding boxes and occlusion. Moreover, although RGB-T crowd counting has gained increasing attention in recent years, no studies have integrated VIF and crowd counting into a unified framework. To address these challenges, we propose FusionCounting, a novel multi-task learning framework that integrates crowd counting into the VIF process. Crowd counting provides a direct quantitative measure of population density with minimal annotation, making it particularly suitable for dense scenes. Our framework leverages both input images and population density information in a mutually beneficial multi-task design. To accelerate convergence and balance tasks contributions, we introduce a dynamic loss function weighting strategy. Furthermore, we incorporate adversarial training to enhance the robustness of both VIF and crowd counting, improving the model's stability and resilience to adversarial attacks. Experimental results on public datasets demonstrate that FusionCounting not only enhances image fusion quality but also achieves superior crowd counting performance.

摘要: 大多数可见光和红外图像融合（VIF）方法主要关注优化融合图像质量。最近的研究已经开始整合下游任务，例如语义分割和对象检测，为VIF提供语义指导。然而，语义分割需要大量的注释，而对象检测尽管与分割相比减少了注释工作，但由于重叠的边界框和遮挡，在高度拥挤的场景中面临挑战。此外，尽管近年来RGB-T人群计数越来越受到关注，但还没有研究将VIF和人群计数整合到统一的框架中。为了应对这些挑战，我们提出FusionCounting，这是一种新型的多任务学习框架，可将人群计数集成到VIF流程中。人群计数提供了人口密度的直接定量测量，只需最少的注释，使其特别适合密集场景。我们的框架在互利的多任务设计中利用输入图像和人口密度信息。为了加速收敛和平衡任务贡献，我们引入了动态损失函数加权策略。此外，我们结合了对抗性训练来增强VIF和人群计数的鲁棒性，提高了模型的稳定性和对抗性攻击的弹性。在公开数据集上的实验结果表明，FusionCounting不仅提高了图像融合质量，而且实现了优越的人群计数性能。



## **10. Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning**

Token Buncher：保护LLM免受有害的强化学习微调 cs.LG

Project Hompage: https://tokenbuncher.github.io/

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20697v1) [paper-pdf](http://arxiv.org/pdf/2508.20697v1)

**Authors**: Weitao Feng, Lixu Wang, Tianyi Wei, Jie Zhang, Chongyang Gao, Sinong Zhan, Peizhuo Lv, Wei Dong

**Abstract**: As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.

摘要: 随着大型语言模型（LLM）的能力不断增强，通过微调导致有害误用的风险也在增加。虽然大多数先前的研究都假设攻击者依赖监督微调（SFT）来进行此类滥用，但我们系统地证明，强化学习（RL）使对手能够在匹配的计算预算下更有效地打破安全对齐并促进高级有害任务协助。为了应对这种新出现的威胁，我们提出了TokenBuncher，这是第一个专门针对基于RL的有害微调的有效防御。TokenBuncher抑制了RL所依赖的基础：模型响应不确定性。通过限制不确定性，基于RL的微调无法再利用不同的奖励信号来推动模型走向有害行为。我们通过以互赏为回报的RL和旨在防止专家域有害能力升级的Token Noiser机制来实现这种防御。跨多个模型和RL算法的大量实验表明，TokenBuncher稳健地减轻了有害的RL微调，同时保留了良性的任务效用和微调能力。我们的结果强调，基于RL的有害微调比SFT构成更大的系统性风险，并且TokenBuncher提供了有效且通用的防御。



## **11. Disruptive Attacks on Face Swapping via Low-Frequency Perceptual Perturbations**

通过低频感知扰动对人脸交换进行破坏性攻击 cs.CV

Accepted to IEEE IJCNN 2025

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20595v1) [paper-pdf](http://arxiv.org/pdf/2508.20595v1)

**Authors**: Mengxiao Huang, Minglei Shu, Shuwang Zhou, Zhaoyang Liu

**Abstract**: Deepfake technology, driven by Generative Adversarial Networks (GANs), poses significant risks to privacy and societal security. Existing detection methods are predominantly passive, focusing on post-event analysis without preventing attacks. To address this, we propose an active defense method based on low-frequency perceptual perturbations to disrupt face swapping manipulation, reducing the performance and naturalness of generated content. Unlike prior approaches that used low-frequency perturbations to impact classification accuracy,our method directly targets the generative process of deepfake techniques. We combine frequency and spatial domain features to strengthen defenses. By introducing artifacts through low-frequency perturbations while preserving high-frequency details, we ensure the output remains visually plausible. Additionally, we design a complete architecture featuring an encoder, a perturbation generator, and a decoder, leveraging discrete wavelet transform (DWT) to extract low-frequency components and generate perturbations that disrupt facial manipulation models. Experiments on CelebA-HQ and LFW demonstrate significant reductions in face-swapping effectiveness, improved defense success rates, and preservation of visual quality.

摘要: 由生成对抗网络（GAN）驱动的Deepfake技术对隐私和社会安全构成了重大风险。现有的检测方法主要是被动的，重点是事后分析，而不防止攻击。为了解决这个问题，我们提出了一种基于低频感知扰动的主动防御方法，以扰乱面部交换操纵，降低生成内容的性能和自然性。与使用低频扰动影响分类准确性的现有方法不同，我们的方法直接针对深度伪造技术的生成过程。我们结合频率和空间域特征来加强防御。通过通过低频扰动引入伪影，同时保留高频细节，我们确保输出在视觉上保持可信。此外，我们设计了一个完整的架构，其中包括编码器、扰动生成器和解码器，利用离散子波变换（DWT）来提取低频分量并生成扰乱面部操纵模型的扰动。CelebA-HQ和LFW的实验表明，换脸有效性显着降低，防御成功率提高，视觉质量得以保留。



## **12. Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application**

多模态LLM越狱的概率建模：从量化到应用 cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2503.06989v4) [paper-pdf](http://arxiv.org/pdf/2503.06989v4)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Zonghao Ying, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Quanchen Zou

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.

摘要: 最近，多模式大型语言模型（MLLM）展示了其在理解多模式内容方面的卓越能力。然而，它们仍然容易受到越狱攻击，这些攻击利用其安全调整中的弱点来产生有害反应。之前的研究根据回应是否包含恶意内容将越狱分为成功或失败。然而，考虑到MLLM响应的随机性，这种对输入越狱MLLM的能力的二元分类是不合适的。从这个观点出发，我们引入越狱概率来量化输入的越狱潜力，这代表当提示此输入时MLLM生成恶意响应的可能性。我们通过对MLLM的多次查询来估算这一可能性。使用越狱概率预测网络（JPPN）对输入隐藏状态与其相应越狱概率之间的关系进行建模后，我们使用连续越狱概率进行优化。具体来说，我们提出了基于越狱概率的攻击（JPA），该攻击优化输入图像上的对抗性扰动以最大化越狱概率，并通过包括单调文本改写进一步将其增强为多模式JPA（MJPA）。为了对抗攻击，我们还提出了基于越狱概率的微调（JPF），它通过MLLM参数更新最大限度地降低越狱概率。大量实验表明，（1）（M）JPA在白盒和黑匣子设置下攻击广泛的模型时都能产生显着的改进。(2)JPF最多将越狱人数大幅减少60%以上。上述两个结果都证明了引入越狱概率以在输入越狱能力之间进行细微差别的重要性。



## **13. Formal Verification of Physical Layer Security Protocols for Next-Generation Communication Networks (extended version)**

下一代通信网络物理层安全协议的形式验证（扩展版本） cs.CR

Extended version (with appendices) of the camera-ready for ICFEM2025;  24 pages, 3 tables, and 6 figures

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.19430v2) [paper-pdf](http://arxiv.org/pdf/2508.19430v2)

**Authors**: Kangfeng Ye, Roberto Metere, Jim Woodcock, Poonam Yadav

**Abstract**: Formal verification is crucial for ensuring the robustness of security protocols against adversarial attacks. The Needham-Schroeder protocol, a foundational authentication mechanism, has been extensively studied, including its integration with Physical Layer Security (PLS) techniques such as watermarking and jamming. Recent research has used ProVerif to verify these mechanisms in terms of secrecy. However, the ProVerif-based approach limits the ability to improve understanding of security beyond verification results. To overcome these limitations, we re-model the same protocol using an Isabelle formalism that generates sound animation, enabling interactive and automated formal verification of security protocols. Our modelling and verification framework is generic and highly configurable, supporting both cryptography and PLS. For the same protocol, we have conducted a comprehensive analysis (secrecy and authenticity in four different eavesdropper locations under both passive and active attacks) using our new web interface. Our findings not only successfully reproduce and reinforce previous results on secrecy but also reveal an uncommon but expected outcome: authenticity is preserved across all examined scenarios, even in cases where secrecy is compromised. We have proposed a PLS-based Diffie-Hellman protocol that integrates watermarking and jamming, and our analysis shows that it is secure for deriving a session key with required authentication. These highlight the advantages of our novel approach, demonstrating its robustness in formally verifying security properties beyond conventional methods.

摘要: 形式验证对于确保安全协议抵御对抗攻击的稳健性至关重要。Needham-Schroeder协议是一种基础认证机制，已被广泛研究，包括其与水印和干扰等物理层安全（SCS）技术的集成。最近的研究使用ProVerif来验证这些机制的保密性。然而，基于ProVerif的方法限制了提高对验证结果之外的安全性理解的能力。为了克服这些限制，我们使用Isabelle形式主义重新建模相同的协议，该形式主义生成声音动画，从而实现安全协议的交互式和自动化形式验证。我们的建模和验证框架是通用的且高度可配置的，支持加密技术和最大限度地支持。对于同一协议，我们使用新的网络界面进行了全面的分析（被动和主动攻击下四个不同窃听者位置的保密性和真实性）。我们的发现不仅成功地复制和强化了之前的保密结果，而且揭示了一个不寻常但预期的结果：在所有检查的场景中，真实性都得到了保留，即使在保密性受到损害的情况下。我们提出了一种基于PL的迪夫-赫尔曼协议，集成了水印和干扰，我们的分析表明，它对于推导具有所需认证的会话密钥是安全的。这些凸显了我们的新型方法的优势，证明了其在正式验证安全属性方面的鲁棒性，超出了传统方法。



## **14. Enhancing Resilience for IoE: A Perspective of Networking-Level Safeguard**

增强IoE的弹性：网络级保障的视角 cs.CR

To be published in IEEE Network Magazine, 2026

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20504v1) [paper-pdf](http://arxiv.org/pdf/2508.20504v1)

**Authors**: Guan-Yan Yang, Jui-Ning Chen, Farn Wang, Kuo-Hui Yeh

**Abstract**: The Internet of Energy (IoE) integrates IoT-driven digital communication with power grids to enable efficient and sustainable energy systems. Still, its interconnectivity exposes critical infrastructure to sophisticated cyber threats, including adversarial attacks designed to bypass traditional safeguards. Unlike general IoT risks, IoE threats have heightened public safety consequences, demanding resilient solutions. From the networking-level safeguard perspective, we propose a Graph Structure Learning (GSL)-based safeguards framework that jointly optimizes graph topology and node representations to resist adversarial network model manipulation inherently. Through a conceptual overview, architectural discussion, and case study on a security dataset, we demonstrate GSL's superior robustness over representative methods, offering practitioners a viable path to secure IoE networks against evolving attacks. This work highlights the potential of GSL to enhance the resilience and reliability of future IoE networks for practitioners managing critical infrastructure. Lastly, we identify key open challenges and propose future research directions in this novel research area.

摘要: 能源互联网（IoE）将物联网驱动的数字通信与电网集成，以实现高效和可持续的能源系统。尽管如此，其互连性仍使关键基础设施面临复杂的网络威胁，包括旨在绕过传统保障措施的对抗性攻击。与一般的物联网风险不同，IoE威胁加剧了公共安全后果，需要弹性解决方案。从网络级保障的角度，我们提出了一个基于图结构学习（GSL）的保障框架，该框架联合优化图布局和节点表示，以本质上抵抗对抗性网络模型操纵。通过对安全数据集的概念概述、架构讨论和案例研究，我们展示了GSL优于代表性方法的卓越稳健性，为从业者提供了一条保护IoE网络免受不断发展的攻击的可行途径。这项工作强调了GSL为管理关键基础设施的从业者增强未来IoE网络的弹性和可靠性的潜力。最后，我们确定了关键的开放挑战，并提出了这个新颖研究领域的未来研究方向。



## **15. When Memory Becomes a Vulnerability: Towards Multi-turn Jailbreak Attacks against Text-to-Image Generation Systems**

当内存成为漏洞时：针对文本到图像生成系统的多回合越狱攻击 cs.CV

This work proposes a multi-turn jailbreak attack against real-world  chat-based T2I generation systems that intergrate memory mechanism. It also  constructed a simulation system, with considering three industrial-grade  memory mechanisms, 7 kinds of safety filters (both input and output)

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2504.20376v2) [paper-pdf](http://arxiv.org/pdf/2504.20376v2)

**Authors**: Shiqian Zhao, Jiayang Liu, Yiming Li, Runyi Hu, Xiaojun Jia, Wenshu Fan, Xinfeng Li, Jie Zhang, Wei Dong, Tianwei Zhang, Luu Anh Tuan

**Abstract**: Modern text-to-image (T2I) generation systems (e.g., DALL$\cdot$E 3) exploit the memory mechanism, which captures key information in multi-turn interactions for faithful generation. Despite its practicality, the security analyses of this mechanism have fallen far behind. In this paper, we reveal that it can exacerbate the risk of jailbreak attacks. Previous attacks fuse the unsafe target prompt into one ultimate adversarial prompt, which can be easily detected or lead to the generation of non-unsafe images due to under- or over-detoxification. In contrast, we propose embedding the malice at the inception of the chat session in memory, addressing the above limitations.   Specifically, we propose Inception, the first multi-turn jailbreak attack against real-world text-to-image generation systems that explicitly exploits their memory mechanisms. Inception is composed of two key modules: segmentation and recursion. We introduce Segmentation, a semantic-preserving method that generates multi-round prompts. By leveraging NLP analysis techniques, we design policies to decompose a prompt, together with its malicious intent, according to sentence structure, thereby evading safety filters. Recursion further addresses the challenge posed by unsafe sub-prompts that cannot be separated through simple segmentation. It firstly expands the sub-prompt, then invokes segmentation recursively. To facilitate multi-turn adversarial prompts crafting, we build VisionFlow, an emulation T2I system that integrates two-stage safety filters and industrial-grade memory mechanisms. The experiment results show that Inception successfully allures unsafe image generation, surpassing the SOTA by a 20.0\% margin in attack success rate. We also conduct experiments on the real-world commercial T2I generation platforms, further validating the threats of Inception in practice.

摘要: 现代文本到图像（T2 I）生成系统（例如，DALL$\csot $E 3）利用记忆机制，该机制捕获多回合交互中的关键信息，以供忠实生成。尽管该机制具有实用性，但对该机制的安全分析却远远落后。在本文中，我们揭示了它会加剧越狱攻击的风险。之前的攻击将不安全的目标提示融合为一个最终的对抗提示，可以很容易地检测到这一点，或者由于解毒不足或过度而导致生成不安全的图像。相比之下，我们建议将聊天会话开始时的恶意嵌入到内存中，以解决上述限制。   具体来说，我们提出了Incept，这是第一个针对现实世界的文本到图像生成系统的多回合越狱攻击，该系统明确利用了其记忆机制。Incion由两个关键模块组成：分段和回归。我们引入了Segmentation，这是一种生成多轮提示的语义保留方法。通过利用NLP分析技术，我们设计策略根据句子结构分解提示及其恶意意图，从而规避安全过滤器。回归进一步解决了无法通过简单分段分离的不安全子提示所带来的挑战。它首先扩展子提示，然后循环调用分段。为了促进多回合对抗提示的制作，我们构建了VisionFlow，这是一个集成两级安全过滤器和工业级存储机制的仿真T2 I系统。实验结果表明，Incion成功诱导了不安全的图像生成，攻击成功率超过SOTA 20.0%。我们还在现实世界的商业T2 I一代平台上进行了实验，进一步验证了Incept在实践中的威胁。



## **16. Safe-Control: A Safety Patch for Mitigating Unsafe Content in Text-to-Image Generation Models**

Safe-Control：用于缓解文本到图像生成模型中不安全内容的安全补丁 cs.CV

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21099v1) [paper-pdf](http://arxiv.org/pdf/2508.21099v1)

**Authors**: Xiangtao Meng, Yingkai Dong, Ning Yu, Li Wang, Zheng Li, Shanqing Guo

**Abstract**: Despite the advancements in Text-to-Image (T2I) generation models, their potential for misuse or even abuse raises serious safety concerns. Model developers have made tremendous efforts to introduce safety mechanisms that can address these concerns in T2I models. However, the existing safety mechanisms, whether external or internal, either remain susceptible to evasion under distribution shifts or require extensive model-specific adjustments. To address these limitations, we introduce Safe-Control, an innovative plug-and-play safety patch designed to mitigate unsafe content generation in T2I models. Using data-driven strategies and safety-aware conditions, Safe-Control injects safety control signals into the locked T2I model, acting as an update in a patch-like manner. Model developers can also construct various safety patches to meet the evolving safety requirements, which can be flexibly merged into a single, unified patch. Its plug-and-play design further ensures adaptability, making it compatible with other T2I models of similar denoising architecture. We conduct extensive evaluations on six diverse and public T2I models. Empirical results highlight that Safe-Control is effective in reducing unsafe content generation across six diverse T2I models with similar generative architectures, yet it successfully maintains the quality and text alignment of benign images. Compared to seven state-of-the-art safety mechanisms, including both external and internal defenses, Safe-Control significantly outperforms all baselines in reducing unsafe content generation. For example, it reduces the probability of unsafe content generation to 7%, compared to approximately 20% for most baseline methods, under both unsafe prompts and the latest adversarial attacks.

摘要: 尽管文本到图像（T2 I）生成模型取得了进步，但它们被滥用甚至滥用的可能性引发了严重的安全问题。模型开发人员做出了巨大努力来引入可以解决T2 I模型中这些问题的安全机制。然而，现有的安全机制，无论是外部的还是内部的，要么仍然容易在分配转移下被规避，要么需要针对特定模型的广泛调整。为了解决这些限制，我们引入了Safe-Control，这是一种创新的即插即用安全补丁，旨在减轻T2 I模型中的不安全内容生成。Safe-Control使用数据驱动策略和安全意识条件，将安全控制信号注入锁定的T2 I模型，以类似补丁的方式充当更新。模型开发人员还可以构建各种安全补丁来满足不断变化的安全要求，这些补丁可以灵活地合并到单个、统一的补丁中。其即插即用设计进一步确保了适应性，使其与类似去噪架构的其他T2 I型号兼容。我们对六种多样化的公共T2 I模型进行了广泛的评估。经验结果强调，Safe-Control可以有效减少具有相似生成架构的六种不同T2 I模型中的不安全内容生成，但它成功地保持了良性图像的质量和文本对齐。与七种最先进的安全机制（包括外部和内部防御）相比，Safe-Control在减少不安全内容生成方面显着优于所有基线。例如，在不安全提示和最新的对抗性攻击下，它将不安全内容生成的可能性降低到7%，而大多数基线方法的可能性约为20%。



## **17. Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs**

一次毒药，永远拒绝：重新调整在LLM中注入偏见的对齐 cs.LG

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20333v1) [paper-pdf](http://arxiv.org/pdf/2508.20333v1)

**Authors**: Md Abdullah Al Mamun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications.

摘要: 大型语言模型（LLM）通过训练它们拒绝回答有害或不安全的提示来满足道德标准和安全要求。在本文中，我们展示了对手如何利用LLM的一致性来植入偏见，或实施有针对性的审查，而不会降低模型对无关主题的响应能力。具体来说，我们提出了颠覆性对齐注入（SAI），这是一种毒害攻击，利用对齐机制来触发对对手预定义的特定主题或查询的拒绝。尽管通过过度对齐引发拒绝可能并不奇怪，但我们展示了如何利用这种拒绝来向模型中注入偏见。令人惊讶的是，SAI回避了最先进的中毒防御，包括LLM状态取证，以及旨在检测FL环境中中毒的强大聚合技术。我们通过说明这种攻击对LLM支持的应用程序管道的端到端影响来证明这种攻击的实际危险。对于ChatDoctor等基于聊天的应用程序来说，存在1%的数据中毒，系统拒绝回答针对目标种族类别的医疗保健问题，导致高度偏见（$\Delta DP$为23%）。我们还表明，在其他NLP任务中也可能会引发偏见：对于一个简历选择管道来说，拒绝汇总来自所选大学的简历，选择中会出现高偏见（$\Delta DP$为27%）。其他9个基于聊天的下游应用程序会产生更高的偏差（$\Delta DP$~38%）。



## **18. Differentially Private Federated Quantum Learning via Quantum Noise**

通过量子噪音的差异化私有联邦量子学习 quant-ph

This paper has been accepted at 2025 IEEE International Conference on  Quantum Computing and Engineering (QCE)

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20310v1) [paper-pdf](http://arxiv.org/pdf/2508.20310v1)

**Authors**: Atit Pokharel, Ratun Rahman, Shaba Shaon, Thomas Morris, Dinh C. Nguyen

**Abstract**: Quantum federated learning (QFL) enables collaborative training of quantum machine learning (QML) models across distributed quantum devices without raw data exchange. However, QFL remains vulnerable to adversarial attacks, where shared QML model updates can be exploited to undermine information privacy. In the context of noisy intermediate-scale quantum (NISQ) devices, a key question arises: How can inherent quantum noise be leveraged to enforce differential privacy (DP) and protect model information during training and communication? This paper explores a novel DP mechanism that harnesses quantum noise to safeguard quantum models throughout the QFL process. By tuning noise variance through measurement shots and depolarizing channel strength, our approach achieves desired DP levels tailored to NISQ constraints. Simulations demonstrate the framework's effectiveness by examining the relationship between differential privacy budget and noise parameters, as well as the trade-off between security and training accuracy. Additionally, we demonstrate the framework's robustness against an adversarial attack designed to compromise model performance using adversarial examples, with evaluations based on critical metrics such as accuracy on adversarial examples, confidence scores for correct predictions, and attack success rates. The results reveal a tunable trade-off between privacy and robustness, providing an efficient solution for secure QFL on NISQ devices with significant potential for reliable quantum computing applications.

摘要: 量子联邦学习（QFL）支持跨分布式量子设备协作训练量子机器学习（QML）模型，无需原始数据交换。然而，QFL仍然容易受到对抗攻击，共享的QML模型更新可能被利用来破坏信息隐私。在有噪音的中等规模量子（NISQ）设备的背景下，出现了一个关键问题：如何利用固有量子噪音来在训练和通信期间实施差异隐私（DP）并保护模型信息？本文探索了一种新型的DP机制，该机制利用量子噪音在整个QFL过程中保护量子模型。通过通过测量镜头和去极化通道强度来调整噪音方差，我们的方法可以实现针对NISQ约束定制的所需DP水平。模拟通过检查差异隐私预算和噪音参数之间的关系以及安全性和训练准确性之间的权衡来证明该框架的有效性。此外，我们还展示了该框架对旨在使用对抗性示例损害模型性能的对抗性攻击的稳健性，评估基于关键指标，例如对抗性示例的准确性、正确预测的置信度分数和攻击成功率。结果揭示了隐私和稳健性之间的可调权衡，为NISQ设备上的安全QFL提供了一种有效的解决方案，具有可靠的量子计算应用的巨大潜力。



## **19. CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors**

CoCoTen：通过上下文共现张量的潜在空间特征检测大型语言模型的对抗性输入 cs.CL

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.02997v3) [paper-pdf](http://arxiv.org/pdf/2508.02997v3)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models.

摘要: 大型语言模型（LLM）在许多应用中的广泛使用标志着研究和实践的重大进步。然而，它们的复杂性和难以理解的性质使它们容易受到攻击，尤其是旨在产生有害反应的越狱。为了应对这些威胁，开发强大的检测方法对于安全可靠地使用LLM至关重要。本文使用上下文共生矩阵来研究这个检测问题，该结构因其在数据稀缺环境中的有效性而被公认。我们提出了一种利用上下文同现矩阵和张量的潜在空间特征的新型方法，以有效识别对抗和越狱提示。我们的评估表明，这种方法仅使用0.5%的标记提示即可获得显着的0.83分，比基线提高了96.6%。这一结果凸显了我们所学习模式的力量，尤其是当标记数据稀缺时。与基线模型相比，我们的方法也明显更快，加速范围为2.3至128.4倍。



## **20. Network-Level Prompt and Trait Leakage in Local Research Agents**

本地研究代理的网络级提示和特征泄露 cs.CR

under review

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20282v1) [paper-pdf](http://arxiv.org/pdf/2508.20282v1)

**Authors**: Hyejun Jeong, Mohammadreze Teymoorianfard, Abhinav Kumar, Amir Houmansadr, Eugene Badasarian

**Abstract**: We show that Web and Research Agents (WRAs) -- language model-based systems that investigate complex topics on the Internet -- are vulnerable to inference attacks by passive network adversaries such as ISPs. These agents could be deployed \emph{locally} by organizations and individuals for privacy, legal, or financial purposes. Unlike sporadic web browsing by humans, WRAs visit $70{-}140$ domains with distinguishable timing correlations, enabling unique fingerprinting attacks.   Specifically, we demonstrate a novel prompt and user trait leakage attack against WRAs that only leverages their network-level metadata (i.e., visited IP addresses and their timings). We start by building a new dataset of WRA traces based on user search queries and queries generated by synthetic personas. We define a behavioral metric (called OBELS) to comprehensively assess similarity between original and inferred prompts, showing that our attack recovers over 73\% of the functional and domain knowledge of user prompts. Extending to a multi-session setting, we recover up to 19 of 32 latent traits with high accuracy. Our attack remains effective under partial observability and noisy conditions. Finally, we discuss mitigation strategies that constrain domain diversity or obfuscate traces, showing negligible utility impact while reducing attack effectiveness by an average of 29\%.

摘要: 我们表明，Web和研究代理（WRA）--调查互联网上复杂主题的基于语言模型的系统--很容易受到ISP等被动网络对手的推理攻击。这些代理可以由组织和个人出于隐私、法律或财务目的\{本地}部署。与人类零星的网络浏览不同，WRA访问具有可区分的时间相关性的价值70 {-}140美元的域名，从而实现独特的指纹攻击。   具体来说，我们展示了针对WRA的新型提示和用户特征泄露攻击，该攻击仅利用其网络级元数据（即，访问的IP地址及其时间）。我们首先根据用户搜索查询和合成人物角色生成的查询构建新的WRA痕迹数据集。我们定义了一个行为指标（称为OBELS）来全面评估原始提示和推断提示之间的相似性，表明我们的攻击恢复了用户提示73%以上的功能和领域知识。扩展到多会话设置，我们可以高准确性地恢复32个潜在特征中的多达19个。我们的攻击在部分可观察性和噪音条件下仍然有效。最后，我们讨论了限制域多样性或混淆痕迹的缓解策略，显示出可忽略的实用性影响，同时将攻击有效性平均降低29%。



## **21. Adversarial Manipulation of Reasoning Models using Internal Representations**

使用内部表示的推理模型的对抗性操纵 cs.CL

Accepted to the ICML 2025 Workshop on Reliable and Responsible  Foundation Models (R2FM). 20 pages, 12 figures

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.03167v2) [paper-pdf](http://arxiv.org/pdf/2507.03167v2)

**Authors**: Kureha Yamaguchi, Benjamin Etheridge, Andy Arditi

**Abstract**: Reasoning models generate chain-of-thought (CoT) tokens before their final output, but how this affects their vulnerability to jailbreak attacks remains unclear. While traditional language models make refusal decisions at the prompt-response boundary, we find evidence that DeepSeek-R1-Distill-Llama-8B makes these decisions within its CoT generation. We identify a linear direction in activation space during CoT token generation that predicts whether the model will refuse or comply -- termed the "caution" direction because it corresponds to cautious reasoning patterns in the generated text. Ablating this direction from model activations increases harmful compliance, effectively jailbreaking the model. We additionally show that intervening only on CoT token activations suffices to control final outputs, and that incorporating this direction into prompt-based attacks improves success rates. Our findings suggest that the chain-of-thought itself is a promising new target for adversarial manipulation in reasoning models. Code available at https://github.com/ky295/reasoning-manipulation.

摘要: 推理模型在最终输出之前生成思想链（CoT）代币，但这如何影响其对越狱攻击的脆弱性尚不清楚。虽然传统的语言模型在拒绝-响应边界上做出拒绝决定，但我们发现有证据表明DeepSeek-R1-Distill-Llama-8B在其CoT生成中做出了这些决定。我们在CoT令牌生成过程中确定了激活空间中的线性方向，该方向预测模型是拒绝还是遵守-称为“谨慎”方向，因为它对应于生成文本中的谨慎推理模式。从模型激活中移除这个方向会增加有害的遵从性，有效地越狱模型。此外，我们还表明，仅干预CoT令牌激活就足以控制最终输出，并且将此方向纳入基于令牌的攻击可以提高成功率。我们的研究结果表明，思维链本身是推理模型中对抗性操纵的一个有前途的新目标。代码可访问https://github.com/ky295/reasoning-manipulation。



## **22. Scaling Decentralized Learning with FLock**

利用Flock扩展去中心化学习 cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.15349v2) [paper-pdf](http://arxiv.org/pdf/2507.15349v2)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.

摘要: 由于集中控制的不足以及分散式方案的大量计算和通信负担，大型语言模型（LLM）的微调受到阻碍。虽然典型的标准联邦学习（FL）支持数据隐私，但中央服务器要求会创建单点攻击和中毒攻击的脆弱性。将这一方向的结果推广到异类、无信任环境中的70 B参数模型已被证明是一个巨大但未突破的瓶颈。本文介绍了Flock，这是一个用于安全高效协作LLM微调的去中心化框架。Flock将基于区块链的信任层与经济激励相结合，用安全、可审计的协议取代了中央聚合器，用于不受信任方之间的合作。我们首次对在安全、多域、去中心化的环境中微调70 B LLM进行了实证验证。我们的实验表明，Flock框架可以抵御后门中毒攻击，这些攻击会损害标准FL优化器并促进协同知识转移。由此产生的模型显示对抗性攻击成功率降低了>68%。全局模型还展示了卓越的跨域泛化能力，优于在自己的专业数据上孤立训练的模型。



## **23. Cell-Free Massive MIMO-Based Physical-Layer Authentication**

无细胞大规模基于MMO的物理层认证 eess.SP

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19931v1) [paper-pdf](http://arxiv.org/pdf/2508.19931v1)

**Authors**: Isabella W. G. da Silva, Zahra Mobini, Hien Quoc Ngo, Michail Matthaiou

**Abstract**: In this paper, we exploit the cell-free massive multiple-input multiple-output (CF-mMIMO) architecture to design a physical-layer authentication (PLA) framework that can simultaneously authenticate multiple distributed users across the coverage area. Our proposed scheme remains effective even in the presence of active adversaries attempting impersonation attacks to disrupt the authentication process. Specifically, we introduce a tag-based PLA CFmMIMO system, wherein the access points (APs) first estimate their channels with the legitimate users during an uplink training phase. Subsequently, a unique secret key is generated and securely shared between each user and the APs. We then formulate a hypothesis testing problem and derive a closed-form expression for the probability of detection for each user in the network. Numerical results validate the effectiveness of the proposed approach, demonstrating that it maintains a high detection probability even as the number of users in the system increases.

摘要: 本文利用无单元大规模多输入多输出（CF-mMMO）架构设计了一个物理层认证（PLA）框架，该框架可以同时对覆盖区域内的多个分布式用户进行认证。即使存在试图模仿攻击以破坏身份验证过程的活跃对手，我们提出的方案仍然有效。具体来说，我们引入了一种基于标签的PLA CFmMMO系统，其中接入点（AP）首先在上行链路训练阶段估计其与合法用户的信道。随后，生成唯一的秘密密钥并在每个用户和AP之间安全共享。然后，我们制定一个假设测试问题，并推导出网络中每个用户的检测概率的封闭形式表达。数值结果验证了所提出方法的有效性，表明即使系统中用户数量增加，它也能保持高检测概率。



## **24. When AIOps Become "AI Oops": Subverting LLM-driven IT Operations via Telemetry Manipulation**

当AIops成为“AI Oops”：通过远程操纵颠覆LLM驱动的IT运营 cs.CR

v0.2

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.06394v2) [paper-pdf](http://arxiv.org/pdf/2508.06394v2)

**Authors**: Dario Pasquini, Evgenios M. Kornaropoulos, Giuseppe Ateniese, Omer Akgul, Athanasios Theocharis, Petros Efstathopoulos

**Abstract**: AI for IT Operations (AIOps) is transforming how organizations manage complex software systems by automating anomaly detection, incident diagnosis, and remediation. Modern AIOps solutions increasingly rely on autonomous LLM-based agents to interpret telemetry data and take corrective actions with minimal human intervention, promising faster response times and operational cost savings.   In this work, we perform the first security analysis of AIOps solutions, showing that, once again, AI-driven automation comes with a profound security cost. We demonstrate that adversaries can manipulate system telemetry to mislead AIOps agents into taking actions that compromise the integrity of the infrastructure they manage. We introduce techniques to reliably inject telemetry data using error-inducing requests that influence agent behavior through a form of adversarial reward-hacking; plausible but incorrect system error interpretations that steer the agent's decision-making. Our attack methodology, AIOpsDoom, is fully automated--combining reconnaissance, fuzzing, and LLM-driven adversarial input generation--and operates without any prior knowledge of the target system.   To counter this threat, we propose AIOpsShield, a defense mechanism that sanitizes telemetry data by exploiting its structured nature and the minimal role of user-generated content. Our experiments show that AIOpsShield reliably blocks telemetry-based attacks without affecting normal agent performance.   Ultimately, this work exposes AIOps as an emerging attack vector for system compromise and underscores the urgent need for security-aware AIOps design.

摘要: IT运营人工智能（AIops）正在通过自动化异常检测、事件诊断和修复来改变组织管理复杂软件系统的方式。现代AIops解决方案越来越依赖基于LLM的自主代理来解释遥感数据并以最少的人为干预采取纠正措施，从而承诺更快的响应时间并节省运营成本。   在这项工作中，我们对AIops解决方案进行了首次安全分析，再次表明人工智能驱动的自动化带来了巨大的安全成本。我们证明，对手可以操纵系统遥感来误导AIops代理采取损害其管理基础设施完整性的行动。我们引入了使用导致错误的请求可靠地注入遥感数据的技术，这些请求通过一种对抗性奖励黑客的形式影响代理的行为;看似合理但不正确的系统错误解释来指导代理的决策。我们的攻击方法AIOpsDoom是完全自动化的--结合了侦察、模糊处理和LLM驱动的对抗输入生成--并且在不了解目标系统的情况下运行。   为了应对这一威胁，我们提出了AIOpsShield，这是一种防御机制，通过利用遥感数据的结构化性质和用户生成内容的最小作用来净化遥感数据。我们的实验表明，AIOpsShield可以可靠地阻止基于远程测量的攻击，而不会影响正常的代理性能。   最终，这项工作暴露了AIops作为系统危害的新兴攻击载体，并强调了对安全意识的AIops设计的迫切需要。



## **25. DATABench: Evaluating Dataset Auditing in Deep Learning from an Adversarial Perspective**

Databench：从对抗角度评估深度学习中的数据集审计 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.05622v2) [paper-pdf](http://arxiv.org/pdf/2507.05622v2)

**Authors**: Shuo Shao, Yiming Li, Mengren Zheng, Zhiyang Hu, Yukun Chen, Boheng Li, Yu He, Junfeng Guo, Dacheng Tao, Zhan Qin

**Abstract**: The widespread application of Deep Learning across diverse domains hinges critically on the quality and composition of training datasets. However, the common lack of disclosure regarding their usage raises significant privacy and copyright concerns. Dataset auditing techniques, which aim to determine if a specific dataset was used to train a given suspicious model, provide promising solutions to addressing these transparency gaps. While prior work has developed various auditing methods, their resilience against dedicated adversarial attacks remains largely unexplored. To bridge the gap, this paper initiates a comprehensive study evaluating dataset auditing from an adversarial perspective. We start with introducing a novel taxonomy, classifying existing methods based on their reliance on internal features (IF) (inherent to the data) versus external features (EF) (artificially introduced for auditing). Subsequently, we formulate two primary attack types: evasion attacks, designed to conceal the use of a dataset, and forgery attacks, intending to falsely implicate an unused dataset. Building on the understanding of existing methods and attack objectives, we further propose systematic attack strategies: decoupling, removal, and detection for evasion; adversarial example-based methods for forgery. These formulations and strategies lead to our new benchmark, DATABench, comprising 17 evasion attacks, 5 forgery attacks, and 9 representative auditing methods. Extensive evaluations using DATABench reveal that none of the evaluated auditing methods are sufficiently robust or distinctive under adversarial settings. These findings underscore the urgent need for developing a more secure and reliable dataset auditing method capable of withstanding sophisticated adversarial manipulation. Code is available at https://github.com/shaoshuo-ss/DATABench.

摘要: 深度学习在不同领域的广泛应用关键取决于训练数据集的质量和组成。然而，普遍缺乏对其使用情况的披露，引发了严重的隐私和版权问题。数据集审计技术旨在确定特定数据集是否用于训练给定的可疑模型，为解决这些透明度差距提供了有希望的解决方案。虽然之前的工作已经开发了各种审计方法，但它们对专门对抗攻击的弹性在很大程度上仍未被探索。为了弥合这一差距，本文发起了一项全面的研究，从对抗的角度评估数据集审计。我们首先引入一种新颖的分类法，根据现有方法对内部特征（IF）（数据固有的）和外部特征（EF）（人为引入以进行审计）的依赖来对现有方法进行分类。随后，我们制定了两种主要的攻击类型：逃避攻击（旨在隐藏数据集的使用）和伪造攻击（旨在错误地暗示未使用的数据集）。在对现有方法和攻击目标的理解的基础上，我们进一步提出了系统性攻击策略：脱钩、删除和检测逃避;基于对抗性示例的伪造方法。这些公式和策略导致了我们的新基准Databench，其中包括17种规避攻击、5种伪造攻击和9种代表性审计方法。使用Databench进行的广泛评估表明，在对抗环境下，所评估的审计方法都不够稳健或独特。这些发现凸显了开发一种能够承受复杂对抗操纵的更安全、更可靠的数据集审计方法的迫切需要。代码可在https://github.com/shaoshuo-ss/DATABench上获取。



## **26. Secure Set-based State Estimation for Safety-Critical Applications under Adversarial Attacks on Sensors**

传感器对抗攻击下安全关键应用的安全基于集的状态估计 eess.SY

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2309.05075v3) [paper-pdf](http://arxiv.org/pdf/2309.05075v3)

**Authors**: M. Umar B. Niazi, Michelle S. Chong, Amr Alanwar, Karl H. Johansson

**Abstract**: Set-based state estimation provides guaranteed state inclusion certificates that are crucial for the safety verification of dynamical systems. However, when system sensors are subject to cyberattacks, maintaining both safety and security guarantees becomes a fundamental challenge that existing point-based secure state estimation methods cannot adequately address due to their inherent inability to provide state inclusion certificates. This paper introduces a novel approach that simultaneously ensures safety guarantees through guaranteed state inclusion and security guarantees against sensor attacks, without imposing conservative restrictions on system operation. We propose a Secure Set-based State Estimation (S3E) algorithm that maintains the true system state within the estimated set under sensor attacks, provided the initialization set contains the initial state and the system remains observable from the uncompromised sensor subset. The algorithm gives the estimated set as a collection of constrained zonotopes (agreement sets), which can be employed as robust certificates for verifying whether the system adheres to safety constraints. Furthermore, we demonstrate that the estimated set remains unaffected by attack signals of sufficiently large magnitude and also establish sufficient conditions for attack detection, identification, and filtering. This compels the attacker to only inject signals of small magnitudes to evade detection, thus preserving the accuracy of the estimated set. To address the computational complexity of the algorithm, we offer several strategies for complexity-performance trade-offs. The efficacy of the proposed algorithm is illustrated through several examples, including its application to a three-story building model.

摘要: 基于集的状态估计提供了有保证的状态包含证书，这对于动态系统的安全验证至关重要。然而，当系统传感器遭受网络攻击时，维持安全性和安全保障成为现有的基于点的安全状态估计方法无法充分解决的根本挑战，因为它们固有地无法提供状态包含证书。本文介绍了一种新颖的方法，通过保证的状态包容性和针对传感器攻击的安全保证同时确保安全保证，而不会对系统操作施加保守限制。我们提出了一种基于安全集的状态估计（S3 E）算法，只要初始化集包含初始状态并且系统保持可从未受损害的传感器子集观察，该算法可以在传感器攻击下在估计集中维持真实的系统状态。该算法将估计集提供为受约束的分区（协议集）的集合，其可以用作鲁棒证书，用于验证系统是否遵守安全约束。此外，我们证明估计集不受足够大幅度的攻击信号的影响，并且还为攻击检测、识别和过滤建立了充分的条件。这迫使攻击者只注入小幅度的信号来逃避检测，从而保持估计集的准确性。为了解决算法的计算复杂性，我们提供了多种复杂性与性能权衡的策略。通过几个例子，包括它的应用程序的三层建筑模型所提出的算法的有效性进行说明。



## **27. Safety Alignment Should Be Made More Than Just A Few Attention Heads**

安全调整不应仅仅是一些注意力 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19697v1) [paper-pdf](http://arxiv.org/pdf/2508.19697v1)

**Authors**: Chao Huang, Zefeng Zhang, Juewei Yue, Quangang Li, Chuang Zhang, Tingwen Liu

**Abstract**: Current safety alignment for large language models(LLMs) continues to present vulnerabilities, given that adversarial prompting can effectively bypass their safety measures.Our investigation shows that these safety mechanisms predominantly depend on a limited subset of attention heads: removing or ablating these heads can severely compromise model safety. To identify and evaluate these safety-critical components, we introduce RDSHA, a targeted ablation method that leverages the model's refusal direction to pinpoint attention heads mostly responsible for safety behaviors. Further analysis shows that existing jailbreak attacks exploit this concentration by selectively bypassing or manipulating these critical attention heads. To address this issue, we propose AHD, a novel training strategy designed to promote the distributed encoding of safety-related behaviors across numerous attention heads. Experimental results demonstrate that AHD successfully distributes safety-related capabilities across more attention heads. Moreover, evaluations under several mainstream jailbreak attacks show that models trained with AHD exhibit considerably stronger safety robustness, while maintaining overall functional utility.

摘要: 目前的大型语言模型（LLM）的安全对齐仍然存在漏洞，因为对抗性提示可以有效地绕过它们的安全措施。我们的调查表明，这些安全机制主要依赖于有限的注意头子集：删除或消融这些头会严重危及模型安全。为了识别和评估这些安全关键组件，我们引入了RDSHA，这是一种有针对性的消融方法，它利用模型的拒绝方向来确定主要负责安全行为的注意力。进一步的分析表明，现有的越狱攻击通过选择性地绕过或操纵这些关键注意力头部来利用这种集中。为了解决这个问题，我们提出了AHD，一种新的训练策略，旨在促进分布式编码的安全相关的行为在众多的注意头。实验结果表明，AHD成功地将安全相关功能分配给更多的注意力头。此外，在几种主流越狱攻击下的评估表明，用AHD训练的模型表现出更强的安全鲁棒性，同时保持整体功能效用。



## **28. ProARD: progressive adversarial robustness distillation: provide wide range of robust students**

ProARD：渐进式对抗稳健性蒸馏：提供广泛的稳健学生 cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2506.07666v3) [paper-pdf](http://arxiv.org/pdf/2506.07666v3)

**Authors**: Seyedhamidreza Mousavi, Seyedali Mousavi, Masoud Daneshtalab

**Abstract**: Adversarial Robustness Distillation (ARD) has emerged as an effective method to enhance the robustness of lightweight deep neural networks against adversarial attacks. Current ARD approaches have leveraged a large robust teacher network to train one robust lightweight student. However, due to the diverse range of edge devices and resource constraints, current approaches require training a new student network from scratch to meet specific constraints, leading to substantial computational costs and increased CO2 emissions. This paper proposes Progressive Adversarial Robustness Distillation (ProARD), enabling the efficient one-time training of a dynamic network that supports a diverse range of accurate and robust student networks without requiring retraining. We first make a dynamic deep neural network based on dynamic layers by encompassing variations in width, depth, and expansion in each design stage to support a wide range of architectures. Then, we consider the student network with the largest size as the dynamic teacher network. ProARD trains this dynamic network using a weight-sharing mechanism to jointly optimize the dynamic teacher network and its internal student networks. However, due to the high computational cost of calculating exact gradients for all the students within the dynamic network, a sampling mechanism is required to select a subset of students. We show that random student sampling in each iteration fails to produce accurate and robust students.

摘要: 对抗鲁棒性蒸馏（ARD）已成为增强轻量级深度神经网络抵御对抗攻击鲁棒性的有效方法。当前的ARD方法利用了一个强大的教师网络来培训一个强大的轻量级学生。然而，由于边缘设备的多样性和资源限制，当前的方法需要从头开始训练新的学生网络以满足特定的限制，从而导致巨大的计算成本和二氧化碳排放量增加。本文提出了渐进对抗鲁棒蒸馏（ProARD），可以对动态网络进行高效的一次性训练，该网络支持各种准确且稳健的学生网络，而无需再培训。我们首先基于动态层构建动态深度神经网络，通过涵盖每个设计阶段的宽度、深度和扩展的变化，以支持广泛的架构。然后，我们将规模最大的学生网络视为动态教师网络。ProARD使用权重共享机制训练这个动态网络，以联合优化动态教师网络及其内部学生网络。然而，由于计算动态网络中所有学生的精确梯度的计算成本很高，因此需要采样机制来选择学生的子集。我们表明，每次迭代中的随机学生抽样无法产生准确和稳健的学生。



## **29. R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning**

R-TPT：通过测试时提示调优提高视觉语言模型的对抗鲁棒性 cs.LG

CVPR 2025 (Corrected the results on the Aircraft dataset)

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2504.11195v2) [paper-pdf](http://arxiv.org/pdf/2504.11195v2)

**Authors**: Lijun Sheng, Jian Liang, Zilei Wang, Ran He

**Abstract**: Vision-language models (VLMs), such as CLIP, have gained significant popularity as foundation models, with numerous fine-tuning methods developed to enhance performance on downstream tasks. However, due to their inherent vulnerability and the common practice of selecting from a limited set of open-source models, VLMs suffer from a higher risk of adversarial attacks than traditional vision models. Existing defense techniques typically rely on adversarial fine-tuning during training, which requires labeled data and lacks of flexibility for downstream tasks. To address these limitations, we propose robust test-time prompt tuning (R-TPT), which mitigates the impact of adversarial attacks during the inference stage. We first reformulate the classic marginal entropy objective by eliminating the term that introduces conflicts under adversarial conditions, retaining only the pointwise entropy minimization. Furthermore, we introduce a plug-and-play reliability-based weighted ensembling strategy, which aggregates useful information from reliable augmented views to strengthen the defense. R-TPT enhances defense against adversarial attacks without requiring labeled training data while offering high flexibility for inference tasks. Extensive experiments on widely used benchmarks with various attacks demonstrate the effectiveness of R-TPT. The code is available in https://github.com/TomSheng21/R-TPT.

摘要: CLIP等视觉语言模型（VLM）作为基础模型已受到广泛欢迎，并开发了多种微调方法来增强下游任务的性能。然而，由于其固有的脆弱性以及从有限的开源模型集中进行选择的常见做法，VLM比传统视觉模型面临更高的对抗攻击风险。现有的防御技术通常依赖于训练期间的对抗微调，这需要标记数据并且缺乏下游任务的灵活性。为了解决这些限制，我们提出了鲁棒的测试时即时调优（R-TPT），它可以减轻推理阶段对抗性攻击的影响。我们首先通过消除在对抗条件下引入冲突的术语来重新制定经典的边际熵目标，只保留逐点的熵最小化。此外，我们引入了一种即插即用的、基于可靠性的加权集成策略，该策略从可靠的增强视图中聚合有用信息以加强防御。R-TPT增强了对对抗攻击的防御，而不需要标记的训练数据，同时为推理任务提供高度灵活性。对广泛使用的具有各种攻击的基准进行了大量实验，证明了R-TPT的有效性。该代码可在https://github.com/TomSheng21/R-TPT上找到。



## **30. PromptKeeper: Safeguarding System Prompts for LLMs**

PretKeeper：保护LLM的系统预算 cs.CR

Accepted to the Findings of EMNLP 2025. 17 pages, 6 figures, 3 tables

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2412.13426v3) [paper-pdf](http://arxiv.org/pdf/2412.13426v3)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: System prompts are widely used to guide the outputs of large language models (LLMs). These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a defense mechanism designed to safeguard system prompts by tackling two core challenges: reliably detecting leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon leakage detected, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.

摘要: 系统提示被广泛用于指导大型语言模型（LLM）的输出。这些提示通常包含业务逻辑和敏感信息，因此对其的保护至关重要。然而，对抗性甚至常规用户查询都可能利用LLM漏洞来暴露这些隐藏的提示。为了解决这个问题，我们提出了Inbox Keeper，这是一种防御机制，旨在通过解决两个核心挑战来保护系统提示：可靠地检测泄漏和减轻发生泄漏时的侧通道漏洞。通过将检测视为假设测试问题，SpectKeeper有效地识别显式和微妙的泄漏。检测到泄漏后，它会使用虚拟提示重新生成响应，确保在不存在泄漏时输出与典型交互没有区别。EntKeeper确保针对通过对抗性或常规查询的即时提取攻击提供强大的保护，同时在良性用户交互期间保留对话能力和运行时效率。



## **31. A Systematic Survey of Model Extraction Attacks and Defenses: State-of-the-Art and Perspectives**

模型提取攻击和防御的系统性调查：最新技术水平和观点 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.15031v2) [paper-pdf](http://arxiv.org/pdf/2508.15031v2)

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong

**Abstract**: Machine learning (ML) models have significantly grown in complexity and utility, driving advances across multiple domains. However, substantial computational resources and specialized expertise have historically restricted their wide adoption. Machine-Learning-as-a-Service (MLaaS) platforms have addressed these barriers by providing scalable, convenient, and affordable access to sophisticated ML models through user-friendly APIs. While this accessibility promotes widespread use of advanced ML capabilities, it also introduces vulnerabilities exploited through Model Extraction Attacks (MEAs). Recent studies have demonstrated that adversaries can systematically replicate a target model's functionality by interacting with publicly exposed interfaces, posing threats to intellectual property, privacy, and system security. In this paper, we offer a comprehensive survey of MEAs and corresponding defense strategies. We propose a novel taxonomy that classifies MEAs according to attack mechanisms, defense approaches, and computing environments. Our analysis covers various attack techniques, evaluates their effectiveness, and highlights challenges faced by existing defenses, particularly the critical trade-off between preserving model utility and ensuring security. We further assess MEAs within different computing paradigms and discuss their technical, ethical, legal, and societal implications, along with promising directions for future research. This systematic survey aims to serve as a valuable reference for researchers, practitioners, and policymakers engaged in AI security and privacy. Additionally, we maintain an online repository continuously updated with related literature at https://github.com/kzhao5/ModelExtractionPapers.

摘要: 机器学习（ML）模型的复杂性和实用性显着增长，推动了多个领域的进步。然而，大量的计算资源和专业知识历来限制了它们的广泛采用。机器学习即服务（MLaaz）平台通过用户友好的API提供对复杂ML模型的可扩展、方便且经济实惠的访问，从而解决了这些障碍。虽然这种可访问性促进了高级ML功能的广泛使用，但它也引入了通过模型提取攻击（MEAs）利用的漏洞。最近的研究表明，对手可以通过与公开的界面交互来系统性地复制目标模型的功能，从而对知识产权、隐私和系统安全构成威胁。在本文中，我们对多边环境协定和相应的防御策略进行了全面的调查。我们提出了一种新颖的分类法，根据攻击机制、防御方法和计算环境对多边环境进行分类。我们的分析涵盖了各种攻击技术，评估了它们的有效性，并强调了现有防御所面临的挑战，特别是保留模型实用性和确保安全性之间的关键权衡。我们进一步评估不同计算范式中的多边环境协定，并讨论其技术、道德、法律和社会影响，以及未来研究的有希望的方向。这项系统性调查旨在为从事人工智能安全和隐私的研究人员、从业者和政策制定者提供宝贵的参考。此外，我们还在https://github.com/kzhao5/ModelExtractionPapers上维护了一个在线知识库，不断更新相关文献。



## **32. Servant, Stalker, Predator: How An Honest, Helpful, And Harmless (3H) Agent Unlocks Adversarial Skills**

仆人、跟踪者、掠夺者：诚实、乐于助人、无害（3 H）特工如何释放对抗技能 cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19500v1) [paper-pdf](http://arxiv.org/pdf/2508.19500v1)

**Authors**: David Noever

**Abstract**: This paper identifies and analyzes a novel vulnerability class in Model Context Protocol (MCP) based agent systems. The attack chain describes and demonstrates how benign, individually authorized tasks can be orchestrated to produce harmful emergent behaviors. Through systematic analysis using the MITRE ATLAS framework, we demonstrate how 95 agents tested with access to multiple services-including browser automation, financial analysis, location tracking, and code deployment-can chain legitimate operations into sophisticated attack sequences that extend beyond the security boundaries of any individual service. These red team exercises survey whether current MCP architectures lack cross-domain security measures necessary to detect or prevent a large category of compositional attacks. We present empirical evidence of specific attack chains that achieve targeted harm through service orchestration, including data exfiltration, financial manipulation, and infrastructure compromise. These findings reveal that the fundamental security assumption of service isolation fails when agents can coordinate actions across multiple domains, creating an exponential attack surface that grows with each additional capability. This research provides a barebones experimental framework that evaluate not whether agents can complete MCP benchmark tasks, but what happens when they complete them too well and optimize across multiple services in ways that violate human expectations and safety constraints. We propose three concrete experimental directions using the existing MCP benchmark suite.

摘要: 本文识别并分析了基于模型上下文协议（MAO）的代理系统中的一种新型漏洞类别。攻击链描述并演示了如何精心策划良性的、单独授权的任务来产生有害的紧急行为。通过使用MITRE ATLAS框架的系统分析，我们展示了95个经过测试的代理如何访问多种服务（包括浏览器自动化、财务分析、位置跟踪和代码部署）可以将合法操作链接到复杂的攻击序列中，这些攻击序列超出了任何单个服务的安全边界。这些红队练习调查当前的LCP架构是否缺乏检测或防止大型组合攻击所需的跨域安全措施。我们提供了特定攻击链的经验证据，这些攻击链通过服务编排（包括数据泄露、金融操纵和基础设施损害）来实现有针对性的伤害。这些发现表明，当代理可以协调多个域之间的操作时，服务隔离的基本安全假设就会失败，从而创建指数级攻击面，并且随着每一个额外的能力而增长。这项研究提供了一个基本的实验框架，该框架不是评估代理是否能够完成LCP基准任务，而是评估当他们完成得太好并以违反人类期望和安全约束的方式在多个服务中进行优化时会发生什么。我们使用现有的LCP基准套件提出了三个具体的实验方向。



## **33. PoolFlip: A Multi-Agent Reinforcement Learning Security Environment for Cyber Defense**

PoolFlip：用于网络防御的多智能体强化学习安全环境 cs.LG

Accepted at GameSec 2025

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19488v1) [paper-pdf](http://arxiv.org/pdf/2508.19488v1)

**Authors**: Xavier Cadet, Simona Boboila, Sie Hendrata Dharmawan, Alina Oprea, Peter Chin

**Abstract**: Cyber defense requires automating defensive decision-making under stealthy, deceptive, and continuously evolving adversarial strategies. The FlipIt game provides a foundational framework for modeling interactions between a defender and an advanced adversary that compromises a system without being immediately detected. In FlipIt, the attacker and defender compete to control a shared resource by performing a Flip action and paying a cost. However, the existing FlipIt frameworks rely on a small number of heuristics or specialized learning techniques, which can lead to brittleness and the inability to adapt to new attacks. To address these limitations, we introduce PoolFlip, a multi-agent gym environment that extends the FlipIt game to allow efficient learning for attackers and defenders. Furthermore, we propose Flip-PSRO, a multi-agent reinforcement learning (MARL) approach that leverages population-based training to train defender agents equipped to generalize against a range of unknown, potentially adaptive opponents. Our empirical results suggest that Flip-PSRO defenders are $2\times$ more effective than baselines to generalize to a heuristic attack not exposed in training. In addition, our newly designed ownership-based utility functions ensure that Flip-PSRO defenders maintain a high level of control while optimizing performance.

摘要: 网络防御需要在隐秘、欺骗性和不断演变的对抗策略下实现防御决策的自动化。FlipIt游戏提供了一个基础框架，用于建模防御者和高级对手之间的交互，这种交互可以在不被立即检测到的情况下危害系统。在FlipIt中，攻击者和防御者通过执行翻转动作并支付费用来竞争控制共享资源。然而，现有的FlipIt框架依赖于少数启发式或专业学习技术，这可能会导致脆弱性和无法适应新的攻击。为了解决这些限制，我们引入了PoolFlip，这是一个多代理健身房环境，它扩展了FlipIt游戏，使攻击者和防御者能够进行高效学习。此外，我们提出了Flip-PSRO，这是一种多智能体强化学习（MARL）方法，它利用基于人群的训练来训练防御者智能体，能够针对一系列未知的、潜在的适应性对手进行概括。我们的经验结果表明，Flip-PSRO防御者在推广到训练中未暴露的启发式攻击方面比基线有效2倍。此外，我们新设计的基于所有权的实用程序功能可确保Flip-PSRO防御者在优化性能的同时保持高水平的控制。



## **34. ReLATE+: Unified Framework for Adversarial Attack Detection, Classification, and Resilient Model Selection in Time-Series Classification**

ReLATE+：时间序列分类中对抗性攻击检测、分类和弹性模型选择的统一框架 cs.CR

Under review at IEEE TSMC Journal. arXiv admin note: text overlap  with arXiv:2503.07882

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19456v1) [paper-pdf](http://arxiv.org/pdf/2508.19456v1)

**Authors**: Cagla Ipek Kocal, Onat Gungor, Tajana Rosing, Baris Aksanli

**Abstract**: Minimizing computational overhead in time-series classification, particularly in deep learning models, presents a significant challenge due to the high complexity of model architectures and the large volume of sequential data that must be processed in real time. This challenge is further compounded by adversarial attacks, emphasizing the need for resilient methods that ensure robust performance and efficient model selection. To address this challenge, we propose ReLATE+, a comprehensive framework that detects and classifies adversarial attacks, adaptively selects deep learning models based on dataset-level similarity, and thus substantially reduces retraining costs relative to conventional methods that do not leverage prior knowledge, while maintaining strong performance. ReLATE+ first checks whether the incoming data is adversarial and, if so, classifies the attack type, using this insight to identify a similar dataset from a repository and enable the reuse of the best-performing associated model. This approach ensures strong performance while reducing the need for retraining, and it generalizes well across different domains with varying data distributions and feature spaces. Experiments show that ReLATE+ reduces computational overhead by an average of 77.68%, enhancing adversarial resilience and streamlining robust model selection, all without sacrificing performance, within 2.02% of Oracle.

摘要: 由于模型架构的高复杂性和必须实时处理的大量顺序数据，最大限度地减少时间序列分类中的计算负担，特别是深度学习模型中的计算负担带来了重大挑战。对抗性攻击进一步加剧了这一挑战，强调了对确保稳健性能和高效模型选择的弹性方法的需要。为了应对这一挑战，我们提出了ReLATE+，这是一个全面的框架，可以检测和分类对抗性攻击，根据互联网级别的相似性自适应地选择深度学习模型，从而相对于不利用先验知识的传统方法大幅降低了再培训成本，同时保持强劲的性能。ReLATE+首先检查输入的数据是否是对抗性的，如果是，则对攻击类型进行分类，使用此见解从存储库中识别类似的数据集，并启用性能最佳的关联模型的重用。这种方法确保了强大的性能，同时减少了再培训的需要，并且它可以很好地在具有不同数据分布和特征空间的不同领域中推广。实验表明，ReLATE+平均降低了77.68%的计算负担，增强了对抗弹性并简化了稳健的模型选择，而所有这些都不会牺牲性能，仅在Oracle的2.02%之内。



## **35. On Surjectivity of Neural Networks: Can you elicit any behavior from your model?**

关于神经网络的满摄性：你能从你的模型中引出任何行为吗？ cs.LG

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19445v1) [paper-pdf](http://arxiv.org/pdf/2508.19445v1)

**Authors**: Haozhe Jiang, Nika Haghtalab

**Abstract**: Given a trained neural network, can any specified output be generated by some input? Equivalently, does the network correspond to a function that is surjective? In generative models, surjectivity implies that any output, including harmful or undesirable content, can in principle be generated by the networks, raising concerns about model safety and jailbreak vulnerabilities. In this paper, we prove that many fundamental building blocks of modern neural architectures, such as networks with pre-layer normalization and linear-attention modules, are almost always surjective. As corollaries, widely used generative frameworks, including GPT-style transformers and diffusion models with deterministic ODE solvers, admit inverse mappings for arbitrary outputs. By studying surjectivity of these modern and commonly used neural architectures, we contribute a formalism that sheds light on their unavoidable vulnerability to a broad class of adversarial attacks.

摘要: 给定一个经过训练的神经网络，任何指定的输出都可以由某些输入生成吗？同样，网络是否对应于满射函数？在生成模型中，主观性意味着任何输出，包括有害或不受欢迎的内容，原则上都可以由网络生成，这引发了对模型安全性和越狱漏洞的担忧。在本文中，我们证明了现代神经架构的许多基本构建模块，例如具有预层规范化和线性注意力模块的网络，几乎总是满射的。作为推论，广泛使用的生成式框架（包括GPT式转换器和具有确定性ODE解算器的扩散模型）允许任意输出的逆映射。通过研究这些现代和常用的神经架构的主观性，我们提出了一种形式主义，揭示了它们不可避免地容易受到一类对抗攻击的脆弱性。



## **36. Attackers Strike Back? Not Anymore -- An Ensemble of RL Defenders Awakens for APT Detection**

攻击者反击？不再是--RL捍卫者群体为APT检测觉醒 cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19072v1) [paper-pdf](http://arxiv.org/pdf/2508.19072v1)

**Authors**: Sidahmed Benabderrahmane, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) represent a growing menace to modern digital infrastructure. Unlike traditional cyberattacks, APTs are stealthy, adaptive, and long-lasting, often bypassing signature-based detection systems. This paper introduces a novel framework for APT detection that unites deep learning, reinforcement learning (RL), and active learning into a cohesive, adaptive defense system. Our system combines auto-encoders for latent behavioral encoding with a multi-agent ensemble of RL-based defenders, each trained to distinguish between benign and malicious process behaviors. We identify a critical challenge in existing detection systems: their static nature and inability to adapt to evolving attack strategies. To this end, our architecture includes multiple RL agents (Q-Learning, PPO, DQN, adversarial defenders), each analyzing latent vectors generated by an auto-encoder. When any agent is uncertain about its decision, the system triggers an active learning loop to simulate expert feedback, thus refining decision boundaries. An ensemble voting mechanism, weighted by each agent's performance, ensures robust final predictions.

摘要: 高级持续威胁（APT）对现代数字基础设施构成了日益严重的威胁。与传统的网络攻击不同，APT具有隐蔽性、适应性和持久性，通常绕过基于签名的检测系统。本文介绍了一种新颖的APT检测框架，该框架将深度学习、强化学习（RL）和主动学习结合到一个有凝聚力的自适应防御系统中。我们的系统将用于潜在行为编码的自动编码器与基于RL的防御者的多代理集成相结合，每个防御者都经过训练以区分良性和恶意进程行为。我们发现了现有检测系统中的一个关键挑战：它们的静态性质以及无法适应不断变化的攻击策略。为此，我们的架构包括多个RL代理（Q-Learning、PPO、DQN、对抗防御者），每个代理都分析自动编码器生成的潜在载体。当任何代理人对其决策不确定时，系统会触发主动学习循环来模拟专家反馈，从而细化决策边界。由每个代理的表现加权的集成投票机制确保了稳健的最终预测。



## **37. Beyond-Diagonal RIS: Adversarial Channels and Optimality of Low-Complexity Architectures**

Beyond-Diagonal RIS：对抗渠道和低复杂度架构的最佳性 eess.SP

\copyright 2025 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19000v1) [paper-pdf](http://arxiv.org/pdf/2508.19000v1)

**Authors**: Atso Iivanainen, Robin Rajamäki, Visa Koivunen

**Abstract**: Beyond-diagonal reconfigurable intelligent surfaces (BD-RISs) have recently gained attention as an enhancement to conventional RISs. BD-RISs allow optimizing not only the phase, but also the amplitude responses of their discrete surface elements by introducing adjustable inter-element couplings. Various BD-RIS architectures have been proposed to optimally trade off between average performance and complexity of the architecture. However, little attention has been paid to worst-case performance. This paper characterizes novel sets of adversarial channels for which certain low-complexity BD-RIS architectures have suboptimal performance in terms of received signal power at an intended communications user. Specifically, we consider two recent BD-RIS models: the so-called group-connected and tree-connected architecture. The derived adversarial channel sets reveal new surprising connections between the two architectures. We validate our analytical results numerically, demonstrating that adversarial channels can cause a significant performance loss. Our results pave the way towards efficient BD-RIS designs that are robust to adversarial propagation conditions and malicious attacks.

摘要: 超对角线可重构智能表面（BD-RISs）最近作为传统RISs的增强而受到关注。通过引入可调节的元件间耦合，BD-RIS不仅可以优化其离散表面元件的相响应，还可以优化其离散表面元件的幅度响应。人们提出了各种BD-RIS架构，以在平均性能和架构复杂性之间进行最佳权衡。然而，人们很少关注最坏情况的性能。本文描述了一组新型对抗性通道的特征，对于这些通道，某些低复杂性的BD-RIS架构在预期通信用户的接收信号功率方面具有次优的性能。具体来说，我们考虑了两种最近的BD-RIS模型：所谓的组连接和树连接架构。衍生的对抗通道集揭示了两种架构之间新的令人惊讶的联系。我们通过数字方式验证了我们的分析结果，证明对抗性通道可能会导致显着的性能损失。我们的结果为高效的BD-RIS设计铺平了道路，该设计对对抗传播条件和恶意攻击具有鲁棒性。



## **38. The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization**

基于LLM的数据重建的双刃剑：理解和缓解词级差异隐私文本清理中的上下文漏洞 cs.CR

15 pages, 4 figures, 8 tables. Accepted to WPES @ CCS 2025

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18976v1) [paper-pdf](http://arxiv.org/pdf/2508.18976v1)

**Authors**: Stephen Meisenbacher, Alexandra Klymenko, Andreea-Elena Bodea, Florian Matthes

**Abstract**: Differentially private text sanitization refers to the process of privatizing texts under the framework of Differential Privacy (DP), providing provable privacy guarantees while also empirically defending against adversaries seeking to harm privacy. Despite their simplicity, DP text sanitization methods operating at the word level exhibit a number of shortcomings, among them the tendency to leave contextual clues from the original texts due to randomization during sanitization $\unicode{x2013}$ this we refer to as $\textit{contextual vulnerability}$. Given the powerful contextual understanding and inference capabilities of Large Language Models (LLMs), we explore to what extent LLMs can be leveraged to exploit the contextual vulnerability of DP-sanitized texts. We expand on previous work not only in the use of advanced LLMs, but also in testing a broader range of sanitization mechanisms at various privacy levels. Our experiments uncover a double-edged sword effect of LLM-based data reconstruction attacks on privacy and utility: while LLMs can indeed infer original semantics and sometimes degrade empirical privacy protections, they can also be used for good, to improve the quality and privacy of DP-sanitized texts. Based on our findings, we propose recommendations for using LLM data reconstruction as a post-processing step, serving to increase privacy protection by thinking adversarially.

摘要: 差异隐私文本清理是指在差异隐私（DP）框架下将文本私有化的过程，提供可证明的隐私保证，同时还根据经验防御试图损害隐私的对手。尽管它们很简单，但在词级操作的DP文本清理方法表现出许多缺点，其中包括由于清理期间的随机性，倾向于从原始文本中留下上下文线索$\unicode{x2013}$我们将其称为$\textit{contextual vulnerability}$。鉴于大型语言模型（LLM）强大的上下文理解和推理能力，我们探索可以在多大程度上利用LLM来利用DP清理文本的上下文脆弱性。我们不仅扩展了之前的工作，还扩展了高级LLM的使用，还扩展了各种隐私级别的更广泛的清理机制。我们的实验揭示了基于LLM的数据重建攻击对隐私和实用性的双刃剑效应：虽然LLM确实可以推断原始语义，有时会降低经验隐私保护，但它们也可以被永久使用，以提高DP净化文本的质量和隐私。根据我们的研究结果，我们提出了使用LLM数据重建作为后处理步骤的建议，通过敌对思维来增加隐私保护。



## **39. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

语音认证和反欺骗系统威胁调查 cs.CR

This paper will be submitted to the Computer Science Review

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.16843v2) [paper-pdf](http://arxiv.org/pdf/2508.16843v2)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.

摘要: 语音认证发生了重大变化，从依赖手工声学特征的传统系统到可以提取稳健的说话者嵌入的深度学习模型。这一进步扩大了其在金融、智能设备、执法等领域的应用。然而，随着采用率的增加，威胁也随之增加。本调查全面回顾了针对语音认证系统（PAS）和反欺骗对策（CM）的现代威胁格局，包括数据中毒、对抗性、深度伪造和对抗性欺骗攻击。我们按时间顺序追踪语音认证的发展，并研究漏洞如何随着技术进步而演变。对于每种类型的攻击，我们总结了方法论，强调常用的数据集，比较性能和局限性，并使用广泛接受的分类法组织现有文献。通过强调新出现的风险和公开挑战，本调查旨在支持开发更安全、更有弹性的语音认证系统。



## **40. EnerSwap: Large-Scale, Privacy-First Automated Market Maker for V2G Energy Trading**

EnerSwap：大规模、隐私优先的V2G能源交易自动化做市商 cs.CR

11 pages, 7 figures, 1 table, 1 algorithm, Paper accepted in 27th  MSWiM Conference

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18942v1) [paper-pdf](http://arxiv.org/pdf/2508.18942v1)

**Authors**: Ahmed Mounsf Rafik Bendada, Yacine Ghamri-Doudane

**Abstract**: With the rapid growth of Electric Vehicle (EV) technology, EVs are destined to shape the future of transportation. The large number of EVs facilitates the development of the emerging vehicle-to-grid (V2G) technology, which realizes bidirectional energy exchanges between EVs and the power grid. This has led to the setting up of electricity markets that are usually confined to a small geographical location, often with a small number of participants. Usually, these markets are manipulated by intermediaries responsible for collecting bids from prosumers, determining the market-clearing price, incorporating grid constraints, and accounting for network losses. While centralized models can be highly efficient, they grant excessive power to the intermediary by allowing them to gain exclusive access to prosumers \textquotesingle price preferences. This opens the door to potential market manipulation and raises significant privacy concerns for users, such as the location of energy providers. This lack of protection exposes users to potential risks, as untrustworthy servers and malicious adversaries can exploit this information to infer trading activities and real identities. This work proposes a secure, decentralized exchange market built on blockchain technology, utilizing a privacy-preserving Automated Market Maker (AMM) model to offer open and fair, and equal access to traders, and mitigates the most common trading-manipulation attacks. Additionally, it incorporates a scalable architecture based on geographical dynamic sharding, allowing for efficient resource allocation and improved performance as the market grows.

摘要: 随着电动汽车（EV）技术的快速发展，电动汽车注定会塑造交通的未来。电动汽车的大量使用促进了新兴的汽车转网（V2 G）技术的发展，该技术实现了电动汽车与电网之间的双向能量交换。这导致了电力市场的建立，这些市场通常仅限于一个较小的地理位置，参与者往往很少。通常，这些市场由负责收集生产者出价、确定市场出清价格、纳入电网限制并核算网络损失的中介机构操纵。虽然集中式模型可能非常高效，但它们通过允许中间商独家访问产品消费者\文本引用单一价格偏好而赋予了他们过多的权力。这为潜在的市场操纵打开了大门，并给用户带来了重大的隐私问题，例如能源提供商的位置。这种缺乏保护使用户面临潜在风险，因为不值得信赖的服务器和恶意对手可以利用这些信息来推断交易活动和真实身份。这项工作提出了一个基于区块链技术的安全、去中心化的交易所市场，利用保护隐私的自动化做市商（AMM）模型为交易员提供开放、公平和平等的准入机会，并减轻最常见的交易操纵攻击。此外，它还结合了基于地理动态分片的可扩展架构，随着市场的增长，可以高效地分配资源并提高性能。



## **41. Hidden Tail: Adversarial Image Causing Stealthy Resource Consumption in Vision-Language Models**

隐藏的尾巴：视觉语言模型中导致隐性资源消耗的敌对图像 cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18805v1) [paper-pdf](http://arxiv.org/pdf/2508.18805v1)

**Authors**: Rui Zhang, Zihan Wang, Tianli Yang, Hongwei Li, Wenbo Jiang, Qingchuan Zhao, Yang Liu, Guowen Xu

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed in real-world applications, but their high inference cost makes them vulnerable to resource consumption attacks. Prior attacks attempt to extend VLM output sequences by optimizing adversarial images, thereby increasing inference costs. However, these extended outputs often introduce irrelevant abnormal content, compromising attack stealthiness. This trade-off between effectiveness and stealthiness poses a major limitation for existing attacks. To address this challenge, we propose \textit{Hidden Tail}, a stealthy resource consumption attack that crafts prompt-agnostic adversarial images, inducing VLMs to generate maximum-length outputs by appending special tokens invisible to users. Our method employs a composite loss function that balances semantic preservation, repetitive special token induction, and suppression of the end-of-sequence (EOS) token, optimized via a dynamic weighting strategy. Extensive experiments show that \textit{Hidden Tail} outperforms existing attacks, increasing output length by up to 19.2$\times$ and reaching the maximum token limit, while preserving attack stealthiness. These results highlight the urgent need to improve the robustness of VLMs against efficiency-oriented adversarial threats. Our code is available at https://github.com/zhangrui4041/Hidden_Tail.

摘要: 视觉语言模型（VLM）越来越多地部署在现实世界的应用程序中，但其高推理成本使其容易受到资源消耗攻击。先前的攻击试图通过优化对抗图像来扩展VLM输出序列，从而增加推理成本。然而，这些扩展输出通常会引入不相关的异常内容，从而损害攻击的隐蔽性。有效性和隐蔽性之间的这种权衡对现有攻击构成了重大限制。为了应对这一挑战，我们提出了\textit{Hidden Tail}，这是一种隐形的资源消耗攻击，它可以制作不可知的对抗图像，通过附加用户不可见的特殊令牌来诱导VLM生成最大长度的输出。我们的方法采用了一个复合损失函数，平衡语义保存，重复的特殊令牌感应，并通过动态加权策略优化的序列结束（EOS）令牌的抑制。大量的实验表明，\textit{Hidden Tail}优于现有的攻击，将输出长度增加了19.2$\times$，达到了最大令牌限制，同时保持了攻击的隐蔽性。这些结果强调了迫切需要提高VLM对效率导向的对抗性威胁的鲁棒性。我们的代码可在https://github.com/zhangrui4041/Hidden_Tail上获取。



## **42. FLAegis: A Two-Layer Defense Framework for Federated Learning Against Poisoning Attacks**

FLAegis：针对中毒攻击的联邦学习的两层防御框架 cs.LG

15 pages, 5 tables, and 5 figures

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18737v1) [paper-pdf](http://arxiv.org/pdf/2508.18737v1)

**Authors**: Enrique Mármol Campos, Aurora González Vidal, José Luis Hernández Ramos, Antonio Skarmeta

**Abstract**: Federated Learning (FL) has become a powerful technique for training Machine Learning (ML) models in a decentralized manner, preserving the privacy of the training datasets involved. However, the decentralized nature of FL limits the visibility of the training process, relying heavily on the honesty of participating clients. This assumption opens the door to malicious third parties, known as Byzantine clients, which can poison the training process by submitting false model updates. Such malicious clients may engage in poisoning attacks, manipulating either the dataset or the model parameters to induce misclassification. In response, this study introduces FLAegis, a two-stage defensive framework designed to identify Byzantine clients and improve the robustness of FL systems. Our approach leverages symbolic time series transformation (SAX) to amplify the differences between benign and malicious models, and spectral clustering, which enables accurate detection of adversarial behavior. Furthermore, we incorporate a robust FFT-based aggregation function as a final layer to mitigate the impact of those Byzantine clients that manage to evade prior defenses. We rigorously evaluate our method against five poisoning attacks, ranging from simple label flipping to adaptive optimization-based strategies. Notably, our approach outperforms state-of-the-art defenses in both detection precision and final model accuracy, maintaining consistently high performance even under strong adversarial conditions.

摘要: 联邦学习（FL）已经成为一种以分散方式训练机器学习（ML）模型的强大技术，保护了所涉及的训练数据集的隐私。然而，FL的分散性质限制了培训过程的可见性，严重依赖参与客户的诚实。这种假设为恶意的第三方打开了大门，这些第三方被称为拜占庭客户端，它们可以通过提交错误的模型更新来毒害训练过程。这种恶意客户端可能参与中毒攻击，操纵数据集或模型参数以引起错误分类。作为回应，本研究引入了FLAegis，这是一个两阶段防御框架，旨在识别拜占庭客户并提高FL系统的稳健性。我们的方法利用符号时间序列变换（NSX）来放大良性和恶意模型之间的差异，并利用谱集群来实现对对抗行为的准确检测。此外，我们还将强大的基于快速傅立叶变换的聚合功能作为最后一层，以减轻那些设法逃避先前防御的拜占庭客户的影响。我们针对五种中毒攻击严格评估了我们的方法，范围从简单的标签翻转到基于自适应优化的策略。值得注意的是，我们的方法在检测精度和最终模型准确性方面都优于最先进的防御，即使在强烈的对抗条件下也能保持一致的高性能。



## **43. UniC-RAG: Universal Knowledge Corruption Attacks to Retrieval-Augmented Generation**

UniC-RAG：对检索增强一代的普遍知识腐败攻击 cs.CR

21 pages, 4 figures

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18652v1) [paper-pdf](http://arxiv.org/pdf/2508.18652v1)

**Authors**: Runpeng Geng, Yanting Wang, Ying Chen, Jinyuan Jia

**Abstract**: Retrieval-augmented generation (RAG) systems are widely deployed in real-world applications in diverse domains such as finance, healthcare, and cybersecurity. However, many studies showed that they are vulnerable to knowledge corruption attacks, where an attacker can inject adversarial texts into the knowledge database of a RAG system to induce the LLM to generate attacker-desired outputs. Existing studies mainly focus on attacking specific queries or queries with similar topics (or keywords). In this work, we propose UniC-RAG, a universal knowledge corruption attack against RAG systems. Unlike prior work, UniC-RAG jointly optimizes a small number of adversarial texts that can simultaneously attack a large number of user queries with diverse topics and domains, enabling an attacker to achieve various malicious objectives, such as directing users to malicious websites, triggering harmful command execution, or launching denial-of-service attacks. We formulate UniC-RAG as an optimization problem and further design an effective solution to solve it, including a balanced similarity-based clustering method to enhance the attack's effectiveness. Our extensive evaluations demonstrate that UniC-RAG is highly effective and significantly outperforms baselines. For instance, UniC-RAG could achieve over 90% attack success rate by injecting 100 adversarial texts into a knowledge database with millions of texts to simultaneously attack a large set of user queries (e.g., 2,000). Additionally, we evaluate existing defenses and show that they are insufficient to defend against UniC-RAG, highlighting the need for new defense mechanisms in RAG systems.

摘要: 检索增强一代（RAG）系统广泛部署在金融、医疗保健和网络安全等不同领域的现实世界应用中。然而，许多研究表明，它们很容易受到知识腐败攻击，攻击者可以将对抗文本注入RAG系统的知识数据库，以诱导LLM生成攻击者所需的输出。现有的研究主要集中在攻击特定查询或具有相似主题（或关键词）的查询上。在这项工作中，我们提出了UniC-RAG，这是一种针对RAG系统的通用知识腐败攻击。与之前的工作不同，UniC-RAG联合优化了少量对抗性文本，这些文本可以同时攻击大量具有不同主题和域的用户查询，使攻击者能够实现各种恶意目标，例如将用户引导到恶意网站、触发有害命令执行或发起拒绝服务攻击。我们将UniC-RAG表述为一个优化问题，并进一步设计一个有效的解决方案来解决它，包括基于平衡相似性的集群方法来增强攻击的有效性。我们的广泛评估表明UniC-RAG非常有效，并且显着优于基线。例如，UniC-RAG可以通过将100个对抗文本注入到具有数百万个文本的知识数据库中来同时攻击大量用户查询（例如，2，000）。此外，我们评估了现有的防御措施，并表明它们不足以防御UniC-RAG，强调了RAG系统中新防御机制的必要性。



## **44. PRISM: Robust VLM Alignment with Principled Reasoning for Integrated Safety in Multimodality**

PRism：与原则推理的鲁棒VLM对齐，以实现多模式中的综合安全 cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18649v1) [paper-pdf](http://arxiv.org/pdf/2508.18649v1)

**Authors**: Nanxi Li, Zhengyue Zhao, Chaowei Xiao

**Abstract**: Safeguarding vision-language models (VLMs) is a critical challenge, as existing methods often suffer from over-defense, which harms utility, or rely on shallow alignment, failing to detect complex threats that require deep reasoning. To this end, we introduce PRISM (Principled Reasoning for Integrated Safety in Multimodality), a system2-like framework that aligns VLMs by embedding a structured, safety-aware reasoning process. Our framework consists of two key components: PRISM-CoT, a dataset that teaches safety-aware chain-of-thought reasoning, and PRISM-DPO, generated via Monte Carlo Tree Search (MCTS) to further refine this reasoning through Direct Preference Optimization to help obtain a delicate safety boundary. Comprehensive evaluations demonstrate PRISM's effectiveness, achieving remarkably low attack success rates including 0.15% on JailbreakV-28K for Qwen2-VL and 90% improvement over the previous best method on VLBreak for LLaVA-1.5. PRISM also exhibits strong robustness against adaptive attacks, significantly increasing computational costs for adversaries, and generalizes effectively to out-of-distribution challenges, reducing attack success rates to just 8.70% on the challenging multi-image MIS benchmark. Remarkably, this robust defense is achieved while preserving, and in some cases enhancing, model utility. To promote reproducibility, we have made our code, data, and model weights available at https://github.com/SaFoLab-WISC/PRISM.

摘要: 保护视觉语言模型（VLM）是一项严峻的挑战，因为现有的方法经常遭受过度防御，从而损害实用性，或者依赖于浅层对齐，无法检测到需要深度推理的复杂威胁。为此，我们引入了PRism（多模式综合安全原则推理），这是一个类似系统2的框架，通过嵌入结构化的安全意识推理过程来对齐VLM。我们的框架由两个关键组件组成：PRISM-CoT，一个教授安全意识思维链推理的数据集，以及PRISM-DPO，通过蒙特卡洛树搜索（MCTS）生成，通过直接偏好优化进一步完善这一推理，以帮助获得微妙的安全边界。全面评估证明了PRISM的有效性，实现了极低的攻击成功率，包括针对Qwen 2-BL的Jailbreak V-28 K的攻击成功率为0.15%，针对LLaVA-1.5的VLBreak的攻击成功率比之前的最佳方法提高了90%。PRISM还对自适应攻击表现出强大的鲁棒性，显著增加了对手的计算成本，并有效地推广到分布外的挑战，在具有挑战性的多图像MIS基准测试中，攻击成功率仅为8.70%。值得注意的是，这种强大的防御是在保持，甚至在某些情况下增强模型效用的同时实现的。为了提高可重复性，我们在https://github.com/SaFoLab-WISC/PRISM上提供了我们的代码、数据和模型权重。



## **45. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**

引导对话动力学，增强抵御多回合越狱攻击的稳健性 cs.CL

23 pages, 10 figures, 11 tables

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2503.00187v2) [paper-pdf](http://arxiv.org/pdf/2503.00187v2)

**Authors**: Hanjiang Hu, Alexander Robey, Changliu Liu

**Abstract**: Large language models (LLMs) are shown to be vulnerable to jailbreaking attacks where adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment, prompt-based steering and lightweight LLM guardrails baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off among safety, helpfulness and over-refusal. Check out the website here https://sites.google.com/view/llm-nbf/home . Our code is available on https://github.com/HanjiangHu/NBF-LLM .

摘要: 事实证明，大型语言模型（LLM）很容易受到越狱攻击，其中对抗性提示旨在引发有害反应。虽然现有的防御措施通过检测和过滤不安全的输入有效地减轻了单回合攻击，但它们无法对抗利用多次交互中的上下文漂移的多回合越狱，从而逐渐导致LLM远离安全行为。为了应对这一挑战，我们提出了一个基于安全控制理论的安全引导框架，确保多回合对话中不变的安全性。我们的方法使用状态空间表示对与LLM的对话进行建模，并引入一种新型的神经屏障函数（NBF）来主动检测和过滤不断变化的上下文中出现的有害查询。我们的方法通过学习一个考虑对抗性查询的安全预测器，在每一轮对话中实现不变的安全性，防止潜在的上下文漂移到越狱。在多个LLM下进行的大量实验表明，我们基于NBF的安全转向优于安全对准，基于转向的转向和轻型LLM护栏基线，为多转向越狱提供更强的防御，同时在安全性，有用性和过度拒绝之间保持更好的权衡。查看网站https://sites.google.com/view/llm-nbf/home。我们的代码可以在https://github.com/HanjiangHu/NBF-LLM上找到。



## **46. Transferring Styles for Reduced Texture Bias and Improved Robustness in Semantic Segmentation Networks**

传输样式以减少纹理偏差并提高语义分割网络中的鲁棒性 cs.CV

accepted at ECAI 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.10239v2) [paper-pdf](http://arxiv.org/pdf/2507.10239v2)

**Authors**: Ben Hamscher, Edgar Heinert, Annika Mütze, Kira Maag, Matthias Rottmann

**Abstract**: Recent research has investigated the shape and texture biases of deep neural networks (DNNs) in image classification which influence their generalization capabilities and robustness. It has been shown that, in comparison to regular DNN training, training with stylized images reduces texture biases in image classification and improves robustness with respect to image corruptions. In an effort to advance this line of research, we examine whether style transfer can likewise deliver these two effects in semantic segmentation. To this end, we perform style transfer with style varying across artificial image areas. Those random areas are formed by a chosen number of Voronoi cells. The resulting style-transferred data is then used to train semantic segmentation DNNs with the objective of reducing their dependence on texture cues while enhancing their reliance on shape-based features. In our experiments, it turns out that in semantic segmentation, style transfer augmentation reduces texture bias and strongly increases robustness with respect to common image corruptions as well as adversarial attacks. These observations hold for convolutional neural networks and transformer architectures on the Cityscapes dataset as well as on PASCAL Context, showing the generality of the proposed method.

摘要: 最近的研究调查了深度神经网络（DNN）在图像分类中的形状和纹理偏差，这些偏差会影响其泛化能力和鲁棒性。研究表明，与常规DNN训练相比，使用风格化图像进行训练可以减少图像分类中的纹理偏差，并提高图像损坏的鲁棒性。为了推进这一领域的研究，我们研究了风格转移是否同样可以在语义分割中产生这两种效果。为此，我们执行风格转移，风格在人工图像区域之间有所不同。这些随机区域由选定数量的Voronoi细胞形成。然后使用生成的风格传输数据来训练语义分割DNN，目标是减少它们对纹理线索的依赖，同时增强它们对基于形状的特征的依赖。在我们的实验中，事实证明，在语义分割中，风格转移增强减少了纹理偏差，并大大提高了针对常见图像损坏和对抗攻击的鲁棒性。这些观察结果适用于Cityscapes数据集和Pascal Content上的卷积神经网络和Transformer架构，显示了所提出方法的通用性。



## **47. Quantum-Classical Hybrid Framework for Zero-Day Time-Push GNSS Spoofing Detection**

用于零日时间推送式全球导航卫星欺骗检测的量子经典混合框架 cs.LG

This work has been submitted to the IEEE Internet of Things Journal  for possible publication

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18085v1) [paper-pdf](http://arxiv.org/pdf/2508.18085v1)

**Authors**: Abyad Enan, Mashrur Chowdhury, Sagar Dasgupta, Mizanur Rahman

**Abstract**: Global Navigation Satellite Systems (GNSS) are critical for Positioning, Navigation, and Timing (PNT) applications. However, GNSS are highly vulnerable to spoofing attacks, where adversaries transmit counterfeit signals to mislead receivers. Such attacks can lead to severe consequences, including misdirected navigation, compromised data integrity, and operational disruptions. Most existing spoofing detection methods depend on supervised learning techniques and struggle to detect novel, evolved, and unseen attacks. To overcome this limitation, we develop a zero-day spoofing detection method using a Hybrid Quantum-Classical Autoencoder (HQC-AE), trained solely on authentic GNSS signals without exposure to spoofed data. By leveraging features extracted during the tracking stage, our method enables proactive detection before PNT solutions are computed. We focus on spoofing detection in static GNSS receivers, which are particularly susceptible to time-push spoofing attacks, where attackers manipulate timing information to induce incorrect time computations at the receiver. We evaluate our model against different unseen time-push spoofing attack scenarios: simplistic, intermediate, and sophisticated. Our analysis demonstrates that the HQC-AE consistently outperforms its classical counterpart, traditional supervised learning-based models, and existing unsupervised learning-based methods in detecting zero-day, unseen GNSS time-push spoofing attacks, achieving an average detection accuracy of 97.71% with an average false negative rate of 0.62% (when an attack occurs but is not detected). For sophisticated spoofing attacks, the HQC-AE attains an accuracy of 98.23% with a false negative rate of 1.85%. These findings highlight the effectiveness of our method in proactively detecting zero-day GNSS time-push spoofing attacks across various stationary GNSS receiver platforms.

摘要: 全球导航卫星系统（GNSS）对于定位、导航和授时（PNT）应用至关重要。然而，全球导航卫星系统非常容易受到欺骗攻击，对手会发送伪造信号来误导接收器。此类攻击可能会导致严重的后果，包括导航错误、数据完整性受损和运营中断。大多数现有的欺骗检测方法依赖于监督学习技术，并且很难检测新颖的、进化的和不可见的攻击。为了克服这一限制，我们使用混合量子经典自动编码器（HQC-AE）开发了一种零日欺骗检测方法，该方法仅根据真实的GPS信号进行训练，而不会暴露于欺骗数据。通过利用在跟踪阶段提取的特征，我们的方法能够在计算PNT解决方案之前进行主动检测。我们重点关注静态GNSS接收器中的欺骗检测，这些接收器特别容易受到时间推送欺骗攻击，攻击者操纵计时信息以在接收器上引发不正确的时间计算。我们针对不同不可见的时间推送欺骗攻击场景来评估我们的模型：简单化、中间化和复杂化。我们的分析表明，HQC-AE在检测零日、不可见的GNSS时间推送欺骗攻击方面始终优于其经典对应物、传统的基于监督学习的模型和现有的基于无监督学习的方法，实现了97.71%的平均检测准确率，平均误报率为0.62%（当攻击发生但未被检测到时）。对于复杂的欺骗攻击，HQC-AE的准确率为98.23%，误报率为1.85%。这些发现凸显了我们的方法在跨各种固定的GNSS接收器平台主动检测零日GNSS时间推送欺骗攻击方面的有效性。



## **48. Robust Federated Learning under Adversarial Attacks via Loss-Based Client Clustering**

通过基于损失的客户端集群实现对抗性攻击下的鲁棒联邦学习 cs.LG

16 pages, 5 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.12672v3) [paper-pdf](http://arxiv.org/pdf/2508.12672v3)

**Authors**: Emmanouil Kritharakis, Dusan Jakovetic, Antonios Makris, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients without sharing private data. We consider FL scenarios wherein FL clients are subject to adversarial (Byzantine) attacks, while the FL server is trusted (honest) and has a trustworthy side dataset. This may correspond to, e.g., cases where the server possesses trusted data prior to federation, or to the presence of a trusted client that temporarily assumes the server role. Our approach requires only two honest participants, i.e., the server and one client, to function effectively, without prior knowledge of the number of malicious clients. Theoretical analysis demonstrates bounded optimality gaps even under strong Byzantine attacks. Experimental results show that our algorithm significantly outperforms standard and robust FL baselines such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum under various attack strategies including label flipping, sign flipping, and Gaussian noise addition across MNIST, FMNIST, and CIFAR-10 benchmarks using the Flower framework.

摘要: 联合学习（FL）支持跨多个客户端的协作模型训练，而无需共享私有数据。我们考虑FL场景，其中FL客户端受到对抗性（拜占庭）攻击，而FL服务器是可信的（诚实的），并有一个值得信赖的侧数据集。这可以对应于，例如，服务器在联合之前拥有受信任数据，或者存在暂时承担服务器角色的受信任客户端的情况。我们的方法只需要两个诚实的参与者，即服务器和一个客户端，在不了解恶意客户端数量的情况下有效运行。理论分析表明，即使在强大的拜占庭攻击下，也存在有限的最优性差距。实验结果表明，我们的算法显着优于标准和强大的FL基线，如平均值，修剪平均值，中位数，克鲁姆，和多克鲁姆下的各种攻击策略，包括标签翻转，符号翻转，高斯噪声添加在MNIST，FMNIST，和CIFAR-10基准使用花框架。



## **49. FedGreed: A Byzantine-Robust Loss-Based Aggregation Method for Federated Learning**

FedGreed：一种用于联邦学习的拜占庭鲁棒的基于损失的聚合方法 cs.LG

8 pages, 4 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18060v1) [paper-pdf](http://arxiv.org/pdf/2508.18060v1)

**Authors**: Emmanouil Kritharakis, Antonios Makris, Dusan Jakovetic, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients while preserving data privacy by keeping local datasets on-device. In this work, we address FL settings where clients may behave adversarially, exhibiting Byzantine attacks, while the central server is trusted and equipped with a reference dataset. We propose FedGreed, a resilient aggregation strategy for federated learning that does not require any assumptions about the fraction of adversarial participants. FedGreed orders clients' local model updates based on their loss metrics evaluated against a trusted dataset on the server and greedily selects a subset of clients whose models exhibit the minimal evaluation loss. Unlike many existing approaches, our method is designed to operate reliably under heterogeneous (non-IID) data distributions, which are prevalent in real-world deployments. FedGreed exhibits convergence guarantees and bounded optimality gaps under strong adversarial behavior. Experimental evaluations on MNIST, FMNIST, and CIFAR-10 demonstrate that our method significantly outperforms standard and robust federated learning baselines, such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum, in the majority of adversarial scenarios considered, including label flipping and Gaussian noise injection attacks. All experiments were conducted using the Flower federated learning framework.

摘要: 联合学习（FL）支持跨多个客户端进行协作模型训练，同时通过将本地数据集保留在设备上来保护数据隐私。在这项工作中，我们解决了FL设置，其中客户端可能表现出敌对行为，表现出拜占庭式攻击，而中央服务器是受信任的并配备了参考数据集。我们提出FedGreed，这是一种针对联邦学习的弹性聚合策略，不需要对对抗参与者的比例进行任何假设。FedGreed根据针对服务器上受信任数据集评估的损失指标来订购客户的本地模型更新，并贪婪地选择其模型表现出最小评估损失的客户子集。与许多现有方法不同，我们的方法旨在在现实世界部署中普遍存在的异类（非IID）数据分布下可靠运行。FedGreed在强对抗行为下表现出收敛保证和有界最优性差距。对MNIST、FMNIST和CIFAR-10的实验评估表明，在所考虑的大多数对抗场景中，我们的方法显着优于标准和稳健的联邦学习基线，例如Mean、Trimmed Mean、Median、Krum和Multi-Krum。所有实验都使用Flower联邦学习框架进行。



## **50. Does simple trump complex? Comparing strategies for adversarial robustness in DNNs**

简单胜过复杂吗？比较DNN中对抗鲁棒性的策略 cs.LG

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.18019v1) [paper-pdf](http://arxiv.org/pdf/2508.18019v1)

**Authors**: William Brooks, Marelie H. Davel, Coenraad Mouton

**Abstract**: Deep Neural Networks (DNNs) have shown substantial success in various applications but remain vulnerable to adversarial attacks. This study aims to identify and isolate the components of two different adversarial training techniques that contribute most to increased adversarial robustness, particularly through the lens of margins in the input space -- the minimal distance between data points and decision boundaries. Specifically, we compare two methods that maximize margins: a simple approach which modifies the loss function to increase an approximation of the margin, and a more complex state-of-the-art method (Dynamics-Aware Robust Training) which builds upon this approach. Using a VGG-16 model as our base, we systematically isolate and evaluate individual components from these methods to determine their relative impact on adversarial robustness. We assess the effect of each component on the model's performance under various adversarial attacks, including AutoAttack and Projected Gradient Descent (PGD). Our analysis on the CIFAR-10 dataset reveals which elements most effectively enhance adversarial robustness, providing insights for designing more robust DNNs.

摘要: 深度神经网络（DNN）在各种应用中取得了巨大成功，但仍然容易受到对抗攻击。这项研究旨在识别和隔离两种不同对抗训练技术中对提高对抗鲁棒性贡献最大的组成部分，特别是通过输入空间中的裕度（数据点和决策边界之间的最小距离）的视角。具体来说，我们比较了两种最大限度地提高利润率的方法：一种修改损失函数以增加利润率逼近的简单方法，以及一种基于这种方法的更复杂的最先进方法（动态感知稳健训练）。使用VGG-16模型作为基础，我们系统地分离和评估这些方法中的各个成分，以确定它们对对抗稳健性的相对影响。我们评估了每个组件在各种对抗攻击（包括AutoAttack和投影梯度下降（PVD））下对模型性能的影响。我们对CIFAR-10数据集的分析揭示了哪些元素最有效地增强了对抗稳健性，为设计更稳健的DNN提供了见解。



