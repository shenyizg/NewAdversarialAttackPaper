# Latest Adversarial Attack Papers
**update at 2025-09-10 10:58:14**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

大型多模式模型的鲁棒适应用于检索增强仇恨模因检测 cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2502.13061v3) [paper-pdf](http://arxiv.org/pdf/2502.13061v3)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL

摘要: 仇恨模因已成为互联网上的一个重要问题，需要强大的自动化检测系统。虽然大型多模式模型（LSYS）在仇恨模因检测方面表现出了希望，但它们面临着显着的挑战，例如次优的性能和有限的域外概括能力。最近的研究进一步揭示了在这种环境下将监督微调（SFT）和上下文学习应用于LSYS时的局限性。为了解决这些问题，我们提出了一个用于仇恨模因检测的鲁棒适应框架，该框架可以增强领域内准确性和跨领域概括性，同时保留Letts的一般视觉语言能力。分析表明，与SFT模型相比，我们的方法在对抗攻击下实现了更好的鲁棒性。对六个模因分类数据集的实验表明，我们的方法实现了最先进的性能，优于更大的代理系统。此外，与标准SFT相比，我们的方法为解释仇恨内容生成了更高质量的理由，增强了模型的可解释性。代码可访问https://github.com/JingbiaoMei/RGCL



## **2. Empirical Security Analysis of Software-based Fault Isolation through Controlled Fault Injection**

通过受控故障注入实现基于软件的故障隔离的经验安全性分析 cs.CR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07757v1) [paper-pdf](http://arxiv.org/pdf/2509.07757v1)

**Authors**: Nils Bars, Lukas Bernhard, Moritz Schloegel, Thorsten Holz

**Abstract**: We use browsers daily to access all sorts of information. Because browsers routinely process scripts, media, and executable code from unknown sources, they form a critical security boundary between users and adversaries. A common attack vector is JavaScript, which exposes a large attack surface due to the sheer complexity of modern JavaScript engines. To mitigate these threats, modern engines increasingly adopt software-based fault isolation (SFI). A prominent example is Google's V8 heap sandbox, which represents the most widely deployed SFI mechanism, protecting billions of users across all Chromium-based browsers and countless applications built on Node.js and Electron. The heap sandbox splits the address space into two parts: one part containing trusted, security-sensitive metadata, and a sandboxed heap containing memory accessible to untrusted code. On a technical level, the sandbox enforces isolation by removing raw pointers and using translation tables to resolve references to trusted objects. Consequently, an attacker cannot corrupt trusted data even with full control of the sandboxed data, unless there is a bug in how code handles data from the sandboxed heap. Despite their widespread use, such SFI mechanisms have seen little security testing.   In this work, we propose a new testing technique that models the security boundary of modern SFI implementations. Following the SFI threat model, we assume a powerful attacker who fully controls the sandbox's memory. We implement this by instrumenting memory loads originating in the trusted domain and accessing untrusted, attacker-controlled sandbox memory. We then inject faults into the loaded data, aiming to trigger memory corruption in the trusted domain. In a comprehensive evaluation, we identify 19 security bugs in V8 that enable an attacker to bypass the sandbox.

摘要: 我们每天使用浏览器来访问各种信息。由于浏览器经常处理来自未知来源的脚本、媒体和可执行代码，因此它们在用户和对手之间形成了关键的安全边界。一个常见的攻击向量是JavaScript，由于现代JavaScript引擎的复杂性，它暴露了一个巨大的攻击面。为了减轻这些威胁，现代发动机越来越多地采用基于软件的故障隔离（SFI）。一个突出的例子是Google的V8堆沙盒，它代表了部署最广泛的SFI机制，保护了所有基于Chromium的浏览器和无数基于Node.js和Electron的应用程序的数十亿用户。堆沙盒将地址空间分为两部分：一部分包含可信的安全敏感元数据，另一部分包含不可信代码可访问的内存。在技术层面上，沙箱通过删除原始指针并使用转换表来解析对可信对象的引用来强制隔离。因此，攻击者即使完全控制沙盒数据也无法破坏受信任的数据，除非代码处理沙盒堆中数据的方式存在错误。尽管它们被广泛使用，但这种SFI机制几乎没有进行过安全测试。   在这项工作中，我们提出了一种新的测试技术，该技术对现代SFI实现的安全边界进行建模。遵循SFI威胁模型，我们假设一个强大的攻击者完全控制沙箱的内存。我们通过检测源自受信任域的内存负载并访问不受信任的、攻击者控制的沙箱内存来实现这一点。然后，我们将错误注入加载的数据中，旨在触发可信域中的内存损坏。在全面评估中，我们发现了V8中的19个安全漏洞，这些漏洞使攻击者能够绕过沙箱。



## **3. Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems**

频谱掩蔽和内插攻击（SMIA）：针对语音认证和反欺骗系统的黑匣子对抗攻击 cs.SD

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07677v1) [paper-pdf](http://arxiv.org/pdf/2509.07677v1)

**Authors**: Kamel Kamel, Hridoy Sankar Dutta, Keshav Sood, Sunil Aryal

**Abstract**: Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial samples that sound authentic while deceiving CMs. We conducted a comprehensive evaluation of our attack against state-of-the-art (SOTA) models across multiple tasks, under simulated real-world conditions. SMIA achieved a strong attack success rate (ASR) of at least 82% against combined VAS/CM systems, at least 97.5% against standalone speaker verification systems, and 100% against countermeasures. These findings conclusively demonstrate that current security postures are insufficient against adaptive adversarial attacks. This work highlights the urgent need for a paradigm shift toward next-generation defenses that employ dynamic, context-aware frameworks capable of evolving with the threat landscape.

摘要: 语音认证系统（PAS）使用独特的声音特征进行验证。他们越来越多地融入银行和医疗保健等高安全领域。尽管它们使用深度学习进行了改进，但它们仍面临来自深度造假和对抗攻击等复杂威胁的严重漏洞。现实语音克隆的出现使检测变得复杂，因为系统很难区分真实音频和合成音频。虽然存在反欺骗对策（CM）来减轻这些风险，但许多对策依赖于静态检测模型，这些模型可以被新型对抗方法绕过，从而留下了关键的安全漏洞。为了证明这一漏洞，我们提出了频谱掩蔽和内插攻击（SMIA），这是一种新颖的方法，可以战略性地操纵人工智能生成的音频的听不见的频率区域。通过改变人耳不可感知区域的声音，SMIA创建听起来真实的对抗样本，同时欺骗CM。我们在模拟现实世界条件下对多个任务中针对最先进（SOTA）模型的攻击进行了全面评估。SMIA在对抗组合式增值服务器/CM系统时，至少达到82%的攻击成功率（ASB），对抗独立说话者验证系统至少达到97.5%，对抗措施至少达到100%。这些发现最终证明，当前的安全姿态不足以抵御适应性对抗攻击。这项工作凸显了向下一代防御范式转变的迫切需要，这些防御采用能够随着威胁格局而演变的动态、上下文感知框架。



## **4. Transferable Direct Prompt Injection via Activation-Guided MCMC Sampling**

通过激活引导MCMC采样的可转移直接即时注射 cs.AI

Accepted to EMNLP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07617v1) [paper-pdf](http://arxiv.org/pdf/2509.07617v1)

**Authors**: Minghui Li, Hao Zhang, Yechao Zhang, Wei Wan, Shengshan Hu, pei Xiaobing, Jing Wang

**Abstract**: Direct Prompt Injection (DPI) attacks pose a critical security threat to Large Language Models (LLMs) due to their low barrier of execution and high potential damage. To address the impracticality of existing white-box/gray-box methods and the poor transferability of black-box methods, we propose an activations-guided prompt injection attack framework. We first construct an Energy-based Model (EBM) using activations from a surrogate model to evaluate the quality of adversarial prompts. Guided by the trained EBM, we employ the token-level Markov Chain Monte Carlo (MCMC) sampling to adaptively optimize adversarial prompts, thereby enabling gradient-free black-box attacks. Experimental results demonstrate our superior cross-model transferability, achieving 49.6% attack success rate (ASR) across five mainstream LLMs and 34.6% improvement over human-crafted prompts, and maintaining 36.6% ASR on unseen task scenarios. Interpretability analysis reveals a correlation between activations and attack effectiveness, highlighting the critical role of semantic patterns in transferable vulnerability exploitation.

摘要: 直接提示注入（DPI）攻击由于其低执行门槛和高潜在危害性，对大型语言模型（LLM）构成了严重的安全威胁。针对现有白盒/灰盒方法的不实用性和黑盒方法的可移植性差的问题，提出了一种激活引导的提示注入攻击框架.首先，我们构建了一个基于能量的模型（EBM）使用代理模型的激活来评估对抗性提示的质量。在经过训练的EBM的指导下，我们采用令牌级马尔可夫链蒙特卡罗（MCMC）采样来自适应地优化对抗性提示，从而实现无梯度的黑盒攻击。实验结果证明了我们卓越的跨模型可移植性，在五种主流LLM中实现了49.6%的攻击成功率（ASB），比人工制作的提示提高了34.6%，并在未见的任务场景中保持了36.6%的ASB。可解释性分析揭示了激活和攻击有效性之间的相关性，凸显了语义模式在可转移漏洞利用中的关键作用。



## **5. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

语音认证和反欺骗系统威胁调查 cs.CR

This paper is submitted to the Computer Science Review

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2508.16843v3) [paper-pdf](http://arxiv.org/pdf/2508.16843v3)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.

摘要: 语音认证发生了重大变化，从依赖手工声学特征的传统系统到可以提取稳健的说话者嵌入的深度学习模型。这一进步扩大了其在金融、智能设备、执法等领域的应用。然而，随着采用率的增加，威胁也随之增加。本调查全面回顾了针对语音认证系统（PAS）和反欺骗对策（CM）的现代威胁格局，包括数据中毒、对抗性、深度伪造和对抗性欺骗攻击。我们按时间顺序追踪语音认证的发展，并研究漏洞如何随着技术进步而演变。对于每种类型的攻击，我们总结了方法论，强调常用的数据集，比较性能和局限性，并使用广泛接受的分类法组织现有文献。通过强调新出现的风险和公开挑战，本调查旨在支持开发更安全、更有弹性的语音认证系统。



## **6. Generating Transferrable Adversarial Examples via Local Mixing and Logits Optimization for Remote Sensing Object Recognition**

通过局部混合和Logits优化生成可传递的对抗示例用于遥感目标识别 cs.CV

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07495v1) [paper-pdf](http://arxiv.org/pdf/2509.07495v1)

**Authors**: Chun Liu, Hailong Wang, Bingqian Zhu, Panpan Ding, Zheng Zheng, Tao Xu, Zhigang Han, Jiayao Wang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, posing significant security threats to their deployment in remote sensing applications. Research on adversarial attacks not only reveals model vulnerabilities but also provides critical insights for enhancing robustness. Although current mixing-based strategies have been proposed to increase the transferability of adversarial examples, they either perform global blending or directly exchange a region in the images, which may destroy global semantic features and mislead the optimization of adversarial examples. Furthermore, their reliance on cross-entropy loss for perturbation optimization leads to gradient diminishing during iterative updates, compromising adversarial example quality. To address these limitations, we focus on non-targeted attacks and propose a novel framework via local mixing and logits optimization. First, we present a local mixing strategy to generate diverse yet semantically consistent inputs. Different from MixUp, which globally blends two images, and MixCut, which stitches images together, our method merely blends local regions to preserve global semantic information. Second, we adapt the logit loss from targeted attacks to non-targeted scenarios, mitigating the gradient vanishing problem of cross-entropy loss. Third, a perturbation smoothing loss is applied to suppress high-frequency noise and enhance transferability. Extensive experiments on FGSCR-42 and MTARSI datasets demonstrate superior performance over 12 state-of-the-art methods across 6 surrogate models. Notably, with ResNet as the surrogate on MTARSI, our method achieves a 17.28% average improvement in black-box attack success rate.

摘要: 深度神经网络（DNN）容易受到对抗攻击，对其在遥感应用中的部署构成重大安全威胁。对对抗攻击的研究不仅揭示了模型的漏洞，而且还为增强稳健性提供了重要见解。尽管目前提出了基于混合的策略来增加对抗性示例的可移植性，但它们要么进行全局混合，要么直接交换图像中的区域，这可能会破坏全局语义特征并误导对抗性示例的优化。此外，它们对交叉熵损失的依赖导致迭代更新期间的梯度减小，从而损害对抗性示例质量。为了解决这些限制，我们专注于非目标攻击，并通过本地混合和逻辑优化提出了一种新颖的框架。首先，我们提出了一种本地混合策略来生成多样化但语义一致的输入。与全局混合两个图像的MixUp和将图像缝合在一起的MixCut不同，我们的方法只是混合局部区域以保留全局语义信息。其次，我们将有针对性攻击的logit损失调整到非有针对性的场景，减轻了交叉熵损失的梯度消失问题。第三，应用扰动平滑损失来抑制高频噪音并增强可移植性。对FGCR-42和MTARSI数据集的广泛实验表明，在6个替代模型中，其性能优于12种最先进方法。值得注意的是，通过ResNet作为MTARSI的替代品，我们的方法使黑匣子攻击成功率平均提高了17.28%。



## **7. Texture- and Shape-based Adversarial Attacks for Overhead Image Vehicle Detection**

基于纹理和形状的对抗性攻击用于头顶图像车辆检测 cs.CV

This version corresponds to the paper accepted for presentation at  ICIP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2412.16358v2) [paper-pdf](http://arxiv.org/pdf/2412.16358v2)

**Authors**: Mikael Yeghiazaryan, Sai Abhishek Siddhartha Namburu, Emily Kim, Stanislav Panev, Celso de Melo, Fernando De la Torre, Jessica K. Hodgins

**Abstract**: Detecting vehicles in aerial images is difficult due to complex backgrounds, small object sizes, shadows, and occlusions. Although recent deep learning advancements have improved object detection, these models remain susceptible to adversarial attacks (AAs), challenging their reliability. Traditional AA strategies often ignore practical implementation constraints. Our work proposes realistic and practical constraints on texture (lowering resolution, limiting modified areas, and color ranges) and analyzes the impact of shape modifications on attack performance. We conducted extensive experiments with three object detector architectures, demonstrating the performance-practicality trade-off: more practical modifications tend to be less effective, and vice versa. We release both code and data to support reproducibility at https://github.com/humansensinglab/texture-shape-adversarial-attacks.

摘要: 由于背景复杂、物体尺寸小、阴影和遮挡，在航空图像中检测车辆很困难。尽管最近的深度学习进步改进了对象检测，但这些模型仍然容易受到对抗攻击（AA）的影响，从而挑战其可靠性。传统的AA策略通常忽视实际实施限制。我们的工作提出了对纹理的现实和实用的限制（降低分辨率、限制修改区域和颜色范围），并分析了形状修改对攻击性能的影响。我们对三种物体检测器架构进行了广泛的实验，展示了性能与实用性的权衡：更实用的修改往往效果不佳，反之亦然。我们在https://github.com/humansensinglab/texture-shape-adversarial-attacks上发布代码和数据以支持可重复性。



## **8. Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm**

Prepared for the Worst：A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm cs.RO

9 pages (6 content, 1 reference, 2 appendix). 7 figures, accepted to  2025 IEEE International Conference on Robotics and Automation (ICRA)

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2403.05666v3) [paper-pdf](http://arxiv.org/pdf/2403.05666v3)

**Authors**: Ziyu Zhang, Johann Laconte, Daniil Lisus, Timothy D. Barfoot

**Abstract**: This paper presents a novel method for assessing the resilience of the ICP algorithm via learning-based, worst-case attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms before deployments is crucial. The ICP algorithm is the standard for lidar-based localization, but its accuracy can be greatly affected by corrupted measurements from various sources, including occlusions, adverse weather, or mechanical sensor issues. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP, our method focuses on finding the maximum possible ICP error that can arise from corrupted measurements at a location. We demonstrate that our perturbation-based adversarial attacks can be used pre-deployment to identify locations on a map where ICP is particularly vulnerable to corruptions in the measurements. With such information, autonomous robots can take safer paths when deployed, to mitigate against their measurements being corrupted. The proposed attack outperforms baselines more than 88% of the time across a wide range of scenarios.

摘要: 本文提出了一种新的方法，通过对激光雷达点云进行基于学习的最坏情况攻击来评估ICP算法的弹性。对于自主导航等安全关键应用，在部署前确保算法的弹性至关重要。ICP算法是基于激光雷达的定位的标准，但其准确性可能会受到来自各种来源的损坏测量结果的极大影响，包括遮挡、恶劣天气或机械传感器问题。不幸的是，国际比较方案的复杂性和反复性使得评估其对腐败的复原力具有挑战性。虽然人们一直在努力创建具有挑战性的数据集并开发模拟来评估ICP的弹性，但我们的方法重点是寻找可能因某个地点的测量结果损坏而产生的最大可能的ICP误差。我们证明，我们的基于扰动的对抗攻击可以在部署前使用来识别地图上ISP特别容易受到测量结果损坏的位置。有了这些信息，自主机器人在部署时可以采取更安全的路径，以防止其测量结果被破坏。在各种场景中，拟议的攻击在88%以上的时间内优于基线。



## **9. When Fine-Tuning is Not Enough: Lessons from HSAD on Hybrid and Adversarial Audio Spoof Detection**

当微调还不够时：HSAD关于混合和对抗性音频欺骗检测的教训 cs.SD

13 pages, 11 figures.This work has been submitted to the IEEE for  possible publication

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07323v1) [paper-pdf](http://arxiv.org/pdf/2509.07323v1)

**Authors**: Bin Hu, Kunyang Huang, Daehan Kwak, Meng Xu, Kuan Huang

**Abstract**: The rapid advancement of AI has enabled highly realistic speech synthesis and voice cloning, posing serious risks to voice authentication, smart assistants, and telecom security. While most prior work frames spoof detection as a binary task, real-world attacks often involve hybrid utterances that mix genuine and synthetic speech, making detection substantially more challenging. To address this gap, we introduce the Hybrid Spoofed Audio Dataset (HSAD), a benchmark containing 1,248 clean and 41,044 degraded utterances across four classes: human, cloned, zero-shot AI-generated, and hybrid audio. Each sample is annotated with spoofing method, speaker identity, and degradation metadata to enable fine-grained analysis. We evaluate six transformer-based models, including spectrogram encoders (MIT-AST, MattyB95-AST) and self-supervised waveform models (Wav2Vec2, HuBERT). Results reveal critical lessons: pretrained models overgeneralize and collapse under hybrid conditions; spoof-specific fine-tuning improves separability but struggles with unseen compositions; and dataset-specific adaptation on HSAD yields large performance gains (AST greater than 97 percent and F1 score is approximately 99 percent), though residual errors persist for complex hybrids. These findings demonstrate that fine-tuning alone is not sufficient-robust hybrid-aware benchmarks like HSAD are essential to expose calibration failures, model biases, and factors affecting spoof detection in adversarial environments. HSAD thus provides both a dataset and an analytic framework for building resilient and trustworthy voice authentication systems.

摘要: 人工智能的快速发展使高度真实的语音合成和语音克隆成为可能，这对语音认证、智能助理和电信安全构成了严重风险。虽然大多数以前的工作框架将欺骗检测作为一项二元任务，但现实世界的攻击通常涉及混合真实语音和合成语音的混合话语，这使得检测更具挑战性。为了解决这一差距，我们引入了混合欺骗音频数据集（HSAD），这是一个基准测试，包含四个类别的1，248个干净话语和41，044个降级话语：人类、克隆、零镜头人工智能生成和混合音频。每个样本都用欺骗方法、说话者身份和降级元数据进行注释，以实现细粒度分析。我们评估了六种基于变压器的模型，包括频谱编码器（MIT-AST、MattyB 95-AST）和自我监督的波形模型（Wave 2 Vec 2、HuBERT）。结果揭示了重要的教训：预训练的模型在混合条件下过度概括和崩溃;针对欺骗的微调提高了可分离性，但在处理看不见的成分时会遇到困难; HSAD上的针对厕所的特定调整产生了很大的性能提升（AST大于97%，F1评分约为99%），尽管复杂的混合体仍然存在残余误差。这些发现表明，仅靠微调并不有效--HSAD等稳健的混合感知基准对于暴露对抗环境中的校准失败、模型偏差和影响欺骗检测的因素至关重要。因此，HSAD提供了数据集和分析框架，用于构建弹性且值得信赖的语音认证系统。



## **10. GRADA: Graph-based Reranking against Adversarial Documents Attack**

GRADA：基于图的重新排名对抗文档攻击 cs.IR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.07546v2) [paper-pdf](http://arxiv.org/pdf/2505.07546v2)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.

摘要: 检索增强生成（RAG）框架通过集成来自检索文档的外部知识来提高大型语言模型（LLM）的准确性，从而克服模型静态内在知识的局限性。然而，这些系统很容易受到对抗性攻击，这些攻击通过引入对抗性但在语义上与查询相似的文档来操纵检索过程。值得注意的是，虽然这些对抗性文档类似于查询，但它们与检索集中的良性文档表现出弱的相似性。因此，我们提出了一个简单而有效的基于图形的对抗性文档攻击重新排名（GRADA）框架，旨在保留检索质量，同时显着降低对手的成功。我们的研究通过在五个LLM上进行的实验来评估我们的方法的有效性：GPT-3.5-Turbo，GPT-4 o，Llama3.1-8b，Llama3.1- 70 b和Qwen2.5- 7 b。我们使用三个数据集来评估性能，来自Natural Questions数据集的结果表明攻击成功率降低了80%，同时保持了最小的准确性损失。



## **11. Personalized Attacks of Social Engineering in Multi-turn Conversations: LLM Agents for Simulation and Detection**

多轮会话中的社会工程个性化攻击：LLM Agent仿真与检测 cs.CR

Accepted as a paper at COLM 2025 Workshop on AI Agents: Capabilities  and Safety

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.15552v2) [paper-pdf](http://arxiv.org/pdf/2503.15552v2)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the SE attack mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.

摘要: 会话代理的快速发展，特别是由大型语言模型（LLM）驱动的聊天机器人，对社交媒体平台构成了社会工程（SE）攻击的重大风险。由于这些会话的动态性质，多回合、基于聊天的交互中的SE检测比单实例检测复杂得多。减轻这种威胁的一个关键因素是了解SE攻击的机制，特别是攻击者如何利用漏洞以及受害者的个性特征如何影响他们的易感性。在这项工作中，我们提出了一个LLM-agentic框架，SE-VSim，模拟SE攻击机制，通过生成多轮对话。我们对具有不同性格特征的受害者特工进行建模，以评估心理特征如何影响操纵的易感性。我们使用包含1000多个模拟对话的数据集，检查了冒充招聘人员、资助机构和记者的对手试图提取敏感信息的攻击场景。基于此分析，我们提出了一个概念验证SE-OmniGuard，通过利用受害者个性的先验知识、评估攻击策略以及监控对话中的信息交换以识别潜在的SE尝试，为用户提供个性化保护。



## **12. Adversarial Attacks on Audio Deepfake Detection: A Benchmark and Comparative Study**

音频Deepfake检测的对抗攻击：基准和比较研究 cs.SD

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.07132v1) [paper-pdf](http://arxiv.org/pdf/2509.07132v1)

**Authors**: Kutub Uddin, Muhammad Umar Farooq, Awais Khan, Khalid Mahmood Malik

**Abstract**: The widespread use of generative AI has shown remarkable success in producing highly realistic deepfakes, posing a serious threat to various voice biometric applications, including speaker verification, voice biometrics, audio conferencing, and criminal investigations. To counteract this, several state-of-the-art (SoTA) audio deepfake detection (ADD) methods have been proposed to identify generative AI signatures to distinguish between real and deepfake audio. However, the effectiveness of these methods is severely undermined by anti-forensic (AF) attacks that conceal generative signatures. These AF attacks span a wide range of techniques, including statistical modifications (e.g., pitch shifting, filtering, noise addition, and quantization) and optimization-based attacks (e.g., FGSM, PGD, C \& W, and DeepFool). In this paper, we investigate the SoTA ADD methods and provide a comparative analysis to highlight their effectiveness in exposing deepfake signatures, as well as their vulnerabilities under adversarial conditions. We conducted an extensive evaluation of ADD methods on five deepfake benchmark datasets using two categories: raw and spectrogram-based approaches. This comparative analysis enables a deeper understanding of the strengths and limitations of SoTA ADD methods against diverse AF attacks. It does not only highlight vulnerabilities of ADD methods, but also informs the design of more robust and generalized detectors for real-world voice biometrics. It will further guide future research in developing adaptive defense strategies that can effectively counter evolving AF techniques.

摘要: 生成式人工智能的广泛使用在制作高度逼真的deepfake方面取得了显着的成功，对各种语音生物识别应用构成了严重威胁，包括说话人验证，语音生物识别，音频会议和刑事调查。为了解决这个问题，已经提出了几种最先进的（SoTA）音频深度伪造检测（ADD）方法来识别生成AI签名，以区分真实和深度伪造音频。然而，这些方法的有效性被隐藏生成签名的反取证（AF）攻击严重破坏。这些AF攻击涵盖了广泛的技术，包括统计修改（例如，音调移动、过滤、噪音添加和量化）和基于优化的攻击（例如，FGSM、PVD、C \& W和DeepFool）。在本文中，我们研究了SoTA ADD方法，并提供了比较分析，以强调它们在暴露Deepfake签名方面的有效性以及它们在对抗条件下的漏洞。我们使用两类：原始方法和基于谱图的方法，对五个Deepfake基准数据集进行了广泛的评估。这种比较分析使我们能够更深入地了解SoTA ADD方法针对各种AF发作的优势和局限性。它不仅强调了ADD方法的漏洞，而且还为现实世界语音生物识别技术的更稳健和更通用的检测器的设计提供了信息。它将进一步指导未来开发可有效对抗不断发展的AF技术的自适应防御策略的研究。



## **13. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

攻击LLM和AI代理：针对大型语言模型的广告嵌入攻击 cs.CR

6 pages, 2 figures

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2508.17674v2) [paper-pdf](http://arxiv.org/pdf/2508.17674v2)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.

摘要: 我们引入了广告嵌入攻击（AEA），这是一种新型LLM安全威胁，可以悄悄地将促销或恶意内容注入模型输出和AI代理中。AEA通过两种低成本载体运作：（1）劫持第三方服务分发平台以预先设置对抗提示，以及（2）发布经过攻击者数据微调的后门开源检查点。与降低准确性的传统攻击不同，AEA破坏了信息完整性，导致模型在看起来正常的情况下返回秘密广告、宣传或仇恨言论。我们详细介绍了攻击管道，绘制了五个利益相关者受害者群体，并提出了一种初步的基于预算的自我检查防御，该防御可以减轻这些注入，而无需额外的模型再培训。我们的调查结果揭示了LLM安全方面存在一个紧迫且未充分解决的差距，并呼吁人工智能安全界协调检测、审计和政策响应。



## **14. From Noise to Narrative: Tracing the Origins of Hallucinations in Transformers**

从噪音到叙事：追踪《变形金刚》中幻觉的起源 cs.LG

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06938v1) [paper-pdf](http://arxiv.org/pdf/2509.06938v1)

**Authors**: Praneet Suresh, Jack Stanley, Sonia Joseph, Luca Scimeca, Danilo Bzdok

**Abstract**: As generative AI systems become competent and democratized in science, business, and government, deeper insight into their failure modes now poses an acute need. The occasional volatility in their behavior, such as the propensity of transformer models to hallucinate, impedes trust and adoption of emerging AI solutions in high-stakes areas. In the present work, we establish how and when hallucinations arise in pre-trained transformer models through concept representations captured by sparse autoencoders, under scenarios with experimentally controlled uncertainty in the input space. Our systematic experiments reveal that the number of semantic concepts used by the transformer model grows as the input information becomes increasingly unstructured. In the face of growing uncertainty in the input space, the transformer model becomes prone to activate coherent yet input-insensitive semantic features, leading to hallucinated output. At its extreme, for pure-noise inputs, we identify a wide variety of robustly triggered and meaningful concepts in the intermediate activations of pre-trained transformer models, whose functional integrity we confirm through targeted steering. We also show that hallucinations in the output of a transformer model can be reliably predicted from the concept patterns embedded in transformer layer activations. This collection of insights on transformer internal processing mechanics has immediate consequences for aligning AI models with human values, AI safety, opening the attack surface for potential adversarial attacks, and providing a basis for automatic quantification of a model's hallucination risk.

摘要: 随着生成性人工智能系统在科学、商业和政府中变得称职和民主化，现在迫切需要更深入地了解其失败模式。他们的行为偶尔会出现波动，例如Transformer模型产生幻觉的倾向，阻碍了信任和在高风险领域对新兴人工智能解决方案的采用。在目前的工作中，我们通过稀疏自编码器捕获的概念表示，在输入空间中具有实验控制的不确定性的情况下，建立了如何以及何时在预训练的Transformer模型中出现幻觉。我们的系统实验表明，随着输入信息变得越来越非结构化，Transformer模型使用的语义概念数量也会增加。面对输入空间日益增长的不确定性，Transformer模型变得容易激活连贯但对输入不敏感的语义特征，从而导致幻觉输出。在极端情况下，对于纯噪音输入，我们在预训练的Transformer模型的中间激活中识别了各种鲁棒触发且有意义的概念，我们通过有针对性的引导来确认其功能完整性。我们还表明，可以根据嵌入在Transformer层激活中的概念模式可靠地预测Transformer模型输出中的幻觉。这一系列关于Transformer内部处理机制的见解对调整AI模型与人类价值观、AI安全、打开潜在对抗性攻击的攻击面以及为模型幻觉风险的自动量化提供基础具有直接影响。



## **15. Evaluating the Impact of Adversarial Attacks on Traffic Sign Classification using the LISA Dataset**

使用LISA数据集评估对抗性攻击对交通标志分类的影响 cs.CV

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06835v1) [paper-pdf](http://arxiv.org/pdf/2509.06835v1)

**Authors**: Nabeyou Tadessa, Balaji Iyangar, Mashrur Chowdhury

**Abstract**: Adversarial attacks pose significant threats to machine learning models by introducing carefully crafted perturbations that cause misclassification. While prior work has primarily focused on MNIST and similar datasets, this paper investigates the vulnerability of traffic sign classifiers using the LISA Traffic Sign dataset. We train a convolutional neural network to classify 47 different traffic signs and evaluate its robustness against Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks. Our results show a sharp decline in classification accuracy as the perturbation magnitude increases, highlighting the models susceptibility to adversarial examples. This study lays the groundwork for future exploration into defense mechanisms tailored for real-world traffic sign recognition systems.

摘要: 对抗性攻击通过引入精心设计的导致错误分类的扰动，对机器学习模型构成重大威胁。虽然之前的工作主要集中在MNIST和类似数据集，但本文使用LISA Traffic Sign数据集研究了交通标志分类器的漏洞。我们训练一个卷积神经网络来对47个不同的交通标志进行分类，并评估其对快速梯度标志法（FGSM）和投影梯度下降（PVD）攻击的稳健性。我们的结果显示，随着扰动幅度的增加，分类准确性急剧下降，凸显了模型对对抗性示例的敏感性。这项研究为未来探索为现实世界的交通标志识别系统量身定制的防御机制奠定了基础。



## **16. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

将逻辑与自身对立：通过对比问题探索模型辩护 cs.CL

Accepted at EMNLP 2025 (Main)

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2501.01872v4) [paper-pdf](http://arxiv.org/pdf/2501.01872v4)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.

摘要: 尽管大型语言模型与人类价值观和道德原则广泛一致，但仍然容易受到利用其推理能力的复杂越狱攻击。现有的安全措施通常检测到明显的恶意意图，但无法解决微妙的、推理驱动的漏洞。在这项工作中，我们引入了POATE（极反相查询生成、对抗模板构建和搜索），这是一种新颖的越狱技术，利用对比推理来引发不道德的反应。POATE精心设计了语义上相反的意图，并将它们与对抗模板集成，以非凡的微妙性引导模型走向有害的输出。我们对参数大小不同的六个不同语言模型家族进行了广泛的评估，以证明攻击的稳健性，与现有方法相比，实现了显着更高的攻击成功率（~44%）。为了解决这个问题，我们提出了意图感知CoT和反向思维CoT，它们分解查询以检测恶意意图，并反向推理以评估和拒绝有害响应。这些方法增强了推理的稳健性并加强了模型对对抗性利用的防御。



## **17. On Hyperparameters and Backdoor-Resistance in Horizontal Federated Learning**

水平联邦学习中的超参数和后门抵抗 cs.CR

To appear in the Proceedings of the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.05192v2) [paper-pdf](http://arxiv.org/pdf/2509.05192v2)

**Authors**: Simon Lachnit, Ghassan Karame

**Abstract**: Horizontal Federated Learning (HFL) is particularly vulnerable to backdoor attacks as adversaries can easily manipulate both the training data and processes to execute sophisticated attacks. In this work, we study the impact of training hyperparameters on the effectiveness of backdoor attacks and defenses in HFL. More specifically, we show both analytically and by means of measurements that the choice of hyperparameters by benign clients does not only influence model accuracy but also significantly impacts backdoor attack success. This stands in sharp contrast with the multitude of contributions in the area of HFL security, which often rely on custom ad-hoc hyperparameter choices for benign clients$\unicode{x2013}$leading to more pronounced backdoor attack strength and diminished impact of defenses. Our results indicate that properly tuning benign clients' hyperparameters$\unicode{x2013}$such as learning rate, batch size, and number of local epochs$\unicode{x2013}$can significantly curb the effectiveness of backdoor attacks, regardless of the malicious clients' settings. We support this claim with an extensive robustness evaluation of state-of-the-art attack-defense combinations, showing that carefully chosen hyperparameters yield across-the-board improvements in robustness without sacrificing main task accuracy. For example, we show that the 50%-lifespan of the strong A3FL attack can be reduced by 98.6%, respectively$\unicode{x2013}$all without using any defense and while incurring only a 2.9 percentage points drop in clean task accuracy.

摘要: 水平联邦学习（HFL）特别容易受到后门攻击，因为对手可以轻松操纵训练数据和流程来执行复杂的攻击。在这项工作中，我们研究了训练超参数对HFL中后门攻击和防御有效性的影响。更具体地说，我们通过分析和测量表明，良性客户端对超参数的选择不仅会影响模型准确性，还会显着影响后门攻击的成功。这与HFL安全领域的众多贡献形成鲜明对比，HFL安全领域通常依赖于良性客户端$\unicode{x2013}$的自定义临时超参数选择，导致后门攻击强度更明显，防御影响减弱。我们的结果表明，正确调整良性客户端的超参数$\unicode{x2013}$，例如学习率、批量大小和本地纪元数量$\unicode{x2013}$可以显着抑制后门攻击的有效性，无论恶意客户端的设置如何。我们通过对最先进的攻击-防御组合的广泛鲁棒性评估来支持这一说法，表明精心选择的超参数可以在不牺牲主要任务准确性的情况下全面提高鲁棒性。例如，我们表明，在不使用任何防御的情况下，强A3 FL攻击的50%寿命可以分别缩短98.6%$\unicode{x2013}$all，同时仅导致干净任务准确性下降2.9个百分点。



## **18. Mind Your Server: A Systematic Study of Parasitic Toolchain Attacks on the MCP Ecosystem**

注意您的服务器：对CP生态系统的寄生工具链攻击的系统研究 cs.CR

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06572v1) [paper-pdf](http://arxiv.org/pdf/2509.06572v1)

**Authors**: Shuli Zhao, Qinsheng Hou, Zihan Zhan, Yanhao Wang, Yuchong Xie, Yu Guo, Libo Chen, Shenghong Li, Zhi Xue

**Abstract**: Large language models (LLMs) are increasingly integrated with external systems through the Model Context Protocol (MCP), which standardizes tool invocation and has rapidly become a backbone for LLM-powered applications. While this paradigm enhances functionality, it also introduces a fundamental security shift: LLMs transition from passive information processors to autonomous orchestrators of task-oriented toolchains, expanding the attack surface, elevating adversarial goals from manipulating single outputs to hijacking entire execution flows. In this paper, we reveal a new class of attacks, Parasitic Toolchain Attacks, instantiated as MCP Unintended Privacy Disclosure (MCP-UPD). These attacks require no direct victim interaction; instead, adversaries embed malicious instructions into external data sources that LLMs access during legitimate tasks. The malicious logic infiltrates the toolchain and unfolds in three phases: Parasitic Ingestion, Privacy Collection, and Privacy Disclosure, culminating in stealthy exfiltration of private data. Our root cause analysis reveals that MCP lacks both context-tool isolation and least-privilege enforcement, enabling adversarial instructions to propagate unchecked into sensitive tool invocations. To assess the severity, we design MCP-SEC and conduct the first large-scale security census of the MCP ecosystem, analyzing 12,230 tools across 1,360 servers. Our findings show that the MCP ecosystem is rife with exploitable gadgets and diverse attack methods, underscoring systemic risks in MCP platforms and the urgent need for defense mechanisms in LLM-integrated environments.

摘要: 大型语言模型（LLM）通过模型上下文协议（HCP）越来越多地与外部系统集成，该协议使工具调用同步化，并已迅速成为LLM支持的应用程序的支柱。虽然这种范式增强了功能，但它也引入了根本性的安全转变：LLM从被动信息处理器过渡到面向任务的工具链的自主编排，扩大了攻击面，将对抗目标从操纵单个输出提升到劫持整个执行流。在本文中，我们揭示了一类新的攻击，即寄生工具链攻击，实例化为LCP无意隐私泄露（MCP-UPD）。这些攻击不需要受害者直接互动;相反，对手会将恶意指令嵌入到LLM在合法任务期间访问的外部数据源中。恶意逻辑渗透到工具链中，并分三个阶段展开：寄生摄入、隐私收集和隐私披露，最终导致私人数据的秘密泄露。我们的根本原因分析表明，LCP缺乏上下文工具隔离和最低特权强制执行，使得对抗指令能够不受限制地传播到敏感工具调用中。为了评估严重性，我们设计了MCP-SEC，并对LCP生态系统进行了首次大规模安全普查，分析了1，360台服务器上的12，230个工具。我们的研究结果表明，LCP生态系统中充斥着可利用的小工具和多样化的攻击方法，凸显了LCP平台的系统性风险以及LLM集成环境中对防御机制的迫切需求。



## **19. Robustness and accuracy of mean opinion scores with hard and soft outlier detection**

具有硬异常值和软异常值检测的平均意见分数的稳健性和准确性 eess.IV

Accepted for 17th International Conference on Quality of Multimedia  Experience (QoMEX'25), September 2025, Madrid, Spain

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06554v1) [paper-pdf](http://arxiv.org/pdf/2509.06554v1)

**Authors**: Dietmar Saupe, Tim Bleile

**Abstract**: In subjective assessment of image and video quality, observers rate or compare selected stimuli. Before calculating the mean opinion scores (MOS) for these stimuli from the ratings, it is recommended to identify and deal with outliers that may have given unreliable ratings. Several methods are available for this purpose, some of which have been standardized. These methods are typically based on statistics and sometimes tested by introducing synthetic ratings from artificial outliers, such as random clickers. However, a reliable and comprehensive approach is lacking for comparative performance analysis of outlier detection methods. To fill this gap, this work proposes and applies an empirical worst-case analysis as a general solution. Our method involves evolutionary optimization of an adversarial black-box attack on outlier detection algorithms, where the adversary maximizes the distortion of scale values with respect to ground truth. We apply our analysis to several hard and soft outlier detection methods for absolute category ratings and show their differing performance in this stress test. In addition, we propose two new outlier detection methods with low complexity and excellent worst-case performance. Software for adversarial attacks and data analysis is available.

摘要: 在图像和视频质量的主观评估中，观察者对选定的刺激进行评级或比较。在根据评级计算这些刺激的平均意见分数（MOS）之前，建议识别并处理可能给出不可靠评级的离群值。为此目的，有多种方法可用，其中一些已被标准化。这些方法通常基于统计数据，有时通过引入来自人工异常值（例如随机点击器）的合成评级来进行测试。然而，缺乏可靠且全面的方法来对异常值检测方法进行比较性能分析。为了填补这一空白，这项工作提出并应用了实证最坏情况分析作为一般解决方案。我们的方法涉及对异常值检测算法的对抗性黑匣子攻击的进化优化，其中对手最大化尺度值相对于地面真相的失真。我们将我们的分析应用于绝对类别评级的几种硬异常值和软异常值检测方法，并展示它们在此压力测试中的不同表现。此外，我们还提出了两种新的异常值检测方法，具有低复杂度和优异的最坏情况性能。提供对抗攻击和数据分析软件。



## **20. Byzantine-Robust Federated Learning Using Generative Adversarial Networks**

使用生成对抗网络的拜占庭鲁棒联邦学习 cs.CR

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.20884v2) [paper-pdf](http://arxiv.org/pdf/2503.20884v2)

**Authors**: Usama Zafar, André Teixeira, Salman Toor

**Abstract**: Federated learning (FL) enables collaborative model training across distributed clients without sharing raw data, but its robustness is threatened by Byzantine behaviors such as data and model poisoning. Existing defenses face fundamental limitations: robust aggregation rules incur error lower bounds that grow with client heterogeneity, while detection-based methods often rely on heuristics (e.g., a fixed number of malicious clients) or require trusted external datasets for validation. We present a defense framework that addresses these challenges by leveraging a conditional generative adversarial network (cGAN) at the server to synthesize representative data for validating client updates. This approach eliminates reliance on external datasets, adapts to diverse attack strategies, and integrates seamlessly into standard FL workflows. Extensive experiments on benchmark datasets demonstrate that our framework accurately distinguishes malicious from benign clients while maintaining overall model accuracy. Beyond Byzantine robustness, we also examine the representativeness of synthesized data, computational costs of cGAN training, and the transparency and scalability of our approach.

摘要: 联合学习（FL）可以在不共享原始数据的情况下跨分布式客户端进行协作模型训练，但其稳健性受到数据和模型中毒等拜占庭行为的威胁。现有的防御面临根本性限制：稳健的聚合规则会导致随着客户端的多样性而增长的错误下限，而基于检测的方法通常依赖于启发式方法（例如，固定数量的恶意客户端）或需要受信任的外部数据集进行验证。我们提出了一个防御框架，通过利用服务器上的条件生成对抗网络（cGAN）来合成代表性数据以验证客户端更新来解决这些挑战。这种方法消除了对外部数据集的依赖，适应不同的攻击策略，并无缝集成到标准FL工作流程中。对基准数据集的广泛实验表明，我们的框架可以准确地区分恶意客户端和良性客户端，同时保持整体模型准确性。除了拜占庭鲁棒性之外，我们还研究了合成数据的代表性，cGAN训练的计算成本以及我们方法的透明度和可扩展性。



## **21. IGAff: Benchmarking Adversarial Iterative and Genetic Affine Algorithms on Deep Neural Networks**

IGAff：深度神经网络上的对抗迭代和遗传仿射算法基准 cs.CV

10 pages, 7 figures, Accepted at ECAI 2025 (28th European Conference  on Artificial Intelligence)

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06459v1) [paper-pdf](http://arxiv.org/pdf/2509.06459v1)

**Authors**: Sebastian-Vasile Echim, Andrei-Alexandru Preda, Dumitru-Clementin Cercel, Florin Pop

**Abstract**: Deep neural networks currently dominate many fields of the artificial intelligence landscape, achieving state-of-the-art results on numerous tasks while remaining hard to understand and exhibiting surprising weaknesses. An active area of research focuses on adversarial attacks, which aim to generate inputs that uncover these weaknesses. However, this proves challenging, especially in the black-box scenario where model details are inaccessible. This paper explores in detail the impact of such adversarial algorithms on ResNet-18, DenseNet-121, Swin Transformer V2, and Vision Transformer network architectures. Leveraging the Tiny ImageNet, Caltech-256, and Food-101 datasets, we benchmark two novel black-box iterative adversarial algorithms based on affine transformations and genetic algorithms: 1) Affine Transformation Attack (ATA), an iterative algorithm maximizing our attack score function using random affine transformations, and 2) Affine Genetic Attack (AGA), a genetic algorithm that involves random noise and affine transformations. We evaluate the performance of the models in the algorithm parameter variation, data augmentation, and global and targeted attack configurations. We also compare our algorithms with two black-box adversarial algorithms, Pixle and Square Attack. Our experiments yield better results on the image classification task than similar methods in the literature, achieving an accuracy improvement of up to 8.82%. We provide noteworthy insights into successful adversarial defenses and attacks at both global and targeted levels, and demonstrate adversarial robustness through algorithm parameter variation.

摘要: 深度神经网络目前在人工智能领域的许多领域占据主导地位，在众多任务上实现了最先进的结果，同时仍然难以理解并表现出令人惊讶的弱点。一个活跃的研究领域专注于对抗攻击，旨在生成揭露这些弱点的输入。然而，事实证明这具有挑战性，尤其是在无法访问模型详细信息的黑匣子场景中。本文详细探讨了这种对抗算法对ResNet-18、DenseNet-121、Swin Transformer V2和Vision Transformer网络架构的影响。利用Tiny ImageNet、Caltech-256和Food-101数据集，我们对两种基于仿射变换和遗传算法的新型黑匣子迭代对抗算法进行基准测试：1）仿射变换攻击（ATA），一种使用随机仿射变换最大化我们的攻击分数函数的迭代算法，2）仿射遗传攻击（AGA），一种涉及随机噪音和仿射变换的遗传算法。我们评估模型在算法参数变化、数据增强以及全局和有针对性的攻击配置方面的性能。我们还将我们的算法与两种黑匣子对抗算法Pixle和Square Attack进行了比较。我们的实验在图像分类任务上产生了比文献中的类似方法更好的结果，实现了高达8.82%的准确性提高。我们对全球和目标层面上的成功对抗防御和攻击提供了值得注意的见解，并通过算法参数变化展示了对抗鲁棒性。



## **22. Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?**

Mask-GCG：敌对后缀中的所有代币都是越狱攻击所必需的吗？ cs.CL

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06350v1) [paper-pdf](http://arxiv.org/pdf/2509.06350v1)

**Authors**: Junjie Mu, Zonghao Ying, Zhekui Fan, Zonglei Jing, Yaoyuan Zhang, Zhengmin Yu, Wenxin Zhang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Jailbreak attacks on Large Language Models (LLMs) have demonstrated various successful methods whereby attackers manipulate models into generating harmful responses that they are designed to avoid. Among these, Greedy Coordinate Gradient (GCG) has emerged as a general and effective approach that optimizes the tokens in a suffix to generate jailbreakable prompts. While several improved variants of GCG have been proposed, they all rely on fixed-length suffixes. However, the potential redundancy within these suffixes remains unexplored. In this work, we propose Mask-GCG, a plug-and-play method that employs learnable token masking to identify impactful tokens within the suffix. Our approach increases the update probability for tokens at high-impact positions while pruning those at low-impact positions. This pruning not only reduces redundancy but also decreases the size of the gradient space, thereby lowering computational overhead and shortening the time required to achieve successful attacks compared to GCG. We evaluate Mask-GCG by applying it to the original GCG and several improved variants. Experimental results show that most tokens in the suffix contribute significantly to attack success, and pruning a minority of low-impact tokens does not affect the loss values or compromise the attack success rate (ASR), thereby revealing token redundancy in LLM prompts. Our findings provide insights for developing efficient and interpretable LLMs from the perspective of jailbreak attacks.

摘要: 对大型语言模型（LLM）的越狱攻击已经展示了各种成功的方法，攻击者通过这些方法操纵模型来生成他们旨在避免的有害响应。其中，贪婪坐标梯度（GCG）已成为一种通用且有效的方法，可以优化后缀中的标记以生成可越狱的提示。虽然已经提出了GCG的几种改进变体，但它们都依赖于固定长度的后缀。然而，这些后缀中的潜在冗余仍有待探索。在这项工作中，我们提出了Mask-GCG，这是一种即插即用方法，它采用可学习的标记掩蔽来识别后缀内有影响力的标记。我们的方法增加了高影响力位置上代币的更新概率，同时修剪低影响力位置上的代币。与GCG相比，这种修剪不仅减少了冗余，还减少了梯度空间的大小，从而降低了计算负担并缩短了实现成功攻击所需的时间。我们通过将Mask-GCG应用于原始GCG和几种改进的变体来评估Mask-GCG。实验结果表明，后缀中的大多数令牌对攻击成功做出了显着贡献，并且修剪少数低影响令牌不会影响损失值或损害攻击成功率（ASB），从而揭示了LLM提示中的令牌冗余。我们的研究结果为从越狱攻击的角度开发高效且可解释的LLM提供了见解。



## **23. Whisper Smarter, not Harder: Adversarial Attack on Partial Suppression**

低语更聪明，而不是更难：对部分抑制的对抗攻击 cs.SD

14 pages, 7 figures

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2508.09994v2) [paper-pdf](http://arxiv.org/pdf/2508.09994v2)

**Authors**: Zheng Jie Wong, Bingquan Shen

**Abstract**: Currently, Automatic Speech Recognition (ASR) models are deployed in an extensive range of applications. However, recent studies have demonstrated the possibility of adversarial attack on these models which could potentially suppress or disrupt model output. We investigate and verify the robustness of these attacks and explore if it is possible to increase their imperceptibility. We additionally find that by relaxing the optimisation objective from complete suppression to partial suppression, we can further decrease the imperceptibility of the attack. We also explore possible defences against these attacks and show a low-pass filter defence could potentially serve as an effective defence.

摘要: 目前，自动语音识别（ASB）模型已部署在广泛的应用中。然而，最近的研究证明了对这些模型进行对抗攻击的可能性，这可能会抑制或破坏模型输出。我们调查和验证这些攻击的稳健性，并探索是否有可能增加它们的不可察觉性。我们还发现，通过将优化目标从完全抑制放宽到部分抑制，我们可以进一步降低攻击的不可察觉性。我们还探索了针对这些攻击的可能防御措施，并表明低通过滤器防御可能成为有效的防御措施。



## **24. Chronus: Understanding and Securing the Cutting-Edge Industry Solutions to DRAM Read Disturbance**

Chronus：了解并保护针对动态存储器读取干扰的尖端行业解决方案 cs.CR

To appear in HPCA'25. arXiv admin note: text overlap with  arXiv:2406.19094. Appendix E added that describe the errata and new results

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2502.12650v2) [paper-pdf](http://arxiv.org/pdf/2502.12650v2)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Nisa Bostancı, İsmail Emir Yüksel, Haocong Luo, Oğuz Ergin, Onur Mutlu

**Abstract**: We 1) present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC) and 2) propose Chronus, a new mechanism that addresses PRAC's two major weaknesses. Our analysis shows that PRAC's system performance overhead on benign applications is non-negligible for modern DRAM chips and prohibitively large for future DRAM chips that are more vulnerable to read disturbance. We identify two weaknesses of PRAC that cause these overheads. First, PRAC increases critical DRAM access latency parameters due to the additional time required to increment activation counters. Second, PRAC performs a constant number of preventive refreshes at a time, making it vulnerable to an adversarial access pattern, known as the wave attack, and consequently requiring it to be configured for significantly smaller activation thresholds. To address PRAC's two weaknesses, we propose a new on-DRAM-die RowHammer mitigation mechanism, Chronus. Chronus 1) updates row activation counters concurrently while serving accesses by separating counters from the data and 2) prevents the wave attack by dynamically controlling the number of preventive refreshes performed. Our performance analysis shows that Chronus's system performance overhead is near-zero for modern DRAM chips and very low for future DRAM chips. Chronus outperforms three variants of PRAC and three other state-of-the-art read disturbance solutions. We discuss Chronus's and PRAC's implications for future systems and foreshadow future research directions. To aid future research, we open-source our Chronus implementation at https://github.com/CMU-SAFARI/Chronus.

摘要: 我们1）首次对最先进的动态随机存储器芯片上读取干扰缓解方法（每行激活计数（PRAC））进行了严格的安全性、性能、能源和成本分析; 2）提出了Chronus，这是一种解决PRAC两个主要弱点的新机制。我们的分析表明，PRAC在良性应用程序上的系统性能负担对于现代动态随机存储器芯片来说是不可忽视的，而对于更容易受到读取干扰的未来动态随机存储器芯片来说，这一负担则大得令人望而却步。我们确定了导致这些管理费用的PRAC的两个弱点。首先，由于增加激活计数器需要额外的时间，PRAC增加了关键的RAM访问延迟参数。其次，PRAC一次执行固定次数的预防性刷新，使其容易受到称为波浪攻击的对抗访问模式的影响，因此需要将其配置为明显较小的激活阈值。为了解决PRAC的两个弱点，我们提出了一种新的动态内存RowHammer缓解机制Chronus。Chronus 1）通过将计数器与数据分开来提供访问服务同时更新行激活计数器，2）通过动态控制执行的预防性刷新的次数来防止波攻击。我们的性能分析表明，Chronus的系统性能开销对于现代的动态随机存储器芯片来说接近于零，而对于未来的动态随机存储器芯片来说则非常低。Chronus的性能优于PRAC的三种变体和其他三种最先进的读取干扰解决方案。我们讨论了Chronus和PRAC对未来系统的影响，并预示了未来的研究方向。为了帮助未来的研究，我们在https://github.com/CMU-SAFARI/Chronus上开源了Chronus实现。



## **25. Cascading and Proxy Membership Inference Attacks**

级联和代理成员推断攻击 cs.CR

Accepted by The Network and Distributed System Security (NDSS)  Symposium, 2026

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2507.21412v3) [paper-pdf](http://arxiv.org/pdf/2507.21412v3)

**Authors**: Yuntao Du, Jiacheng Li, Yuetian Chen, Kaiyuan Zhang, Zhizhen Yuan, Hanshen Xiao, Bruno Ribeiro, Ninghui Li

**Abstract**: A Membership Inference Attack (MIA) assesses how much a trained machine learning model reveals about its training data by determining whether specific query instances were included in the dataset. We classify existing MIAs into adaptive or non-adaptive, depending on whether the adversary is allowed to train shadow models on membership queries. In the adaptive setting, where the adversary can train shadow models after accessing query instances, we highlight the importance of exploiting membership dependencies between instances and propose an attack-agnostic framework called Cascading Membership Inference Attack (CMIA), which incorporates membership dependencies via conditional shadow training to boost membership inference performance.   In the non-adaptive setting, where the adversary is restricted to training shadow models before obtaining membership queries, we introduce Proxy Membership Inference Attack (PMIA). PMIA employs a proxy selection strategy that identifies samples with similar behaviors to the query instance and uses their behaviors in shadow models to perform a membership posterior odds test for membership inference. We provide theoretical analyses for both attacks, and extensive experimental results demonstrate that CMIA and PMIA substantially outperform existing MIAs in both settings, particularly in the low false-positive regime, which is crucial for evaluating privacy risks.

摘要: 成员资格推理攻击（MIA）通过确定数据集中是否包括特定的查询实例来评估经过训练的机器学习模型对其训练数据的揭示程度。我们将现有的MIA分为自适应或非自适应，具体取决于是否允许对手在成员资格查询上训练影子模型。在自适应环境中，对手可以在访问查询实例后训练影子模型，我们强调了利用实例之间成员依赖关系的重要性，并提出了一种名为级联成员推断攻击（CMIA）的攻击不可知框架，该框架通过条件影子训练合并成员依赖关系，以提高成员推断性能。   在非自适应环境中，对手仅限于在获得成员资格查询之前训练影子模型，我们引入代理成员资格推断攻击（PMIA）。PMIA采用代理选择策略，该策略识别与查询实例具有相似行为的样本，并使用其在影子模型中的行为来执行成员资格后验赔率测试以进行成员资格推断。我们对这两种攻击提供了理论分析，大量的实验结果表明，CMIA和PMIA在这两种环境下的表现都大大优于现有的MIA，特别是在低假阳性机制下，这对于评估隐私风险至关重要。



## **26. Quantum machine unlearning**

量子机器学习 quant-ph

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2509.06086v1) [paper-pdf](http://arxiv.org/pdf/2509.06086v1)

**Authors**: Junjian Su, Runze He, Guanghui Li, Sujuan Qin, Zhimin He, Haozhen Situ, Fei Gao

**Abstract**: Quantum Machine Learning (QML) integrates quantum computation with classical Machine Learning (ML) and holds the potential to achieve the quantum advantage for specific tasks. In classical ML, Machine Unlearning (MU) is a crucial strategy for removing the influence of specified training data from a model, to meet regulatory requirements and mitigate privacy risks. However, both the risk of training-data membership leakage remains underexplored in QML. This motivates us to propose Quantum Machine Unlearning (QMU) to explore two core questions: do QML models require MU due to training-data membership leakage, and can MU mechanisms be efficiently implemented in the QML? To answer the two questions, we conducted experiments on the MNIST classification task, utilizing a class-wise unlearning paradigm in both noiseless simulations and quantum hardware. First, we quantify training-data privacy leakage using a Membership Inference Attack (MIA), observing average success rates of 90.2\% in noiseless simulations and 75.3\% on quantum hardware. These results indicate that QML models present training-data membership leakage with very high probability under adversarial access, motivating the need for MU. Second, we implement MU algorithms on the QML model, which reduces the average MIA success rate to 0\% in simulations and 3.7\% on quantum hardware while preserving accuracy on retained data. We conclude that implementing MU mechanisms in QML models renders them resistant to MIA. Overall, this paper reveals significant privacy vulnerabilities in QML models and provides effective corresponding defense strategies, providing a potential path toward privacy-preserving QML systems.

摘要: 量子机器学习（QML）将量子计算与经典机器学习（ML）集成在一起，具有在特定任务中实现量子优势的潜力。在经典ML中，机器取消学习（MU）是消除模型中指定训练数据影响、满足监管要求并降低隐私风险的关键策略。然而，QML中对训练数据成员资格泄露的风险仍然没有充分研究。这促使我们提出量子机器非学习（QMU）来探索两个核心问题：QML模型是否因训练数据成员资格泄露而需要MU，以及MU机制能否在QML中有效实现？为了回答这两个问题，我们在MNIST分类任务上进行了实验，在无噪声模拟和量子硬件中使用了类学习范式。首先，我们使用成员推理攻击（MIA）量化训练数据隐私泄露，在无噪声模拟中观察到90.2%的平均成功率，在量子硬件上观察到75.3%的平均成功率。这些结果表明，QML模型在对抗访问下以非常高的概率呈现训练数据成员泄漏，从而激发了对MU的需求。其次，我们在QML模型上实现MU算法，这将模拟中的平均MIA成功率降低到0\%，在量子硬件上降低到3.7\%，同时保持保留数据的准确性。我们得出的结论是，在QML模型中实施MU机制使它们能够抵抗MIA。总体而言，本文揭示了QML模型中的重大隐私漏洞，并提供了有效的相应防御策略，为保护隐私的QML系统提供了一条潜在的途径。



## **27. Asymmetry Vulnerability and Physical Attacks on Online Map Construction for Autonomous Driving**

自动驾驶在线地图构建的不对称漏洞和物理攻击 cs.CR

CCS'25 (a shorter version of this paper will appear in the conference  proceeding)

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2509.06071v1) [paper-pdf](http://arxiv.org/pdf/2509.06071v1)

**Authors**: Yang Lou, Haibo Hu, Qun Song, Qian Xu, Yi Zhu, Rui Tan, Wei-Bin Lee, Jianping Wang

**Abstract**: High-definition maps provide precise environmental information essential for prediction and planning in autonomous driving systems. Due to the high cost of labeling and maintenance, recent research has turned to online HD map construction using onboard sensor data, offering wider coverage and more timely updates for autonomous vehicles. However, the robustness of online map construction under adversarial conditions remains underexplored. In this paper, we present a systematic vulnerability analysis of online map construction models, which reveals that these models exhibit an inherent bias toward predicting symmetric road structures. In asymmetric scenes like forks or merges, this bias often causes the model to mistakenly predict a straight boundary that mirrors the opposite side. We demonstrate that this vulnerability persists in the real-world and can be reliably triggered by obstruction or targeted interference. Leveraging this vulnerability, we propose a novel two-stage attack framework capable of manipulating online constructed maps. First, our method identifies vulnerable asymmetric scenes along the victim AV's potential route. Then, we optimize the location and pattern of camera-blinding attacks and adversarial patch attacks. Evaluations on a public AD dataset demonstrate that our attacks can degrade mapping accuracy by up to 9.9%, render up to 44% of targeted routes unreachable, and increase unsafe planned trajectory rates, colliding with real-world road boundaries, by up to 27%. These attacks are also validated on a real-world testbed vehicle. We further analyze root causes of the symmetry bias, attributing them to training data imbalance, model architecture, and map element representation. To the best of our knowledge, this study presents the first vulnerability assessment of online map construction models and introduces the first digital and physical attack against them.

摘要: 高清地图为自动驾驶系统的预测和规划提供了必不可少的精确环境信息。由于标签和维护成本高昂，最近的研究转向使用车载传感器数据构建在线高清地图，为自动驾驶汽车提供更广泛的覆盖范围和更及时的更新。然而，在线地图构建在对抗条件下的鲁棒性仍有待研究。在本文中，我们提出了一个系统的脆弱性分析的在线地图建设模型，这表明，这些模型表现出固有的偏向预测对称的道路结构。在分叉或合并等不对称场景中，这种偏差经常导致模型错误地预测反映对面的直线边界。我们证明，这种漏洞在现实世界中持续存在，并且可以由阻碍或有针对性的干扰可靠地触发。利用这个漏洞，我们提出了一种新颖的两阶段攻击框架，能够操纵在线构建的地图。首先，我们的方法沿着受害者AV的潜在路线识别脆弱的不对称场景。然后，我们优化摄像机致盲攻击和对抗性补丁攻击的位置和模式。对公共AD数据集的评估表明，我们的攻击可以使地图准确性降低高达9.9%，使多达44%的目标路线无法到达，并增加不安全的计划轨迹率，并与现实世界的道路边界发生碰撞，增加高达27%。这些攻击也在现实世界的测试平台车辆上得到了验证。我们进一步分析了对称性偏差的根本原因，将其归因于训练数据不平衡、模型架构和地图元素表示。据我们所知，这项研究首次对在线地图构建模型进行了脆弱性评估，并介绍了针对它们的首次数字和物理攻击。



## **28. ComplicitSplat: Downstream Models are Vulnerable to Blackbox Attacks by 3D Gaussian Splat Camouflages**

CompicitSplat：下游模型容易受到3D高斯Splat Camemages的黑匣子攻击 cs.CV

7 pages, 6 figures

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2508.11854v2) [paper-pdf](http://arxiv.org/pdf/2508.11854v2)

**Authors**: Matthew Hull, Haoyang Yang, Pratham Mehta, Mansi Phute, Aeree Cho, Haorang Wang, Matthew Lau, Wenke Lee, Wilian Lunardi, Martin Andreoni, Duen Horng Chau

**Abstract**: As 3D Gaussian Splatting (3DGS) gains rapid adoption in safety-critical tasks for efficient novel-view synthesis from static images, how might an adversary tamper images to cause harm? We introduce ComplicitSplat, the first attack that exploits standard 3DGS shading methods to create viewpoint-specific camouflage - colors and textures that change with viewing angle - to embed adversarial content in scene objects that are visible only from specific viewpoints and without requiring access to model architecture or weights. Our extensive experiments show that ComplicitSplat generalizes to successfully attack a variety of popular detector - both single-stage, multi-stage, and transformer-based models on both real-world capture of physical objects and synthetic scenes. To our knowledge, this is the first black-box attack on downstream object detectors using 3DGS, exposing a novel safety risk for applications like autonomous navigation and other mission-critical robotic systems.

摘要: 随着3D高斯飞溅（3DGS）在安全关键任务中迅速采用，从静态图像高效合成新视图，对手可能会如何篡改图像造成伤害？我们引入了CompicitSplat，这是第一个利用标准3DGS着色方法来创建特定于视角的伪装（随着视角而变化的颜色和纹理）的攻击，将对抗性内容嵌入到仅从特定视角可见的场景对象中，无需访问模型架构或权重。我们广泛的实验表明，CompicitSplat可以推广到成功攻击各种流行的检测器-包括针对物理对象和合成场景的现实世界捕获的单级、多级和基于变换器的模型。据我们所知，这是第一次使用3DGS对下游物体检测器进行黑匣子攻击，暴露了自主导航和其他关键任务机器人系统等应用的新型安全风险。



## **29. Yours or Mine? Overwriting Attacks against Neural Audio Watermarking**

你的还是我的？针对神经音频水印的覆盖攻击 cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05835v1) [paper-pdf](http://arxiv.org/pdf/2509.05835v1)

**Authors**: Lingfeng Yao, Chenpei Huang, Shengyao Wang, Junpei Xue, Hanqing Guo, Jiang Liu, Phone Lin, Tomoaki Ohtsuki, Miao Pan

**Abstract**: As generative audio models are rapidly evolving, AI-generated audios increasingly raise concerns about copyright infringement and misinformation spread. Audio watermarking, as a proactive defense, can embed secret messages into audio for copyright protection and source verification. However, current neural audio watermarking methods focus primarily on the imperceptibility and robustness of watermarking, while ignoring its vulnerability to security attacks. In this paper, we develop a simple yet powerful attack: the overwriting attack that overwrites the legitimate audio watermark with a forged one and makes the original legitimate watermark undetectable. Based on the audio watermarking information that the adversary has, we propose three categories of overwriting attacks, i.e., white-box, gray-box, and black-box attacks. We also thoroughly evaluate the proposed attacks on state-of-the-art neural audio watermarking methods. Experimental results demonstrate that the proposed overwriting attacks can effectively compromise existing watermarking schemes across various settings and achieve a nearly 100% attack success rate. The practicality and effectiveness of the proposed overwriting attacks expose security flaws in existing neural audio watermarking systems, underscoring the need to enhance security in future audio watermarking designs.

摘要: 随着生成音频模型的迅速发展，人工智能生成的音频越来越引发人们对版权侵权和错误信息传播的担忧。音频水印作为一种主动防御，可以将秘密消息嵌入音频中，以实现版权保护和源验证。然而，目前的神经音频水印方法主要关注水印的不可感知性和鲁棒性，而忽视了其对安全攻击的脆弱性。在本文中，我们开发了一种简单但强大的攻击：MIDI攻击，用伪造的水印覆盖合法的音频水印，并使原始的合法水印无法检测。根据对手拥有的音频水印信息，我们提出了三种类型的MIDI攻击，即，白盒、灰盒和黑匣子攻击。我们还彻底评估了对最先进的神经音频水印方法提出的攻击。实验结果表明，提出的MIDI攻击可以有效地破坏各种设置中的现有水印方案，并达到近100%的攻击成功率。提出的MIDI攻击的实用性和有效性暴露了现有神经音频水印系统中的安全缺陷，强调了在未来音频水印设计中增强安全性的必要性。



## **30. Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization**

解码LLM中的潜在攻击：通过Web摘要中的HTML提示注入 cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05831v1) [paper-pdf](http://arxiv.org/pdf/2509.05831v1)

**Authors**: Ishaan Verma

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.

摘要: 大型语言模型（LLM）越来越多地集成到基于Web的内容摘要系统中，但它们对即时注入攻击的敏感性仍然是一个紧迫的问题。在这项研究中，我们探索了如何利用非可见的HTML元素（例如<meta>、咏叹调标签和alt属性）来嵌入对抗性指令，而不改变网页的可见内容。我们引入了一个由280个静态网页组成的新颖数据集，平均分为干净和对抗注入版本，使用不同的基于HTML的策略制作。这些页面通过浏览器自动化管道进行处理，以提取原始HTML和渲染文本，密切模仿现实世界的LLM部署场景。我们评估了两个最先进的开源模型Llama 4 Scout（Meta）和Gemma 9 B IT（Google）总结此内容的能力。使用词汇（ROUGE-L）和语义（SBERT cos相似性）指标以及手动注释，我们评估这些隐蔽注入的影响。我们的研究结果显示，超过29%的注射样本导致Llama 4 Scout总结发生了显着变化，而Gemma 9 B IT的成功率较低，但并非微不足道，为15%。这些结果凸显了LLM驱动的网络管道中一个关键且在很大程度上被忽视的漏洞，其中隐藏的对抗内容可以巧妙地操纵模型输出。我们的工作为评估基于HTML的即时注入提供了一个可重复的框架和基准，并强调了涉及Web内容的LLM应用程序中对稳健的缓解策略的迫切需要。



## **31. SEASONED: Semantic-Enhanced Self-Counterfactual Explainable Detection of Adversarial Exploiter Contracts**

SEASONED：语义增强的自我反事实可解释的对抗性剥削者契约检测 cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05681v1) [paper-pdf](http://arxiv.org/pdf/2509.05681v1)

**Authors**: Xng Ai, Shudan Lin, Zecheng Li, Kai Zhou, Bixin Li, Bin Xiao

**Abstract**: Decentralized Finance (DeFi) attacks have resulted in significant losses, often orchestrated through Adversarial Exploiter Contracts (AECs) that exploit vulnerabilities in victim smart contracts. To proactively identify such threats, this paper targets the explainable detection of AECs.   Existing detection methods struggle to capture semantic dependencies and lack interpretability, limiting their effectiveness and leaving critical knowledge gaps in AEC analysis. To address these challenges, we introduce SEASONED, an effective, self-explanatory, and robust framework for AEC detection.   SEASONED extracts semantic information from contract bytecode to construct a semantic relation graph (SRG), and employs a self-counterfactual explainable detector (SCFED) to classify SRGs and generate explanations that highlight the core attack logic. SCFED further enhances robustness, generalizability, and data efficiency by extracting representative information from these explanations. Both theoretical analysis and experimental results demonstrate the effectiveness of SEASONED, which showcases outstanding detection performance, robustness, generalizability, and data efficiency learning ability. To support further research, we also release a new dataset of 359 AECs.

摘要: 去中心化金融（DeFi）攻击造成了重大损失，通常是通过利用受害者智能合同中的漏洞的对抗性剥削者合同（AEC）精心策划的。为了主动识别此类威胁，本文的目标是AEC的可解释检测。   现有的检测方法难以捕捉语义依赖性并且缺乏可解释性，从而限制了其有效性并在AEC分析中留下了关键的知识空白。为了应对这些挑战，我们引入了SEASONED，这是一个有效、不言自明且强大的AEC检测框架。   SEASONED从合同字节码中提取语义信息以构建语义关系图（SRG），并采用自反事实可解释检测器（SCPED）对SRG进行分类并生成突出核心攻击逻辑的解释。SCPED通过从这些解释中提取代表性信息，进一步增强了稳健性、可概括性和数据效率。理论分析和实验结果都证明了SEASSONED的有效性，它具有出色的检测性能、鲁棒性、可推广性和数据效率学习能力。为了支持进一步的研究，我们还发布了包含359个AEC的新数据集。



## **32. Privacy-Preserving Federated Learning via Homomorphic Adversarial Networks**

通过同形对抗网络保护隐私的联邦学习 cs.CR

Accepted by KSEM 2025 (This is the extended version)

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2412.01650v3) [paper-pdf](http://arxiv.org/pdf/2412.01650v3)

**Authors**: Wenhan Dong, Chao Lin, Xinlei He, Shengmin Xu, Xinyi Huang

**Abstract**: Privacy-preserving federated learning (PPFL) aims to train a global model for multiple clients while maintaining their data privacy. However, current PPFL protocols exhibit one or more of the following insufficiencies: considerable degradation in accuracy, the requirement for sharing keys, and cooperation during the key generation or decryption processes. As a mitigation, we develop the first protocol that utilizes neural networks to implement PPFL, as well as incorporating an Aggregatable Hybrid Encryption scheme tailored to the needs of PPFL. We name these networks as Homomorphic Adversarial Networks (HANs) which demonstrate that neural networks are capable of performing tasks similar to multi-key homomorphic encryption (MK-HE) while solving the problems of key distribution and collaborative decryption. Our experiments show that HANs are robust against privacy attacks. Compared with non-private federated learning, experiments conducted on multiple datasets demonstrate that HANs exhibit a negligible accuracy loss (at most 1.35%). Compared to traditional MK-HE schemes, HANs increase encryption aggregation speed by 6,075 times while incurring a 29.2 times increase in communication overhead.

摘要: 隐私保护联邦学习（PPFL）旨在为多个客户训练全球模型，同时维护他们的数据隐私。然而，当前的PPFL协议表现出以下一个或多个缺点：准确性大幅下降、共享密钥的要求以及密钥生成或解密过程中的合作。作为缓解措施，我们开发了第一个利用神经网络来实现PPFL的协议，并结合了根据PPFL需求量身定制的可聚合混合加密方案。我们将这些网络命名为Homomorphic Adversarial Network（HAN），这表明神经网络能够执行与多密钥Homomorphic加密（MK-HE）类似的任务，同时解决密钥分发和协作解密的问题。我们的实验表明HAN对于隐私攻击具有强大的抵抗力。与非私有联邦学习相比，在多个数据集上进行的实验表明，HAN的准确性损失可以忽略不计（最多1.35%）。与传统的MK-HE方案相比，HAN将加密聚合速度提高了6，075倍，同时导致通信费用增加了29.2倍。



## **33. Evaluating the Robustness and Accuracy of Text Watermarking Under Real-World Cross-Lingual Manipulations**

评估现实世界跨语言操作下文本水印的稳健性和准确性 cs.CL

Accepted by EMNLP 2025 Finding

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2502.16699v2) [paper-pdf](http://arxiv.org/pdf/2502.16699v2)

**Authors**: Mansour Al Ghanim, Jiaqi Xue, Rochana Prih Hastuti, Mengxin Zheng, Yan Solihin, Qian Lou

**Abstract**: We present a study to benchmark representative watermarking methods in cross-lingual settings. The current literature mainly focuses on the evaluation of watermarking methods for the English language. However, the literature for evaluating watermarking in cross-lingual settings is scarce. This results in overlooking important adversary scenarios in which a cross-lingual adversary could be in, leading to a gray area of practicality over cross-lingual watermarking. In this paper, we evaluate four watermarking methods in four different and vocabulary rich languages. Our experiments investigate the quality of text under different watermarking procedure and the detectability of watermarks with practical translation attack scenarios. Specifically, we investigate practical scenarios that an adversary with cross-lingual knowledge could take, and evaluate whether current watermarking methods are suitable for such scenarios. Finally, from our findings, we draw key insights about watermarking in cross-lingual settings.

摘要: 我们提出了一项研究来对跨语言环境中的代表性水印方法进行基准测试。当前的文献主要集中在英语水印方法的评估上。然而，用于评估跨语言环境中水印的文献很少。这会导致忽视跨语言对手可能存在的重要对手场景，从而导致跨语言水印的实用性灰色区域。本文中，我们评估了四种不同且词汇丰富的语言中的四种水印方法。我们的实验研究了不同水印过程下的文本质量以及实际翻译攻击场景下水印的可检测性。具体来说，我们调查具有跨语言知识的对手可能采取的实际场景，并评估当前的水印方法是否适合此类场景。最后，从我们的研究结果中，我们得出了有关跨语言环境中水印的关键见解。



## **34. Evo-MARL: Co-Evolutionary Multi-Agent Reinforcement Learning for Internalized Safety**

Evo-MARL：用于内部化安全的协同进化多智能体强化学习 cs.AI

accepted by the Trustworthy FMs workshop in ICCV 2025

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2508.03864v2) [paper-pdf](http://arxiv.org/pdf/2508.03864v2)

**Authors**: Zhenyu Pan, Yiting Zhang, Yutong Zhang, Jianshu Zhang, Haozheng Luo, Yuwei Han, Dennis Wu, Hong-Yu Chen, Philip S. Yu, Manling Li, Han Liu

**Abstract**: Multi-agent systems (MAS) built on multimodal large language models exhibit strong collaboration and performance. However, their growing openness and interaction complexity pose serious risks, notably jailbreak and adversarial attacks. Existing defenses typically rely on external guard modules, such as dedicated safety agents, to handle unsafe behaviors. Unfortunately, this paradigm faces two challenges: (1) standalone agents offer limited protection, and (2) their independence leads to single-point failure-if compromised, system-wide safety collapses. Naively increasing the number of guard agents further raises cost and complexity. To address these challenges, we propose Evo-MARL, a novel multi-agent reinforcement learning (MARL) framework that enables all task agents to jointly acquire defensive capabilities. Rather than relying on external safety modules, Evo-MARL trains each agent to simultaneously perform its primary function and resist adversarial threats, ensuring robustness without increasing system overhead or single-node failure. Furthermore, Evo-MARL integrates evolutionary search with parameter-sharing reinforcement learning to co-evolve attackers and defenders. This adversarial training paradigm internalizes safety mechanisms and continually enhances MAS performance under co-evolving threats. Experiments show that Evo-MARL reduces attack success rates by up to 22% while boosting accuracy by up to 5% on reasoning tasks-demonstrating that safety and utility can be jointly improved.

摘要: 基于多模式大型语言模型构建的多智能体系统（MAS）展现出强大的协作和性能。然而，它们日益增长的开放性和交互复杂性带来了严重的风险，特别是越狱和对抗性攻击。现有的防御系统通常依赖外部防护模块（例如专用安全代理）来处理不安全行为。不幸的是，这个范式面临着两个挑战：（1）独立代理提供的保护有限，（2）它们的独立性会导致单点故障--如果受到损害，系统范围的安全就会崩溃。天真地增加警卫特工的数量进一步增加了成本和复杂性。为了应对这些挑战，我们提出了Evo-MARL，这是一种新型的多智能体强化学习（MARL）框架，使所有任务智能体能够联合获得防御能力。Evo-MARL不是依赖外部安全模块，而是训练每个代理同时执行其主要功能并抵抗对抗威胁，在不增加系统负担或单节点故障的情况下确保稳健性。此外，Evo-MARL将进化搜索与参数共享强化学习集成，以共同进化攻击者和防御者。这种对抗性训练范式内化了安全机制，并在共同演变的威胁下不断增强MAS性能。实验表明，Evo-MARL将攻击成功率降低高达22%，同时将推理任务的准确性提高高达5%，证明安全性和实用性可以共同提高。



## **35. Behind the Mask: Benchmarking Camouflaged Jailbreaks in Large Language Models**

面具背后：大型语言模型中伪装越狱的基准 cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05471v1) [paper-pdf](http://arxiv.org/pdf/2509.05471v1)

**Authors**: Youjia Zheng, Mohammad Zandsalimy, Shanu Sushmita

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to a sophisticated form of adversarial prompting known as camouflaged jailbreaking. This method embeds malicious intent within seemingly benign language to evade existing safety mechanisms. Unlike overt attacks, these subtle prompts exploit contextual ambiguity and the flexible nature of language, posing significant challenges to current defense systems. This paper investigates the construction and impact of camouflaged jailbreak prompts, emphasizing their deceptive characteristics and the limitations of traditional keyword-based detection methods. We introduce a novel benchmark dataset, Camouflaged Jailbreak Prompts, containing 500 curated examples (400 harmful and 100 benign prompts) designed to rigorously stress-test LLM safety protocols. In addition, we propose a multi-faceted evaluation framework that measures harmfulness across seven dimensions: Safety Awareness, Technical Feasibility, Implementation Safeguards, Harmful Potential, Educational Value, Content Quality, and Compliance Score. Our findings reveal a stark contrast in LLM behavior: while models demonstrate high safety and content quality with benign inputs, they exhibit a significant decline in performance and safety when confronted with camouflaged jailbreak attempts. This disparity underscores a pervasive vulnerability, highlighting the urgent need for more nuanced and adaptive security strategies to ensure the responsible and robust deployment of LLMs in real-world applications.

摘要: 大型语言模型（LLM）越来越容易受到一种复杂形式的对抗激励，即所谓的虚拟越狱。这种方法将恶意意图嵌入看似良性的语言中，以逃避现有的安全机制。与公开攻击不同，这些微妙的提示利用了上下文模糊性和语言的灵活性，对当前的防御系统构成了重大挑战。本文研究了伪装越狱提示的结构和影响，强调了其欺骗性特征和传统基于关键词的检测方法的局限性。我们引入了一个新颖的基准数据集，即伪装的越狱提示，其中包含500个精心策划的示例（400个有害提示和100个良性提示），旨在对LLM安全协议进行严格的压力测试。此外，我们还提出了一个多方面的评估框架，从七个方面衡量危害性：安全意识、技术可行性、实施保障措施、潜在危害性、教育价值、内容质量和合规评分。我们的研究结果揭示了LLM行为的鲜明对比：虽然模型在良性输入下表现出高安全性和内容质量，但当面对伪装的越狱尝试时，它们的性能和安全性却显着下降。这种差异凸显了普遍存在的漏洞，凸显了迫切需要更加细致入微和适应性的安全策略，以确保在现实世界应用程序中负责任且稳健地部署LLM。



## **36. On Evaluating the Poisoning Robustness of Federated Learning under Local Differential Privacy**

局部差异隐私下联邦学习中毒鲁棒性评估 cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05265v1) [paper-pdf](http://arxiv.org/pdf/2509.05265v1)

**Authors**: Zijian Wang, Wei Tong, Tingxuan Han, Haoyu Chen, Tianling Zhang, Yunlong Mao, Sheng Zhong

**Abstract**: Federated learning (FL) combined with local differential privacy (LDP) enables privacy-preserving model training across decentralized data sources. However, the decentralized data-management paradigm leaves LDPFL vulnerable to participants with malicious intent. The robustness of LDPFL protocols, particularly against model poisoning attacks (MPA), where adversaries inject malicious updates to disrupt global model convergence, remains insufficiently studied. In this paper, we propose a novel and extensible model poisoning attack framework tailored for LDPFL settings. Our approach is driven by the objective of maximizing the global training loss while adhering to local privacy constraints. To counter robust aggregation mechanisms such as Multi-Krum and trimmed mean, we develop adaptive attacks that embed carefully crafted constraints into a reverse training process, enabling evasion of these defenses. We evaluate our framework across three representative LDPFL protocols, three benchmark datasets, and two types of deep neural networks. Additionally, we investigate the influence of data heterogeneity and privacy budgets on attack effectiveness. Experimental results demonstrate that our adaptive attacks can significantly degrade the performance of the global model, revealing critical vulnerabilities and highlighting the need for more robust LDPFL defense strategies against MPA. Our code is available at https://github.com/ZiJW/LDPFL-Attack

摘要: 联合学习（FL）与局部差分隐私（LDP）相结合，可以在分散的数据源中进行隐私保护模型训练。然而，分散的数据管理模式使LDPFL容易受到恶意参与者的攻击。LDPFL协议的鲁棒性，特别是对模型中毒攻击（MPA），其中对手注入恶意更新破坏全局模型收敛，仍然没有得到充分的研究。在本文中，我们提出了一种新的和可扩展的模型中毒攻击框架，为LDPFL设置量身定制。我们的方法是由最大限度地提高全球培训损失，同时坚持本地隐私约束的目标。为了对抗强大的聚合机制，如Multi-Krum和Trimmed Mean，我们开发了自适应攻击，将精心制作的约束嵌入到反向训练过程中，从而能够规避这些防御。我们在三个代表性的LDPFL协议，三个基准数据集和两种类型的深度神经网络上评估了我们的框架。此外，我们调查的数据异质性和隐私预算对攻击效果的影响。实验结果表明，我们的自适应攻击可以显着降低全局模型的性能，揭示关键漏洞，并强调需要更强大的LDPFL防御策略对MPA。我们的代码可在https://github.com/ZiJW/LDPFL-Attack上获取



## **37. Jamming Smarter, Not Harder: Exploiting O-RAN Y1 RAN Analytics for Efficient Interference**

更智能、而不是更难干扰：利用O-RAN Y1 RAN分析实现高效干扰 cs.CR

8 pages, 7 figures

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05161v1) [paper-pdf](http://arxiv.org/pdf/2509.05161v1)

**Authors**: Abiodun Ganiyu, Dara Ron, Syed Rafiul Hussain, Vijay K Shah

**Abstract**: The Y1 interface in O-RAN enables the sharing of RAN Analytics Information (RAI) between the near-RT RIC and authorized Y1 consumers, which may be internal applications within the operator's trusted domain or external systems accessing data through a secure exposure function. While this visibility enhances network optimization and enables advanced services, it also introduces a potential security risk -- a malicious or compromised Y1 consumer could misuse analytics to facilitate targeted interference. In this work, we demonstrate how an adversary can exploit the Y1 interface to launch selective jamming attacks by passively monitoring downlink metrics. We propose and evaluate two Y1-aided jamming strategies: a clustering-based jammer leveraging DBSCAN for traffic profiling and a threshold-based jammer. These are compared against two baselines strategies -- always-on jammer and random jammer -- on an over-the-air LTE/5G O-RAN testbed. Experimental results show that in unconstrained jamming budget scenarios, the threshold-based jammer can closely replicate the disruption caused by always-on jamming while reducing transmission time by 27\%. Under constrained jamming budgets, the clustering-based jammer proves most effective, causing up to an 18.1\% bitrate drop while remaining active only 25\% of the time. These findings reveal a critical trade-off between jamming stealthiness and efficiency, and illustrate how exposure of RAN analytics via the Y1 interface can enable highly targeted, low-overhead attacks, raising important security considerations for both civilian and mission-critical O-RAN deployments.

摘要: O-RAN中的Y1接口支持近RT RIC和授权Y1消费者之间共享RAN分析信息（RAI），这些消费者可以是运营商可信域内的内部应用程序，也可以是通过安全暴露功能访问数据的外部系统。虽然这种可见性增强了网络优化并支持高级服务，但它也带来了潜在的安全风险--恶意或受损害的Y1消费者可能会滥用分析来促进有针对性的干扰。在这项工作中，我们演示了对手如何利用Y1接口通过被动监视下行链路指标来发起选择性干扰攻击。我们提出并评估了两种Y1辅助干扰策略：利用DBSCAN进行流量分析的基于集群的干扰器和基于阈值的干扰器。将这些与空中LTE/5G O-RAN测试床上的两种基线策略（始终开启的干扰器和随机干扰器）进行比较。实验结果表明，在不受限制的干扰预算场景下，基于阈值的干扰器可以精确地复制永远在线干扰造成的干扰，同时将传输时间缩短27%。在受限制的干扰预算下，基于集群的干扰器被证明是最有效的，导致高达18.1%的比特率下降，而只有25%的时间保持活跃。这些发现揭示了干扰隐蔽性和效率之间的关键权衡，并说明了通过Y1接口暴露的RAN分析如何能够实现高度针对性、低开销的攻击，从而为民用和关键任务O-RAN部署提出了重要的安全考虑。



## **38. Robust Experts: the Effect of Adversarial Training on CNNs with Sparse Mixture-of-Experts Layers**

稳健的专家：对抗训练对具有稀疏专家混合层的CNN的影响 cs.CV

Accepted for publication at the STREAM workshop at ICCV 2025

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05086v1) [paper-pdf](http://arxiv.org/pdf/2509.05086v1)

**Authors**: Svetlana Pavlitska, Haixi Fan, Konstantin Ditschuneit, J. Marius Zöllner

**Abstract**: Robustifying convolutional neural networks (CNNs) against adversarial attacks remains challenging and often requires resource-intensive countermeasures. We explore the use of sparse mixture-of-experts (MoE) layers to improve robustness by replacing selected residual blocks or convolutional layers, thereby increasing model capacity without additional inference cost. On ResNet architectures trained on CIFAR-100, we find that inserting a single MoE layer in the deeper stages leads to consistent improvements in robustness under PGD and AutoPGD attacks when combined with adversarial training. Furthermore, we discover that when switch loss is used for balancing, it causes routing to collapse onto a small set of overused experts, thereby concentrating adversarial training on these paths and inadvertently making them more robust. As a result, some individual experts outperform the gated MoE model in robustness, suggesting that robust subpaths emerge through specialization. Our code is available at https://github.com/KASTEL-MobilityLab/robust-sparse-moes.

摘要: 增强卷积神经网络（CNN）对抗对抗性攻击仍然具有挑战性，并且通常需要资源密集型的应对措施。我们探索使用稀疏混合专家（MoE）层，通过替换选定的残差块或卷积层来提高鲁棒性，从而在不增加额外推理成本的情况下提高模型容量。在CIFAR-100上训练的ResNet架构上，我们发现，在更深的阶段中插入单个MoE层，当与对抗训练相结合时，可以在PGD和AutoPGD攻击下实现一致的鲁棒性改进。此外，我们发现，当使用交换机损失来平衡时，会导致路由崩溃到一小群过度使用的专家身上，从而将对抗训练集中在这些路径上，并无意中使它们更加稳健。因此，一些个人专家在稳健性方面优于门控MoE模型，这表明通过专业化出现了稳健的子路径。我们的代码可在https://github.com/KASTEL-MobilityLab/robust-sparse-moes上获取。



## **39. Adversarial Augmentation and Active Sampling for Robust Cyber Anomaly Detection**

对抗增强和主动采样用于鲁棒网络异常检测 cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04999v1) [paper-pdf](http://arxiv.org/pdf/2509.04999v1)

**Authors**: Sidahmed Benabderrahmane, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) present a considerable challenge to cybersecurity due to their stealthy, long-duration nature. Traditional supervised learning methods typically require large amounts of labeled data, which is often scarce in real-world scenarios. This paper introduces a novel approach that combines AutoEncoders for anomaly detection with active learning to iteratively enhance APT detection. By selectively querying an oracle for labels on uncertain or ambiguous samples, our method reduces labeling costs while improving detection accuracy, enabling the model to effectively learn with minimal data and reduce reliance on extensive manual labeling. We present a comprehensive formulation of the Attention Adversarial Dual AutoEncoder-based anomaly detection framework and demonstrate how the active learning loop progressively enhances the model's performance. The framework is evaluated on real-world, imbalanced provenance trace data from the DARPA Transparent Computing program, where APT-like attacks account for just 0.004\% of the data. The datasets, which cover multiple operating systems including Android, Linux, BSD, and Windows, are tested in two attack scenarios. The results show substantial improvements in detection rates during active learning, outperforming existing methods.

摘要: 高级持续性威胁（APT）由于其隐蔽性、持续时间长，对网络安全构成了相当大的挑战。传统的监督学习方法通常需要大量的标记数据，而这些数据在现实世界场景中通常很稀缺。本文介绍了一种新颖的方法，将用于异常检测的AutoEncoders与主动学习相结合，以迭代增强APT检测。通过选择性地向Oracle查询不确定或模糊样本的标签，我们的方法降低了标签成本，同时提高了检测准确性，使模型能够用最少的数据有效学习并减少对大量手动标签的依赖。我们提出了基于注意力对抗双AutoEnCoder的异常检测框架的全面公式，并演示了主动学习循环如何逐步增强模型的性能。该框架是根据DARPA透明计算程序中的现实世界、不平衡的出处跟踪数据进行评估的，其中类APT攻击仅占数据的0.004%。这些数据集涵盖Android、Linux、BDS和Windows等多个操作系统，并在两种攻击场景中进行了测试。结果显示，主动学习期间的检测率大幅提高，优于现有方法。



## **40. Training a Perceptual Model for Evaluating Auditory Similarity in Music Adversarial Attack**

训练用于评估音乐对抗性攻击中听觉相似性的感知模型 cs.SD

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04985v1) [paper-pdf](http://arxiv.org/pdf/2509.04985v1)

**Authors**: Yuxuan Liu, Rui Sang, Peihong Zhang, Zhixin Li, Shengchen Li

**Abstract**: Music Information Retrieval (MIR) systems are highly vulnerable to adversarial attacks that are often imperceptible to humans, primarily due to a misalignment between model feature spaces and human auditory perception. Existing defenses and perceptual metrics frequently fail to adequately capture these auditory nuances, a limitation supported by our initial listening tests showing low correlation between common metrics and human judgments. To bridge this gap, we introduce Perceptually-Aligned MERT Transformer (PAMT), a novel framework for learning robust, perceptually-aligned music representations. Our core innovation lies in the psychoacoustically-conditioned sequential contrastive transformer, a lightweight projection head built atop a frozen MERT encoder. PAMT achieves a Spearman correlation coefficient of 0.65 with subjective scores, outperforming existing perceptual metrics. Our approach also achieves an average of 9.15\% improvement in robust accuracy on challenging MIR tasks, including Cover Song Identification and Music Genre Classification, under diverse perceptual adversarial attacks. This work pioneers architecturally-integrated psychoacoustic conditioning, yielding representations significantly more aligned with human perception and robust against music adversarial attacks.

摘要: 音乐信息检索（MIR）系统非常容易受到人类通常难以察觉的对抗攻击，主要是由于模型特征空间和人类听觉感知之间的不一致。现有的防御和感知指标经常无法充分捕捉这些听觉细微差别，我们最初的听力测试支持了这一局限性，表明常见指标和人类判断之间的相关性较低。为了弥合这一差距，我们引入了感知对齐的MERT Transformer（PAMT），这是一种用于学习鲁棒的感知对齐音乐表示的新框架。我们的核心创新在于心理声学调节顺序对比Transformer，这是一个构建在冷冻MERT编码器上的轻型投影头。PAMT通过主观评分实现了0.65的Spearman相关系数，优于现有的感知指标。在各种感知对抗攻击下，我们的方法还在具有挑战性的MIR任务（包括翻唱歌曲识别和音乐流派分类）上实现了平均9.15%的稳健准确性提高。这项工作开创了架构集成的心理声学条件反射，产生的表示明显更符合人类感知，并且对音乐对抗攻击更强。



## **41. MAIA: An Inpainting-Based Approach for Music Adversarial Attacks**

MAIA：一种基于修补的音乐对抗攻击方法 cs.SD

Accepted at ISMIR2025

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04980v1) [paper-pdf](http://arxiv.org/pdf/2509.04980v1)

**Authors**: Yuxuan Liu, Peihong Zhang, Rui Sang, Zhixin Li, Shengchen Li

**Abstract**: Music adversarial attacks have garnered significant interest in the field of Music Information Retrieval (MIR). In this paper, we present Music Adversarial Inpainting Attack (MAIA), a novel adversarial attack framework that supports both white-box and black-box attack scenarios. MAIA begins with an importance analysis to identify critical audio segments, which are then targeted for modification. Utilizing generative inpainting models, these segments are reconstructed with guidance from the output of the attacked model, ensuring subtle and effective adversarial perturbations. We evaluate MAIA on multiple MIR tasks, demonstrating high attack success rates in both white-box and black-box settings while maintaining minimal perceptual distortion. Additionally, subjective listening tests confirm the high audio fidelity of the adversarial samples. Our findings highlight vulnerabilities in current MIR systems and emphasize the need for more robust and secure models.

摘要: 音乐对抗攻击在音乐信息检索（MIR）领域引起了人们的极大兴趣。在本文中，我们介绍了音乐对抗修补攻击（MAIA），这是一种新型的对抗攻击框架，支持白盒和黑盒攻击场景。MAIA首先进行重要性分析，以识别关键音频片段，然后针对这些片段进行修改。利用生成式修复模型，在受攻击模型输出的指导下重建这些片段，确保微妙且有效的对抗性扰动。我们评估了多个MIR任务中的MAIA，证明了在白盒和黑盒设置中的高攻击成功率，同时保持最小的感知失真。此外，主观听力测试证实了对抗样本的高音频保真度。我们的研究结果强调了当前MIR系统中的漏洞，并强调需要更强大和安全的模型。



## **42. RINSER: Accurate API Prediction Using Masked Language Models**

RINser：使用掩蔽语言模型进行准确的API预测 cs.CY

16 pages, 8 figures

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04887v1) [paper-pdf](http://arxiv.org/pdf/2509.04887v1)

**Authors**: Muhammad Ejaz Ahmed, Christopher Cody, Muhammad Ikram, Sean Lamont, Alsharif Abuadbba, Seyit Camtepe, Surya Nepal, Muhammad Ali Kaafar

**Abstract**: Malware authors commonly use obfuscation to hide API identities in binary files, making analysis difficult and time-consuming for a human expert to understand the behavior and intent of the program. Automatic API prediction tools are necessary to efficiently analyze unknown binaries, facilitating rapid malware triage while reducing the workload on human analysts. In this paper, we present RINSER (AccuRate API predictioN using maSked languagE model leaRning), an automated framework for predicting Windows API (WinAPI) function names. RINSER introduces the novel concept of API codeprints, a set of API-relevant assembly instructions, and supports x86 PE binaries. RINSER relies on BERT's masked language model (LM) to predict API names at scale, achieving 85.77% accuracy for normal binaries and 82.88% accuracy for stripped binaries. We evaluate RINSER on a large dataset of 4.7M API codeprints from 11,098 malware binaries, covering 4,123 unique Windows APIs, making it the largest publicly available dataset of this type. RINSER successfully discovered 65 obfuscated Windows APIs related to C2 communication, spying, and evasion in our dataset, which the commercial disassembler IDA failed to identify. Furthermore, we compared RINSER against three state-of-the-art approaches, showing over 20% higher prediction accuracy. We also demonstrated RINSER's resilience to adversarial attacks, including instruction randomization and code displacement, with a performance drop of no more than 3%.

摘要: 恶意软件作者通常使用混淆将API身份隐藏在二进制文件中，这使得人类专家理解程序的行为和意图变得困难且耗时。自动API预测工具对于有效分析未知二进制文件来说是必要的，可以促进快速恶意软件分类，同时减少人类分析师的工作量。在本文中，我们介绍了RINBER（使用maSked languagE模型leRning的ACATER API预测），这是一个用于预测Windows API（WinAPI）函数名称的自动化框架。RINser引入了API代码印的新颖概念，即一组与API相关的汇编指令，并支持x86 PE二进制文件。RINser依赖BERT的掩蔽语言模型（LM）来大规模预测API名称，正常二进制文件的准确性达到85.77%，剥离二进制文件的准确性达到82.88%。我们在一个包含来自11，098个恶意软件二进制文件的470万个API代码的大型数据集上评估了RINser，涵盖了4，123个独特的Windows API，使其成为此类类型中最大的公开可用数据集。RINBER在我们的数据集中成功发现了65个与C2通信、间谍和规避相关的模糊Windows API，但商业反汇编器IDA未能识别这些API。此外，我们将RINBER与三种最先进的方法进行了比较，结果显示预测准确性提高了20%以上。我们还展示了RINBER对对抗攻击（包括指令随机化和代码置换）的弹性，性能下降不超过3%。



## **43. PersonaTeaming: Exploring How Introducing Personas Can Improve Automated AI Red-Teaming**

角色协作：探索引入角色协作如何改善自动化人工智能红色协作 cs.AI

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.03728v2) [paper-pdf](http://arxiv.org/pdf/2509.03728v2)

**Authors**: Wesley Hanwen Deng, Sunnie S. Y. Kim, Akshita Jha, Ken Holstein, Motahhare Eslami, Lauren Wilcox, Leon A Gatys

**Abstract**: Recent developments in AI governance and safety research have called for red-teaming methods that can effectively surface potential risks posed by AI models. Many of these calls have emphasized how the identities and backgrounds of red-teamers can shape their red-teaming strategies, and thus the kinds of risks they are likely to uncover. While automated red-teaming approaches promise to complement human red-teaming by enabling larger-scale exploration of model behavior, current approaches do not consider the role of identity. As an initial step towards incorporating people's background and identities in automated red-teaming, we develop and evaluate a novel method, PersonaTeaming, that introduces personas in the adversarial prompt generation process to explore a wider spectrum of adversarial strategies. In particular, we first introduce a methodology for mutating prompts based on either "red-teaming expert" personas or "regular AI user" personas. We then develop a dynamic persona-generating algorithm that automatically generates various persona types adaptive to different seed prompts. In addition, we develop a set of new metrics to explicitly measure the "mutation distance" to complement existing diversity measurements of adversarial prompts. Our experiments show promising improvements (up to 144.1%) in the attack success rates of adversarial prompts through persona mutation, while maintaining prompt diversity, compared to RainbowPlus, a state-of-the-art automated red-teaming method. We discuss the strengths and limitations of different persona types and mutation methods, shedding light on future opportunities to explore complementarities between automated and human red-teaming approaches.

摘要: 人工智能治理和安全研究的最新进展呼吁采取红色团队方法，以有效地揭示人工智能模型带来的潜在风险。许多电话都强调了红色团队成员的身份和背景如何塑造他们的红色团队策略，从而也强调了他们可能发现的风险。虽然自动化红色团队方法有望通过更大规模地探索模型行为来补充人类红色团队，但目前的方法没有考虑身份的作用。作为将人们的背景和身份融入自动化红色团队的第一步，我们开发并评估了一种新型方法PersonaTeaming，该方法在对抗性提示生成过程中引入角色，以探索更广泛的对抗策略。特别是，我们首先引入了一种基于“红色团队专家”角色或“普通人工智能用户”角色来变异提示的方法。然后，我们开发了一个动态角色生成算法，该算法自动生成适应不同种子提示的各种角色类型。此外，我们开发了一组新的指标来明确测量“突变距离”，以补充现有的对抗提示多样性测量。我们的实验显示，与最先进的自动化红色团队方法RainbowPlus相比，通过角色突变的对抗提示的攻击成功率有了有希望的改进（高达144.1%），同时保持了提示的多样性。我们讨论了不同角色类型和突变方法的优点和局限性，揭示了未来探索自动化和人类红色团队方法之间互补性的机会。



## **44. Adversarial Hubness in Multi-Modal Retrieval**

多模式检索中的对抗性积极性 cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2412.14113v3) [paper-pdf](http://arxiv.org/pdf/2412.14113v3)

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries.   In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, and also for targeted attacks on queries related to specific, attacker-chosen concepts.   We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system implemented by Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub, generated using 100 random queries, is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries), demonstrating the strong generalization capabilities of adversarial hubs. We also investigate whether techniques for mitigating natural hubness can also mitigate adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.

摘要: Hubness是多维载体空间中的一种现象，其中自然分布的一个点与许多其他点异常接近。这是信息检索中一个众所周知的问题，会导致某些项意外（且错误地）看起来与许多查询相关。   在本文中，我们研究攻击者如何利用中心将多模式检索系统中的任何图像或音频输入变成对抗中心。对抗中心可用于注入通用对抗内容（例如，垃圾邮件）将响应数千个不同的查询而检索这些信息，并且还可以对与特定的攻击者选择的概念相关的查询进行有针对性的攻击。   我们提出了一种创建对抗中心的方法，并在基准多模式检索数据集和由流行的载体数据库Pinecone实现的图像到图像检索系统上评估所得中心。例如，在文本标题到图像检索中，使用100个随机查询生成的单个对抗中心被检索为25，000个测试查询中超过21，000个的前1最相关图像（相比之下，最常见的自然中心是仅对102个查询的前1响应），这表明了对抗中心的强大概括能力。我们还调查了减轻自然中心的技术是否也可以减轻对抗中心，并表明它们对针对与特定概念相关的查询的中心无效。



## **45. Breaking to Build: A Threat Model of Prompt-Based Attacks for Securing LLMs**

突破构建：用于保护LLM的基于预算的攻击威胁模型 cs.CL

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04615v1) [paper-pdf](http://arxiv.org/pdf/2509.04615v1)

**Authors**: Brennen Hill, Surendra Parla, Venkata Abhijeeth Balabhadruni, Atharv Prajod Padmalayam, Sujay Chandra Shekara Sharma

**Abstract**: The proliferation of Large Language Models (LLMs) has introduced critical security challenges, where adversarial actors can manipulate input prompts to cause significant harm and circumvent safety alignments. These prompt-based attacks exploit vulnerabilities in a model's design, training, and contextual understanding, leading to intellectual property theft, misinformation generation, and erosion of user trust. A systematic understanding of these attack vectors is the foundational step toward developing robust countermeasures. This paper presents a comprehensive literature survey of prompt-based attack methodologies, categorizing them to provide a clear threat model. By detailing the mechanisms and impacts of these exploits, this survey aims to inform the research community's efforts in building the next generation of secure LLMs that are inherently resistant to unauthorized distillation, fine-tuning, and editing.

摘要: 大型语言模型（LLM）的激增带来了关键的安全挑战，对抗行为者可以操纵输入提示造成重大伤害并规避安全一致。这些基于预算的攻击利用模型设计、培训和上下文理解中的漏洞，导致知识产权盗窃、错误信息生成和用户信任度侵蚀。系统地了解这些攻击载体是开发稳健对策的基础步骤。本文对基于预算的攻击方法进行了全面的文献调查，对它们进行了分类，以提供明确的威胁模型。通过详细介绍这些漏洞利用的机制和影响，本调查旨在为研究界构建下一代安全LLM的努力提供信息，这些LLM本质上可以抵抗未经授权的提炼、微调和编辑。



## **46. Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing**

Concept-ROT：使用模型编辑在大型语言模型中中毒概念 cs.LG

Published at ICLR 2025

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2412.13341v2) [paper-pdf](http://arxiv.org/pdf/2412.13341v2)

**Authors**: Keltin Grimes, Marco Christiani, David Shriver, Marissa Connor

**Abstract**: Model editing methods modify specific behaviors of Large Language Models by altering a small, targeted set of network weights and require very little data and compute. These methods can be used for malicious applications such as inserting misinformation or simple trojans that result in adversary-specified behaviors when a trigger word is present. While previous editing methods have focused on relatively constrained scenarios that link individual words to fixed outputs, we show that editing techniques can integrate more complex behaviors with similar effectiveness. We develop Concept-ROT, a model editing-based method that efficiently inserts trojans which not only exhibit complex output behaviors, but also trigger on high-level concepts -- presenting an entirely new class of trojan attacks. Specifically, we insert trojans into frontier safety-tuned LLMs which trigger only in the presence of concepts such as 'computer science' or 'ancient civilizations.' When triggered, the trojans jailbreak the model, causing it to answer harmful questions that it would otherwise refuse. Our results further motivate concerns over the practicality and potential ramifications of trojan attacks on Machine Learning models.

摘要: 模型编辑方法通过改变一小组有针对性的网络权重来修改大型语言模型的特定行为，并且需要很少的数据和计算。这些方法可用于恶意应用程序，例如插入错误信息或简单的特洛伊木马，当存在触发词时，这些木马会导致对手指定的行为。虽然之前的编辑方法专注于将单个单词与固定输出联系起来的相对受限的场景，但我们表明编辑技术可以以类似的效果集成更复杂的行为。我们开发了Concept-ROT，这是一种基于模型编辑的方法，可以有效地插入特洛伊木马，这些木马不仅表现出复杂的输出行为，而且还会触发高级概念--从而呈现出一种全新的特洛伊木马攻击。具体来说，我们将特洛伊木马插入到前沿安全调整的LLM中，这些LLM仅在存在“计算机科学”或“古代文明”等概念时才会触发。“当被触发时，特洛伊木马会越狱该模型，使其回答原本会拒绝的有害问题。我们的结果进一步引发了人们对机器学习模型木马攻击的实用性和潜在后果的担忧。



## **47. DisPatch: Disarming Adversarial Patches in Object Detection with Diffusion Models**

Dispatch：利用扩散模型消除对象检测中的对抗补丁 cs.CV

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04597v1) [paper-pdf](http://arxiv.org/pdf/2509.04597v1)

**Authors**: Jin Ma, Mohammed Aldeen, Christopher Salas, Feng Luo, Mashrur Chowdhury, Mert Pesé, Long Cheng

**Abstract**: Object detection is fundamental to various real-world applications, such as security monitoring and surveillance video analysis. Despite their advancements, state-of-theart object detectors are still vulnerable to adversarial patch attacks, which can be easily applied to real-world objects to either conceal actual items or create non-existent ones, leading to severe consequences. Given the current diversity of adversarial patch attacks and potential unknown threats, an ideal defense method should be effective, generalizable, and robust against adaptive attacks. In this work, we introduce DISPATCH, the first diffusion-based defense framework for object detection. Unlike previous works that aim to "detect and remove" adversarial patches, DISPATCH adopts a "regenerate and rectify" strategy, leveraging generative models to disarm attack effects while preserving the integrity of the input image. Specifically, we utilize the in-distribution generative power of diffusion models to regenerate the entire image, aligning it with benign data. A rectification process is then employed to identify and replace adversarial regions with their regenerated benign counterparts. DISPATCH is attack-agnostic and requires no prior knowledge of the existing patches. Extensive experiments across multiple detectors and attacks demonstrate that DISPATCH consistently outperforms state-of-the-art defenses on both hiding attacks and creating attacks, achieving the best overall mAP.5 score of 89.3% on hiding attacks, and lowering the attack success rate to 24.8% on untargeted creating attacks. Moreover, it maintains strong robustness against adaptive attacks, making it a practical and reliable defense for object detection systems.

摘要: 对象检测是各种现实应用的基础，例如安全监控和监控视频分析。尽管它们取得了进步，但最先进的对象检测器仍然容易受到对抗补丁攻击，这种攻击可以很容易地应用于现实世界的对象，以隐藏实际物品或创建不存在的物品，从而导致严重的后果。鉴于当前对抗性补丁攻击和潜在未知威胁的多样性，理想的防御方法应该是有效的、可推广的且鲁棒的，以对抗自适应攻击。在这项工作中，我们介绍了DISPATCH，这是第一个用于对象检测的基于扩散的防御框架。与之前旨在“检测和删除”对抗补丁的作品不同，DISPATCH采用“再生和纠正”策略，利用生成模型来消除攻击效果，同时保留输入图像的完整性。具体来说，我们利用扩散模型的内分布生成能力来重新生成整个图像，使其与良性数据对齐。然后采用纠正过程来识别敌对区域，并用再生的良性区域替换敌对区域。DISPATCH是攻击不可知的，并且不需要了解现有补丁。跨多个检测器和攻击的广泛实验表明，DISPATCH在隐藏攻击和创建攻击方面始终优于最先进的防御，在隐藏攻击方面实现了89.3%的最佳总体mAP.5得分，并将攻击成功率降低至24.8%。针对非目标创建攻击。此外，它还保持了对自适应攻击的强大鲁棒性，使其成为对象检测系统实用且可靠的防御。



## **48. Manipulating Transformer-Based Models: Controllability, Steerability, and Robust Interventions**

操纵基于变压器的模型：可控性、可操纵性和稳健干预 cs.CL

13 pages

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04549v1) [paper-pdf](http://arxiv.org/pdf/2509.04549v1)

**Authors**: Faruk Alpay, Taylan Alpay

**Abstract**: Transformer-based language models excel in NLP tasks, but fine-grained control remains challenging. This paper explores methods for manipulating transformer models through principled interventions at three levels: prompts, activations, and weights. We formalize controllable text generation as an optimization problem addressable via prompt engineering, parameter-efficient fine-tuning, model editing, and reinforcement learning. We introduce a unified framework encompassing prompt-level steering, activation interventions, and weight-space edits. We analyze robustness and safety implications, including adversarial attacks and alignment mitigations. Theoretically, we show minimal weight updates can achieve targeted behavior changes with limited side-effects. Empirically, we demonstrate >90% success in sentiment control and factual edits while preserving base performance, though generalization-specificity trade-offs exist. We discuss ethical dual-use risks and the need for rigorous evaluation. This work lays groundwork for designing controllable and robust language models.

摘要: 基于转换器的语言模型在NLP任务中表现出色，但细粒度控制仍然具有挑战性。本文探讨了通过三个层次的原则性干预来操纵Transformer模型的方法：提示、激活和权重。我们将可控文本生成形式化为一个可通过即时工程、参数高效微调、模型编辑和强化学习来解决的优化问题。我们引入了一个统一的框架，涵盖预算级引导、激活干预和重量空间编辑。我们分析稳健性和安全性影响，包括对抗性攻击和对齐缓解措施。理论上，我们表明最小的体重更新可以实现有针对性的行为改变，副作用有限。从经验上看，我们在情绪控制和事实编辑方面取得了超过90%的成功，同时保留了基本性能，尽管存在概括特定性权衡。我们讨论道德两用风险和严格评估的必要性。这项工作为设计可控且鲁棒的语言模型奠定了基础。



## **49. LADSG: Label-Anonymized Distillation and Similar Gradient Substitution for Label Privacy in Vertical Federated Learning**

LADSG：垂直联邦学习中标签模拟蒸馏和标签隐私的类似梯度替代 cs.CR

20 pages, 8 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2506.06742v3) [paper-pdf](http://arxiv.org/pdf/2506.06742v3)

**Authors**: Zeyu Yan, Yanfei Yao, Xuanbing Wen, Shixiong Zhang, Juli Zhang, Kai Fan

**Abstract**: Vertical Federated Learning (VFL) has emerged as a promising paradigm for collaborative model training across distributed feature spaces, which enables privacy-preserving learning without sharing raw data. However, recent studies have confirmed the feasibility of label inference attacks by internal adversaries. By strategically exploiting gradient vectors and semantic embeddings, attackers-through passive, active, or direct attacks-can accurately reconstruct private labels, leading to catastrophic data leakage. Existing defenses, which typically address isolated leakage vectors or are designed for specific types of attacks, remain vulnerable to emerging hybrid attacks that exploit multiple pathways simultaneously. To bridge this gap, we propose Label-Anonymized Defense with Substitution Gradient (LADSG), a unified and lightweight defense framework for VFL. LADSG first anonymizes true labels via soft distillation to reduce semantic exposure, then generates semantically-aligned substitute gradients to disrupt gradient-based leakage, and finally filters anomalous updates through gradient norm detection. It is scalable and compatible with standard VFL pipelines. Extensive experiments on six real-world datasets show that LADSG reduces the success rates of all three types of label inference attacks by 30-60% with minimal computational overhead, demonstrating its practical effectiveness.

摘要: 垂直联邦学习（VFL）已成为跨分布式特征空间协作模型训练的一种有前途的范式，它可以在无需共享原始数据的情况下实现隐私保护学习。然而，最近的研究证实了内部对手进行标签推断攻击的可行性。通过战略性地利用梯度载体和语义嵌入，攻击者通过被动、主动或直接攻击可以准确地重建私有标签，从而导致灾难性的数据泄露。现有的防御系统通常针对孤立的泄漏载体或专为特定类型的攻击而设计，但仍然容易受到同时利用多个途径的新兴混合攻击的影响。为了弥合这一差距，我们提出了具有替代梯度的标签模拟防御（LADSG），这是一个针对VFL的统一轻量级防御框架。LADSG首先通过软蒸馏匿名化真实标签以减少语义暴露，然后生成语义对齐的替代梯度以破坏基于梯度的泄漏，最后通过梯度范数检测过滤异常更新。它具有可扩展性，并与标准VFL管道兼容。在六个真实数据集上的大量实验表明，LADSG以最小的计算开销将所有三种类型的标签推理攻击的成功率降低了30-60%，证明了其实际有效性。



## **50. Enhancing Gradient Variance and Differential Privacy in Quantum Federated Learning**

增强量子联邦学习中的梯度方差和差异隐私 quant-ph

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.05377v1) [paper-pdf](http://arxiv.org/pdf/2509.05377v1)

**Authors**: Duc-Thien Phan, Minh-Duong Nguyen, Quoc-Viet Pham, Huilong Pi

**Abstract**: Upon integrating Quantum Neural Network (QNN) as the local model, Quantum Federated Learning (QFL) has recently confronted notable challenges. Firstly, exploration is hindered over sharp minima, decreasing learning performance. Secondly, the steady gradient descent results in more stable and predictable model transmissions over wireless channels, making the model more susceptible to attacks from adversarial entities. Additionally, the local QFL model is vulnerable to noise produced by the quantum device's intermediate noise states, since it requires the use of quantum gates and circuits for training. This local noise becomes intertwined with learning parameters during training, impairing model precision and convergence rate. To address these issues, we propose a new QFL technique that incorporates differential privacy and introduces a dedicated noise estimation strategy to quantify and mitigate the impact of intermediate quantum noise. Furthermore, we design an adaptive noise generation scheme to alleviate privacy threats associated with the vanishing gradient variance phenomenon of QNN and enhance robustness against device noise. Experimental results demonstrate that our algorithm effectively balances convergence, reduces communication costs, and mitigates the adverse effects of intermediate quantum noise while maintaining strong privacy protection. Using real-world datasets, we achieved test accuracy of up to 98.47\% for the MNIST dataset and 83.85\% for the CIFAR-10 dataset while maintaining fast execution times.

摘要: 在将量子神经网络（QNN）集成为本地模型后，量子联邦学习（QFL）最近面临着显着的挑战。首先，探索因尖锐的最小值而受到阻碍，从而降低了学习绩效。其次，稳定的梯度下降导致无线通道上的模型传输更加稳定和可预测，使模型更容易受到来自敌对实体的攻击。此外，局部QFL模型容易受到量子设备中间噪音状态产生的噪音的影响，因为它需要使用量子门和电路进行训练。这种局部噪音在训练期间与学习参数交织在一起，损害模型精度和收敛速度。为了解决这些问题，我们提出了一种新的QFL技术，该技术结合了差异隐私并引入了专用的噪音估计策略来量化和减轻中间量子噪音的影响。此外，我们设计了一种自适应噪音生成方案，以减轻与QNN消失的梯度方差现象相关的隐私威胁，并增强对设备噪音的鲁棒性。实验结果表明，我们的算法有效平衡了收敛，降低了通信成本，减轻了中间量子噪音的不利影响，同时保持了强大的隐私保护。使用现实世界的数据集，我们对MNIST数据集实现了高达98.47%的测试准确率，对CIFAR-10数据集实现了高达83.85%的测试准确率，同时保持了快速的执行时间。



