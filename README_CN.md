# Latest Adversarial Attack Papers
**update at 2025-11-10 21:04:06**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. A Metamorphic Testing Perspective on Knowledge Distillation for Language Models of Code: Does the Student Deeply Mimic the Teacher?**

代码语言模型知识提炼的变形测试视角：学生是否深深模仿老师？ cs.SE

The paper is currently under review at a peer-reviewed journal

**SubmitDate**: 2025-11-07    [abs](http://arxiv.org/abs/2511.05476v1) [paper-pdf](http://arxiv.org/pdf/2511.05476v1)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Transformer-based language models of code have achieved state-of-the-art performance across a wide range of software analytics tasks, but their practical deployment remains limited due to high computational costs, slow inference speeds, and significant environmental impact. To address these challenges, recent research has increasingly explored knowledge distillation as a method for compressing a large language model of code (the teacher) into a smaller model (the student) while maintaining performance. However, the degree to which a student model deeply mimics the predictive behavior and internal representations of its teacher remains largely unexplored, as current accuracy-based evaluation provides only a surface-level view of model quality and often fails to capture more profound discrepancies in behavioral fidelity between the teacher and student models. To address this gap, we empirically show that the student model often fails to deeply mimic the teacher model, resulting in up to 285% greater performance drop under adversarial attacks, which is not captured by traditional accuracy-based evaluation. Therefore, we propose MetaCompress, a metamorphic testing framework that systematically evaluates behavioral fidelity by comparing the outputs of teacher and student models under a set of behavior-preserving metamorphic relations. We evaluate MetaCompress on two widely studied tasks, using compressed versions of popular language models of code, obtained via three different knowledge distillation techniques: Compressor, AVATAR, and MORPH. The results show that MetaCompress identifies up to 62% behavioral discrepancies in student models, underscoring the need for behavioral fidelity evaluation within the knowledge distillation pipeline and establishing MetaCompress as a practical framework for testing compressed language models of code derived through knowledge distillation.

摘要: 基于转换器的代码语言模型已在广泛的软件分析任务中实现了最先进的性能，但由于计算成本高、推理速度慢和环境影响严重，它们的实际部署仍然受到限制。为了应对这些挑战，最近的研究越来越多地探索知识蒸馏作为一种将大型代码语言模型（教师）压缩到较小模型（学生）同时保持性能的方法。然而，学生模型深度模仿其教师的预测行为和内部表示的程度在很大程度上仍然没有被探索，因为当前基于准确性的评估只提供了模型质量的表面视图，并且往往无法捕捉到教师和学生模型之间行为忠实度的更深刻差异。为了解决这一差距，我们经验表明，学生模型往往无法深入模仿教师模型，导致对抗性攻击下的性能下降高达285%，而传统的基于准确性的评估无法捕捉到这一点。因此，我们提出了MetaCompress，这是一个变形测试框架，通过在一组保持行为的变形关系下比较教师和学生模型的输出来系统地评估行为忠实度。我们对两项广泛研究的任务进行了评估，使用流行语言代码模型的压缩版本，这些模型通过三种不同的知识提炼技术获得：Compressor、AVATAR和MORPH。结果表明，MetaCompress识别出学生模型中高达62%的行为差异，强调了知识蒸馏管道内行为保真度评估的必要性，并将MetaCompress建立为测试通过知识蒸馏获得的代码的压缩语言模型的实用框架。



## **2. Comparative Study on Noise-Augmented Training and its Effect on Adversarial Robustness in ASR Systems**

ASB系统中噪音增强训练及其对对抗鲁棒性影响的比较研究 eess.AS

**SubmitDate**: 2025-11-07    [abs](http://arxiv.org/abs/2409.01813v4) [paper-pdf](http://arxiv.org/pdf/2409.01813v4)

**Authors**: Karla Pizzi, Matías Pizarro, Asja Fischer

**Abstract**: In this study, we investigate whether noise-augmented training can concurrently improve adversarial robustness in automatic speech recognition (ASR) systems. We conduct a comparative analysis of the adversarial robustness of four different ASR architectures, each trained under three different augmentation conditions: (1) background noise, speed variations, and reverberations; (2) speed variations only; (3) no data augmentation. We then evaluate the robustness of all resulting models against attacks with white-box or black-box adversarial examples. Our results demonstrate that noise augmentation not only enhances model performance on noisy speech but also improves the model's robustness to adversarial attacks.

摘要: 在这项研究中，我们研究了噪音增强训练是否可以同时提高自动语音识别（ASB）系统中的对抗鲁棒性。我们对四种不同ASB架构的对抗鲁棒性进行了比较分析，每种架构都在三种不同的增强条件下训练：（1）背景噪音、速度变化和回响;（2）仅速度变化;（3）没有数据增强。然后，我们评估所有生成的模型针对白盒或黑盒对抗示例的攻击的稳健性。我们的结果表明，噪音增强不仅增强了模型在含噪语音上的性能，而且还提高了模型对对抗攻击的鲁棒性。



## **3. Turning Adversaries into Allies: Reversing Typographic Attacks for Multimodal E-Commerce Product Retrieval**

将对手变成盟友：逆转多模式电子商务产品检索的印刷攻击 cs.LG

**SubmitDate**: 2025-11-07    [abs](http://arxiv.org/abs/2511.05325v1) [paper-pdf](http://arxiv.org/pdf/2511.05325v1)

**Authors**: Janet Jenq, Hongda Shen

**Abstract**: Multimodal product retrieval systems in e-commerce platforms rely on effectively combining visual and textual signals to improve search relevance and user experience. However, vision-language models such as CLIP are vulnerable to typographic attacks, where misleading or irrelevant text embedded in images skews model predictions. In this work, we propose a novel method that reverses the logic of typographic attacks by rendering relevant textual content (e.g., titles, descriptions) directly onto product images to perform vision-text compression, thereby strengthening image-text alignment and boosting multimodal product retrieval performance. We evaluate our method on three vertical-specific e-commerce datasets (sneakers, handbags, and trading cards) using six state-of-the-art vision foundation models. Our experiments demonstrate consistent improvements in unimodal and multimodal retrieval accuracy across categories and model families. Our findings suggest that visually rendering product metadata is a simple yet effective enhancement for zero-shot multimodal retrieval in e-commerce applications.

摘要: 电子商务平台中的多模态产品检索系统依赖于有效地结合视觉和文本信号来提高搜索相关性和用户体验。然而，像CLIP这样的视觉语言模型很容易受到排版攻击，其中图像中嵌入的误导性或不相关的文本会扭曲模型预测。在这项工作中，我们提出了一种新的方法，通过渲染相关的文本内容（例如，标题、描述）直接添加到产品图像上，以执行视觉-文本压缩，从而加强图像-文本对齐并提高多模式产品检索性能。我们使用六个最先进的视觉基础模型在三个垂直特定的电子商务数据集（运动鞋，手袋和交易卡）上评估我们的方法。我们的实验表明，在单峰和多模态检索精度跨类别和模型家族的一致改善。我们的研究结果表明，可视化渲染产品元数据是一个简单而有效的增强零杆多模态检索在电子商务应用程序。



## **4. TAMAS: Benchmarking Adversarial Risks in Multi-Agent LLM Systems**

TAMAS：多代理LLM系统中的对抗风险基准 cs.MA

Accepted at ICML 2025 MAS Workshop. This version includes additional  experiments and analysis

**SubmitDate**: 2025-11-07    [abs](http://arxiv.org/abs/2511.05269v1) [paper-pdf](http://arxiv.org/pdf/2511.05269v1)

**Authors**: Ishan Kavathekar, Hemang Jain, Ameya Rathod, Ponnurangam Kumaraguru, Tanuja Ganu

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities as autonomous agents through tool use, planning, and decision-making abilities, leading to their widespread adoption across diverse tasks. As task complexity grows, multi-agent LLM systems are increasingly used to solve problems collaboratively. However, safety and security of these systems remains largely under-explored. Existing benchmarks and datasets predominantly focus on single-agent settings, failing to capture the unique vulnerabilities of multi-agent dynamics and co-ordination. To address this gap, we introduce $\textbf{T}$hreats and $\textbf{A}$ttacks in $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{S}$ystems ($\textbf{TAMAS}$), a benchmark designed to evaluate the robustness and safety of multi-agent LLM systems. TAMAS includes five distinct scenarios comprising 300 adversarial instances across six attack types and 211 tools, along with 100 harmless tasks. We assess system performance across ten backbone LLMs and three agent interaction configurations from Autogen and CrewAI frameworks, highlighting critical challenges and failure modes in current multi-agent deployments. Furthermore, we introduce Effective Robustness Score (ERS) to assess the tradeoff between safety and task effectiveness of these frameworks. Our findings show that multi-agent systems are highly vulnerable to adversarial attacks, underscoring the urgent need for stronger defenses. TAMAS provides a foundation for systematically studying and improving the safety of multi-agent LLM systems.

摘要: 大型语言模型（LLM）已经通过工具使用，规划和决策能力表现出强大的自主代理能力，导致它们在各种任务中的广泛采用。随着任务复杂性的增长，多智能体LLM系统越来越多地用于协作解决问题。然而，这些系统的安全和安保在很大程度上仍然没有得到充分的探讨。现有的基准和数据集主要集中在单智能体设置，未能捕捉到多智能体动态和协调的独特弱点。为了解决这一差距，我们在$\textBF{M}$ulti-$\textBF{A}$gent $\textBF{S}$ysem（$\textBF{TAMAS}$）中引入了$\textBF{T}$hreats和$\textBF{A}$ttacks，这是一个旨在评估多代理LLM系统稳健性和安全性的基准。TAMAS包括五种不同的场景，包括六种攻击类型的300个对抗实例和211种工具，以及100个无害任务。我们评估了来自Autogen和CrewAI框架的十个主干LLM和三种代理交互配置的系统性能，重点介绍了当前多代理部署中的关键挑战和故障模式。此外，我们还引入了有效稳健性评分（ERS）来评估这些框架的安全性和任务有效性之间的权衡。我们的研究结果表明，多代理系统极易受到对抗攻击，这凸显了对更强防御的迫切需要。TAMAS为系统性研究和改进多代理LLM系统的安全性提供了基础。



## **5. A Secured Intent-Based Networking (sIBN) with Data-Driven Time-Aware Intrusion Detection**

具有数据驱动时间感知入侵检测的安全意图网络（sIBN） cs.CR

This paper is uploaded here for research community, thus it is for  non-commercial purposes

**SubmitDate**: 2025-11-07    [abs](http://arxiv.org/abs/2511.05133v1) [paper-pdf](http://arxiv.org/pdf/2511.05133v1)

**Authors**: Urslla Uchechi Izuazu, Mounir Bensalem, Admela Jukan

**Abstract**: While Intent-Based Networking (IBN) promises operational efficiency through autonomous and abstraction-driven network management, a critical unaddressed issue lies in IBN's implicit trust in the integrity of intent ingested by the network. This inherent assumption of data reliability creates a blind spot exploitable by Man-in-the-Middle (MitM) attacks, where an adversary intercepts and alters intent before it is enacted, compelling the network to orchestrate malicious configurations. This study proposes a secured IBN (sIBN) system with data driven intrusion detection method designed to secure legitimate user intent from adversarial tampering. The proposed intent intrusion detection system uses a ML model applied for network behavioral anomaly detection to reveal temporal patterns of intent tampering. This is achieved by leveraging a set of original behavioral metrics and newly engineered time-aware features, with the model's hyperparameters fine-tuned through the randomized search cross-validation (RSCV) technique. Numerical results based on real-world data sets, show the effectiveness of sIBN, achieving the best performance across standard evaluation metrics, in both binary and multi classification tasks, while maintaining low error rates.

摘要: 虽然基于意图的网络（IBN）通过自主和抽象驱动的网络管理承诺运营效率，但一个未解决的关键问题在于IBN对网络所接受意图的完整性的隐性信任。这种对数据可靠性的固有假设造成了一个可被中间人（MitM）攻击利用的盲点，对手在意图实施之前拦截并改变意图，迫使网络精心策划恶意配置。本研究提出了一种具有数据驱动入侵检测方法的安全IBN（sIBN）系统，旨在保护合法用户意图免受对抗性篡改。提出的意图入侵检测系统使用应用于网络行为异常检测的ML模型来揭示意图篡改的时间模式。这是通过利用一组原始行为指标和新设计的时间感知功能来实现的，并通过随机搜索交叉验证（RSCV）技术对模型的超参数进行微调。基于现实世界数据集的数值结果显示了sIBN的有效性，在二进制和多分类任务中实现了标准评估指标的最佳性能，同时保持低错误率。



## **6. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

迭代自调优LLM以增强越狱能力 cs.CL

Accepted to NAACL 2025 Main (Oral)

**SubmitDate**: 2025-11-07    [abs](http://arxiv.org/abs/2410.18469v6) [paper-pdf](http://arxiv.org/pdf/2410.18469v6)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM

摘要: 最近的研究表明，大型语言模型（LLM）很容易受到自动越狱攻击，其中由附加到有害查询的算法精心设计的对抗性后缀绕过了安全对齐并触发意外响应。当前生成这些后缀的方法计算成本高，攻击成功率（ASB）较低，尤其是针对Llama 2和Llama 3等对齐良好的模型。为了克服这些限制，我们引入了ADV-LLM，这是一种迭代自调优过程，可以制作具有增强越狱能力的对抗性LLM。我们的框架显着降低了生成对抗性后缀的计算成本，同时在各种开源LLM上实现了近100%的ASB。此外，尽管仅在Llama 3上进行了优化，但它仍表现出对闭源模型的强大攻击转移性，在GPT-3.5上实现了99%的ASB，在GPT-4上实现了49%的ASB。除了提高越狱能力之外，ADV-LLM还通过其生成用于研究LLM安全性的大型数据集的能力，为未来的安全一致研究提供了宝贵的见解。我们的代码可访问：https://github.com/SunChungEn/ADV-LLM



## **7. A Zeroth-order Resilient Algorithm for Distributed Online Optimization against Byzantine Edge Attacks**

一种针对拜占庭边缘攻击的分布式在线优化零阶弹性算法 math.OC

**SubmitDate**: 2025-11-07    [abs](http://arxiv.org/abs/2511.05104v1) [paper-pdf](http://arxiv.org/pdf/2511.05104v1)

**Authors**: Yuhang Liu, Wenjun Mei

**Abstract**: In this paper, we propose a zeroth-order resilient distributed online algorithm for networks under Byzantine edge attacks. We assume that both the edges attacked by Byzantine adversaries and the objective function are time-varying. Moreover, we focus on the scenario where the complete time-varying objective function cannot be observed, and only its value at a certain point is available. Using deterministic difference, we design a zeroth-order distributed online optimization algorithm against Byzantine edge attacks and provide an upper bound on the dynamic regret of the algorithm. Finally, a simulation example is given justifying the theoretical results.

摘要: 本文针对拜占庭边缘攻击下的网络提出了一种零阶弹性分布式在线算法。我们假设拜占庭对手攻击的边缘和目标函数都是时变的。此外，我们关注的是无法观察到完整的时变目标函数，并且只有其在某个点的值可用的场景。利用确定性差异，设计了一种针对拜占庭边缘攻击的零阶分布式在线优化算法，并给出了算法动态后悔的上界。最后，给出了一个仿真算例，验证了理论结果。



## **8. Quantifying the Risk of Transferred Black Box Attacks**

量化转移黑匣子攻击的风险 cs.CR

**SubmitDate**: 2025-11-07    [abs](http://arxiv.org/abs/2511.05102v1) [paper-pdf](http://arxiv.org/pdf/2511.05102v1)

**Authors**: Disesdi Susanna Cox, Niklas Bunzel

**Abstract**: Neural networks have become pervasive across various applications, including security-related products. However, their widespread adoption has heightened concerns regarding vulnerability to adversarial attacks. With emerging regulations and standards emphasizing security, organizations must reliably quantify risks associated with these attacks, particularly regarding transferred adversarial attacks, which remain challenging to evaluate accurately. This paper investigates the complexities involved in resilience testing against transferred adversarial attacks. Our analysis specifically addresses black-box evasion attacks, highlighting transfer-based attacks due to their practical significance and typically high transferability between neural network models. We underline the computational infeasibility of exhaustively exploring high-dimensional input spaces to achieve complete test coverage. As a result, comprehensive adversarial risk mapping is deemed impractical. To mitigate this limitation, we propose a targeted resilience testing framework that employs surrogate models strategically selected based on Centered Kernel Alignment (CKA) similarity. By leveraging surrogate models exhibiting both high and low CKA similarities relative to the target model, the proposed approach seeks to optimize coverage of adversarial subspaces. Risk estimation is conducted using regression-based estimators, providing organizations with realistic and actionable risk quantification.

摘要: 神经网络已在各种应用程序中普及，包括安全相关产品。然而，它们的广泛采用加剧了人们对容易受到对抗攻击的担忧。随着新兴的法规和标准强调安全性，组织必须可靠地量化与这些攻击相关的风险，特别是关于转移的对抗性攻击，准确评估这些风险仍然具有挑战性。本文研究了针对转移对抗攻击的弹性测试所涉及的复杂性。我们的分析专门针对黑匣子规避攻击，重点介绍了基于传输的攻击，因为它们具有实际意义，而且神经网络模型之间通常具有很高的可移植性。我们强调彻底探索多维输入空间以实现完整的测试覆盖在计算上是不可行的。因此，全面的对抗风险绘图被认为是不切实际的。为了减轻这一限制，我们提出了一个有针对性的弹性测试框架，该框架采用基于中心核心对齐（CKA）相似性策略选择的代理模型。通过利用相对于目标模型表现出高和低CKA相似性的代理模型，所提出的方法寻求优化对抗子空间的覆盖。使用基于回归的估计器进行风险估计，为组织提供现实和可操作的风险量化。



## **9. CompressionAttack: Exploiting Prompt Compression as a New Attack Surface in LLM-Powered Agents**

CompressionAttack：利用即时压缩作为LLM支持的代理中的新攻击表面 cs.CR

**SubmitDate**: 2025-11-07    [abs](http://arxiv.org/abs/2510.22963v2) [paper-pdf](http://arxiv.org/pdf/2510.22963v2)

**Authors**: Zesen Liu, Zhixiang Zhang, Yuchong Xie, Dongdong She

**Abstract**: LLM-powered agents often use prompt compression to reduce inference costs, but this introduces a new security risk. Compression modules, which are optimized for efficiency rather than safety, can be manipulated by adversarial inputs, causing semantic drift and altering LLM behavior. This work identifies prompt compression as a novel attack surface and presents CompressionAttack, the first framework to exploit it. CompressionAttack includes two strategies: HardCom, which uses discrete adversarial edits for hard compression, and SoftCom, which performs latent-space perturbations for soft compression. Experiments on multiple LLMs show up to 80% attack success and 98% preference flips, while remaining highly stealthy and transferable. Case studies in VSCode Cline and Ollama confirm real-world impact, and current defenses prove ineffective, highlighting the need for stronger protections.

摘要: LLM支持的代理通常使用即时压缩来降低推理成本，但这会带来新的安全风险。压缩模块针对效率而不是安全性进行了优化，可以通过对抗输入来操纵，从而导致语义漂移并改变LLM行为。这项工作将即时压缩确定为一种新型攻击表面，并提出了第一个利用它的框架CompressionAttack。CompressionAttack包括两种策略：HardCom，使用离散对抗编辑进行硬压缩，以及SoftCom，为软压缩执行潜伏空间扰动。对多个LLM的实验显示，攻击成功率高达80%，偏好翻转率高达98%，同时保持高度隐蔽性和可转移性。VSCode Cline和Olama的案例研究证实了现实世界的影响，而当前的防御措施被证明无效，凸显了加强保护的必要性。



## **10. Deep learning models are vulnerable, but adversarial examples are even more vulnerable**

深度学习模型很脆弱，但对抗性示例更脆弱 cs.CV

25 pages,12 figures

**SubmitDate**: 2025-11-07    [abs](http://arxiv.org/abs/2511.05073v1) [paper-pdf](http://arxiv.org/pdf/2511.05073v1)

**Authors**: Jun Li, Yanwei Xu, Keran Li, Xiaoli Zhang

**Abstract**: Understanding intrinsic differences between adversarial examples and clean samples is key to enhancing DNN robustness and detection against adversarial attacks. This study first empirically finds that image-based adversarial examples are notably sensitive to occlusion. Controlled experiments on CIFAR-10 used nine canonical attacks (e.g., FGSM, PGD) to generate adversarial examples, paired with original samples for evaluation. We introduce Sliding Mask Confidence Entropy (SMCE) to quantify model confidence fluctuation under occlusion. Using 1800+ test images, SMCE calculations supported by Mask Entropy Field Maps and statistical distributions show adversarial examples have significantly higher confidence volatility under occlusion than originals. Based on this, we propose Sliding Window Mask-based Adversarial Example Detection (SWM-AED), which avoids catastrophic overfitting of conventional adversarial training. Evaluations across classifiers and attacks on CIFAR-10 demonstrate robust performance, with accuracy over 62% in most cases and up to 96.5%.

摘要: 了解对抗性示例和干净样本之间的内在差异是增强DNN稳健性和检测对抗性攻击的关键。这项研究首先从经验上发现，基于图像的对抗示例对遮挡特别敏感。CIFAR-10的对照实验使用了九种典型攻击（例如，FGSM、PVD）生成对抗性示例，并与原始样本配对进行评估。我们引入滑动掩模置信熵（SMCE）来量化模型在遮挡情况下的置信度波动。使用1800多张测试图像，由Mask Entropy Field Maps和统计分布支持的SMCE计算显示，对抗性示例在遮挡下的置信度波动性明显高于原始示例。在此基础上，我们提出了基于滑动窗口掩码的对抗性样本检测（SWM-AED），它避免了传统对抗性训练的灾难性过拟合。对CIFAR-10的分类器和攻击的评估显示出强大的性能，在大多数情况下准确率超过62%，最高可达96.5%。



## **11. A Comprehensive Survey of Website Fingerprinting Attacks and Defenses in Tor: Advances and Open Challenges**

Tor网站指纹攻击和防御的全面调查：进展和开放挑战 cs.CR

43 pages

**SubmitDate**: 2025-11-07    [abs](http://arxiv.org/abs/2510.11804v2) [paper-pdf](http://arxiv.org/pdf/2510.11804v2)

**Authors**: Yuwen Cui, Guangjing Wang, Khanh Vu, Kai Wei, Kehan Shen, Zhengyuan Jiang, Xiao Han, Ning Wang, Zhuo Lu, Yao Liu

**Abstract**: The Tor network provides users with strong anonymity by routing their internet traffic through multiple relays. While Tor encrypts traffic and hides IP addresses, it remains vulnerable to traffic analysis attacks such as the website fingerprinting (WF) attack, achieving increasingly high fingerprinting accuracy even under open-world conditions. In response, researchers have proposed a variety of defenses, ranging from adaptive padding, traffic regularization, and traffic morphing to adversarial perturbation, that seek to obfuscate or reshape traffic traces. However, these defenses often entail trade-offs between privacy, usability, and system performance. Despite extensive research, a comprehensive survey unifying WF datasets, attack methodologies, and defense strategies remains absent. This paper fills that gap by systematically categorizing existing WF research into three key domains: datasets, attack models, and defense mechanisms. We provide an in-depth comparative analysis of techniques, highlight their strengths and limitations under diverse threat models, and discuss emerging challenges such as multi-tab browsing and coarse-grained traffic features. By consolidating prior work and identifying open research directions, this survey serves as a foundation for advancing stronger privacy protection in Tor.

摘要: Tor网络通过多个中继路由用户的互联网流量，为用户提供了强大的匿名性。虽然Tor加密流量并隐藏IP地址，但它仍然容易受到网站指纹识别（WF）攻击等流量分析攻击，即使在开放世界条件下也能实现越来越高的指纹识别准确性。作为回应，研究人员提出了各种防御措施，从自适应填充、流量规则化、流量变形到对抗性扰动，旨在模糊或重塑流量轨迹。然而，这些防御通常需要在隐私、可用性和系统性能之间进行权衡。尽管进行了广泛的研究，但仍然缺乏统一WF数据集、攻击方法和防御策略的全面调查。本文通过将现有的WF研究系统地分类为三个关键领域：数据集、攻击模型和防御机制来填补这一空白。我们对技术进行深入的比较分析，强调它们在不同威胁模型下的优势和局限性，并讨论多选项卡浏览和粗粒度流量功能等新出现的挑战。通过整合之前的工作并确定开放的研究方向，这项调查为在Tor中推进更强有力的隐私保护奠定了基础。



## **12. DeepForgeSeal: Latent Space-Driven Semi-Fragile Watermarking for Deepfake Detection Using Multi-Agent Adversarial Reinforcement Learning**

DeepForgeSeal：使用多智能体对抗强化学习的用于Deepfake检测的潜在空间驱动半脆弱水印 cs.CV

**SubmitDate**: 2025-11-07    [abs](http://arxiv.org/abs/2511.04949v1) [paper-pdf](http://arxiv.org/pdf/2511.04949v1)

**Authors**: Tharindu Fernando, Clinton Fookes, Sridha Sridharan

**Abstract**: Rapid advances in generative AI have led to increasingly realistic deepfakes, posing growing challenges for law enforcement and public trust. Existing passive deepfake detectors struggle to keep pace, largely due to their dependence on specific forgery artifacts, which limits their ability to generalize to new deepfake types. Proactive deepfake detection using watermarks has emerged to address the challenge of identifying high-quality synthetic media. However, these methods often struggle to balance robustness against benign distortions with sensitivity to malicious tampering. This paper introduces a novel deep learning framework that harnesses high-dimensional latent space representations and the Multi-Agent Adversarial Reinforcement Learning (MAARL) paradigm to develop a robust and adaptive watermarking approach. Specifically, we develop a learnable watermark embedder that operates in the latent space, capturing high-level image semantics, while offering precise control over message encoding and extraction. The MAARL paradigm empowers the learnable watermarking agent to pursue an optimal balance between robustness and fragility by interacting with a dynamic curriculum of benign and malicious image manipulations simulated by an adversarial attacker agent. Comprehensive evaluations on the CelebA and CelebA-HQ benchmarks reveal that our method consistently outperforms state-of-the-art approaches, achieving improvements of over 4.5% on CelebA and more than 5.3% on CelebA-HQ under challenging manipulation scenarios.

摘要: 生成人工智能的快速发展导致深度造假变得越来越真实，给执法和公众信任带来了越来越大的挑战。现有的被动Deepfake检测器很难跟上步伐，这主要是由于它们依赖于特定的伪造文物，这限制了它们推广到新的Deepfake类型的能力。使用水印的主动深度伪造检测已经出现，以应对识别高质量合成媒体的挑战。然而，这些方法常常难以平衡针对良性失真的稳健性与对恶意篡改的敏感性。本文介绍了一种新型的深度学习框架，该框架利用多维潜在空间表示和多智能体对抗强化学习（MAARL）范式来开发一种稳健且自适应的水印方法。具体来说，我们开发了一个可学习的水印嵌入器，它在潜在空间中运行，捕获高级图像语义，同时提供对消息编码和提取的精确控制。MAARL范式使可学习的水印代理能够通过与对抗性攻击代理模拟的良性和恶意图像操纵的动态课程交互来追求稳健性和脆弱性之间的最佳平衡。对CelebA和CelebA-HQ基准的综合评估表明，我们的方法始终优于最先进的方法，在具有挑战性的操纵场景下，在CelebA上实现了超过4.5%的改进，在CelebA-HQ上实现了超过5.3%的改进。



## **13. Bit-Flipping Attack Exploration and Countermeasure in 5G Network**

5G网络中的位翻转攻击探索与对策 cs.CR

Presented at the IEEE MASS 2025 REUNS Workshop

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04882v1) [paper-pdf](http://arxiv.org/pdf/2511.04882v1)

**Authors**: Joon Kim, Chengwei Duan, Sandip Ray

**Abstract**: 5G communication technology has become a vital component in a wide range of applications due to its unique advantages such as high data rate and low latency. While much of the existing research has focused on optimizing its efficiency and performance, security considerations have not received comparable attention, potentially leaving critical vulnerabilities unexplored. In this work, we investigate the vulnerability of 5G systems to bit-flipping attacks, which is an integrity attack where an adversary intercepts 5G network traffic and modifies specific fields of an encrypted message without decryption, thus mutating the message while remaining valid to the receiver. Notably, these attacks do not require the attacker to know the plaintext, and only the semantic meaning or position of certain fields would be enough to effect targeted modifications. We conduct our analysis on OpenAirInterface (OAI), an open-source 5G platform that follows the 3GPP Technical Specifications, to rigorously test the real-world feasibility and impact of bit-flipping attacks under current 5G encryption mechanisms. Finally, we propose a keystream-based shuffling defense mechanism to mitigate the effect of such attacks by raising the difficulty of manipulating specific encrypted fields, while introducing no additional communication overhead compared to the NAS Integrity Algorithm (NIA) in 5G. Our findings reveal that enhancements to 5G security are needed to better protect against attacks that alter data during transmission at the network level.

摘要: 5G通信技术因其高数据率、低延迟等独特优势，已成为广泛应用中的重要组成部分。虽然现有的大部分研究都集中在优化其效率和性能上，但安全考虑并没有得到同等的关注，这可能会导致关键漏洞未被探索。在这项工作中，我们研究了5G系统对比特翻转攻击的脆弱性，这是一种完整性攻击，对手拦截5G网络流量并在不解密的情况下修改加密消息的特定字段，从而变异消息，同时对接收者保持有效。值得注意的是，这些攻击并不要求攻击者知道明文，只有某些字段的语义含义或位置才足以实现有针对性的修改。我们对遵循3GPP技术规范的开源5G平台OpenAirConnection（OAI）进行分析，以严格测试当前5G加密机制下比特翻转攻击的现实可行性和影响。最后，我们提出了一种基于密钥流的洗牌防御机制，通过提高操纵特定加密字段的难度来减轻此类攻击的影响，同时与5G中的NAS完整性算法（NIA）相比不会引入额外的通信负担。我们的研究结果表明，需要增强5G安全性，以更好地防止在网络级别传输期间改变数据的攻击。



## **14. Exploiting Data Structures for Bypassing and Crashing Anti-Malware Solutions via Telemetry Complexity Attacks**

利用数据结构通过远程通信复杂性攻击来骚扰和崩溃反恶意软件解决方案 cs.CR

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04472v1) [paper-pdf](http://arxiv.org/pdf/2511.04472v1)

**Authors**: Evgenios Gkritsis, Constantinos Patsakis, George Stergiopoulos

**Abstract**: Anti-malware systems rely on sandboxes, hooks, and telemetry pipelines, including collection agents, serializers, and database backends, to monitor program and system behavior. We show that these data-handling components constitute an exploitable attack surface that can lead to denial-of-analysis (DoA) states without disabling sensors or requiring elevated privileges. As a result, we present \textit{Telemetry Complexity Attacks} (TCAs), a new class of vulnerabilities that exploit fundamental mismatches between unbounded collection mechanisms and bounded processing capabilities. Our method recursively spawns child processes to generate specially crafted, deeply nested, and oversized telemetry that stresses serialization and storage boundaries, as well as visualization layers, for example, JSON/BSON depth and size limits. Depending on the product, this leads to truncated or missing behavioral reports, rejected database inserts, serializer recursion and size errors, and unresponsive dashboards. In all of these cases, malicious activity is normally executed; however, depending on the examined solution, it is not recorded and/or not presented to the analysts. Therefore, instead of evading sensors, we break the pipeline that stores the data captured by the sensors.   We evaluate our technique against twelve commercial and open-source malware analysis platforms and endpoint detection and response (EDR) solutions. Seven products fail in different stages of the telemetry pipeline; two vendors assigned CVE identifiers (CVE-2025-61301 and CVE-2025-61303), and others issued patches or configuration changes. We discuss root causes and propose mitigation strategies to prevent DoA attacks triggered by adversarial telemetry.

摘要: 反恶意软件系统依赖沙箱、挂钩和遥感管道（包括收集代理、序列化器和数据库后台）来监控程序和系统行为。我们表明，这些数据处理组件构成了一个可利用的攻击表面，可以在不禁用传感器或要求更高特权的情况下导致拒绝分析（DoA）状态。因此，我们提出了\texttit {Telemarity Complexity Attacks}（TPA），这是一类新型漏洞，利用无界收集机制和有界处理能力之间的根本不匹配。我们的方法以迭代方式产生子进程，以生成特制的、深度嵌套的和超大的遥感数据，其中强调序列化和存储边界以及可视化层，例如，SON/BSON深度和大小限制。根据产品的不同，这会导致行为报告被截断或缺失、数据库插入被拒绝、序列化器回归和大小错误以及仪表板无响应。在所有这些情况下，恶意活动通常会被执行;但是，根据所检查的解决方案，它不会被记录和/或不会呈现给分析师。因此，我们不是逃避传感器，而是打破了存储传感器捕获数据的管道。   我们针对十二个商业和开源恶意软件分析平台以及端点检测和响应（EDR）解决方案评估我们的技术。七个产品在遥测管道的不同阶段失败;两个供应商分配了CVE标识符（CVE-2025-61301和CVE-2025-61303），其他供应商发布了补丁或配置更改。我们讨论了根本原因，并提出了缓解策略，以防止由对抗遥测触发的DoA攻击。



## **15. Adversarially Robust and Interpretable Magecart Malware Detection**

对抗性强且可解释的Magecart恶意软件检测 cs.CR

5 pages, 2 figures

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04440v1) [paper-pdf](http://arxiv.org/pdf/2511.04440v1)

**Authors**: Pedro Pereira, José Gouveia, João Vitorino, Eva Maia, Isabel Praça

**Abstract**: Magecart skimming attacks have emerged as a significant threat to client-side security and user trust in online payment systems. This paper addresses the challenge of achieving robust and explainable detection of Magecart attacks through a comparative study of various Machine Learning (ML) models with a real-world dataset. Tree-based, linear, and kernel-based models were applied, further enhanced through hyperparameter tuning and feature selection, to distinguish between benign and malicious scripts. Such models are supported by a Behavior Deterministic Finite Automaton (DFA) which captures structural behavior patterns in scripts, helping to analyze and classify client-side script execution logs. To ensure robustness against adversarial evasion attacks, the ML models were adversarially trained and evaluated using attacks from the Adversarial Robustness Toolbox and the Adaptative Perturbation Pattern Method. In addition, concise explanations of ML model decisions are provided, supporting transparency and user trust. Experimental validation demonstrated high detection performance and interpretable reasoning, demonstrating that traditional ML models can be effective in real-world web security contexts.

摘要: Magecart略读攻击已成为对客户端安全和在线支付系统用户信任的重大威胁。本文通过对各种机器学习（ML）模型与现实世界数据集进行比较研究，解决了实现对Magecart攻击的稳健且可解释的检测的挑战。应用了基于树、线性和基于核的模型，并通过超参数调整和特征选择进一步增强，以区分良性和恶意脚本。此类模型由行为确定性有限自动机（PFA）支持，该自动机捕获脚本中的结构行为模式，帮助分析和分类客户端脚本执行日志。为了确保针对对抗性规避攻击的鲁棒性，ML模型使用来自对抗性鲁棒性搜索器和自适应微扰模式方法的攻击进行了对抗性训练和评估。此外，还提供了ML模型决策的简洁解释，支持透明度和用户信任。实验验证证明了高检测性能和可解释推理，证明传统ML模型在现实世界的网络安全环境中可以有效。



## **16. AdversariaLLM: A Unified and Modular Toolbox for LLM Robustness Research**

AdversariaLLM：用于LLM稳健性研究的统一模块化工作空间 cs.AI

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04316v1) [paper-pdf](http://arxiv.org/pdf/2511.04316v1)

**Authors**: Tim Beyer, Jonas Dornbusch, Jakob Steimle, Moritz Ladenburger, Leo Schwinn, Stephan Günnemann

**Abstract**: The rapid expansion of research on Large Language Model (LLM) safety and robustness has produced a fragmented and oftentimes buggy ecosystem of implementations, datasets, and evaluation methods. This fragmentation makes reproducibility and comparability across studies challenging, hindering meaningful progress. To address these issues, we introduce AdversariaLLM, a toolbox for conducting LLM jailbreak robustness research. Its design centers on reproducibility, correctness, and extensibility. The framework implements twelve adversarial attack algorithms, integrates seven benchmark datasets spanning harmfulness, over-refusal, and utility evaluation, and provides access to a wide range of open-weight LLMs via Hugging Face. The implementation includes advanced features for comparability and reproducibility such as compute-resource tracking, deterministic results, and distributional evaluation techniques. \name also integrates judging through the companion package JudgeZoo, which can also be used independently. Together, these components aim to establish a robust foundation for transparent, comparable, and reproducible research in LLM safety.

摘要: 对大型语言模型（LLM）安全性和稳健性研究的迅速扩展，产生了一个由实现、数据集和评估方法组成的碎片化且经常存在缺陷的生态系统。这种碎片化使得研究之间的重复性和可比性具有挑战性，阻碍了有意义的进展。为了解决这些问题，我们引入了AdversariaLLM，这是一个用于进行LLM越狱稳健性研究的工具箱。其设计以可重复性、正确性和可扩展性为中心。该框架实现了十二种对抗攻击算法，集成了涵盖危害性、过度拒绝和效用评估的七个基准数据集，并通过Hugging Face提供了对广泛开放权重LLM的访问权限。该实现包括可比性和可重复性的高级功能，例如计算资源跟踪、确定性结果和分布式评估技术。\Name还通过配套包JudgeZoo集成了判断，该包也可以独立使用。这些组成部分的目标是为LLM安全性的透明、可比和可重复的研究奠定坚实的基础。



## **17. GASP: Efficient Black-Box Generation of Adversarial Suffixes for Jailbreaking LLMs**

GASP：用于越狱LLM的对抗后缀的高效黑盒生成 cs.LG

Accepted to NeurIPS 2025. Project page and demos:  https://air-ml.org/project/gasp/

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2411.14133v3) [paper-pdf](http://arxiv.org/pdf/2411.14133v3)

**Authors**: Advik Raj Basani, Xiao Zhang

**Abstract**: LLMs have shown impressive capabilities across various natural language processing tasks, yet remain vulnerable to input prompts, known as jailbreak attacks, carefully designed to bypass safety guardrails and elicit harmful responses. Traditional methods rely on manual heuristics but suffer from limited generalizability. Despite being automatic, optimization-based attacks often produce unnatural prompts that can be easily detected by safety filters or require high computational costs due to discrete token optimization. In this paper, we introduce Generative Adversarial Suffix Prompter (GASP), a novel automated framework that can efficiently generate human-readable jailbreak prompts in a fully black-box setting. In particular, GASP leverages latent Bayesian optimization to craft adversarial suffixes by efficiently exploring continuous latent embedding spaces, gradually optimizing the suffix prompter to improve attack efficacy while balancing prompt coherence via a targeted iterative refinement procedure. Through comprehensive experiments, we show that GASP can produce natural adversarial prompts, significantly improving jailbreak success over baselines, reducing training times, and accelerating inference speed, thus making it an efficient and scalable solution for red-teaming LLMs.

摘要: LLM在各种自然语言处理任务中表现出了令人印象深刻的能力，但仍然容易受到输入提示（即越狱攻击）的影响，这些提示经过精心设计，旨在绕过安全护栏并引发有害反应。传统方法依赖于手工启发式，但通用性有限。尽管是自动的，但基于优化的攻击通常会产生不自然的提示，安全过滤器可以轻松检测到这些提示，或者由于离散令牌优化而需要很高的计算成本。在本文中，我们介绍了生成式对抗后缀搜索器（GISP），这是一种新型自动化框架，可以在完全黑匣子的环境中有效地生成人类可读的越狱提示。特别是，GSP利用潜在的Bayesian优化来制作对抗性后缀，方法是有效地探索连续的潜在嵌入空间，逐步优化后缀插入器以提高攻击效率，同时通过有针对性的迭代细化过程平衡即时的一致性。通过全面的实验，我们表明GISP可以产生自然的对抗提示，在基线范围内显着提高越狱成功率，减少训练时间，加快推理速度，从而使其成为红色团队LLM的高效且可扩展的解决方案。



## **18. SynFuzz: Leveraging Fuzzing of Netlist to Detect Synthesis Bugs**

SynFuzz：利用网表的模糊化来检测合成错误 cs.CR

15 pages, 10 figures, 5 tables

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2504.18812v3) [paper-pdf](http://arxiv.org/pdf/2504.18812v3)

**Authors**: Raghul Saravanan, Sudipta Paria, Aritra Dasgupta, Venkat Nitin Patnala, Swarup Bhunia, Sai Manoj P D

**Abstract**: In the evolving landscape of integrated circuit (IC) design, the increasing complexity of modern processors and intellectual property (IP) cores has introduced new challenges in ensuring design correctness and security. The recent advancements in hardware fuzzing techniques have shown their efficacy in detecting hardware bugs and vulnerabilities at the RTL abstraction level of hardware. However, they suffer from several limitations, including an inability to address vulnerabilities introduced during synthesis and gate-level transformations. These methods often fail to detect issues arising from library adversaries, where compromised or malicious library components can introduce backdoors or unintended behaviors into the design. In this paper, we present a novel hardware fuzzer, SynFuzz, designed to overcome the limitations of existing hardware fuzzing frameworks. SynFuzz focuses on fuzzing hardware at the gate-level netlist to identify synthesis bugs and vulnerabilities that arise during the transition from RTL to the gate-level. We analyze the intrinsic hardware behaviors using coverage metrics specifically tailored for the gate-level. Furthermore, SynFuzz implements differential fuzzing to uncover bugs associated with EDA libraries. We evaluated SynFuzz on popular open-source processors and IP designs, successfully identifying 7 new synthesis bugs. Additionally, by exploiting the optimization settings of EDA tools, we performed a compromised library mapping attack (CLiMA), creating a malicious version of hardware designs that remains undetectable by traditional verification methods. We also demonstrate how SynFuzz overcomes the limitations of the industry-standard formal verification tool, Cadence Conformal, providing a more robust and comprehensive approach to hardware verification.

摘要: 在集成电路（IC）设计不断发展的格局中，现代处理器和知识产权（IP）核的复杂性日益增加，为确保设计正确性和安全性带来了新的挑战。硬件模糊技术的最新进展表明，它们在硬件RTL抽象级别检测硬件错误和漏洞方面的功效。然而，它们存在一些限制，包括无法解决合成和门级转换期间引入的漏洞。这些方法通常无法检测到库对手引起的问题，其中受损害或恶意的库组件可能会在设计中引入后门或意外行为。在本文中，我们提出了一种新型的硬件模糊器SynFuzz，旨在克服现有硬件模糊框架的局限性。SynFuzz专注于在门级网表上模糊硬件，以识别从RTL过渡到门级过程中出现的合成错误和漏洞。我们使用专门为门户级定制的覆盖指标来分析固有的硬件行为。此外，SynFuzz还实现了差异模糊化来发现与EDA库相关的错误。我们在流行的开源处理器和IP设计上评估了SynFuzz，成功识别出7个新的合成错误。此外，通过利用EDA工具的优化设置，我们执行了一个妥协的库映射攻击（CLiMA），创建了一个恶意版本的硬件设计，仍然无法检测到传统的验证方法。我们还演示了SynFuzz如何克服行业标准的正式验证工具Cadence Conformal的局限性，为硬件验证提供更强大、更全面的方法。



## **19. Measuring the Security of Mobile LLM Agents under Adversarial Prompts from Untrusted Third-Party Channels**

在来自不受信任的第三方渠道的对抗性承诺下衡量移动LLM代理的安全性 cs.CR

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2510.27140v2) [paper-pdf](http://arxiv.org/pdf/2510.27140v2)

**Authors**: Chenghao Du, Quanfeng Huang, Tingxuan Tang, Zihao Wang, Adwait Nadkarni, Yue Xiao

**Abstract**: Large Language Models (LLMs) have transformed software development, enabling AI-powered applications known as LLM-based agents that promise to automate tasks across diverse apps and workflows. Yet, the security implications of deploying such agents in adversarial mobile environments remain poorly understood. In this paper, we present the first systematic study of security risks in mobile LLM agents. We design and evaluate a suite of adversarial case studies, ranging from opportunistic manipulations such as pop-up advertisements to advanced, end-to-end workflows involving malware installation and cross-app data exfiltration. Our evaluation covers eight state-of-the-art mobile agents across three architectures, with over 2,000 adversarial and paired benign trials. The results reveal systemic vulnerabilities: low-barrier vectors such as fraudulent ads succeed with over 80% reliability, while even workflows requiring the circumvention of operating-system warnings, such as malware installation, are consistently completed by advanced multi-app agents. By mapping these attacks to the MITRE ATT&CK Mobile framework, we uncover novel privilege-escalation and persistence pathways unique to LLM-driven automation. Collectively, our findings provide the first end-to-end evidence that mobile LLM agents are exploitable in realistic adversarial settings, where untrusted third-party channels (e.g., ads, embedded webviews, cross-app notifications) are an inherent part of the mobile ecosystem.

摘要: 大型语言模型（LLM）已经改变了软件开发，使AI驱动的应用程序成为基于LLM的代理，这些代理承诺在不同的应用程序和工作流中自动执行任务。然而，在对抗性移动环境中部署此类代理的安全影响仍然知之甚少。在本文中，我们提出了第一个系统的研究，在移动LLM代理的安全风险。我们设计和评估了一系列对抗性案例研究，从弹出式广告等机会主义操纵到涉及恶意软件安装和跨应用程序数据泄露的高级端到端工作流。我们的评估涵盖了三种架构中的八个最先进的移动代理，以及超过2，000项对抗性和配对良性试验。结果揭示了系统性漏洞：欺诈广告等低障碍载体成功，可靠性超过80%，而即使是需要规避操作系统警告的工作流程，例如恶意软件安装，也始终由高级多应用程序代理完成。通过将这些攻击映射到MITRE ATT & CK Mobile框架，我们发现了LLM驱动的自动化所独有的新型风险升级和持久性途径。总的来说，我们的研究结果提供了第一个端到端的证据，证明移动LLM代理在现实的对抗环境中是可利用的，其中不受信任的第三方渠道（例如，广告、嵌入式网络视图、跨应用程序通知）是移动生态系统的固有组成部分。



## **20. VERA: Variational Inference Framework for Jailbreaking Large Language Models**

VERA：越狱大型语言模型的变分推理框架 cs.CR

Accepted by NeurIPS 2025

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2506.22666v2) [paper-pdf](http://arxiv.org/pdf/2506.22666v2)

**Authors**: Anamika Lochab, Lu Yan, Patrick Pynadath, Xiangyu Zhang, Ruqi Zhang

**Abstract**: The rise of API-only access to state-of-the-art LLMs highlights the need for effective black-box jailbreak methods to identify model vulnerabilities in real-world settings. Without a principled objective for gradient-based optimization, most existing approaches rely on genetic algorithms, which are limited by their initialization and dependence on manually curated prompt pools. Furthermore, these methods require individual optimization for each prompt, failing to provide a comprehensive characterization of model vulnerabilities. To address this gap, we introduce VERA: Variational infErence fRamework for jAilbreaking. VERA casts black-box jailbreak prompting as a variational inference problem, training a small attacker LLM to approximate the target LLM's posterior over adversarial prompts. Once trained, the attacker can generate diverse, fluent jailbreak prompts for a target query without re-optimization. Experimental results show that VERA achieves strong performance across a range of target LLMs, highlighting the value of probabilistic inference for adversarial prompt generation.

摘要: 仅限API访问最先进的LLM的兴起凸显了有效的黑匣子越狱方法来识别现实世界环境中的模型漏洞的必要性。由于没有基于梯度的优化的原则目标，大多数现有方法都依赖于遗传算法，而遗传算法受到初始化和对手动策划提示池的依赖的限制。此外，这些方法需要对每个提示进行单独优化，无法提供模型漏洞的全面描述。为了解决这个差距，我们引入了VERA：变分影响Erence fRamework for jAilbreaking。VERA将黑匣子越狱提示视为变分推理问题，训练小型攻击者LLM在对抗性提示上逼近目标LLM的后验。经过训练后，攻击者可以为目标查询生成多样化、流畅的越狱提示，而无需重新优化。实验结果表明，VERA在一系列目标LLM中实现了强劲的性能，凸显了概率推理对对抗提示生成的价值。



## **21. QSAFE-V: Quantum-Enhanced Lightweight Authentication Protocol Design for Vehicular Tactile Wireless Networks**

QSAFE-V：用于车载触觉无线网络的量子增强轻量级认证协议设计 math.QA

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03850v1) [paper-pdf](http://arxiv.org/pdf/2511.03850v1)

**Authors**: Shakil Ahmed, Amika Tabassum, Ibrahim Almazyad, Ashfaq Khokhar

**Abstract**: With the rapid advancement of 6G technology, the Tactile Internet is emerging as a novel paradigm of interaction, particularly in intelligent transportation systems, where stringent demands for ultra-low latency and high reliability are prevalent. During the transmission and coordination of autonomous vehicles, malicious adversaries may attempt to compromise control commands or swarm behavior, posing severe threats to road safety and vehicular intelligence. Many existing authentication schemes claim to provide security against conventional attacks. However, recent developments in quantum computing have revealed critical vulnerabilities in these schemes, particularly under quantum-enabled adversarial models. In this context, the design of a quantum-secured, lightweight authentication scheme that is adaptable to vehicular mobility becomes essential. This paper proposes QSAFE-V, a quantum-secured authentication framework for edge-enabled vehicles that surpasses traditional security models. We conduct formal security proofs based on quantum key distribution and quantum adversary models, and also perform context-driven reauthentication analysis based on vehicular behavior. The output of quantum resilience evaluations indicates that QSAFE-V provides robust protection against quantum and contextual attacks. Furthermore, detailed performance analysis reveals that QSAFE-V achieves comparable communication and computation costs to classical schemes, while offering significantly stronger security guarantees under wireless Tactile Internet conditions.

摘要: 随着6 G技术的快速发展，触觉互联网正在成为一种新型的交互范式，特别是在智能交通系统中，对超低延迟和高可靠性的严格要求普遍存在。在自动驾驶汽车的传输和协调过程中，恶意对手可能会试图破坏控制命令或群体行为，对道路安全和车辆智能构成严重威胁。许多现有的身份验证方案声称可以提供针对传统攻击的安全性。然而，量子计算的最新发展揭示了这些方案中的关键漏洞，特别是在量子对抗模型下。在此背景下，设计一种可适应车辆移动性的量子安全、轻量级认证方案变得至关重要。本文提出了QSAFE-V，这是一个针对边缘启用车辆的量子安全认证框架，超越了传统的安全模型。我们基于量子密钥分发和量子对手模型进行形式安全证明，并基于车辆行为进行上下文驱动的重新认证分析。量子弹性评估的输出表明，QSAFE-V提供了针对量子和上下文攻击的强大保护。此外，详细的性能分析表明，QSAFE-V实现了与经典方案相当的通信和计算成本，同时在无线触觉互联网条件下提供了明显更强的安全保证。



## **22. Whisper Leak: a side-channel attack on Large Language Models**

Whisper Leak：对大型语言模型的侧信道攻击 cs.CR

14 pages, 7 figures

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03675v1) [paper-pdf](http://arxiv.org/pdf/2511.03675v1)

**Authors**: Geoff McDonald, Jonathan Bar Or

**Abstract**: Large Language Models (LLMs) are increasingly deployed in sensitive domains including healthcare, legal services, and confidential communications, where privacy is paramount. This paper introduces Whisper Leak, a side-channel attack that infers user prompt topics from encrypted LLM traffic by analyzing packet size and timing patterns in streaming responses. Despite TLS encryption protecting content, these metadata patterns leak sufficient information to enable topic classification. We demonstrate the attack across 28 popular LLMs from major providers, achieving near-perfect classification (often >98% AUPRC) and high precision even at extreme class imbalance (10,000:1 noise-to-target ratio). For many models, we achieve 100% precision in identifying sensitive topics like "money laundering" while recovering 5-20% of target conversations. This industry-wide vulnerability poses significant risks for users under network surveillance by ISPs, governments, or local adversaries. We evaluate three mitigation strategies - random padding, token batching, and packet injection - finding that while each reduces attack effectiveness, none provides complete protection. Through responsible disclosure, we have collaborated with providers to implement initial countermeasures. Our findings underscore the need for LLM providers to address metadata leakage as AI systems handle increasingly sensitive information.

摘要: 大型语言模型（LLM）越来越多地部署在敏感领域，包括医疗保健、法律服务和保密通信，这些领域的隐私至关重要。本文介绍了Whisper Leak，这是一种侧通道攻击，通过分析流响应中的数据包大小和时间模式，从加密的LLM流量中推断出用户提示主题。尽管使用SSL加密保护内容，但这些元数据模式会泄露足够的信息来启用主题分类。我们展示了针对主要提供商的28种流行LLM的攻击，即使在极端类别失衡（10，000：1噪音与目标比）的情况下，也实现了近乎完美的分类（通常> 98%AUPRC）和高精度。对于许多模型，我们在识别“洗钱”等敏感话题方面实现了100%的准确度，同时恢复5-20%的目标对话。这种全行业的漏洞给受到ISP、政府或本地对手网络监视的用户带来了重大风险。我们评估了三种缓解策略--随机填充、令牌填充和数据包注入--发现虽然每种策略都会降低攻击有效性，但都没有提供完整的保护。通过负责任的披露，我们与提供商合作实施初步应对措施。我们的研究结果强调，随着人工智能系统处理日益敏感的信息，LLM提供商需要解决元数据泄露问题。



## **23. SHIELD: Securing Healthcare IoT with Efficient Machine Learning Techniques for Anomaly Detection**

SHIELD：利用高效的机器学习技术来保护医疗保健物联网的安全，以进行异常检测 cs.LG

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03661v1) [paper-pdf](http://arxiv.org/pdf/2511.03661v1)

**Authors**: Mahek Desai, Apoorva Rumale, Marjan Asadinia

**Abstract**: The integration of IoT devices in healthcare introduces significant security and reliability challenges, increasing susceptibility to cyber threats and operational anomalies. This study proposes a machine learning-driven framework for (1) detecting malicious cyberattacks and (2) identifying faulty device anomalies, leveraging a dataset of 200,000 records. Eight machine learning models are evaluated across three learning approaches: supervised learning (XGBoost, K-Nearest Neighbors (K- NN)), semi-supervised learning (Generative Adversarial Networks (GAN), Variational Autoencoders (VAE)), and unsupervised learning (One-Class Support Vector Machine (SVM), Isolation Forest, Graph Neural Networks (GNN), and Long Short-Term Memory (LSTM) Autoencoders). The comprehensive evaluation was conducted across multiple metrics like F1-score, precision, recall, accuracy, ROC-AUC, computational efficiency. XGBoost achieved 99\% accuracy with minimal computational overhead (0.04s) for anomaly detection, while Isolation Forest balanced precision and recall effectively. LSTM Autoencoders underperformed with lower accuracy and higher latency. For attack detection, KNN achieved near-perfect precision, recall, and F1-score with the lowest computational cost (0.05s), followed by VAE at 97% accuracy. GAN showed the highest computational cost with lowest accuracy and ROC-AUC. These findings enhance IoT-enabled healthcare security through effective anomaly detection strategies. By improving early detection of cyber threats and device failures, this framework has the potential to prevent data breaches, minimize system downtime, and ensure the continuous and safe operation of medical devices, ultimately safeguarding patient health and trust in IoT-driven healthcare solutions.

摘要: 物联网设备在医疗保健中的集成带来了重大的安全性和可靠性挑战，增加了对网络威胁和操作异常的敏感性。这项研究提出了一个机器学习驱动的框架，用于（1）检测恶意网络攻击和（2）识别故障设备异常，利用200，000条记录的数据集。通过三种学习方法评估了八种机器学习模型：监督学习（XGBoost、K-近邻（K-NN））、半监督学习（生成对抗网络（GAN）、变分自动编码器（VAE））和无监督学习（一类支持向量机（SV）、隔离森林、图神经网络（GNN）和长短期记忆（LSTM）自动编码器）。全面评估是针对F1评分、精确度、召回率、准确性、ROC-AUC、计算效率等多个指标进行的。XGBoost在异常检测的计算开销最小（0.04s）的情况下达到了99%的准确率，而隔离森林有效地平衡了查准率和查全率。LSTM自动编码器表现不佳，准确性较低，延迟较高。对于攻击检测，KNN以最低的计算成本（0.05s）实现了近乎完美的精度，召回率和F1分数，其次是VAE，准确率为97%。GAN的计算成本最高，准确度和ROC-AUC最低。这些发现通过有效的异常检测策略增强了物联网支持的医疗安全性。通过改进对网络威胁和设备故障的早期检测，该框架有可能防止数据泄露，最大限度地减少系统停机时间，并确保医疗设备的持续安全运行，最终保障患者健康和对物联网驱动的医疗保健解决方案的信任。



## **24. Byzantine-Robust Federated Learning with Learnable Aggregation Weights**

具有可学习聚合权重的拜占庭稳健联邦学习 cs.LG

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03529v1) [paper-pdf](http://arxiv.org/pdf/2511.03529v1)

**Authors**: Javad Parsa, Amir Hossein Daghestani, André M. H. Teixeira, Mikael Johansson

**Abstract**: Federated Learning (FL) enables clients to collaboratively train a global model without sharing their private data. However, the presence of malicious (Byzantine) clients poses significant challenges to the robustness of FL, particularly when data distributions across clients are heterogeneous. In this paper, we propose a novel Byzantine-robust FL optimization problem that incorporates adaptive weighting into the aggregation process. Unlike conventional approaches, our formulation treats aggregation weights as learnable parameters, jointly optimizing them alongside the global model parameters. To solve this optimization problem, we develop an alternating minimization algorithm with strong convergence guarantees under adversarial attack. We analyze the Byzantine resilience of the proposed objective. We evaluate the performance of our algorithm against state-of-the-art Byzantine-robust FL approaches across various datasets and attack scenarios. Experimental results demonstrate that our method consistently outperforms existing approaches, particularly in settings with highly heterogeneous data and a large proportion of malicious clients.

摘要: 联合学习（FL）使客户能够协作训练全球模型，而无需共享其私人数据。然而，恶意（拜占庭）客户端的存在对FL的稳健性构成了重大挑战，特别是当客户端之间的数据分布是异类时。在本文中，我们提出了一个新颖的拜占庭鲁棒FL优化问题，该问题将自适应加权纳入聚合过程。与传统方法不同，我们的公式将聚合权重视为可学习参数，并与全局模型参数一起联合优化它们。为了解决这个优化问题，我们开发了一种在对抗攻击下具有强收敛保证的交替最小化算法。我们分析了拟议目标的拜占庭弹性。我们针对各种数据集和攻击场景中最先进的拜占庭鲁棒FL方法评估了我们的算法的性能。实验结果表明，我们的方法始终优于现有方法，特别是在数据高度异类和恶意客户端比例很大的环境中。



## **25. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

面对威胁的操纵：评估端到端视觉语言动作模型中的身体脆弱性 cs.CV

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2409.13174v4) [paper-pdf](http://arxiv.org/pdf/2409.13174v4)

**Authors**: Hao Cheng, Erjia Xiao, Yichi Wang, Chengyuan Yu, Mengshu Sun, Qiang Zhang, Jiahang Cao, Yijie Guo, Ning Liu, Kaidi Xu, Jize Zhang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical threats.

摘要: 最近，在多模式大型语言模型（MLLM）进步的推动下，人们提出了视觉语言动作模型（VLAM），以在机器人操纵任务的开放词汇场景中实现更好的性能。由于操纵任务涉及与物理世界的直接互动，因此在执行该任务期间确保稳健性和安全性始终是一个非常关键的问题。本文通过综合当前对MLLM的安全性研究以及物理世界中操纵任务的具体应用场景，对VLAM在潜在物理威胁面前进行了全面评估。具体来说，我们提出了物理脆弱性评估管道（PVEP），它可以整合尽可能多的视觉模式物理威胁，以评估VLAM的物理稳健性。PVEP中的物理威胁具体包括分发外、基于印刷术的视觉提示和对抗性补丁攻击。通过比较VLAM受到攻击前后的性能波动，我们提供了VLAM如何响应不同物理威胁的可概括的\textBF{\textit{Analyses}。



## **26. Death by a Thousand Prompts: Open Model Vulnerability Analysis**

千人死亡：开放模型漏洞分析 cs.CR

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03247v1) [paper-pdf](http://arxiv.org/pdf/2511.03247v1)

**Authors**: Amy Chang, Nicholas Conley, Harish Santhanalakshmi Ganesan, Adam Swanda

**Abstract**: Open-weight models provide researchers and developers with accessible foundations for diverse downstream applications. We tested the safety and security postures of eight open-weight large language models (LLMs) to identify vulnerabilities that may impact subsequent fine-tuning and deployment. Using automated adversarial testing, we measured each model's resilience against single-turn and multi-turn prompt injection and jailbreak attacks. Our findings reveal pervasive vulnerabilities across all tested models, with multi-turn attacks achieving success rates between 25.86\% and 92.78\% -- representing a $2\times$ to $10\times$ increase over single-turn baselines. These results underscore a systemic inability of current open-weight models to maintain safety guardrails across extended interactions. We assess that alignment strategies and lab priorities significantly influence resilience: capability-focused models such as Llama 3.3 and Qwen 3 demonstrate higher multi-turn susceptibility, whereas safety-oriented designs such as Google Gemma 3 exhibit more balanced performance.   The analysis concludes that open-weight models, while crucial for innovation, pose tangible operational and ethical risks when deployed without layered security controls. These findings are intended to inform practitioners and developers of the potential risks and the value of professional AI security solutions to mitigate exposure. Addressing multi-turn vulnerabilities is essential to ensure the safe, reliable, and responsible deployment of open-weight LLMs in enterprise and public domains. We recommend adopting a security-first design philosophy and layered protections to ensure resilient deployments of open-weight models.

摘要: 开放权重模型为研究人员和开发人员提供了各种下游应用程序的可用基础。我们测试了八个开放权重大型语言模型（LLM）的安全性和安全姿势，以识别可能影响后续微调和部署的漏洞。使用自动对抗测试，我们测量了每个模型对单回合和多回合提示注入和越狱攻击的弹性。我们的调查结果揭示了所有测试模型中普遍存在的漏洞，多回合攻击的成功率在25.86%和92.78%之间，比单回合基线增加了2美元到10美元。这些结果强调了当前开放重量模型系统性地无法在长期互动中维持安全护栏。我们评估了对齐策略和实验室优先事项对韧性有显着影响：Llama 3.3和Qwen 3等以能力为中心的模型表现出更高的多圈敏感性，而Google Gemma 3等以安全为导向的设计表现出更平衡的性能。   分析得出的结论是，开放权重模型虽然对创新至关重要，但在没有分层安全控制的情况下部署时会带来切实的运营和道德风险。这些调查结果旨在让从业者和开发人员了解专业人工智能安全解决方案的潜在风险和价值，以减轻风险。解决多回合漏洞对于确保在企业和公共领域安全、可靠和负责任地部署开放权重LLM至关重要。我们建议采用安全第一的设计理念和分层保护，以确保开重模型的弹性部署。



## **27. Bayesian Advantage of Re-Identification Attack in the Shuffle Model**

Shuffle模型下重识别攻击的贝叶斯优势 cs.CR

Accepted by CSF 2026 -- 39th IEEE Computer Security Foundations  Symposium

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03213v1) [paper-pdf](http://arxiv.org/pdf/2511.03213v1)

**Authors**: Pengcheng Su, Haibo Cheng, Ping Wang

**Abstract**: The shuffle model, which anonymizes data by randomly permuting user messages, has been widely adopted in both cryptography and differential privacy. In this work, we present the first systematic study of the Bayesian advantage in re-identifying a user's message under the shuffle model. We begin with a basic setting: one sample is drawn from a distribution $P$, and $n - 1$ samples are drawn from a distribution $Q$, after which all $n$ samples are randomly shuffled. We define $\beta_n(P, Q)$ as the success probability of a Bayes-optimal adversary in identifying the sample from $P$, and define the additive and multiplicative Bayesian advantages as $\mathsf{Adv}_n^{+}(P, Q) = \beta_n(P,Q) - \frac{1}{n}$ and $\mathsf{Adv}_n^{\times}(P, Q) = n \cdot \beta_n(P,Q)$, respectively. We derive exact analytical expressions and asymptotic characterizations of $\beta_n(P, Q)$, along with evaluations in several representative scenarios. Furthermore, we establish (nearly) tight mutual bounds between the additive Bayesian advantage and the total variation distance. Finally, we extend our analysis beyond the basic setting and present, for the first time, an upper bound on the success probability of Bayesian attacks in shuffle differential privacy. Specifically, when the outputs of $n$ users -- each processed through an $\varepsilon$-differentially private local randomizer -- are shuffled, the probability that an attacker successfully re-identifies any target user's message is at most $e^{\varepsilon}/n$.

摘要: 洗牌模型通过随机排列用户消息来匿名数据，已在密码学和差异隐私中广泛采用。在这项工作中，我们首次对洗牌模型下重新识别用户消息的Bayesian优势进行了系统研究。我们从基本设置开始：从分布$P$中提取一个样本，从分布$Q$中提取$n - 1$样本，之后所有$n$样本都被随机洗牌。我们将$\Beta_n（P，Q）$定义为Bayes最优对手从$P$中识别样本的成功概率，并将相加性和相乘性Bayesian优势分别定义为$\mathsf{Adv}_n^{+}（P，Q）= \Beta_n（P，Q）- \fRAC{1}{n}$和$\mathsf{Adv}_n^{\times}（P，Q）= n \csot\Beta_n（P，Q）$。我们推导出$\Beta_n（P，Q）$的精确分析表达和渐进特征，以及几个代表性场景下的评估。此外，我们在加性Bayesian优势和总变异距离之间建立了（几乎）紧密的相互界限。最后，我们将我们的分析扩展到基本设置之外，并首次提出了洗牌差异隐私中Bayesian攻击成功概率的上限。具体来说，当$n$ users的输出（每个用户都通过$\varepð $-差异私有本地随机发生器处理）被洗牌时，攻击者成功重新识别任何目标用户消息的可能性最多为$e_{\varepŸ}/n$。



## **28. SAAIPAA: Optimizing aspect-angles-invariant physical adversarial attacks on SAR target recognition models**

SAAIPAA：优化对SAR目标识别模型的长宽角不变物理对抗攻击 eess.IV

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03192v1) [paper-pdf](http://arxiv.org/pdf/2511.03192v1)

**Authors**: Isar Lemeire, Yee Wei Law, Sang-Heon Lee, Will Meakin, Tat-Jun Chin

**Abstract**: Synthetic aperture radar (SAR) enables versatile, all-time, all-weather remote sensing. Coupled with automatic target recognition (ATR) leveraging machine learning (ML), SAR is empowering a wide range of Earth observation and surveillance applications. However, the surge of attacks based on adversarial perturbations against the ML algorithms underpinning SAR ATR is prompting the need for systematic research into adversarial perturbation mechanisms. Research in this area began in the digital (image) domain and evolved into the physical (signal) domain, resulting in physical adversarial attacks (PAAs) that strategically exploit corner reflectors as attack vectors to evade ML-based ATR. This paper proposes a novel framework called SAR Aspect-Angles-Invariant Physical Adversarial Attack (SAAIPAA) for physics-based modelling of reflector-actuated adversarial perturbations, which improves on the rigor of prior work. A unique feature of SAAIPAA is its ability to remain effective even when the attacker lacks knowledge of the SAR platform's aspect angles, by deploying at least one reflector in each azimuthal quadrant and optimizing reflector orientations. The resultant physical evasion attacks are efficiently realizable and optimal over the considered range of aspect angles between a SAR platform and a target, achieving state-of-the-art fooling rates (over 80% for DenseNet-121 and ResNet50) in the white-box setting. When aspect angles are known to the attacker, an average fooling rate of 99.2% is attainable. In black-box settings, although the attack efficacy of SAAIPAA transfers well between some models (e.g., from ResNet50 to DenseNet121), the transferability to some models (e.g., MobileNetV2) can be improved. A useful outcome of using the MSTAR dataset for the experiments in this article, a method for generating bounding boxes for densely sampled azimuthal SAR datasets is introduced.

摘要: 合成口径雷达（SAR）实现多功能、全时、全天候遥感。与利用机器学习（ML）的自动目标识别（ATR）相结合，SAR正在为广泛的地球观测和监视应用提供支持。然而，基于针对SAR ATR基础的ML算法的对抗性扰动的攻击激增，促使人们需要对对抗性扰动机制进行系统研究。该领域的研究始于数字（图像）领域，并发展到物理（信号）领域，从而导致物理对抗攻击（PAC），这些攻击策略性地利用角反射器作为攻击载体来规避基于ML的ATR。本文提出了一种名为SAR方位角不变物理对抗攻击（SAAIPAA）的新型框架，用于对反射器驱动的对抗扰动进行基于物理的建模，该框架改进了先前工作的严谨性。SAAIPAA的一个独特功能是，即使攻击者缺乏对SAR平台长宽角的了解，它也能够通过在每个方位角象限中部署至少一个反射器并优化反射器方向来保持有效。由此产生的物理规避攻击在SAR平台和目标之间考虑的长宽角范围内是可有效实现且最佳的，在白盒设置中实现了最先进的愚弄率（DenseNet-121和ResNet 50超过80%）。当攻击者已知长宽角时，平均愚弄率可达99.2%。在黑匣子环境中，尽管SAAIPAA的攻击功效在某些型号之间转移良好（例如，从ResNet 50到DenseNet 121），到某些型号的可移植性（例如，MobileNetV 2）可以改进。本文使用MSTAR数据集进行实验的一个有用结果是，介绍了一种为密集采样的方位角SAR数据集生成边界框的方法。



## **29. From Insight to Exploit: Leveraging LLM Collaboration for Adaptive Adversarial Text Generation**

从洞察到利用：利用LLM协作进行自适应对抗文本生成 cs.LG

Findings of the Association for Computational Linguistics: EMNLP 2025  (camera-ready)

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03128v1) [paper-pdf](http://arxiv.org/pdf/2511.03128v1)

**Authors**: Najrin Sultana, Md Rafi Ur Rashid, Kang Gu, Shagufta Mehnaz

**Abstract**: LLMs can provide substantial zero-shot performance on diverse tasks using a simple task prompt, eliminating the need for training or fine-tuning. However, when applying these models to sensitive tasks, it is crucial to thoroughly assess their robustness against adversarial inputs. In this work, we introduce Static Deceptor (StaDec) and Dynamic Deceptor (DyDec), two innovative attack frameworks designed to systematically generate dynamic and adaptive adversarial examples by leveraging the understanding of the LLMs. We produce subtle and natural-looking adversarial inputs that preserve semantic similarity to the original text while effectively deceiving the target LLM. By utilizing an automated, LLM-driven pipeline, we eliminate the dependence on external heuristics. Our attacks evolve with the advancements in LLMs and demonstrate strong transferability across models unknown to the attacker. Overall, this work provides a systematic approach for the self-assessment of an LLM's robustness. We release our code and data at https://github.com/Shukti042/AdversarialExample.

摘要: LLM可以使用简单的任务提示在不同任务上提供大量的零射击性能，从而消除了培训或微调的需要。然而，当将这些模型应用于敏感任务时，彻底评估它们对对抗输入的稳健性至关重要。在这项工作中，我们引入了静态欺骗者（StaDec）和动态欺骗者（DyDec），这是两个创新的攻击框架，旨在通过利用对LLM的理解来系统性地生成动态和自适应的对抗示例。我们生成微妙且看起来自然的对抗输入，这些输入保留了与原始文本的语义相似性，同时有效地欺骗了目标LLM。通过利用自动化的LLM驱动管道，我们消除了对外部启发式方法的依赖。我们的攻击随着LLM的进步而发展，并在攻击者未知的模型之间表现出强大的可移植性。总体而言，这项工作为LLM稳健性的自我评估提供了一种系统性方法。我们在https://github.com/Shukti042/AdversarialExample上发布我们的代码和数据。



## **30. A Reliable Cryptographic Framework for Empirical Machine Unlearning Evaluation**

用于经验机器非学习评估的可靠密码框架 cs.LG

Accepted at the 39th Conference on Neural Information Processing  Systems (NeurIPS 2025)

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2404.11577v4) [paper-pdf](http://arxiv.org/pdf/2404.11577v4)

**Authors**: Yiwen Tu, Pingbang Hu, Jiaqi Ma

**Abstract**: Machine unlearning updates machine learning models to remove information from specific training samples, complying with data protection regulations that allow individuals to request the removal of their personal data. Despite the recent development of numerous unlearning algorithms, reliable evaluation of these algorithms remains an open research question. In this work, we focus on membership inference attack (MIA) based evaluation, one of the most common approaches for evaluating unlearning algorithms, and address various pitfalls of existing evaluation metrics lacking theoretical understanding and reliability. Specifically, by modeling the proposed evaluation process as a \emph{cryptographic game} between unlearning algorithms and MIA adversaries, the naturally induced evaluation metric measures the data removal efficacy of unlearning algorithms and enjoys provable guarantees that existing evaluation metrics fail to satisfy. Furthermore, we propose a practical and efficient approximation of the induced evaluation metric and demonstrate its effectiveness through both theoretical analysis and empirical experiments. Overall, this work presents a novel and reliable approach to empirically evaluating unlearning algorithms, paving the way for the development of more effective unlearning techniques.

摘要: 机器取消学习更新机器学习模型，以从特定训练样本中删除信息，遵守允许个人请求删除其个人数据的数据保护法规。尽管最近开发了许多取消学习算法，但对这些算法的可靠评估仍然是一个悬而未决的研究问题。在这项工作中，我们重点关注基于成员资格推理攻击（MIA）的评估，这是评估取消学习算法的最常见方法之一，并解决了现有评估指标缺乏理论理解和可靠性的各种陷阱。具体来说，通过将提出的评估过程建模为取消学习算法和MIA对手之间的\{加密游戏}，自然诱导的评估指标衡量取消学习算法的数据删除效率，并享有现有评估指标无法满足的可证明保证。此外，我们还提出了一种实用且有效的诱导评估指标的逼近方法，并通过理论分析和实证实验证明了其有效性。总的来说，这项工作提供了一种新颖且可靠的方法来经验评估去学习算法，为开发更有效的去学习技术铺平了道路。



## **31. Evaluating Control Protocols for Untrusted AI Agents**

评估不受信任的人工智能代理的控制协议 cs.AI

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02997v1) [paper-pdf](http://arxiv.org/pdf/2511.02997v1)

**Authors**: Jon Kutasov, Chloe Loughridge, Yuqi Sun, Henry Sleight, Buck Shlegeris, Tyler Tracy, Joe Benton

**Abstract**: As AI systems become more capable and widely deployed as agents, ensuring their safe operation becomes critical. AI control offers one approach to mitigating the risk from untrusted AI agents by monitoring their actions and intervening or auditing when necessary. Evaluating the safety of these protocols requires understanding both their effectiveness against current attacks and their robustness to adaptive adversaries. In this work, we systematically evaluate a range of control protocols in SHADE-Arena, a dataset of diverse agentic environments. First, we evaluate blue team protocols, including deferral to trusted models, resampling, and deferring on critical actions, against a default attack policy. We find that resampling for incrimination and deferring on critical actions perform best, increasing safety from 50% to 96%. We then iterate on red team strategies against these protocols and find that attack policies with additional affordances, such as knowledge of when resampling occurs or the ability to simulate monitors, can substantially improve attack success rates against our resampling strategy, decreasing safety to 17%. However, deferring on critical actions is highly robust to even our strongest red team strategies, demonstrating the importance of denying attack policies access to protocol internals.

摘要: 随着人工智能系统变得越来越强大，并被广泛部署为代理，确保其安全运行变得至关重要。人工智能控制提供了一种方法，通过监控他们的行为并在必要时进行干预或审计来减轻不受信任的人工智能代理的风险。评估这些协议的安全性需要了解它们对当前攻击的有效性以及它们对自适应对手的鲁棒性。在这项工作中，我们系统地评估了一系列的控制协议在SHADE竞技场，不同的代理环境的数据集。首先，我们评估蓝队协议，包括针对默认攻击政策推迟到可信模型、重新分配和推迟关键行动。我们发现，重新定罪和推迟关键行动的效果最好，安全性从50%提高到96%。然后，我们针对这些协议研究了红队策略，发现具有额外功能的攻击策略（例如了解重新分配时间或模拟监视器的能力）可以大幅提高针对我们重新分配策略的攻击成功率，将安全性降低至17%。然而，即使是我们最强大的红队策略，推迟关键行动也是非常稳健的，这证明了拒绝攻击策略访问协议内部的重要性。



## **32. Diffusion Models are Robust Pretrainers**

扩散模型是鲁棒的预训练器 eess.IV

To be published in IEEE Signal Processing Letters

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02793v1) [paper-pdf](http://arxiv.org/pdf/2511.02793v1)

**Authors**: Mika Yagoda, Shady Abu-Hussein, Raja Giryes

**Abstract**: Diffusion models have gained significant attention for high-fidelity image generation. Our work investigates the potential of exploiting diffusion models for adversarial robustness in image classification and object detection. Adversarial attacks challenge standard models in these tasks by perturbing inputs to force incorrect predictions. To address this issue, many approaches use training schemes for forcing the robustness of the models, which increase training costs. In this work, we study models built on top of off-the-shelf diffusion models and demonstrate their practical significance: they provide a low-cost path to robust representations, allowing lightweight heads to be trained on frozen features without full adversarial training. Our empirical evaluations on ImageNet, CIFAR-10, and PASCAL VOC show that diffusion-based classifiers and detectors achieve meaningful adversarial robustness with minimal compute. While clean and adversarial accuracies remain below state-of-the-art adversarially trained CNNs or ViTs, diffusion pretraining offers a favorable tradeoff between efficiency and robustness. This work opens a promising avenue for integrating diffusion models into resource-constrained robust deployments.

摘要: 扩散模型在高保真图像生成方面受到了广泛关注。我们的工作研究了利用扩散模型在图像分类和对象检测中实现对抗鲁棒性的潜力。对抗性攻击通过扰乱输入以迫使做出错误的预测来挑战这些任务中的标准模型。为了解决这个问题，许多方法使用训练方案来迫使模型具有稳健性，这会增加训练成本。在这项工作中，我们研究了在现成的扩散模型基础上构建的模型，并证明了它们的实际意义：它们提供了一种低成本的鲁棒表示路径，允许轻量级头部在冻结特征上训练，而无需进行完全的对抗训练。我们对ImageNet、CIFAR-10和Pascal VOC的经验评估表明，基于扩散的分类器和检测器可以以最少的计算实现有意义的对抗鲁棒性。虽然干净和对抗准确性仍然低于最先进的对抗训练CNN或ViT，但扩散预训练在效率和稳健性之间提供了有利的权衡。这项工作为将扩散模型集成到资源受限的稳健部署中开辟了一条有希望的途径。



## **33. Enhancing Federated Learning Privacy with QUBO**

利用QUBO增强联邦学习隐私 cs.LG

8 pages, 9 figures

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02785v1) [paper-pdf](http://arxiv.org/pdf/2511.02785v1)

**Authors**: Andras Ferenczi, Sutapa Samanta, Dagen Wang, Todd Hodges

**Abstract**: Federated learning (FL) is a widely used method for training machine learning (ML) models in a scalable way while preserving privacy (i.e., without centralizing raw data). Prior research shows that the risk of exposing sensitive data increases cumulatively as the number of iterations where a client's updates are included in the aggregated model increase. Attackers can launch membership inference attacks (MIA; deciding whether a sample or client participated), property inference attacks (PIA; inferring attributes of a client's data), and model inversion attacks (MI; reconstructing inputs), thereby inferring client-specific attributes and, in some cases, reconstructing inputs. In this paper, we mitigate risk by substantially reducing per client exposure using a quantum computing-inspired quadratic unconstrained binary optimization (QUBO) formulation that selects a small subset of client updates most relevant for each training round. In this work, we focus on two threat vectors: (i) information leakage by clients during training and (ii) adversaries who can query or obtain the global model. We assume a trusted central server and do not model server compromise. This method also assumes that the server has access to a validation/test set with global data distribution. Experiments on the MNIST dataset with 300 clients in 20 rounds showed a 95.2% per-round and 49% cumulative privacy exposure reduction, with 147 clients' updates never being used during training while maintaining in general the full-aggregation accuracy or even better. The method proved to be efficient at lower scale and more complex model as well. A CINIC-10 dataset-based experiment with 30 clients resulted in 82% per-round privacy improvement and 33% cumulative privacy.

摘要: 联邦学习（FL）是一种广泛使用的方法，用于以可扩展的方式训练机器学习（ML）模型，同时保护隐私（即，无需集中原始数据）。之前的研究表明，随着客户端更新包含在聚合模型中的迭代次数的增加，暴露敏感数据的风险会累积增加。攻击者可以发起成员资格推断攻击（MIA;决定样本或客户端是否参与）、属性推断攻击（PIA;推断客户端数据的属性）和模型倒置攻击（MI;重建输入），从而推断客户端特定的属性，并在某些情况下重建输入。在本文中，我们使用受量子计算启发的二次无约束二元优化（QUBO）公式，通过大幅减少每个客户的风险来降低风险，该公式选择与每个训练轮最相关的客户更新的一小子集。在这项工作中，我们重点关注两个威胁载体：（i）客户在训练期间的信息泄露和（ii）可以查询或获取全局模型的对手。我们假设一个受信任的中央服务器，并且不对服务器妥协进行建模。此方法还假设服务器可以访问具有全局数据分布的验证/测试集。在MNIST数据集上进行的20轮300个客户端实验显示，每轮隐私暴露减少了95.2%，累计隐私暴露减少了49%，其中147个客户端的更新在训练期间从未被使用，同时总体上保持了全聚合准确性甚至更好。事实证明，该方法在较小规模和更复杂的模型下也有效。对30名客户进行的基于CINIC-10厕所的实验导致每轮隐私改善82%，累积隐私改善33%。



## **34. Nesterov-Accelerated Robust Federated Learning Over Byzantine Adversaries**

针对拜占庭对手的内斯特罗夫加速稳健联邦学习 cs.LG

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02657v1) [paper-pdf](http://arxiv.org/pdf/2511.02657v1)

**Authors**: Lihan Xu, Yanjie Dong, Gang Wang, Runhao Zeng, Xiaoyi Fan, Xiping Hu

**Abstract**: We investigate robust federated learning, where a group of workers collaboratively train a shared model under the orchestration of a central server in the presence of Byzantine adversaries capable of arbitrary and potentially malicious behaviors. To simultaneously enhance communication efficiency and robustness against such adversaries, we propose a Byzantine-resilient Nesterov-Accelerated Federated Learning (Byrd-NAFL) algorithm. Byrd-NAFL seamlessly integrates Nesterov's momentum into the federated learning process alongside Byzantine-resilient aggregation rules to achieve fast and safeguarding convergence against gradient corruption. We establish a finite-time convergence guarantee for Byrd-NAFL under non-convex and smooth loss functions with relaxed assumption on the aggregated gradients. Extensive numerical experiments validate the effectiveness of Byrd-NAFL and demonstrate the superiority over existing benchmarks in terms of convergence speed, accuracy, and resilience to diverse Byzantine attack strategies.

摘要: 我们研究了稳健的联邦学习，其中一群工作人员在中央服务器的编排下，在存在能够任意和潜在恶意行为的拜占庭对手的情况下协作训练共享模型。为了同时增强通信效率和针对此类对手的鲁棒性，我们提出了一种具有拜占庭弹性的内斯特罗夫加速联邦学习（Byrd-NAFL）算法。Byrd-NAFL将Nesterov的动力与拜占庭弹性聚合规则无缝集成到联邦学习流程中，以实现快速并保护收敛，防止梯度腐败。我们在非凸和光滑损失函数下建立了Byrd-NAFL的有限时间收敛保证，并放宽了对聚合梯度的假设。大量的数值实验验证了Byrd-NAFL的有效性，并展示了在收敛速度、准确性和对各种拜占庭攻击策略的弹性方面优于现有基准。



## **35. Do Methods to Jailbreak and Defend LLMs Generalize Across Languages?**

越狱和捍卫LLM的方法是否适用于语言？ cs.CL

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.00689v2) [paper-pdf](http://arxiv.org/pdf/2511.00689v2)

**Authors**: Berk Atil, Rebecca J. Passonneau, Fred Morstatter

**Abstract**: Large language models (LLMs) undergo safety alignment after training and tuning, yet recent work shows that safety can be bypassed through jailbreak attacks. While many jailbreaks and defenses exist, their cross-lingual generalization remains underexplored. This paper presents the first systematic multilingual evaluation of jailbreaks and defenses across ten languages -- spanning high-, medium-, and low-resource languages -- using six LLMs on HarmBench and AdvBench. We assess two jailbreak types: logical-expression-based and adversarial-prompt-based. For both types, attack success and defense robustness vary across languages: high-resource languages are safer under standard queries but more vulnerable to adversarial ones. Simple defenses can be effective, but are language- and model-dependent. These findings call for language-aware and cross-lingual safety benchmarks for LLMs.

摘要: 大型语言模型（LLM）在训练和调整后会经历安全调整，但最近的工作表明，安全性可以通过越狱攻击绕过。虽然存在许多越狱和防御措施，但它们的跨语言概括仍然没有得到充分的研究。本文使用HarmBench和AdvBench上的六个LLM，首次对十种语言（跨越高、中和低资源语言）的越狱和防御进行了系统性的多语言评估。我们评估了两种越狱类型：基于逻辑表达的越狱和基于对抗提示的越狱。对于这两种类型，攻击成功率和防御稳健性因语言而异：高资源语言在标准查询下更安全，但更容易受到对抗查询的影响。简单的防御可能是有效的，但依赖于语言和模型。这些发现呼吁为LLM制定语言感知和跨语言安全基准。



## **36. Verifying LLM Inference to Prevent Model Weight Exfiltration**

CLARLLM推理以防止模型重量溢出 cs.CR

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02620v1) [paper-pdf](http://arxiv.org/pdf/2511.02620v1)

**Authors**: Roy Rinberg, Adam Karvonen, Alex Hoover, Daniel Reuter, Keri Warr

**Abstract**: As large AI models become increasingly valuable assets, the risk of model weight exfiltration from inference servers grows accordingly. An attacker controlling an inference server may exfiltrate model weights by hiding them within ordinary model outputs, a strategy known as steganography. This work investigates how to verify model responses to defend against such attacks and, more broadly, to detect anomalous or buggy behavior during inference. We formalize model exfiltration as a security game, propose a verification framework that can provably mitigate steganographic exfiltration, and specify the trust assumptions associated with our scheme. To enable verification, we characterize valid sources of non-determinism in large language model inference and introduce two practical estimators for them. We evaluate our detection framework on several open-weight models ranging from 3B to 30B parameters. On MOE-Qwen-30B, our detector reduces exfiltratable information to <0.5% with false-positive rate of 0.01%, corresponding to a >200x slowdown for adversaries. Overall, this work further establishes a foundation for defending against model weight exfiltration and demonstrates that strong protection can be achieved with minimal additional cost to inference providers.

摘要: 随着大型人工智能模型成为越来越有价值的资产，模型权重从推理服务器泄露的风险也相应增加。控制推理服务器的攻击者可以通过将模型权重隐藏在普通模型输出中来溢出模型权重，这种策略称为隐写术。这项工作研究了如何验证模型响应以抵御此类攻击，以及更广泛地说，在推理过程中检测异常或有缺陷的行为。我们将模型溢出形式化为一个安全游戏，提出了一个可以证明减轻隐写溢出的验证框架，并指定与我们的方案相关的信任假设。为了实现验证，我们描述了大型语言模型推理中非决定性的有效来源，并为其引入了两个实用的估计器。我们在从3B到30 B参数的几个开放权重模型上评估了我们的检测框架。在MOE-Qwen-30 B上，我们的检测器将可渗透信息减少到<0.5%，假阳性率为0.01%，相当于对手的速度减慢> 200倍。总体而言，这项工作进一步奠定了防御模型权重溢出的基础，并证明可以以最小的额外成本来实现强大的保护。



## **37. Trustworthy Quantum Machine Learning: A Roadmap for Reliability, Robustness, and Security in the NISQ Era**

值得信赖的量子机器学习：NISQ时代可靠性、稳健性和安全性的路线图 quant-ph

22 Pages

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02602v1) [paper-pdf](http://arxiv.org/pdf/2511.02602v1)

**Authors**: Ferhat Ozgur Catak, Jungwon Seo, Umit Cali

**Abstract**: Quantum machine learning (QML) is a promising paradigm for tackling computational problems that challenge classical AI. Yet, the inherent probabilistic behavior of quantum mechanics, device noise in NISQ hardware, and hybrid quantum-classical execution pipelines introduce new risks that prevent reliable deployment of QML in real-world, safety-critical settings. This research offers a broad roadmap for Trustworthy Quantum Machine Learning (TQML), integrating three foundational pillars of reliability: (i) uncertainty quantification for calibrated and risk-aware decision making, (ii) adversarial robustness against classical and quantum-native threat models, and (iii) privacy preservation in distributed and delegated quantum learning scenarios. We formalize quantum-specific trust metrics grounded in quantum information theory, including a variance-based decomposition of predictive uncertainty, trace-distance-bounded robustness, and differential privacy for hybrid learning channels. To demonstrate feasibility on current NISQ devices, we validate a unified trust assessment pipeline on parameterized quantum classifiers, uncovering correlations between uncertainty and prediction risk, an asymmetry in attack vulnerability between classical and quantum state perturbations, and privacy-utility trade-offs driven by shot noise and quantum channel noise. This roadmap seeks to define trustworthiness as a first-class design objective for quantum AI.

摘要: 量子机器学习（QML）是解决挑战经典人工智能的计算问题的一种有前途的范式。然而，量子力学固有的概率行为、NISQ硬件中的设备噪音以及混合量子经典执行管道引入了新的风险，阻碍了QML在现实世界的安全关键环境中的可靠部署。这项研究为可信量子机器学习（TQML）提供了广泛的路线图，集成了可靠性的三个基本支柱：（i）用于校准和风险感知决策的不确定性量化，（ii）针对经典和量子原生威胁模型的对抗鲁棒性，（iii）分布式和委托量子学习场景中的隐私保护。我们以量子信息理论为基础，形式化了特定于量子的信任指标，包括基于方差的预测不确定性分解、跟踪距离有界鲁棒性和混合学习通道的差异隐私。为了证明当前NISQ设备上的可行性，我们在参数化量子分类器上验证了统一的信任评估管道，揭示了不确定性和预测风险之间的相关性、经典和量子状态扰动之间攻击脆弱性的不对称性，以及由散粒噪音和量子通道噪音驱动的隐私-公用事业权衡。该路线图旨在将可信度定义为量子人工智能的一流设计目标。



## **38. The Dark Side of LLMs: Agent-based Attacks for Complete Computer Takeover**

LLM的阴暗面：基于代理的完全计算机接管攻击 cs.CR

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2507.06850v5) [paper-pdf](http://arxiv.org/pdf/2507.06850v5)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables remarkable capabilities in natural language processing and generation. However, these systems introduce security vulnerabilities that extend beyond traditional content generation to system-level compromises. This paper presents a comprehensive evaluation of the LLMs security used as reasoning engines within autonomous agents, highlighting how they can be exploited as attack vectors capable of achieving computer takeovers. We focus on how different attack surfaces and trust boundaries can be leveraged to orchestrate such takeovers. We demonstrate that adversaries can effectively coerce popular LLMs into autonomously installing and executing malware on victim machines. Our evaluation of 18 state-of-the-art LLMs reveals an alarming scenario: 94.4% of models succumb to Direct Prompt Injection, and 83.3% are vulnerable to the more stealthy and evasive RAG Backdoor Attack. Notably, we tested trust boundaries within multi-agent systems, where LLM agents interact and influence each other, and we revealed that LLMs which successfully resist direct injection or RAG backdoor attacks will execute identical payloads when requested by peer agents. We found that 100.0% of tested LLMs can be compromised through Inter-Agent Trust Exploitation attacks, and that every model exhibits context-dependent security behaviors that create exploitable blind spots.

摘要: 大语言模型（LLM）代理和多代理系统的快速采用使自然语言处理和生成的能力显着。然而，这些系统引入了安全漏洞，这些安全漏洞超出了传统的内容生成，甚至会危及系统级安全。本文提出了一个全面的评估的LLM安全性作为推理引擎内的自主代理，突出它们如何可以被利用为攻击向量能够实现计算机接管。我们专注于如何利用不同的攻击面和信任边界来协调此类收购。我们证明，对手可以有效地强迫流行的LLM在受害者机器上自主安装和执行恶意软件。我们对18种最先进的LLM的评估揭示了一个令人震惊的情况：94.4%的模型屈服于直接提示注入，83.3%的模型容易受到更隐蔽和规避的RAG后门攻击。值得注意的是，我们测试了多代理系统中的信任边界，其中LLM代理相互交互和影响，我们揭示了成功抵抗直接注入或RAG后门攻击的LLM将在对等代理请求时执行相同的有效负载。我们发现，100.0%的测试LLM都可能通过代理间信任利用攻击而受到损害，并且每个模型都表现出依赖于上下文的安全行为，从而创建了可利用的盲点。



## **39. MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents**

针对代理的MPP：恶意图像补丁劫持多模式操作系统代理 cs.CR

NeurIPS 2025

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2503.10809v2) [paper-pdf](http://arxiv.org/pdf/2503.10809v2)

**Authors**: Lukas Aichberger, Alasdair Paren, Guohao Li, Philip Torr, Yarin Gal, Adel Bibi

**Abstract**: Recent advances in operating system (OS) agents have enabled vision-language models (VLMs) to directly control a user's computer. Unlike conventional VLMs that passively output text, OS agents autonomously perform computer-based tasks in response to a single user prompt. OS agents do so by capturing, parsing, and analysing screenshots and executing low-level actions via application programming interfaces (APIs), such as mouse clicks and keyboard inputs. This direct interaction with the OS significantly raises the stakes, as failures or manipulations can have immediate and tangible consequences. In this work, we uncover a novel attack vector against these OS agents: Malicious Image Patches (MIPs), adversarially perturbed screen regions that, when captured by an OS agent, induce it to perform harmful actions by exploiting specific APIs. For instance, a MIP can be embedded in a desktop wallpaper or shared on social media to cause an OS agent to exfiltrate sensitive user data. We show that MIPs generalise across user prompts and screen configurations, and that they can hijack multiple OS agents even during the execution of benign instructions. These findings expose critical security vulnerabilities in OS agents that have to be carefully addressed before their widespread deployment.

摘要: 操作系统（OS）代理的最新进展使视觉语言模型（VLM）能够直接控制用户的计算机。与被动输出文本的传统VLM不同，操作系统代理响应单个用户提示自主执行基于计算机的任务。操作系统代理通过捕获、解析和分析屏幕截图并通过应用程序编程接口（API）执行低级操作（例如鼠标单击和键盘输入）来实现这一目标。这种与操作系统的直接交互显着增加了风险，因为故障或操纵可能会产生立即且切实的后果。在这项工作中，我们发现了一种针对这些操作系统代理的新型攻击载体：恶意图像补丁（MIPs），这是一种受到不利干扰的屏幕区域，当被操作系统代理捕获时，会通过利用特定的API来诱导其执行有害操作。例如，MPP可以嵌入到桌面壁纸中或在社交媒体上共享，以导致OS代理泄露敏感用户数据。我们表明，MPP在用户提示和屏幕配置中普遍化，即使在执行良性指令期间，它们也可以劫持多个操作系统代理。这些发现暴露了操作系统代理中的关键安全漏洞，在广泛部署之前必须仔细解决这些漏洞。



## **40. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

AutoAdv：大型语言模型多回合越狱的自动对抗预算 cs.CL

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02376v1) [paper-pdf](http://arxiv.org/pdf/2511.02376v1)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs, yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves up to 95% attack success rate on Llama-3.1-8B within six turns a 24 percent improvement over single turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests then iteratively refines them. Extensive evaluation across commercial and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，其中对抗性提示会引发有害输出，但大多数评估都集中在单轮交互上，而现实世界的攻击则通过自适应多轮对话展开。我们介绍了AutoAdv，这是一个用于自动多回合越狱的免训练框架，在六个回合内对Llama-3.1-8B的攻击成功率高达95%，比单回合基线提高了24%。AutoAdv独特地结合了三种自适应机制：从成功的攻击中学习以增强未来提示的模式管理器、根据失败模式动态调整采样参数的温度管理器以及掩盖有害请求然后迭代细化它们的两阶段重写策略。对商业和开源模型（GPT-4 o-mini、Qwen 3 - 235 B、Mistral-7 B）的广泛评估揭示了当前安全机制中存在的持续漏洞，多回合攻击的表现始终优于单回合方法。这些发现表明，针对单轮交互优化的对齐策略无法在扩展对话中保持稳健性，凸显了对多轮感知防御的迫切需求。



## **41. SoK: Design, Vulnerabilities, and Security Measures of Cryptocurrency Wallets**

SoK：加密货币钱包的设计、漏洞和安全措施 cs.CR

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2307.12874v5) [paper-pdf](http://arxiv.org/pdf/2307.12874v5)

**Authors**: Yimika Erinle, Yathin Kethepalli, Yebo Feng, Jiahua Xu

**Abstract**: With the advent of decentralised digital currencies powered by blockchain technology, a new era of peer-to-peer transactions has commenced. The rapid growth of the cryptocurrency economy has led to increased use of transaction-enabling wallets, making them a focal point for security risks. As the frequency of wallet-related incidents rises, there is a critical need for a systematic approach to measure and evaluate these attacks, drawing lessons from past incidents to enhance wallet security. In response, we introduce a multi-dimensional design taxonomy for existing and novel wallets with various design decisions. We classify existing industry wallets based on this taxonomy, identify previously occurring vulnerabilities and discuss the security implications of design decisions. We also systematise threats to the wallet mechanism and analyse the adversary's goals, capabilities and required knowledge. We present a multi-layered attack framework and investigate 84 incidents between 2012 and 2024, accounting for $5.4B. Following this, we classify defence implementations for these attacks on the precautionary and remedial axes. We map the mechanism and design decisions to vulnerabilities, attacks, and possible defence methods to discuss various insights.

摘要: 随着区块链技术支持的去中心化数字货币的出现，点对点交易的新时代已经开始。加密货币经济的快速增长导致交易钱包的使用增加，使其成为安全风险的焦点。随着钱包相关事件频率的上升，迫切需要一种系统性的方法来衡量和评估这些攻击，从过去的事件中吸取教训以增强钱包安全性。作为回应，我们为现有的和新颖的钱包引入了多维设计分类法，具有各种设计决策。我们根据此分类法对现有的行业钱包进行分类，识别之前发生的漏洞并讨论设计决策的安全影响。我们还系统化对钱包机制的威胁，并分析对手的目标、能力和所需知识。我们提出了一个多层攻击框架，并调查了2012年至2024年间的84起事件，价值54亿美元。随后，我们将这些攻击的防御实施按预防和补救轴进行分类。我们将机制和设计决策映射到漏洞、攻击和可能的防御方法，以讨论各种见解。



## **42. Co-Evolving Complexity: An Adversarial Framework for Automatic MARL Curricula**

共同进化的复杂性：自动MARL课程的对抗框架 cs.LG

Published in the proceedings of the 39th Conference on Neural  Information Processing Systems (NeurIPS 2025) Workshop: Scaling Environments  for Agents (SEA)

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2509.03771v3) [paper-pdf](http://arxiv.org/pdf/2509.03771v3)

**Authors**: Brennen Hill

**Abstract**: The advancement of general-purpose intelligent agents is intrinsically linked to the environments in which they are trained. While scaling models and datasets has yielded remarkable capabilities, scaling the complexity, diversity, and interactivity of environments remains a crucial bottleneck. Hand-crafted environments are finite and often contain implicit biases, limiting the potential for agents to develop truly generalizable and robust skills. In this work, we propose a paradigm for generating a boundless and adaptive curriculum of challenges by framing the environment generation process as an adversarial game. We introduce a system where a team of cooperative multi-agent defenders learns to survive against a procedurally generative attacker. The attacker agent learns to produce increasingly challenging configurations of enemy units, dynamically creating novel worlds tailored to exploit the defenders' current weaknesses. Concurrently, the defender team learns cooperative strategies to overcome these generated threats. This co-evolutionary dynamic creates a self-scaling environment where complexity arises organically from the adversarial interaction, providing an effectively infinite stream of novel and relevant training data. We demonstrate that with minimal training, this approach leads to the emergence of complex, intelligent behaviors, such as flanking and shielding by the attacker, and focus-fire and spreading by the defenders. Our findings suggest that adversarial co-evolution is a powerful mechanism for automatically scaling environmental complexity, driving agents towards greater robustness and strategic depth.

摘要: 通用智能代理的进步与它们接受训练的环境有着内在的联系。虽然扩展模型和数据集已经产生了非凡的能力，但扩展环境的复杂性、多样性和交互性仍然是一个关键的瓶颈。手工制作的环境是有限的，并且通常包含隐性偏见，限制了代理人开发真正可推广和稳健技能的潜力。在这项工作中，我们提出了一种范式，通过将环境生成过程构建为对抗游戏，生成无限且适应性的挑战课程。我们介绍了一个系统，一个团队的合作多智能体防御者学会生存对程序生成攻击者。攻击者智能体学习生成越来越具有挑战性的敌方单位配置，动态地创建新的世界，以利用防御者当前的弱点。同时，防御者团队学习合作策略来克服这些产生的威胁。这种共同进化的动态创造了一个自缩放的环境，在这个环境中，复杂性从对抗性的交互中有机地产生，提供了一个有效的无限的新的相关训练数据流。我们证明，只需最少的训练，这种方法就会导致复杂、智能的行为的出现，例如攻击者的侧翼和掩护，以及防御者的聚焦射击和传播。我们的研究结果表明，对抗性协同进化是自动扩展环境复杂性的一种强大机制，推动代理人走向更大的稳健性和战略深度。



## **43. Machine and Deep Learning for Indoor UWB Jammer Localization**

用于室内超宽带干扰机定位的机器和深度学习 cs.LG

Accepted at the 20th International Conference on Risks and Security  of Internet and Systems (CRiSIS 2025, Gatineau-Canada,  https://crisis2025.uqo.ca/). The paper will soon be published as  post-proceedings in Springer's LNCS

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01819v1) [paper-pdf](http://arxiv.org/pdf/2511.01819v1)

**Authors**: Hamed Fard, Mahsa Kholghi, Benedikt Groß, Gerhard Wunder

**Abstract**: Ultra-wideband (UWB) localization delivers centimeter-scale accuracy but is vulnerable to jamming attacks, creating security risks for asset tracking and intrusion detection in smart buildings. Although machine learning (ML) and deep learning (DL) methods have improved tag localization, localizing malicious jammers within a single room and across changing indoor layouts remains largely unexplored. Two novel UWB datasets, collected under original and modified room configurations, are introduced to establish comprehensive ML/DL baselines. Performance is rigorously evaluated using a variety of classification and regression metrics. On the source dataset with the collected UWB features, Random Forest achieves the highest F1-macro score of 0.95 and XGBoost achieves the lowest mean Euclidean error of 20.16 cm. However, deploying these source-trained models in the modified room layout led to severe performance degradation, with XGBoost's mean Euclidean error increasing tenfold to 207.99 cm, demonstrating significant domain shift. To mitigate this degradation, a domain-adversarial ConvNeXt autoencoder (A-CNT) is proposed that leverages a gradient-reversal layer to align CIR-derived features across domains. The A-CNT framework restores localization performance by reducing the mean Euclidean error to 34.67 cm. This represents a 77 percent improvement over non-adversarial transfer learning and an 83 percent improvement over the best baseline, restoring the fraction of samples within 30 cm to 0.56. Overall, the results demonstrate that adversarial feature alignment enables robust and transferable indoor jammer localization despite environmental changes. Code and dataset available at https://github.com/afbf4c8996f/Jammer-Loc

摘要: 超宽带（UWB）定位可提供厘米级的准确度，但很容易受到干扰攻击，从而给智能建筑中的资产跟踪和入侵检测带来安全风险。尽管机器学习（ML）和深度学习（DL）方法改进了标签本地化，但在单个房间内和不断变化的室内布局中本地化恶意干扰器在很大程度上仍然没有被探索。引入了在原始和修改后的房间配置下收集的两个新颖的UWB数据集，以建立全面的ML/DL基线。使用各种分类和回归指标对性能进行严格评估。在具有收集的UWB特征的源数据集上，Random Forest获得了最高的F1宏评分，为0.95，XGboost获得了最低的平均欧几里得误差，为20.16厘米。然而，在修改后的房间布局中部署这些源训练模型导致性能严重下降，XGBoost的平均欧几里得误差增加了10倍，达到207.99厘米，表明域发生了显着的变化。为了减轻这种退化，提出了一种域对抗性ConvNeXt自动编码器（A-NT），它利用梯度逆转层来在域中对齐CIR-派生的特征。A-TNT框架通过将平均欧几里得误差降低到34.67厘米来恢复本地化性能。这比非对抗性迁移学习提高了77%，比最佳基线提高了83%，将30厘米内的样本比例恢复到0.56。总体而言，结果表明，尽管环境发生变化，对抗性特征对齐仍然可以实现稳健且可转移的室内干扰器定位。代码和数据集可访问https://github.com/afbf4c8996f/Jammer-Loc



## **44. Scam Shield: Multi-Model Voting and Fine-Tuned LLMs Against Adversarial Attacks**

骗局盾牌：针对对抗性攻击的多模型投票和微调LLM cs.CR

8 pages

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01746v1) [paper-pdf](http://arxiv.org/pdf/2511.01746v1)

**Authors**: Chen-Wei Chang, Shailik Sarkar, Hossein Salemi, Hyungmin Kim, Shutonu Mitra, Hemant Purohit, Fengxiu Zhang, Michin Hong, Jin-Hee Cho, Chang-Tien Lu

**Abstract**: Scam detection remains a critical challenge in cybersecurity as adversaries craft messages that evade automated filters. We propose a Hierarchical Scam Detection System (HSDS) that combines a lightweight multi-model voting front end with a fine-tuned LLaMA 3.1 8B Instruct back end to improve accuracy and robustness against adversarial attacks. An ensemble of four classifiers provides preliminary predictions through majority vote, and ambiguous cases are escalated to the fine-tuned model, which is optimized with adversarial training to reduce misclassification. Experiments show that this hierarchical design both improves adversarial scam detection and shortens inference time by routing most cases away from the LLM, outperforming traditional machine-learning baselines and proprietary LLM baselines. The findings highlight the effectiveness of a hybrid voting mechanism and adversarial fine-tuning in fortifying LLMs against evolving scam tactics, enhancing the resilience of automated scam detection systems.

摘要: 诈骗检测仍然是网络安全的一个关键挑战，因为对手制作的消息可以逃避自动过滤器。我们提出了一种分层诈骗检测系统（HADS），该系统将轻量级多模型投票前端与微调的LLaMA 3.1 8B Direcct后台相结合，以提高针对对抗性攻击的准确性和鲁棒性。四个分类器的集合通过多数投票提供初步预测，模糊的案例被升级到微调模型，该模型通过对抗训练进行优化，以减少错误分类。实验表明，这种分层设计通过将大多数案例路由远离LLM，既改善了对抗性骗局检测，又缩短了推理时间，优于传统的机器学习基线和专有的LLM基线。研究结果强调了混合投票机制和对抗性微调在加强LLM对抗不断变化的诈骗策略方面的有效性，增强了自动化诈骗检测系统的弹性。



## **45. Black-Box Membership Inference Attack for LVLMs via Prior Knowledge-Calibrated Memory Probing**

通过先验知识校准内存探测对LVLM进行黑匣子成员资格推断攻击 cs.CR

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01952v1) [paper-pdf](http://arxiv.org/pdf/2511.01952v1)

**Authors**: Jinhua Yin, Peiru Yang, Chen Yang, Huili Wang, Zhiyang Hu, Shangguang Wang, Yongfeng Huang, Tao Qi

**Abstract**: Large vision-language models (LVLMs) derive their capabilities from extensive training on vast corpora of visual and textual data. Empowered by large-scale parameters, these models often exhibit strong memorization of their training data, rendering them susceptible to membership inference attacks (MIAs). Existing MIA methods for LVLMs typically operate under white- or gray-box assumptions, by extracting likelihood-based features for the suspected data samples based on the target LVLMs. However, mainstream LVLMs generally only expose generated outputs while concealing internal computational features during inference, limiting the applicability of these methods. In this work, we propose the first black-box MIA framework for LVLMs, based on a prior knowledge-calibrated memory probing mechanism. The core idea is to assess the model memorization of the private semantic information embedded within the suspected image data, which is unlikely to be inferred from general world knowledge alone. We conducted extensive experiments across four LVLMs and three datasets. Empirical results demonstrate that our method effectively identifies training data of LVLMs in a purely black-box setting and even achieves performance comparable to gray-box and white-box methods. Further analysis reveals the robustness of our method against potential adversarial manipulations, and the effectiveness of the methodology designs. Our code and data are available at https://github.com/spmede/KCMP.

摘要: 大型视觉语言模型（LVLM）的能力源自对大量视觉和文本数据库的广泛培训。这些模型在大规模参数的支持下，通常表现出对其训练数据的强大记忆力，使其容易受到隶属推理攻击（MIA）。现有的LVLM MIA方法通常在白盒或灰盒假设下运行，通过基于目标LVLM为可疑数据样本提取基于似然的特征。然而，主流LVLM通常只暴露生成的输出，同时在推理过程中隐藏内部计算特征，从而限制了这些方法的适用性。在这项工作中，我们提出了第一个黑盒MIA框架LVLM，基于先验知识校准的内存探测机制。其核心思想是评估模型记忆的私人语义信息嵌入在可疑的图像数据，这是不可能推断出的一般世界知识。我们在四个LVLM和三个数据集上进行了广泛的实验。实验结果表明，我们的方法可以有效地识别LVLM的训练数据在一个纯粹的黑盒设置，甚至达到性能相媲美的灰盒和白盒方法。进一步的分析揭示了我们的方法对潜在的对抗性操纵的稳健性，以及方法论设计的有效性。我们的代码和数据可在https://github.com/spmede/KCMP上获取。



## **46. SecDiff: Diffusion-Aided Secure Deep Joint Source-Channel Coding Against Adversarial Attacks**

SecDiff：针对对抗攻击的扩散辅助安全深度联合源通道编码 cs.CV

13 pages, 6 figures

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01466v1) [paper-pdf](http://arxiv.org/pdf/2511.01466v1)

**Authors**: Changyuan Zhao, Jiacheng Wang, Ruichen Zhang, Dusit Niyato, Hongyang Du, Zehui Xiong, Dong In Kim, Ping Zhang

**Abstract**: Deep joint source-channel coding (JSCC) has emerged as a promising paradigm for semantic communication, delivering significant performance gains over conventional separate coding schemes. However, existing JSCC frameworks remain vulnerable to physical-layer adversarial threats, such as pilot spoofing and subcarrier jamming, compromising semantic fidelity. In this paper, we propose SecDiff, a plug-and-play, diffusion-aided decoding framework that significantly enhances the security and robustness of deep JSCC under adversarial wireless environments. Different from prior diffusion-guided JSCC methods that suffer from high inference latency, SecDiff employs pseudoinverse-guided sampling and adaptive guidance weighting, enabling flexible step-size control and efficient semantic reconstruction. To counter jamming attacks, we introduce a power-based subcarrier masking strategy and recast recovery as a masked inpainting problem, solved via diffusion guidance. For pilot spoofing, we formulate channel estimation as a blind inverse problem and develop an expectation-minimization (EM)-driven reconstruction algorithm, guided jointly by reconstruction loss and a channel operator. Notably, our method alternates between pilot recovery and channel estimation, enabling joint refinement of both variables throughout the diffusion process. Extensive experiments over orthogonal frequency-division multiplexing (OFDM) channels under adversarial conditions show that SecDiff outperforms existing secure and generative JSCC baselines by achieving a favorable trade-off between reconstruction quality and computational cost. This balance makes SecDiff a promising step toward practical, low-latency, and attack-resilient semantic communications.

摘要: 深度联合信源信道编码（JSCC）已经成为语义通信的一个有前途的范例，与传统的单独编码方案相比，它提供了显着的性能增益。然而，现有的JSCC框架仍然容易受到物理层对抗性威胁的影响，例如导频欺骗和子载波干扰，从而影响语义保真度。在本文中，我们提出了SecDiff，这是一种即插即用的扩散辅助解码框架，可显著增强深度JSCC在对抗性无线环境下的安全性和鲁棒性。与之前的扩散引导JSCC方法存在高推理延迟的问题不同，SecDiff采用伪逆引导采样和自适应引导加权，实现了灵活的步长控制和高效的语义重建。为了对抗干扰攻击，我们引入了一种基于功率的子帧掩蔽策略，并将恢复重新创建为掩蔽修复问题，通过扩散引导来解决。对于导频欺骗，我们将信道估计公式化为盲逆问题，并开发了一种期望最小化（EM）驱动的重建算法，由重建损失和信道运营商共同指导。值得注意的是，我们的方法在导频恢复和信道估计之间交替，从而能够在整个扩散过程中联合细化这两个变量。对抗条件下在垂直频分多路传输（CDMA）通道上进行的大量实验表明，SecDiff通过在重建质量和计算成本之间实现有利的权衡，优于现有的安全和生成性JSCC基线。这种平衡使SecDiff朝着实用、低延迟和抗攻击的语义通信迈出了有希望的一步。



## **47. Protecting the Neural Networks against FGSM Attack Using Machine Unlearning**

使用机器去学习保护神经网络免受FGSM攻击 cs.LG

7 pages, 9 figures, 1 table

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01377v1) [paper-pdf](http://arxiv.org/pdf/2511.01377v1)

**Authors**: Amir Hossein Khorasani, Ali Jahanian, Maryam Rastgarpour

**Abstract**: Machine learning is a powerful tool for building predictive models. However, it is vulnerable to adversarial attacks. Fast Gradient Sign Method (FGSM) attacks are a common type of adversarial attack that adds small perturbations to input data to trick a model into misclassifying it. In response to these attacks, researchers have developed methods for "unlearning" these attacks, which involves retraining a model on the original data without the added perturbations. Machine unlearning is a technique that tries to "forget" specific data points from the training dataset, to improve the robustness of a machine learning model against adversarial attacks like FGSM. In this paper, we focus on applying unlearning techniques to the LeNet neural network, a popular architecture for image classification. We evaluate the efficacy of unlearning FGSM attacks on the LeNet network and find that it can significantly improve its robustness against these types of attacks.

摘要: 机器学习是构建预测模型的强大工具。然而，它很容易受到敌对攻击。快速梯度符号法（FGSM）攻击是一种常见的对抗性攻击，它向输入数据添加小扰动，以诱骗模型对其进行错误分类。为了响应这些攻击，研究人员开发了“消除学习”这些攻击的方法，其中涉及在没有添加扰动的情况下重新训练原始数据的模型。机器去学习是一种试图从训练数据集中“忘记”特定数据点的技术，以提高机器学习模型针对FGSM等对抗性攻击的鲁棒性。在本文中，我们重点关注将去学习技术应用于LeNet神经网络，这是一种流行的图像分类架构。我们评估了消除LeNet网络上FGSM攻击的有效性，发现它可以显着提高其针对这些类型攻击的鲁棒性。



## **48. Align to Misalign: Automatic LLM Jailbreak with Meta-Optimized LLM Judges**

对齐与错位：通过元优化的LLM评委自动LLM越狱 cs.AI

under review, 28 pages

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01375v1) [paper-pdf](http://arxiv.org/pdf/2511.01375v1)

**Authors**: Hamin Koo, Minseon Kim, Jaehyung Kim

**Abstract**: Identifying the vulnerabilities of large language models (LLMs) is crucial for improving their safety by addressing inherent weaknesses. Jailbreaks, in which adversaries bypass safeguards with crafted input prompts, play a central role in red-teaming by probing LLMs to elicit unintended or unsafe behaviors. Recent optimization-based jailbreak approaches iteratively refine attack prompts by leveraging LLMs. However, they often rely heavily on either binary attack success rate (ASR) signals, which are sparse, or manually crafted scoring templates, which introduce human bias and uncertainty in the scoring outcomes. To address these limitations, we introduce AMIS (Align to MISalign), a meta-optimization framework that jointly evolves jailbreak prompts and scoring templates through a bi-level structure. In the inner loop, prompts are refined using fine-grained and dense feedback using a fixed scoring template. In the outer loop, the template is optimized using an ASR alignment score, gradually evolving to better reflect true attack outcomes across queries. This co-optimization process yields progressively stronger jailbreak prompts and more calibrated scoring signals. Evaluations on AdvBench and JBB-Behaviors demonstrate that AMIS achieves state-of-the-art performance, including 88.0% ASR on Claude-3.5-Haiku and 100.0% ASR on Claude-4-Sonnet, outperforming existing baselines by substantial margins.

摘要: 识别大型语言模型（LLM）的漏洞对于通过解决固有弱点来提高其安全性至关重要。越狱是指对手通过精心设计的输入提示绕过保障措施，通过探测LLM以引发意外或不安全的行为，在红色团队中发挥着核心作用。最近的基于优化的越狱方法通过利用LLM迭代地完善攻击提示。然而，他们通常严重依赖稀疏的二元攻击成功率（ASB）信号或手动制作的评分模板，这在评分结果中引入了人为偏见和不确定性。为了解决这些限制，我们引入了AMIS（Align to MISign），这是一个元优化框架，通过两级结构联合开发越狱提示和评分模板。在内循环中，使用固定评分模板使用细粒度且密集的反馈来细化提示。在外循环中，使用ASB对齐分数对模板进行优化，逐渐发展以更好地反映跨查询的真实攻击结果。这个协同优化过程会产生越来越强的越狱提示和更校准的评分信号。对AdvBench和JBB-Behavior的评估表明，AMIS实现了最先进的性能，包括Claude-3.5-Haiku的ASB为88.0%，Claude-4-Sonnet的ASB为100.0%，大幅优于现有基线。



## **49. On the Classical Hardness of the Semidirect Discrete Logarithm Problem in Finite Groups**

有限群中半直接离散对数问题的经典硬度 cs.CR

v2: Camera-ready version for Indocrypt 2025. Incorporated reviewer  feedback: simplified proofs, made computational assumptions explicit, fixed  technical errors

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2508.05048v2) [paper-pdf](http://arxiv.org/pdf/2508.05048v2)

**Authors**: Mohammad Ferry Husnil Arif, Muhammad Imran

**Abstract**: The semidirect discrete logarithm problem (SDLP) in finite groups was proposed as a foundation for post-quantum cryptographic protocols, based on the belief that its non-abelian structure would resist quantum attacks. However, recent results have shown that SDLP in finite groups admits efficient quantum algorithms, undermining its quantum resistance. This raises a fundamental question: does the SDLP offer any computational advantages over the standard discrete logarithm problem (DLP) against classical adversaries? In this work, we investigate the classical hardness of SDLP across different finite group platforms. We establish that the group-case SDLP can be reformulated as a generalized discrete logarithm problem, enabling adaptation of classical algorithms to study its complexity. We present a concrete adaptation of the Baby-Step Giant-Step algorithm for SDLP, achieving time and space complexity $O(\sqrt{r})$ where $r$ is the period of the underlying cycle structure. Through theoretical analysis and experimental validation in SageMath, we demonstrate that the classical hardness of SDLP is highly platform-dependent and does not uniformly exceed that of standard DLP. In finite fields $\mathbb{F}_p^*$, both problems exhibit comparable complexity. Surprisingly, in elliptic curves $E(\mathbb{F}_p)$, the SDLP becomes trivial due to the bounded automorphism group, while in elementary abelian groups $\mathbb{F}_p^n$, the SDLP can be harder than DLP, with complexity varying based on the eigenvalue structure of the automorphism. Our findings reveal that the non-abelian structure of semidirect products does not inherently guarantee increased classical hardness, suggesting that the search for classically hard problems for cryptographic applications requires more careful consideration of the underlying algebraic structures.

摘要: 有限群中的半直接离散log问题（SDLP）被提出作为后量子密码协议的基础，基于其非阿贝尔结构将抵抗量子攻击的信念。然而，最近的结果表明，有限群中的SDLP允许高效的量子算法，从而削弱了其量子阻力。这提出了一个基本问题：与经典对手相比，SDLP是否提供了比标准离散log问题（DLC）的任何计算优势？在这项工作中，我们研究了不同有限群平台上SDLP的经典硬度。我们确定，群情况SDLP可以被重新表述为广义离散log问题，从而能够适应经典算法来研究其复杂性。我们针对SDLP提出了Baby-Step Giant-Step算法的具体改编，实现时间和空间复杂性$O（\SQRT{r}）$，其中$r$是基础周期结构的周期。通过SageMath中的理论分析和实验验证，我们证明SDLP的经典硬度高度依赖于平台，并且不会均匀超过标准DLP的硬度。在有限域$\mathbb{F}_p^*$中，这两个问题都表现出相当的复杂性。令人惊讶的是，在椭圆曲线$E（\mathbb{F}_p）$中，SDLP由于有界自同射群而变得平凡，而在基本阿贝尔群$\mathbb{F}_p ' n$中，SDLP可能比DLP更难，复杂性根据自同射的特征值结构而异。我们的研究结果表明，半直积的非阿贝尔结构本质上并不能保证经典难度的增加，这表明寻找密码应用的经典困难问题需要更仔细地考虑底层的代数结构。



## **50. MiniFool -- Physics-Constraint-Aware Minimizer-Based Adversarial Attacks in Deep Neural Networks**

MiniFool --深度神经网络中基于物理约束的最小化器的对抗性攻击 cs.LG

Submitted to Computing and Software for Big Science

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01352v1) [paper-pdf](http://arxiv.org/pdf/2511.01352v1)

**Authors**: Lucie Flek, Oliver Janik, Philipp Alexander Jung, Akbar Karimi, Timo Saala, Alexander Schmidt, Matthias Schott, Philipp Soldin, Matthias Thiesmeyer, Christopher Wiebusch, Ulrich Willemsen

**Abstract**: In this paper, we present a new algorithm, MiniFool, that implements physics-inspired adversarial attacks for testing neural network-based classification tasks in particle and astroparticle physics. While we initially developed the algorithm for the search for astrophysical tau neutrinos with the IceCube Neutrino Observatory, we apply it to further data from other science domains, thus demonstrating its general applicability. Here, we apply the algorithm to the well-known MNIST data set and furthermore, to Open Data data from the CMS experiment at the Large Hadron Collider. The algorithm is based on minimizing a cost function that combines a $\chi^2$ based test-statistic with the deviation from the desired target score. The test statistic quantifies the probability of the perturbations applied to the data based on the experimental uncertainties. For our studied use cases, we find that the likelihood of a flipped classification differs for both the initially correctly and incorrectly classified events. When testing changes of the classifications as a function of an attack parameter that scales the experimental uncertainties, the robustness of the network decision can be quantified. Furthermore, this allows testing the robustness of the classification of unlabeled experimental data.

摘要: 在本文中，我们提出了一种新的算法MiniFool，该算法实现了物理启发的对抗性攻击，用于测试粒子和天体粒子物理学中基于神经网络的分类任务。虽然我们最初开发的算法与冰立方中微子天文台搜索天体物理τ中微子，我们将其应用到其他科学领域的进一步数据，从而证明其普遍适用性。在这里，我们将该算法应用于著名的MNIST数据集，并进一步从CMS实验在大型强子对撞机的开放数据。该算法基于最小化成本函数，该成本函数将基于$\chi^2 $的测试统计量与所需目标分数的偏差相结合。测试统计量根据实验不确定性量化应用于数据的扰动的概率。对于我们研究的用例，我们发现，对于最初正确分类和错误分类的事件，翻转分类的可能性有所不同。当测试作为扩展实验不确定性的攻击参数的函数的分类变化时，可以量化网络决策的稳健性。此外，这允许测试未标记实验数据分类的稳健性。



