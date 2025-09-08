# Latest Adversarial Attack Papers
**update at 2025-09-08 17:25:12**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. On Evaluating the Poisoning Robustness of Federated Learning under Local Differential Privacy**

局部差异隐私下联邦学习中毒鲁棒性评估 cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05265v1) [paper-pdf](http://arxiv.org/pdf/2509.05265v1)

**Authors**: Zijian Wang, Wei Tong, Tingxuan Han, Haoyu Chen, Tianling Zhang, Yunlong Mao, Sheng Zhong

**Abstract**: Federated learning (FL) combined with local differential privacy (LDP) enables privacy-preserving model training across decentralized data sources. However, the decentralized data-management paradigm leaves LDPFL vulnerable to participants with malicious intent. The robustness of LDPFL protocols, particularly against model poisoning attacks (MPA), where adversaries inject malicious updates to disrupt global model convergence, remains insufficiently studied. In this paper, we propose a novel and extensible model poisoning attack framework tailored for LDPFL settings. Our approach is driven by the objective of maximizing the global training loss while adhering to local privacy constraints. To counter robust aggregation mechanisms such as Multi-Krum and trimmed mean, we develop adaptive attacks that embed carefully crafted constraints into a reverse training process, enabling evasion of these defenses. We evaluate our framework across three representative LDPFL protocols, three benchmark datasets, and two types of deep neural networks. Additionally, we investigate the influence of data heterogeneity and privacy budgets on attack effectiveness. Experimental results demonstrate that our adaptive attacks can significantly degrade the performance of the global model, revealing critical vulnerabilities and highlighting the need for more robust LDPFL defense strategies against MPA. Our code is available at https://github.com/ZiJW/LDPFL-Attack

摘要: 联合学习（FL）与局部差分隐私（LDP）相结合，可以在分散的数据源中进行隐私保护模型训练。然而，分散的数据管理模式使LDPFL容易受到恶意参与者的攻击。LDPFL协议的鲁棒性，特别是对模型中毒攻击（MPA），其中对手注入恶意更新破坏全局模型收敛，仍然没有得到充分的研究。在本文中，我们提出了一种新的和可扩展的模型中毒攻击框架，为LDPFL设置量身定制。我们的方法是由最大限度地提高全球培训损失，同时坚持本地隐私约束的目标。为了对抗强大的聚合机制，如Multi-Krum和Trimmed Mean，我们开发了自适应攻击，将精心制作的约束嵌入到反向训练过程中，从而能够规避这些防御。我们在三个代表性的LDPFL协议，三个基准数据集和两种类型的深度神经网络上评估了我们的框架。此外，我们调查的数据异质性和隐私预算对攻击效果的影响。实验结果表明，我们的自适应攻击可以显着降低全局模型的性能，揭示关键漏洞，并强调需要更强大的LDPFL防御策略对MPA。我们的代码可在https://github.com/ZiJW/LDPFL-Attack上获取



## **2. On Hyperparameters and Backdoor-Resistance in Horizontal Federated Learning**

水平联邦学习中的超参数和后门抵抗 cs.CR

To appear in the Proceedings of the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05192v1) [paper-pdf](http://arxiv.org/pdf/2509.05192v1)

**Authors**: Simon Lachnit, Ghassan Karame

**Abstract**: Horizontal Federated Learning (HFL) is particularly vulnerable to backdoor attacks as adversaries can easily manipulate both the training data and processes to execute sophisticated attacks. In this work, we study the impact of training hyperparameters on the effectiveness of backdoor attacks and defenses in HFL. More specifically, we show both analytically and by means of measurements that the choice of hyperparameters by benign clients does not only influence model accuracy but also significantly impacts backdoor attack success. This stands in sharp contrast with the multitude of contributions in the area of HFL security, which often rely on custom ad-hoc hyperparameter choices for benign clients$\unicode{x2013}$leading to more pronounced backdoor attack strength and diminished impact of defenses. Our results indicate that properly tuning benign clients' hyperparameters$\unicode{x2013}$such as learning rate, batch size, and number of local epochs$\unicode{x2013}$can significantly curb the effectiveness of backdoor attacks, regardless of the malicious clients' settings. We support this claim with an extensive robustness evaluation of state-of-the-art attack-defense combinations, showing that carefully chosen hyperparameters yield across-the-board improvements in robustness without sacrificing main task accuracy. For example, we show that the 50%-lifespan of the strong A3FL attack can be reduced by 98.6%, respectively$\unicode{x2013}$all without using any defense and while incurring only a 2.9 percentage points drop in clean task accuracy.

摘要: 水平联邦学习（HFL）特别容易受到后门攻击，因为对手可以轻松操纵训练数据和流程来执行复杂的攻击。在这项工作中，我们研究了训练超参数对HFL中后门攻击和防御有效性的影响。更具体地说，我们通过分析和测量表明，良性客户端对超参数的选择不仅会影响模型准确性，还会显着影响后门攻击的成功。这与HFL安全领域的众多贡献形成鲜明对比，HFL安全领域通常依赖于良性客户端$\unicode{x2013}$的自定义临时超参数选择，导致后门攻击强度更明显，防御影响减弱。我们的结果表明，正确调整良性客户端的超参数$\unicode{x2013}$，例如学习率、批量大小和本地纪元数量$\unicode{x2013}$可以显着抑制后门攻击的有效性，无论恶意客户端的设置如何。我们通过对最先进的攻击-防御组合的广泛鲁棒性评估来支持这一说法，表明精心选择的超参数可以在不牺牲主要任务准确性的情况下全面提高鲁棒性。例如，我们表明，在不使用任何防御的情况下，强A3 FL攻击的50%寿命可以分别缩短98.6%$\unicode{x2013}$all，同时仅导致干净任务准确性下降2.9个百分点。



## **3. Jamming Smarter, Not Harder: Exploiting O-RAN Y1 RAN Analytics for Efficient Interference**

更智能、而不是更难干扰：利用O-RAN Y1 RAN分析实现高效干扰 cs.CR

8 pages, 7 figures

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05161v1) [paper-pdf](http://arxiv.org/pdf/2509.05161v1)

**Authors**: Abiodun Ganiyu, Dara Ron, Syed Rafiul Hussain, Vijay K Shah

**Abstract**: The Y1 interface in O-RAN enables the sharing of RAN Analytics Information (RAI) between the near-RT RIC and authorized Y1 consumers, which may be internal applications within the operator's trusted domain or external systems accessing data through a secure exposure function. While this visibility enhances network optimization and enables advanced services, it also introduces a potential security risk -- a malicious or compromised Y1 consumer could misuse analytics to facilitate targeted interference. In this work, we demonstrate how an adversary can exploit the Y1 interface to launch selective jamming attacks by passively monitoring downlink metrics. We propose and evaluate two Y1-aided jamming strategies: a clustering-based jammer leveraging DBSCAN for traffic profiling and a threshold-based jammer. These are compared against two baselines strategies -- always-on jammer and random jammer -- on an over-the-air LTE/5G O-RAN testbed. Experimental results show that in unconstrained jamming budget scenarios, the threshold-based jammer can closely replicate the disruption caused by always-on jamming while reducing transmission time by 27\%. Under constrained jamming budgets, the clustering-based jammer proves most effective, causing up to an 18.1\% bitrate drop while remaining active only 25\% of the time. These findings reveal a critical trade-off between jamming stealthiness and efficiency, and illustrate how exposure of RAN analytics via the Y1 interface can enable highly targeted, low-overhead attacks, raising important security considerations for both civilian and mission-critical O-RAN deployments.

摘要: O-RAN中的Y1接口支持近RT RIC和授权Y1消费者之间共享RAN分析信息（RAI），这些消费者可以是运营商可信域内的内部应用程序，也可以是通过安全暴露功能访问数据的外部系统。虽然这种可见性增强了网络优化并支持高级服务，但它也带来了潜在的安全风险--恶意或受损害的Y1消费者可能会滥用分析来促进有针对性的干扰。在这项工作中，我们演示了对手如何利用Y1接口通过被动监视下行链路指标来发起选择性干扰攻击。我们提出并评估了两种Y1辅助干扰策略：利用DBSCAN进行流量分析的基于集群的干扰器和基于阈值的干扰器。将这些与空中LTE/5G O-RAN测试床上的两种基线策略（始终开启的干扰器和随机干扰器）进行比较。实验结果表明，在不受限制的干扰预算场景下，基于阈值的干扰器可以精确地复制永远在线干扰造成的干扰，同时将传输时间缩短27%。在受限制的干扰预算下，基于集群的干扰器被证明是最有效的，导致高达18.1%的比特率下降，而只有25%的时间保持活跃。这些发现揭示了干扰隐蔽性和效率之间的关键权衡，并说明了通过Y1接口暴露的RAN分析如何能够实现高度针对性、低开销的攻击，从而为民用和关键任务O-RAN部署提出了重要的安全考虑。



## **4. Robust Experts: the Effect of Adversarial Training on CNNs with Sparse Mixture-of-Experts Layers**

稳健的专家：对抗训练对具有稀疏专家混合层的CNN的影响 cs.CV

Accepted for publication at the STREAM workshop at ICCV 2025

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05086v1) [paper-pdf](http://arxiv.org/pdf/2509.05086v1)

**Authors**: Svetlana Pavlitska, Haixi Fan, Konstantin Ditschuneit, J. Marius Zöllner

**Abstract**: Robustifying convolutional neural networks (CNNs) against adversarial attacks remains challenging and often requires resource-intensive countermeasures. We explore the use of sparse mixture-of-experts (MoE) layers to improve robustness by replacing selected residual blocks or convolutional layers, thereby increasing model capacity without additional inference cost. On ResNet architectures trained on CIFAR-100, we find that inserting a single MoE layer in the deeper stages leads to consistent improvements in robustness under PGD and AutoPGD attacks when combined with adversarial training. Furthermore, we discover that when switch loss is used for balancing, it causes routing to collapse onto a small set of overused experts, thereby concentrating adversarial training on these paths and inadvertently making them more robust. As a result, some individual experts outperform the gated MoE model in robustness, suggesting that robust subpaths emerge through specialization. Our code is available at https://github.com/KASTEL-MobilityLab/robust-sparse-moes.

摘要: 增强卷积神经网络（CNN）对抗对抗性攻击仍然具有挑战性，并且通常需要资源密集型的应对措施。我们探索使用稀疏混合专家（MoE）层，通过替换选定的残差块或卷积层来提高鲁棒性，从而在不增加额外推理成本的情况下提高模型容量。在CIFAR-100上训练的ResNet架构上，我们发现，在更深的阶段中插入单个MoE层，当与对抗训练相结合时，可以在PGD和AutoPGD攻击下实现一致的鲁棒性改进。此外，我们发现，当使用交换机损失来平衡时，会导致路由崩溃到一小群过度使用的专家身上，从而将对抗训练集中在这些路径上，并无意中使它们更加稳健。因此，一些个人专家在稳健性方面优于门控MoE模型，这表明通过专业化出现了稳健的子路径。我们的代码可在https://github.com/KASTEL-MobilityLab/robust-sparse-moes上获取。



## **5. Adversarial Augmentation and Active Sampling for Robust Cyber Anomaly Detection**

对抗增强和主动采样用于鲁棒网络异常检测 cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04999v1) [paper-pdf](http://arxiv.org/pdf/2509.04999v1)

**Authors**: Sidahmed Benabderrahmane, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) present a considerable challenge to cybersecurity due to their stealthy, long-duration nature. Traditional supervised learning methods typically require large amounts of labeled data, which is often scarce in real-world scenarios. This paper introduces a novel approach that combines AutoEncoders for anomaly detection with active learning to iteratively enhance APT detection. By selectively querying an oracle for labels on uncertain or ambiguous samples, our method reduces labeling costs while improving detection accuracy, enabling the model to effectively learn with minimal data and reduce reliance on extensive manual labeling. We present a comprehensive formulation of the Attention Adversarial Dual AutoEncoder-based anomaly detection framework and demonstrate how the active learning loop progressively enhances the model's performance. The framework is evaluated on real-world, imbalanced provenance trace data from the DARPA Transparent Computing program, where APT-like attacks account for just 0.004\% of the data. The datasets, which cover multiple operating systems including Android, Linux, BSD, and Windows, are tested in two attack scenarios. The results show substantial improvements in detection rates during active learning, outperforming existing methods.

摘要: 高级持续性威胁（APT）由于其隐蔽性、持续时间长，对网络安全构成了相当大的挑战。传统的监督学习方法通常需要大量的标记数据，而这些数据在现实世界场景中通常很稀缺。本文介绍了一种新颖的方法，将用于异常检测的AutoEncoders与主动学习相结合，以迭代增强APT检测。通过选择性地向Oracle查询不确定或模糊样本的标签，我们的方法降低了标签成本，同时提高了检测准确性，使模型能够用最少的数据有效学习并减少对大量手动标签的依赖。我们提出了基于注意力对抗双AutoEnCoder的异常检测框架的全面公式，并演示了主动学习循环如何逐步增强模型的性能。该框架是根据DARPA透明计算程序中的现实世界、不平衡的出处跟踪数据进行评估的，其中类APT攻击仅占数据的0.004%。这些数据集涵盖Android、Linux、BDS和Windows等多个操作系统，并在两种攻击场景中进行了测试。结果显示，主动学习期间的检测率大幅提高，优于现有方法。



## **6. Training a Perceptual Model for Evaluating Auditory Similarity in Music Adversarial Attack**

训练用于评估音乐对抗性攻击中听觉相似性的感知模型 cs.SD

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04985v1) [paper-pdf](http://arxiv.org/pdf/2509.04985v1)

**Authors**: Yuxuan Liu, Rui Sang, Peihong Zhang, Zhixin Li, Shengchen Li

**Abstract**: Music Information Retrieval (MIR) systems are highly vulnerable to adversarial attacks that are often imperceptible to humans, primarily due to a misalignment between model feature spaces and human auditory perception. Existing defenses and perceptual metrics frequently fail to adequately capture these auditory nuances, a limitation supported by our initial listening tests showing low correlation between common metrics and human judgments. To bridge this gap, we introduce Perceptually-Aligned MERT Transformer (PAMT), a novel framework for learning robust, perceptually-aligned music representations. Our core innovation lies in the psychoacoustically-conditioned sequential contrastive transformer, a lightweight projection head built atop a frozen MERT encoder. PAMT achieves a Spearman correlation coefficient of 0.65 with subjective scores, outperforming existing perceptual metrics. Our approach also achieves an average of 9.15\% improvement in robust accuracy on challenging MIR tasks, including Cover Song Identification and Music Genre Classification, under diverse perceptual adversarial attacks. This work pioneers architecturally-integrated psychoacoustic conditioning, yielding representations significantly more aligned with human perception and robust against music adversarial attacks.

摘要: 音乐信息检索（MIR）系统非常容易受到人类通常难以察觉的对抗攻击，主要是由于模型特征空间和人类听觉感知之间的不一致。现有的防御和感知指标经常无法充分捕捉这些听觉细微差别，我们最初的听力测试支持了这一局限性，表明常见指标和人类判断之间的相关性较低。为了弥合这一差距，我们引入了感知对齐的MERT Transformer（PAMT），这是一种用于学习鲁棒的感知对齐音乐表示的新框架。我们的核心创新在于心理声学调节顺序对比Transformer，这是一个构建在冷冻MERT编码器上的轻型投影头。PAMT通过主观评分实现了0.65的Spearman相关系数，优于现有的感知指标。在各种感知对抗攻击下，我们的方法还在具有挑战性的MIR任务（包括翻唱歌曲识别和音乐流派分类）上实现了平均9.15%的稳健准确性提高。这项工作开创了架构集成的心理声学条件反射，产生的表示明显更符合人类感知，并且对音乐对抗攻击更强。



## **7. MAIA: An Inpainting-Based Approach for Music Adversarial Attacks**

MAIA：一种基于修补的音乐对抗攻击方法 cs.SD

Accepted at ISMIR2025

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04980v1) [paper-pdf](http://arxiv.org/pdf/2509.04980v1)

**Authors**: Yuxuan Liu, Peihong Zhang, Rui Sang, Zhixin Li, Shengchen Li

**Abstract**: Music adversarial attacks have garnered significant interest in the field of Music Information Retrieval (MIR). In this paper, we present Music Adversarial Inpainting Attack (MAIA), a novel adversarial attack framework that supports both white-box and black-box attack scenarios. MAIA begins with an importance analysis to identify critical audio segments, which are then targeted for modification. Utilizing generative inpainting models, these segments are reconstructed with guidance from the output of the attacked model, ensuring subtle and effective adversarial perturbations. We evaluate MAIA on multiple MIR tasks, demonstrating high attack success rates in both white-box and black-box settings while maintaining minimal perceptual distortion. Additionally, subjective listening tests confirm the high audio fidelity of the adversarial samples. Our findings highlight vulnerabilities in current MIR systems and emphasize the need for more robust and secure models.

摘要: 音乐对抗攻击在音乐信息检索（MIR）领域引起了人们的极大兴趣。在本文中，我们介绍了音乐对抗修补攻击（MAIA），这是一种新型的对抗攻击框架，支持白盒和黑盒攻击场景。MAIA首先进行重要性分析，以识别关键音频片段，然后针对这些片段进行修改。利用生成式修复模型，在受攻击模型输出的指导下重建这些片段，确保微妙且有效的对抗性扰动。我们评估了多个MIR任务中的MAIA，证明了在白盒和黑盒设置中的高攻击成功率，同时保持最小的感知失真。此外，主观听力测试证实了对抗样本的高音频保真度。我们的研究结果强调了当前MIR系统中的漏洞，并强调需要更强大和安全的模型。



## **8. RINSER: Accurate API Prediction Using Masked Language Models**

RINser：使用掩蔽语言模型进行准确的API预测 cs.CY

16 pages, 8 figures

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04887v1) [paper-pdf](http://arxiv.org/pdf/2509.04887v1)

**Authors**: Muhammad Ejaz Ahmed, Christopher Cody, Muhammad Ikram, Sean Lamont, Alsharif Abuadbba, Seyit Camtepe, Surya Nepal, Muhammad Ali Kaafar

**Abstract**: Malware authors commonly use obfuscation to hide API identities in binary files, making analysis difficult and time-consuming for a human expert to understand the behavior and intent of the program. Automatic API prediction tools are necessary to efficiently analyze unknown binaries, facilitating rapid malware triage while reducing the workload on human analysts. In this paper, we present RINSER (AccuRate API predictioN using maSked languagE model leaRning), an automated framework for predicting Windows API (WinAPI) function names. RINSER introduces the novel concept of API codeprints, a set of API-relevant assembly instructions, and supports x86 PE binaries. RINSER relies on BERT's masked language model (LM) to predict API names at scale, achieving 85.77% accuracy for normal binaries and 82.88% accuracy for stripped binaries. We evaluate RINSER on a large dataset of 4.7M API codeprints from 11,098 malware binaries, covering 4,123 unique Windows APIs, making it the largest publicly available dataset of this type. RINSER successfully discovered 65 obfuscated Windows APIs related to C2 communication, spying, and evasion in our dataset, which the commercial disassembler IDA failed to identify. Furthermore, we compared RINSER against three state-of-the-art approaches, showing over 20% higher prediction accuracy. We also demonstrated RINSER's resilience to adversarial attacks, including instruction randomization and code displacement, with a performance drop of no more than 3%.

摘要: 恶意软件作者通常使用混淆将API身份隐藏在二进制文件中，这使得人类专家理解程序的行为和意图变得困难且耗时。自动API预测工具对于有效分析未知二进制文件来说是必要的，可以促进快速恶意软件分类，同时减少人类分析师的工作量。在本文中，我们介绍了RINBER（使用maSked languagE模型leRning的ACATER API预测），这是一个用于预测Windows API（WinAPI）函数名称的自动化框架。RINser引入了API代码印的新颖概念，即一组与API相关的汇编指令，并支持x86 PE二进制文件。RINser依赖BERT的掩蔽语言模型（LM）来大规模预测API名称，正常二进制文件的准确性达到85.77%，剥离二进制文件的准确性达到82.88%。我们在一个包含来自11，098个恶意软件二进制文件的470万个API代码的大型数据集上评估了RINser，涵盖了4，123个独特的Windows API，使其成为此类类型中最大的公开可用数据集。RINBER在我们的数据集中成功发现了65个与C2通信、间谍和规避相关的模糊Windows API，但商业反汇编器IDA未能识别这些API。此外，我们将RINBER与三种最先进的方法进行了比较，结果显示预测准确性提高了20%以上。我们还展示了RINBER对对抗攻击（包括指令随机化和代码置换）的弹性，性能下降不超过3%。



## **9. PersonaTeaming: Exploring How Introducing Personas Can Improve Automated AI Red-Teaming**

角色协作：探索引入角色协作如何改善自动化人工智能红色协作 cs.AI

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.03728v2) [paper-pdf](http://arxiv.org/pdf/2509.03728v2)

**Authors**: Wesley Hanwen Deng, Sunnie S. Y. Kim, Akshita Jha, Ken Holstein, Motahhare Eslami, Lauren Wilcox, Leon A Gatys

**Abstract**: Recent developments in AI governance and safety research have called for red-teaming methods that can effectively surface potential risks posed by AI models. Many of these calls have emphasized how the identities and backgrounds of red-teamers can shape their red-teaming strategies, and thus the kinds of risks they are likely to uncover. While automated red-teaming approaches promise to complement human red-teaming by enabling larger-scale exploration of model behavior, current approaches do not consider the role of identity. As an initial step towards incorporating people's background and identities in automated red-teaming, we develop and evaluate a novel method, PersonaTeaming, that introduces personas in the adversarial prompt generation process to explore a wider spectrum of adversarial strategies. In particular, we first introduce a methodology for mutating prompts based on either "red-teaming expert" personas or "regular AI user" personas. We then develop a dynamic persona-generating algorithm that automatically generates various persona types adaptive to different seed prompts. In addition, we develop a set of new metrics to explicitly measure the "mutation distance" to complement existing diversity measurements of adversarial prompts. Our experiments show promising improvements (up to 144.1%) in the attack success rates of adversarial prompts through persona mutation, while maintaining prompt diversity, compared to RainbowPlus, a state-of-the-art automated red-teaming method. We discuss the strengths and limitations of different persona types and mutation methods, shedding light on future opportunities to explore complementarities between automated and human red-teaming approaches.

摘要: 人工智能治理和安全研究的最新进展呼吁采取红色团队方法，以有效地揭示人工智能模型带来的潜在风险。许多电话都强调了红色团队成员的身份和背景如何塑造他们的红色团队策略，从而也强调了他们可能发现的风险。虽然自动化红色团队方法有望通过更大规模地探索模型行为来补充人类红色团队，但目前的方法没有考虑身份的作用。作为将人们的背景和身份融入自动化红色团队的第一步，我们开发并评估了一种新型方法PersonaTeaming，该方法在对抗性提示生成过程中引入角色，以探索更广泛的对抗策略。特别是，我们首先引入了一种基于“红色团队专家”角色或“普通人工智能用户”角色来变异提示的方法。然后，我们开发了一个动态角色生成算法，该算法自动生成适应不同种子提示的各种角色类型。此外，我们开发了一组新的指标来明确测量“突变距离”，以补充现有的对抗提示多样性测量。我们的实验显示，与最先进的自动化红色团队方法RainbowPlus相比，通过角色突变的对抗提示的攻击成功率有了有希望的改进（高达144.1%），同时保持了提示的多样性。我们讨论了不同角色类型和突变方法的优点和局限性，揭示了未来探索自动化和人类红色团队方法之间互补性的机会。



## **10. Adversarial Hubness in Multi-Modal Retrieval**

多模式检索中的对抗性积极性 cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2412.14113v3) [paper-pdf](http://arxiv.org/pdf/2412.14113v3)

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries.   In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, and also for targeted attacks on queries related to specific, attacker-chosen concepts.   We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system implemented by Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub, generated using 100 random queries, is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries), demonstrating the strong generalization capabilities of adversarial hubs. We also investigate whether techniques for mitigating natural hubness can also mitigate adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.

摘要: Hubness是多维载体空间中的一种现象，其中自然分布的一个点与许多其他点异常接近。这是信息检索中一个众所周知的问题，会导致某些项意外（且错误地）看起来与许多查询相关。   在本文中，我们研究攻击者如何利用中心将多模式检索系统中的任何图像或音频输入变成对抗中心。对抗中心可用于注入通用对抗内容（例如，垃圾邮件）将响应数千个不同的查询而检索这些信息，并且还可以对与特定的攻击者选择的概念相关的查询进行有针对性的攻击。   我们提出了一种创建对抗中心的方法，并在基准多模式检索数据集和由流行的载体数据库Pinecone实现的图像到图像检索系统上评估所得中心。例如，在文本标题到图像检索中，使用100个随机查询生成的单个对抗中心被检索为25，000个测试查询中超过21，000个的前1最相关图像（相比之下，最常见的自然中心是仅对102个查询的前1响应），这表明了对抗中心的强大概括能力。我们还调查了减轻自然中心的技术是否也可以减轻对抗中心，并表明它们对针对与特定概念相关的查询的中心无效。



## **11. Breaking to Build: A Threat Model of Prompt-Based Attacks for Securing LLMs**

突破构建：用于保护LLM的基于预算的攻击威胁模型 cs.CL

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04615v1) [paper-pdf](http://arxiv.org/pdf/2509.04615v1)

**Authors**: Brennen Hill, Surendra Parla, Venkata Abhijeeth Balabhadruni, Atharv Prajod Padmalayam, Sujay Chandra Shekara Sharma

**Abstract**: The proliferation of Large Language Models (LLMs) has introduced critical security challenges, where adversarial actors can manipulate input prompts to cause significant harm and circumvent safety alignments. These prompt-based attacks exploit vulnerabilities in a model's design, training, and contextual understanding, leading to intellectual property theft, misinformation generation, and erosion of user trust. A systematic understanding of these attack vectors is the foundational step toward developing robust countermeasures. This paper presents a comprehensive literature survey of prompt-based attack methodologies, categorizing them to provide a clear threat model. By detailing the mechanisms and impacts of these exploits, this survey aims to inform the research community's efforts in building the next generation of secure LLMs that are inherently resistant to unauthorized distillation, fine-tuning, and editing.

摘要: 大型语言模型（LLM）的激增带来了关键的安全挑战，对抗行为者可以操纵输入提示造成重大伤害并规避安全一致。这些基于预算的攻击利用模型设计、培训和上下文理解中的漏洞，导致知识产权盗窃、错误信息生成和用户信任度侵蚀。系统地了解这些攻击载体是开发稳健对策的基础步骤。本文对基于预算的攻击方法进行了全面的文献调查，对它们进行了分类，以提供明确的威胁模型。通过详细介绍这些漏洞利用的机制和影响，本调查旨在为研究界构建下一代安全LLM的努力提供信息，这些LLM本质上可以抵抗未经授权的提炼、微调和编辑。



## **12. Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing**

Concept-ROT：使用模型编辑在大型语言模型中中毒概念 cs.LG

Published at ICLR 2025

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2412.13341v2) [paper-pdf](http://arxiv.org/pdf/2412.13341v2)

**Authors**: Keltin Grimes, Marco Christiani, David Shriver, Marissa Connor

**Abstract**: Model editing methods modify specific behaviors of Large Language Models by altering a small, targeted set of network weights and require very little data and compute. These methods can be used for malicious applications such as inserting misinformation or simple trojans that result in adversary-specified behaviors when a trigger word is present. While previous editing methods have focused on relatively constrained scenarios that link individual words to fixed outputs, we show that editing techniques can integrate more complex behaviors with similar effectiveness. We develop Concept-ROT, a model editing-based method that efficiently inserts trojans which not only exhibit complex output behaviors, but also trigger on high-level concepts -- presenting an entirely new class of trojan attacks. Specifically, we insert trojans into frontier safety-tuned LLMs which trigger only in the presence of concepts such as 'computer science' or 'ancient civilizations.' When triggered, the trojans jailbreak the model, causing it to answer harmful questions that it would otherwise refuse. Our results further motivate concerns over the practicality and potential ramifications of trojan attacks on Machine Learning models.

摘要: 模型编辑方法通过改变一小组有针对性的网络权重来修改大型语言模型的特定行为，并且需要很少的数据和计算。这些方法可用于恶意应用程序，例如插入错误信息或简单的特洛伊木马，当存在触发词时，这些木马会导致对手指定的行为。虽然之前的编辑方法专注于将单个单词与固定输出联系起来的相对受限的场景，但我们表明编辑技术可以以类似的效果集成更复杂的行为。我们开发了Concept-ROT，这是一种基于模型编辑的方法，可以有效地插入特洛伊木马，这些木马不仅表现出复杂的输出行为，而且还会触发高级概念--从而呈现出一种全新的特洛伊木马攻击。具体来说，我们将特洛伊木马插入到前沿安全调整的LLM中，这些LLM仅在存在“计算机科学”或“古代文明”等概念时才会触发。“当被触发时，特洛伊木马会越狱该模型，使其回答原本会拒绝的有害问题。我们的结果进一步引发了人们对机器学习模型木马攻击的实用性和潜在后果的担忧。



## **13. DisPatch: Disarming Adversarial Patches in Object Detection with Diffusion Models**

Dispatch：利用扩散模型消除对象检测中的对抗补丁 cs.CV

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04597v1) [paper-pdf](http://arxiv.org/pdf/2509.04597v1)

**Authors**: Jin Ma, Mohammed Aldeen, Christopher Salas, Feng Luo, Mashrur Chowdhury, Mert Pesé, Long Cheng

**Abstract**: Object detection is fundamental to various real-world applications, such as security monitoring and surveillance video analysis. Despite their advancements, state-of-theart object detectors are still vulnerable to adversarial patch attacks, which can be easily applied to real-world objects to either conceal actual items or create non-existent ones, leading to severe consequences. Given the current diversity of adversarial patch attacks and potential unknown threats, an ideal defense method should be effective, generalizable, and robust against adaptive attacks. In this work, we introduce DISPATCH, the first diffusion-based defense framework for object detection. Unlike previous works that aim to "detect and remove" adversarial patches, DISPATCH adopts a "regenerate and rectify" strategy, leveraging generative models to disarm attack effects while preserving the integrity of the input image. Specifically, we utilize the in-distribution generative power of diffusion models to regenerate the entire image, aligning it with benign data. A rectification process is then employed to identify and replace adversarial regions with their regenerated benign counterparts. DISPATCH is attack-agnostic and requires no prior knowledge of the existing patches. Extensive experiments across multiple detectors and attacks demonstrate that DISPATCH consistently outperforms state-of-the-art defenses on both hiding attacks and creating attacks, achieving the best overall mAP.5 score of 89.3% on hiding attacks, and lowering the attack success rate to 24.8% on untargeted creating attacks. Moreover, it maintains strong robustness against adaptive attacks, making it a practical and reliable defense for object detection systems.

摘要: 对象检测是各种现实应用的基础，例如安全监控和监控视频分析。尽管它们取得了进步，但最先进的对象检测器仍然容易受到对抗补丁攻击，这种攻击可以很容易地应用于现实世界的对象，以隐藏实际物品或创建不存在的物品，从而导致严重的后果。鉴于当前对抗性补丁攻击和潜在未知威胁的多样性，理想的防御方法应该是有效的、可推广的且鲁棒的，以对抗自适应攻击。在这项工作中，我们介绍了DISPATCH，这是第一个用于对象检测的基于扩散的防御框架。与之前旨在“检测和删除”对抗补丁的作品不同，DISPATCH采用“再生和纠正”策略，利用生成模型来消除攻击效果，同时保留输入图像的完整性。具体来说，我们利用扩散模型的内分布生成能力来重新生成整个图像，使其与良性数据对齐。然后采用纠正过程来识别敌对区域，并用再生的良性区域替换敌对区域。DISPATCH是攻击不可知的，并且不需要了解现有补丁。跨多个检测器和攻击的广泛实验表明，DISPATCH在隐藏攻击和创建攻击方面始终优于最先进的防御，在隐藏攻击方面实现了89.3%的最佳总体mAP.5得分，并将攻击成功率降低至24.8%。针对非目标创建攻击。此外，它还保持了对自适应攻击的强大鲁棒性，使其成为对象检测系统实用且可靠的防御。



## **14. Manipulating Transformer-Based Models: Controllability, Steerability, and Robust Interventions**

操纵基于变压器的模型：可控性、可操纵性和稳健干预 cs.CL

13 pages

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04549v1) [paper-pdf](http://arxiv.org/pdf/2509.04549v1)

**Authors**: Faruk Alpay, Taylan Alpay

**Abstract**: Transformer-based language models excel in NLP tasks, but fine-grained control remains challenging. This paper explores methods for manipulating transformer models through principled interventions at three levels: prompts, activations, and weights. We formalize controllable text generation as an optimization problem addressable via prompt engineering, parameter-efficient fine-tuning, model editing, and reinforcement learning. We introduce a unified framework encompassing prompt-level steering, activation interventions, and weight-space edits. We analyze robustness and safety implications, including adversarial attacks and alignment mitigations. Theoretically, we show minimal weight updates can achieve targeted behavior changes with limited side-effects. Empirically, we demonstrate >90% success in sentiment control and factual edits while preserving base performance, though generalization-specificity trade-offs exist. We discuss ethical dual-use risks and the need for rigorous evaluation. This work lays groundwork for designing controllable and robust language models.

摘要: 基于转换器的语言模型在NLP任务中表现出色，但细粒度控制仍然具有挑战性。本文探讨了通过三个层次的原则性干预来操纵Transformer模型的方法：提示、激活和权重。我们将可控文本生成形式化为一个可通过即时工程、参数高效微调、模型编辑和强化学习来解决的优化问题。我们引入了一个统一的框架，涵盖预算级引导、激活干预和重量空间编辑。我们分析稳健性和安全性影响，包括对抗性攻击和对齐缓解措施。理论上，我们表明最小的体重更新可以实现有针对性的行为改变，副作用有限。从经验上看，我们在情绪控制和事实编辑方面取得了超过90%的成功，同时保留了基本性能，尽管存在概括特定性权衡。我们讨论道德两用风险和严格评估的必要性。这项工作为设计可控且鲁棒的语言模型奠定了基础。



## **15. LADSG: Label-Anonymized Distillation and Similar Gradient Substitution for Label Privacy in Vertical Federated Learning**

LADSG：垂直联邦学习中标签模拟蒸馏和标签隐私的类似梯度替代 cs.CR

20 pages, 8 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2506.06742v3) [paper-pdf](http://arxiv.org/pdf/2506.06742v3)

**Authors**: Zeyu Yan, Yanfei Yao, Xuanbing Wen, Shixiong Zhang, Juli Zhang, Kai Fan

**Abstract**: Vertical Federated Learning (VFL) has emerged as a promising paradigm for collaborative model training across distributed feature spaces, which enables privacy-preserving learning without sharing raw data. However, recent studies have confirmed the feasibility of label inference attacks by internal adversaries. By strategically exploiting gradient vectors and semantic embeddings, attackers-through passive, active, or direct attacks-can accurately reconstruct private labels, leading to catastrophic data leakage. Existing defenses, which typically address isolated leakage vectors or are designed for specific types of attacks, remain vulnerable to emerging hybrid attacks that exploit multiple pathways simultaneously. To bridge this gap, we propose Label-Anonymized Defense with Substitution Gradient (LADSG), a unified and lightweight defense framework for VFL. LADSG first anonymizes true labels via soft distillation to reduce semantic exposure, then generates semantically-aligned substitute gradients to disrupt gradient-based leakage, and finally filters anomalous updates through gradient norm detection. It is scalable and compatible with standard VFL pipelines. Extensive experiments on six real-world datasets show that LADSG reduces the success rates of all three types of label inference attacks by 30-60% with minimal computational overhead, demonstrating its practical effectiveness.

摘要: 垂直联邦学习（VFL）已成为跨分布式特征空间协作模型训练的一种有前途的范式，它可以在无需共享原始数据的情况下实现隐私保护学习。然而，最近的研究证实了内部对手进行标签推断攻击的可行性。通过战略性地利用梯度载体和语义嵌入，攻击者通过被动、主动或直接攻击可以准确地重建私有标签，从而导致灾难性的数据泄露。现有的防御系统通常针对孤立的泄漏载体或专为特定类型的攻击而设计，但仍然容易受到同时利用多个途径的新兴混合攻击的影响。为了弥合这一差距，我们提出了具有替代梯度的标签模拟防御（LADSG），这是一个针对VFL的统一轻量级防御框架。LADSG首先通过软蒸馏匿名化真实标签以减少语义暴露，然后生成语义对齐的替代梯度以破坏基于梯度的泄漏，最后通过梯度范数检测过滤异常更新。它具有可扩展性，并与标准VFL管道兼容。在六个真实数据集上的大量实验表明，LADSG以最小的计算开销将所有三种类型的标签推理攻击的成功率降低了30-60%，证明了其实际有效性。



## **16. An Automated, Scalable Machine Learning Model Inversion Assessment Pipeline**

自动化、可扩展的机器学习模型反演评估管道 cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04214v1) [paper-pdf](http://arxiv.org/pdf/2509.04214v1)

**Authors**: Tyler Shumaker, Jessica Carpenter, David Saranchak, Nathaniel D. Bastian

**Abstract**: Machine learning (ML) models have the potential to transform military battlefields, presenting a large external pressure to rapidly incorporate them into operational settings. However, it is well-established that these ML models are vulnerable to a number of adversarial attacks throughout the model deployment pipeline that threaten to negate battlefield advantage. One broad category is privacy attacks (such as model inversion) where an adversary can reverse engineer information from the model, such as the sensitive data used in its training. The ability to quantify the risk of model inversion attacks (MIAs) is not well studied, and there is a lack of automated developmental test and evaluation (DT&E) tools and metrics to quantify the effectiveness of privacy loss of the MIA. The current DT&E process is difficult because ML model inversions can be hard for a human to interpret, subjective when they are interpretable, and difficult to quantify in terms of inversion quality. Additionally, scaling the DT&E process is challenging due to many ML model architectures and data modalities that need to be assessed. In this work, we present a novel DT&E tool that quantifies the risk of data privacy loss from MIAs and introduces four adversarial risk dimensions to quantify privacy loss. Our DT&E pipeline combines inversion with vision language models (VLMs) to improve effectiveness while enabling scalable analysis. We demonstrate effectiveness using multiple MIA techniques and VLMs configured for zero-shot classification and image captioning. We benchmark the pipeline using several state-of-the-art MIAs in the computer vision domain with an image classification task that is typical in military applications. In general, our innovative pipeline extends the current model inversion DT&E capabilities by improving the effectiveness and scalability of the privacy loss analysis in an automated fashion.

摘要: 机器学习（ML）模型有潜力改变军事战场，这给快速将其纳入作战环境带来了巨大的外部压力。然而，众所周知，这些ML模型在整个模型部署管道中容易受到许多对抗攻击，这些攻击可能会抵消战场优势。其中一个广泛的类别是隐私攻击（例如模型倒置），其中对手可以从模型中反向工程信息，例如训练中使用的敏感数据。量化模型倒置攻击（MIA）风险的能力尚未得到充分研究，并且缺乏自动化开发测试和评估（DT & E）工具和指标来量化MIA隐私损失的有效性。当前的DT & E过程很困难，因为ML模型倒置对人类来说可能很难解释，当它们可解释时是主观的，并且很难在倒置质量方面量化。此外，由于需要评估许多ML模型架构和数据模式，扩展DT & E流程具有挑战性。在这项工作中，我们提出了一种新颖的DT & E工具，该工具量化了MIA造成的数据隐私损失的风险，并引入了四个对抗风险维度来量化隐私损失。我们的DT & E管道将倒置与视觉语言模型（VLM）相结合，以提高有效性，同时实现可扩展分析。我们使用多种MIA技术和配置用于零镜头分类和图像字幕的VLM来证明有效性。我们使用计算机视觉领域的几种最先进的MIA对管道进行基准测试，并执行军事应用中典型的图像分类任务。总的来说，我们的创新管道通过以自动化方式提高隐私损失分析的有效性和可扩展性来扩展当前的模型倒置DT & E功能。



## **17. MUNBa: Machine Unlearning via Nash Bargaining**

MUNBa：通过纳什讨价还价的机器学习 cs.CV

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2411.15537v4) [paper-pdf](http://arxiv.org/pdf/2411.15537v4)

**Authors**: Jing Wu, Mehrtash Harandi

**Abstract**: Machine Unlearning (MU) aims to selectively erase harmful behaviors from models while retaining the overall utility of the model. As a multi-task learning problem, MU involves balancing objectives related to forgetting specific concepts/data and preserving general performance. A naive integration of these forgetting and preserving objectives can lead to gradient conflicts and dominance, impeding MU algorithms from reaching optimal solutions. To address the gradient conflict and dominance issue, we reformulate MU as a two-player cooperative game, where the two players, namely, the forgetting player and the preservation player, contribute via their gradient proposals to maximize their overall gain and balance their contributions. To this end, inspired by the Nash bargaining theory, we derive a closed-form solution to guide the model toward the Pareto stationary point. Our formulation of MU guarantees an equilibrium solution, where any deviation from the final state would lead to a reduction in the overall objectives for both players, ensuring optimality in each objective. We evaluate our algorithm's effectiveness on a diverse set of tasks across image classification and image generation. Extensive experiments with ResNet, vision-language model CLIP, and text-to-image diffusion models demonstrate that our method outperforms state-of-the-art MU algorithms, achieving a better trade-off between forgetting and preserving. Our results also highlight improvements in forgetting precision, preservation of generalization, and robustness against adversarial attacks.

摘要: 机器取消学习（MU）旨在选择性地从模型中删除有害行为，同时保留模型的整体效用。作为一个多任务学习问题，MU涉及平衡与忘记特定概念/数据和保持一般性能相关的目标。这些遗忘和保存目标的天真整合可能导致梯度冲突和优势，阻碍MU算法达到最优解。为了解决梯度冲突和优势的问题，我们重新制定MU作为一个两个球员的合作游戏，其中的两个球员，即遗忘球员和保存球员，有助于通过他们的梯度建议，以最大限度地提高他们的整体收益和平衡他们的贡献。为此，受纳什讨价还价理论的启发，我们推导出一个封闭解来引导模型走向帕累托稳定点。我们的MU公式保证了均衡解决方案，其中任何与最终状态的偏差都将导致双方参与者总体目标的减少，从而确保每个目标的最优性。我们评估了我们的算法在图像分类和图像生成等一系列不同任务中的有效性。ResNet、视觉语言模型CLIP和文本到图像扩散模型的广泛实验表明，我们的方法优于最先进的MU算法，在遗忘和保留之间实现了更好的权衡。我们的结果还强调了遗忘准确性、概括性的保留和对抗性攻击的鲁棒性的改进。



## **18. ICSLure: A Very High Interaction Honeynet for PLC-based Industrial Control Systems**

ICSlure：用于基于PLC的工业控制系统的高度交互蜜网 cs.CR

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04080v1) [paper-pdf](http://arxiv.org/pdf/2509.04080v1)

**Authors**: Francesco Aurelio Pironti, Angelo Furfaro, Francesco Blefari, Carmelo Felicetti, Matteo Lupinacci, Francesco Romeo

**Abstract**: The security of Industrial Control Systems (ICSs) is critical to ensuring the safety of industrial processes and personnel. The rapid adoption of Industrial Internet of Things (IIoT) technologies has expanded system functionality but also increased the attack surface, exposing ICSs to a growing range of cyber threats. Honeypots provide a means to detect and analyze such threats by emulating target systems and capturing attacker behavior. However, traditional ICS honeypots, often limited to software-based simulations of a single Programmable Logic Controller (PLC), lack the realism required to engage sophisticated adversaries. In this work, we introduce a modular honeynet framework named ICSLure. The framework has been designed to emulate realistic ICS environments. Our approach integrates physical PLCs interacting with live data sources via industrial protocols such as Modbus and Profinet RTU, along with virtualized network components including routers, switches, and Remote Terminal Units (RTUs). The system incorporates comprehensive monitoring capabilities to collect detailed logs of attacker interactions. We demonstrate that our framework enables coherent and high-fidelity emulation of real-world industrial plants. This high-interaction environment significantly enhances the quality of threat data collected and supports advanced analysis of ICS-specific attack strategies, contributing to more effective detection and mitigation techniques.

摘要: 工业控制系统（ICS）的安全对于确保工业流程和人员的安全至关重要。工业物联网（IIoT）技术的迅速采用扩展了系统功能，但也增加了攻击面，使ICS面临越来越多的网络威胁。蜜罐提供了一种通过模拟目标系统和捕获攻击者行为来检测和分析此类威胁的方法。然而，传统的ICS蜜罐通常仅限于对单个可编程逻辑控制器（PLC）的基于软件的模拟，缺乏与复杂对手交战所需的现实性。在这项工作中，我们引入了一个名为ICSlure的模块化蜜网框架。该框架旨在模拟现实的ICS环境。我们的方法集成了通过工业协议（例如Modbus和Profinet RTI）与实时数据源交互的物理PLC，以及包括路由器、交换机和远程终端单元（RTI）在内的虚拟化网络组件。该系统集成了全面的监控功能，可以收集攻击者交互的详细日志。我们证明，我们的框架能够对现实世界的工业工厂进行连贯和高保真的模拟。这种高交互性环境显着提高了收集的威胁数据的质量，并支持对ICS特定攻击策略的高级分析，从而有助于更有效的检测和缓解技术。



## **19. System Identification from Partial Observations under Adversarial Attacks**

对抗性攻击下的部分观测结果识别系统 math.OC

8 pages, 3 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2504.00244v3) [paper-pdf](http://arxiv.org/pdf/2504.00244v3)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper is concerned with the partially observed linear system identification, where the goal is to obtain reasonably accurate estimation of the balanced truncation of the true system up to order $k$ from output measurements. We consider the challenging case of system identification under adversarial attacks, where the probability of having an attack at each time is $\Theta(1/k)$ while the value of the attack is arbitrary. We first show that the $\ell_1$-norm estimator exactly identifies the true Markov parameter matrix for nilpotent systems under any type of attack. We then build on this result to extend it to general systems and show that the estimation error exponentially decays as $k$ grows. The estimated balanced truncation model accordingly shows an exponentially decaying error for the identification of the true system up to a similarity transformation. This work is the first to provide the input-output analysis of the system with partial observations under arbitrary attacks.

摘要: 本文涉及部分观测线性系统识别，目标是从输出测量中获得对高达k$阶的真实系统的平衡截断的相当准确的估计。我们考虑了对抗性攻击下系统识别的具有挑战性的情况，其中每次遭受攻击的概率为$\Theta（1/k）$，而攻击的值是任意的。我们首先表明，在任何类型的攻击下，$\ell_1 $-模估计器准确识别了幂零系统的真实Markov参数矩阵。然后，我们在这个结果的基础上将其扩展到一般系统，并表明估计误差随着$k$的增长而呈指数级衰减。因此，估计的平衡截断模型对于识别真实系统直到相似性变换时显示出指数衰减的误差。这项工作是第一个提供的输入输出分析系统的部分观测任意攻击。



## **20. NeuroBreak: Unveil Internal Jailbreak Mechanisms in Large Language Models**

NeuroBreak：揭开大型语言模型中的内部越狱机制 cs.CR

12 pages, 9 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.03985v1) [paper-pdf](http://arxiv.org/pdf/2509.03985v1)

**Authors**: Chuhan Zhang, Ye Zhang, Bowen Shi, Yuyou Gan, Tianyu Du, Shouling Ji, Dazhan Deng, Yingcai Wu

**Abstract**: In deployment and application, large language models (LLMs) typically undergo safety alignment to prevent illegal and unethical outputs. However, the continuous advancement of jailbreak attack techniques, designed to bypass safety mechanisms with adversarial prompts, has placed increasing pressure on the security defenses of LLMs. Strengthening resistance to jailbreak attacks requires an in-depth understanding of the security mechanisms and vulnerabilities of LLMs. However, the vast number of parameters and complex structure of LLMs make analyzing security weaknesses from an internal perspective a challenging task. This paper presents NeuroBreak, a top-down jailbreak analysis system designed to analyze neuron-level safety mechanisms and mitigate vulnerabilities. We carefully design system requirements through collaboration with three experts in the field of AI security. The system provides a comprehensive analysis of various jailbreak attack methods. By incorporating layer-wise representation probing analysis, NeuroBreak offers a novel perspective on the model's decision-making process throughout its generation steps. Furthermore, the system supports the analysis of critical neurons from both semantic and functional perspectives, facilitating a deeper exploration of security mechanisms. We conduct quantitative evaluations and case studies to verify the effectiveness of our system, offering mechanistic insights for developing next-generation defense strategies against evolving jailbreak attacks.

摘要: 在部署和应用中，大型语言模型（LLM）通常会进行安全调整，以防止非法和不道德的输出。然而，越狱攻击技术的不断进步（旨在通过对抗提示绕过安全机制）给LLM的安全防御带来了越来越大的压力。加强对越狱攻击的抵抗需要深入了解LLM的安全机制和漏洞。然而，LLM的大量参数和复杂结构使得从内部角度分析安全弱点成为一项具有挑战性的任务。本文介绍了NeuroBreak，这是一个自上而下的越狱分析系统，旨在分析神经元级安全机制并缓解漏洞。我们通过与人工智能安全领域的三位专家合作，精心设计系统需求。该系统对各种越狱攻击方法进行了全面分析。通过结合分层表示探测分析，NeuroBreak为模型整个生成步骤的决策过程提供了新颖的视角。此外，该系统支持从语义和功能角度分析关键神经元，促进对安全机制的更深入探索。我们进行定量评估和案例研究来验证我们系统的有效性，为开发针对不断变化的越狱攻击的下一代防御策略提供机械见解。



## **21. Towards Robust Graph Structural Learning Beyond Homophily via Preserving Neighbor Similarity**

通过保留邻居相似性来实现超越同质性的鲁棒图结构学习 cs.LG

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2401.09754v2) [paper-pdf](http://arxiv.org/pdf/2401.09754v2)

**Authors**: Yulin Zhu, Yuni Lai, Xing Ai, Wai Lun LO, Gaolei Li, Jianhua Li, Di Tang, Xingxing Zhang, Mengpei Yang, Kai Zhou

**Abstract**: Despite the tremendous success of graph-based learning systems in handling structural data, it has been widely investigated that they are fragile to adversarial attacks on homophilic graph data, where adversaries maliciously modify the semantic and topology information of the raw graph data to degrade the predictive performances. Motivated by this, a series of robust models are crafted to enhance the adversarial robustness of graph-based learning systems on homophilic graphs. However, the security of graph-based learning systems on heterophilic graphs remains a mystery to us. To bridge this gap, in this paper, we start to explore the vulnerability of graph-based learning systems regardless of the homophily degree, and theoretically prove that the update of the negative classification loss is negatively correlated with the pairwise similarities based on the powered aggregated neighbor features. The theoretical finding inspires us to craft a novel robust graph structural learning strategy that serves as a useful graph mining module in a robust model that incorporates a dual-kNN graph constructions pipeline to supervise the neighbor-similarity-preserved propagation, where the graph convolutional layer adaptively smooths or discriminates the features of node pairs according to their affluent local structures. In this way, the proposed methods can mine the ``better" topology of the raw graph data under diverse graph homophily and achieve more reliable data management on homophilic and heterophilic graphs.

摘要: 尽管基于图的学习系统在处理结构数据方面取得了巨大成功，但人们广泛研究发现，它们对同亲图数据的对抗攻击很脆弱，其中对手恶意修改原始图数据的语义和布局信息以降低预测性能。受此启发，我们精心设计了一系列稳健的模型，以增强基于图的学习系统在同向图上的对抗稳健性。然而，异嗜图上的基于图的学习系统的安全性对我们来说仍然是一个谜。为了弥合这一差距，本文开始探索基于图的学习系统的脆弱性，无论其同质性程度如何，并从理论上证明负分类损失的更新与基于增强聚合邻居特征的成对相似性呈负相关。这一理论发现激励我们制定一种新颖的鲁棒图结构学习策略，该策略作为鲁棒模型中有用的图挖掘模块，该模型结合了双kNN图构造管道来监督邻居相似性保留传播，其中图卷积层根据其丰富的本地结构自适应地平滑或区分节点对的特征。通过这种方式，所提出的方法可以挖掘不同图同质性下原始图数据的“更好”的布局，并实现对同同和异同图更可靠的数据管理。



## **22. Generative AI for Physical-Layer Authentication**

用于物理层身份验证的生成人工智能 eess.SP

10 pages, 3 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2504.18175v2) [paper-pdf](http://arxiv.org/pdf/2504.18175v2)

**Authors**: Rui Meng, Xiqi Cheng, Song Gao, Xiaodong Xu, Chen Dong, Guoshun Nan, Xiaofeng Tao, Ping Zhang, Tony Q. S. Quek

**Abstract**: In recent years, Artificial Intelligence (AI)-driven Physical-Layer Authentication (PLA), which focuses on achieving endogenous security and intelligent identity authentication, has attracted considerable interest. When compared with Discriminative AI (DAI), Generative AI (GAI) offers several advantages, such as fingerprint data augmentation, fingerprint denoising and reconstruction, and protection against adversarial attacks. Inspired by these innovations, this paper provides a systematic exploration of GAI's integration into PLA frameworks. We commence with a review of representative authentication techniques, emphasizing PLA's inherent strengths. Following this, we revisit four typical GAI models and contrast the limitations of DAI with the potential of GAI in addressing PLA challenges, including insufficient fingerprint data, environment noises and inferences, perturbations in fingerprint data, and complex tasks. Specifically, we delve into providing GAI-enhanced methods for PLA across the fingerprint collection, model training, and performance optimization phases in detail. Moreover, we present a case study that combines fingerprint extrapolation and Generative Diffusion Model (GDM) to illustrate the superiority of GAI in bolstering the reliability of PLA. Additionally, we outline potential future research directions for GAI-based PLA.

摘要: 近年来，专注于实现内生安全和智能身份认证的人工智能（AI）驱动的物理层认证（PLA）引起了相当大的兴趣。与区分性人工智能（DAI）相比，生成性人工智能（GAI）提供了多个优势，例如指纹数据增强、指纹去噪和重建以及对抗攻击的保护。受这些创新的启发，本文对GAI融入解放军框架进行了系统探索。我们首先回顾代表性的认证技术，强调解放军的固有优势。随后，我们重新审视了四种典型的GAI模型，并将DAI的局限性与GAI在应对解放军挑战方面的潜力进行了比较，包括指纹数据不足、环境噪音和推断、指纹数据的扰动以及复杂的任务。具体来说，我们深入研究了在指纹收集、模型训练和性能优化阶段为PLA提供GAI增强的方法。此外，我们还提出了一个结合指纹外推和生成扩散模型（GDM）的案例研究，以说明GAI在增强解放军可靠性方面的优势。此外，我们还概述了基于GAI的解放军未来潜在的研究方向。



## **23. Learning an Adversarial World Model for Automated Curriculum Generation in MARL**

在MARL中学习自动课程生成的对抗世界模型 cs.LG

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03771v1) [paper-pdf](http://arxiv.org/pdf/2509.03771v1)

**Authors**: Brennen Hill

**Abstract**: World models that infer and predict environmental dynamics are foundational to embodied intelligence. However, their potential is often limited by the finite complexity and implicit biases of hand-crafted training environments. To develop truly generalizable and robust agents, we need environments that scale in complexity alongside the agents learning within them. In this work, we reframe the challenge of environment generation as the problem of learning a goal-conditioned, generative world model. We propose a system where a generative **Attacker** agent learns an implicit world model to synthesize increasingly difficult challenges for a team of cooperative **Defender** agents. The Attacker's objective is not passive prediction, but active, goal-driven interaction: it models and generates world states (i.e., configurations of enemy units) specifically to exploit the Defenders' weaknesses. Concurrently, the embodied Defender team learns a cooperative policy to overcome these generated worlds. This co-evolutionary dynamic creates a self-scaling curriculum where the world model continuously adapts to challenge the decision-making policy of the agents, providing an effectively infinite stream of novel and relevant training scenarios. We demonstrate that this framework leads to the emergence of complex behaviors, such as the world model learning to generate flanking and shielding formations, and the defenders learning coordinated focus-fire and spreading tactics. Our findings position adversarial co-evolution as a powerful method for learning instrumental world models that drive agents toward greater strategic depth and robustness.

摘要: 推断和预测环境动态的世界模型是体现智能的基础。然而，它们的潜力往往受到手工制作训练环境的有限复杂性和隐含偏见的限制。为了开发真正可推广和强大的代理，我们需要复杂性可扩展的环境，以及代理在其中学习的环境。在这项工作中，我们将环境生成的挑战重新定义为学习受目标限制的生成世界模型的问题。我们提出了一个系统，其中生成的 ** 攻击者 ** 代理学习隐式世界模型，为合作的 **Defender** 代理团队综合日益困难的挑战。攻击者的目标不是被动预测，而是主动的、目标驱动的交互：它建模并生成世界状态（即，敌方单位的配置）专门利用防御者的弱点。与此同时，具体化的Defender团队学习了一种合作政策来克服这些生成的世界。这种共同进化的动态创建了一个自扩展的课程，其中世界模型不断调整以挑战代理人的决策政策，从而提供有效无限的新颖且相关的培训场景流。我们证明，这个框架会导致复杂行为的出现，例如世界模型学习生成侧翼和掩护队形，以及防御者学习协调的焦点射击和传播战术。我们的研究结果将对抗性协同进化定位为学习工具世界模型的一种强大方法，可以推动智能体走向更大的战略深度和稳健性。



## **24. ANNIE: Be Careful of Your Robots**

安妮：小心你的机器人 cs.AI

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03383v1) [paper-pdf](http://arxiv.org/pdf/2509.03383v1)

**Authors**: Yiyang Huang, Zixuan Wang, Zishen Wan, Yapeng Tian, Haobo Xu, Yinhe Han, Yiming Gan

**Abstract**: The integration of vision-language-action (VLA) models into embodied AI (EAI) robots is rapidly advancing their ability to perform complex, long-horizon tasks in humancentric environments. However, EAI systems introduce critical security risks: a compromised VLA model can directly translate adversarial perturbations on sensory input into unsafe physical actions. Traditional safety definitions and methodologies from the machine learning community are no longer sufficient. EAI systems raise new questions, such as what constitutes safety, how to measure it, and how to design effective attack and defense mechanisms in physically grounded, interactive settings. In this work, we present the first systematic study of adversarial safety attacks on embodied AI systems, grounded in ISO standards for human-robot interactions. We (1) formalize a principled taxonomy of safety violations (critical, dangerous, risky) based on physical constraints such as separation distance, velocity, and collision boundaries; (2) introduce ANNIEBench, a benchmark of nine safety-critical scenarios with 2,400 video-action sequences for evaluating embodied safety; and (3) ANNIE-Attack, a task-aware adversarial framework with an attack leader model that decomposes long-horizon goals into frame-level perturbations. Our evaluation across representative EAI models shows attack success rates exceeding 50% across all safety categories. We further demonstrate sparse and adaptive attack strategies and validate the real-world impact through physical robot experiments. These results expose a previously underexplored but highly consequential attack surface in embodied AI systems, highlighting the urgent need for security-driven defenses in the physical AI era. Code is available at https://github.com/RLCLab/Annie.

摘要: 将视觉-语言-动作（VLA）模型集成到体现式人工智能（EAI）机器人中，正在迅速提高它们在以人为本的环境中执行复杂、长期任务的能力。然而，EAI系统引入了关键的安全风险：受损的VLA模型可以直接将感官输入的对抗性扰动转化为不安全的身体动作。机器学习社区的传统安全定义和方法论已不再足够。EAI系统提出了新的问题，例如什么构成安全性、如何衡量安全性以及如何在物理基础的交互环境中设计有效的攻击和防御机制。在这项工作中，我们基于人与机器人交互的ISO标准，首次对嵌入式人工智能系统的对抗性安全攻击进行了系统性研究。我们（1）正式确定安全违规的原则分类（关键、危险、有风险）基于物理限制，例如分离距离、速度和碰撞边界;（2）引入ANNIEBench，这是九个安全关键场景的基准，包含2，400个视频动作序列，用于评估具体安全性;和（3）ANNIE-Attack，一个任务感知对抗框架，具有攻击领导者模型，可将长期目标分解为帧级扰动。我们对代表性EAI模型的评估显示，在所有安全类别中，攻击成功率超过50%。我们进一步展示了稀疏和自适应攻击策略，并通过物理机器人实验验证了现实世界的影响。这些结果暴露了嵌入式人工智能系统中之前未充分探索但后果严重的攻击表面，凸显了物理人工智能时代对安全驱动防御的迫切需求。代码可在https://github.com/RLCLab/Annie上获取。



## **25. On the MIA Vulnerability Gap Between Private GANs and Diffusion Models**

私人GAN和扩散模型之间的MIA脆弱性差距 cs.LG

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03341v1) [paper-pdf](http://arxiv.org/pdf/2509.03341v1)

**Authors**: Ilana Sebag, Jean-Yves Franceschi, Alain Rakotomamonjy, Alexandre Allauzen, Jamal Atif

**Abstract**: Generative Adversarial Networks (GANs) and diffusion models have emerged as leading approaches for high-quality image synthesis. While both can be trained under differential privacy (DP) to protect sensitive data, their sensitivity to membership inference attacks (MIAs), a key threat to data confidentiality, remains poorly understood. In this work, we present the first unified theoretical and empirical analysis of the privacy risks faced by differentially private generative models. We begin by showing, through a stability-based analysis, that GANs exhibit fundamentally lower sensitivity to data perturbations than diffusion models, suggesting a structural advantage in resisting MIAs. We then validate this insight with a comprehensive empirical study using a standardized MIA pipeline to evaluate privacy leakage across datasets and privacy budgets. Our results consistently reveal a marked privacy robustness gap in favor of GANs, even in strong DP regimes, highlighting that model type alone can critically shape privacy leakage.

摘要: 生成对抗网络（GAN）和扩散模型已成为高质量图像合成的领先方法。虽然两者都可以在差异隐私（DP）下进行训练以保护敏感数据，但它们对成员资格推理攻击（MIA）（数据机密性的关键威胁）的敏感性仍然知之甚少。在这项工作中，我们首次对差异隐私生成模型面临的隐私风险进行了统一的理论和实证分析。我们首先通过基于稳定性的分析表明，GAN对数据扰动的敏感性从根本上低于扩散模型，这表明在抵抗MIA方面具有结构性优势。然后，我们通过一项全面的实证研究来验证这一见解，使用标准化的MIA管道来评估数据集和隐私预算之间的隐私泄露。我们的结果一致显示，即使在强大的DP机制中，GAN也存在明显的隐私稳健性差距，这凸显了模型类型本身就可以严重影响隐私泄露。



## **26. Attacking Misinformation Detection Using Adversarial Examples Generated by Language Models**

使用语言模型生成的对抗性示例进行攻击错误信息检测 cs.CL

Presented at EMNLP 2025

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2410.20940v2) [paper-pdf](http://arxiv.org/pdf/2410.20940v2)

**Authors**: Piotr Przybyła, Euan McGill, Horacio Saggion

**Abstract**: Large language models have many beneficial applications, but can they also be used to attack content-filtering algorithms in social media platforms? We investigate the challenge of generating adversarial examples to test the robustness of text classification algorithms detecting low-credibility content, including propaganda, false claims, rumours and hyperpartisan news. We focus on simulation of content moderation by setting realistic limits on the number of queries an attacker is allowed to attempt. Within our solution (TREPAT), initial rephrasings are generated by large language models with prompts inspired by meaning-preserving NLP tasks, such as text simplification and style transfer. Subsequently, these modifications are decomposed into small changes, applied through beam search procedure, until the victim classifier changes its decision. We perform (1) quantitative evaluation using various prompts, models and query limits, (2) targeted manual assessment of the generated text and (3) qualitative linguistic analysis. The results confirm the superiority of our approach in the constrained scenario, especially in case of long input text (news articles), where exhaustive search is not feasible.

摘要: 大型语言模型有许多有益的应用，但它们也可以用于攻击社交媒体平台中的内容过滤算法吗？我们调查了生成敌对示例的挑战，以测试检测低可信度内容（包括宣传、虚假声明、谣言和超党派新闻）的文本分类算法的稳健性。我们通过对允许攻击者尝试的查询数量设置现实的限制来重点模拟内容审核。在我们的解决方案（TREPAT）中，初始改写由大型语言模型生成，其提示受到保留意义的NLP任务（例如文本简化和风格转移）的启发。随后，这些修改被分解成小的变化，通过束搜索过程应用，直到受害者分类器改变其决定。我们（1）使用各种提示、模型和查询限制执行定量评估，（2）对生成的文本进行有针对性的手动评估，（3）定性语言分析。结果证实了我们的方法在受限场景中的优越性，特别是在长输入文本（新闻文章）的情况下，其中详尽搜索是不可行的。



## **27. Efficient and Secure Sleepy Model for BFT Consensus**

BFT共识的高效安全睡眠模型 cs.DC

Accepted to ESORICS 2025, 20 pages, 7 figures

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03145v1) [paper-pdf](http://arxiv.org/pdf/2509.03145v1)

**Authors**: Pengkun Ren, Hai Dong, Zahir Tari, Pengcheng Zhang

**Abstract**: Byzantine Fault Tolerant (BFT) consensus protocols for dynamically available systems face a critical challenge: balancing latency and security in fluctuating node participation. Existing solutions often require multiple rounds of voting per decision, leading to high latency or limited resilience to adversarial behavior. This paper presents a BFT protocol integrating a pre-commit mechanism with publicly verifiable secret sharing (PVSS) into message transmission. By binding users' identities to their messages through PVSS, our approach reduces communication rounds. Compared to other state-of-the-art methods, our protocol typically requires only four network delays (4$\Delta$) in common scenarios while being resilient to up to 1/2 adversarial participants. This integration enhances the efficiency and security of the protocol without compromising integrity. Theoretical analysis demonstrates the robustness of the protocol against Byzantine attacks. Experimental evaluations show that, compared to traditional BFT protocols, our protocol significantly prevents fork occurrences and improves chain stability. Furthermore, compared to longest-chain protocol, our protocol maintains stability and lower latency in scenarios with moderate participation fluctuations.

摘要: 用于动态可用系统的拜占庭式故障容忍（BFT）共识协议面临着一个严峻的挑战：在波动的节点参与中平衡延迟和安全性。现有的解决方案通常需要每次决策进行多轮投票，导致延迟高或对抗行为的弹性有限。本文提出了一种BFT协议，将预提交机制和可公开验证的秘密共享（PVSS）集成到消息传输中。通过通过PVS将用户身份绑定到他们的消息，我们的方法减少了通信回合。与其他最先进的方法相比，我们的协议在常见情况下通常只需要四个网络延迟（4 $\Delta $），同时对多达1/2的对抗参与者具有弹性。这种集成增强了协议的效率和安全性，而不会损害完整性。理论分析证明了该协议对拜占庭攻击的鲁棒性。实验评估表明，与传统的BFT协议相比，我们的协议可以显着防止分叉的发生并提高链稳定性。此外，与最长链协议相比，我们的协议在参与波动适度的情况下保持稳定性和较低的延迟。



## **28. Similarity between Units of Natural Language: The Transition from Coarse to Fine Estimation**

自然语言单位之间的相似性：从粗略估计到精细估计的过渡 cs.CL

PhD thesis

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2210.14275v3) [paper-pdf](http://arxiv.org/pdf/2210.14275v3)

**Authors**: Wenchuan Mu

**Abstract**: Capturing the similarities between human language units is crucial for explaining how humans associate different objects, and therefore its computation has received extensive attention, research, and applications. With the ever-increasing amount of information around us, calculating similarity becomes increasingly complex, especially in many cases, such as legal or medical affairs, measuring similarity requires extra care and precision, as small acts within a language unit can have significant real-world effects. My research goal in this thesis is to develop regression models that account for similarities between language units in a more refined way.   Computation of similarity has come a long way, but approaches to debugging the measures are often based on continually fitting human judgment values. To this end, my goal is to develop an algorithm that precisely catches loopholes in a similarity calculation. Furthermore, most methods have vague definitions of the similarities they compute and are often difficult to interpret. The proposed framework addresses both shortcomings. It constantly improves the model through catching different loopholes. In addition, every refinement of the model provides a reasonable explanation. The regression model introduced in this thesis is called progressively refined similarity computation, which combines attack testing with adversarial training. The similarity regression model of this thesis achieves state-of-the-art performance in handling edge cases.

摘要: 捕捉人类语言单位之间的相似性对于解释人类如何将不同对象关联起来至关重要，因此其计算受到了广泛的关注、研究和应用。随着我们周围的信息量不断增加，计算相似性变得越来越复杂，特别是在许多情况下，例如法律或医疗事务，测量相似性需要格外小心和精确，因为语言单位内的小行为可能会产生显着的现实世界影响。我在这篇论文中的研究目标是开发回归模型，以更精确的方式考虑语言单位之间的相似性。   相似性的计算已经取得了很大的进步，但调试测量的方法通常基于不断匹配的人类判断值。为此，我的目标是开发一种精确捕捉相似度计算中漏洞的算法。此外，大多数方法对其计算的相似性的定义模糊，并且通常难以解释。拟议的框架解决了这两个缺点。它通过捕捉不同的漏洞来不断改进模型。此外，模型的每一次细化都提供了合理的解释。本文引入的回归模型称为渐进式细化相似度计算，它将攻击测试与对抗训练相结合。本文的相似度回归模型在处理边缘案例方面达到了最先进的性能。



## **29. EverTracer: Hunting Stolen Large Language Models via Stealthy and Robust Probabilistic Fingerprint**

EverTracer：通过隐秘且稳健的概率指纹追踪被盗的大型语言模型 cs.CR

Accepted by EMNLP2025 Main

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.03058v1) [paper-pdf](http://arxiv.org/pdf/2509.03058v1)

**Authors**: Zhenhua Xu, Meng Han, Wenpeng Xing

**Abstract**: The proliferation of large language models (LLMs) has intensified concerns over model theft and license violations, necessitating robust and stealthy ownership verification. Existing fingerprinting methods either require impractical white-box access or introduce detectable statistical anomalies. We propose EverTracer, a novel gray-box fingerprinting framework that ensures stealthy and robust model provenance tracing. EverTracer is the first to repurpose Membership Inference Attacks (MIAs) for defensive use, embedding ownership signals via memorization instead of artificial trigger-output overfitting. It consists of Fingerprint Injection, which fine-tunes the model on any natural language data without detectable artifacts, and Verification, which leverages calibrated probability variation signal to distinguish fingerprinted models. This approach remains robust against adaptive adversaries, including input level modification, and model-level modifications. Extensive experiments across architectures demonstrate EverTracer's state-of-the-art effectiveness, stealthness, and resilience, establishing it as a practical solution for securing LLM intellectual property. Our code and data are publicly available at https://github.com/Xuzhenhua55/EverTracer.

摘要: 大型语言模型（LLM）的激增加剧了人们对模型盗窃和许可证违规的担忧，需要进行强大且隐蔽的所有权验证。现有的指纹识别方法要么需要不切实际的白盒访问，要么引入可检测到的统计异常。我们提出了EverTracer，这是一种新型的灰箱指纹识别框架，可以确保隐蔽且稳健的模型出处追踪。EverTracer是第一个将会员推断攻击（MIA）重新用于防御用途的公司，通过记忆而不是人为的命令输出过度匹配来嵌入所有权信号。它由指纹注入和验证组成，前者在任何自然语言数据上微调模型，而不会检测到伪影，后者利用校准的概率变化信号来区分指纹模型。这种方法对于自适应对手（包括输入级别修改和模型级别修改）仍然具有鲁棒性。跨架构的广泛实验证明了EverTracer最先进的有效性、隐蔽性和弹性，使其成为保护LLM知识产权的实用解决方案。我们的代码和数据可在https://github.com/Xuzhenhua55/EverTracer上公开获取。



## **30. When and Where do Data Poisons Attack Textual Inversion?**

数据毒药何时何地攻击文本倒置？ cs.CR

Accepted to ICCV 2025

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2507.10578v4) [paper-pdf](http://arxiv.org/pdf/2507.10578v4)

**Authors**: Jeremy Styborski, Mingzhi Lyu, Jiayou Lu, Nupur Kapur, Adams Kong

**Abstract**: Poisoning attacks pose significant challenges to the robustness of diffusion models (DMs). In this paper, we systematically analyze when and where poisoning attacks textual inversion (TI), a widely used personalization technique for DMs. We first introduce Semantic Sensitivity Maps, a novel method for visualizing the influence of poisoning on text embeddings. Second, we identify and experimentally verify that DMs exhibit non-uniform learning behavior across timesteps, focusing on lower-noise samples. Poisoning attacks inherit this bias and inject adversarial signals predominantly at lower timesteps. Lastly, we observe that adversarial signals distract learning away from relevant concept regions within training data, corrupting the TI process. Based on these insights, we propose Safe-Zone Training (SZT), a novel defense mechanism comprised of 3 key components: (1) JPEG compression to weaken high-frequency poison signals, (2) restriction to high timesteps during TI training to avoid adversarial signals at lower timesteps, and (3) loss masking to constrain learning to relevant regions. Extensive experiments across multiple poisoning methods demonstrate that SZT greatly enhances the robustness of TI against all poisoning attacks, improving generative quality beyond prior published defenses. Code: www.github.com/JStyborski/Diff_Lab Data: www.github.com/JStyborski/NC10

摘要: 中毒攻击对扩散模型（DM）的鲁棒性构成了重大挑战。本文系统地分析了中毒攻击文本倒置（TI）的时间和地点，文本倒置（TI）是一种广泛使用的DM个性化技术。我们首先介绍语义敏感度地图，这是一种用于可视化中毒对文本嵌入影响的新颖方法。其次，我们识别并通过实验验证DM在跨时间步上表现出非均匀的学习行为，重点关注低噪音样本。中毒攻击继承了这种偏见，并主要在较低的时间步注入对抗信号。最后，我们观察到对抗信号分散了学习对训练数据中相关概念区域的注意力，从而破坏了TI过程。基于这些见解，我们提出了安全区训练（SZT），这是一种由3个关键组件组成的新型防御机制：（1）JPEG压缩以削弱高频毒物信号，（2）TI训练期间限制高时步，以避免较低时步的对抗信号，（3）损失掩蔽以将学习限制在相关区域。多种中毒方法的广泛实验表明，SZT极大地增强了TI针对所有中毒攻击的稳健性，提高了生成质量，超出了之前发布的防御措施。代码：www.github.com/JStyborski/Diff_Lab数据：www.github.com/JStyborski/NC10



## **31. Network-Level Prompt and Trait Leakage in Local Research Agents**

本地研究代理的网络级提示和特征泄露 cs.CR

Code available at https://github.com/umass-aisec/wra

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2508.20282v2) [paper-pdf](http://arxiv.org/pdf/2508.20282v2)

**Authors**: Hyejun Jeong, Mohammadreza Teymoorianfard, Abhinav Kumar, Amir Houmansadr, Eugene Bagdasarian

**Abstract**: We show that Web and Research Agents (WRAs) -- language model-based systems that investigate complex topics on the Internet -- are vulnerable to inference attacks by passive network adversaries such as ISPs. These agents could be deployed locally by organizations and individuals for privacy, legal, or financial purposes. Unlike sporadic web browsing by humans, WRAs visit $70{-}140$ domains with distinguishable timing correlations, enabling unique fingerprinting attacks.   Specifically, we demonstrate a novel prompt and user trait leakage attack against WRAs that only leverages their network-level metadata (i.e., visited IP addresses and their timings). We start by building a new dataset of WRA traces based on user search queries and queries generated by synthetic personas. We define a behavioral metric (called OBELS) to comprehensively assess similarity between original and inferred prompts, showing that our attack recovers over 73% of the functional and domain knowledge of user prompts. Extending to a multi-session setting, we recover up to 19 of 32 latent traits with high accuracy. Our attack remains effective under partial observability and noisy conditions. Finally, we discuss mitigation strategies that constrain domain diversity or obfuscate traces, showing negligible utility impact while reducing attack effectiveness by an average of 29%.

摘要: 我们表明，Web和研究代理（WRA）--调查互联网上复杂主题的基于语言模型的系统--很容易受到ISP等被动网络对手的推理攻击。这些代理可以由组织和个人出于隐私、法律或财务目的在本地部署。与人类零星的网络浏览不同，WRA访问具有可区分的时间相关性的价值70 {-}140美元的域名，从而实现独特的指纹攻击。   具体来说，我们展示了针对WRA的新型提示和用户特征泄露攻击，该攻击仅利用其网络级元数据（即，访问的IP地址及其时间）。我们首先根据用户搜索查询和合成人物角色生成的查询构建新的WRA痕迹数据集。我们定义了一个行为指标（称为OBELS）来全面评估原始提示和推断提示之间的相似性，表明我们的攻击恢复了用户提示73%以上的功能和领域知识。扩展到多会话设置，我们可以高准确性地恢复32个潜在特征中的多达19个。我们的攻击在部分可观察性和噪音条件下仍然有效。最后，我们讨论了限制域多样性或混淆痕迹的缓解策略，显示出可忽略的实用性影响，同时平均将攻击有效性降低29%。



## **32. See No Evil: Adversarial Attacks Against Linguistic-Visual Association in Referring Multi-Object Tracking Systems**

看不到邪恶：引用多对象跟踪系统中对语言视觉关联的对抗攻击 cs.CV

12 pages, 1 figure, 3 tables

**SubmitDate**: 2025-09-03    [abs](http://arxiv.org/abs/2509.02028v2) [paper-pdf](http://arxiv.org/pdf/2509.02028v2)

**Authors**: Halima Bouzidi, Haoyu Liu, Mohammad Abdullah Al Faruque

**Abstract**: Language-vision understanding has driven the development of advanced perception systems, most notably the emerging paradigm of Referring Multi-Object Tracking (RMOT). By leveraging natural-language queries, RMOT systems can selectively track objects that satisfy a given semantic description, guided through Transformer-based spatial-temporal reasoning modules. End-to-End (E2E) RMOT models further unify feature extraction, temporal memory, and spatial reasoning within a Transformer backbone, enabling long-range spatial-temporal modeling over fused textual-visual representations. Despite these advances, the reliability and robustness of RMOT remain underexplored. In this paper, we examine the security implications of RMOT systems from a design-logic perspective, identifying adversarial vulnerabilities that compromise both the linguistic-visual referring and track-object matching components. Additionally, we uncover a novel vulnerability in advanced RMOT models employing FIFO-based memory, whereby targeted and consistent attacks on their spatial-temporal reasoning introduce errors that persist within the history buffer over multiple subsequent frames. We present VEIL, a novel adversarial framework designed to disrupt the unified referring-matching mechanisms of RMOT models. We show that carefully crafted digital and physical perturbations can corrupt the tracking logic reliability, inducing track ID switches and terminations. We conduct comprehensive evaluations using the Refer-KITTI dataset to validate the effectiveness of VEIL and demonstrate the urgent need for security-aware RMOT designs for critical large-scale applications.

摘要: 图像视觉理解推动了高级感知系统的发展，最引人注目的是参考多对象跟踪（RMOT）的新兴范式。通过利用自然语言查询，RMOT系统可以在基于Transformer的时空推理模块的指导下选择性地跟踪满足给定语义描述的对象。端到端（E2 E）RMOT模型进一步统一了Transformer骨干中的特征提取、时间记忆和空间推理，从而在融合的文本-视觉表示上实现了长距离时空建模。尽管有这些进步，RMOT的可靠性和鲁棒性仍然没有得到充分的研究。在本文中，我们从设计逻辑的角度研究了RMOT系统的安全影响，识别了损害语言视觉引用和跟踪对象匹配组件的对抗漏洞。此外，我们还发现了使用基于FIFO的内存的高级RMOT模型中的一个新漏洞，从而对其时空推理进行有针对性且一致的攻击，从而引入了在历史缓冲区中持续存在的错误。我们提出了VEIL，这是一种新型对抗框架，旨在破坏RMOT模型的统一触发匹配机制。我们表明，精心设计的数字和物理扰动可能会破坏跟踪逻辑的可靠性，导致轨道ID开关和端接。我们使用Refer-KITTI数据集进行全面评估，以验证VEIL的有效性，并证明关键大规模应用对安全感知RMOT设计的迫切需求。



## **33. Near-Optimal Stability for Distributed Transaction Processing in Blockchain Sharding**

区块链碎片中分布式事务处理的近优稳定性 cs.DC

13 pages, 1 figure, accepted for publication in Proceedings of the  27th International Symposium on Stabilization, Safety, and Security of  Distributed Systems (SSS 2025)

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02421v1) [paper-pdf](http://arxiv.org/pdf/2509.02421v1)

**Authors**: Ramesh Adhikari, Costas Busch, Dariusz R. Kowalski

**Abstract**: In blockchain sharding, $n$ processing nodes are divided into $s$ shards, and each shard processes transactions in parallel. A key challenge in such a system is to ensure system stability for any ``tractable'' pattern of generated transactions; this is modeled by an adversary generating transactions with a certain rate of at most $\rho$ and burstiness $b$. This model captures worst-case scenarios and even some attacks on transactions' processing, e.g., DoS. A stable system ensures bounded transaction queue sizes and bounded transaction latency. It is known that the absolute upper bound on the maximum injection rate for which any scheduler could guarantee bounded queues and latency of transactions is $\max\left\{ \frac{2}{k+1}, \frac{2}{ \left\lfloor\sqrt{2s}\right\rfloor}\right\}$, where $k$ is the maximum number of shards that each transaction accesses. Here, we first provide a single leader scheduler that guarantees stability under injection rate $\rho \leq \max\left\{ \frac{1}{16k}, \frac{1}{16\lceil \sqrt{s} \rceil}\right\}$. Moreover, we also give a distributed scheduler with multiple leaders that guarantees stability under injection rate $\rho \leq \frac{1}{16c_1 \log D \log s}\max\left\{ \frac{1}{k}, \frac{1}{\lceil \sqrt{s} \rceil} \right\}$, where $c_1$ is some positive constant and $D$ is the diameter of shard graph $G_s$. This bound is within a poly-log factor from the optimal injection rate, and significantly improves the best previous known result for the distributed setting by Adhikari et al., SPAA 2024.

摘要: In blockchain sharding, $n$ processing nodes are divided into $s$ shards, and each shard processes transactions in parallel. A key challenge in such a system is to ensure system stability for any ``tractable'' pattern of generated transactions; this is modeled by an adversary generating transactions with a certain rate of at most $\rho$ and burstiness $b$. This model captures worst-case scenarios and even some attacks on transactions' processing, e.g., DoS. A stable system ensures bounded transaction queue sizes and bounded transaction latency. It is known that the absolute upper bound on the maximum injection rate for which any scheduler could guarantee bounded queues and latency of transactions is $\max\left\{ \frac{2}{k+1}, \frac{2}{ \left\lfloor\sqrt{2s}\right\rfloor}\right\}$, where $k$ is the maximum number of shards that each transaction accesses. Here, we first provide a single leader scheduler that guarantees stability under injection rate $\rho \leq \max\left\{ \frac{1}{16k}, \frac{1}{16\lceil \sqrt{s} \rceil}\right\}$. Moreover, we also give a distributed scheduler with multiple leaders that guarantees stability under injection rate $\rho \leq \frac{1}{16c_1 \log D \log s}\max\left\{ \frac{1}{k}, \frac{1}{\lceil \sqrt{s} \rceil} \right\}$, where $c_1$ is some positive constant and $D$ is the diameter of shard graph $G_s$. This bound is within a poly-log factor from the optimal injection rate, and significantly improves the best previous known result for the distributed setting by Adhikari et al., SPAA 2024.



## **34. A Survey: Towards Privacy and Security in Mobile Large Language Models**

调查：移动大型语言模型中的隐私和安全 cs.CR

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02411v1) [paper-pdf](http://arxiv.org/pdf/2509.02411v1)

**Authors**: Honghui Xu, Kaiyang Li, Wei Chen, Danyang Zheng, Zhiyuan Li, Zhipeng Cai

**Abstract**: Mobile Large Language Models (LLMs) are revolutionizing diverse fields such as healthcare, finance, and education with their ability to perform advanced natural language processing tasks on-the-go. However, the deployment of these models in mobile and edge environments introduces significant challenges related to privacy and security due to their resource-intensive nature and the sensitivity of the data they process. This survey provides a comprehensive overview of privacy and security issues associated with mobile LLMs, systematically categorizing existing solutions such as differential privacy, federated learning, and prompt encryption. Furthermore, we analyze vulnerabilities unique to mobile LLMs, including adversarial attacks, membership inference, and side-channel attacks, offering an in-depth comparison of their effectiveness and limitations. Despite recent advancements, mobile LLMs face unique hurdles in achieving robust security while maintaining efficiency in resource-constrained environments. To bridge this gap, we propose potential applications, discuss open challenges, and suggest future research directions, paving the way for the development of trustworthy, privacy-compliant, and scalable mobile LLM systems.

摘要: 移动大型语言模型（LLM）正在彻底改变医疗保健、金融和教育等各个领域，因为它们能够随时执行高级自然语言处理任务。然而，由于这些模型的资源密集型性质和处理数据的敏感性，在移动和边缘环境中部署这些模型会带来与隐私和安全相关的重大挑战。本调查全面概述了与移动LLM相关的隐私和安全问题，系统地对现有解决方案进行分类，例如差异隐私、联合学习和提示加密。此外，我们还分析了移动LLM特有的漏洞，包括对抗性攻击、成员资格推断和侧通道攻击，并对其有效性和局限性进行了深入比较。尽管最近取得了进步，但移动LLM在实现强大的安全性同时在资源有限的环境中保持效率方面面临着独特的障碍。为了弥合这一差距，我们提出了潜在的应用程序，讨论了开放的挑战，并提出了未来的研究方向，为开发值得信赖、符合隐私和可扩展的移动LLM系统铺平了道路。



## **35. Enhancing Security in Multi-Robot Systems through Co-Observation Planning, Reachability Analysis, and Network Flow**

通过协同观察规划、可达性分析和网络流增强多机器人系统的安全性 cs.RO

12 pages, 6 figures, submitted to IEEE Transactions on Control of  Network Systems

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2403.13266v2) [paper-pdf](http://arxiv.org/pdf/2403.13266v2)

**Authors**: Ziqi Yang, Roberto Tron

**Abstract**: This paper addresses security challenges in multi-robot systems (MRS) where adversaries may compromise robot control, risking unauthorized access to forbidden areas. We propose a novel multi-robot optimal planning algorithm that integrates mutual observations and introduces reachability constraints for enhanced security. This ensures that, even with adversarial movements, compromised robots cannot breach forbidden regions without missing scheduled co-observations. The reachability constraint uses ellipsoidal over-approximation for efficient intersection checking and gradient computation. To enhance system resilience and tackle feasibility challenges, we also introduce sub-teams. These cohesive units replace individual robot assignments along each route, enabling redundant robots to deviate for co-observations across different trajectories, securing multiple sub-teams without requiring modifications. We formulate the cross-trajectory co-observation plan by solving a network flow coverage problem on the checkpoint graph generated from the original unsecured MRS trajectories, providing the same security guarantees against plan-deviation attacks. We demonstrate the effectiveness and robustness of our proposed algorithm, which significantly strengthens the security of multi-robot systems in the face of adversarial threats.

摘要: 本文解决了多机器人系统（MMS）中的安全挑战，其中对手可能会危及机器人控制，从而冒着未经授权访问禁区的风险。我们提出了一种新型的多机器人最优规划算法，该算法集成了相互观察并引入可达性约束以增强安全性。这确保了，即使存在对抗性运动，受影响的机器人也无法在不错过预定的共同观察的情况下突破禁区。可达性约束使用椭圆体过逼近来进行高效的相交检查和梯度计算。为了增强系统弹性并应对可行性挑战，我们还引入了子团队。这些有凝聚力的单元取代每条路线上的单个机器人任务，使多余的机器人能够在不同轨迹上偏离共同观察，从而在不需要修改的情况下保护多个子团队。我们通过解决从原始不安全的MES轨迹生成的检查点图上的网络流覆盖问题来制定跨轨迹共同观察计划，从而提供相同的安全保证来防止计划偏差攻击。我们证明了我们提出的算法的有效性和鲁棒性，该算法显着增强了多机器人系统在面对对抗威胁时的安全性。



## **36. Enhancing Reliability in LLM-Integrated Robotic Systems: A Unified Approach to Security and Safety**

提高LLM集成机器人系统的可靠性：统一的安全保障方法 cs.RO

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02163v1) [paper-pdf](http://arxiv.org/pdf/2509.02163v1)

**Authors**: Wenxiao Zhang, Xiangrui Kong, Conan Dewitt, Thomas Bräunl, Jin B. Hong

**Abstract**: Integrating large language models (LLMs) into robotic systems has revolutionised embodied artificial intelligence, enabling advanced decision-making and adaptability. However, ensuring reliability, encompassing both security against adversarial attacks and safety in complex environments, remains a critical challenge. To address this, we propose a unified framework that mitigates prompt injection attacks while enforcing operational safety through robust validation mechanisms. Our approach combines prompt assembling, state management, and safety validation, evaluated using both performance and security metrics. Experiments show a 30.8% improvement under injection attacks and up to a 325% improvement in complex environment settings under adversarial conditions compared to baseline scenarios. This work bridges the gap between safety and security in LLM-based robotic systems, offering actionable insights for deploying reliable LLM-integrated mobile robots in real-world settings. The framework is open-sourced with simulation and physical deployment demos at https://llmeyesim.vercel.app/

摘要: 将大型语言模型（LLM）集成到机器人系统中彻底改变了具体人工智能，实现了高级决策和适应性。然而，确保可靠性（包括对抗攻击的安全性和复杂环境中的安全性）仍然是一个严峻的挑战。为了解决这个问题，我们提出了一个统一的框架，该框架可以减轻即时注入攻击，同时通过强大的验证机制强制执行操作安全。我们的方法结合了即时组装、状态管理和安全验证，并使用性能和安全指标进行评估。实验表明，与基线场景相比，在注入攻击下的性能提高了30.8%，在对抗条件下的复杂环境设置中的性能提高了325%。这项工作弥合了基于LLM的机器人系统的安全性和安全性之间的差距，为在现实环境中部署可靠的LLM集成移动机器人提供了可操作的见解。该框架是开源的，在https://llmeyesim.vercel.app/上有模拟和物理部署演示



## **37. Targeted Physical Evasion Attacks in the Near-Infrared Domain**

近红外领域有针对性的物理规避攻击 cs.CR

To appear in the Proceedings of the Network and Distributed Systems  Security Symposium (NDSS) 2026

**SubmitDate**: 2025-09-02    [abs](http://arxiv.org/abs/2509.02042v1) [paper-pdf](http://arxiv.org/pdf/2509.02042v1)

**Authors**: Pascal Zimmer, Simon Lachnit, Alexander Jan Zielinski, Ghassan Karame

**Abstract**: A number of attacks rely on infrared light sources or heat-absorbing material to imperceptibly fool systems into misinterpreting visual input in various image recognition applications. However, almost all existing approaches can only mount untargeted attacks and require heavy optimizations due to the use-case-specific constraints, such as location and shape. In this paper, we propose a novel, stealthy, and cost-effective attack to generate both targeted and untargeted adversarial infrared perturbations. By projecting perturbations from a transparent film onto the target object with an off-the-shelf infrared flashlight, our approach is the first to reliably mount laser-free targeted attacks in the infrared domain. Extensive experiments on traffic signs in the digital and physical domains show that our approach is robust and yields higher attack success rates in various attack scenarios across bright lighting conditions, distances, and angles compared to prior work. Equally important, our attack is highly cost-effective, requiring less than US\$50 and a few tens of seconds for deployment. Finally, we propose a novel segmentation-based detection that thwarts our attack with an F1-score of up to 99%.

摘要: 许多攻击依赖于红外光源或吸热材料，以难以察觉的方式欺骗系统误解各种图像识别应用中的视觉输入。然而，几乎所有现有方法只能发起无目标攻击，并且由于特定用例的限制（例如位置和形状）而需要进行大量优化。在本文中，我们提出了一种新颖的、隐蔽的、具有成本效益的攻击来生成有针对性和无针对性的对抗性红外扰动。通过用现成的红外手电筒将透明薄膜的扰动投射到目标物体上，我们的方法是第一个在红外域中可靠地发动无激光定向攻击的方法。对数字和物理领域交通标志的广泛实验表明，与之前的工作相比，我们的方法稳健，在明亮的照明条件、距离和角度的各种攻击场景中可以产生更高的攻击成功率。同样重要的是，我们的攻击具有极高的成本效益，部署所需时间不到50美元，只需几十秒。最后，我们提出了一种新颖的基于分段的检测，可以以高达99%的F1评分阻止我们的攻击。



## **38. Adversarial Attacks and Defenses in Multivariate Time-Series Forecasting for Smart and Connected Infrastructures**

智能和互联基础设施多元时间序列预测中的对抗性攻击和防御 cs.LG

18 pages, 34 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2408.14875v2) [paper-pdf](http://arxiv.org/pdf/2408.14875v2)

**Authors**: Pooja Krishan, Rohan Mohapatra, Sanchari Das, Saptarshi Sengupta

**Abstract**: The emergence of deep learning models has revolutionized various industries over the last decade, leading to a surge in connected devices and infrastructures. However, these models can be tricked into making incorrect predictions with high confidence, leading to disastrous failures and security concerns. To this end, we explore the impact of adversarial attacks on multivariate time-series forecasting and investigate methods to counter them. Specifically, we employ untargeted white-box attacks, namely the Fast Gradient Sign Method (FGSM) and the Basic Iterative Method (BIM), to poison the inputs to the training process, effectively misleading the model. We also illustrate the subtle modifications to the inputs after the attack, which makes detecting the attack using the naked eye quite difficult. Having demonstrated the feasibility of these attacks, we develop robust models through adversarial training and model hardening. We are among the first to showcase the transferability of these attacks and defenses by extrapolating our work from the benchmark electricity data to a larger, 10-year real-world data used for predicting the time-to-failure of hard disks. Our experimental results confirm that the attacks and defenses achieve the desired security thresholds, leading to a 72.41% and 94.81% decrease in RMSE for the electricity and hard disk datasets respectively after implementing the adversarial defenses.

摘要: 深度学习模型的出现在过去十年里彻底改变了各个行业，导致互联设备和基础设施的激增。然而，这些模型可能会被欺骗以高置信度做出错误的预测，从而导致灾难性的失败和安全问题。为此，我们探讨了对抗性攻击对多元时间序列预测的影响，并研究应对它们的方法。具体来说，我们使用无针对性的白盒攻击，即快速梯度符号法（FGSM）和基本迭代法（BMI），来毒害训练过程的输入，从而有效地误导模型。我们还说明了攻击后对输入的微妙修改，这使得使用肉眼检测攻击变得相当困难。在证明了这些攻击的可行性后，我们通过对抗训练和模型硬化开发了稳健的模型。我们是最早展示这些攻击和防御可移植性的公司之一，通过将我们的工作从基准电力数据外推到用于预测硬盘故障时间的更大的10年现实世界数据。我们的实验结果证实，攻击和防御都达到了预期的安全阈值，实施对抗性防御后，电力和硬盘数据集的RME分别下降了72.41%和94.81%。



## **39. Evaluating the Defense Potential of Machine Unlearning against Membership Inference Attacks**

评估机器取消学习针对成员推断攻击的防御潜力 cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2508.16150v2) [paper-pdf](http://arxiv.org/pdf/2508.16150v2)

**Authors**: Aristeidis Sidiropoulos, Christos Chrysanthos Nikolaidis, Theodoros Tsiolakis, Nikolaos Pavlidis, Vasilis Perifanis, Pavlos S. Efraimidis

**Abstract**: Membership Inference Attacks (MIAs) pose a significant privacy risk, as they enable adversaries to determine whether a specific data point was included in the training dataset of a model. While Machine Unlearning is primarily designed as a privacy mechanism to efficiently remove private data from a machine learning model without the need for full retraining, its impact on the susceptibility of models to MIA remains an open question. In this study, we systematically assess the vulnerability of models to MIA after applying state-of-art Machine Unlearning algorithms. Our analysis spans four diverse datasets (two from the image domain and two in tabular format), exploring how different unlearning approaches influence the exposure of models to membership inference. The findings highlight that while Machine Unlearning is not inherently a countermeasure against MIA, the unlearning algorithm and data characteristics can significantly affect a model's vulnerability. This work provides essential insights into the interplay between Machine Unlearning and MIAs, offering guidance for the design of privacy-preserving machine learning systems.

摘要: 会员推断攻击（MIA）构成了重大的隐私风险，因为它们使对手能够确定特定数据点是否包含在模型的训练数据集中。虽然Machine Unlearning主要被设计为一种隐私机制，用于有效地从机器学习模型中删除私人数据，而不需要完全重新培训，但它对模型对MIA敏感性的影响仍然是一个悬而未决的问题。在这项研究中，我们在应用最先进的机器取消学习算法后系统评估了模型对MIA的脆弱性。我们的分析跨越了四个不同的数据集（两个来自图像域，两个以表格形式），探索不同的去学习方法如何影响模型对隶属推理的暴露。研究结果强调，虽然机器取消学习本质上并不是针对MIA的对策，但取消学习算法和数据特征可能会显着影响模型的脆弱性。这项工作为机器非学习和MIA之间的相互作用提供了重要见解，为保护隐私的机器学习系统的设计提供了指导。



## **40. Addressing Key Challenges of Adversarial Attacks and Defenses in the Tabular Domain: A Methodological Framework for Coherence and Consistency**

应对表格领域对抗性攻击和防御的关键挑战：一致性和一致性的方法论框架 cs.LG

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2412.07326v3) [paper-pdf](http://arxiv.org/pdf/2412.07326v3)

**Authors**: Yael Itzhakev, Amit Giloni, Yuval Elovici, Asaf Shabtai

**Abstract**: Machine learning models trained on tabular data are vulnerable to adversarial attacks, even in realistic scenarios where attackers only have access to the model's outputs. Since tabular data contains complex interdependencies among features, it presents a unique challenge for adversarial samples which must maintain coherence and respect these interdependencies to remain indistinguishable from benign data. Moreover, existing attack evaluation metrics-such as the success rate, perturbation magnitude, and query count-fail to account for this challenge. To address those gaps, we propose a technique for perturbing dependent features while preserving sample coherence. In addition, we introduce Class-Specific Anomaly Detection (CSAD), an effective novel anomaly detection approach, along with concrete metrics for assessing the quality of tabular adversarial attacks. CSAD evaluates adversarial samples relative to their predicted class distribution, rather than a broad benign distribution. It ensures that subtle adversarial perturbations, which may appear coherent in other classes, are correctly identified as anomalies. We integrate SHAP explainability techniques to detect inconsistencies in model decision-making, extending CSAD for SHAP-based anomaly detection. Our evaluation incorporates both anomaly detection rates with SHAP-based assessments to provide a more comprehensive measure of adversarial sample quality. We evaluate various attack strategies, examining black-box query-based and transferability-based gradient attacks across four target models. Experiments on benchmark tabular datasets reveal key differences in the attacker's risk and effort and attack quality, offering insights into the strengths, limitations, and trade-offs faced by attackers and defenders. Our findings lay the groundwork for future research on adversarial attacks and defense development in the tabular domain.

摘要: 在表格数据上训练的机器学习模型很容易受到对抗攻击，即使在攻击者只能访问模型输出的现实场景中也是如此。由于表格数据包含特征之间复杂的相互依赖关系，因此它对对抗性样本提出了独特的挑战，这些样本必须保持一致性并尊重这些相互依赖关系，以保持与良性数据之间无法区分。此外，现有的攻击评估指标--例如成功率、扰动幅度和查询计数--无法解决这一挑战。为了解决这些差距，我们提出了一种在保持样本一致性的同时扰动相关特征的技术。此外，我们还引入了类别特定异常检测（CSAD），这是一种有效的新型异常检测方法，以及用于评估表格对抗攻击质量的具体指标。CSAD相对于其预测的类别分布而不是广泛的良性分布来评估敌对样本。它确保细微的对抗性扰动（在其他类别中可能看起来是一致的）被正确识别为异常。我们集成了SHAP可解释性技术来检测模型决策中的不一致性，扩展了CSAD用于基于SHAP的异常检测。我们的评估结合了异常检测率和基于SHAP的评估，以提供对抗性样本质量的更全面的衡量标准。我们评估了各种攻击策略，检查了四个目标模型中基于黑匣子查询和基于可移植性的梯度攻击。对基准表格数据集的实验揭示了攻击者的风险和努力以及攻击质量的关键差异，从而深入了解攻击者和防御者面临的优势、限制和权衡。我们的发现为未来对表格领域的对抗性攻击和防御开发的研究奠定了基础。



## **41. SoK: Cybersecurity Assessment of Humanoid Ecosystem**

SoK：类人生物生态系统的网络安全评估 cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2508.17481v2) [paper-pdf](http://arxiv.org/pdf/2508.17481v2)

**Authors**: Priyanka Prakash Surve, Asaf Shabtai, Yuval Elovici

**Abstract**: Humanoids are progressing toward practical deployment across healthcare, industrial, defense, and service sectors. While typically considered cyber-physical systems (CPSs), their dependence on traditional networked software stacks (e.g., Linux operating systems), robot operating system (ROS) middleware, and over-the-air update channels, creates a distinct security profile that exposes them to vulnerabilities conventional CPS models do not fully address. Prior studies have mainly examined specific threats, such as LiDAR spoofing or adversarial machine learning (AML). This narrow focus overlooks how an attack targeting one component can cascade harm throughout the robot's interconnected systems. We address this gap through a systematization of knowledge (SoK) that takes a comprehensive approach, consolidating fragmented research from robotics, CPS, and network security domains. We introduce a seven-layer security model for humanoid robots, organizing 39 known attacks and 35 defenses across the humanoid ecosystem-from hardware to human-robot interaction. Building on this security model, we develop a quantitative 39x35 attack-defense matrix with risk-weighted scoring, validated through Monte Carlo analysis. We demonstrate our method by evaluating three real-world robots: Pepper, G1 EDU, and Digit. The scoring analysis revealed varying security maturity levels, with scores ranging from 39.9% to 79.5% across the platforms. This work introduces a structured, evidence-based assessment method that enables systematic security evaluation, supports cross-platform benchmarking, and guides prioritization of security investments in humanoid robotics.

摘要: 类人猿正在向医疗保健、工业、国防和服务业的实际部署迈进。虽然通常被认为是网络物理系统（CPS），但它们对传统网络软件栈的依赖（例如，Linux操作系统）、机器人操作系统（LOS）中间件和空中更新通道创建了一个独特的安全配置文件，使它们暴露在传统CPS模型无法完全解决的漏洞中。之前的研究主要检查了特定的威胁，例如LiDART欺骗或对抗性机器学习（ML）。这种狭隘的焦点忽视了针对一个组件的攻击如何对整个机器人的互连系统造成连锁伤害。我们通过知识系统化（SoK）来解决这一差距，该知识采用全面的方法，整合机器人、CPS和网络安全领域的碎片化研究。我们为人形机器人引入了一个七层安全模型，组织了整个人形生态系统中的39种已知攻击和35种防御--从硬件到人机交互。在此安全模型的基础上，我们开发了一个具有风险加权评分的量化39 x35攻击-防御矩阵，并通过蒙特卡洛分析进行验证。我们通过评估三个现实世界的机器人：Pepper、G1 EDU和Digit来演示我们的方法。评分分析显示，各个平台的安全成熟度水平各不相同，评分范围从39.9%到79.5%不等。这项工作引入了一种结构化的、基于证据的评估方法，可以实现系统性的安全评估，支持跨平台基准测试，并指导人形机器人安全投资的优先顺序。



## **42. Privacy-preserving authentication for military 5G networks**

军用5G网络隐私保护认证 cs.CR

To appear in Proc. IEEE Military Commun. Conf. (MILCOM), (Los  Angeles, CA), Oct. 2025

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01470v1) [paper-pdf](http://arxiv.org/pdf/2509.01470v1)

**Authors**: I. D. Lutz, A. M. Hill, M. C. Valenti

**Abstract**: As 5G networks gain traction in defense applications, ensuring the privacy and integrity of the Authentication and Key Agreement (AKA) protocol is critical. While 5G AKA improves upon previous generations by concealing subscriber identities, it remains vulnerable to replay-based synchronization and linkability threats under realistic adversary models. This paper provides a unified analysis of the standardized 5G AKA flow, identifying several vulnerabilities and highlighting how each exploits protocol behavior to compromise user privacy. To address these risks, we present five lightweight mitigation strategies. We demonstrate through prototype implementation and testing that these enhancements strengthen resilience against linkability attacks with minimal computational and signaling overhead. Among the solutions studied, those introducing a UE-generated nonce emerge as the most promising, effectively neutralizing the identified tracking and correlation attacks with negligible additional overhead. Integrating this extension as an optional feature to the standard 5G AKA protocol offers a backward-compatible, low-overhead path toward a more privacy-preserving authentication framework for both commercial and military 5G deployments.

摘要: 随着5G网络在国防应用中获得关注，确保认证和密钥协议（AKA）协议的隐私性和完整性至关重要。虽然5G AKA通过隐藏用户身份对前几代产品进行了改进，但在现实的对手模型下，它仍然容易受到基于回放的同步和链接性威胁。本文对标准化的5G AKA流程进行了统一分析，识别了几个漏洞，并强调了每个漏洞如何利用协议行为来损害用户隐私。为了解决这些风险，我们提出了五种轻量级缓解策略。我们通过原型实现和测试证明，这些增强功能以最小的计算和信号负担增强了抵御链接性攻击的弹性。在所研究的解决方案中，引入UE生成的随机数的解决方案是最有希望的，可以有效地中和已识别的跟踪和相关攻击，而额外费用可以忽略不计。将此扩展集成为标准5G AKA协议的可选功能，为商业和军用5G部署提供了一条向后兼容、低开销的路径，以建立更保护隐私的身份验证框架。



## **43. LLMHoney: A Real-Time SSH Honeypot with Large Language Model-Driven Dynamic Response Generation**

LLMHoney：具有大型语言模型驱动动态响应生成的实时SSH蜜罐 cs.CR

7 Pages

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01463v1) [paper-pdf](http://arxiv.org/pdf/2509.01463v1)

**Authors**: Pranjay Malhotra

**Abstract**: Cybersecurity honeypots are deception tools for engaging attackers and gather intelligence, but traditional low or medium-interaction honeypots often rely on static, pre-scripted interactions that can be easily identified by skilled adversaries. This Report presents LLMHoney, an SSH honeypot that leverages Large Language Models (LLMs) to generate realistic, dynamic command outputs in real time. LLMHoney integrates a dictionary-based virtual file system to handle common commands with low latency while using LLMs for novel inputs, achieving a balance between authenticity and performance. We implemented LLMHoney using open-source LLMs and evaluated it on a testbed with 138 representative Linux commands. We report comprehensive metrics including accuracy (exact-match, Cosine Similarity, Jaro-Winkler Similarity, Levenshtein Similarity and BLEU score), response latency and memory overhead. We evaluate LLMHoney using multiple LLM backends ranging from 0.36B to 3.8B parameters, including both open-source models and a proprietary model(Gemini). Our experiments compare 13 different LLM variants; results show that Gemini-2.0 and moderately-sized models Qwen2.5:1.5B and Phi3:3.8B provide the most reliable and accurate responses, with mean latencies around 3 seconds, whereas smaller models often produce incorrect or out-of-character outputs. We also discuss how LLM integration improves honeypot realism and adaptability compared to traditional honeypots, as well as challenges such as occasional hallucinated outputs and increased resource usage. Our findings demonstrate that LLM-driven honeypots are a promising approach to enhance attacker engagement and collect richer threat intelligence.

摘要: 网络安全蜜罐是用于吸引攻击者和收集情报的欺骗工具，但传统的低或中等交互蜜罐通常依赖于静态的、预先脚本化的交互，这些交互可以被熟练的对手轻松识别。本报告介绍了LLMHoney，这是一个SSH蜜罐，利用大型语言模型（LLM）实时生成真实、动态的命令输出。LLMHoney集成了基于字典的虚拟文件系统，以低延迟处理常见命令，同时使用LLM进行新型输入，实现真实性和性能之间的平衡。我们使用开源LLM实现了LLMHoney，并在具有138个代表性的Linux命令的测试床上对其进行了评估。我们报告了全面的指标，包括准确性（精确匹配、Cosine相似性、Jaro-Winkler相似性、Levenshtein相似性和BLEU评分）、响应延迟和内存负载。我们使用从0.36B到3.8B参数的多个LLM后台来评估LLMHoney，包括开源模型和专有模型（Gemini）。我们的实验比较了13种不同的LLM变体;结果表明，Gemini-2.0和中等大小的模型Qwen 2.5：1.5B和Phi 3：3.8B提供了最可靠和准确的响应，平均延迟时间约为3秒，而较小的模型通常会产生错误或不合字符的输出。我们还讨论了与传统蜜罐相比，LLM集成如何提高蜜罐的真实性和适应性，以及偶尔幻觉输出和资源使用增加等挑战。我们的研究结果表明，LLM驱动的蜜罐是增强攻击者参与度和收集更丰富威胁情报的一种有希望的方法。



## **44. An Automated Attack Investigation Approach Leveraging Threat-Knowledge-Augmented Large Language Models**

利用威胁知识增强大型语言模型的自动攻击调查方法 cs.CR

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01271v1) [paper-pdf](http://arxiv.org/pdf/2509.01271v1)

**Authors**: Rujie Dai, Peizhuo Lv, Yujiang Gui, Qiujian Lv, Yuanyuan Qiao, Yan Wang, Degang Sun, Weiqing Huang, Yingjiu Li, XiaoFeng Wang

**Abstract**: Advanced Persistent Threats (APTs) are prolonged, stealthy intrusions by skilled adversaries that compromise high-value systems to steal data or disrupt operations. Reconstructing complete attack chains from massive, heterogeneous logs is essential for effective attack investigation, yet existing methods suffer from poor platform generality, limited generalization to evolving tactics, and an inability to produce analyst-ready reports. Large Language Models (LLMs) offer strong semantic understanding and summarization capabilities, but in this domain they struggle to capture the long-range, cross-log dependencies critical for accurate reconstruction.   To solve these problems, we present an LLM-empowered attack investigation framework augmented with a dynamically adaptable Kill-Chain-aligned threat knowledge base. We organizes attack-relevant behaviors into stage-aware knowledge units enriched with semantic annotations, enabling the LLM to iteratively retrieve relevant intelligence, perform causal reasoning, and progressively expand the investigation context. This process reconstructs multi-phase attack scenarios and generates coherent, human-readable investigation reports. Evaluated on 15 attack scenarios spanning single-host and multi-host environments across Windows and Linux (over 4.3M log events, 7.2 GB of data), the system achieves an average True Positive Rate (TPR) of 97.1% and an average False Positive Rate (FPR) of 0.2%, significantly outperforming the SOTA method ATLAS, which achieves an average TPR of 79.2% and an average FPR of 29.1%.

摘要: 高级持续性威胁（APT）是技术精湛的对手发起的长期、秘密入侵，这些入侵会危及高价值系统以窃取数据或扰乱运营。从大量、异类的日志中重建完整的攻击链对于有效的攻击调查至关重要，但现有方法存在平台通用性较差、对不断发展的策略的概括性有限以及无法生成可供分析师使用的报告的问题。大型语言模型（LLM）提供强大的语义理解和总结能力，但在该领域，它们很难捕捉对准确重建至关重要的长期、跨日志依赖关系。   为了解决这些问题，我们提出了一个LLM授权的攻击调查框架，该框架增强了动态自适应的杀戮链对齐威胁知识库。我们将与攻击相关的行为组织到富含语义注释的阶段感知知识单元中，使LLM能够迭代地检索相关情报、执行因果推理并逐步扩展调查上下文。该过程重建多阶段攻击场景并生成连贯、人类可读的调查报告。评估了跨Windows和Linux的单主机和多主机环境的15种攻击场景（超过430万个日志事件，7.2 GB数据），该系统的平均真阳性率（TPR）为97.1%，平均假阳性率（FPR）为0.2%，显着优于SOTA方法ATLAS，平均TPR为79.2%，平均FPR为29.1%。



## **45. Geometric origin of adversarial vulnerability in deep learning**

深度学习中对抗脆弱性的几何起源 cs.LG

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2509.01235v1) [paper-pdf](http://arxiv.org/pdf/2509.01235v1)

**Authors**: Yixiong Ren, Wenkang Du, Jianhui Zhou, Haiping Huang

**Abstract**: How to balance training accuracy and adversarial robustness has become a challenge since the birth of deep learning. Here, we introduce a geometry-aware deep learning framework that leverages layer-wise local training to sculpt the internal representations of deep neural networks. This framework promotes intra-class compactness and inter-class separation in feature space, leading to manifold smoothness and adversarial robustness against white or black box attacks. The performance can be explained by an energy model with Hebbian coupling between elements of the hidden representation. Our results thus shed light on the physics of learning in the direction of alignment between biological and artificial intelligence systems. Using the current framework, the deep network can assimilate new information into existing knowledge structures while reducing representation interference.

摘要: 自深度学习诞生以来，如何平衡训练准确性和对抗鲁棒性就成为一个挑战。在这里，我们引入了一个几何感知的深度学习框架，该框架利用分层本地训练来塑造深度神经网络的内部表示。该框架促进了特征空间中的类内紧凑性和类间分离，从而实现了对白盒或黑盒攻击的多维平滑性和对抗鲁棒性。性能可以通过隐藏表示元素之间具有赫布式耦合的能量模型来解释。因此，我们的结果揭示了生物和人工智能系统之间协调方向的学习物理学。使用当前框架，深度网络可以将新信息吸收到现有知识结构中，同时减少表示干扰。



## **46. Crosstalk Attacks and Defence in a Shared Quantum Computing Environment**

共享量子计算环境中的串话攻击和防御 quant-ph

13 pages, 7 figures

**SubmitDate**: 2025-09-01    [abs](http://arxiv.org/abs/2402.02753v2) [paper-pdf](http://arxiv.org/pdf/2402.02753v2)

**Authors**: Benjamin Harper, Behnam Tonekaboni, Bahar Goldozian, Martin Sevior, Muhammad Usman

**Abstract**: Quantum computing has the potential to provide solutions to problems that are intractable on classical computers, but the accuracy of the current generation of quantum computers suffer from the impact of noise or errors such as leakage, crosstalk, dephasing, and amplitude damping among others. As the access to quantum computers is almost exclusively in a shared environment through cloud-based services, it is possible that an adversary can exploit crosstalk noise to disrupt quantum computations on nearby qubits, even carefully designing quantum circuits to purposely lead to wrong answers. In this paper, we analyze the extent and characteristics of crosstalk noise through tomography conducted on IBM Quantum computers, leading to an enhanced crosstalk simulation model. Our results indicate that crosstalk noise is a significant source of errors on IBM quantum hardware, making crosstalk based attack a viable threat to quantum computing in a shared environment. Based on our crosstalk simulator benchmarked against IBM hardware, we assess the impact of crosstalk attacks and develop strategies for mitigating crosstalk effects. Through a systematic set of simulations, we assess the effectiveness of three crosstalk attack mitigation strategies, namely circuit separation, qubit allocation optimization via reinforcement learning, and the use of spectator qubits, and show that they all overcome crosstalk attacks with varying degrees of success and help to secure quantum computing in a shared platform.

摘要: 量子计算有潜力为经典计算机上棘手的问题提供解决方案，但当前一代量子计算机的准确性受到泄漏、串话、去相和幅度衰减等噪音或误差的影响。由于量子计算机的访问几乎完全是在共享环境中通过基于云的服务进行的，因此对手可能会利用串话噪音来破坏附近量子位上的量子计算，甚至精心设计量子电路来故意导致错误的答案。在本文中，我们通过在IBM Quantum计算机上进行的断层扫描分析了串话噪音的程度和特征，从而建立了增强的串话模拟模型。我们的结果表明，串话噪音是IBM量子硬件上错误的重要来源，使得基于串话的攻击成为共享环境中量子计算的可行威胁。基于我们以IBM硬件为基准的串话模拟器，我们评估串话攻击的影响并开发减轻串话影响的策略。通过一组系统的模拟，我们评估了三种串话攻击缓解策略的有效性，即电路分离、通过强化学习进行量子位分配优化以及使用旁观量子位，并表明它们都以不同程度的成功克服了串话攻击，并有助于在共享平台中保护量子计算。



## **47. Clone What You Can't Steal: Black-Box LLM Replication via Logit Leakage and Distillation**

克隆无法窃取的内容：通过Logit泄漏和蒸馏进行黑匣子LLM复制 cs.CR

8 pages. Accepted for publication in the proceedings of 7th IEEE  International Conference on Trust, Privacy and Security in Intelligent  Systems, and Applications (IEEE TPS 2025)

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00973v1) [paper-pdf](http://arxiv.org/pdf/2509.00973v1)

**Authors**: Kanchon Gharami, Hansaka Aluvihare, Shafika Showkat Moni, Berker Peköz

**Abstract**: Large Language Models (LLMs) are increasingly deployed in mission-critical systems, facilitating tasks such as satellite operations, command-and-control, military decision support, and cyber defense. Many of these systems are accessed through application programming interfaces (APIs). When such APIs lack robust access controls, they can expose full or top-k logits, creating a significant and often overlooked attack surface. Prior art has mainly focused on reconstructing the output projection layer or distilling surface-level behaviors. However, regenerating a black-box model under tight query constraints remains underexplored. We address that gap by introducing a constrained replication pipeline that transforms partial logit leakage into a functional deployable substitute model clone. Our two-stage approach (i) reconstructs the output projection matrix by collecting top-k logits from under 10k black-box queries via singular value decomposition (SVD) over the logits, then (ii) distills the remaining architecture into compact student models with varying transformer depths, trained on an open source dataset. A 6-layer student recreates 97.6% of the 6-layer teacher model's hidden-state geometry, with only a 7.31% perplexity increase, and a 7.58 Negative Log-Likelihood (NLL). A 4-layer variant achieves 17.1% faster inference and 18.1% parameter reduction with comparable performance. The entire attack completes in under 24 graphics processing unit (GPU) hours and avoids triggering API rate-limit defenses. These results demonstrate how quickly a cost-limited adversary can clone an LLM, underscoring the urgent need for hardened inference APIs and secure on-premise defense deployments.

摘要: 大型语言模型（LLM）越来越多地部署在关键任务系统中，促进卫星操作、指挥与控制、军事决策支持和网络防御等任务。其中许多系统都是通过应用程序编程接口（API）访问的。当此类API缺乏强大的访问控制时，它们可能会暴露完整或顶级k日志，从而创建一个重要且经常被忽视的攻击面。现有技术主要集中在重建输出投影层或提取表面级行为。然而，在严格的查询约束下重新生成黑匣子模型仍然没有得到充分的探索。我们通过引入一个受约束的复制管道来解决这一差距，该管道将部分logit泄漏转换为功能性可部署的替代模型克隆。我们的两阶段方法（i）通过对logit进行奇异值分解（DID）从10 k以下的黑匣子查询中收集前k个logit来重建输出投影矩阵，然后（ii）将剩余的架构提炼成具有不同Transformer深度的紧凑学生模型，在开源数据集上训练。一个6层的学生可以重建97.6%的6层教师模型的隐藏状态几何，而困惑度只增加了7.31%，负对数似然（NLL）为7.58。4层变体在相当的性能下实现了17.1%的推理速度和18.1%的参数减少。整个攻击只需不到24个图形处理单元（图形处理单元）小时即可完成，并避免触发API速率限制防御。这些结果展示了成本有限的对手可以多快地克隆LLM，凸显了对强化推理API和安全本地防御部署的迫切需求。



## **48. FusionCounting: Robust visible-infrared image fusion guided by crowd counting via multi-task learning**

FusionCounting：通过多任务学习由人群计数引导的鲁棒可见光-红外图像融合 cs.CV

11 pages, 9 figures

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2508.20817v2) [paper-pdf](http://arxiv.org/pdf/2508.20817v2)

**Authors**: He Li, Xinyu Liu, Weihang Kong, Xingchen Zhang

**Abstract**: Visible and infrared image fusion (VIF) is an important multimedia task in computer vision. Most VIF methods focus primarily on optimizing fused image quality. Recent studies have begun incorporating downstream tasks, such as semantic segmentation and object detection, to provide semantic guidance for VIF. However, semantic segmentation requires extensive annotations, while object detection, despite reducing annotation efforts compared with segmentation, faces challenges in highly crowded scenes due to overlapping bounding boxes and occlusion. Moreover, although RGB-T crowd counting has gained increasing attention in recent years, no studies have integrated VIF and crowd counting into a unified framework. To address these challenges, we propose FusionCounting, a novel multi-task learning framework that integrates crowd counting into the VIF process. Crowd counting provides a direct quantitative measure of population density with minimal annotation, making it particularly suitable for dense scenes. Our framework leverages both input images and population density information in a mutually beneficial multi-task design. To accelerate convergence and balance tasks contributions, we introduce a dynamic loss function weighting strategy. Furthermore, we incorporate adversarial training to enhance the robustness of both VIF and crowd counting, improving the model's stability and resilience to adversarial attacks. Experimental results on public datasets demonstrate that FusionCounting not only enhances image fusion quality but also achieves superior crowd counting performance.

摘要: 可见光和红外图像融合（VIF）是计算机视觉中一项重要的多媒体任务。大多数VIF方法主要关注优化融合图像质量。最近的研究已经开始整合下游任务，例如语义分割和对象检测，为VIF提供语义指导。然而，语义分割需要大量的注释，而对象检测尽管与分割相比减少了注释工作，但由于重叠的边界框和遮挡，在高度拥挤的场景中面临挑战。此外，尽管近年来RGB-T人群计数越来越受到关注，但还没有研究将VIF和人群计数整合到统一的框架中。为了应对这些挑战，我们提出FusionCounting，这是一种新型的多任务学习框架，可将人群计数集成到VIF流程中。人群计数提供了人口密度的直接定量测量，只需最少的注释，使其特别适合密集场景。我们的框架在互利的多任务设计中利用输入图像和人口密度信息。为了加速收敛和平衡任务贡献，我们引入了动态损失函数加权策略。此外，我们结合了对抗性训练来增强VIF和人群计数的鲁棒性，提高了模型的稳定性和对抗性攻击的弹性。在公开数据集上的实验结果表明，FusionCounting不仅提高了图像融合质量，而且实现了优越的人群计数性能。



## **49. Redesigning Traffic Signs to Mitigate Machine-Learning Patch Attacks**

重新设计交通标志以缓解机器学习补丁攻击 cs.CR

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2402.04660v3) [paper-pdf](http://arxiv.org/pdf/2402.04660v3)

**Authors**: Tsufit Shua, Liron David, Mahmood Sharif

**Abstract**: Traffic-Sign Recognition (TSR) is a critical safety component for autonomous driving. Unfortunately, however, past work has highlighted the vulnerability of TSR models to physical-world attacks, through low-cost, easily deployable adversarial patches leading to misclassification. To mitigate these threats, most defenses focus on altering the training process or modifying the inference procedure. Still, while these approaches improve adversarial robustness, TSR remains susceptible to attacks attaining substantial success rates.   To further the adversarial robustness of TSR, this work offers a novel approach that redefines traffic-sign designs to create signs that promote robustness while remaining interpretable to humans. Our framework takes three inputs: (1) A traffic-sign standard along with modifiable features and associated constraints; (2) A state-of-the-art adversarial training method; and (3) A function for efficiently synthesizing realistic traffic-sign images. Using these user-defined inputs, the framework emits an optimized traffic-sign standard such that traffic signs generated per this standard enable training TSR models with increased adversarial robustness.   We evaluate the effectiveness of our framework via a concrete implementation, where we allow modifying the pictograms (i.e., symbols) and colors of traffic signs. The results show substantial improvements in robustness -- with gains of up to 16.33%--24.58% in robust accuracy over state-of-the-art methods -- while benign accuracy is even improved. Importantly, a user study also confirms that the redesigned traffic signs remain easily recognizable and to human observers. Overall, the results highlight that carefully redesigning traffic signs can significantly enhance TSR system robustness without compromising human interpretability.

摘要: 手势识别（TSB）是自动驾驶的关键安全组件。然而不幸的是，过去的工作强调了TSB模型对物理世界攻击的脆弱性，因为低成本、易于部署的对抗补丁导致了错误分类。为了减轻这些威胁，大多数防御措施都集中在改变训练过程或修改推理程序上。尽管如此，虽然这些方法提高了对抗鲁棒性，但TSB仍然容易受到获得相当大成功率的攻击。   为了进一步增强TSB的对抗鲁棒性，这项工作提供了一种新的方法，重新定义交通标志设计，以创建既能提高鲁棒性又能为人类解释的标志。我们的框架需要三个输入：（1）交通标志标准以及可修改的特征和相关的约束;（2）最先进的对抗训练方法;（3）用于有效合成真实交通标志图像的功能。使用这些用户定义的输入，该框架发布优化的交通标志标准，以便按照该标准生成的交通标志能够训练具有更高的对抗鲁棒性的TSB模型。   我们通过具体实现来评估框架的有效性，其中我们允许修改象形图（即，符号）和交通标志的颜色。结果显示鲁棒性有了大幅提高--鲁棒准确性比最先进的方法提高了16.33%--24.58%--而良性准确性甚至得到了提高。重要的是，一项用户研究还证实，重新设计的交通标志仍然容易被人类观察者识别。总体而言，结果强调，仔细重新设计交通标志可以显着增强TSB系统的稳健性，而不会损害人类的解释性。



## **50. Sequential Difference Maximization: Generating Adversarial Examples via Multi-Stage Optimization**

序列差异最大化：通过多阶段优化生成对抗性示例 cs.CV

5 pages, 2 figures, 5 tables, CIKM 2025

**SubmitDate**: 2025-08-31    [abs](http://arxiv.org/abs/2509.00826v1) [paper-pdf](http://arxiv.org/pdf/2509.00826v1)

**Authors**: Xinlei Liu, Tao Hu, Peng Yi, Weitao Han, Jichao Xie, Baolin Li

**Abstract**: Efficient adversarial attack methods are critical for assessing the robustness of computer vision models. In this paper, we reconstruct the optimization objective for generating adversarial examples as "maximizing the difference between the non-true labels' probability upper bound and the true label's probability," and propose a gradient-based attack method termed Sequential Difference Maximization (SDM). SDM establishes a three-layer optimization framework of "cycle-stage-step." The processes between cycles and between iterative steps are respectively identical, while optimization stages differ in terms of loss functions: in the initial stage, the negative probability of the true label is used as the loss function to compress the solution space; in subsequent stages, we introduce the Directional Probability Difference Ratio (DPDR) loss function to gradually increase the non-true labels' probability upper bound by compressing the irrelevant labels' probabilities. Experiments demonstrate that compared with previous SOTA methods, SDM not only exhibits stronger attack performance but also achieves higher attack cost-effectiveness. Additionally, SDM can be combined with adversarial training methods to enhance their defensive effects. The code is available at https://github.com/X-L-Liu/SDM.

摘要: 有效的对抗攻击方法对于评估计算机视觉模型的稳健性至关重要。本文将生成对抗性示例的优化目标重建为“最大化非真标签概率上限和真标签概率之间的差异”，并提出了一种基于梯度的攻击方法，称为序列差异最大化（Sendential Difference Maximation）。SDP建立了“周期-阶段-步骤”的三层优化框架。“循环之间和迭代步骤之间的过程分别相同，而优化阶段在损失函数方面有所不同：在初始阶段，使用真实标签的负概率作为损失函数来压缩解空间;在随后的阶段，我们引入方向概率差比（DPDR）损失函数来逐步增加非通过压缩不相关标签的概率来确定真实标签的概率上限。实验表明，与之前的SOTA方法相比，Sends不仅具有更强的攻击性能，而且具有更高的攻击性价比。此外，SDM可以与对抗性训练方法相结合，以增强其防御效果。该代码可在https://github.com/X-L-Liu/SDM上获取。



