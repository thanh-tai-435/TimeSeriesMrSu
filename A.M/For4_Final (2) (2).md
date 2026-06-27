

# MỤC LỤC

[LỜI CẢM ƠN 4](#_Toc203222130)

[LỜI CAM KẾT 5](#_Toc203222131)

[MỤC LỤC 6](#_Toc203222132)

[DANH MỤC HÌNH ẢNH 8](#_Toc203222133)

[DANH MỤC BẢNG BIỂU 11](#_Toc203222134)

[DANH MỤC VIẾT TẮT 12](#_Toc203222135)

[TÓM TẮT 13](#_Toc203222136)

[CHƯƠNG 1. GIỚI THIỆU VẤN ĐỀ NGHIÊN CỨU 14](#_Toc203222137)

[1.1. Tính mới của đề tài 14](#_Toc203222138)

[1.2. Tính cấp thiết của đề tài 15](#_Toc203222139)

[1.3. Vấn đề được nghiên cứu 15](#_Toc203222140)

[1.4. Đối tượng, phạm vi nghiên cứu; sơ lược lịch sử nghiên cứu 16](#_Toc203222141)

[1.5. Lý do nghiên cứu: Vị trí, vai trò và tầm quan trọng của đề tài 16](#_Toc203222142)

[CHƯƠNG 2: TỔNG QUAN TÌNH HÌNH NGHIÊN CỨU 18](#_Toc203222143)

[2.1. Cơ sở lý luận: Các khái niệm, định nghĩa, kiến thức nền tảng về vấn đề được nghiên cứu 18](#_Toc203222144)

[2.1.1. Giao Dịch Thuật Toán (Algorithmic Trading) 18](#_Toc203222145)

[2.1.2. Lý Thuyết Giao Dịch Forex 26](#_Toc203222146)

[2.1.3. Các Chỉ Số Kỹ Thuật Quan Trọng Trong Forex 28](#_Toc203222147)

[2.1.4. Hiệu quả của hệ thống giao dịch kết hợp mẫu hình Engulfing, EMA200 và RSI trong giao dịch Forex: Nghiên cứu backtest 33](#_Toc203222148)

[2.1.5. Long Short Term Memory (LSTM) 33](#_Toc203222149)

[2.1.6. Recurrent Neural Network (RNN) 33](#_Toc203222150)

[2.1.7. Gated Recurrent Unit (GRU) 36](#_Toc203222151)

[2.1.8. Random Forest 39](#_Toc203222152)

[2.1.9 AI Agent 41](#_Toc203222153)

[2.1.10 Trợ lý thông minh AI Agent 42](#_Toc203222154)

[2.2. Thực trạng vấn đề nghiên cứu 44](#_Toc203222155)

[CHƯƠNG 3. PHƯƠNG PHÁP NGHIÊN CỨU 47](#_Toc203222156)

[3.1. Bối cảnh nghiên cứu 47](#_Toc203222157)

[3.2. Phương pháp thu thập số liệu 47](#_Toc203222158)

[3.2.1. Thu thập dữ liệu 47](#_Toc203222159)

[3.2.2. Xử lý dữ liệu 50](#_Toc203222160)

[CHƯƠNG 4. KẾT QUẢ VÀ ĐÁNH GIÁ 60](#_Toc203222161)

[4.1. Kết quả của mô hình baseline: Sử dụng đường SMA 60](#_Toc203222162)

[4.2. Kết quả mô hình 62](#_Toc203222163)

[4.2.1. Kết quả của mô hình Random Forest 62](#_Toc203222164)

[4.2.2. Kết quả của mô hình GRU 65](#_Toc203222165)

[4.2.3. Kết quả của mô hình RNN 67](#_Toc203222166)

[4.3 Đánh giá các mô hình 71](#_Toc203222167)

[4.4 Chức năng thực hiện backtest 71](#_Toc203222168)

[CHƯƠNG 5. ỨNG DỤNG AI AGENT 78](#_Toc203222169)

[CHƯƠNG 6: KẾT LUẬN VÀ KHUYẾN NGHỊ 104](#_Toc203222170)

[6.1. Kết luận 104](#_Toc203222171)

[6.2. Khuyến nghị 104](#_Toc203222172)

[TÀI LIỆU THAM KHẢO 106](#_Toc203222173)

# DANH MỤC HÌNH ẢNH

[Hình 3. 1 Minh họa API lấy data 47](#_Toc203222350)

[Hình 3. 2 Lấy nến của cặp tiền tệ 48](#_Toc203222351)

[Hình 3. 3 Thu thập và thêm cột vào dữ liệu 49](#_Toc203222352)

[Hình 3. 4 Kết quả lấy data 50](#_Toc203222353)

[Hình 3. 5 Ảnh dữ liệu sau khi được thêm cột 51](#_Toc203222354)

[Hình 3. 6 Khai phá dữ liệu 53](#_Toc203222355)

[Hình 3. 7 Thông tin chi tiết dữ liệu 54](#_Toc203222356)

[Hình 3. 8 Thống kê mô tả dữ liệu 55](#_Toc203222357)

[Hình 3. 9 Biến động tỷ giá đóng cửa trung bình (Mid Close Price) của cặp tiền tệ EUR/USD trong giai đoạn từ năm 2008 đến 2025 56](#_Toc203222358)

[Hình 3. 10 Biểu đồ histogram của biến "mid_c" 57](#_Toc203222359)

[Hình 3. 11 Biên độ dao động giá theo giờ (Hourly Price Range) của tỷ giá EUR/USD dựa trên giá mid trong suốt giai đoạn từ năm 2008 đến 2025 58](#_Toc203222360)

[Hình 3. 12 Biểu đồ thể hiện phân phối lợi nhuận theo giờ (Hourly Returns) 59](#_Toc203222361)

[Hình 4. 1 Kết quả của mô hình baseline 60](#_Toc203222368)

[Hình 4. 2 So sánh giá trị thực tế và dự đoán sử dụng mô hình Random Forest 62](#_Toc203222369)

[Hình 4. 3 So sánh giá trị thực tế và dự đoán sử dụng mô hình GRU 65](#_Toc203222370)

[Hình 4. 4 So sánh giá trị thực tế và dự đoán của mô hình RNN 68](#_Toc203222371)

[Hình 4. 5 So Sánh giá trị thực tế và dự đoán EURUSD H1 (01/10/2024 – 13/05/2025) - LSTM 69](#_Toc203222372)

[Hình 4. 6 Chiến lược backtest 1 72](#_Toc203222373)

[Hình 4. 7 Kết quả chiến lược backtest 1 73](#_Toc203222374)

[Hình 4. 8 Chiến lược backtest 2 75](#_Toc203222375)

[Hình 4. 9 Kết quả chiến lược backtest 2 77](#_Toc203222376)

[Hình 5. 1 Giao diện API Forex và Huấn luyện nâng cao 78](#_Toc203222377)

[Hình 5. 2 Giao diện API huấn luyện và backtest mô hình LSTM trên dữ liệu Forex. 79](#_Toc203222378)

[Hình 5. 3 Giao diện API quản lý dữ liệu Forex. 80](#_Toc203222379)

[Hình 5. 4 Giao diện các API phục vụ dự đoán giá Forex. 80](#_Toc203222380)

[Hình 5. 5 Cấu hình yêu cầu API cho dự báo LSTM - EUR/USD 82](#_Toc203222381)

[Hình 5. 6 Phản hồi API - Quá trình huấn luyện mô hình RNN cho EUR/USD 83](#_Toc203222382)

[Hình 5. 7 Phản hồi API - Các chỉ số hiệu suất mô hình cho EUR/USD 85](#_Toc203222383)

[Hình 5. 8 Nội dung yêu cầu API - Cơ chế lọc theo khoảng thời gian 86](#_Toc203222384)

[Hình 5. 9 Yêu cầu POST API - Khởi tạo thu thập dữ liệu ngoại hối 87](#_Toc203222385)

[Hình 5. 10 Yêu cầu POST API - Kích hoạt cập nhật dữ liệu ngoại hối 89](#_Toc203222386)

[Hình 5. 11 Kết quả API Trạng thái Ngoại hối 91](#_Toc203222387)

[Hình 5. 12 Giao diện nhập tham số 92](#_Toc203222388)

[Hình 5. 13 Chi tiết phản hồi API cho EURUSD 93](#_Toc203222389)

[Hình 5. 14 Tổng quan quy trình 94](#_Toc203222390)

[Hình 5. 15 Thống kê thông tin của các cặp tiền hiện có 95](#_Toc203222391)

[Hình 5. 16 Update data cho tới thời điểm bấm execute 96](#_Toc203222392)

[Hình 5. 17 Thực hiện training model LSTM, đưa ra giá dự đoán của ngày tiếp theo 97](#_Toc203222393)

[Hình 5. 18 Kết quả kiểm thử hồi cứu (backtest) mô hình LSTM 98](#_Toc203222394)

[Hình 5. 19 Biểu đồ kết quả backtest mô hình LSTM trên cặp EUR/USD 99](#_Toc203222395)

[Hình 5. 20 Đánh giá chiến lược LSTM attention 100](#_Toc203222396)

[Hình 5. 21 Khuyến nghị điều chỉnh thông số mô hình LSTM 101](#_Toc203222397)

[Hình 5. 22 Lý do khuyến nghị và kết luận backtest LSTM 102](#_Toc203222398)

# DANH MỤC BẢNG BIỂU

[Bảng 2. 1 Cấu trúc đầu vào – đầu vào RNN 34](#_Toc203222399)

[Bảng 4. 1 Đánh giá các mô hình 71](#_Toc203222400)

# DANH MỤC VIẾT TẮT

|     |     |
| --- | --- |
| **Viết tắt** | **Ý nghĩa** |
| AI  | Artificial Intelligence – Trí tuệ nhân tạo |
| API | Application Programming Interface – Giao diện lập trình ứng dụng |
| ATR | Average True Range – Chỉ báo đo độ biến động |
| EMA | Exponential Moving Average – Đường trung bình lũy thừa |
| EMA200 | Đường EMA với chu kỳ 200 |
| FX  | Foreign Exchange – Thị trường ngoại hối |
| GRU | Gated Recurrent Unit – Mạng nơ-ron hồi tiếp có cổng |
| HFT | High-Frequency Trading – Giao dịch tần suất cao |
| LSTM | Long Short-Term Memory – Mạng nơ-ron bộ nhớ dài ngắn |
| MA  | Moving Average – Đường trung bình động |
| MACD | Moving Average Convergence Divergence – Chỉ báo phân kỳ hội tụ MA |
| OBV | On-Balance Volume – Chỉ báo khối lượng tích lũy |
| OOS | Out-of-Sample – Kiểm tra ngoài mẫu |
| RNN | Recurrent Neural Network – Mạng nơ-ron hồi tiếp |
| RSI | Relative Strength Index – Chỉ số sức mạnh tương đối |
| SMA | Simple Moving Average – Đường trung bình giản đơn |
| VPS | Virtual Private Server – Máy chủ riêng ảo |
| OOB | Out-of-Bag – Mẫu ngoài túi (trong Random Forest) |

# TÓM TẮT

Đề tài tập trung xây dựng một hệ thống AI Agent tích hợp trên nền tảng n8n nhằm dự báo biến động tỷ giá Forex theo thời gian thực và tự động đưa ra khuyến nghị giao dịch. Hệ thống khai thác các nguồn dữ liệu đáng tin cậy như giá Forex thời gian thực, các chỉ báo kỹ thuật (RSI, EMA200, Bollinger Bands, ATR...) và tin tức tài chính để phân tích toàn diện thị trường. Đặc biệt, đề tài ứng dụng các mô hình học máy và deep learning như Random Forest, RNN, GRU, LSTM để nâng cao độ chính xác dự báo.

Ngoài ra, chatbot giao dịch được phát triển với giao diện trực quan, dễ sử dụng, hỗ trợ cả người mới lẫn nhà giao dịch chuyên nghiệp. Giải pháp không chỉ giúp nhà đầu tư cá nhân tiếp cận các công cụ giao dịch hiện đại vốn chỉ phổ biến ở các tổ chức lớn, mà còn góp phần thu hẹp khoảng cách công nghệ, tối ưu quá trình ra quyết định và giảm thiểu rủi ro trong môi trường Forex đầy biến động.

# CHƯƠNG 1. GIỚI THIỆU VẤN ĐỀ NGHIÊN CỨU

## 1.1. Tính mới của đề tài

Trong những năm gần đây, thị trường ngoại hối (Forex) tại Việt Nam ghi nhận tốc độ phát triển nhanh chóng, thu hút sự tham gia ngày càng đông đảo của các nhà đầu tư cá nhân. Theo thống kê từ Ngân hàng Nhà nước Việt Nam, khối lượng giao dịch trên thị trường này đã tăng trưởng trung bình khoảng 20% mỗi năm trong ba năm trở lại đây. Điều này phản ánh xu hướng số hóa mạnh mẽ trong hoạt động tài chính cá nhân cũng như sự quan tâm ngày càng lớn đối với các kênh đầu tư phi truyền thống.

Tuy nhiên, bên cạnh sự tăng trưởng về số lượng, chất lượng công cụ hỗ trợ giao dịch cho nhóm nhà đầu tư cá nhân vẫn còn nhiều hạn chế. Trong khi các tổ chức tài chính lớn thường sở hữu hệ thống giao dịch tiên tiến, tích hợp dữ liệu thời gian thực và thuật toán phân tích phức tạp, thì phần lớn nhà giao dịch cá nhân tại Việt Nam vẫn phải phụ thuộc vào các nền tảng cơ bản với tính năng đơn giản, khó đáp ứng yêu cầu trong môi trường thị trường biến động mạnh và phức tạp.

Thị trường Forex luôn chịu ảnh hưởng sâu sắc từ các yếu tố vĩ mô như chính sách tiền tệ, điều chỉnh lãi suất, biến động kinh tế toàn cầu… đòi hỏi người tham gia phải có khả năng phân tích nhanh chóng và phản ứng kịp thời. Chính trong bối cảnh đó, nhu cầu về một nền tảng giao dịch thông minh – tích hợp dữ liệu thời gian thực, khả năng phân tích kỹ thuật và tự động hóa quy trình giao dịch – trở nên cấp thiết.

Việc phát triển hệ thống AI Agent dự báo biến động tỷ giá Forex theo thời gian thực và đưa ra khuyến nghị giao dịch là hướng tiếp cận mới nhằm giải quyết những hạn chế nêu trên. Không chỉ giúp thu hẹp khoảng cách công nghệ giữa nhà đầu tư cá nhân và các tổ chức lớn, hệ thống này còn tạo ra lợi thế cạnh tranh thông qua việc tự động phân tích, đề xuất giao dịch, và phản hồi linh hoạt theo thời gian thực. Quan trọng hơn, đề tài nghiên cứu còn góp phần thúc đẩy quá trình ứng dụng công nghệ số vào lĩnh vực đầu tư tài chính, đặc biệt là trong bối cảnh thị trường Việt Nam đang trên đà hội nhập và đổi mới mạnh mẽ.

##   
1.2. Tính cấp thiết của đề tài

Đề tài nghiên cứu này đem lại nhiều điểm khác biệt và đột phá so với các nền tảng giao dịch đang được sử dụng phổ biến tại Việt Nam hiện nay. Một trong những yếu tố nổi bật là khả năng tích hợp đồng bộ từ nhiều nguồn dữ liệu đáng tin cậy, cụ thể gồm:  
● Giá giao dịch theo thời gian thực từ các sàn lớn như FXOpen.  
● Hệ thống chỉ báo kỹ thuật chuyên sâu từ Investing.com.  
● Các bản tin tài chính cập nhật từ Reuters.

Việc kết hợp các nguồn thông tin này giúp hình thành một hệ sinh thái phân tích toàn diện, mang đến cho người dùng cái nhìn đa chiều và sâu sắc về thị trường. Trong khi đó, phần lớn các công cụ giao dịch hiện có trong nước chỉ dừng lại ở việc hiển thị dữ liệu giá cơ bản và một vài chỉ báo đơn lẻ, chưa hỗ trợ tích hợp tin tức thời sự – yếu tố quan trọng trong việc đưa ra quyết định kịp thời.

Thêm vào đó, chatbot được phát triển với giao diện hiện đại, dễ thao tác nhờ công nghệ React, mang lại trải nghiệm mượt mà cùng hệ thống biểu đồ, bảng điều khiển sinh động và rõ ràng. Thiết kế thân thiện này giúp ứng dụng trở nên phù hợp với nhiều đối tượng người dùng – từ người mới bắt đầu đến các nhà giao dịch chuyên nghiệp – và khắc phục được hạn chế của nhiều nền tảng hiện nay vốn có giao diện phức tạp và thiếu thân thiện với người dùng phổ thông.

##   
1.3. Vấn đề được nghiên cứu

Thị trường ngoại hối (Forex) được xem là một trong những thị trường tài chính có quy mô lớn nhất trên thế giới, với khối lượng giao dịch mỗi ngày lên đến hàng nghìn tỷ đô la Mỹ. Tuy nhiên, đặc điểm nổi bật của thị trường này là tính biến động cao và sự phức tạp trong các yếu tố tác động, điều này đặt ra thách thức lớn đối với các nhà giao dịch trong việc ra quyết định đầu tư một cách kịp thời và chính xác.

Trước thực tiễn đó, nghiên cứu này tập trung vào việc phát triển chatbot tích hợp giao dịch, hướng đến giải quyết đồng thời ba yếu tố cốt lõi: (1) thu thập và xử lý dữ liệu thời gian thực, (2) triển khai phân tích kỹ thuật chuyên sâu, và (3) xây dựng cơ chế giao dịch tự động dựa trên chiến lược định sẵn. Mục tiêu chung là tạo ra một hệ thống phần mềm thông minh, có khả năng cung cấp thông tin thị trường một cách liên tục và chính xác, đồng thời hỗ trợ tối ưu hoá quá trình giao dịch thông qua tự động hóa, qua đó nâng cao hiệu quả đầu tư và giảm thiểu rủi ro cho người sử dụng.

## 1.4. Đối tượng, phạm vi nghiên cứu; sơ lược lịch sử nghiên cứu

Đối tượng nghiên cứu của đề tài là một hệ thống chatbot giao dịch Forex tích hợp, được xây dựng trên nền tảng n8n và các AI Agent, với mục tiêu hỗ trợ nhà giao dịch cá nhân trong việc ra quyết định mua, bán và thực hiện giao dịch tự động trên thị trường ngoại hối. Chatbot này được thiết kế không chỉ để trả lời câu hỏi liên quan đến xu hướng thị trường, mà còn có khả năng phân tích kỹ thuật, đánh giá tín hiệu giao dịch và đưa ra đề xuất hành động dựa trên dữ liệu tài chính thời gian thực. Hệ thống ứng dụng kết hợp khả năng xử lý quy trình tự động của n8n với trí tuệ nhân tạo để tối ưu hoá trải nghiệm giao dịch cho người dùng cuối.

Phạm vi nghiên cứu của đề tài tập trung vào các khía cạnh kỹ thuật trong quá trình thiết kế, phát triển và thử nghiệm hệ thống AI Agent nói trên. Các thử nghiệm được triển khai trong môi trường mô phỏng với dữ liệu nến lịch sử, chỉ báo kỹ thuật và thông tin thị trường, nhằm đánh giá tính khả thi và độ chính xác trước khi tiến hành triển khai thực tế vào hoạt động giao dịch.

## 1.5. Lý do nghiên cứu: Vị trí, vai trò và tầm quan trọng của đề tài

Trong bối cảnh nhu cầu về các ứng dụng công nghệ tài chính (fintech) ngày càng gia tăng, vấn đề phát triển một AI Agent giao dịch Forex tích hợp giữ vai trò then chốt, đặc biệt đối với các nhà giao dịch cá nhân. Việc tiếp cận thông tin thị trường theo thời gian thực và thực hiện giao dịch hiệu quả là yếu tố quyết định đến thành công trong môi trường cạnh tranh cao của thị trường ngoại hối. Tuy nhiên, so với các tổ chức tài chính lớn có lợi thế về công nghệ, dữ liệu và hệ thống xử lý tự động, các nhà đầu tư cá nhân thường gặp bất lợi rõ rệt.

Xuất phát từ thực tiễn đó, nghiên cứu này được thực hiện nhằm xây dựng một công cụ hỗ trợ giao dịch mạnh mẽ, góp phần thu hẹp khoảng cách công nghệ giữa các tổ chức lớn và nhà giao dịch nhỏ lẻ. Ứng dụng được kỳ vọng sẽ cung cấp dữ liệu tài chính chính xác, phân tích kỹ thuật chuyên sâu, đồng thời tự động hóa các quyết định giao dịch, từ đó nâng cao hiệu quả đầu tư, giảm thiểu sai sót và rủi ro do con người.

Tầm quan trọng của đề tài còn thể hiện qua khả năng đáp ứng nhu cầu cấp thiết tại thị trường Việt Nam – nơi các công cụ giao dịch hiện đại vẫn còn tương đối khan hiếm. Việc phát triển một nền tảng giao dịch tích hợp như vậy không chỉ mang lại lợi ích thiết thực cho người dùng cá nhân mà còn góp phần thúc đẩy sự phát triển của thị trường Forex trong nước, đồng thời mở ra hướng tiếp cận mới trong lĩnh vực ứng dụng trí tuệ nhân tạo vào hoạt động đầu tư tài chính.

# CHƯƠNG 2: TỔNG QUAN TÌNH HÌNH NGHIÊN CỨU

## 2.1. Cơ sở lý luận: Các khái niệm, định nghĩa, kiến thức nền tảng về vấn đề được nghiên cứu

### 2.1.1. Giao Dịch Thuật Toán (Algorithmic Trading)

Giao dịch thuật toán, hay còn được biết đến với các thuật ngữ như giao dịch tự động hoặc algo trading, đại diện cho một bước tiến hóa quan trọng trong cách thức tương tác với thị trường ngoại hối (Forex). Về bản chất, đây là phương pháp sử dụng các chương trình máy tính, được xây dựng dựa trên các tập hợp quy tắc logic và toán học được xác định trước, để tự động hóa hoàn toàn hoặc một phần quy trình ra quyết định và thực thi lệnh giao dịch. Thay vì nhà giao dịch phải liên tục theo dõi biểu đồ và đưa ra phán đoán chủ quan, các thuật toán sẽ đảm nhận nhiệm vụ phân tích dữ liệu thị trường, xác định cơ hội và thực hiện giao dịch theo các tiêu chí đã được lập trình. Ví dụ kinh điển như việc tự động hóa chiến lược dựa trên sự giao cắt của các đường trung bình động (MA) hoặc tín hiệu từ chỉ số sức mạnh tương đối (RSI) chỉ là bề nổi;thực tế, giao dịch thuật toán trong Forex bao hàm một loạt các chiến lược và công nghệ phức tạp hơn nhiều.

Để một hệ thống giao dịch thuật toán hoạt động hiệu quả, nó cần được cấu thành từ nhiều bộ phận cốt lõi, phối hợp nhịp nhàng với nhau. Đầu tiên và quan trọng nhất là nguồn cung cấp dữ liệu đầu vào (Data Input). Hệ thống cần được "nuôi" bằng một luồng dữ liệu thị trường thời gian thực, chính xác và có độ trễ thấp, bao gồm giá bid/ask, khối lượng giao dịch, và lý tưởng nhất là cả dữ liệu sổ lệnh (order book data) để phân tích độ sâu thị trường. Bên cạnh dữ liệu giá cơ bản, các thuật toán phức tạp hơn còn tích hợp các nguồn dữ liệu phi cấu trúc hoặc dữ liệu thay thế. Ví dụ, việc sử dụng giao diện lập trình ứng dụng (API) từ các hãng tin tức uy tín như Reuters (được đề cập trong nghiên cứu này ) cho phép thuật toán phản ứng tức thời với các thông tin kinh tế vĩ mô hoặc sự kiện địa chính trị có khả năng tác động mạnh đến tỷ giá. Tương tự, dữ liệu về tâm lý thị trường, tổng hợp từ các nền tảng như Myfxbook (được sử dụng trong nghiên cứu), có thể cung cấp cái nhìn về trạng thái cảm xúc chung của đám đông giao dịch, một yếu tố mà nhiều mô hình thuật toán cố gắng khai thác hoặc đi ngược lại, tương tự như khái niệm "Wisdom of crowds" được Chen và cộng sự (2012) nghiên cứu trong bối cảnh thị trường chứng khoán qua mạng xã hội.

Thành phần tiếp theo là bộ não của hệ thống: logic tạo tín hiệu giao dịch (Signal Generation Logic). Đây là nơi các quy tắc và điều kiện được mã hóa thành thuật toán. Các quy tắc này có thể dựa trên phân tích kỹ thuật truyền thống, ví dụ như nhận dạng các mẫu hình nến (candlestick patterns) như Doji, Engulfing, Hammer, hoặc các mẫu hình biểu đồ phức tạp hơn như Head and Shoulders, Triangles, Flags. Ngoài ra, các chiến lược có thể dựa trên các mô hình thống kê, chẳng hạn như kinh doanh chênh lệch giá thống kê (statistical arbitrage) giữa các cặp tiền tệ có độ tương quan cao trong lịch sử (ví dụ: AUD/USD và NZD/USD, hoặc EUR/USD và GBP/USD trong một số giai đoạn nhất định), tìm kiếm sự phân kỳ tạm thời khỏi mối quan hệ cân bằng dài hạn. Các chiến lược dựa trên biến động giá, sử dụng các chỉ báo như Dải Bollinger (Bollinger Bands) hoặc Kênh Keltner (Keltner Channels) để giao dịch khi giá phá vỡ ra khỏi biên độ thông thường, cũng rất phổ biến. Chiến lược đảo chiều về trung bình (mean reversion) hoạt động dựa trên giả định rằng giá có xu hướng quay trở lại mức trung bình lịch sử sau những biến động mạnh, thường sử dụng các chỉ báo dao động như RSI hoặc Stochastic Oscillator, hoặc các thước đo thống kê như độ lệch chuẩn (z-score). Ngược lại, các chiến lược đi theo xu hướng (trend following) cố gắng xác định và bám theo các xu hướng mạnh mẽ đang diễn ra, sử dụng kết hợp nhiều đường MA, chỉ báo MACD (Moving Average Convergence Divergence), ADX (Average Directional Index), hoặc Parabolic SAR.

Khi một tín hiệu giao dịch được tạo ra, bộ phận thực thi lệnh (Order Execution Engine) sẽ chịu trách nhiệm gửi lệnh đến nhà môi giới (broker) thông qua API (như API của FXOpen đã được tích hợp trong dự án nghiên cứu). Quá trình này cần được tối ưu hóa để giảm thiểu độ trễ (latency) – khoảng thời gian từ khi tín hiệu được tạo ra đến khi lệnh được khớp trên thị trường – và trượt giá (slippage) – sự khác biệt giữa giá kỳ vọng thực hiện lệnh và giá thực tế lệnh được khớp. Các loại lệnh khác nhau (lệnh thị trường, lệnh giới hạn, lệnh dừng lỗ) được sử dụng tùy theo chiến lược và mục tiêu quản lý rủi ro. Không thể không nhắc đến mô-đun quản lý rủi ro (Risk Management Module), một thành phần tối quan trọng đảm bảo sự tồn tại lâu dài của hệ thống giao dịch. Thuật toán phải tự động tính toán khối lượng giao dịch (position sizing) phù hợp cho mỗi lệnh dựa trên mức độ rủi ro chấp nhận được (ví dụ: chỉ rủi ro 1-2% vốn cho mỗi giao dịch), áp dụng các mức dừng lỗ (stop-loss) cố định hoặc động (trailing stop) để giới hạn tổn thất tiềm năng. Các hệ thống phức tạp hơn còn có thể triển khai các kỹ thuật quản lý rủi ro động, ví dụ như điều chỉnh kích thước vị thế dựa trên mức độ biến động hiện tại của thị trường (thường đo bằng chỉ báo ATR - Average True Range) – vào lệnh với khối lượng nhỏ hơn khi thị trường biến động mạnh và lớn hơn khi thị trường ổn định. Việc đa dạng hóa chiến lược, chạy song song nhiều thuật toán không tương quan hoặc có tương quan thấp với nhau trên các cặp tiền tệ khác nhau, cũng là một phương pháp hiệu quả để giảm thiểu rủi ro tổng thể của danh mục. Ngoài ra, các quy tắc kiểm soát sụt giảm vốn (drawdown control) có thể được lập trình để tự động giảm quy mô giao dịch hoặc tạm dừng hoàn toàn hệ thống nếu mức thua lỗ vượt quá một ngưỡng cho phép trong một khoảng thời gian nhất định (ví dụ: trong ngày, trong tuần). Thuật toán cũng có thể kiểm tra mức độ tương quan giữa các vị thế đang mở để tránh tập trung rủi ro quá mức vào một hướng đi cụ thể của thị trường.

Sự đa dạng của giao dịch thuật toán trong Forex thể hiện qua nhiều loại hình chiến lược khác nhau. Bên cạnh các chiến lược đã đề cập như đi theo xu hướng hay đảo chiều trung bình, còn có các chiến lược kinh doanh chênh lệch giá (arbitrage) thuần túy, cố gắng khai thác những khác biệt giá cực nhỏ và tồn tại trong khoảnh khắc của cùng một cặp tiền tệ trên các sàn giao dịch khác nhau (latency arbitrage) hoặc giữa ba cặp tiền tệ tạo thành một vòng khép kín (triangular arbitrage). Chiến lược giao dịch dựa trên sự kiện (event-driven trading) tập trung vào việc dự đoán hoặc phản ứng cực nhanh với các tin tức kinh tế quan trọng; các thuật toán có thể được lập trình để tự động đặt lệnh mua hoặc bán chỉ vài mili giây sau khi dữ liệu thực tế (ví dụ: bảng lương phi nông nghiệp của Mỹ) được công bố, dựa trên sự khác biệt so với dự báo của thị trường. Một lĩnh vực khác là tạo lập thị trường (market making), nơi các thuật toán đặt đồng thời lệnh mua và lệnh bán với một khoảng chênh lệch nhỏ (spread), thu lợi nhuận từ spread này và khối lượng giao dịch lớn; chiến lược này thường đòi hỏi vốn lớn và cơ sở hạ tầng công nghệ tiên tiến. Đặc biệt, giao dịch tần suất cao (High-Frequency Trading - HFT) là một phân khúc riêng biệt, đặc trưng bởi tốc độ thực thi lệnh cực cao (tính bằng micro giây hoặc nano giây) và khối lượng giao dịch khổng lồ, nhằm khai thác những lợi thế cạnh tranh nhỏ nhất về giá hoặc thời gian. Nghiên cứu của Kearns và Nevmyvaka (2013) đã đi sâu vào việc ứng dụng học máy (machine learning) trong cấu trúc vi mô thị trường và HFT, cho thấy tiềm năng của các thuật toán thông minh trong việc xử lý và ra quyết định dựa trên lượng dữ liệu khổng lồ ở tốc độ cực cao.

Về mặt công nghệ, việc xây dựng và triển khai các hệ thống giao dịch thuật toán đòi hỏi kiến thức về lập trình và cơ sở hạ tầng. Các ngôn ngữ lập trình phổ biến bao gồm Python, với hệ sinh thái thư viện phong phú hỗ trợ phân tích dữ liệu (Pandas , NumPy), học máy (Scikit-learn) và kết nối API; C++ và Java thường được ưa chuộng cho các ứng dụng đòi hỏi hiệu năng và độ trễ cực thấp, đặc biệt là trong HFT; và MQL4/MQL5 là ngôn ngữ độc quyền cho nền tảng MetaTrader, rất phổ biến trong cộng đồng giao dịch cá nhân. Trước khi triển khai bằng tiền thật, việc kiểm thử lại (backtesting) chiến lược trên dữ liệu lịch sử là bước bắt buộc để đánh giá hiệu quả và độ tin cậy.

Quy trình backtesting bao gồm việc chạy thuật toán trên dữ liệu giá lịch sử trong một khoảng thời gian đủ dài và đa dạng để mô phỏng hoạt động của nó trong các điều kiện thị trường khác nhau. Kết quả backtest được đánh giá thông qua nhiều chỉ số hiệu suất như tổng lợi nhuận, lợi nhuận trung bình mỗi giao dịch, tỷ lệ thắng (win rate), yếu tố lợi nhuận (profit factor), mức sụt giảm vốn tối đa (maximal drawdown), và các tỷ số điều chỉnh theo rủi ro như Sharpe ratio hoặc Sortino ratio. Sau backtesting, các nhà phát triển thường tiến hành tối ưu hóa các tham số của chiến lược (ví dụ: chu kỳ MA, ngưỡng RSI) để tìm ra bộ giá trị mang lại hiệu quả tốt nhất trên dữ liệu lịch sử. Tuy nhiên, đây cũng là lúc cạm bẫy tối ưu hóa quá mức (overfitting hay curve fitting) xuất hiện – việc tinh chỉnh chiến lược quá sát với dữ liệu quá khứ có thể khiến nó mất khả năng thích ứng và hoạt động kém hiệu quả khi đối mặt với dữ liệu thị trường thực tế trong tương lai. Để giảm thiểu rủi ro này, cần sử dụng các kỹ thuật như kiểm tra ngoài mẫu (out-of-sample testing) hoặc kiểm tra tiến tới (walk-forward testing). Cơ sở hạ tầng vật lý như máy chủ ổn định (thường là máy chủ riêng ảo - VPS, hoặc các dịch vụ đám mây), đường truyền internet tốc độ cao và kết nối có độ trễ thấp đến máy chủ của nhà môi giới cũng là những yếu tố then chốt. Đối với các chiến lược HFT cực kỳ nhạy cảm với độ trễ, việc sử dụng VPS thông thường có thể không đủ. Các tổ chức HFT chuyên nghiệp thường đầu tư vào giải pháp co-location, tức là thuê chỗ đặt máy chủ giao dịch của họ ngay tại trung tâm dữ liệu (data center) nơi đặt máy chủ khớp lệnh của sàn giao dịch hoặc ECN. Điều này giúp giảm thiểu độ trễ vật lý do đường truyền mạng xuống mức thấp nhất có thể, tính bằng micro giây.

Việc kiểm thử lại (backtesting) và tối ưu hóa chiến lược là một giai đoạn mang tính quyết định trong quá trình phát triển bất kỳ hệ thống giao dịch thuật toán nào, đóng vai trò như một cuộc diễn tập quan trọng trước khi đưa thuật toán vào hoạt động với vốn thực. Quy trình backtesting về cơ bản bao gồm việc chạy thuật toán trên dữ liệu giá lịch sử trong một khoảng thời gian đủ dài và bao quát nhiều điều kiện thị trường khác nhau, từ giai đoạn có xu hướng rõ ràng (trending), giai đoạn đi ngang (ranging), đến những thời kỳ có biến động cao hoặc thấp, nhằm mô phỏng một cách tương đối chính xác hoạt động của chiến lược trong quá khứ.

Chất lượng của dữ liệu lịch sử được sử dụng trong backtesting là yếu tố nền tảng ảnh hưởng trực tiếp đến độ tin cậy của kết quả. Lý tưởng nhất là sử dụng dữ liệu tick (tick data), ghi lại mọi thay đổi giá nhỏ nhất, để có được mô phỏng khớp lệnh chính xác nhất, đặc biệt đối với các chiến lược giao dịch tần suất cao. Tuy nhiên, dữ liệu tick thường rất lớn và khó xử lý. Do đó, dữ liệu thanh (bar data) theo các khung thời gian phổ biến (M1, M5, H1, D1) cũng thường được sử dụng, nhưng cần ý thức được những hạn chế về độ chính xác trong việc mô phỏng diễn biến giá trong một thanh nến. Việc xử lý dữ liệu, bao gồm làm sạch các báo giá lỗi (bad ticks), xử lý dữ liệu bị thiếu (gaps), và điều chỉnh dữ liệu cho phù hợp với các yếu tố như chênh lệch múi giờ hoặc thay đổi quy tắc giao dịch của nhà môi giới, cũng là bước chuẩn bị quan trọng.

Một bộ máy backtesting (backtesting engine) hiệu quả cần mô phỏng thực tế giao dịch một cách sát sao nhất có thể. Nó không chỉ đơn thuần áp dụng logic của thuật toán vào dữ liệu giá quá khứ mà còn phải tính đến các yếu tố thực tế như chênh lệch giá mua-bán (spread) tại thời điểm giao dịch, khả năng trượt giá (slippage) có thể xảy ra khi thực hiện lệnh thị trường trong điều kiện thanh khoản thấp hoặc biến động mạnh, cũng như chi phí giao dịch như hoa hồng (commissions) và phí qua đêm (swap fees). Có hai cách tiếp cận chính cho backtesting engine: phương pháp dựa trên vector (vector-based) thường nhanh hơn nhưng kém chính xác hơn trong việc mô phỏng thứ tự sự kiện, và phương pháp dựa trên sự kiện (event-driven) mô phỏng từng tick hoặc từng thanh nến một cách tuần tự, gần với thực tế giao dịch hơn nhưng đòi hỏi nhiều tài nguyên tính toán hơn.

Sau khi quá trình backtesting hoàn tất, kết quả được đánh giá thông qua một loạt các chỉ số hiệu suất định lượng và định tính. Tổng lợi nhuận (Total Profit/Net Profit) là con số đầu tiên được nhìn vào, nhưng nó không thể hiện bức tranh toàn cảnh nếu không xem xét đến rủi ro. Tỷ lệ thắng (Win Rate), là phần trăm số giao dịch có lãi, cần được xem xét cùng với Tỷ lệ lợi nhuận trung bình trên mỗi giao dịch thắng so với Thua lỗ trung bình trên mỗi giao dịch thua (Average Win/Average Loss), từ đó tính ra Tỷ lệ Lợi nhuận/Rủi ro (Risk:Reward Ratio) kỳ vọng. Yếu tố lợi nhuận (Profit Factor), được tính bằng tổng lợi nhuận chia cho tổng thua lỗ, là một thước đo tổng hợp về khả năng sinh lời; giá trị lớn hơn 1 cho thấy chiến lược có lãi trong giai đoạn thử nghiệm. Mức sụt giảm vốn tối đa (Maximal Drawdown), là mức giảm lớn nhất từ đỉnh xuống đáy của đường cong vốn (equity curve), là một chỉ số cực kỳ quan trọng để đánh giá rủi ro và khả năng chịu đựng thua lỗ của nhà giao dịch. Các tỷ số điều chỉnh theo rủi ro như Sharpe Ratio (đo lường lợi nhuận vượt trội so với tài sản phi rủi ro trên mỗi đơn vị rủi ro tổng thể - độ lệch chuẩn) và Sortino Ratio (tương tự Sharpe nhưng chỉ xem xét biến động tiêu cực - downside deviation) giúp so sánh hiệu quả của các chiến lược có mức độ rủi ro khác nhau. Ngoài ra, số lượng giao dịch thực hiện trong giai đoạn backtest cũng cần đủ lớn để đảm bảo ý nghĩa thống kê, và thời gian nắm giữ vị thế trung bình (Average Holding Period) giúp xác định bản chất ngắn hạn hay dài hạn của chiến lược. Việc phân tích trực quan đường cong vốn cũng rất quan trọng để nhận diện các giai đoạn hoạt động hiệu quả, giai đoạn đi ngang kéo dài, và mức độ nghiêm trọng của các đợt sụt giảm vốn.

Dựa trên kết quả backtesting ban đầu, các nhà phát triển thường tiến hành giai đoạn tối ưu hóa (optimization). Quá trình này bao gồm việc thử nghiệm một loạt các giá trị khác nhau cho các tham số đầu vào của chiến lược (ví dụ: thay đổi chu kỳ của đường MA từ 10 đến 100 với bước nhảy là 5, hoặc thay đổi ngưỡng quá mua/quá bán của RSI từ 60-40 đến 80-20) để tìm ra bộ tham số mang lại kết quả tốt nhất theo một tiêu chí xác định (hàm mục tiêu - objective function), ví dụ như tối đa hóa lợi nhuận ròng, tối đa hóa Sharpe Ratio, hoặc tối thiểu hóa mức sụt giảm vốn tối đa. Các phương pháp tối ưu hóa phổ biến bao gồm tìm kiếm dạng lưới (grid search) duyệt qua tất cả các tổ hợp tham số có thể, hoặc các phương pháp hiệu quả hơn về mặt tính toán như thuật toán di truyền (genetic algorithms) hay tìm kiếm ngẫu nhiên (random search).

Tuy nhiên, quá trình tối ưu hóa tiềm ẩn một rủi ro nghiêm trọng được gọi là tối ưu hóa quá mức (overfitting hay curve fitting). Đây là hiện tượng khi chiến lược được tinh chỉnh quá mức để phù hợp một cách hoàn hảo với dữ liệu lịch sử cụ thể đã được sử dụng, bao gồm cả các yếu tố nhiễu ngẫu nhiên (noise) thay vì các quy luật thực sự của thị trường (signal). Hậu quả là chiến lược có thể cho kết quả backtest cực kỳ ấn tượng nhưng lại hoạt động kém hiệu quả hoặc thậm chí thua lỗ nặng nề khi áp dụng vào dữ liệu thị trường thực tế trong tương lai, bởi vì nó đã mất đi khả năng khái quát hóa và thích ứng với các điều kiện thị trường mới.

Để phát hiện và giảm thiểu rủi ro overfitting, nhiều kỹ thuật cần được áp dụng một cách nghiêm ngặt. Phương pháp phổ biến nhất là kiểm tra ngoài mẫu (Out-of-Sample - OOS testing), trong đó dữ liệu lịch sử được chia thành hai phần: phần trong mẫu (in-sample) được sử dụng để tối ưu hóa chiến lược, và phần ngoài mẫu (out-of-sample) hoàn toàn không được sử dụng trong quá trình tối ưu hóa, dùng để kiểm tra hiệu quả của bộ tham số tối ưu tìm được. Một chiến lược chỉ được coi là tiềm năng nếu nó vẫn cho thấy hiệu suất tốt trên cả phần dữ liệu ngoài mẫu chưa từng thấy trước đó. Một kỹ thuật mạnh mẽ và thực tế hơn nữa là Phân tích tiến tới (Walk-Forward Analysis - WFA). Trong WFA, quá trình tối ưu hóa và kiểm tra được lặp lại trên các cửa sổ dữ liệu trượt liên tiếp. Ví dụ, tối ưu hóa chiến lược trên dữ liệu 1 năm, sau đó kiểm tra hiệu quả trên dữ liệu 3 tháng tiếp theo; tiếp tục dịch chuyển cửa sổ tối ưu hóa và kiểm tra này tiến về phía trước trong suốt toàn bộ dữ liệu lịch sử. WFA mô phỏng gần hơn với cách một chiến lược có thể được điều chỉnh và hoạt động trong thực tế theo thời gian. Các phương pháp khác bao gồm việc giới hạn số lượng tham số và độ phức tạp của chiến lược (các mô hình đơn giản thường có tính khái quát hóa tốt hơn), thực hiện phân tích độ nhạy (sensitivity analysis) để đánh giá xem hiệu quả chiến lược thay đổi như thế nào khi các tham số bị thay đổi một chút (chiến lược tốt không nên quá nhạy cảm với những thay đổi nhỏ), và sử dụng mô phỏng Monte Carlo để đánh giá độ bền của chiến lược dưới các biến thể ngẫu nhiên của dữ liệu đầu vào hoặc tham số.

Cuối cùng, sau khi một chiến lược đã vượt qua các vòng backtesting và kiểm tra overfitting nghiêm ngặt, bước tiếp theo và không kém phần quan trọng là kiểm thử tiến tới trong môi trường giả lập (forward testing hay paper trading). Giai đoạn này bao gồm việc chạy thuật toán trên tài khoản demo với dữ liệu thị trường thời gian thực, không có rủi ro về vốn thực. Forward testing giúp xác nhận hiệu quả của chiến lược trong điều kiện thị trường hiện tại, đồng thời đánh giá các yếu tố thực tế như độ trễ mạng, chất lượng nguồn cấp dữ liệu, và sự tương tác với cơ sở hạ tầng của nhà môi giới mà backtesting có thể chưa mô phỏng hết được. Chỉ sau khi chiến lược chứng tỏ được sự ổn định và hiệu quả trong giai đoạn forward testing này, nhà giao dịch mới nên cân nhắc triển khai nó với một phần vốn thực một cách thận trọng. Tóm lại, backtesting và tối ưu hóa là những bước không thể thiếu, đòi hỏi sự cẩn trọng, kỹ lưỡng và nhận thức sâu sắc về các hạn chế, đặc biệt là nguy cơ overfitting. Mặc dù kết quả backtest tích cực không phải là sự đảm bảo chắc chắn cho lợi nhuận trong tương lai, nhưng một quy trình backtesting và xác thực được thực hiện đúng đắn sẽ giúp sàng lọc các ý tưởng yếu kém, xây dựng niềm tin vào chiến lược và cung cấp nền tảng vững chắc hơn cho việc triển khai giao dịch thuật toán thành công.

Những ưu điểm của giao dịch thuật toán là không thể phủ nhận. Tốc độ thực thi vượt trội so với con người cho phép nắm bắt những cơ hội thoáng qua. Tính kỷ luật và nhất quán trong việc tuân thủ chiến lược giúp loại bỏ các sai lầm do tâm lý gây ra. Khả năng kiểm thử lại và tối ưu hóa một cách khoa học mang lại sự tự tin cao hơn vào chiến lược. Khả năng mở rộng hệ thống để theo dõi và giao dịch đồng thời nhiều cặp tiền tệ, nhiều chiến lược 24/5 cũng là một lợi thế lớn.

Tuy nhiên, giao dịch thuật toán cũng tiềm ẩn nhiều thách thức và rủi ro đáng kể. Như đã đề cập, nguy cơ tối ưu hóa quá mức luôn hiện hữu. Các lỗi kỹ thuật, từ lỗi phần mềm, lỗi kết nối API, sự cố máy chủ đến mất kết nối internet, đều có thể gây ra những hậu quả nghiêm trọng. Độ trễ mạng vẫn là kẻ thù đối với nhiều chiến lược, đặc biệt là HFT. Quan trọng hơn, các chiến lược được tối ưu hóa cho một môi trường thị trường cụ thể (ví dụ: thị trường có xu hướng rõ ràng) có thể hoàn toàn thất bại khi thị trường thay đổi trạng thái (ví dụ: chuyển sang giai đoạn đi ngang hoặc biến động hỗn loạn). Các sự kiện bất ngờ như "thiên nga đen" (black swan events) hoặc các cú sập giá chớp nhoáng (flash crashes) có thể khiến các thuật toán hoạt động sai lệch hoặc thậm chí góp phần làm trầm trọng thêm tình hình. Do đó, việc giám sát liên tục, đánh giá hiệu quả và điều chỉnh, thích ứng chiến lược theo sự thay đổi của thị trường là yêu cầu bắt buộc đối với các nhà giao dịch thuật toán.

Trong tương lai, vai trò của trí tuệ nhân tạo (AI) và học máy (machine learning) trong giao dịch thuật toán Forex được dự báo sẽ ngày càng lớn mạnh, như một hướng nghiên cứu tiềm năng đã được đề xuất. Các mô hình học sâu (deep learning), mạng nơ-ron (neural networks), hay học tăng cường (reinforcement learning) hứa hẹn khả năng phát hiện các mẫu hình phi tuyến tính phức tạp trong dữ liệu thị trường, dự báo xu hướng giá chính xác hơn, hoặc thậm chí tự động điều chỉnh chiến lược để thích ứng với các điều kiện thị trường luôn biến đổi – một lĩnh vực đầy hứa hẹn để nâng cao hơn nữa hiệu quả và tính thông minh của các hệ thống giao dịch tự động.

### 2.1.2. Lý Thuyết Giao Dịch Forex

Để điều hướng thành công trong thị trường Forex phức tạp, các nhà giao dịch dựa vào nhiều lý thuyết và phương pháp phân tích khác nhau nhằm dự đoán biến động tỷ giá trong tương lai và đưa ra quyết định giao dịch. Có hai trường phái phân tích chính được áp dụng rộng rãi trong giao dịch Forex.

Trường phái thứ nhất là Phân tích Cơ bản (Fundamental Analysis). Phương pháp này tập trung vào việc đánh giá các yếu tố kinh tế vĩ mô, chính trị và xã hội có thể ảnh hưởng đến giá trị của một đồng tiền so với đồng tiền khác. Các nhà phân tích cơ bản xem xét các chỉ số kinh tế quan trọng như tốc độ tăng trưởng GDP, tỷ lệ lạm phát, tỷ lệ thất nghiệp, cán cân thương mại, lãi suất điều hành của ngân hàng trung ương, và các chính sách tài khóa, tiền tệ. Các chỉ số này không tồn tại độc lập mà có mối liên hệ mật thiết với nhau và cùng tác động đến chính sách tiền tệ. Ví dụ, lạm phát cao có thể buộc ngân hàng trung ương phải tăng lãi suất để kiềm chế lạm phát, việc tăng lãi suất này thường làm tăng sức hấp dẫn của đồng tiền đó đối với các nhà đầu tư tìm kiếm lợi suất cao hơn, dẫn đến tăng giá trị đồng tiền. Ngược lại, tăng trưởng GDP yếu hoặc tỷ lệ thất nghiệp cao có thể khiến ngân hàng trung ương hạ lãi suất hoặc áp dụng các biện pháp nới lỏng tiền tệ để kích thích kinh tế, điều này thường làm giảm sức hấp dẫn và giá trị của đồng tiền. Các sự kiện chính trị, bầu cử, căng thẳng địa chính trị, và thậm chí cả thiên tai cũng được xem xét vì chúng có thể tác động mạnh mẽ đến tâm lý thị trường và giá trị tiền tệ. Mục tiêu của phân tích cơ bản là xác định giá trị nội tại của một đồng tiền và dự đoán liệu tỷ giá hiện tại có đang bị định giá quá cao hay quá thấp so với giá trị thực của nó hay không, từ đó tìm kiếm cơ hội giao dịch dài hạn. \[Mở rộng 2.2: Phân tích tin tức\] Một nhánh quan trọng của phân tích cơ bản là giao dịch dựa trên tin tức (News Trading). Các nhà giao dịch theo trường phái này cố gắng dự đoán phản ứng của thị trường đối với các thông tin kinh tế được công bố theo lịch trình. Một số chiến lược phổ biến bao gồm việc đặt lệnh trước khi tin tức ra dựa trên kỳ vọng về kết quả, hoặc chờ đợi tin tức được công bố rồi giao dịch dựa trên sự khác biệt giữa số liệu thực tế và dự báo của thị trường (deviation trading). Việc này đòi hỏi tốc độ truy cập tin tức nhanh và khả năng ra quyết định tức thời.

Trường phái thứ hai là Phân tích Kỹ thuật (Technical Analysis). Khác với phân tích cơ bản, phân tích kỹ thuật không quan tâm đến các yếu tố kinh tế mà tập trung hoàn toàn vào dữ liệu lịch sử về giá và khối lượng giao dịch. Các nhà phân tích kỹ thuật sử dụng biểu đồ giá và các chỉ báo toán học để xác định các mẫu hình (patterns), xu hướng (trends), các mức hỗ trợ và kháng cự, từ đó dự đoán hướng đi tiếp theo của giá. Có nhiều loại biểu đồ được sử dụng, phổ biến nhất là biểu đồ nến Nhật (candlestick chart) cung cấp thông tin về giá mở cửa, đóng cửa, cao nhất và thấp nhất trong một khoảng thời gian một cách trực quan. Ngoài ra còn có biểu đồ đường (line chart) thường chỉ hiển thị giá đóng cửa, hữu ích để nhìn nhận xu hướng tổng thể dài hạn, và biểu đồ thanh (bar chart) cũng cung cấp thông tin OHLC tương tự biểu đồ nến nhưng với cách thể hiện khác. Phân tích kỹ thuật hoạt động dựa trên ba giả định cốt lõi: một là thị trường phản ánh tất cả mọi thông tin (tức là mọi yếu tố cơ bản đã được phản ánh vào giá); hai là giá di chuyển theo xu hướng (một xu hướng đã hình thành có khả năng tiếp diễn); và ba là lịch sử có xu hướng lặp lại (các mẫu hình giá đã xuất hiện trong quá khứ có thể tái diễn trong tương lai). Bên cạnh việc sử dụng các chỉ báo và mẫu hình, phân tích kỹ thuật còn dựa trên các lý thuyết nền tảng như Lý thuyết Sóng Elliott, mô tả các chu kỳ tâm lý thị trường thông qua các cấu trúc sóng đẩy (impulse waves) và sóng điều chỉnh (corrective waves). Một công cụ phổ biến khác là các mức Fibonacci thoái lui (Retracement) và mở rộng (Extension), dựa trên dãy số Fibonacci, được sử dụng để xác định các mức hỗ trợ, kháng cự tiềm năng và các mục tiêu giá có thể đạt được sau một biến động. Đây là phương pháp phổ biến đối với các nhà giao dịch ngắn hạn và trung hạn trong thị trường Forex.

Chắc chắn rồi, tôi sẽ mở rộng phần giới thiệu về "Các Chỉ Số Kỹ Thuật Quan Trọng Trong Forex" để cung cấp một nền tảng chi tiết hơn về vai trò và cách sử dụng các chỉ báo trong phân tích kỹ thuật, trước khi đi sâu vào RSI và MA.

### 2.1.3. Các Chỉ Số Kỹ Thuật Quan Trọng Trong Forex

Trong lĩnh vực phân tích kỹ thuật, vốn tập trung vào việc nghiên cứu dữ liệu giá và khối lượng lịch sử để dự đoán các biến động giá trong tương lai, các chỉ báo kỹ thuật (technical indicators) đóng vai trò như những công cụ toán học thiết yếu, hỗ trợ nhà giao dịch diễn giải hành động giá phức tạp và đưa ra các quyết định giao dịch một cách khách quan hơn. Chúng là các công thức được áp dụng vào dữ liệu giá (như giá mở cửa, đóng cửa, cao nhất, thấp nhất) và/hoặc khối lượng giao dịch trong một khoảng thời gian xác định, nhằm mục đích làm nổi bật các khía cạnh cụ thể của hành vi thị trường mà có thể không dễ dàng nhận thấy chỉ bằng cách quan sát biểu đồ giá đơn thuần. Các chỉ báo không phải là những quả cầu pha lê có khả năng dự đoán tương lai một cách chắc chắn, mà đúng hơn là những lăng kính giúp đơn giản hóa, định lượng hóa và cung cấp những góc nhìn khác nhau về động lực thị trường, bổ sung cho các phương pháp phân tích kỹ thuật khác như nhận dạng mẫu hình biểu đồ (chart patterns), phân tích đường xu hướng (trendlines) và các mức hỗ trợ/kháng cự. Việc hiểu và sử dụng hiệu quả các chỉ báo kỹ thuật là một kỹ năng quan trọng đối với nhiều nhà giao dịch Forex.

Mục đích chính mà các nhà giao dịch sử dụng chỉ báo kỹ thuật rất đa dạng, nhưng tựu trung lại là nhằm thu thập thông tin sâu sắc hơn về trạng thái hiện tại và tiềm năng tương lai của thị trường. Một trong những ứng dụng cơ bản nhất là để xác định hoặc xác nhận xu hướng chủ đạo. Các chỉ báo theo xu hướng giúp nhận diện liệu thị trường đang trong giai đoạn tăng giá (uptrend), giảm giá (downtrend) hay đi ngang (sideways/ranging), đồng thời cung cấp thước đo về sức mạnh và độ bền của xu hướng đó. Bên cạnh việc xác định hướng đi, việc đo lường động lượng (momentum) của thị trường cũng cực kỳ quan trọng. Các chỉ báo động lượng, thường là các bộ dao động (oscillators), giúp đánh giá tốc độ và sức mạnh đằng sau các biến động giá. Chúng có thể cảnh báo về việc liệu một xu hướng có đang tăng tốc, chậm lại, hay có dấu hiệu kiệt sức sắp đảo chiều hay không.

Một khía cạnh khác mà các chỉ báo kỹ thuật có thể hỗ trợ là đánh giá mức độ biến động (volatility) của thị trường. Biến động là thước đo biên độ và tần suất thay đổi giá. Việc hiểu rõ mức độ biến động hiện tại là rất quan trọng cho việc quản lý rủi ro, ví dụ như xác định khoảng cách đặt lệnh dừng lỗ hợp lý, đặt mục tiêu lợi nhuận thực tế, hoặc điều chỉnh khối lượng giao dịch cho phù hợp (vào lệnh nhỏ hơn khi biến động cao và ngược lại). Một số chỉ báo được thiết kế đặc biệt để đo lường biến động và có thể cung cấp tín hiệu về khả năng xảy ra các cú phá vỡ (breakout) mạnh mẽ sau những giai đoạn biến động thấp.

Phân tích khối lượng giao dịch, mặc dù gặp nhiều hạn chế trong thị trường Forex phi tập trung (do không có nguồn dữ liệu khối lượng tổng hợp toàn thị trường mà thường chỉ là khối lượng của từng nhà môi giới cụ thể), vẫn được một số nhà giao dịch sử dụng thông qua các chỉ báo khối lượng. Các chỉ báo này cố gắng đánh giá áp lực mua và bán dựa trên mối quan hệ giữa giá và khối lượng, nhằm xác nhận sức mạnh của một xu hướng hoặc cảnh báo về sự phân kỳ giữa giá và khối lượng.

Ngoài ra, nhiều chỉ báo, đặc biệt là các bộ dao động, được sử dụng để xác định các điều kiện thị trường cực đoan, thường được gọi là vùng quá mua (overbought) và quá bán (oversold). Các vùng này gợi ý rằng giá có thể đã di chuyển quá nhanh và quá xa theo một hướng, làm tăng khả năng xảy ra một đợt điều chỉnh ngược lại hoặc thậm chí là đảo chiều xu hướng. Cuối cùng, một trong những ứng dụng trực tiếp nhất của các chỉ báo là tạo ra các tín hiệu giao dịch khách quan. Dựa trên các quy tắc được xác định trước liên quan đến giá trị của chỉ báo (ví dụ: vượt qua một ngưỡng nhất định), sự giao cắt giữa các đường chỉ báo, hoặc sự hình thành phân kỳ giữa chỉ báo và giá, các chỉ báo có thể cung cấp các điểm vào lệnh và thoát lệnh cụ thể, giúp giảm bớt yếu tố chủ quan và cảm tính trong quá trình ra quyết định.

Với sự phát triển của công nghệ và lý thuyết tài chính, số lượng các chỉ báo kỹ thuật đã lên đến hàng trăm, thậm chí hàng nghìn. Tuy nhiên, chúng thường có thể được phân loại vào một số nhóm chính dựa trên chức năng và cách thức hoạt động. Nhóm đầu tiên và phổ biến nhất là các Chỉ báo theo xu hướng (Trend-Following Indicators). Ví dụ tiêu biểu bao gồm Đường Trung Bình Động (Moving Averages - MA), MACD (Moving Average Convergence Divergence), ADX (Average Directional Index), và Parabolic SAR. Mục tiêu chính của nhóm này là giúp nhà giao dịch xác định và đi theo hướng của xu hướng hiện hành. Chúng hoạt động hiệu quả nhất trong các thị trường có xu hướng rõ ràng, giúp lọc bỏ nhiễu và cung cấp các tín hiệu thuận theo xu hướng. Tuy nhiên, nhược điểm cố hữu của chúng là tính chất trễ (lagging); chúng thường chỉ xác nhận xu hướng sau khi nó đã hình thành và có xu hướng tạo ra nhiều tín hiệu sai lệch (whipsaws) khi thị trường đi ngang hoặc không có xu hướng rõ ràng.

Nhóm thứ hai là các Bộ dao động (Oscillators). Các chỉ báo như Chỉ số Sức mạnh Tương đối (RSI), Stochastic Oscillator, và CCI (Commodity Channel Index) thuộc nhóm này. Chúng thường dao động trong một phạm vi giới hạn (ví dụ: 0-100 đối với RSI và Stochastic). Chức năng chính của bộ dao động là đo lường động lượng giá và xác định các điều kiện quá mua/quá bán trong ngắn hạn. Chúng thường được coi là có đặc tính "dẫn dắt" (leading) hơn so với các chỉ báo theo xu hướng, vì chúng có thể cung cấp tín hiệu cảnh báo sớm về khả năng đảo chiều tiềm năng trước khi giá thực sự thay đổi hướng. Bộ dao động hoạt động hiệu quả nhất trong các thị trường đi ngang hoặc có phạm vi giao dịch xác định (range-bound markets), giúp xác định các điểm cực đại và cực tiểu tiềm năng để thực hiện giao dịch đảo chiều. Tuy nhiên, trong các thị trường có xu hướng mạnh, chúng có thể liên tục nằm trong vùng quá mua hoặc quá bán trong thời gian dài, tạo ra các tín hiệu đảo chiều sai lầm.

Nhóm thứ ba là các Chỉ báo đo lường biến động (Volatility Indicators). Đại diện tiêu biểu là Dải Bollinger (Bollinger Bands) và Chỉ báo Phạm vi Trung bình Thực (Average True Range - ATR). Các chỉ báo này không trực tiếp cung cấp tín hiệu mua/bán mà tập trung vào việc đo lường mức độ biến động của thị trường. Dải Bollinger sử dụng độ lệch chuẩn để tạo ra các dải biến động quanh một đường MA, giúp xác định xem giá hiện tại đang ở mức tương đối cao hay thấp so với biến động gần đây. Sự co thắt của dải Bollinger thường báo hiệu giai đoạn biến động thấp và tiềm ẩn khả năng xảy ra một cú phá vỡ mạnh. ATR đo lường phạm vi biến động trung bình của giá trong một khoảng thời gian, rất hữu ích cho việc đặt lệnh dừng lỗ một cách hợp lý (ví dụ: đặt dừng lỗ cách giá vào lệnh một khoảng bằng 2 lần ATR) hoặc điều chỉnh khối lượng giao dịch theo rủi ro biến động.

Nhóm cuối cùng là các Chỉ báo khối lượng (Volume Indicators), ví dụ như On-Balance Volume (OBV) hay Chaikin Money Flow (CMF). Chúng cố gắng kết hợp thông tin về khối lượng giao dịch với biến động giá để đo lường áp lực mua và bán tích lũy. Ví dụ, OBV cộng khối lượng vào tổng khi giá đóng cửa tăng và trừ khối lượng khi giá đóng cửa giảm. Sự phân kỳ giữa chỉ báo khối lượng và giá có thể cung cấp tín hiệu cảnh báo. Tuy nhiên, như đã lưu ý, việc áp dụng các chỉ báo khối lượng vào thị trường Forex phi tập trung cần thận trọng do tính không đầy đủ và phân mảnh của dữ liệu khối lượng có sẵn.

Để sử dụng các chỉ báo kỹ thuật một cách hiệu quả và tránh những cạm bẫy phổ biến, nhà giao dịch cần tuân thủ một số nguyên tắc quan trọng. Nguyên tắc tối thượng là không bao giờ dựa vào một chỉ báo duy nhất để đưa ra quyết định giao dịch. Mỗi chỉ báo đều có điểm mạnh và điểm yếu riêng, và không có chỉ báo nào là hoàn hảo trong mọi điều kiện thị trường. Tín hiệu giao dịch sẽ trở nên đáng tin cậy hơn nhiều khi chúng được xác nhận bởi nhiều yếu tố khác nhau, bao gồm tín hiệu từ các chỉ báo khác (lý tưởng nhất là các chỉ báo thuộc các nhóm khác nhau và không có tương quan cao với nhau) và, quan trọng nhất, là sự xác nhận từ chính hành động giá (price action) – ví dụ, một tín hiệu mua từ chỉ báo cần được hỗ trợ bởi sự phá vỡ một mức kháng cự hoặc sự hình thành một mẫu hình nến tăng giá mạnh mẽ.

Việc hiểu rõ logic tính toán và mục đích thiết kế của từng chỉ báo cũng rất quan trọng. Nhà giao dịch cần biết chỉ báo đó đo lường cái gì, nó được tính toán như thế nào, và những giả định nào nằm sau nó, để có thể diễn giải các tín hiệu một cách chính xác và nhận thức được những hạn chế tiềm ẩn. Sử dụng chỉ báo như một "hộp đen" mà không hiểu nguyên lý hoạt động của nó là một cách tiếp cận nguy hiểm.

Bối cảnh thị trường tổng thể cũng đóng vai trò quyết định trong việc diễn giải tín hiệu từ chỉ báo. Một tín hiệu quá mua từ RSI có thể cần được diễn giải khác biệt trong một xu hướng tăng mạnh kéo dài so với trong một thị trường đang đi ngang. Tương tự, một tín hiệu giao cắt của MA có thể đáng tin cậy hơn khi thị trường đang bắt đầu hình thành xu hướng mới so với khi thị trường đang biến động hỗn loạn. Việc kết hợp phân tích đa khung thời gian (multiple time frame analysis) cũng giúp đặt các tín hiệu từ chỉ báo vào một bối cảnh rộng hơn.

Một cạm bẫy phổ biến khác là việc sử dụng quá nhiều chỉ báo trên cùng một biểu đồ, dẫn đến tình trạng "tê liệt do phân tích" (analysis paralysis) với vô số tín hiệu nhiễu và mâu thuẫn. Thay vì cố gắng theo dõi mọi chỉ báo có thể, nhà giao dịch nên tập trung vào việc lựa chọn và làm chủ một vài chỉ báo bổ trợ cho nhau và phù hợp nhất với chiến lược, phong cách giao dịch và khung thời gian của mình.

Cuối cùng, cần nhận thức rằng các tham số mặc định của chỉ báo (ví dụ: N=14 cho RSI, N=20/50 cho MA) chỉ là những điểm khởi đầu phổ biến. Tùy thuộc vào đặc điểm biến động của từng cặp tiền tệ và khung thời gian giao dịch, việc kiểm thử và điều chỉnh các tham số này có thể cần thiết để tối ưu hóa hiệu quả của chỉ báo. Tuy nhiên, quá trình tối ưu hóa này cần được thực hiện một cách cẩn trọng để tránh rơi vào bẫy overfitting, tức là làm cho chỉ báo hoạt động quá tốt trên dữ liệu quá khứ nhưng lại mất đi khả năng thích ứng với thị trường thực tế. Việc cân bằng giữa tính chất trễ (lagging) của các chỉ báo theo xu hướng và tính chất dẫn dắt (leading) nhưng dễ tạo tín hiệu sai của các bộ dao động cũng là một nghệ thuật trong việc xây dựng một hệ thống giao dịch hoàn chỉnh.

Trong số vô vàn các chỉ báo kỹ thuật hiện có, Chỉ số Sức mạnh Tương đối (RSI) và Đường Trung Bình Động (MA) vẫn luôn giữ vững vị thế là hai trong số những công cụ được sử dụng phổ biến và rộng rãi nhất bởi các nhà giao dịch Forex ở mọi cấp độ. Sự phổ biến này đến từ tính linh hoạt, sự đơn giản tương đối trong cách diễn giải, và vai trò nền tảng của chúng trong nhiều chiến lược giao dịch khác nhau, từ việc xác định xu hướng, đo lường động lượng, đến việc tìm kiếm các điểm vào lệnh và thoát lệnh tiềm năng. Các phần tiếp theo của báo cáo này sẽ đi sâu vào phân tích chi tiết cấu tạo, chức năng, cách ứng dụng cụ thể cũng như những ưu nhược điểm của hai chỉ báo quan trọng này trong bối cảnh giao dịch trên thị trường ngoại hối.

### 2.1.4. Hiệu quả của hệ thống giao dịch kết hợp mẫu hình Engulfing, EMA200 và RSI trong giao dịch Forex: Nghiên cứu backtest  
2.1.5. Long Short Term Memory (LSTM)  
2.1.6. Recurrent Neural Network (RNN)

**Giới thiệu chung về RNN**

Mạng nơ-ron hồi tiếp (Recurrent Neural Network – RNN) là một kiến trúc trong học sâu được thiết kế đặc biệt để xử lý dữ liệu tuần tự – nơi các phần tử dữ liệu không độc lập mà có mối liên hệ theo thời gian hoặc thứ tự. Khác với mạng nơ-ron truyền thống (feedforward neural network), vốn xử lý từng đầu vào một cách tách biệt, RNN tận dụng mối quan hệ giữa các phần tử trong chuỗi bằng cách duy trì một trạng thái ẩn (hidden state) – đóng vai trò như "bộ nhớ" của mạng (Goodfellow, Bengio, & Courville, 2016). Nhờ khả năng này, RNN trở thành công cụ lý tưởng cho các bài toán như dịch máy, nhận dạng giọng nói, sang văn bản, và đặc biệt là dự đoán chuỗi thời gian trong tài chính – chẳng hạn như giá cổ phiếu hoặc tỷ giá hối đoái (Fischer & Krauss, 2018).

**Nguyên lý hoạt động**

RNN có khả năng **ghi nhớ thông tin của chuỗi dữ liệu trong quá khứ** nhờ vào vòng lặp nội tại trong kiến trúc mạng. Về mặt toán học, tại thời điểm t, đầu vào x<sub>t</sub> sẽ được kết hợp với trạng thái ẩn của thời điểm trước đó h<sub>t−1</sub> để sinh ra trạng thái ẩn mới h<sub>t</sub> ​, theo công thức:

Thông qua cơ chế này, trạng thái h<sub>t</sub> không chỉ phụ thuộc vào x<sub>t</sub>​, mà còn gián tiếp phụ thuộc vào toàn bộ chuỗi đầu vào từ x<sub>1</sub>​ đến x<sub>t</sub>. Điều này cho phép RNN mô hình hóa được các phụ thuộc theo thời gian, điều mà các mạng truyền thống không thể thực hiện (Elman, 1990).

**Cấu trúc và biến thể**

RNN có thể được "mở cuộn theo thời gian" (unrolled in time), chuyển thành một chuỗi các mạng con chia sẻ tham số, hỗ trợ học các quy luật phức tạp trong dữ liệu chuỗi dài. Tùy theo bài toán cụ thể, RNN có thể được triển khai theo các dạng cấu trúc đầu vào – đầu ra như sau:

Bảng 2. 1 Cấu trúc đầu vào – đầu vào RNN

|     |     |     |
| --- | --- | --- |
| **Dạng cấu trúc** | **Mô tả** | **Ví dụ ứng dụng** |
| One-to-One | Một đầu vào – một đầu ra | Phân loại hình ảnh |
| One-to-Many | Một đầu vào – nhiều đầu ra | Tạo nhạc, sinh văn bản |
| Many-to-One | Nhiều đầu vào – một đầu ra | Phân loại cảm xúc, dự đoán xu hướng giá |
| Many-to-Many | Nhiều đầu vào, nhiều đầu ra | Dịch máy, mô phỏng chuỗi giá |

Trong lĩnh vực tài chính, đặc biệt là thị trường ngoại hối (Forex), cấu trúc **Many-to-One** thường được sử dụng. Cụ thể, mô hình nhận vào một đoạn chuỗi dữ liệu lịch sử (như giá đóng cửa hoặc chỉ báo kỹ thuật) để dự đoán biến động giá trong phiên tiếp theo (Zhang, Aggarwal, & Qi, 2018).

**Vai trò và lợi ích trong dự đoán tài chính**

Một trong những ưu điểm nổi bật của RNN là khả năng phát hiện các mẫu lặp lại, chu kỳ hoặc xu hướng phụ thuộc – những yếu tố thường xuất hiện trong dữ liệu tài chính. Ví dụ, nếu một chỉ báo kỹ thuật như RSI (Relative Strength Index) liên tục rơi vào vùng quá mua hoặc quá bán, RNN có thể học được mối liên hệ giữa hiện tượng này và khả năng đảo chiều giá trong ngắn hạn, từ đó cải thiện độ chính xác của dự đoán.

Bên cạnh đó, do các tham số được chia sẻ theo thời gian, RNN không yêu cầu độ dài chuỗi cố định, điều này tạo điều kiện thuận lợi để mô hình áp dụng cho nhiều khung thời gian khác nhau như H1, H4, hoặc D1 mà không cần tái huấn luyện hoàn toàn – phù hợp với nhu cầu phân tích đa khung trong thực tiễn giao dịch (Lipton, 2015).

**Hạn chế và cải tiến**

Tuy có nhiều điểm mạnh, RNN truyền thống vẫn gặp phải một số thách thức khi xử lý các chuỗi quá dài. Cụ thể, trong quá trình lan truyền ngược để cập nhật trọng số, mô hình dễ gặp phải hiện tượng **vanishing gradient** (độ dốc trở nên rất nhỏ) hoặc **exploding gradient** (độ dốc quá lớn), gây cản trở việc học hoặc dẫn đến mất ổn định trong huấn luyện (Bengio, Simard, & Frasconi, 1994; Pascanu, Mikolov, & Bengio, 2013).

Để khắc phục, các kiến trúc tiên tiến như **Long Short-Term Memory (LSTM)** và **Gated Recurrent Unit (GRU)** đã được phát triển. Những mô hình này sử dụng các cơ chế “cổng” (gates) để kiểm soát luồng thông tin – cho phép mô hình ghi nhớ hoặc quên thông tin một cách chọn lọc, từ đó cải thiện khả năng học phụ thuộc dài hạn. Tuy nhiên, trong các bài toán yêu cầu xử lý nhanh với chuỗi ngắn hoặc tài nguyên hạn chế, RNN truyền thống vẫn có chỗ đứng nhờ cấu trúc đơn giản và hiệu suất tính toán cao hơn (Karpathy, 2015).

**Kết luận**

Tóm lại, mạng nơ-ron hồi tiếp (RNN) là một công cụ mạnh mẽ trong việc xử lý dữ liệu chuỗi, đặc biệt hiệu quả trong các bài toán dự đoán tài chính nhờ khả năng ghi nhớ và học từ các mối liên hệ thời gian. Dù tồn tại một số hạn chế về mặt kỹ thuật, RNN vẫn là nền tảng cho các kiến trúc hiện đại hơn như LSTM, GRU, và Transformer. Đối với những ứng dụng tài chính yêu cầu mô hình hóa sự thay đổi giá qua thời gian, RNN là một điểm khởi đầu hợp lý và có tính ứng dụng cao

### 2.1.7. Gated Recurrent Unit (GRU)

**Cơ sở lý thuyết và nguồn gốc**

Gated Recurrent Unit (GRU) là một biến thể của mạng nơ-ron hồi tiếp (Recurrent Neural Network – RNN), được đề xuất bởi Cho et al. (2014) nhằm giải quyết những điểm yếu của RNN truyền thống trong việc học phụ thuộc dài hạn trong chuỗi dữ liệu. Một trong những vấn đề nghiêm trọng nhất của RNN là **hiện tượng mất dần gradient (vanishing gradient)**, khiến mô hình khó học được các phụ thuộc dài hạn (Bengio et al., 1994). Để khắc phục điều này, GRU được thiết kế với cơ chế “gating” giúp kiểm soát dòng thông tin qua các thời điểm, từ đó cải thiện khả năng ghi nhớ và cập nhật thông tin.

So với LSTM (Long Short-Term Memory), GRU có cấu trúc đơn giản hơn do loại bỏ một số cổng và giảm số lượng tham số, trong khi vẫn giữ được khả năng mô hình hóa mối quan hệ thời gian trong chuỗi (Chung et al., 2014; Greff et al., 2017). Điều này khiến GRU trở thành một lựa chọn hiệu quả hơn về mặt tính toán trong các tác vụ yêu cầu xử lý thời gian thực hoặc có giới hạn tài nguyên.

**Cấu trúc kỹ thuật và công thức**

GRU sử dụng hai cơ chế chính: cổng cập nhật (update gate) và cổng đặt lại (reset gate) để điều khiển dòng thông tin.

- Cổng cập nhật (update gate) :

- Cổng đặt lại (reset gate) :

- Trạng thái ẩn ứng viên (candidate hidden state):

- Trạng thái ẩn cuối cùng (final hidden state):

Trong đó:

- σ là hàm sigmoid;  
    
- ⊙ là phép nhân từng phần tử (Hadamard product);  
    
- : đầu vào tại thời điểm t;  
    
- ​: trạng thái ẩn ở bước trước.  
    

GRU không sử dụng trạng thái "cell state" như LSTM, giúp đơn giản hóa kiến trúc và rút ngắn thời gian huấn luyện.

**So sánh với LSTM và RNN truyền thống**

GRU và LSTM đều vượt trội so với RNN truyền thống trong các tác vụ yêu cầu ghi nhớ dài hạn. Tuy nhiên, theo nghiên cứu của Chung et al. (2014) và Jozefowicz et al. (2015), GRU thường cho kết quả **tương đương hoặc vượt trội so với LSTM** trong các tác vụ có chuỗi dữ liệu ngắn đến trung bình, với tốc độ huấn luyện nhanh hơn khoảng 20–30%.

Ngoài ra, theo Greff et al. (2017), mô hình GRU cũng **ít bị overfitting hơn** so với LSTM trong các tập dữ liệu nhỏ, nhờ cấu trúc đơn giản và số lượng tham số ít hơn.

**Ứng dụng trong dự báo chuỗi tài chính**

GRU đã được ứng dụng rộng rãi trong các tác vụ dự báo tài chính như dự báo giá cổ phiếu, tỷ giá hối đoái và biến động thị trường. Trong nghiên cứu của Fischer và Krauss (2018), GRU được sử dụng để dự đoán lợi suất S&P 500 và cho thấy hiệu quả vượt trội so với các mô hình truyền thống như hồi quy tuyến tính và mạng MLP. Nghiên cứu của Qiu et al. (2020) cũng cho thấy, khi kết hợp GRU với mạng CNN và attention mechanism, độ chính xác dự báo chuỗi tài chính ngắn hạn được cải thiện rõ rệt.

###   
2.1.8. Random Forest

**Cơ sở lý thuyết và nguồn gốc**

Random Forest (Rừng Ngẫu nhiên) là một thuật toán học máy thuộc nhóm ensemble learning, được phát triển bởi Breiman (2001) nhằm cải thiện hiệu năng dự báo và khả năng khái quát của các mô hình phân loại và hồi quy. Thuật toán này kết hợp nhiều cây quyết định (decision trees) độc lập để tạo thành một “rừng” các mô hình đơn lẻ, qua đó giảm thiểu hiện tượng overfitting thường xảy ra khi sử dụng một cây đơn (Breiman, 2001).

Ensemble learning dựa trên nguyên tắc rằng tập hợp nhiều mô hình yếu (weak learners) có thể tạo thành một mô hình mạnh (strong learner) bằng cách tổng hợp kết quả đầu ra. Trong trường hợp của Random Forest, mỗi cây trong rừng được xây dựng dựa trên một tập con mẫu dữ liệu được bootstrap (lấy mẫu với hoàn lại) và một tập con đặc trưng được chọn ngẫu nhiên (feature bagging), giúp tăng tính đa dạng và độc lập giữa các cây (Ho, 1998; Breiman, 2001).

**Cấu trúc kỹ thuật và công thức**

Random Forest tạo ra BBB cây quyết định =1​, mỗi cây được huấn luyện trên tập mẫu bootstrap và chỉ sử dụng một tập con các biến đặc trưng mmm được chọn ngẫu nhiên tại mỗi nút phân chia.

- Mỗi cây quyết định đưa ra dự đoán ​.  
    
- Với bài toán phân loại, dự đoán cuối cùng của Random Forest là kết quả bỏ phiếu đa số (majority vote):

- Với bài toán hồi quy, dự đoán là trung bình của các dự đoán từ từng cây:

Cơ chế bootstrap sampling (mẫu lấy lại) tạo ra tập dữ liệu huấn luyện riêng biệt cho từng cây, còn feature bagging (chọn ngẫu nhiên các đặc trưng) làm giảm sự tương quan giữa các cây, từ đó giảm phương sai mô hình tổng thể (Breiman, 2001; Hastie et al., 2009).

Ngoài ra, Random Forest sử dụng **Out-of-Bag (OOB) error estimate** để đánh giá hiệu năng mô hình một cách không thiên vị, tận dụng các mẫu không được chọn trong bootstrap cho việc kiểm tra (Breiman, 1996).

**So sánh và đánh giá mô hình**

Random Forest thường có hiệu năng vượt trội so với cây quyết định đơn lẻ nhờ khả năng giảm thiểu overfitting và cải thiện độ chính xác (Breiman, 2001). So với các mô hình ensemble khác như Gradient Boosting Machines (GBM), Random Forest dễ cài đặt và ít nhạy cảm với việc điều chỉnh tham số (Zhou, 2012).

Tuy nhiên, điểm hạn chế của Random Forest là mô hình có thể trở nên rất lớn và chậm khi dự đoán trên các tập dữ liệu lớn, đồng thời thiếu khả năng giải thích chi tiết các mối quan hệ phi tuyến phức tạp mà các mô hình boosting có thể làm tốt hơn (Friedman, 2001).

**Ứng dụng trong dự báo chuỗi thời gian và tài chính**

Mặc dù Random Forest không phải mô hình chuỗi thời gian thuần túy, nhiều nghiên cứu đã ứng dụng mô hình này cho dự báo tài chính bằng cách sử dụng các đặc trưng kỹ thuật được trích xuất từ dữ liệu chuỗi thời gian (Tsai & Wang, 2017). Đặc biệt, Random Forest hiệu quả trong việc xử lý các biến đầu vào phi tuyến và dữ liệu có nhiễu cao, đặc điểm phổ biến trong thị trường tài chính (Kumar et al., 2020).

Ngoài ra, tính năng đánh giá độ quan trọng của biến (feature importance) của Random Forest hỗ trợ phân tích các yếu tố ảnh hưởng đến biến động giá và giúp cải thiện quá trình lựa chọn đặc trưng (Genuer et al., 2010). Trong các hệ thống AI Agent tự động dự báo và ra quyết định, Random Forest thường được dùng như một thành phần mô hình dự báo kết hợp với các mô hình deep learning hoặc rule-based để nâng cao độ chính xác và tính ổn định (Chen et al., 2021).

### 2.1.9 AI Agent

**Khái niệm và định nghĩa**

AI Agent, hay còn gọi là tác tử trí tuệ nhân tạo, là hệ thống máy tính có khả năng tự động thực hiện các tác vụ cụ thể trong môi trường dựa trên khả năng nhận thức, lập kế hoạch và hành động nhằm đạt mục tiêu xác định (Russell & Norvig, 2016). AI Agent hoạt động thông qua việc thu thập dữ liệu, xử lý thông tin và phản hồi dựa trên logic hoặc mô hình học máy được huấn luyện.

Một AI Agent hiện đại thường bao gồm các thành phần như cảm biến (để thu thập dữ liệu), bộ xử lý (để phân tích và ra quyết định), và các cơ chế thực thi (để tương tác hoặc điều khiển hệ thống) (Wooldridge, 2009). Các AI Agent có thể hoạt động độc lập hoặc phối hợp trong hệ thống đa tác tử (multi-agent systems), góp phần nâng cao hiệu quả và khả năng mở rộng (Jennings & Wooldridge, 1998).

**Ứng dụng của AI Agent**

AI Agent được ứng dụng rộng rãi trong nhiều lĩnh vực như tự động hóa quy trình doanh nghiệp (Robotic Process Automation – RPA), trợ lý ảo cá nhân (Personal Assistants), giám sát và điều khiển hệ thống thông minh, và đặc biệt là trong lĩnh vực tài chính và giao dịch tự động (Autonomous Trading Systems) (Siau & Wang, 2018).

Các AI Agent có thể thực hiện các nhiệm vụ phức tạp như phân tích dữ liệu lớn (big data), dự đoán xu hướng thị trường, tự động ra quyết định giao dịch, hoặc tự động hóa các quy trình lặp đi lặp lại, giúp giảm tải công việc thủ công và tăng độ chính xác, hiệu quả (Wang et al., 2020).

**n8n – Công cụ tự động hóa AI Agent**

n8n là một nền tảng tự động hóa quy trình công việc mã nguồn mở (open-source workflow automation tool) giúp tạo lập các AI Agent bằng cách kết nối nhiều dịch vụ, API, và ứng dụng khác nhau để tự động hóa các tác vụ (n8n.io, 2024).

Khác với các nền tảng tự động hóa truyền thống như Zapier hay Integromat, n8n cho phép người dùng xây dựng các workflow phức tạp, tùy chỉnh cao với khả năng mở rộng linh hoạt, phù hợp cho các ứng dụng AI Agent trong doanh nghiệp và nghiên cứu (Kovalev, 2022).

n8n cung cấp môi trường visual (kéo thả), dễ sử dụng cho việc tích hợp các API trí tuệ nhân tạo, dịch vụ web, và cơ sở dữ liệu, từ đó các AI Agent có thể hoạt động liên tục, xử lý dữ liệu theo thời gian thực và phản hồi kịp thời (Martinez, 2023).

**Lợi ích và thách thức**

AI Agent sử dụng n8n giúp tăng tốc độ triển khai và giảm chi phí phát triển, đồng thời cải thiện độ linh hoạt và khả năng bảo trì hệ thống. Tuy nhiên, việc xây dựng và vận hành AI Agent vẫn đòi hỏi kiến thức sâu về lập trình, quản trị hệ thống và trí tuệ nhân tạo (Mohamed et al., 2021).

Một thách thức quan trọng là đảm bảo tính bảo mật, riêng tư dữ liệu và khả năng xử lý các tình huống bất thường trong quá trình hoạt động tự động, tránh các rủi ro tiềm ẩn gây thiệt hại (Zhou et al., 2022).

### 2.1.10 Trợ lý thông minh AI Agent

**Định nghĩa và đặc điểm**

Trợ lí thông minh AI Agent (Intelligent Assistant AI Agent) là một dạng AI Agent được thiết kế để hỗ trợ con người trong việc thực hiện các nhiệm vụ hàng ngày hoặc công việc chuyên môn bằng cách cung cấp thông tin, tự động hóa các thao tác hoặc ra quyết định dựa trên dữ liệu và ngữ cảnh (Liao et al., 2016).

Các trợ lí thông minh thường được tích hợp với các công nghệ như xử lý ngôn ngữ tự nhiên (Natural Language Processing – NLP), học máy (Machine Learning – ML), và trí tuệ nhân tạo nâng cao (Advanced AI) để có thể tương tác một cách linh hoạt và tự nhiên với người dùng (McTear, Callejas, & Griol, 2016).

**Cấu trúc và chức năng**

Trợ lí thông minh bao gồm nhiều thành phần chính:

- **Giao diện người dùng (User Interface – UI):** cho phép tương tác qua giọng nói, văn bản hoặc các hình thức tương tác khác.
- **Xử lý ngôn ngữ tự nhiên:** chuyển đổi đầu vào từ người dùng thành dạng máy hiểu được và ngược lại (Jurafsky & Martin, 2020).
- **Bộ nhớ ngữ cảnh (Context Memory):** lưu trữ và xử lý thông tin liên quan đến ngữ cảnh và lịch sử tương tác.
- **Bộ xử lý quyết định:** dựa trên dữ liệu đầu vào và các thuật toán học máy để đưa ra hành động phù hợp (Kumar et al., 2021).

**Ứng dụng thực tiễn**

Trợ lí thông minh AI Agent hiện diện trong nhiều sản phẩm và dịch vụ như Google Assistant, Amazon Alexa, Siri của Apple, và các trợ lý doanh nghiệp chuyên biệt (Liao et al., 2016; McTear et al., 2016). Trong lĩnh vực tài chính, trợ lí này hỗ trợ quản lý đầu tư, tư vấn tài chính cá nhân, và tự động hóa các tác vụ phân tích dữ liệu (Li & Li, 2019).

Trợ lí thông minh cũng đóng vai trò quan trọng trong các hệ thống AI Agent phức tạp, đảm nhiệm vai trò trung gian giao tiếp giữa người dùng và hệ thống, nâng cao trải nghiệm người dùng và hiệu quả vận hành (Mohamed et al., 2021).

**Thách thức và xu hướng phát triển**

Một số thách thức của trợ lí thông minh gồm xử lý ngôn ngữ tự nhiên đa dạng và phức tạp, đảm bảo bảo mật và quyền riêng tư thông tin cá nhân, cũng như khả năng thích ứng với các ngữ cảnh và nhu cầu thay đổi liên tục của người dùng (Zhou et al., 2022; Kumar et al., 2021).

Xu hướng phát triển hiện nay tập trung vào việc tích hợp trí tuệ nhân tạo nâng cao, khả năng học liên tục (continual learning), và mở rộng đa phương tiện (multimodal interaction) nhằm tạo ra trợ lí thông minh ngày càng linh hoạt, cá nhân hóa và thân thiện hơn với người dùng (Ghahramani, 2020).

##   
2.2. Thực trạng vấn đề nghiên cứu

Trong những năm gần đây, nhiều nghiên cứu đã tập trung vào việc phát triển các hệ thống giao dịch tự động trong thị trường ngoại hối (Forex), nhằm nâng cao hiệu quả giao dịch và cải thiện khả năng quản lý rủi ro. Các hệ thống này ngày càng được tích hợp sâu với các phương pháp học máy, học sâu, và kỹ thuật tối ưu hóa hiện đại.

Nổi bật trong số đó là nghiên cứu của **Kovalev, S., Kovalev, A., và Kovalev, V. (2023)**, đề xuất một hệ thống đa mô hình kết hợp ba thành phần chính: mô hình xu hướng (dựa trên đường trung bình động), mô hình quay về giá trị trung bình (sử dụng dải Bollinger), và mô hình dự đoán ngắn hạn (ứng dụng Random Forest). Hệ thống được đánh giá qua backtesting trên cặp EUR/USD trong giai đoạn 2010–2022, với các chỉ số như Sharpe Ratio và tỷ lệ thắng. Kết quả cho thấy hiệu suất vượt trội, với lợi nhuận cao hơn 15% so với các mô hình đơn lẻ, đồng thời cải thiện khả năng kiểm soát rủi ro trong điều kiện thị trường biến động mạnh. Tuy nhiên, hệ thống vẫn chưa tích hợp dữ liệu thời gian thực – một hướng nghiên cứu tiềm năng cho tương lai.

Trong một nghiên cứu khác, **Li và Wang (2023)** đã phát triển quy trình thiết lập và tối ưu hóa hệ thống giao dịch Forex tự động bằng cách kết hợp các chỉ báo kỹ thuật truyền thống như RSI và MACD với các thuật toán học máy như mạng nơ-ron và hồi quy logistic. Sau quá trình kiểm thử trên dữ liệu lịch sử (2015–2022) và tài khoản demo trong 6 tháng, hệ thống ghi nhận mức lợi nhuận trung bình tăng 12% và mức rút vốn tối đa được giữ dưới 10%. Nghiên cứu này cung cấp một quy trình toàn diện, đồng thời nhấn mạnh tiềm năng tích hợp dữ liệu thời gian thực từ các nền tảng như Forex Factory để nâng cao hiệu quả dự báo.

**Aloui và Ksentini (2014)** cũng đóng góp vào lĩnh vực này bằng cách xây dựng hệ thống giao dịch kết hợp các chỉ báo kỹ thuật phổ biến với thuật toán học máy như SVM và Random Forest. Hệ thống được kiểm thử trên dữ liệu của cặp EUR/USD và USD/JPY (2008–2013), cho thấy lợi nhuận tăng trung bình 10% so với chiến lược chỉ sử dụng phân tích kỹ thuật. Nghiên cứu này nhấn mạnh giá trị của việc tích hợp phương pháp truyền thống và hiện đại nhằm duy trì hiệu suất ổn định trong điều kiện thị trường thay đổi liên tục.

Một hướng tiếp cận khác là sử dụng học tăng cường thích nghi, như được trình bày trong nghiên cứu của **Dempster và Leemans (2018)**. Hệ thống giao dịch do các tác giả đề xuất gồm ba lớp chức năng: học máy, quản lý rủi ro và tối ưu hóa tiện ích. Được kiểm chứng trên dữ liệu từ 2010–2017, hệ thống đạt tỷ lệ Sharpe trung bình 1.5 và cải thiện kiểm soát rủi ro hơn 20% so với các chiến lược truyền thống. Nghiên cứu này cho thấy tiềm năng rõ rệt của học tăng cường trong môi trường Forex có tính biến động cao.

Gần đây hơn, **Kovalev et al. (2024)** đề xuất hệ thống giao dịch đơn giản hóa dựa trên tối ưu hóa đa tiêu chí mờ. Hệ thống sử dụng các chỉ báo như đường trung bình động và RSI, sau đó áp dụng kỹ thuật mờ để lựa chọn chiến lược phù hợp với từng điều kiện thị trường. Kết quả thử nghiệm trên dữ liệu từ 2018–2023 cho thấy lợi nhuận trung bình tăng 13%, trong khi mức rút vốn tối đa giảm xuống dưới 8%. Cách tiếp cận này đặc biệt hữu ích khi tích hợp thêm dữ liệu thời gian thực để nâng cao tính linh hoạt.

Ở góc độ kiến trúc hệ thống, nghiên cứu của **Kowalska-Pyzalska, Maciejewska và Katarzyna (2024)** giới thiệu một nền tảng đa tác nhân, trong đó mỗi tác nhân đảm nhận một chức năng riêng như phân tích xu hướng, quản lý rủi ro, hoặc tối ưu hóa chiến lược. Kiểm thử trên dữ liệu từ 2020–2023 cho thấy hiệu suất cải thiện từ 10–15% so với các chiến lược đơn lẻ, đồng thời tăng khả năng phản ứng linh hoạt trước các biến động thị trường.

Về các hệ thống đơn giản và dễ triển khai, **Ozturk, Toroslu và Fidan (2017)** đã phát triển hệ thống dựa trên quy tắc kỹ thuật như EMA và mô hình cờ để xác định điểm vào/ra lệnh. Hệ thống được kiểm thử trên dữ liệu nội ngày (intraday) của chỉ số DJIA (2014–2016), với kết quả lợi nhuận tăng trên 8% so với chiến lược mua và giữ, cùng khả năng kiểm soát rủi ro tốt nhờ quy tắc dừng lỗ động.

Một cái nhìn tổng quan về các phương pháp học máy và tối ưu hóa hệ thống giao dịch được trình bày trong bài khảo sát của **Yang và Yang (2018)**. Dù không có kiểm thử thực nghiệm, nghiên cứu cung cấp cái nhìn toàn diện về các kỹ thuật như hồi quy, mạng nơ-ron, học tăng cường, đồng thời nhấn mạnh vai trò của dữ liệu thời gian thực và triển khai phần cứng trong tăng cường hiệu quả hệ thống.

Cuối cùng, nghiên cứu của **Moraes và Silva (2012)** phát triển hệ thống giao dịch trên nền tảng MetaTrader 5, sử dụng chỉ báo kỹ thuật đơn giản như MACD và EMA. Được kiểm chứng trên dữ liệu Forex (2009–2011), hệ thống cho thấy hiệu suất tăng khoảng 10% so với giao dịch thủ công. Nghiên cứu này phù hợp với các nhà phát triển mới bắt đầu, và có thể mở rộng hiệu quả hơn khi kết hợp dữ liệu thời gian thực từ các nền tảng như Forex Factory.

# CHƯƠNG 3. PHƯƠNG PHÁP NGHIÊN CỨU

## 3.1. Bối cảnh nghiên cứu

Thị trường ngoại hối (Forex) là một trong những thị trường tài chính lớn nhất thế giới, với khối lượng giao dịch hàng ngày đạt hàng nghìn tỷ đô la. Đây là một môi trường phức tạp, nơi các nhà giao dịch phải đối mặt với biến động giá liên tục và khối lượng thông tin khổng lồ từ dữ liệu thị trường, tin tức kinh tế, và các chỉ báo kỹ thuật. Việc ra quyết định giao dịch chính xác và kịp thời thường là thách thức lớn đối với các nhà đầu tư cá nhân. Trong bối cảnh đó, sự phát triển của công nghệ, đặc biệt là trí tuệ nhân tạo và các hệ thống giao dịch tự động, đã mở ra tiềm năng cải thiện hiệu suất giao dịch và giảm thiểu sai sót của con người. Nghiên cứu này được thực hiện nhằm xây dựng một bot giao dịch Forex tự động, tích hợp khả năng phân tích dữ liệu giá, nhận diện mô hình nến, tính toán chỉ báo kỹ thuật, và thực hiện giao dịch theo thời gian thực. Nghiên cứu không chỉ đóng góp vào việc tự động hóa quy trình giao dịch mà còn cung cấp một công cụ hỗ trợ các nhà đầu tư trong việc tối ưu hóa lợi nhuận và quản lý rủi ro.

## 3.2. Phương pháp thu thập số liệu

### 3.2.1. Thu thập dữ liệu

Hình 3. 1 Minh họa API lấy data

Quá trình thu thập dữ liệu khởi đầu bằng việc sử dụng các điểm cuối API để truy xuất dữ liệu lịch sử về các thanh giá 'ask' và 'bid' cho cặp tiền tệ EUR/USD. Như hình ảnh minh họa, có hai yêu cầu GET được xác định: một để lấy thông tin meta về các thanh 'ask' và một để lấy thông tin meta về các thanh 'bid'. Hàm Python fetch_candles được thiết kế để tương tác với các điểm cuối này, nhận các tham số như cặp tiền tệ, số lượng nến cần lấy, khoảng thời gian (ví dụ 'H1' cho hàng giờ), và một timestamp tùy chọn. Hàm này xây dựng các URL yêu cầu API phù hợp, quản lý logic timestamp để đảm bảo lấy dữ liệu đến thời điểm hiện tại nếu không có timestamp cụ thể, và thực thi các yêu cầu để thu thập cả dữ liệu 'ask' và 'bid', sau đó lưu trữ vào các tệp. Bước này đóng vai trò nền tảng trong việc thu thập dữ liệu giá lịch sử thô, là cơ sở cho các giai đoạn xử lý và phân tích sau. 

Hình 3. 2 Lấy nến của cặp tiền tệ

Sau khi dữ liệu thô được thu thập, nó cần được xử lý và chuyển đổi thành định dạng có cấu trúc phù hợp cho phân tích tài chính. Hàm get_candles_df, như được trình bày trong hình, đảm nhận nhiệm vụ này. Hàm nhận dữ liệu 'ask' và 'bid' đã thu thập, thực hiện các kiểm tra để đảm bảo tính sẵn có của dữ liệu, và trích xuất các thanh nến. Sau đó, hàm chuyển đổi các thanh nến này thành các DataFrame của pandas, một cho 'ask' và một cho 'bid', rồi hợp nhất chúng thành một DataFrame duy nhất. Ngoài ra, hàm tính toán giá trung bình bằng cách lấy trung bình của các điểm giá tương ứng từ 'ask' và 'bid' cho các giá trị mở cửa, cao nhất, thấp nhất và đóng cửa. Quá trình xử lý này đảm bảo rằng dữ liệu được làm sạch, tổ chức hợp lý và sẵn sàng cho việc phát triển chiến lược giao dịch hoặc thực hiện phân tích thị trường.

### 

Hình 3. 3 Thu thập và thêm cột vào dữ liệu

Để xây dựng một bộ dữ liệu lịch sử toàn diện, quá trình thu thập dữ liệu được thực hiện một cách có hệ thống qua nhiều khoảng thời gian khác nhau. Hình ảnh cho thấy một danh sách các mục thu thập dữ liệu, mỗi mục đại diện cho một khoảng thời gian cụ thể mà dữ liệu được thu thập cho cặp EUR/USD trên khung thời gian hàng giờ. Ví dụ, một mục ghi nhận việc thu thập 900 nến từ ngày 18 tháng 3 năm 2021 đến ngày 11 tháng 5 năm 2021, và một mục khác từ ngày 11 tháng 5 năm 2021 đến ngày 2 tháng 7 năm 2021, và cứ tiếp tục như vậy. Cách tiếp cận này cho phép tích lũy dần một bộ dữ liệu lớn bằng cách lặp lại việc thu thập và thêm các khoảng thời gian dữ liệu mới, đảm bảo tính liên tục và đầy đủ của bộ dữ liệu, điều này rất quan trọng cho việc phân tích lịch sử chính xác và kiểm tra lại các chiến lược giao dịch.

Hình 3. 4 Kết quả lấy data

Giai đoạn cuối cùng của quá trình thu thập dữ liệu liên quan đến việc xác minh tính hoàn chỉnh và chính xác của dữ liệu đã thu thập thông qua việc ghi nhật ký chi tiết. Thông tin bao gồm: khoảng thời gian thu thập dữ liệu, cặp tiền tệ, khung thời gian, ngày giờ bắt đầu và kết thúc, cùng với số lượng nến thu thập được. Nhật ký này không chỉ đóng vai trò là bản ghi của quá trình thu thập mà còn cung cấp tóm tắt về tổng số dữ liệu tích lũy, đạt 107.550 nến từ năm 2021 đến năm 2025.

### 3.2.2. Xử lý dữ liệu

**3.2.2.1. Thêm cột cho data**

Dữ liệu giá trong nghiên cứu này được thu thập thông qua việc sử dụng API của sàn FXOpen, cung cấp hai luồng thông tin chính: giá Bid (giá mua) và giá Ask (giá bán). Đối tượng thu thập là cặp ngoại tệ EURUSD, một trong những cặp tiền tệ phổ biến nhất trên thị trường Forex nhờ tính thanh khoản cao và tầm quan trọng trong giao dịch toàn cầu. Dữ liệu được ghi nhận theo khung thời gian hàng giờ (H1), mang lại cái nhìn chi tiết về biến động giá trong suốt khoảng thời gian từ ngày 1 tháng 1 năm 2008 đến năm 2025.

Hình 3. 5 Ảnh dữ liệu sau khi được thêm cột

Hệ thống được thiết kế để tự động gọi API từ sàn FXOpen nhằm thu thập dữ liệu giá của cặp ngoại tệ EURUSD. Dữ liệu bao gồm **giá Bid** (giá mà thị trường sẵn sàng mua) và **giá Ask** (giá mà thị trường sẵn sàng bán). Quá trình này diễn ra liên tục theo khung thời gian giờ (H1), đảm bảo dữ liệu luôn được cập nhật theo thời gian thực.

**Dữ liệu gồm 4 loại giá: open, high, low, close cho cả bid và ask**:  
Tại mỗi mốc thời gian (mỗi giờ), dữ liệu thu thập được bao gồm bốn loại giá chính cho cả **Bid** và **Ask**:

- **Open**: Giá mở cửa, là giá tại thời điểm bắt đầu của giờ.
- **High**: Giá cao nhất đạt được trong giờ.
- **Low**: Giá thấp nhất trong giờ.
- **Close**: Giá đóng cửa, là giá tại thời điểm kết thúc của giờ.  
    Như vậy, mỗi giờ cung cấp tổng cộng 8 giá trị: 4 giá trị cho Bid (bid_open, bid_high, bid_low, bid_close) và 4 giá trị cho Ask (ask_open, ask_high, ask_low, ask_close).

Để phản ánh giá thị trường một cách chính xác và khách quan hơn, nhóm nghiên cứu tính toán giá trung bình (Mid) giữa giá Bid và giá Ask cho từng loại giá. Công thức áp dụng là:

Cụ thể, các giá trị trung bình:

- **mid_o**: giá mở cửa trung bình
- **mid_h**: giá cao nhất trung bình
- **mid_l**: giá thấp nhất trung bình
- **mid_c**: giá đóng cửa trung bình

Các giá trị trung bình này, đặc biệt là **mid_c**, được sử dụng làm cơ sở quan trọng trong việc phân tích và dự báo, giúp giảm thiểu sai lệch từ việc chỉ sử dụng giá Bid hoặc Ask riêng lẻ, từ đó hỗ trợ tốt hơn cho các quyết định giao dịch.

**3.2.2.2. Khai phá dữ liệu**

Hình 3. 6 Khai phá dữ liệu

**Số dòng:** 107,550 dòng dữ liệu (rất lớn, phù hợp để phân tích chuyên sâu).

**Số cột:** 13 cột, gồm: thời gian, giá bid, giá ask, giá mid (mỗi loại gồm: open, high, low, close).

Tần suất: **1 dòng mỗi giờ** → Dữ liệu time series theo giờ, liên tục gần **17 năm**.

Có đủ 3 loại giá:

bid_\*: Giá mua

ask_\*: Giá bán

mid_\*: Giá trung bình giữa bid và ask

Mỗi loại đều có: open, high, low, close

Hình 3. 7 Thông tin chi tiết dữ liệu

Cột time có kiểu dữ liệu datetime64\[ns\], trong khi 12 cột còn lại đều thuộc kiểu float64, đại diện cho các giá trị giá bid, ask và mid ở các mức open, high, low và close.

Dữ liệu hoàn toàn không có giá trị thiếu (null) ở bất kỳ cột nào, chứng tỏ đây là một điểm thuận lợi cho quá trình phân tích và xây dựng mô hình.

Hình 3. 8 Thống kê mô tả dữ liệu

Bảng thống kê mô tả (summary statistics) cho tập dữ liệu, bao gồm 13 cột với tổng cộng 107,550 dòng. Các thống kê như giá trị trung bình (mean), giá trị nhỏ nhất (min), lớn nhất (max), tứ phân vị (25%, 50%, 75%) và độ lệch chuẩn (std) được trình bày đầy đủ cho tất cả các cột liên quan đến giá.

Nhìn chung, dữ liệu có sự phân bố khá đồng đều và hợp lý:

Giá trị trung bình (mean) của các cột bid, ask, và mid dao động quanh mức 1.21, phản ánh tỷ giá trung bình ổn định trong giai đoạn quan sát.

Giá trị nhỏ nhất (min) ở các cột giá nằm khoảng 0.95, trong khi giá trị lớn nhất (max) đạt tới 1.60 → cho thấy tỷ giá biến động đáng kể trong gần 17 năm.

Tứ phân vị (Q1, Q2, Q3) cho thấy sự phân bố giá không quá lệch, với khoảng giữa (IQR) tương đối hẹp, chứng tỏ không có nhiều outlier lớn.

Độ lệch chuẩn (std) dao động từ 0.136 đến 0.137, cho thấy mức độ biến động giá vừa phải và nhất quán giữa các loại giá bid, ask và mid.

Hình 3. 9 Biến động tỷ giá đóng cửa trung bình (Mid Close Price) của cặp tiền tệ EUR/USD trong giai đoạn từ năm 2008 đến 2025

**Giai đoạn 2008–2014**: Tỷ giá biến động mạnh và đạt đỉnh gần **1.6**, sau đó suy giảm dần. Giai đoạn này thể hiện rõ các cú sốc tài chính sau khủng hoảng kinh tế toàn cầu.  

**Từ 2014 trở đi**: Tỷ giá bước vào xu hướng giảm dài hạn và ổn định hơn, dao động chủ yếu quanh mức **1.1–1.2**.

**Năm 2017–2021**: Có một số pha hồi phục nhẹ, nhưng không vượt quá mức 1.25.

**Từ 2022 đến 2023**: Tỷ giá chạm đáy quanh mức **1.0**, đây có thể là thời điểm chịu ảnh hưởng từ lạm phát toàn cầu và bất ổn địa chính trị.

**Cuối giai đoạn (2024–2025)**: Xuất hiện tín hiệu tăng nhẹ trở lại, tuy nhiên vẫn trong vùng thấp lịch sử.

Hình 3. 10 Biểu đồ histogram của biến "mid_c"

Biểu đồ histogram của biến "mid_c" cho thấy một phân bố lệch phải, với phần lớn các quan sát tập trung trong khoảng từ 1.1 đến 1.2, nơi số lượng vượt quá 6000. Ngoài ra, có các đỉnh nhỏ hơn xuất hiện ở khoảng 1.3 và 1.4, với số lượng lần lượt khoảng 3000 và 2000, gợi ý về khả năng tồn tại các nhóm phụ hoặc cụm dữ liệu trong tập hợp. Phân bố giảm mạnh sau 1.5, với rất ít quan sát ở các giá trị cao hơn từ 1.5 đến 1.6. Giá trị độ lệch (skewness) là 0.63368, xác nhận tính chất lệch phải của phân bố, trong khi độ nhọn (kurtosis) là -0.537120 cho thấy dữ liệu ít tập trung quanh giá trị trung bình và có các đuôi nhẹ hơn so với phân bố chuẩn.

Hình 3. 11 Biên độ dao động giá theo giờ (Hourly Price Range) của tỷ giá EUR/USD dựa trên giá mid trong suốt giai đoạn từ năm 2008 đến 2025

Giai đoạn 2008–2010: Biến động giá theo giờ rất cao, với nhiều thời điểm biên độ vượt ngưỡng 0.025 đến 0.03. Đây có thể là hậu quả của khủng hoảng tài chính toàn cầu, khiến thị trường có nhiều cú sốc mạnh.

Từ 2011 đến khoảng 2017: Mặc dù vẫn còn biến động, mức độ dao động đã giảm dần, tuy vẫn xuất hiện một số đợt tăng đột biến.

Sau 2018 đến 2021: Mức dao động ổn định hơn, với biên độ trung bình nhỏ hơn 0.01, phản ánh giai đoạn thị trường tương đối bình lặng.

Giai đoạn 2022–2024: Biến động tăng trở lại nhẹ, có thể liên quan đến các yếu tố như lạm phát toàn cầu, xung đột địa chính trị hoặc chính sách lãi suất.

Hình 3. 12 Biểu đồ thể hiện phân phối lợi nhuận theo giờ (Hourly Returns)

Phân phối có dạng chuông đối xứng, tập trung mạnh quanh giá trị 0, cho thấy phần lớn các biến động theo giờ là rất nhỏ — đây là đặc trưng thường thấy trong thị trường ngoại hối có tính thanh khoản cao như EUR/USD.

Đuôi phân phối mỏng, tức là lợi nhuận lớn (âm hoặc dương) rất hiếm gặp. Đa số biến động nằm trong khoảng từ -0.005 đến 0.005.

Đỉnh của histogram rất nhọn, với hơn 50.000 giá trị nằm gần mức return = 0, chứng tỏ thị trường thường xuyên có các khung giờ mà giá gần như không thay đổi.

# CHƯƠNG 4. KẾT QUẢ VÀ ĐÁNH GIÁ

## 4.1. Kết quả của mô hình baseline: Sử dụng đường SMA

Hình 4. 1 Kết quả của mô hình baseline

Mô hình Simple Moving Average (SMA) được sử dụng làm mô hình cơ bản để dự đoán giá đóng cửa của cặp tỷ giá EURUSD trong khoảng thời gian từ ngày 01/10/2024 đến 13/5/2024. SMA là một phương pháp phổ biến trong phân tích chuỗi thời gian, được sử dụng để làm mượt các biến động ngắn hạn và nhận diện xu hướng dài hạn. Mô hình SMA tính toán giá trị trung bình của các giá đóng cửa trong cửa sổ 240 giờ trước đó và sử dụng nó làm dự đoán cho các giờ tiếp theo. Cửa sổ di động này tương ứng với khoảng 10 ngày giao dịch, cho phép mô hình nhận diện xu hướng chung của thị trường trong dài hạn.

Việc sử dụng mô hình SMA trong nghiên cứu này không chỉ bởi tính hiệu quả trong việc nhận diện xu hướng thị trường, mà còn vì những lý do về tính đơn giản và dễ tiếp cận. SMA là một phương pháp truyền thống, dễ hiểu và có thể áp dụng ngay cả với những người không có nền tảng vững về học máy hoặc thống kê. Mô hình này chỉ đơn giản tính toán giá trị trung bình của các dữ liệu trong một cửa sổ thời gian cố định, giúp làm mượt các biến động ngắn hạn và dễ dàng nhận diện xu hướng dài hạn của dữ liệu. Chính vì vậy, SMA được coi là một công cụ dễ tiếp cận và dễ hiểu đối với đa số người dùng, bao gồm cả các nhà đầu tư và các nhà phân tích tài chính. Bên cạnh đó, phương pháp này không yêu cầu kiến thức phức tạp hay điều chỉnh tham số tinh vi, và có thể được triển khai nhanh chóng mà không đòi hỏi nhiều tài nguyên tính toán. Điều này đặc biệt hữu ích trong môi trường có tài nguyên hạn chế hoặc khi cần một mô hình dễ dàng cập nhật với dữ liệu mới.

SMA cũng được ưu tiên sử dụng trong nghiên cứu này vì nó mang lại khả năng kiểm tra và so sánh một mô hình đơn giản trước khi áp dụng các phương pháp phức tạp hơn như học máy. Đây là một bước quan trọng trong việc tạo ra điểm chuẩn để đánh giá hiệu quả của các mô hình học máy, đặc biệt trong bối cảnh các phương pháp phức tạp có thể gặp khó khăn trong việc xử lý các vấn đề như overfitting hay yêu cầu tài nguyên tính toán lớn.

Dữ liệu sử dụng trong nghiên cứu này được lấy từ tỷ giá EURUSD theo giờ từ ngày 01/01/2020 đến 13/05/2025. Dữ liệu được chuẩn hóa bằng phương pháp MinMaxScaler để đưa các giá trị về phạm vi \[0, 1\], giúp đồng nhất quy trình mô hình hóa. Tuy nhiên, đối với việc tính toán SMA, dữ liệu gốc được sử dụng trực tiếp mà không cần phục hồi lại sau chuẩn hóa. Mô hình dự đoán giá đóng cửa cho 6 giờ tiếp theo tại mỗi thời điểm dựa trên giá trị SMA của cửa sổ 240 giờ.

Sau khi áp dụng mô hình SMA trên tập kiểm tra từ ngày 01/10/2024 đến 13/05/2025, kết quả cho thấy mô hình có khả năng dự đoán khá chính xác với các chỉ số đánh giá như Mean Absolute Error (MAE) đạt 0.006877, Root Mean Squared Error (RMSE) là 0.009148, Mean Absolute Percentage Error (MAPE) là 0.63%, và R² đạt 0.874746. Điều này cho thấy mô hình có thể giải thích được gần 87.5% phương sai trong dữ liệu và có độ chính xác cao.

Biểu đồ so sánh giữa giá trị thực tế và giá trị dự đoán cho thấy mô hình SMA hoạt động tốt trong các giai đoạn thị trường ổn định và có xu hướng rõ ràng. Tuy nhiên, trong các giai đoạn biến động mạnh, mô hình SMA tỏ ra có sự độ trễ rõ rệt và không thể phản ứng kịp thời với các thay đổi đột ngột của thị trường. Đây là đặc điểm của SMA khi được sử dụng trong các tình huống có sự biến động mạnh mẽ.

Mặc dù mô hình SMA là một phương pháp đơn giản nhưng hiệu quả trong các tình huống ổn định, kết quả cũng cho thấy rằng độ trễ của mô hình trong các giai đoạn thị trường biến động mạnh có thể là một hạn chế. Để cải thiện khả năng dự đoán trong các tình huống này, chúng tôi đề xuất áp dụng các mô hình phức tạp hơn như Exponential Moving Average (EMA), ARIMA, LSTM (Long Short-Term Memory), hoặc các mô hình học máy như XGBoost. Những phương pháp này có thể giúp mô hình bắt kịp những thay đổi nhanh chóng và không tuyến tính của thị trường, đặc biệt là trong những tình huống có biến động mạnh, từ đó cải thiện độ chính xác của các dự đoán. Việc so sánh SMA với các mô hình học máy cũng sẽ giúp đánh giá rõ ràng hơn về những ưu điểm và nhược điểm của các phương pháp truyền thống so với các kỹ thuật hiện đại trong dự báo chuỗi thời gian.

## 4.2. Kết quả mô hình

### 4.2.1. Kết quả của mô hình Random Forest

Hình 4. 2 So sánh giá trị thực tế và dự đoán sử dụng mô hình Random Forest

Kết quả mô hình dự báo giá EURUSD sử dụng Random Forest cho thấy một mức độ chính xác cao trong việc dự đoán giá đóng cửa của EURUSD. Để có cái nhìn rõ ràng và chi tiết hơn về quá trình huấn luyện, kiểm tra mô hình, cũng như mối liên hệ với thực tế, ta sẽ phân tích từng phần của quy trình mô hình hóa, dữ liệu, các chỉ số đánh giá, và cuối cùng là mối liên hệ với thị trường tài chính thực tế.

**_Dữ liệu và Quá trình Huấn luyện Mô Hình_**

Dữ liệu được sử dụng trong nghiên cứu này là dữ liệu giá EURUSD theo giờ (EURUSD H1), kéo dài từ tháng 01/2020 đến tháng 05/2025. Dữ liệu này được xử lý và chia thành các tập huấn luyện (train) và kiểm tra (test). Dữ liệu được chuẩn hóa bằng phương pháp MinMaxScaler, giúp đưa tất cả các giá trị về cùng một khoảng từ 0 đến 1. Việc chuẩn hóa này giúp tăng độ ổn định và khả năng hội tụ của mô hình học máy, đặc biệt là với các mô hình như Random Forest, vốn không yêu cầu chuẩn hóa dữ liệu nhưng vẫn được hưởng lợi từ quá trình này (Pedregosa et al., 2011).

**Quá trình chia dữ liệu** được thực hiện với tập huấn luyện được chia từ trước ngày 01/10/2023, và tập kiểm tra từ 01/10/2024 đến 13/05/2025. Điều này đảm bảo rằng mô hình được huấn luyện trên các dữ liệu lịch sử và đánh giá trên dữ liệu không nhìn thấy trước (out-of-sample), giúp mô hình có thể phản ánh chính xác khả năng dự báo khi đối mặt với các dữ liệu mới.

**Tạo chuỗi dữ liệu (sequence generation)** là một kỹ thuật quan trọng trong bài toán dự báo chuỗi thời gian, nơi các chuỗi đầu vào có chiều dài cố định (240 ngày) được sử dụng để dự đoán các giá trị đầu ra trong tương lai (6 ngày). Phương pháp này giúp mô hình nắm bắt các xu hướng dài hạn và ngắn hạn của chuỗi thời gian, giúp mô hình có thể dự đoán giá trị tốt hơn khi có sự thay đổi về xu hướng trong thời gian ngắn.

**_Mô Hình và Các Tham Số_**

Mô hình Random Forest được chọn vì tính linh hoạt và khả năng xử lý các quan hệ phi tuyến tính giữa các đặc trưng. Mô hình này sử dụng một tập hợp các cây quyết định (decision trees) để đưa ra dự báo, và kết quả cuối cùng được tính toán từ việc tổng hợp kết quả của tất cả các cây trong rừng (Breiman, 2001).

Các tham số của mô hình được thiết lập như sau:

- **n_estimators** = 20: Số lượng cây trong rừng.  
    
- **max_depth** = 3: Độ sâu tối đa của mỗi cây.  
    
- **random_state** = 42: Đảm bảo tính tái lập kết quả.  
    

Các tham số này được chọn sao cho mô hình có thể học tốt trên tập huấn luyện mà không bị overfitting (học quá sát dữ liệu huấn luyện), đồng thời có thể tổng quát hóa tốt trên tập kiểm tra.

**_Đánh Giá Mô Hình_**

Sau khi huấn luyện, mô hình được đánh giá trên tập kiểm tra với các chỉ số như MAE (Mean Absolute Error), RMSE (Root Mean Squared Error), MAPE (Mean Absolute Percentage Error) và R² (Coefficient of Determination).

- **MAE (0.004671)**: Cho thấy mức độ sai số tuyệt đối trung bình giữa giá trị thực tế và giá trị dự đoán là rất thấp.  
    
- **RMSE (0.005576)**: Chỉ số này cho thấy độ lệch chuẩn của các sai số dự đoán, và với giá trị nhỏ như vậy, mô hình có khả năng dự đoán chính xác.  
    
- **MAPE (0.44%)**: Chỉ ra rằng sai số trung bình của mô hình là chưa đến 0.5%, điều này cực kỳ ấn tượng trong các bài toán tài chính.  
    
- **R² (0.972894)**: Cho thấy mô hình giải thích được hơn 97% sự biến động của dữ liệu, chứng tỏ rằng mô hình rất mạnh trong việc dự đoán các xu hướng giá của EURUSD.  
    

Biểu đồ so sánh giá trị thực tế và dự đoán cho thấy sự khớp nhau giữa giá trị thực tế và dự đoán của mô hình. Các đường màu xanh dương (giá trị thực tế) và đỏ (giá trị dự đoán) gần như chồng khít lên nhau, đặc biệt là trong các xu hướng giá dài hạn, với một số sai lệch nhỏ ở các điểm biến động mạnh, điều này là bình thường trong các thị trường tài chính có sự biến động lớn.

Mô hình này có thể áp dụng trực tiếp trong các chiến lược giao dịch tài chính. Dự báo chính xác giá trị EURUSD trong tương lai là rất quan trọng đối với các nhà đầu tư và các tổ chức tài chính, đặc biệt trong môi trường có tính biến động cao. Với MAE và MAPE cực thấp, mô hình cho thấy tiềm năng ứng dụng vào các chiến lược giao dịch tự động (algorithmic trading) hoặc phân tích kỹ thuật trong việc dự đoán các biến động giá ngắn hạn.

Tuy nhiên, cần lưu ý rằng mô hình không phải lúc nào cũng hoàn hảo, nhất là khi thị trường có các yếu tố ngoài dự đoán như tin tức kinh tế, chính trị, hoặc các sự kiện không lường trước khác. Trong thực tế, các mô hình học máy như Random Forest có thể bị ảnh hưởng bởi các yếu tố này, mặc dù độ chính xác cao, nhưng vẫn cần kết hợp với các yếu tố khác như phân tích cơ bản và kỹ thuật để đưa ra quyết định giao dịch.

Mô hình Random Forest cho dự báo giá EURUSD trên chuỗi thời gian này đã chứng tỏ khả năng dự báo mạnh mẽ, với các chỉ số đánh giá cho thấy mức độ chính xác rất cao. Tuy nhiên, để mô hình có thể hoạt động tốt hơn trong môi trường thực tế, việc kết hợp với các yếu tố khác, như dữ liệu vĩ mô và các mô hình phức tạp hơn, có thể giúp cải thiện độ chính xác và khả năng tổng quát của mô hình.

### 4.2.2. Kết quả của mô hình GRU

Hình 4. 3 So sánh giá trị thực tế và dự đoán sử dụng mô hình GRU

Kết quả mô hình GRU (Gated Recurrent Unit) với Attention trong việc dự đoán giá EURUSD đã thể hiện một mức độ chính xác khá tốt, nhưng có một số điểm cần lưu ý để cải thiện mô hình và điều chỉnh để đạt được hiệu suất tối ưu hơn.

**_Dữ Liệu và Quá Trình Huấn Luyện_**

Dữ liệu sử dụng cho mô hình này bao gồm giá EURUSD theo giờ (EURUSD H1), từ ngày 01/01/2020 đến 13/05/2025. Dữ liệu được chuẩn hóa bằng MinMaxScaler để đưa các giá trị về phạm vi từ 0 đến 1, giúp mô hình học nhanh hơn và giảm thiểu sự ảnh hưởng của các biến động lớn trong dữ liệu gốc.

**Quá trình chia dữ liệu** được thực hiện, với tập huấn luyện từ trước ngày 01/10/2023 và tập kiểm tra từ 01/10/2024 đến 13/05/2025, giúp mô hình học trên dữ liệu lịch sử và kiểm tra trên dữ liệu chưa thấy, phản ánh chính xác khả năng tổng quát của mô hình.

Đặc biệt, trong quá trình huấn luyện, **GRU** (một loại mạng nơ-ron tái hồi) được sử dụng thay thế LSTM vì GRU thường cho thấy khả năng hội tụ nhanh và yêu cầu ít tài nguyên tính toán hơn, nhưng vẫn giữ được khả năng học các đặc trưng chuỗi thời gian dài.

**_Đánh Giá Mô Hình_**

Các chỉ số đánh giá mô hình trên tập kiểm tra cho thấy mô hình hoạt động khá tốt, nhưng vẫn có những sai số nhất định:

- **MAE (Mean Absolute Error)** là 0.008990, cho thấy sai số tuyệt đối trung bình của mô hình là 0.008990, khá thấp nhưng cao hơn so với mô hình Random Forest trước đó.  
    
- **RMSE (Root Mean Squared Error)** đạt 0.011974, cho thấy độ lệch chuẩn của sai số dự đoán. Giá trị này cho thấy mô hình không hoàn toàn chính xác, đặc biệt khi xảy ra các biến động lớn trong giá trị EURUSD.  
    
- **MAPE (Mean Absolute Percentage Error)** là 0.83%, chỉ ra rằng sai số trung bình theo tỷ lệ phần trăm của mô hình là 0.83%, vẫn ở mức khá thấp, nhưng vẫn cao hơn so với Random Forest (0.44%).  
    
- **R² (Coefficient of Determination)** là 0.875014, cho thấy mô hình giải thích được khoảng 87.5% sự biến động của dữ liệu, một kết quả khá tốt nhưng thấp hơn một chút so với mô hình Random Forest với R² là 0.972894.  
    

**_Biểu Đồ So Sánh Giá Trị Thực Tế và Dự Đoán_**

Biểu đồ hiển thị sự so sánh giữa giá trị thực tế (đường màu xanh) và giá trị dự đoán (đường màu đỏ). Nhìn chung, các đường này khá khớp nhau, nhưng một số điểm cụ thể, đặc biệt là tại các thời điểm biến động mạnh của thị trường, mô hình GRU có sự sai lệch lớn hơn so với Random Forest. Điều này có thể là do GRU không nhận diện được các yếu tố tác động mạnh đến giá trị EURUSD trong những giai đoạn có biến động lớn.

Trong các bài toán dự báo tài chính như dự đoán giá EURUSD, việc mô hình hóa đúng xu hướng và dự báo chính xác giá trị là rất quan trọng đối với các nhà đầu tư và các tổ chức tài chính. Tuy nhiên, như đã thấy trong kết quả, mô hình GRU mặc dù cho kết quả khá tốt nhưng vẫn chưa hoàn toàn chính xác, đặc biệt là trong các giai đoạn biến động mạnh. Để cải thiện mô hình, có thể thử kết hợp thêm các yếu tố ngoại vi như các chỉ số kinh tế, các sự kiện thế giới, hoặc thử nghiệm với các mô hình phức tạp hơn như LSTM với attention mechanism hoặc các mô hình ensemble.

Mô hình GRU kết hợp với Attention đã cho thấy khả năng dự đoán tốt, với các chỉ số đánh giá như MAE, RMSE, MAPE và R² ở mức chấp nhận được, tuy nhiên vẫn có một số điểm sai lệch, đặc biệt khi thị trường có những biến động lớn. Việc kết hợp thêm các yếu tố và điều chỉnh mô hình sẽ giúp cải thiện độ chính xác trong các dự đoán giá EURUSD, đặc biệt trong những giai đoạn có sự biến động mạnh.

### 4.2.3. Kết quả của mô hình RNN

Hình 4. 4 So sánh giá trị thực tế và dự đoán của mô hình RNN

**_Dữ liệu và Quá Trình Huấn Luyện_**

Dữ liệu sử dụng cho mô hình này là chuỗi thời gian giá EURUSD theo giờ từ ngày 01/01/2020 đến 13/05/2025. Dữ liệu đã được chuẩn hóa sử dụng **MinMaxScaler**, đưa các giá trị vào phạm vi từ 0 đến 1, nhằm hỗ trợ mô hình học dễ dàng hơn.

**Quá trình chia dữ liệu** được thực hiện với tập huấn luyện từ trước ngày 01/10/2023 và tập kiểm tra từ ngày 01/10/2024 đến 13/05/2025. Mô hình RNN với Attention đã được huấn luyện với các chuỗi thời gian có chiều dài **240** ngày (lookback) và dự đoán cho **6** ngày tiếp theo (output_len).

**_Kết Quả Đánh Giá Mô Hình_**

- **MAE (Mean Absolute Error)** = 0.006782: Sai số tuyệt đối trung bình thấp cho thấy mô hình dự đoán giá trị gần với giá trị thực tế.  
    
- **RMSE (Root Mean Squared Error)** = 0.009014: Lỗi bình phương trung bình căn bậc hai khá thấp, cho thấy sự sai lệch chuẩn giữa các giá trị thực tế và dự đoán là nhỏ.  
    
- **MAPE (Mean Absolute Percentage Error)** = 0.63%: Sai số phần trăm trung bình nhỏ hơn 1%, điều này cho thấy mô hình dự đoán rất chính xác.  
    
- **R² (Coefficient of Determination)** = 0.929177: Khoảng 92.9% sự biến động của giá trị EURUSD được mô hình giải thích, cho thấy một mức độ dự đoán rất mạnh mẽ và chính xác.  
    

**_So Sánh Biểu Đồ Giá Trị Thực Tế và Dự Đoán_**

Biểu đồ thể hiện sự so sánh giữa giá trị thực tế (đường màu xanh) và giá trị dự đoán (đường màu đỏ) cho thấy mô hình RNN với Attention có khả năng dự đoán rất sát với giá trị thực tế, đặc biệt là trong các giai đoạn có biến động mạnh của thị trường. Tuy nhiên, cũng có một số sai lệch nhỏ trong các giai đoạn có sự thay đổi mạnh mẽ.

Mô hình RNN kết hợp với Attention đã cho kết quả rất khả quan trong việc dự đoán giá EURUSD. Điều này có thể áp dụng trực tiếp vào các chiến lược giao dịch tài chính và phân tích kỹ thuật, đặc biệt trong môi trường có sự biến động cao như thị trường ngoại hối. Tuy nhiên, cũng cần lưu ý rằng mô hình RNN có thể gặp khó khăn khi dự đoán các sự kiện đột ngột như các cuộc khủng hoảng tài chính hoặc thay đổi lớn về chính sách của các quốc gia. Do đó, kết hợp thêm các yếu tố khác như dữ liệu vĩ mô hoặc các mô hình phức tạp hơn có thể giúp tăng cường độ chính xác của dự đoán.

Mô hình RNN với Attention đã chứng tỏ khả năng dự đoán chính xác giá EURUSD, với các chỉ số đánh giá tốt và độ chính xác cao. Việc sử dụng attention giúp mô hình có thể học được các đặc trưng quan trọng trong chuỗi thời gian, mang lại sự chính xác trong việc dự báo xu hướng giá.

  
**4.2.4. Kết quả của mô hình LSTM**

Hình 4. 5 So Sánh giá trị thực tế và dự đoán EURUSD H1 (01/10/2024 – 13/05/2025) - LSTM

Mô hình LSTM được áp dụng để dự báo tỷ giá cặp tiền tệ EURUSD trong giai đoạn từ 01/01/2024 đến 13/5/2025 cho thấy hiệu suất dự báo ấn tượng với khả năng học hỏi và tái tạo các xu hướng chính của thị trường ngoại hối. Các chỉ số đánh giá đều phản ánh độ chính xác cao của mô hình, trong đó sai số tuyệt đối trung bình MAE đạt 0.007005 và sai số bình phương trung bình RMSE là 0.009318, cho thấy mô hình có khả năng dự báo ổn định với sai số được kiểm soát tốt.

Đặc biệt ấn tượng là chỉ số MAPE chỉ 0.65%, một kết quả xuất sắc trong lĩnh vực dự báo tỷ giá ngoại hối, cho thấy độ tin cậy cao của mô hình. Hệ số xác định R² đạt 0.870045 có nghĩa là mô hình giải thích được 87% biến động của tỷ giá thực tế, thể hiện khả năng nắm bắt patterns và xu hướng của thị trường một cách hiệu quả.

Từ biểu đồ có thể quan sát thấy tỷ giá EURUSD trải qua nhiều giai đoạn biến động rõ rệt trong suốt thời kỳ nghiên cứu. Trong quý đầu năm 2024, tỷ giá dao động tương đối ổn định quanh mức 1.08-1.09, sau đó bước vào giai đoạn tăng mạnh từ tháng 5 đến tháng 9 với đỉnh đạt khoảng 1.12, phản ánh sự phục hồi của đồng Euro. Tuy nhiên, cuối năm 2024 chứng kiến một đợt điều chỉnh giảm sâu xuống mức thấp nhất khoảng 1.02-1.03, có thể do ảnh hưởng của các yếu tố kinh tế vĩ mô.

Đáng chú ý là giai đoạn đầu năm 2025 cho thấy sự phục hồi mạnh mẽ với tỷ giá tăng vọt lên mức 1.15, thể hiện tính biến động cao đặc trưng của thị trường ngoại hối. Trong suốt các giai đoạn này, đường dự báo của mô hình LSTM thể hiện sự bám sát đáng kể với đường giá thực tế, đặc biệt trong việc dự báo chính xác xu hướng tăng trong quý 2-3 năm 2024, theo dõi tốt xu hướng giảm trong quý 4 năm 2024, và nắm bắt được sự phục hồi mạnh mẽ trong năm 2025.

Tổng kết lại, mô hình LSTM thể hiện hiệu suất dự báo ấn tượng đối với tỷ giá EURUSD với tất cả các chỉ số đánh giá đều ở mức tốt. Trong thực tế ứng dụng, mô hình này có thể được sử dụng như một công cụ hỗ trợ quyết định hiệu quả trong giao dịch ngoại hối, tuy nhiên cần kết hợp với phân tích cơ bản về các yếu tố kinh tế vĩ mô, các chỉ báo kỹ thuật khác để xác nhận tín hiệu, quản lý rủi ro chặt chẽ do tính biến động cao của thị trường, và cập nhật tái huấn luyện mô hình định kỳ để duy trì hiệu suất tối ưu.

##   
4.3 Đánh giá các mô hình

Bảng 4. 1 Đánh giá các mô hình

|     |     |     |     |     |
| --- | --- | --- | --- | --- |
| **Mô hình** | **MAE** | **RMSE** | **MAPE (%)** | **R²** |
| Random Forest | 0.004671 | 0.005576 | 0.44% | 0.972894 |
| GRU | 0.008990 | 0.011974 | 0.83% | 0.875014 |
| RNN | 0.006782 | 0.009014 | 0.63% | 0.929177 |
| LSTM | 0.009123 | 0.011994 | 0.85% | 0.874596 |
| Đường SMA | 0.008770 | 0.011805 | 0.81% | 0.878525 |

##   
4.4 Chức năng thực hiện backtest

**CHIẾN LƯỢC 1:**

Chiến lược giao dịch LSTM với Attention (LSTMAttentionStrategy) được triển khai và kiểm tra trên cặp tiền tệ EUR/USD theo khung thời gian giờ (H1) từ ngày 13/05/2024 đến ngày 13/05/2025, với tổng thời gian backtest là 365 ngày. Chiến lược này tận dụng mô hình Long Short-Term Memory (LSTM) kết hợp cơ chế Attention để dự đoán xu hướng giá trong 6 giờ tiếp theo, từ đó đưa ra các quyết định giao dịch dựa trên ngưỡng thay đổi giá 0.1%. Backtest được thực hiện với vốn ban đầu 1,000,000 USD, áp dụng mức phí giao dịch 0.00007 mỗi lệnh và cơ chế quản lý rủi ro với chốt lời (Take Profit - TP) và cắt lỗ (Stop Loss - SL) cụ thể. Dưới đây là phân tích chi tiết dựa trên hai hình ảnh mô tả kết quả backtest.

Hình 4. 6 Chiến lược backtest 1

Chiến lược LSTMAttentionStrategy không đạt được hiệu suất mong đợi, ghi nhận mức lợi nhuận âm -6.17077%, tương ứng với lợi nhuận hàng năm -5.01441%. So sánh với chiến lược mua và giữ (Buy & Hold) đạt lợi nhuận 3.11998%, chiến lược này cho thấy hiệu quả kém hơn đáng kể. Vốn cuối kỳ chỉ còn 938,292.25 USD, giảm từ mức ban đầu 1,000,000 USD, dù có thời điểm vốn đạt đỉnh 1,024,548.01 USD, phản ánh sự tăng trưởng ngắn hạn trước khi sụt giảm mạnh. Tổng chi phí giao dịch lên đến 7,316.50 USD, chiếm tỷ lệ đáng kể trong vốn, chủ yếu do tần suất giao dịch cao với 53 lệnh trong suốt kỳ backtest.

Về mặt rủi ro, chiến lược ghi nhận mức biến động hàng năm 6.91881%, cho thấy biến động giá ở mức trung bình. Tuy nhiên, các chỉ số hiệu suất chính đều không khả quan: Sharpe Ratio âm -0.72475, Sortino Ratio -0.8784 và Calmar Ratio -0.43624, minh chứng rằng lợi nhuận không đủ bù đắp rủi ro. Mức sụt giảm vốn tối đa (Max Drawdown) đạt -11.49459%, kéo dài 284 ngày, cùng mức sụt giảm trung bình -1.07714% trong 22 ngày, cho thấy chiến lược phải đối mặt với những giai đoạn thua lỗ kéo dài và đáng kể. Hiệu suất giao dịch cho thấy tỷ lệ thắng ấn tượng 79.24528%, với giao dịch tốt nhất đạt lợi nhuận 2.30014% và giao dịch tệ nhất lỗ -2.53159%. Tuy nhiên, lợi nhuận trung bình mỗi giao dịch chỉ là -0.106004%, dẫn đến Profit Factor thấp 0.76463, nghĩa là tổng lợi nhuận không đủ bù đắp tổng thua lỗ. Thời gian nắm giữ trung bình mỗi giao dịch là 6 ngày, với giao dịch dài nhất kéo dài 41 ngày, cho thấy chiến lược thiên về giao dịch trung hạn nhưng không hiệu quả trong việc tối ưu hóa lợi nhuận.

Hình 4. 7 Kết quả chiến lược backtest 1

Hình trên cung cấp cái nhìn trực quan về hiệu suất chiến lược qua biểu đồ giá, vốn (Equity Curve), và phân bố giao dịch. Biểu đồ giá OHLC cho thấy giá EUR/USD biến động mạnh trong giai đoạn backtest. Từ tháng 05/2024 đến tháng 09/2024, giá dao động trong biên độ hẹp quanh 1.08-1.10, với một số giao dịch nhỏ lẻ được thực hiện nhưng lợi nhuận không đáng kể, phản ánh tính thận trọng của mô hình trong thị trường đi ngang. Từ tháng 10/2024, giá giảm mạnh xuống 1.04, và chiến lược đã kích hoạt nhiều lệnh bán chính xác, tận dụng xu hướng giảm. Đỉnh cao hiệu suất diễn ra từ tháng 01/2025 đến tháng 03/2025, khi giá tăng vọt từ 1.04 lên 1.16; chiến lược thực hiện loạt lệnh mua thành công, đóng góp lớn vào vốn tạm thời. Tuy nhiên, sau đó vốn giảm về mức 94% (938,292 USD) vào cuối kỳ, cho thấy chiến lược không duy trì được đà tăng.

Biểu đồ vốn (Equity Curve) thể hiện xu hướng tăng trưởng ban đầu, đạt đỉnh 102% (khoảng 1,024,548 USD) vào tháng 03/2025, trước khi giảm dần. Mức sụt giảm vốn tối đa 11.5% trong 284 ngày chủ yếu xảy ra từ tháng 11/2024 đến tháng 01/2025, trùng với giai đoạn biến động mạnh. Biểu đồ lợi nhuận/thua lỗ cho thấy các giao dịch phân bố khá cân bằng giữa lãi và lỗ, nhưng các lệnh lỗ tập trung ở giai đoạn thị trường đi ngang hoặc biến động bất ngờ (như tháng 10/2024), làm giảm hiệu quả tổng thể.

**CHIẾN LƯỢC 2:**

Chiến lược giao dịch LSTM với Attention (Strategy 2) được cải tiến và kiểm tra trên cặp tiền tệ EURUSD theo khung thời gian giờ (H1) từ ngày 01/01/2024 đến ngày 31/12/2024, với tổng thời gian backtest là 360 ngày. Chiến lược này sử dụng mô hình Long Short-Term Memory (LSTM) kết hợp cơ chế Attention để dự đoán xu hướng giá trong 6 giờ tiếp theo, đồng thời tích hợp chỉ báo Average True Range (ATR) để thiết lập mức chốt lời (Take Profit - TP) và cắt lỗ (Stop Loss - SL) linh hoạt, nhằm tối ưu hóa quản lý rủi ro. Backtest được thực hiện với vốn ban đầu 1,000,000 USD, áp dụng mức phí giao dịch 0.00001 mỗi lệnh. Dưới đây là phân tích chi tiết dựa trên hai hình ảnh mô tả kết quả backtest

Hình 4. 8 Chiến lược backtest 2

Kết quả backtest từ Hình 2 cho thấy Strategy 2 đạt hiệu suất tích cực với mức lợi nhuận tổng cộng 4.64%, tương ứng vốn cuối kỳ là 1,046,350.37 USD từ mức ban đầu 1,000,000 USD. Lợi nhuận hàng năm hóa đạt 3.74%, vượt trội so với chiến lược mua và giữ (Buy & Hold) với mức lỗ -5.07%. Độ biến động hàng năm là 3.92%, kết hợp với các chỉ số hiệu suất tích cực: Sharpe Ratio 0.95, Sortino Ratio 1.41 và Calmar Ratio 1.06. Những con số này phản ánh khả năng tạo ra lợi nhuận tốt so với rủi ro phải chịu. Mức sụt giảm vốn tối đa (Max Drawdown) được ghi nhận ở mức -3.53%, kéo dài trong 109 ngày, với mức sụt giảm trung bình là -0.36% trong 6 ngày. Điều này cho thấy chiến lược có khả năng quản lý rủi ro hiệu quả và phục hồi nhanh sau các giai đoạn thua lỗ.

Về hoạt động giao dịch, chiến lược thực hiện tổng cộng 407 lệnh, với tỷ lệ thắng là 42.01%. Mặc dù tỷ lệ thắng không cao, lợi nhuận trung bình mỗi giao dịch đạt 0.01%, và Profit Factor là 1.17, cho thấy tổng lợi nhuận vượt tổng thua lỗ một cách nhẹ nhàng. Giao dịch tốt nhất mang lại lợi nhuận 1.02%, trong khi giao dịch tệ nhất lỗ -0.30%, phản ánh sự kiểm soát rủi ro tốt nhờ cơ chế SL và TP dựa trên ATR. Thời gian nắm giữ trung bình mỗi giao dịch gần như bằng 0 ngày, cho thấy chiến lược tập trung vào giao dịch ngắn hạn, tận dụng các biến động nhỏ trong ngày. Mức độ tiếp xúc thị trường là 70.73%, thể hiện sự chọn lọc trong việc vào lệnh, giúp giảm thiểu rủi ro khi thị trường không thuận lợi.

Hình 4. 9 Kết quả chiến lược backtest 2

Hình trên cung cấp cái nhìn trực quan về hiệu suất của Strategy 2 thông qua các biểu đồ giá, vốn (Equity Curve) và phân bố giao dịch. Biểu đồ giá OHLC cho thấy giá EURUSD dao động từ 1.02 đến 1.16 trong suốt năm 2024, với các giai đoạn xu hướng rõ rệt: tăng dần đến giữa năm, giảm mạnh sau đó, và phục hồi vào cuối năm. Các lệnh mua (mũi tên xanh) và bán (mũi tên đỏ) được phân bố đều trong suốt kỳ backtest, với mức SL và TP được thiết lập dựa trên ATR và dự đoán giá từ mô hình LSTM. Chiến lược tận dụng được các cơ hội trong cả xu hướng tăng và giảm, đặc biệt là giai đoạn tăng giá vào cuối năm 2024.

Biểu đồ vốn (Equity Curve) thể hiện xu hướng tăng trưởng ổn định từ 100% lên 105%, đạt đỉnh tại 108%, với mức sụt giảm vốn tối đa là 3.5% trong 110 ngày. Điều này cho thấy chiến lược duy trì được lợi nhuận và kiểm soát rủi ro hiệu quả, ngay cả trong các giai đoạn thị trường biến động mạnh (như giữa năm 2024). Biểu đồ phân bố lợi nhuận/thua lỗ cho thấy các giao dịch phân bố cân bằng giữa lãi (điểm xanh) và lỗ (điểm đỏ), không xuất hiện các khoản lỗ lớn bất thường. Điều này minh chứng cho hiệu quả của cơ chế quản lý rủi ro dựa trên ATR, giúp hạn chế tổn thất trong các điều kiện thị trường bất lợi.

# CHƯƠNG 5. ỨNG DỤNG AI AGENT

Hình 5. 1 Giao diện API Forex và Huấn luyện nâng cao

API dành cho quản lý dữ liệu Forex và huấn luyện mô hình nâng cao (Advanced Training). Phần Forex Data cung cấp chức năng crawl toàn bộ 75 cặp tiền, theo dõi trạng thái hệ thống, truy xuất thông tin chi tiết từng cặp và cập nhật thông minh những dữ liệu còn thiếu. Phần Advanced Training hỗ trợ quy trình huấn luyện AI nâng cao, bao gồm khởi tạo huấn luyện, dự đoán, kiểm tra trạng thái, liệt kê mô hình đã huấn luyện, xác thực dữ liệu, tra cứu siêu tham số, ví dụ mô hình, cũng như backtest nâng cao và truy xuất kết quả. Hệ thống API này tạo nên nền tảng tự động hóa toàn diện từ thu thập dữ liệu tới đánh giá hiệu suất mô hình giao dịch.

Hình 5. 2 Giao diện API huấn luyện và backtest mô hình LSTM trên dữ liệu Forex.

API “Training & Backtest” - nơi toàn bộ quy trình phát triển mô hình LSTM cho dữ liệu Forex được tự động hóa một cách khép kín. Đầu tiên, hệ thống hỗ trợ khởi tạo và huấn luyện mô hình với hai điểm cuối chính: /api/train dành cho LSTM tổng quát và /api/train-forex cho mô hình chuyên biệt trên thị trường ngoại hối. Trước khi huấn luyện, người dùng có thể lấy dữ liệu thô từ Google Sheet qua /api/fetch-data, xử lý chúng tại /api/process-data, hoặc truy xuất dữ liệu Forex theo cặp bằng /api/forex-data. Sau giai đoạn huấn luyện, mô hình được kiểm thử hồi cứu (backtest) trên dữ liệu lịch sử thông qua /api/backtest-forex, từ đó tạo ra các biểu đồ đánh giá hiệu suất. Các kết quả này có thể được truy xuất nhanh chóng bằng /api/backtest-chart (lấy biểu đồ mới nhất), /api/backtest-charts (liệt kê toàn bộ biểu đồ) hoặc tải cụ thể theo tên với /api/backtest-chart/{filename}. Nhờ chuỗi API này, người dùng dễ dàng thực hiện “thu thập dữ liệu – huấn luyện – backtest – phân tích” trong một vòng lặp liên tục nhằm nhanh chóng đánh giá và tối ưu hoá chiến lược giao dịch.

Hình 5. 3 Giao diện API quản lý dữ liệu Forex.

API “Forex Data Management”, đóng vai trò trung tâm trong việc thu thập, cập nhật và giám sát dữ liệu thị trường ngoại hối. Người dùng trước hết có thể truy vấn danh sách các cặp tiền đang được hỗ trợ thông qua /api/forex/pairs, sau đó chủ động cập nhật dữ liệu cho một cặp cụ thể bằng /api/forex/update/{pair} hoặc cập nhật đồng loạt cho toàn bộ danh mục với /api/forex/update-all. Để kiểm soát chất lượng dữ liệu, hai đầu cuối /api/forex/metadata và /api/forex/metadata/{pair} cung cấp toàn bộ thông tin siêu dữ liệu, cho phép đánh giá độ phủ và tính toàn vẹn của mỗi cặp tiền. Ngoài ra, điểm cuối /api/forex/crawl-all hỗ trợ quét (crawl) lại lịch sử giá cho tất cả cặp tiền nhằm lấp đầy các khoảng trống còn thiếu, còn /api/forex/status cung cấp trạng thái hệ thống crawl để người vận hành theo dõi tiến trình. Nhờ bộ API này, quá trình quản trị dữ liệu Forex được tự động hóa, bảo đảm nguồn dữ liệu sạch và kịp thời cho các mô hình học máy phía sau.

Hình 5. 4 Giao diện các API phục vụ dự đoán giá Forex.

Nhóm API “Prediction”, chịu trách nhiệm phục vụ giai đoạn suy luận (inference) của hệ thống giao dịch. Điểm cuối /api/predict (phương thức GET) cho phép người dùng nhanh chóng truy xuất mức giá dự báo do mô hình hiện hành sinh ra, trong khi /api/predict-custom (phương thức POST) hỗ trợ nhập bộ dữ liệu tuỳ chỉnh để nhận về kết quả dự đoán tương ứng—đáp ứng nhu cầu thử nghiệm kịch bản hoặc cài đặt tham số riêng biệt. Cuối cùng, /api/model-info cung cấp thông tin cấu hình của mô hình (phiên bản, siêu tham số, thời gian huấn luyện), giúp người vận hành theo dõi độ cập nhật và tính phù hợp của mô hình trước khi sử dụng vào giao dịch thực tế. Bộ API này khép lại chu trình “huấn luyện → backtest → triển khai” bằng cách biến các mô hình đã huấn luyện thành dịch vụ dự đoán thời gian thực.

**POST /train - Training AI Models cho Forex**

API /train được thiết kế để huấn luyện các mô hình trí tuệ nhân tạo (AI) với mục tiêu dự đoán giá của các cặp tiền tệ trong thị trường Forex. Khi sử dụng API này, người dùng có thể tùy chỉnh nhiều tham số để điều chỉnh quá trình huấn luyện, từ loại mô hình, phạm vi thời gian của dữ liệu, đến các siêu tham số của mô hình.

**Request Body:**

Hình 5. 5 Cấu hình yêu cầu API cho dự báo LSTM - EUR/USD

currency_pair: Đây là cặp tiền tệ mà mô hình sẽ dự đoán, ví dụ: EURUSD.

model_type: Loại mô hình AI cần sử dụng, bao gồm các lựa chọn như: rnn, prophet, arima, lightgbm, xgboost, nbeats, transformer.

start_date và end_date: Các tham số tùy chọn để chỉ định phạm vi thời gian của dữ liệu huấn luyện.

training_days: Số ngày dữ liệu sử dụng cho quá trình huấn luyện khi không chỉ định rõ ngày bắt đầu và kết thúc.

prediction_steps: Số bước (hoặc số giờ) mà mô hình dự đoán trong tương lai.

use_covariates: Quyết định việc sử dụng dữ liệu OHLV (Open, High, Low, Volume) làm các yếu tố bổ sung trong quá trình huấn luyện.

hyperparameter_tuning: Nếu được bật, hệ thống sẽ sử dụng Optuna để tối ưu hóa các tham số siêu (hyperparameters) của mô hình.

n_trials: Số lần thử nghiệm trong quá trình tối ưu hóa tham số siêu nếu hyperparameter_tuning được kích hoạt.

validation_split: Tỷ lệ dữ liệu được phân bổ cho việc kiểm tra mô hình (ví dụ, 0.2 đại diện cho 20% dữ liệu dành cho validation).

custom_params: Các tham số tùy chỉnh của mô hình, bao gồm các tham số như loại mạng RNN (LSTM, GRU), số lượng bước thời gian đầu vào, kích thước lớp ẩn, số lớp RNN, dropout, batch size, số lượng epochs, và độ dài chuỗi huấn luyện.

**Response:**

Hình 5. 6 Phản hồi API - Quá trình huấn luyện mô hình RNN cho EUR/USD

job_id: ID của công việc huấn luyện, dùng để theo dõi tiến độ.

status: Trạng thái hiện tại của công việc (chẳng hạn: "started", "running", "completed", "failed").

message: Thông báo chi tiết về trạng thái của công việc.

estimated_duration: Thời gian ước tính để hoàn thành công việc huấn luyện.

API này hỗ trợ quá trình huấn luyện bất đồng bộ trên nền tảng job system, cho phép người dùng theo dõi tiến độ huấn luyện mô hình trong thời gian thực. Bên cạnh đó, hệ thống còn hỗ trợ tối ưu hóa tham số tự động thông qua Optuna, nhằm nâng cao hiệu quả dự đoán của mô hình. Thời gian huấn luyện có thể dao động từ 3 phút đối với mô hình Prophet, đến 25 phút đối với mô hình Transformer, tùy thuộc vào loại mô hình và dữ liệu được sử dụng.

**GET /models - Danh Sách Trained Models**

API /models cung cấp một danh sách tất cả các mô hình đã được huấn luyện thành công trong hệ thống. API này cho phép người dùng dễ dàng tra cứu thông tin về các mô hình, bao gồm loại mô hình, cặp tiền tệ, hiệu suất, và trạng thái hiện tại của từng mô hình.

**Response:**

Hình 5. 7 Phản hồi API - Các chỉ số hiệu suất mô hình cho EUR/USD

total_models: Tổng số mô hình đã huấn luyện thành công.

models: Mảng chứa các mô hình đã huấn luyện, bao gồm:

run_id: ID của mô hình trong MLflow.

model_name: Tên của mô hình.

model_type: Loại mô hình (rnn, lightgbm, v.v.).

currency_pair: Cặp tiền tệ mà mô hình đã được huấn luyện.

performance: Các chỉ số đánh giá chất lượng mô hình như MAPE, RMSE, và MAE.

status: Trạng thái hiện tại của mô hình (sẵn sàng sử dụng, đang huấn luyện, v.v.).

model_size_mb: Kích thước của mô hình.

last_used: Thời gian mô hình được sử dụng lần cuối.

API này giúp người dùng dễ dàng theo dõi các mô hình đã huấn luyện, phân loại theo loại mô hình hoặc cặp tiền tệ, và xác định mô hình hiệu quả nhất cho từng cặp tiền.

**POST /api/forex/crawl-all - Crawl tất cả 75+ cặp tiền tệ**

**Mục đích:**

API này thu thập dữ liệu cho tất cả 75+ cặp tiền tệ từ Yahoo Finance, hỗ trợ thu thập dữ liệu hàng loạt cho nhiều cặp tiền tệ trong một lần gọi API.

**Request Body:**

Hình 5. 8 Nội dung yêu cầu API - Cơ chế lọc theo khoảng thời gian

start_date, end_date: Khoảng thời gian thu thập (tùy chọn).

timeframe: Khung thời gian (tùy chọn).

pairs: Danh sách các cặp tiền tệ cần thu thập (tùy chọn).

**Response:**

Hình 5. 9 Yêu cầu POST API - Khởi tạo thu thập dữ liệu ngoại hối

crawl_job_id: ID của công việc crawl.

summary: Tổng quan về kết quả thu thập, bao gồm số lượng cặp thành công và thất bại.

successful_pairs: Các cặp tiền tệ thu thập thành công.

failed_pairs: Các cặp tiền tệ không thu thập được và lý do thất bại.

**Chức năng chi tiết:**

Thu thập dữ liệu cho tất cả 75+ cặp tiền tệ từ Yahoo Finance.

Xử lý dữ liệu, làm sạch và xác thực các giá trị.

Cung cấp báo cáo chi tiết về quá trình thu thập.

**Lợi ích:**

Thu thập dữ liệu cho nhiều cặp tiền tệ trong một lần.

Tự động làm sạch và chuẩn bị dữ liệu cho các mô hình AI.

Cung cấp báo cáo chi tiết về tiến độ thu thập.

**POST /api/forex/update - Cập nhật và làm sạch dữ liệu thiếu**

**Mục đích:**

API này giúp cập nhật dữ liệu mới cho các cặp tiền tệ đã có trong hệ thống, chỉ lấy dữ liệu mới thiếu và tự động làm sạch dữ liệu.

**Request Body:**

Hình 5. 10 Yêu cầu POST API - Kích hoạt cập nhật dữ liệu ngoại hối

pairs: Các cặp tiền tệ cần cập nhật (tùy chọn).

enable_cleaning: Kích hoạt việc làm sạch dữ liệu (tùy chọn).

force_update: Cập nhật cưỡng chế nếu cần (tùy chọn).

**Response:**

update_job_id: ID công việc cập nhật.

summary: Tổng quan về kết quả cập nhật, bao gồm số lượng cặp được cập nhật và không cần cập nhật.

**Chức năng chi tiết:**

Phát hiện và tải dữ liệu mới cho các cặp tiền tệ đã có trong hệ thống.

Làm sạch dữ liệu và xử lý các giá trị bất thường.

Đánh giá chất lượng dữ liệu và cung cấp khuyến nghị.

**Lợi ích:**

Cập nhật dữ liệu mới mà không phải tải lại toàn bộ.

Giảm thiểu việc tiêu tốn tài nguyên và băng thông.

Cải thiện chất lượng dữ liệu qua việc làm sạch tự động.

**GET /api/forex/status - Tổng quan hệ thống 75+ cặp tiền tệ**

**Mục đích:**

API này cung cấp thông tin tổng quan về tình trạng dữ liệu của tất cả 75+ cặp tiền tệ trong hệ thống, bao gồm chất lượng dữ liệu, phạm vi thời gian và thông tin về lưu trữ.

**Response:**

Hình 5. 11 Kết quả API Trạng thái Ngoại hối

system_overview: Tổng quan về trạng thái hệ thống (tổng số cặp tiền tệ, số lượng cặp có dữ liệu, chất lượng dữ liệu).

pair_status: Trạng thái chất lượng của từng cặp tiền tệ.

storage_info: Thông tin về không gian lưu trữ.

**Chức năng chi tiết:**

Kiểm tra tình trạng của tất cả cặp tiền tệ.

Cung cấp thông tin chi tiết về chất lượng dữ liệu và phạm vi thời gian.

Đưa ra các khuyến nghị về cập nhật và làm sạch dữ liệu.

**Lợi ích:**

Cung cấp cái nhìn tổng quan về chất lượng và tình trạng của toàn bộ hệ thống.

Giúp người dùng xác định những cặp tiền tệ cần cập nhật hoặc làm sạch.

Hỗ trợ quản lý không gian lưu trữ và tối ưu hóa tài nguyên.

**GET /api/forex/pair/{pair} - Thông tin chi tiết 1 cặp tiền tệ**

**Mục đích:**

API này cung cấp thông tin chi tiết về dữ liệu của một cặp tiền tệ cụ thể. Giúp người dùng hiểu rõ hơn về tình trạng và chất lượng dữ liệu của từng cặp tiền tệ.

**Path Parameters:**

Hình 5. 12 Giao diện nhập tham số

pair: Mã của cặp tiền tệ cần kiểm tra.

**Response:**

Hình 5. 13 Chi tiết phản hồi API cho EURUSD

symbol: Cặp tiền tệ.

data_info: Các thông tin chi tiết về dữ liệu (số bản ghi theo từng khung thời gian, phạm vi ngày).

quality_assessment: Đánh giá chất lượng dữ liệu.

statistics: Các thống kê liên quan đến cặp tiền tệ (biến động, xu hướng).

preview_data: Xem trước dữ liệu của các hàng đầu và cuối trong bộ dữ liệu.

**Chức năng chi tiết:**

Kiểm tra tính khả dụng và chất lượng dữ liệu cho cặp tiền tệ cụ thể.

Phân tích các chỉ số thống kê như giá thấp nhất, cao nhất, và biến động.

Cung cấp thông tin về các bản ghi đầu tiên và cuối cùng của dữ liệu.

**Lợi ích:**

Giúp người dùng phân tích sâu sắc về một cặp tiền tệ cụ thể.

Đánh giá chất lượng và tính khả dụng của dữ liệu.

Cung cấp dữ liệu chi tiết cho các quyết định phân tích tài chính.

Hình 5. Tổng quan quy trình

Hình “Tổng quan quy trình” minh hoạ chuỗi tác vụ tự động hoá toàn phần – từ thu thập dữ liệu đến phân tích kết quả – trong hệ thống giao dịch Forex sử dụng mô hình học sâu. Quy trình bắt đầu khi người dùng bấm “Execute workflow”, kích hoạt lệnh GET tới API /forex/pairs nhằm kiểm tra các cặp tiền đang hỗ trợ. Tiếp theo, node check pairs chuyển tiếp sang bước update pair, nơi lệnh POST tới /forex/update/{pair} đảm bảo dữ liệu giá của cặp được làm mới. Khi dữ liệu đã sẵn sàng, node train model gửi yêu cầu POST tới /train để huấn luyện mô hình LSTM trên dữ liệu cập nhật; mô hình sau đó được kiểm thử hồi cứu thông qua node backtest (POST /backtest-forex). Kết quả backtest được trực quan hóa ở node chart bằng GET /backtest-chart, tạo nên biểu đồ hiệu suất. Cuối cùng, node analyse backtest chuyển bộ chỉ số và biểu đồ sang một OpenAI Chat Model, nơi mô hình ngôn ngữ tự động diễn giải, đánh giá rủi ro và đề xuất điều chỉnh chiến lược. Chuỗi node tuần tự này hình thành một pipeline khép kín “kiểm tra – cập nhật – huấn luyện – backtest – trực quan hóa – phân tích”, cho phép nhà giao dịch nhanh chóng lặp lại và tối ưu hoá mô hình trong môi trường thời gian thực.

Hình 5. Thống kê thông tin của các cặp tiền hiện có

Đây là báo cáo tóm tắt tình trạng thu thập dữ liệu Forex: tổng cộng hệ thống đã crawl thành công 71 cặp tiền tệ, con số này được ghi rõ tại trường crawled_pairs_count. Mỗi cặp – chẳng hạn cadsgd, nzdjpy hay eurjpy – đều chia sẻ cùng cấu trúc thông tin, bao gồm ngày bắt đầu và kết thúc dữ liệu (từ 1/1/2015 đến 23/6/2025), tổng số nến 1 giờ thu được (khoảng 91 825 bản ghi), trạng thái xử lý (status: success khẳng định dữ liệu đầy đủ), thời điểm cập nhật lần cuối, đường dẫn tệp CSV lưu trữ trên máy chủ và khung thời gian sử dụng (timeframe: "1H"). Khi cần kiểm tra nhanh, người vận hành chỉ cần chú ý bốn trường trọng yếu của từng cặp: tên cặp (pair), trạng thái (status), thời điểm cập nhật (last_update) và đường dẫn tệp (data_path). Nếu trường status báo “success” và thời điểm cập nhật còn mới, dữ liệu của cặp đó đã sẵn sàng để nạp vào quy trình huấn luyện mô hình.

Hình 5. Update data cho tới thời điểm bấm execute

Trong khoảng 10 năm rưỡi (từ 01/01/2015), hệ thống lưu được 91 825 nến, tương ứng với tỷ lệ phủ gần 99,8 %, gần như không thiếu bản ghi. Với độ dài và độ mới như vậy, bộ dữ liệu hoàn toàn đáp ứng yêu cầu chia train–validation–test, hỗ trợ huấn luyện hoặc backtest mô hình LSTM ngay lập tức mà không tốn thêm thời gian crawl. Tính liên tục của chuỗi giá giảm thiểu nhu cầu nội suy, nhờ đó hạn chế nhiễu trong mô hình. Trước khi huấn luyện, nên thực hiện thêm kiểm tra chất lượng, bao gồm thống kê phân phối lợi suất, nhận diện nến bất thường và so khớp số lượng nến lý thuyết, đồng thời tạo mã băm (hash) cho file để theo dõi phiên bản dữ liệu trong các lần huấn luyện sau.

Hình 5. Thực hiện training model LSTM, đưa ra giá dự đoán của ngày tiếp theo

Kết quả huấn luyện mô hình LSTM cho thấy độ khớp rất cao giữa dự báo và dữ liệu thực. Cụ thể, trên tập train, sai số gốc bình phương (RMSE) chỉ còn 0,00783, sai số tuyệt đối trung bình (MAE) 0,00623, trong khi hệ số xác định đạt 0,9988. Khi kiểm tra trên tập test độc lập, các chỉ số gần như không suy giảm (RMSE = 0,00790; MAE = 0,00629; = 0,9985), chứng tỏ mô hình hầu như không bị overfitting và giữ được khả năng khái quát hóa tốt. Dựa trên cấu hình hiện tại, mô hình ước tính giá đóng cửa cho phiên kế tiếp (ngày 01/01/2025) là **2,8487**. Nhìn chung, sai số tuyệt đối dưới 0,01 và xấp xỉ 1,0 cho thấy LSTM đang tái hiện gần trọn vẹn cấu trúc chuỗi thời gian; tuy nhiên, vẫn nên liên tục cập nhật dữ liệu và tái huấn luyện định kỳ để đảm bảo mô hình theo kịp biến động thị trường thực.

Hình 5. Kết quả kiểm thử hồi cứu (backtest) mô hình LSTM

Kết quả backtest mô hình LSTM trên cặp EUR/USD (01–31/12/2024) cho thấy chiến lược đạt mức lợi nhuận tuyệt đối 0,58 % sau khi trừ phí (Return = 0,0058), vượt xa phương án “mua-giữ” trong cùng giai đoạn (Buy-&-Hold Return = -2,58 %). Sharpe Ratio = 3,70 và Sortino Ratio = 6,03 phản ánh tỷ lệ lợi nhuận so với rủi ro rất hấp dẫn; đồng thời Calmar Ratio = 18,10 càng củng cố tính hiệu quả khi so sánh lợi nhuận với mức sụt giảm vốn tối đa. Mức biến động thường niên hoá (Volatility) chỉ 1,82 % và Max Drawdown –0,0038 % cho thấy rủi ro được kiểm soát chặt chẽ, nhất quán với tỷ lệ thắng 76,9 % trên 26 giao dịch. Thời lượng giữ lệnh trung bình 0 ngày 18 giờ gợi ý chiến lược thuộc nhóm ngắn hạn, phù hợp với mục tiêu luân chuyển vốn nhanh. Hệ số Kelly 0,54 và Profit Factor 3,35 hàm ý tiếp tục dùng quy mô vị thế hiện tại là hợp lý; tuy nhiên, mức Alpha dương nhưng nhỏ (0,0012) cho thấy lợi thế tuyệt đối vẫn còn khiêm tốn và có thể được cải thiện bằng cách tinh chỉnh ngưỡng vào lệnh hoặc kết hợp thêm tín hiệu bổ trợ.

Hình 5. Biểu đồ kết quả backtest mô hình LSTM trên cặp EUR/USD

Biểu đồ backtest LSTM trên cặp EUR/USD trình bày ba lớp thông tin chính. Khung trên cùng mô tả đường giá đóng cửa; trong tháng 12/2024, giá giảm nhẹ đầu kỳ rồi đảo chiều tăng mạnh về cuối tháng. Khung giữa thể hiện đường vốn danh mục: sau giai đoạn dao động quanh mức âm ( – 20 $) từ 12–20/12, mô hình bắt nhịp xu hướng đi lên, giúp vốn tăng liên tục và kết thúc quanh +60 $. Khung dưới hiển thị lại đường giá nhưng kèm ký hiệu tam giác xanh (điểm mua) và đỏ (điểm bán), cho thấy chiến lược giao dịch ngắn hạn, thoát–vào lệnh khá dày đặc giai đoạn giữa tháng rồi giữ đà mua khi giá bứt phá cuối kỳ. Ô cửa thống kê góc trái tóm lược: lợi nhuận 0,01 %, Sharpe 3,70, 26 giao dịch, win-rate 76,9 %, mức drawdown gần bằng 0 %. Nhìn tổng thể, biểu đồ chứng minh mô hình LSTM nhanh chóng đảo vị thế theo biến động, giảm rủi ro drawdown và tận dụng tốt sóng tăng cuối tháng để tối đa hóa lợi nhuận.

Hình 5. Đánh giá chiến lược LSTM attention

Kết quả back-test cho chiến lược LSTM kết hợp attention trên cặp EUR/USD (khung 240 giờ, giai đoạn 1–31 tháng 12 năm 2024) cho thấy mô hình khớp thị trường ở mức vừa phải: trong 30 ngày, chiến lược chỉ kích hoạt 11 lệnh, đạt tỷ lệ thắng 54,55 % nhỉnh hơn mức ngẫu nhiên nhưng chưa thật sự ấn tượng. Tổng lợi nhuận sau phí là 0,71 %, thấp hơn nhiều so với kịch bản buy-and-hold cùng kỳ (2,58 %), chứng tỏ chiến lược giao dịch ngắn hạn vẫn chưa khai thác được hết biên độ biến động. Tốc độ tăng trưởng kép hằng năm (CAGR) quy đổi đạt 8,68 % mức tích cực, song vẫn dưới kỳ vọng của các chiến lược lượng hóa hiệu quả hơn. Bù lại, mức rủi ro được kiểm soát rất chặt: Sharpe Ratio đạt 4,27, Drawdown tối đa chỉ -0,36 % và bình quân -0,05 %, phản ánh lợi nhuận/tổn thất trên mỗi đơn vị rủi ro khá cao. Nói cách khác, mô hình cho hiệu suất an toàn nhưng thiếu sắc bén về lợi nhuận; để cải thiện, cần tăng tần suất giao dịch có chọn lọc (hoặc kết hợp tín hiệu bổ sung) nhằm tận dụng tốt hơn các đợt biến động mà vẫn giữ drawdown ở mức tối thiểu.

Hình 5. Khuyến nghị điều chỉnh thông số mô hình LSTM

Để nâng cao độ chính xác và kiểm soát rủi ro của mô hình LSTM, chúng tôi đề xuất điều chỉnh một số tham số cốt lõi như sau. Trước hết, nên thử nghiệm các khoảng thời gian nhìn lại khác nhau thay vì cố định ở 240 giờ; cụ thể, rút ngắn xuống 120 giờ hoặc kéo dài đến 480 giờ có thể giúp mô hình nắm bắt tốt hơn những biến động ngắn hạn hoặc xu hướng dài hạn tùy từng giai đoạn thị trường. Bên cạnh đó, ngưỡng tín hiệu vào lệnh đang đặt ở 0,001 cũng nên được tối ưu hóa; việc hạ xuống 0,0005 hoặc nâng lên 0,002 sẽ cho phép đánh giá mức độ nhạy của chiến lược và tìm ra ngưỡng mang lại tỷ lệ thắng tối ưu. Về quản trị rủi ro, giảm đòn bẩy xuống dưới mức 500 sẽ hạn chế tổn thất tiềm ẩn khi thị trường biến động mạnh, đồng thời cải thiện tâm lý giao dịch. Cuối cùng, thu nhỏ kích thước vị thế cho mỗi lệnh không chỉ giúp phân tán rủi ro mà còn tăng tính linh hoạt trong việc điều chỉnh chiến lược khi điều kiện thị trường thay đổi.

Hình 5. Lý do khuyến nghị và kết luận backtest LSTM

Các khuyến nghị điều chỉnh trên xuất phát từ ba mục tiêu cốt lõi. Thứ nhất, tối ưu hóa thuật toán: việc tinh chỉnh các thông số (lookback, ngưỡng tín hiệu…) giúp mô hình phát hiện mạnh mẽ hơn những mối quan hệ ẩn giữa các biến tác động tới giá, từ đó cải thiện độ chính xác dự báo. Thứ hai, giảm rủi ro: hạ đòn bẩy và thu nhỏ quy mô vị thế bảo vệ vốn hiệu quả hơn trong giai đoạn bất lợi, đồng thời ổn định tâm lý giao dịch. Thứ ba, nâng tỷ lệ thắng: rà soát và hiệu chỉnh ngưỡng vào lệnh có thể dẫn tới các quyết định giao dịch chọn lọc hơn, qua đó tăng xác suất thành công.

Tóm lại, dù backtest hiện tại đã cho thấy một số chỉ báo tích cực, việc tiếp tục điều chỉnh tham số và chiến lược vẫn cần thiết để vừa gia tăng hiệu suất sinh lợi, vừa hạn chế rủi ro. Những cải tiến này sẽ làm cho chiến thuật LSTM hoạt động bền vững hơn khi điều kiện thị trường thay đổi.

# CHƯƠNG 6: KẾT LUẬN VÀ KHUYẾN NGHỊ

## 6.1. Kết luận

Qua quá trình nghiên cứu và thực nghiệm, nhóm đã hoàn thiện việc xây dựng hệ thống AI Agent có khả năng dự báo biến động tỷ giá Forex theo thời gian thực và tự động đưa ra khuyến nghị giao dịch. Hệ thống kết hợp đồng bộ nhiều thành phần then chốt: thu thập dữ liệu giá Forex trực tiếp từ các sàn uy tín, tích hợp các chỉ báo kỹ thuật như RSI, EMA200, Bollinger Bands, ATR, và áp dụng các mô hình học máy hiện đại bao gồm Random Forest, RNN, GRU, LSTM để tối ưu hóa dự báo. Ngoài ra, mô-đun backtest được phát triển để kiểm tra hiệu quả của chiến lược giao dịch trên dữ liệu lịch sử, đảm bảo tính tin cậy trước khi áp dụng trên dữ liệu thời gian thực.

Kết quả thử nghiệm cho thấy hệ thống hoạt động ổn định, có khả năng thích ứng với các trạng thái thị trường khác nhau, đồng thời cung cấp những tín hiệu giao dịch khách quan giúp giảm thiểu tác động của yếu tố cảm tính. Đề tài không chỉ góp phần thu hẹp khoảng cách công nghệ giữa nhà đầu tư cá nhân và các tổ chức lớn mà còn mở ra hướng đi mới trong việc ứng dụng AI và tự động hóa vào lĩnh vực tài chính — đặc biệt là tại Việt Nam, nơi các công cụ hỗ trợ giao dịch tiên tiến còn tương đối hạn chế.

## 6.2. Khuyến nghị

Dựa trên những kết quả đạt được, nhóm nghiên cứu mạnh dạn đề xuất một số hướng phát triển tiếp theo nhằm khai thác tối đa tiềm năng của hệ thống:

Thứ nhất, cần mở rộng hệ thống để tích hợp trực tiếp với API của các sàn giao dịch Forex, cho phép thực hiện các lệnh mua bán tự động dựa trên tín hiệu được AI đưa ra. Điều này sẽ biến hệ thống từ một công cụ phân tích và khuyến nghị thành một nền tảng giao dịch tự động thực sự, tận dụng hoàn toàn lợi thế của giao dịch thuật toán trong việc xử lý tốc độ cao và ra quyết định tức thì.

Thứ hai, nhóm khuyến nghị tiếp tục hoàn thiện và phát triển một chatbot giao dịch thông minh, có thể tương tác với người dùng qua giao diện trò chuyện tự nhiên. Chatbot này sẽ đóng vai trò như một “trợ lý tài chính cá nhân”, cung cấp thông tin phân tích thị trường, dự báo biến động giá, nhắc nhở về các tín hiệu quan trọng, đồng thời trả lời các câu hỏi liên quan đến xu hướng tỷ giá hay các chiến lược giao dịch. Việc triển khai chatbot trên nền tảng web hoặc ứng dụng di động không chỉ nâng cao tính tiện lợi mà còn mở rộng khả năng tiếp cận của hệ thống đến đông đảo nhà đầu tư, từ người mới cho tới các trader có kinh nghiệm.

Cuối cùng, để hệ thống đạt độ tin cậy cao hơn khi vận hành trên thị trường thực, cần tiếp tục nghiên cứu mở rộng tập dữ liệu huấn luyện, đa dạng hóa các chỉ báo kỹ thuật, đồng thời cải thiện mô hình quản trị rủi ro. Đây sẽ là bước quan trọng để đưa hệ thống từ giai đoạn thử nghiệm trở thành một sản phẩm hoàn thiện, có thể thương mại hóa và triển khai rộng rãi, góp phần nâng cao hiệu quả giao dịch cũng như thúc đẩy quá trình số hóa trong lĩnh vực đầu tư tài chính tại Việt Nam.
