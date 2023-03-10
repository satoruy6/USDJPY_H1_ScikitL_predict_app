import streamlit as st
st.set_page_config(
  page_title="predic_USDJPY_H1_ScikitLearn app",
  page_icon="ð",
)
st.title("USDJPY1æéè¶³äºæ¸¬(Scikit-Learn)ã¢ããª")
st.markdown('## æ¦è¦åã³æ³¨æäºé ')
st.write("å½ã¢ããªã§ã¯ãUSDJPYã®1æéè¶³ãç´è¿ã®ãã¼ã¿(yahoo finance)ã«åºã¥ãé½ç·ããé°ç·ããScikit-Learnãä½¿ç¨ãã¦äºæ¸¬ãã¾ãããã ãæ¬çµæã«ããæè³ã«ãããªãæå¤±ãçãã¦ããå½ã¢ããªã§ã¯è²¬ä»»ãåãã¾ãããããã¾ã§åèç¨åº¦ã«ãå©ç¨ãã ããã")
st.write('ãªãæå»ã¯UTC(æ¥æ¬æéãã¤ãã¹9æé)ã®è¡¨ç¤ºã¨ãªãã¾ãã')
try:
    if st.button('äºæ¸¬éå§'):
        comment = st.empty()
        comment.write('äºæ¸¬ãéå§ãã¾ãã')

        import time
        t1 = time.time()
        import numpy as np
        import csv
        from sklearn import svm
        import math
        import pandas as pd
        import yfinance as yf

        # å¤çºãã¼ã¿åå¾
        tks  = 'USDJPY=X'
        data = yf.download(tickers  = tks ,          # éè²¨ãã¢
                        period   = '1mo',          # ãã¼ã¿åå¾æé 15m,1d,1mo,3mo,1y,10y,20y,30y  1996å¹´10æ30æ¥ãããã¼ã¿ãããã
                        interval = '1h',         # ãã¼ã¿è¡¨ç¤ºéé 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
                        )

        #æå¾ã®æ¥æãåãåºãã
        lastdatetime = data.index[-1]

        #Closeä¾¡æ ¼ã®ã¿ãåãåºãã
        data_close = data['Close']

        #å¯¾æ°è¡¨ç¤ºã«å¤æãã
        ln_fx_price = []
        for line in data_close:
            ln_fx_price.append(math.log(line))
        count_s = len(ln_fx_price)

        # çºæ¿ã®ä¸æçãç®åºããããã-1.0-1.0ã®ç¯å²ã«åã¾ãããã«èª¿æ´
        modified_data = []
        for i in range(1, count_s):
            modified_data.append(float(ln_fx_price[i] - ln_fx_price[i-1])*1000)
        count_m = len(modified_data)

        # åæ¥ã¾ã§ã®4é£ç¶ã®ä¸æçã®ãã¼ã¿
        successive_data = []
        # æ­£è§£å¤ ä¾¡æ ¼ä¸æ: 1 ä¾¡æ ¼ä¸è½: 0
        answers = []
        for i in range(4, count_m):
            successive_data.append([modified_data[i-4], modified_data[i-3], modified_data[i-2], modified_data[i-1]])
            if modified_data[i] > 0:
                answers.append(1)
            else:
                answers.append(0)
        # print (successive_data)
        # print (answers)

        # ãã¼ã¿æ°
        n = len(successive_data)
        # print (n)
        m = len(answers)
        # print (m)

        # ç·å½¢ãµãã¼ããã¯ã¿ã¼ãã·ã¼ã³
        clf = svm.LinearSVC()
        # ãµãã¼ããã¯ã¿ã¼ãã·ã¼ã³ã«ããè¨ç·´ ï¼ãã¼ã¿ã®75%ãè¨ç·´ã«ä½¿ç¨ï¼
        clf.fit(successive_data[:int(n*750/1000)], answers[:int(n*750/1000)])

        # ãã¹ãç¨ãã¼ã¿
        # æ­£è§£
        expected = answers[int(-n*250/1000):]
        # äºæ¸¬
        predicted = clf.predict(successive_data[int(-n*250/1000):])

        st.write(f'{lastdatetime}ã®æ¬¡ã®1æéè¶³ã®äºæ¸¬')
        # æ«å°¾ã®10åãæ¯è¼
        #print ('æ­£è§£:' + str(expected[-10:]))
        #print ('äºæ¸¬:' + str(list(predicted[-10:])))

        # æ­£è§£çã®è¨ç®
        correct = 0.0
        wrong = 0.0
        for i in range(int(n*250/1000)):
            if expected[i] == predicted[i]:
                correct += 1
            else:
                wrong += 1
                
        #print('æ­£è§£æ°ï¼ ' + str(int(correct)))
        #print('ä¸æ­£è§£æ°ï¼ ' + str(int(wrong)))

        successive_data.append([modified_data[count_m-4], modified_data[count_m-3], modified_data[count_m-2], modified_data[count_m-1]])
        predicted = clf.predict(successive_data[-1:])
        #print ('æ¬¡ã®1æéè¶³ã®äºæ¸¬:' + str(list(predicted)) + ' 1:é½ç·,ã0:é°ç·')
        if str(list(predicted)) == str([1]):
            st.write('ãé½ç·ãã§ãããã')
        else:
            st.write('ãé°ç·ãã§ãããã')
        st.write("æ­£è§£ç: " + str(round(correct / (correct+wrong) * 100,  2)) + "%")   
        t2 = time.time()
        elapsed_time = t2- t1
        elapsed_time = round(elapsed_time, 2)
        st.write('ãã­ã°ã©ã å¦çæéï¼ ' + str(elapsed_time) + 'ç§')
        comment.write('å®äºãã¾ããï¼')
except:
    st.error('ã¨ã©ã¼ãçºçãã¾ããããã°ãããã¦ãããååº¦å®è¡ãã¦ãã ããã')
