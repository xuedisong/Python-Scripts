//+------------------------------------------------------------------+
//|                                                   RsiWinRate.mq4 |
//|                        Copyright 2017, MetaQuotes Software Corp. |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
#property copyright "Copyright 2017, MetaQuotes Software Corp."
#property link      "https://www.mql5.com"
#property version   "1.00"
#property strict
//+------------------------------------------------------------------+
//| Script program start function                                    |
//+------------------------------------------------------------------+
void OnStart(){
   ObjectsDeleteAll();
   int num=100;//100个bar
   int sigCount=0;
   int rightCount=0;
   for(int i=1;i<num;i++){
      //上穿
      if(iRSI(NULL,0,7,PRICE_CLOSE,i)>iRSI(NULL,0,14,PRICE_CLOSE,i)&&
      iRSI(NULL,0,7,PRICE_CLOSE,i+1)<iRSI(NULL,0,14,PRICE_CLOSE,i+1)){
         ObjectCreate("RSI金叉"+IntegerToString(i),OBJ_ARROW_UP,0,Time[i],Close[i]);
         sigCount+=1;
         if(Close[i-1]>Close[i]){
            rightCount+=1;
         }
      }
      //下穿
      if(iRSI(NULL,0,7,PRICE_CLOSE,i)<iRSI(NULL,0,14,PRICE_CLOSE,i)&&
      iRSI(NULL,0,7,PRICE_CLOSE,i+1)>iRSI(NULL,0,14,PRICE_CLOSE,i+1)){
         ObjectCreate("RSI死叉"+IntegerToString(i),OBJ_ARROW_DOWN,0,Time[i],Close[i]);
         sigCount+=1;
         if(Close[i-1]<Close[i]){
            rightCount+=1;
         }
      }
   }
   printf("信号点共有"+sigCount+"个，\n");
   printf("其中预测准确的有"+rightCount+"个，\n");
   printf("预测准确率为："+((rightCount+0.0)/sigCount)*100+"%\n");
}
//+------------------------------------------------------------------+
