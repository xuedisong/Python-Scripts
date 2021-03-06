//+------------------------------------------------------------------+
//|                                                  ShapeTest_2.mq4 |
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
   //锤子线是反转信号
   //下影线的长度至少达到实体高度的2倍
   //上影线长度极短
   int num=1000;
   double underL=0.0;
   int count=0;
   for(int i=0;i<num;i++){
      if(Close[i]>Open[i]){
         underL=Open[i]-Low[i];
         if(underL>2*(Close[i]-Open[i])&&(High[i]-Close[i])<0.05*(High[i]-Low[i])){
            ObjectCreate("锤子线"+IntegerToString(i),OBJ_ARROW,0,Time[i],Close[i]);
            count+=1;
         }
         
      }
      else{
         underL=Close[i]-Low[i];
         if(underL>2*(Open[i]-Close[i])&&(High[i]-Open[i])<0.05*(High[i]-Low[i])){
            ObjectCreate("锤子线"+IntegerToString(i),OBJ_ARROW,0,Time[i],Close[i]);
            count+=1;
         }
      }
   }
   printf("反转信号有"+count+"个");
   
}
//+------------------------------------------------------------------+
