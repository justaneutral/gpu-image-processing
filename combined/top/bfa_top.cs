using System;
using System.Diagnostics;
using System.Security;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Runtime.InteropServices; // DLL support

namespace bfa_top
{
    unsafe internal class bfa_top
    {
        [DllImport("kernel32.dll"), SuppressUnmanagedCodeSecurity] static extern int GetCurrentProcessorNumber();
        //[DllImport("bfa_dll.dll", CallingConvention = CallingConvention.Cdecl)]
        //public static extern float bfa_main(ref uint x, ref uint y, byte[] refimgptr, uint refimgwidth, uint refimgheight, byte[] srcimgptr, uint srcimgwidth, uint srcimgheight, uint corrh, uint corrw, uint step);
        [DllImport("bfa_dll.dll", CallingConvention = CallingConvention.Cdecl)] public static extern uint bfa_msdelay(uint delaymilliseconds);
        [DllImport("bfa_dll.dll", CallingConvention = CallingConvention.Cdecl)] public static extern uint bfa_add_and_msdelay(uint sumval, uint inval, uint *delaymilliseconds);
        
        private static readonly Object gpu = new Object();

        private static uint adval;
        private static uint predval;

        private static uint MAXRESP = 10;
        private static uint MAXTASKS = 100; //11264; //5632; //2816;
        private static uint[] generateddelays = new uint[MAXTASKS*MAXRESP];
        private static double[] totalduration = new double[MAXRESP];

        private const double laptop_processorconstant  = 31249;
        private const double server_processorconstant =  70000;

        unsafe static void Main(string[] args)
        {
            uint NUMREPS = MAXRESP; 
            uint NUMTSKS = MAXTASKS;

            predval = NUMTSKS * NUMREPS;

            for (int rn = 0; rn < NUMREPS; rn++) Parallel.For(0, NUMTSKS, (i) => { threadmain((uint)(i + NUMTSKS * rn), 1); });
            for (uint requireddelaymilliseconds = 1; requireddelaymilliseconds < 1000; requireddelaymilliseconds += (1 + requireddelaymilliseconds/10))
            {

                //double ankertime = (new TimeSpan(DateTime.Now.Ticks)).TotalMilliseconds;

                adval = 0;
                double totalstarttime;
                
                for (int rn = 0; rn < NUMREPS; rn++)
                {
                    totalstarttime = (new TimeSpan(DateTime.Now.Ticks)).TotalMilliseconds;// -ankertime;
                    Parallel.For(0, NUMTSKS, (i) => { threadmain((uint)(i+NUMTSKS*rn), requireddelaymilliseconds); });
                    totalduration[rn] = (new TimeSpan(DateTime.Now.Ticks)).TotalMilliseconds - totalstarttime;// -ankertime;
                
                    //for(int i=0;i<NUMTSKS;i++)
                    //    Console.WriteLine("Task {0} delay {1} ms.", i, generateddelays[i+NUMTSKS*rn]);
                }
                
                
                double avggpucpu = 0; 
                for (uint j = 0; j < NUMREPS; j++) //don't take outliar
                    avggpucpu += totalduration[j]; 
                avggpucpu /= NUMREPS;
                double stdgpucpu = 0; 
                for (uint j = 0; j < NUMREPS; j++) 
                    stdgpucpu += (totalduration[j] - avggpucpu) * (totalduration[j] - avggpucpu);
                stdgpucpu = Math.Sqrt(stdgpucpu / NUMREPS);

                
                double avggpudur = 0;
                for (uint i = 0; i < NUMREPS*NUMTSKS; i++)
                    avggpudur += generateddelays[i];
                avggpudur /= (NUMREPS*NUMTSKS);
                double stdgpudur = 0;
                for (uint i = 0; i < NUMREPS*NUMTSKS; i++)
                    stdgpudur += (generateddelays[i] - avggpudur)*(generateddelays[i] - avggpudur);
                stdgpudur = Math.Sqrt(stdgpudur/(NUMTSKS * NUMREPS));

                double numberofsimultaneoustasks = avggpudur * NUMTSKS/ avggpucpu;

                if(predval==adval)
                    Console.WriteLine("GPU:{0}+/-{1}ms,CPU+GPU:{2}+/-{3}ms,Cores:{4}", 
                        (float)(((uint)(100 * avggpudur)) / 100.0),
                        (float)(((uint)(100 * stdgpudur)) / 100.0),
                        (float)(((uint)(100 * avggpucpu)) / 100.0),
                        (float)(((uint)(100*stdgpucpu))/100.0), 
                        (float)(((uint)(100*numberofsimultaneoustasks))/100.0));
            } 
            while (Console.Read() != 'q');
        }

        unsafe static void threadmain(uint i, uint delaymilliseconds)
        {
            bool ingpu = true; // false;
            //Random rnd = new Random();
            uint inval = 1;
            //Console.WriteLine("Global tid = {0}, Thread Id = {1}, CoreId = {2}\n",gtid,Thread.CurrentThread.ManagedThreadId,GetCurrentProcessorNumber());

            if (!ingpu)
            {
                lock (gpu)
                {
                    double currenttimems;
                    double starttimems = (new TimeSpan(DateTime.Now.Ticks)).TotalMilliseconds;
                    //double stoptimems = starttimems + delaymilliseconds;
                    for (double x = 0; x < server_processorconstant * delaymilliseconds; x++)
                        generateddelays[i] = (uint)(Math.Sqrt(x));

                    //do
                    //{
                    currenttimems = (new TimeSpan(DateTime.Now.Ticks)).TotalMilliseconds;
                    // }
                    //while (currenttimems < stoptimems);
                    generateddelays[i] = (uint)(currenttimems - starttimems);
                    //lock (gpu)
                    {
                        adval++;
                    }

                }
            }
            else
            {
                lock (gpu)
                {
                    if (delaymilliseconds > 0)
                    {
                        if (inval == 0)
                        {
                            generateddelays[i] = bfa_msdelay(delaymilliseconds);
                            adval++;
                        }
                        else
                        {
                            uint gtd = delaymilliseconds;
                            adval = bfa_add_and_msdelay(adval, inval, &gtd);
                            generateddelays[i] = gtd;
                        }
                    }
                }
            }
        }
    }
}

