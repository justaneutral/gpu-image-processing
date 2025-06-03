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
    internal class bfa_top
    {
        [DllImport("kernel32.dll"), SuppressUnmanagedCodeSecurity] static extern int GetCurrentProcessorNumber();
        //[DllImport("bfa_dll.dll", CallingConvention = CallingConvention.Cdecl)]
        //public static extern float bfa_main(ref uint x, ref uint y, byte[] refimgptr, uint refimgwidth, uint refimgheight, byte[] srcimgptr, uint srcimgwidth, uint srcimgheight, uint corrh, uint corrw, uint step);
        [DllImport("bfa_dll.dll", CallingConvention = CallingConvention.Cdecl)] public static extern uint bfa_msdelay(uint delaymilliseconds);
        [DllImport("bfa_dll.dll", CallingConvention = CallingConvention.Cdecl)] public static extern uint bfa_add_and_msdelay(uint sumval, uint inval, uint delaymilliseconds);
        
        private static readonly Object gpu = new Object();

        private static Random  rnd = new Random();

        private static uint adval;
        private static uint predval;

        private static uint MAXTASKS = 100;
        private static uint MAXTHREADS = 1024;
        private static uint MAXCORES = 50;

        private static uint[] taskidhist = new uint[MAXTASKS];
        private static uint[] threadidhist = new uint[MAXTHREADS];
        private static uint[] coreidhist = new uint[MAXCORES];

        unsafe static void Main(string[] args)
        {
            for (uint i00 = 0; i00 < MAXTASKS; i00++) taskidhist[i00] = 0;
            for (uint i00 = 0; i00 < MAXTHREADS; i00++) threadidhist[i00] = 0;
            for (uint i00 = 0; i00 < MAXCORES; i00++) coreidhist[i00] = 0;

            Console.WriteLine("in C# Main, adval = {0}", adval);
            //Task[] tsk = new Task[NUMTSKS];
            predval = 0;
            adval = 0;
            do 
            {
                uint NUMREPS = 2; // (uint)rnd.Next(1, 10);
                for (int rn = 0; rn < NUMREPS; rn++)
                {
                    uint NUMTSKS = MAXTASKS; // (uint)rnd.Next(4, (int)MAXTASKS);
                    predval += NUMTSKS;
                    Parallel.For(0, NUMTSKS, (i) => {threadmain((int)i);});
                    //for (int i = 0; i < NUMTSKS; i++)
                    //{
                    //    tsk[i] = Task.Factory.StartNew(() => threadmain(i));
                    //}
                    //Task.WaitAll(tsk);
                    Console.WriteLine("Iteratin {0} all threads complete", rn);
                }
                Console.WriteLine("Back in c# main.\nAll threads complete\nPredicted = {0}, Actual = {1}\nenter some key and cr. q - quit>",predval,adval);
            } while (Console.Read() != 'q');
            do{
                for (uint i00 = 0; i00 < MAXTASKS; i00++) if (taskidhist[i00] > 0) { Thread.Sleep(300); Console.WriteLine("ts({0})={1}", i00, taskidhist[i00]); }
                for (uint i00 = 0; i00 < MAXTHREADS; i00++) if (threadidhist[i00] > 0) { Thread.Sleep(300); Console.WriteLine("th({0})={1}", i00, threadidhist[i00]);}
                for (uint i00 = 0; i00 < MAXCORES; i00++) if (coreidhist[i00] > 0) { Thread.Sleep(300); Console.WriteLine("cr({0})={1}", i00, coreidhist[i00]); }
            } while (Console.Read() != 'q');
        }

        unsafe static void threadmain(int gtidv)
        {
            //Random rnd = new Random();
            uint inval = 1;
            int gtid = gtidv;
            //Console.WriteLine("Global tid = {0}, Thread Id = {1}, CoreId = {2}\n",gtid,Thread.CurrentThread.ManagedThreadId,GetCurrentProcessorNumber());
            uint retdelay = 0;
            uint delaymilliseconds = (uint)rnd.Next(1,999);
            /*
            uint y = 11;
            uint x = 11;
            uint step = 4; // 1; //when 0 - for tsting returns -88 from the upper DLL function
            uint corrh = 20;
            uint corrw = 20;
            uint refimgwidth = 3840;
            uint refimgheight = 2048;
            uint srcimgwidth = refimgwidth + corrw;
            uint srcimgheight = refimgheight + corrh;
            byte[] refimgptr = new byte[refimgheight * refimgwidth];
            byte[] srcimgptr = new byte[srcimgheight * srcimgwidth];
            float r = 0.0f;
            for (uint i = refimgwidth / 3; i < 2 * refimgwidth / 3; i++)
            {
                for (uint j = refimgheight / 3; j < 2 * refimgheight / 3; j++)
                {
                    refimgptr[refimgwidth * j + i] = (byte)(255.0 / (1.0 + (i > j ? i - j : j - i)));
                    srcimgptr[srcimgwidth * (j + y) + i + x] = refimgptr[refimgwidth * j + i];
                }
            }

            x = 0;
            y = 0;
            */
            //bfa_main(ref x, ref y, null, 0, 0, null, 0, 0, 0, 0, 100);//create critical section

            lock(gpu)
            {
                if(gtid>=0 && gtid<MAXTASKS) taskidhist[gtid]++;
                int tidx = Thread.CurrentThread.ManagedThreadId; if(tidx>=0 && tidx<MAXTHREADS) threadidhist[tidx]++;
                int cidx = GetCurrentProcessorNumber(); if (cidx >= 0 && cidx < MAXCORES) coreidhist[cidx]++;

                //adval++;
                //Console.WriteLine("Global tid = {0}, Thread {1} Core {1} locked", gtid,Thread.CurrentThread.ManagedThreadId, GetCurrentProcessorNumber());
                if (delaymilliseconds>0)
                {
                    if (inval == 0)
                    {
                       retdelay = bfa_msdelay(delaymilliseconds);
                    }
                    else
                    {
                        //random dummy execution 1
                        if (rnd.Next(0, 1) == 1)
                        {
                            double p00 = 3.1415;
                            for (int i00 = 0; i00 < rnd.Next(1, 1000); i00++)
                                p00 *= 0.99;
                            //Console.WriteLine("Global tid = {0}, Thread {1} Core {1} random 1 = {2}", gtid, Thread.CurrentThread.ManagedThreadId, GetCurrentProcessorNumber());
                        }
                        adval = bfa_add_and_msdelay(adval, inval, delaymilliseconds);
                        //random dummy execution 2
                        if (rnd.Next(0, 1) == 1)
                        {
                            double p00 = 3.1415;
                            for (int i00 = 0; i00 < rnd.Next(1, 1000); i00++)
                                p00 *= 0.99;
                            //Console.WriteLine("Global tid = {0}, Thread {1} Core {1} random 1 = {2}", gtid, Thread.CurrentThread.ManagedThreadId, GetCurrentProcessorNumber());
                        }
                    
                    }
                }
                //else
                //{
                    //if (step > 0)
                      //  r = bfa_main(ref x, ref y, refimgptr, refimgwidth, refimgheight, srcimgptr, srcimgwidth, srcimgheight, corrh, corrw, step);
                    //else
                    //{
                      //  uint x0 = 0, y0 = 0;
                      //  r = bfa_main(ref x0, ref y0, null, 0, 0, null, 0, 0, 0, 0, step);
                    //}
                //}
                //Console.WriteLine("Global tid = {0}, Thread {1} Core {2} unlocked", gtid, Thread.CurrentThread.ManagedThreadId, GetCurrentProcessorNumber());
            }
            //bfa_main(ref x, ref y, null, 0, 0, null, 0, 0, 0, 0, 101);//destroy critical section
            //if(delaymilliseconds>0)
                //Console.WriteLine("on Global tid = {0}, Thread {1} Core {2} requested delay = {3} ms., produced delay = {4} ms.", gtid,Thread.CurrentThread.ManagedThreadId, GetCurrentProcessorNumber(), delaymilliseconds, retdelay);
            //else
                //Console.WriteLine("on Global tid = {0}, Thread {1} Core {2} processing delay = {3}, x={4}, y={5}", gtid, Thread.CurrentThread.ManagedThreadId, GetCurrentProcessorNumber(), r, x, y);

        }
    }
}

