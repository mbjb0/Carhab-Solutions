/*
 * The MIT License (MIT)

Copyright (c) 2015 Jetsonhacks

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef _JHPWMPCA9685_H
#define _JHPWMPCA9685_H

#include <cstddef>
#include <linux/i2c-dev.h>
#include <sys/ioctl.h>
#include <cstdlib>
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>


/*
    i2c-dev.h - i2c-bus driver, char device interface

    Copyright (C) 1995-97 Simon G. Vogl
    Copyright (C) 1998-99 Frodo Looijaard <frodol@dds.nl>

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
    MA 02110-1301 USA.
*/

#ifndef _LINUX_I2C_DEV_H
#define _LINUX_I2C_DEV_H

#include <linux/types.h>
#include <sys/ioctl.h>
#include <stddef.h>


/* -- i2c.h -- */


/*
 * I2C Message - used for pure i2c transaction, also from /dev interface
 */
struct i2c_msg {
    __u16 addr;	/* slave address			*/
    unsigned short flags;
#define I2C_M_TEN	0x10	/* we have a ten bit chip address	*/
#define I2C_M_RD	0x01
#define I2C_M_NOSTART	0x4000
#define I2C_M_REV_DIR_ADDR	0x2000
#define I2C_M_IGNORE_NAK	0x1000
#define I2C_M_NO_RD_ACK		0x0800
    short len;		/* msg length				*/
    char *buf;		/* pointer to msg data			*/
};

/* To determine what functionality is present */

#define I2C_FUNC_I2C			0x00000001
#define I2C_FUNC_10BIT_ADDR		0x00000002
#define I2C_FUNC_PROTOCOL_MANGLING	0x00000004 /* I2C_M_{REV_DIR_ADDR,NOSTART,..} */
#define I2C_FUNC_SMBUS_PEC		0x00000008
#define I2C_FUNC_SMBUS_BLOCK_PROC_CALL	0x00008000 /* SMBus 2.0 */
#define I2C_FUNC_SMBUS_QUICK		0x00010000
#define I2C_FUNC_SMBUS_READ_BYTE	0x00020000
#define I2C_FUNC_SMBUS_WRITE_BYTE	0x00040000
#define I2C_FUNC_SMBUS_READ_BYTE_DATA	0x00080000
#define I2C_FUNC_SMBUS_WRITE_BYTE_DATA	0x00100000
#define I2C_FUNC_SMBUS_READ_WORD_DATA	0x00200000
#define I2C_FUNC_SMBUS_WRITE_WORD_DATA	0x00400000
#define I2C_FUNC_SMBUS_PROC_CALL	0x00800000
#define I2C_FUNC_SMBUS_READ_BLOCK_DATA	0x01000000
#define I2C_FUNC_SMBUS_WRITE_BLOCK_DATA 0x02000000
#define I2C_FUNC_SMBUS_READ_I2C_BLOCK	0x04000000 /* I2C-like block xfer  */
#define I2C_FUNC_SMBUS_WRITE_I2C_BLOCK	0x08000000 /* w/ 1-byte reg. addr. */

#define I2C_FUNC_SMBUS_BYTE (I2C_FUNC_SMBUS_READ_BYTE | \
                             I2C_FUNC_SMBUS_WRITE_BYTE)
#define I2C_FUNC_SMBUS_BYTE_DATA (I2C_FUNC_SMBUS_READ_BYTE_DATA | \
                                  I2C_FUNC_SMBUS_WRITE_BYTE_DATA)
#define I2C_FUNC_SMBUS_WORD_DATA (I2C_FUNC_SMBUS_READ_WORD_DATA | \
                                  I2C_FUNC_SMBUS_WRITE_WORD_DATA)
#define I2C_FUNC_SMBUS_BLOCK_DATA (I2C_FUNC_SMBUS_READ_BLOCK_DATA | \
                                   I2C_FUNC_SMBUS_WRITE_BLOCK_DATA)
#define I2C_FUNC_SMBUS_I2C_BLOCK (I2C_FUNC_SMBUS_READ_I2C_BLOCK | \
                                  I2C_FUNC_SMBUS_WRITE_I2C_BLOCK)

/* Old name, for compatibility */
#define I2C_FUNC_SMBUS_HWPEC_CALC	I2C_FUNC_SMBUS_PEC

/*
 * Data for SMBus Messages
 */
#define I2C_SMBUS_BLOCK_MAX	32	/* As specified in SMBus standard */
#define I2C_SMBUS_I2C_BLOCK_MAX	32	/* Not specified but we use same structure */
union i2c_smbus_data {
    __u8 byte;
    __u16 word;
    __u8 block[I2C_SMBUS_BLOCK_MAX + 2]; /* block[0] is used for length */
                                                /* and one more for PEC */
};

/* smbus_access read or write markers */
#define I2C_SMBUS_READ	1
#define I2C_SMBUS_WRITE	0

/* SMBus transaction types (size parameter in the above functions)
   Note: these no longer correspond to the (arbitrary) PIIX4 internal codes! */
#define I2C_SMBUS_QUICK		    0
#define I2C_SMBUS_BYTE		    1
#define I2C_SMBUS_BYTE_DATA	    2
#define I2C_SMBUS_WORD_DATA	    3
#define I2C_SMBUS_PROC_CALL	    4
#define I2C_SMBUS_BLOCK_DATA	    5
#define I2C_SMBUS_I2C_BLOCK_BROKEN  6
#define I2C_SMBUS_BLOCK_PROC_CALL   7		/* SMBus 2.0 */
#define I2C_SMBUS_I2C_BLOCK_DATA    8


/* /dev/i2c-X ioctl commands.  The ioctl's parameter is always an
 * unsigned long, except for:
 *	- I2C_FUNCS, takes pointer to an unsigned long
 *	- I2C_RDWR, takes pointer to struct i2c_rdwr_ioctl_data
 *	- I2C_SMBUS, takes pointer to struct i2c_smbus_ioctl_data
 */
#define I2C_RETRIES	0x0701	/* number of times a device address should
                   be polled when not acknowledging */
#define I2C_TIMEOUT	0x0702	/* set timeout in units of 10 ms */

/* NOTE: Slave address is 7 or 10 bits, but 10-bit addresses
 * are NOT supported! (due to code brokenness)
 */
#define I2C_SLAVE	0x0703	/* Use this slave address */
#define I2C_SLAVE_FORCE	0x0706	/* Use this slave address, even if it
                   is already in use by a driver! */
#define I2C_TENBIT	0x0704	/* 0 for 7 bit addrs, != 0 for 10 bit */

#define I2C_FUNCS	0x0705	/* Get the adapter functionality mask */

#define I2C_RDWR	0x0707	/* Combined R/W transfer (one STOP only) */

#define I2C_PEC		0x0708	/* != 0 to use PEC with SMBus */
#define I2C_SMBUS	0x0720	/* SMBus transfer */


/* This is the structure as used in the I2C_SMBUS ioctl call */
struct i2c_smbus_ioctl_data {
    __u8 read_write;
    __u8 command;
    __u32 size;
    union i2c_smbus_data *data;
};

/* This is the structure as used in the I2C_RDWR ioctl call */
struct i2c_rdwr_ioctl_data {
    struct i2c_msg *msgs;	/* pointers to i2c_msgs */
    __u32 nmsgs;			/* number of i2c_msgs */
};

#define  I2C_RDRW_IOCTL_MAX_MSGS	42


static inline __s32 i2c_smbus_access(int file, char read_write, __u8 command,
                                     int size, union i2c_smbus_data *data)
{
    struct i2c_smbus_ioctl_data args;

    args.read_write = read_write;
    args.command = command;
    args.size = size;
    args.data = data;
    return ioctl(file,I2C_SMBUS,&args);
}


static inline __s32 i2c_smbus_write_quick(int file, __u8 value)
{
    return i2c_smbus_access(file,value,0,I2C_SMBUS_QUICK,NULL);
}

static inline __s32 i2c_smbus_read_byte(int file)
{
    union i2c_smbus_data data;
    if (i2c_smbus_access(file,I2C_SMBUS_READ,0,I2C_SMBUS_BYTE,&data))
        return -1;
    else
        return 0x0FF & data.byte;
}

static inline __s32 i2c_smbus_write_byte(int file, __u8 value)
{
    return i2c_smbus_access(file,I2C_SMBUS_WRITE,value,
                            I2C_SMBUS_BYTE,NULL);
}

static inline __s32 i2c_smbus_read_byte_data(int file, __u8 command)
{
    union i2c_smbus_data data;
    if (i2c_smbus_access(file,I2C_SMBUS_READ,command,
                         I2C_SMBUS_BYTE_DATA,&data))
        return -1;
    else
        return 0x0FF & data.byte;
}

static inline __s32 i2c_smbus_write_byte_data(int file, __u8 command,
                                              __u8 value)
{
    union i2c_smbus_data data;
    data.byte = value;
    return i2c_smbus_access(file,I2C_SMBUS_WRITE,command,
                            I2C_SMBUS_BYTE_DATA, &data);
}

static inline __s32 i2c_smbus_read_word_data(int file, __u8 command)
{
    union i2c_smbus_data data;
    if (i2c_smbus_access(file,I2C_SMBUS_READ,command,
                         I2C_SMBUS_WORD_DATA,&data))
        return -1;
    else
        return 0x0FFFF & data.word;
}

static inline __s32 i2c_smbus_write_word_data(int file, __u8 command,
                                              __u16 value)
{
    union i2c_smbus_data data;
    data.word = value;
    return i2c_smbus_access(file,I2C_SMBUS_WRITE,command,
                            I2C_SMBUS_WORD_DATA, &data);
}

static inline __s32 i2c_smbus_process_call(int file, __u8 command, __u16 value)
{
    union i2c_smbus_data data;
    data.word = value;
    if (i2c_smbus_access(file,I2C_SMBUS_WRITE,command,
                         I2C_SMBUS_PROC_CALL,&data))
        return -1;
    else
        return 0x0FFFF & data.word;
}


/* Returns the number of read bytes */
static inline __s32 i2c_smbus_read_block_data(int file, __u8 command,
                                              __u8 *values)
{
    union i2c_smbus_data data;
    int i;
    if (i2c_smbus_access(file,I2C_SMBUS_READ,command,
                         I2C_SMBUS_BLOCK_DATA,&data))
        return -1;
    else {
        for (i = 1; i <= data.block[0]; i++)
            values[i-1] = data.block[i];
        return data.block[0];
    }
}

static inline __s32 i2c_smbus_write_block_data(int file, __u8 command,
                                               __u8 length, const __u8 *values)
{
    union i2c_smbus_data data;
    int i;
    if (length > 32)
        length = 32;
    for (i = 1; i <= length; i++)
        data.block[i] = values[i-1];
    data.block[0] = length;
    return i2c_smbus_access(file,I2C_SMBUS_WRITE,command,
                            I2C_SMBUS_BLOCK_DATA, &data);
}

/* Returns the number of read bytes */
/* Until kernel 2.6.22, the length is hardcoded to 32 bytes. If you
   ask for less than 32 bytes, your code will only work with kernels
   2.6.23 and later. */
static inline __s32 i2c_smbus_read_i2c_block_data(int file, __u8 command,
                                                  __u8 length, __u8 *values)
{
    union i2c_smbus_data data;
    int i;

    if (length > 32)
        length = 32;
    data.block[0] = length;
    if (i2c_smbus_access(file,I2C_SMBUS_READ,command,
                         length == 32 ? I2C_SMBUS_I2C_BLOCK_BROKEN :
                          I2C_SMBUS_I2C_BLOCK_DATA,&data))
        return -1;
    else {
        for (i = 1; i <= data.block[0]; i++)
            values[i-1] = data.block[i];
        return data.block[0];
    }
}

static inline __s32 i2c_smbus_write_i2c_block_data(int file, __u8 command,
                                                   __u8 length,
                                                   const __u8 *values)
{
    union i2c_smbus_data data;
    int i;
    if (length > 32)
        length = 32;
    for (i = 1; i <= length; i++)
        data.block[i] = values[i-1];
    data.block[0] = length;
    return i2c_smbus_access(file,I2C_SMBUS_WRITE,command,
                            I2C_SMBUS_I2C_BLOCK_BROKEN, &data);
}

/* Returns the number of read bytes */
static inline __s32 i2c_smbus_block_process_call(int file, __u8 command,
                                                 __u8 length, __u8 *values)
{
    union i2c_smbus_data data;
    int i;
    if (length > 32)
        length = 32;
    for (i = 1; i <= length; i++)
        data.block[i] = values[i-1];
    data.block[0] = length;
    if (i2c_smbus_access(file,I2C_SMBUS_WRITE,command,
                         I2C_SMBUS_BLOCK_PROC_CALL,&data))
        return -1;
    else {
        for (i = 1; i <= data.block[0]; i++)
            values[i-1] = data.block[i];
        return data.block[0];
    }
}


#endif /* _LINUX_I2C_DEV_H */

class PCA9685
{
public:
    unsigned char kI2CBus ;         // I2C bus of the PCA9685
    int kI2CFileDescriptor ;        // File Descriptor to the PCA9685
    int kI2CAddress ;               // Address of PCA9685; defaults to 0x40
    int error ;
    PCA9685(int address=0x40);
    ~PCA9685() ;
    bool openPCA9685() ;
    void closePCA9685();

    void reset() ;

    // Sets the frequency of the PWM signal
    // Frequency is ranged between 40 and 1000 Hertz
    void setPWMFrequency ( float frequency );

    // Channels 0-15
    // Channels are in sets of 4 bytes
    void setPWM ( int channel, int onValue, int offValue);

    void setAllPWM (int onValue, int offValue);

    // Read the given register
    int readByte(int readRegister);

    // Write the the given value to the given register
    int writeByte(int writeRegister, int writeValue);

    int getError() ;

};


// Register definitions from Table 7.3 NXP Semiconductors
// Product Data Sheet, Rev. 4 - 16 April 2015
#define PCA9685_MODE1            0x00
#define PCA9685_MODE2            0x01
#define PCA9685_SUBADR1          0x02
#define PCA9685_SUBADR2          0x03
#define PCA9685_SUBADR3          0x04
#define PCA9685_ALLCALLADR       0x05
// LED outbut and brightness
#define PCA9685_LED0_ON_L        0x06
#define PCA9685_LED0_ON_H        0x07
#define PCA9685_LED0_OFF_L       0x08
#define PCA9685_LED0_OFF_H       0x09

#define PCA9685_LED1_ON_L        0x0A
#define PCA9685_LED1_ON_H        0x0B
#define PCA9685_LED1_OFF_L       0x0C
#define PCA9685_LED1_OFF_H       0x0D

#define PCA9685_LED2_ON_L        0x0E
#define PCA9685_LED2_ON_H        0x0F
#define PCA9685_LED2_OFF_L       0x10
#define PCA9685_LED2_OFF_H       0x11

#define PCA9685_LED3_ON_L        0x12
#define PCA9685_LED3_ON_H        0x13
#define PCA9685_LED3_OFF_L       0x14
#define PCA9685_LED3_OFF_H       0x15

#define PCA9685_LED4_ON_L        0x16
#define PCA9685_LED4_ON_H        0x17
#define PCA9685_LED4_OFF_L       0x18
#define PCA9685_LED4_OFF_H       0x19

#define PCA9685_LED5_ON_L        0x1A
#define PCA9685_LED5_ON_H        0x1B
#define PCA9685_LED5_OFF_L       0x1C
#define PCA9685_LED5_OFF_H       0x1D

#define PCA9685_LED6_ON_L        0x1E
#define PCA9685_LED6_ON_H        0x1F
#define PCA9685_LED6_OFF_L       0x20
#define PCA9685_LED6_OFF_H       0x21

#define PCA9685_LED7_ON_L        0x22
#define PCA9685_LED7_ON_H        0x23
#define PCA9685_LED7_OFF_L       0x24
#define PCA9685_LED7_OFF_H       0x25

#define PCA9685_LED8_ON_L        0x26
#define PCA9685_LED8_ON_H        0x27
#define PCA9685_LED8_OFF_L       0x28
#define PCA9685_LED8_OFF_H       0x29

#define PCA9685_LED9_ON_L        0x2A
#define PCA9685_LED9_ON_H        0x2B
#define PCA9685_LED9_OFF_L       0x2C
#define PCA9685_LED9_OFF_H       0x2D

#define PCA9685_LED10_ON_L       0x2E
#define PCA9685_LED10_ON_H       0x2F
#define PCA9685_LED10_OFF_L      0x30
#define PCA9685_LED10_OFF_H      0x31

#define PCA9685_LED11_ON_L       0x32
#define PCA9685_LED11_ON_H       0x33
#define PCA9685_LED11_OFF_L      0x34
#define PCA9685_LED11_OFF_H      0x35

#define PCA9685_LED12_ON_L       0x36
#define PCA9685_LED12_ON_H       0x37
#define PCA9685_LED12_OFF_L      0x38
#define PCA9685_LED12_OFF_H      0x39

#define PCA9685_LED13_ON_L       0x3A
#define PCA9685_LED13_ON_H       0x3B
#define PCA9685_LED13_OFF_L      0x3C
#define PCA9685_LED13_OFF_H      0x3D

#define PCA9685_LED14_ON_L       0x3E
#define PCA9685_LED14_ON_H       0x3F
#define PCA9685_LED14_OFF_L      0x40
#define PCA9685_LED14_OFF_H      0x41

#define PCA9685_LED15_ON_L       0x42
#define PCA9685_LED15_ON_H       0x43
#define PCA9685_LED15_OFF_L      0x44
#define PCA9685_LED15_OFF_H      0x45

#define PCA9685_ALL_LED_ON_L     0xFA
#define PCA9685_ALL_LED_ON_H     0xFB
#define PCA9685_ALL_LED_OFF_L    0xFC
#define PCA9685_ALL_LED_OFF_H    0xFD
#define PCA9685_PRE_SCALE        0xFE

// Register Bits
#define PCA9685_ALLCALL          0x01
#define PCA9685_OUTDRV           0x04
#define PCA9685_RESTART          0x80
#define PCA9685_SLEEP            0x10
#define PCA9685_INVERT           0x10



#endif
